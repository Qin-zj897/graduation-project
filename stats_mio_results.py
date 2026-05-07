#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from glob import glob
from typing import Optional

from stats_pynguin_results import (
    build_failure_section,
    build_success_section,
    extract_code_id,
    infer_label_from_module,
    split_items,
    _evaluate_program_entry,
)


BASE_DIR = os.path.dirname(__file__)
DEFAULT_MIO_DIR = os.path.join(BASE_DIR, "testcase_result", "pynguin", "mio")
DEFAULT_DATASET_ROOT = BASE_DIR


def build_output_path(mio_dir: str, problem_id: Optional[str] = None) -> str:
    statistics_dir = os.path.join(os.path.dirname(os.path.dirname(mio_dir.rstrip(os.sep))), "statistics")
    os.makedirs(statistics_dir, exist_ok=True)
    prefix = f"mio_stats_{problem_id}_" if problem_id else "mio_stats_"
    return os.path.join(statistics_dir, f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


def discover_mio_pairs(mio_root: str, dataset_root: str, problem_id: Optional[str] = None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pattern = os.path.join(mio_root, "**", "test_*.py")
    for test_file in glob(pattern, recursive=True):
        if not os.path.isfile(test_file):
            continue
        module_name = os.path.basename(os.path.dirname(test_file))
        label = infer_label_from_module(module_name)
        if label is None:
            continue
        parts = module_name.split("_")
        if len(parts) < 3:
            continue
        current_problem_id = parts[1]
        if problem_id and current_problem_id != problem_id:
            continue
        bucket = "success" if label == "s" else "failure"
        if current_problem_id in {"03039", "03025", "02882"}:
            program_file = os.path.join(dataset_root, "pynguin_wrapped", current_problem_id, bucket, f"{module_name}.py")
        else:
            program_file = os.path.join(dataset_root, current_problem_id, bucket, f"{module_name}.py")
        if os.path.isfile(program_file):
            pairs.append((program_file, test_file))
    return sorted(set(pairs))


def evaluate_mio_program_with_timeout(program_file: str, test_file: str, timeout_seconds: int = 20) -> dict:
    completed = subprocess.run(
        [sys.executable, __file__, "--internal-evaluate", program_file, test_file],
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        raise RuntimeError(stderr or stdout or f"subprocess exit={completed.returncode}")
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("No evaluation output produced")
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 MIO(Pynguin --algorithm MIO) 结果")
    parser.add_argument("--mio-dir", default=DEFAULT_MIO_DIR, help="MIO 结果目录，默认 testcase_result/pynguin/mio")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="数据集根目录")
    parser.add_argument("--problem", default=None, help="只统计指定题号，例如 2910 / 03025 / 02882")
    parser.add_argument("--limit", type=int, default=None, help="只统计前 N 个程序，便于快速验证")
    parser.add_argument("--module-timeout", type=int, default=20, help="单模块统计超时时间（秒）")
    parser.add_argument("--internal-evaluate", nargs=2, metavar=("PROGRAM_FILE", "TEST_FILE"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.internal_evaluate:
        _evaluate_program_entry(args.internal_evaluate[0], args.internal_evaluate[1])
        return

    pairs = discover_mio_pairs(args.mio_dir, args.dataset_root, args.problem)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    total = len(pairs)
    items = []
    for index, (program_file, test_file) in enumerate(pairs, start=1):
        module_name = os.path.splitext(os.path.basename(program_file))[0]
        print(f"[PROGRESS] {index}/{total} START {module_name}", flush=True)
        try:
            item = evaluate_mio_program_with_timeout(program_file, test_file, timeout_seconds=args.module_timeout)
            items.append(item)
            print(f"[PROGRESS] {index}/{total} DONE {module_name}", flush=True)
        except subprocess.TimeoutExpired:
            label = infer_label_from_module(module_name)
            items.append({
                "module": module_name,
                "code_id": extract_code_id(module_name),
                "label": label,
                "bucket": "success" if label == "s" else "failure" if label == "f" else None,
                "edge_coverage": None,
                "block_coverage": None,
                "program_correct": False,
                "raw_case_count": 0,
                "valid_case_count": 0,
                "invalid_input_count": 0,
                "auxiliary_call_count": 0,
                "mismatches": -1,
                "generation_failed": True,
                "error": f"Module timeout > {args.module_timeout}s",
            })
            print(f"[PROGRESS] {index}/{total} TIMEOUT {module_name}", flush=True)
        except Exception as exc:
            label = infer_label_from_module(module_name)
            items.append({
                "module": module_name,
                "code_id": extract_code_id(module_name),
                "label": label,
                "bucket": "success" if label == "s" else "failure" if label == "f" else None,
                "edge_coverage": None,
                "block_coverage": None,
                "program_correct": False,
                "raw_case_count": 0,
                "valid_case_count": 0,
                "invalid_input_count": 0,
                "auxiliary_call_count": 0,
                "mismatches": -1,
                "generation_failed": True,
                "error": str(exc),
            })
            print(f"[PROGRESS] {index}/{total} ERROR {module_name}: {type(exc).__name__}: {exc}", flush=True)

    success_items, failure_items = split_items(items)
    summary = {
        "algorithm": "mio",
        "problem_id": args.problem,
        "file_count": len(items),
        "input_validity_rule": "same_as_stats_pynguin_results",
        "source_result_dir": args.mio_dir,
        "success": build_success_section(success_items),
        "failure": build_failure_section(failure_items),
    }
    output_path = build_output_path(args.mio_dir, args.problem)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已写入: {output_path}")


if __name__ == "__main__":
    main()
