#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import os
import re
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

from dynamic_analyzer import DynamicAnalyzer
from static_analyzer import StaticAnalyzer

EDGE_RE = re.compile(r"边覆盖率\s*:\s*\d+/\d+\s*=\s*([0-9.]+)%")
BLOCK_RE = re.compile(r"块覆盖率\s*:\s*\d+/\d+\s*=\s*([0-9.]+)%")
INPUT_RE = re.compile(r"输入\s*:\s*(.*)$")
EXPECTED_RE = re.compile(r"期望输出:\s*(.*)$")
ACTUAL_RE = re.compile(r"实际输出:\s*(.*)$")
CODE_FILE_RE = re.compile(r"代码文件:\s*(.*?)\s+XML种子:")


def outputs_match(actual: str, expected: str, problem_id: Optional[str] = None) -> bool:
    actual = (actual or "").strip()
    expected = (expected or "").strip()
    if problem_id == "03025":
        return actual.isdigit() and actual == expected
    if actual == expected:
        return True
    try:
        return abs(float(actual) - float(expected)) < 1e-6
    except Exception:
        return False


def infer_label_from_filename(path: str) -> Optional[str]:
    name = os.path.basename(path)
    if name.startswith("testcase_s_"):
        return "s"
    if name.startswith("testcase_f_"):
        return "f"
    return None


def extract_code_id(path: str) -> str:
    code_name = os.path.basename(path)
    code_id_match = re.match(r"testcase_([sf]_\d+_\d+)_", code_name)
    return code_id_match.group(1) if code_id_match else os.path.splitext(code_name)[0]


def extract_problem_id(path: str) -> Optional[str]:
    code_name = os.path.basename(path)
    problem_id_match = re.match(r"testcase_[sf]_(\d+)_\d+_", code_name)
    return problem_id_match.group(1) if problem_id_match else None


def matches_problem(path: str, problem_id: Optional[str]) -> bool:
    if not problem_id:
        return True
    return extract_problem_id(path) == problem_id


def keep_latest_result_per_code(files: List[str]) -> List[str]:
    latest: Dict[str, str] = {}
    for path in files:
        code_id = extract_code_id(path)
        current = latest.get(code_id)
        if current is None or os.path.basename(path) > os.path.basename(current):
            latest[code_id] = path
    return sorted(latest.values())


def parse_three_ints(text: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if text is None:
        return None
    m = re.match(r"\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def parse_two_values(text: Optional[str]) -> Optional[Tuple[str, str]]:
    if text is None:
        return None
    if " | " in text:
        left, right = text.split(" | ", 1)
        return left.strip(), right.strip()
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if len(parts) == 2:
        return parts[0], parts[1]
    return None


def is_valid_case(problem_id: Optional[str], input_text: Optional[str]) -> bool:
    values = parse_three_ints(input_text)
    if problem_id == "03039":
        if values is None:
            return False
        n, m, k = values
        return n >= 1 and m >= 1 and 2 <= k <= n * m
    if problem_id == "2910":
        pair = parse_two_values(input_text)
        if pair is None:
            return False
        try:
            h = float(pair[0])
            n = int(pair[1])
        except Exception:
            return False
        return h >= 0 and n >= 1
    if problem_id in {"3039", "3226"}:
        if input_text is None:
            return False
        try:
            value = ast.literal_eval(input_text.strip())
        except Exception:
            return False
        if not isinstance(value, list) or not value:
            return False
        return all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    if values is None:
        return True
    return True


def split_input_lines(input_text: str) -> List[str]:
    if " | " in input_text:
        return [part.strip() for part in input_text.split(" | ")]
    return [line.strip() for line in input_text.splitlines() if line.strip()]


def extract_code_file_from_lines(lines: List[str]) -> Optional[str]:
    for line in lines:
        match = CODE_FILE_RE.search(line)
        if match:
            return match.group(1).strip()
    return None


def build_analyzer(py_file: str, timeout: float = 5.0) -> DynamicAnalyzer:
    with open(py_file, "r", encoding="utf-8") as f:
        source_code = f.read()
    sa = StaticAnalyzer(source_code=source_code)
    static_result = {
        "cfg": sa.build_control_flow_graph(),
        "predicates": sa.extract_predicates_and_constraints(),
        "data_dependencies": sa.build_data_dependency_graph(),
        "variable_types": sa.get_variable_types(),
        "branch_constraint_map": sa.get_branch_constraint_map(),
    }
    return DynamicAnalyzer(source_code, static_result, timeout=timeout)


def is_structural_edge(edge: Dict[str, Any]) -> bool:
    return edge.get("lineno") == "virtual" or edge.get("label") in {"true_end", "false_end", "continue"}


def is_structural_block(block: Dict[str, Any]) -> bool:
    node_id = block.get("node_id", "")
    structural_prefixes = (
        "if_merge",
        "while_exit",
        "for_exit",
        "while_else",
        "main_entry",
        "func_entry",
        "func_exit",
        "for_body_start",
        "while_body_start",
    )
    return block.get("lineno") == "virtual" or any(node_id.startswith(prefix) for prefix in structural_prefixes)


def recompute_coverage_from_cases(code_file: Optional[str], inputs: List[str]) -> Tuple[Optional[float], Optional[float]]:
    if not code_file or not os.path.isfile(code_file) or not inputs:
        return None, None
    analyzer = build_analyzer(code_file)
    for input_text in inputs:
        try:
            analyzer.run_with_input(split_input_lines(input_text))
        except Exception:
            pass
    agg = analyzer.aggregate_coverage([])
    all_edges = agg.get("covered_edges", []) + agg.get("uncovered_edges", [])
    all_blocks = agg.get("covered_blocks", []) + agg.get("uncovered_blocks", [])
    real_edge_total = sum(1 for edge in all_edges if not is_structural_edge(edge))
    real_edge_covered = sum(1 for edge in agg.get("covered_edges", []) if not is_structural_edge(edge))
    real_block_total = sum(1 for block in all_blocks if not is_structural_block(block))
    real_block_covered = sum(1 for block in agg.get("covered_blocks", []) if not is_structural_block(block))
    edge_coverage = round(real_edge_covered * 100.0 / real_edge_total, 2) if real_edge_total else 0.0
    block_coverage = round(real_block_covered * 100.0 / real_block_total, 2) if real_block_total else 0.0
    return edge_coverage, block_coverage


def parse_result_file(path: str, recompute_coverage: bool = False) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    code_file = extract_code_file_from_lines(lines)
    case_inputs: List[str] = []
    edge_coverage = None
    block_coverage = None
    current_input = None
    case_matches: List[bool] = []
    valid_case_matches: List[bool] = []
    program_correct = False
    expected_bucket = None
    problem_id = extract_problem_id(path)

    current_expected = None
    current_actual_lines: Optional[List[str]] = None

    def finalize_current_case() -> None:
        nonlocal current_expected, current_actual_lines
        if current_expected is None or current_actual_lines is None:
            current_expected = None
            current_actual_lines = None
            return
        actual = "\n".join(current_actual_lines).strip()
        matched = outputs_match(actual, current_expected, problem_id)
        case_matches.append(matched)
        if is_valid_case(problem_id, current_input):
            valid_case_matches.append(matched)
        current_expected = None
        current_actual_lines = None

    for line in lines:
        if current_actual_lines is not None:
            if not (
                EDGE_RE.search(line)
                or BLOCK_RE.search(line)
                or INPUT_RE.search(line)
                or EXPECTED_RE.search(line)
                or ACTUAL_RE.search(line)
                or line.startswith("  代码类型判断 :")
                or line.startswith("  代码是否正确判断成功 :")
                or line.startswith("     结果判断:")
                or line.startswith("     覆盖分支:")
                or line.startswith("================================================================================")
                or line.startswith("覆盖率报告")
                or line.startswith("结果已保存到:")
                or line.startswith("  分支覆盖率")
                or line.startswith("  边覆盖率")
                or line.startswith("  块覆盖率")
                or line.startswith("  已覆盖分支")
                or line.startswith("  未覆盖分支")
                or line.startswith("  未覆盖真实CFG边")
                or line.startswith("  未覆盖真实块")
                or line.startswith("  结构节点/边")
                or line.startswith("    - ")
                or line.startswith("    □ ")
            ):
                current_actual_lines.append(line.strip())
                continue
            finalize_current_case()

        edge_match = EDGE_RE.search(line)
        if edge_match:
            edge_coverage = float(edge_match.group(1))
            continue
        block_match = BLOCK_RE.search(line)
        if block_match:
            block_coverage = float(block_match.group(1))
            continue
        input_match = INPUT_RE.search(line)
        if input_match:
            current_input = input_match.group(1).strip()
            case_inputs.append(current_input)
            continue
        expected_match = EXPECTED_RE.search(line)
        if expected_match:
            current_expected = expected_match.group(1).strip()
            continue
        actual_match = ACTUAL_RE.search(line)
        if actual_match:
            current_actual_lines = [actual_match.group(1).strip()]
            continue
        if line.startswith("  代码类型判断 :"):
            judgement = line.split(":", 1)[1].strip()
            if "success" in judgement:
                expected_bucket = "success"
            elif "failure" in judgement:
                expected_bucket = "failure"
            continue
        if line.startswith("  代码是否正确判断成功 :"):
            program_correct = line.split(":", 1)[1].strip() == "是"

    finalize_current_case()

    if recompute_coverage:
        recomputed_edge, recomputed_block = recompute_coverage_from_cases(code_file, case_inputs)
        if recomputed_edge is not None:
            edge_coverage = recomputed_edge
        if recomputed_block is not None:
            block_coverage = recomputed_block

    label = infer_label_from_filename(path)
    generation_failed = not lines or (edge_coverage is None and block_coverage is None and not case_matches)

    if expected_bucket is None:
        if label == "s":
            expected_bucket = "success"
        elif label == "f":
            expected_bucket = "failure"

    if not generation_failed:
        effective_matches = case_matches
        if label == "s":
            effective_matches = valid_case_matches
        if label == "s":
            program_correct = bool(effective_matches) and all(effective_matches)
        elif label == "f":
            program_correct = any(not matched for matched in effective_matches)

    return {
        "path": path,
        "label": label,
        "bucket": expected_bucket,
        "code_id": extract_code_id(path),
        "edge_coverage": edge_coverage,
        "block_coverage": block_coverage,
        "program_correct": program_correct,
        "case_count": len(case_matches),
        "valid_case_count": len(valid_case_matches),
        "generation_failed": generation_failed,
    }


def average(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 2) if valid else None


def split_items(items: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    return [item for item in items if item["bucket"] == "success"], [item for item in items if item["bucket"] == "failure"]


def build_success_section(items: List[Dict]) -> Dict:
    generation_failed_items = [item for item in items if item.get("generation_failed")]
    valid_items = [item for item in items if not item.get("generation_failed")]
    hidden_bug_items = [item for item in valid_items if not item["program_correct"]]
    consistent_items = [item for item in valid_items if item["program_correct"]]
    return {
        "program_count": len(items),
        "valid_program_count": len(valid_items),
        "generation_failed_ids": [item["code_id"] for item in generation_failed_items],
        "generation_failed_count": len(generation_failed_items),
        "avg_edge_coverage": average([item["edge_coverage"] for item in valid_items]),
        "avg_block_coverage": average([item["block_coverage"] for item in valid_items]),
        "hidden_bug_detected_ids": [item["code_id"] for item in hidden_bug_items],
        "hidden_bug_detected_count": len(hidden_bug_items),
        "hidden_bug_detection_rate": round(len(hidden_bug_items) * 100.0 / len(valid_items), 2) if valid_items else None,
        "consistent_code_count": len(consistent_items),
        "consistent_rate": round(len(consistent_items) * 100.0 / len(valid_items), 2) if valid_items else None,
    }


def build_failure_section(items: List[Dict]) -> Dict:
    generation_failed_items = [item for item in items if item.get("generation_failed")]
    valid_items = [item for item in items if not item.get("generation_failed")]
    detected_items = [item for item in valid_items if item["program_correct"]]
    undetected_items = [item for item in valid_items if not item["program_correct"]]
    return {
        "program_count": len(items),
        "valid_program_count": len(valid_items),
        "generation_failed_ids": [item["code_id"] for item in generation_failed_items],
        "generation_failed_count": len(generation_failed_items),
        "avg_edge_coverage": average([item["edge_coverage"] for item in valid_items]),
        "avg_block_coverage": average([item["block_coverage"] for item in valid_items]),
        "bug_detected_count": len(detected_items),
        "bug_detection_rate": round(len(detected_items) * 100.0 / len(valid_items), 2) if valid_items else None,
        "undetected_failure_ids": [item["code_id"] for item in undetected_items],
        "undetected_failure_count": len(undetected_items),
    }


def build_output_path(result_dir: str, problem_id: Optional[str]) -> str:
    statistics_dir = os.path.join(result_dir, "statistics")
    os.makedirs(statistics_dir, exist_ok=True)
    prefix = f"stats_{problem_id}_" if problem_id else "stats_"
    return os.path.join(statistics_dir, f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="统计我的算法生成结果")
    parser.add_argument("--result-dir", default=os.path.join(os.path.dirname(__file__), "testcase_result"), help="结果目录，默认是 testcase_result")
    parser.add_argument("--problem", default=None, help="只统计指定题号，例如 2910 / 03039 / 3039 / 3226")
    parser.add_argument("--limit", type=int, default=None, help="只统计前 N 个文件，便于快速验证")
    parser.add_argument("--recompute-coverage", action="store_true", help="复用已有测试输入，重新运行 dynamic_analyzer 计算边/块覆盖率")
    args = parser.parse_args()

    files = sorted(
        p for p in glob(os.path.join(args.result_dir, "testcase_*.txt"))
        if os.path.isfile(p) and matches_problem(p, args.problem)
    )
    files = keep_latest_result_per_code(files)
    if args.limit is not None:
        files = files[: args.limit]

    items = [parse_result_file(path, recompute_coverage=args.recompute_coverage) for path in files]
    success_items, failure_items = split_items(items)
    summary = {
        "problem_id": args.problem,
        "file_count": len(items),
        "success": build_success_section(success_items),
        "failure": build_failure_section(failure_items),
    }

    output_path = build_output_path(args.result_dir, args.problem)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已写入: {output_path}")


if __name__ == "__main__":
    main()
