#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
STATS_DIR = BASE_DIR / "testcase_result" / "statistics"
OUTPUT_DIR = STATS_DIR / "figures"

PROBLEMS = ["03039", "03025", "02882", "2910", "3039", "3226"]
METHODS = ["mine", "pynguin", "mio"]

METHOD_LABELS = {
    "mine": "我的方法",
    "pynguin": "Pynguin-DynaMOSA",
    "mio": "Pynguin-MIO",
}

FILE_PREFIXES = {
    "mine": "stats",
    "pynguin": "pynguin_stats",
    "mio": "mio_stats",
}

COLORS = {
    "mine": "#4C78A8",
    "pynguin": "#F58518",
    "mio": "#54A24B",
}

METRIC_SPECS = [
    ("success_hidden_bug_rate", "success 隐藏错误检出率", "比例 (%)", (0, 100), "{:.1f}"),
    ("failure_bug_detection_rate", "failure 错误检出率", "比例 (%)", (0, 100), "{:.1f}"),
    ("success_edge_coverage", "success 平均边覆盖率", "覆盖率 (%)", (0, 100), "{:.1f}"),
    ("failure_edge_coverage", "failure 平均边覆盖率", "覆盖率 (%)", (0, 100), "{:.1f}"),
    ("success_valid_program_ratio", "success 有效程序比例", "比例 (%)", (0, 100), "{:.1f}"),
    ("failure_valid_program_ratio", "failure 有效程序比例", "比例 (%)", (0, 100), "{:.1f}"),
    ("success_hidden_bug_count", "找到隐藏错误的数量", "数量", None, "{:.0f}"),
    ("success_block_coverage", "success 平均块覆盖率", "覆盖率 (%)", (0, 100), "{:.1f}"),
    ("failure_block_coverage", "failure 平均块覆盖率", "覆盖率 (%)", (0, 100), "{:.1f}"),
    ("exclusive_detected_count", "独有错误检出代码数量", "数量", None, "{:.0f}"),
]


def latest_stats_file(method: str, problem_id: str) -> Optional[Path]:
    prefix = FILE_PREFIXES[method]
    pattern = f"{prefix}_{problem_id}_*.json"
    candidates = sorted(STATS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(num: float, den: float) -> float:
    if not den:
        return 0.0
    return round(num * 100.0 / den, 2)


def safe_float(value) -> float:
    return float(value or 0.0)


def collect_one_method_metrics(data: Optional[Dict]) -> Optional[Dict[str, float]]:
    if data is None:
        return None
    success = data.get("success", {})
    failure = data.get("failure", {})
    success_program_count = success.get("program_count", 0)
    failure_program_count = failure.get("program_count", 0)
    success_valid_program_count = success.get("valid_program_count", 0)
    failure_valid_program_count = failure.get("valid_program_count", 0)

    success_block_coverage = safe_float(success.get("avg_block_coverage"))
    failure_block_coverage = safe_float(failure.get("avg_block_coverage"))
    block_values = [value for value in [success_block_coverage, failure_block_coverage] if value > 0]

    return {
        "success_hidden_bug_rate": safe_float(success.get("hidden_bug_detection_rate")),
        "failure_bug_detection_rate": safe_float(failure.get("bug_detection_rate")),
        "success_edge_coverage": safe_float(success.get("avg_edge_coverage")),
        "failure_edge_coverage": safe_float(failure.get("avg_edge_coverage")),
        "success_block_coverage": success_block_coverage,
        "failure_block_coverage": failure_block_coverage,
        "average_block_coverage": round(sum(block_values) / len(block_values), 2) if block_values else 0.0,
        "success_valid_program_ratio": pct(success_valid_program_count, success_program_count),
        "failure_valid_program_ratio": pct(failure_valid_program_count, failure_program_count),
        "success_hidden_bug_count": safe_float(success.get("hidden_bug_detected_count")),
        "exclusive_detected_count": 0.0,
    }


def collect_hidden_bug_codes(data: Optional[Dict]) -> set[str]:
    if data is None:
        return set()
    return set(data.get("success", {}).get("hidden_bug_detected_ids", []))


def collect_detected_error_codes(data: Optional[Dict]) -> set[str]:
    return collect_hidden_bug_codes(data)


def annotate_bars(ax, bars, values: List[Optional[float]], fmt: str = "{:.1f}") -> None:
    for bar, value in zip(bars, values):
        height = bar.get_height()
        label = "N/A" if value is None else fmt.format(height)
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def values_for_metric(metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]], problems: List[str], method: str, metric_name: str) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for problem_id in problems:
        method_metrics = metrics[problem_id].get(method)
        values.append(None if method_metrics is None else method_metrics[metric_name])
    return values


def plot_grouped_bars(
    ax,
    problems: List[str],
    metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]],
    metric_name: str,
    title: str,
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = (0, 100),
    fmt: str = "{:.1f}",
) -> None:
    x = np.arange(len(problems))
    width = 0.24
    offsets = {
        "mine": -width,
        "pynguin": 0.0,
        "mio": width,
    }
    for method in METHODS:
        raw_values = values_for_metric(metrics, problems, method, metric_name)
        plot_values = [0.0 if value is None else value for value in raw_values]
        bars = ax.bar(
            x + offsets[method],
            plot_values,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            alpha=0.35 if any(value is None for value in raw_values) else 1.0,
        )
        annotate_bars(ax, bars, raw_values, fmt=fmt)
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=25)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def dynamic_count_ylim(metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]], problems: List[str], metric_name: str) -> Tuple[float, float]:
    values: List[float] = []
    for problem_id in problems:
        for method in METHODS:
            method_metrics = metrics[problem_id].get(method)
            if method_metrics is not None:
                values.append(method_metrics.get(metric_name, 0.0))
    return 0.0, max(values + [1.0]) * 1.2


def metric_ylim(metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]], metric_name: str, configured_ylim: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if configured_ylim is not None:
        return configured_ylim
    return dynamic_count_ylim(metrics, PROBLEMS, metric_name)


def metric_filename(metric_name: str) -> str:
    return f"{metric_name}.png"


def save_single_metric_chart(
    metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]],
    metric_name: str,
    title: str,
    ylabel: str,
    ylim: Tuple[float, float],
    fmt: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    plot_grouped_bars(ax, PROBLEMS, metrics, metric_name, title, ylabel, ylim=ylim, fmt=fmt)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_metric_charts(metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]]) -> List[Path]:
    metric_dir = OUTPUT_DIR / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for metric_name, title, ylabel, configured_ylim, fmt in METRIC_SPECS:
        output_path = metric_dir / metric_filename(metric_name)
        save_single_metric_chart(
            metrics,
            metric_name,
            title,
            ylabel,
            metric_ylim(metrics, metric_name, configured_ylim),
            fmt,
            output_path,
        )
        paths.append(output_path)
    return paths


def save_coverage_combined_chart(metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]]) -> Tuple[Path, Path]:
    # Success 覆盖率组合图
    fig_success, axes_success = plt.subplots(1, 2, figsize=(16, 6))
    fig_success.suptitle("成功程序的边覆盖率和块覆盖率对比", fontsize=16, fontweight="bold")
    
    plot_grouped_bars(axes_success[0], PROBLEMS, metrics, "success_edge_coverage", "success 平均边覆盖率", "覆盖率 (%)", ylim=(0, 100))
    plot_grouped_bars(axes_success[1], PROBLEMS, metrics, "success_block_coverage", "success 平均块覆盖率", "覆盖率 (%)", ylim=(0, 100))
    
    handles, labels = axes_success[0].get_legend_handles_labels()
    fig_success.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig_success.tight_layout(rect=(0, 0.05, 1, 0.97))
    
    success_path = OUTPUT_DIR / "coverage_success_combined.png"
    fig_success.savefig(success_path, dpi=300, bbox_inches="tight")
    success_pdf_path = OUTPUT_DIR / "coverage_success_combined.pdf"
    fig_success.savefig(success_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig_success)
    
    # Failure 覆盖率组合图
    fig_failure, axes_failure = plt.subplots(1, 2, figsize=(16, 6))
    fig_failure.suptitle("失败程序的边覆盖率和块覆盖率对比", fontsize=16, fontweight="bold")
    
    plot_grouped_bars(axes_failure[0], PROBLEMS, metrics, "failure_edge_coverage", "failure 平均边覆盖率", "覆盖率 (%)", ylim=(0, 100))
    plot_grouped_bars(axes_failure[1], PROBLEMS, metrics, "failure_block_coverage", "failure 平均块覆盖率", "覆盖率 (%)", ylim=(0, 100))
    
    handles, labels = axes_failure[0].get_legend_handles_labels()
    fig_failure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig_failure.tight_layout(rect=(0, 0.05, 1, 0.97))
    
    failure_path = OUTPUT_DIR / "coverage_failure_combined.png"
    fig_failure.savefig(failure_path, dpi=300, bbox_inches="tight")
    failure_pdf_path = OUTPUT_DIR / "coverage_failure_combined.pdf"
    fig_failure.savefig(failure_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig_failure)
    
    return success_path, failure_path


def save_valid_program_combined_chart(metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("成功程序与失败程序的有效程序比例对比", fontsize=16, fontweight="bold")
    
    plot_grouped_bars(axes[0], PROBLEMS, metrics, "success_valid_program_ratio", "success 有效程序比例", "比例 (%)", ylim=(0, 100))
    plot_grouped_bars(axes[1], PROBLEMS, metrics, "failure_valid_program_ratio", "failure 有效程序比例", "比例 (%)", ylim=(0, 100))
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    
    output_path = OUTPUT_DIR / "valid_program_combined.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = OUTPUT_DIR / "valid_program_combined.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    
    return output_path


def save_bug_detection_rate_combined_chart(metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("成功程序与失败程序的错误检出率对比", fontsize=16, fontweight="bold")
    
    plot_grouped_bars(axes[0], PROBLEMS, metrics, "success_hidden_bug_rate", "success 隐藏错误检出率", "比例 (%)", ylim=(0, 100))
    plot_grouped_bars(axes[1], PROBLEMS, metrics, "failure_bug_detection_rate", "failure 错误检出率", "比例 (%)", ylim=(0, 100))
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    
    output_path = OUTPUT_DIR / "bug_detection_rate_combined.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = OUTPUT_DIR / "bug_detection_rate_combined.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    
    return output_path


def save_bug_count_combined_chart(metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("隐藏错误检出数量对比", fontsize=16, fontweight="bold")
    
    plot_grouped_bars(
        axes[0],
        PROBLEMS,
        metrics,
        "success_hidden_bug_count",
        "找到隐藏错误的数量",
        "数量",
        ylim=dynamic_count_ylim(metrics, PROBLEMS, "success_hidden_bug_count"),
        fmt="{:.0f}",
    )
    plot_grouped_bars(
        axes[1],
        PROBLEMS,
        metrics,
        "exclusive_detected_count",
        "独有错误检出代码数量",
        "数量",
        ylim=dynamic_count_ylim(metrics, PROBLEMS, "exclusive_detected_count"),
        fmt="{:.0f}",
    )
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    
    output_path = OUTPUT_DIR / "bug_count_combined.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = OUTPUT_DIR / "bug_count_combined.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    
    return output_path


def build_exclusive_hidden_bug_lines(data_by_problem: Dict[str, Dict[str, Optional[Dict]]]) -> List[str]:
    lines = ["Exclusive success hidden-bug detected codes by problem and method", ""]
    for problem_id in PROBLEMS:
        method_codes = {method: collect_detected_error_codes(data_by_problem[problem_id].get(method)) for method in METHODS}
        lines.append(f"Problem {problem_id}")
        for method in METHODS:
            other_codes = set().union(*(codes for other, codes in method_codes.items() if other != method))
            exclusive = sorted(method_codes[method] - other_codes)
            lines.append(f"{METHOD_LABELS[method]} only: {len(exclusive)} code(s)")
            if exclusive:
                lines.append(", ".join(exclusive))
        lines.append("")
    return lines


def apply_exclusive_detected_counts(data_by_problem: Dict[str, Dict[str, Optional[Dict]]], metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]]) -> None:
    for problem_id in PROBLEMS:
        method_codes = {method: collect_detected_error_codes(data_by_problem[problem_id].get(method)) for method in METHODS}
        for method in METHODS:
            method_metrics = metrics[problem_id].get(method)
            if method_metrics is None:
                continue
            other_codes = set().union(*(codes for other, codes in method_codes.items() if other != method))
            method_metrics["exclusive_detected_count"] = float(len(method_codes[method] - other_codes))


def build_visualization() -> Tuple[Path, Path, Path, List[Path], Path, Path, Path, Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_by_problem: Dict[str, Dict[str, Optional[Dict]]] = {}
    metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}
    selected_files: Dict[str, Dict[str, Optional[Path]]] = {}

    for problem_id in PROBLEMS:
        data_by_problem[problem_id] = {}
        metrics[problem_id] = {}
        selected_files[problem_id] = {}
        for method in METHODS:
            path = latest_stats_file(method, problem_id)
            selected_files[problem_id][method] = path
            data = load_json(path) if path is not None else None
            data_by_problem[problem_id][method] = data
            metrics[problem_id][method] = collect_one_method_metrics(data)

    apply_exclusive_detected_counts(data_by_problem, metrics)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(4, 3, figsize=(21, 19))
    fig.suptitle("我的方法、Pynguin-DynaMOSA 与 Pynguin-MIO 在各题目上的对比", fontsize=16, fontweight="bold")

    plot_grouped_bars(axes[0, 0], PROBLEMS, metrics, "success_hidden_bug_rate", "success 隐藏错误检出率", "比例 (%)")
    plot_grouped_bars(axes[0, 1], PROBLEMS, metrics, "failure_bug_detection_rate", "failure 错误检出率", "比例 (%)")
    plot_grouped_bars(axes[0, 2], PROBLEMS, metrics, "success_edge_coverage", "success 平均边覆盖率", "覆盖率 (%)")
    plot_grouped_bars(axes[1, 0], PROBLEMS, metrics, "failure_edge_coverage", "failure 平均边覆盖率", "覆盖率 (%)")
    plot_grouped_bars(axes[1, 1], PROBLEMS, metrics, "success_valid_program_ratio", "success 有效程序比例", "比例 (%)")
    plot_grouped_bars(axes[1, 2], PROBLEMS, metrics, "failure_valid_program_ratio", "failure 有效程序比例", "比例 (%)")
    plot_grouped_bars(
        axes[2, 0],
        PROBLEMS,
        metrics,
        "success_hidden_bug_count",
        "找到隐藏错误的数量",
        "数量",
        ylim=dynamic_count_ylim(metrics, PROBLEMS, "success_hidden_bug_count"),
        fmt="{:.0f}",
    )
    plot_grouped_bars(
        axes[2, 1],
        PROBLEMS,
        metrics,
        "success_block_coverage",
        "success 平均块覆盖率",
        "覆盖率 (%)",
    )
    plot_grouped_bars(
        axes[2, 2],
        PROBLEMS,
        metrics,
        "failure_block_coverage",
        "failure 平均块覆盖率",
        "覆盖率 (%)",
    )
    plot_grouped_bars(
        axes[3, 0],
        PROBLEMS,
        metrics,
        "exclusive_detected_count",
        "独有错误检出代码数量",
        "数量",
        ylim=dynamic_count_ylim(metrics, PROBLEMS, "exclusive_detected_count"),
        fmt="{:.0f}",
    )
    axes[3, 1].axis("off")
    axes[3, 2].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    chart_path = OUTPUT_DIR / "comparison.png"
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    chart_pdf_path = OUTPUT_DIR / "comparison.pdf"
    fig.savefig(chart_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    metric_paths = save_metric_charts(metrics)
    coverage_success_path, coverage_failure_path = save_coverage_combined_chart(metrics)
    valid_program_combined_path = save_valid_program_combined_chart(metrics)
    bug_detection_rate_combined_path = save_bug_detection_rate_combined_chart(metrics)
    bug_count_combined_path = save_bug_count_combined_chart(metrics)

    exclusive_path = OUTPUT_DIR / "exclusive_hidden_bug_codes_with_mio.txt"
    exclusive_path.write_text("\n".join(build_exclusive_hidden_bug_lines(data_by_problem)), encoding="utf-8")

    selected_lines = ["Selected statistics files", ""]
    for problem_id in PROBLEMS:
        selected_lines.append(f"Problem {problem_id}")
        for method in METHODS:
            path = selected_files[problem_id][method]
            selected_lines.append(f"{METHOD_LABELS[method]}: {path.name if path else 'N/A'}")
        selected_lines.append("")
    selected_path = OUTPUT_DIR / "selected_stats_files_with_mio.txt"
    selected_path.write_text("\n".join(selected_lines), encoding="utf-8")

    return chart_path, exclusive_path, selected_path, metric_paths, coverage_success_path, coverage_failure_path, valid_program_combined_path, bug_detection_rate_combined_path, bug_count_combined_path


def main() -> None:
    chart_path, exclusive_path, selected_path, metric_paths, coverage_success_path, coverage_failure_path, valid_program_combined_path, bug_detection_rate_combined_path, bug_count_combined_path = build_visualization()
    print(f"[OK] 总览图像已生成: {chart_path}")
    print(f"[OK] 总览矢量图已生成: {chart_path.with_suffix('.pdf')}")
    print(f"[OK] 成功程序覆盖率组合图已生成: {coverage_success_path}")
    print(f"[OK] 成功程序覆盖率组合矢量图已生成: {coverage_success_path.with_suffix('.pdf')}")
    print(f"[OK] 失败程序覆盖率组合图已生成: {coverage_failure_path}")
    print(f"[OK] 失败程序覆盖率组合矢量图已生成: {coverage_failure_path.with_suffix('.pdf')}")
    print(f"[OK] 有效程序比例组合图已生成: {valid_program_combined_path}")
    print(f"[OK] 有效程序比例组合矢量图已生成: {valid_program_combined_path.with_suffix('.pdf')}")
    print(f"[OK] 错误检出率组合图已生成: {bug_detection_rate_combined_path}")
    print(f"[OK] 错误检出率组合矢量图已生成: {bug_detection_rate_combined_path.with_suffix('.pdf')}")
    print(f"[OK] 错误检出数量组合图已生成: {bug_count_combined_path}")
    print(f"[OK] 错误检出数量组合矢量图已生成: {bug_count_combined_path.with_suffix('.pdf')}")
    print(f"[OK] 单指标图像已生成: {len(metric_paths)} 张，目录: {OUTPUT_DIR / 'metrics'}")
    for path in metric_paths:
        print(f"     - {path.name} (PNG + PDF)")
    print(f"[OK] 独有隐藏错误代码列表已生成: {exclusive_path}")
    print(f"[OK] 使用的统计文件列表已生成: {selected_path}")


if __name__ == "__main__":
    main()

