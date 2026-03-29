#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TestcaseGenerator 测试脚本

针对指定源码文件运行 TestcaseGenerator，输出生成的测试用例及覆盖率报告。
结果保存到 testcase_result/ 文件夹，文件命名格式：
    testcase_{code_filename}_{timestamp}.txt

用法：
    python test_testcase.py                              # 使用默认配置
    python test_testcase.py 3039/success/s-3039-002.py 3039.xml
    python test_testcase.py 3039/success/s-3039-001.py none
"""

import os
import sys
import json
from datetime import datetime
from testcase_generator import TestcaseGenerator

# 配置（可通过命令行参数覆盖）
DEFAULT_CODE_FILE = "3226/success/s-3226-003.py"
DEFAULT_XML_FILE  = "3226.xml"   # 设为 None 表示不使用 XML 种子

OUTPUT_DIR = "testcase_result"

# GA 参数（可按需调整）
GA_CONFIG = dict(
    population_size  = 20,
    max_generations  = 30,
    tournament_size  = 3,
    crossover_rate   = 0.8,
    mutation_rate    = 0.3,
    elite_ratio      = 0.1,
    stagnation_limit = 8,
    timeout          = 5.0,
)


# ============================================================
# 工具函数
# ============================================================

def make_serializable(obj):
    """将对象递归转换为 JSON 可序列化格式"""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def sep(char="=", width=80):
    return char * width


def print_section(title, fout=None):
    s = "\n" + sep() + "\n" + title + "\n" + sep()
    print(s, file=fout)


# ============================================================
# 主逻辑
# ============================================================

def run_test(code_file, xml_file):
    """运行 TestcaseGenerator 并将结果输出到文件"""

    code_filename = os.path.splitext(os.path.basename(code_file))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"testcase_{code_filename}_{timestamp}.txt")

    term = sys.__stdout__
    print(f"[INFO] 开始测试: {code_file}", file=term, flush=True)
    if xml_file:
        print(f"[INFO] XML 种子: {xml_file}", file=term, flush=True)
    print(
        f"[INFO] GA 参数 — 种群:{GA_CONFIG['population_size']} "
        f"代数:{GA_CONFIG['max_generations']} 超时:{GA_CONFIG['timeout']}s",
        file=term, flush=True,
    )

    with open(output_file, "w", encoding="utf-8") as fh:
        _do_run(code_file, xml_file, code_filename, output_file, fh, term)

    print(f"[OK]  结果已保存到: {output_file}", file=term, flush=True)


def _do_run(code_file, xml_file, code_filename, output_file, fh, term):
    """实际执行，所有输出写入 fh，进度写入 term"""

    def fprint(*args, **kwargs):
        kwargs.setdefault("file", fh)
        print(*args, **kwargs)

    with open(code_file, "r", encoding="utf-8") as f:
        source_code = f.read()

    gen = TestcaseGenerator(
        source_code=source_code,
        xml_file=xml_file,
        **GA_CONFIG,
    )
    print("[INFO] GA 运行中，请稍候…", file=term, flush=True)
    start_ts = datetime.now()
    test_cases = gen.generate()
    elapsed = (datetime.now() - start_ts).total_seconds()
    print(
        f"[OK]  生成完成，耗时 {elapsed:.1f} 秒，共生成 {len(test_cases)} 个用例",
        file=term, flush=True,
    )

    # 覆盖率计算
    covered_ids = set()
    for tc in test_cases:
        covered_ids |= set(tc.get("covered_branches", []))

    branch_infos = [
        (b.get("branch_id", ""), b.get("pred_id", ""), b.get("lineno", 0), b.get("cfg_node_id", ""))
        for b in gen._branches
    ]
    total_cnt   = len(branch_infos)
    covered_cnt = sum(1 for bid, *_ in branch_infos if bid in covered_ids)
    uncovered   = [(bid, pred_id, lineno, node_id)
                    for bid, pred_id, lineno, node_id in branch_infos
                    if bid not in covered_ids]
    branch_cov_rate = covered_cnt / total_cnt * 100 if total_cnt else 0.0

    # ── 测试用例集 ───────────────────────────────────────
    fprint(sep())
    fprint(f"测试用例集 - {code_filename}  （共 {len(test_cases)} 个）")
    fprint(f"代码文件: {code_file}    XML种子: {xml_file or "（无）"}    耗时: {elapsed:.1f}s")
    fprint(sep())

    for i, tc in enumerate(test_cases, 1):
        inputs  = tc.get("input", [])
        covered = sorted(tc.get("covered_branches", []))
        out     = tc.get("output", "").strip()
        fprint(f"\n[{i:02d}] 输入    : {" | ".join(inputs)}")
        fprint(f"     输出    : {out if out else "（无）"}")
        fprint(f"     覆盖分支: {covered if covered else "（无分支信息）"}")

    # ── 边/块覆盖率（通过 DynamicAnalyzer 聚合）────────────────
    da_results = []
    for tc in test_cases:
        try:
            inp_lines = tc.get("input", [])
            res = gen._da.run_with_input(inp_lines)
            da_results.append(res)
        except Exception:
            pass
    if da_results:
        agg = gen._da.aggregate_coverage(da_results)
        all_uncov  = agg.get("uncovered_edges", [])
        all_cov    = agg.get("covered_edges", [])
        all_uncov_blk  = agg.get("uncovered_blocks", [])
        all_cov_blk    = agg.get("covered_blocks", [])
        # 结构边：lineno=="virtual" 或 label 为 true_end/false_end/continue 等内部连接边
        STRUCT_LABELS = {"true_end", "false_end", "continue"}
        # 结构节点：只能通过结构边到达的纯合并节点，无实际语句
        STRUCT_NODE_PREFIXES = ("if_merge", "while_exit", "for_exit", "while_else", "main_entry")
        def is_structural_edge(e):
            return e.get("lineno") == "virtual" or e.get("label") in STRUCT_LABELS
        def is_structural_block(b):
            nid = b.get("node_id", "")
            return b.get("lineno") == "virtual" or any(nid.startswith(p) for p in STRUCT_NODE_PREFIXES)
        # 真实边（对测试有意义）
        real_uncov   = [e for e in all_uncov if not is_structural_edge(e)]
        struct_uncov = [e for e in all_uncov if is_structural_edge(e)]
        real_total   = sum(1 for e in all_cov + all_uncov if not is_structural_edge(e))
        real_covered = sum(1 for e in all_cov if not is_structural_edge(e))
        edge_rate    = real_covered / real_total * 100 if real_total else 0.0
        # 真实块（排除结构节点）
        real_uncov_blk  = [b for b in all_uncov_blk if not is_structural_block(b)]
        struct_uncov_blk = [b for b in all_uncov_blk if is_structural_block(b)]
        block_real_total   = sum(1 for b in all_cov_blk + all_uncov_blk if not is_structural_block(b))
        block_real_covered = sum(1 for b in all_cov_blk if not is_structural_block(b))
        block_rate  = block_real_covered / block_real_total * 100 if block_real_total else 0.0
        block_covered = block_real_covered
        block_total   = block_real_total
    else:
        real_covered = real_total = block_covered = block_total = 0
        edge_rate = block_rate = 0.0
        real_uncov = struct_uncov = real_uncov_blk = struct_uncov_blk = []

    # ── 覆盖率报告 ───────────────────────────────────────
    fprint("\n" + sep())
    fprint("覆盖率报告")
    fprint(sep())
    fprint(f"  分支覆盖率 : {covered_cnt}/{total_cnt} = {branch_cov_rate:.2f}%")
    fprint(f"  边覆盖率   : {real_covered}/{real_total} = {edge_rate:.2f}%  (已排除 {len(struct_uncov)} 条结构边)")
    fprint(f"  块覆盖率   : {block_covered}/{block_total} = {block_rate:.2f}%")
    fprint(f"  已覆盖分支 : {sorted(covered_ids) if covered_ids else "（无）"}")

    if uncovered:
        fprint("  未覆盖分支 :")
        for bid, pred_id, lineno, node_id in uncovered:
            desc = f"pred_id={pred_id}" if pred_id else f"for循环 第{lineno}行"
            fprint(f"    - {bid}  ({desc})  node={node_id}")
    else:
        fprint("  未覆盖分支 : 无（所有分支均已覆盖）")

    if real_uncov:
        fprint("  未覆盖真实CFG边 :")
        for e in real_uncov:
            bid_tag = f"[{e['branch_id']}]" if e.get('branch_id') else ""
            fprint(f"    - {e['from']} -[{e['label']}]->{e['to']}  line={e['lineno']}  {bid_tag}")
    else:
        fprint("  未覆盖真实CFG边 : 无")

    if real_uncov_blk:
        fprint("  未覆盖真实块 :")
        for b in real_uncov_blk:
            bid_tag = f"[{b['branch_id']}]" if b.get('branch_id') else ""
            fprint(f"    - {b['node_id']}  line={b['lineno']}  {bid_tag}")
    else:
        fprint("  未覆盖真实块   : 无")

    struct_excl = struct_uncov + struct_uncov_blk
    if struct_excl:
        struct_node_ids = {b.get("node_id","") for b in struct_uncov_blk}
        fprint(f"  结构节点/边（不计入覆盖率，共 {len(struct_excl)} 项）:")
        for e in struct_uncov:
            fprint(f"    □ 边: {e['from']} -[{e['label']}]->{e['to']}  [分支体内部连接边]")
        for b in struct_uncov_blk:
            fprint(f"    □ 块: {b['node_id']}  [纯合并/入口节点，无实际语句]")
    fprint("\n" + sep())
    fprint(f"结果已保存到: {output_file}")
    fprint(sep())



if __name__ == "__main__":
    argv = sys.argv[1:]
    code_file = argv[0] if len(argv) >= 1 else DEFAULT_CODE_FILE
    xml_file  = argv[1] if len(argv) >= 2 else DEFAULT_XML_FILE
    # 允许传 "none" / "null" 显式表示不使用 XML 种子
    if xml_file and xml_file.lower() in ("none", "null", ""):
        xml_file = None

    if not os.path.isfile(code_file):
        print(f"[ERROR] 源码文件不存在: {code_file}", file=sys.stderr)
        sys.exit(1)
    if xml_file and not os.path.isfile(xml_file):
        print(f"[WARN]  XML 种子文件不存在，将忽略: {xml_file}", file=sys.stderr)
        xml_file = None

    run_test(code_file, xml_file)
