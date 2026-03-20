#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态分析器简化测试
输出保存到test_result文件夹
"""

import json
import xml.etree.ElementTree as ET
from static_analyzer import StaticAnalyzer
from dynamic_analyzer import DynamicAnalyzer
from datetime import datetime
import os
import sys

# 配置
CODE_FILE = "3039/success/s-3039-002.py"  # 要测试的代码文件
XML_FILE = "3039.xml"  # 测试用例文件

# 读取测试代码
with open(CODE_FILE, 'r', encoding='utf-8') as f:
    code = f.read()

# 提取问题编号和文件名（用于输出文件命名）
code_filename = os.path.basename(CODE_FILE).replace('.py', '')

# 创建输出文件
os.makedirs('test_result', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"test_result/test_{code_filename}_{timestamp}.txt"

# 重定向输出到文件
original_stdout = sys.stdout
output_handle = open(output_file, 'w', encoding='utf-8')
sys.stdout = output_handle

def parse_test_cases_from_xml(xml_file):
    """从XML文件解析测试用例"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    test_cases = []
    # 遍历所有子元素
    for i, test_data in enumerate(root, 1):
        if test_data.tag.startswith('testData'):
            input_elem = test_data.find('input')
            output_elem = test_data.find('output')
            
            if input_elem is not None and output_elem is not None:
                input_str = input_elem.text.strip() if input_elem.text else ""
                output_str = output_elem.text.strip() if output_elem.text else ""
                
                # 将多行输入拆分成列表（每行作为一个input()调用）
                input_lines = input_str.split('\n') if input_str else []
                
                test_cases.append({
                    'id': i,
                    'input': input_lines,  # 改为列表
                    'input_display': input_str,  # 保留原始字符串用于显示
                    'expected_output': output_str
                })
    
    return test_cases

def print_result(title, method_name, description, result):
    """统一格式打印结果"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print(f"   方法: {method_name}")
    print(f"   返回: {description}")
    print("-" * 80)
    
    # 尝试 JSON 格式化输出
    try:
        # 转换为可序列化的格式
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                if obj and isinstance(obj[0], tuple):
                    return [make_serializable(list(item)) for item in obj]
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable = make_serializable(result)
        print(json.dumps(serializable, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"JSON序列化失败: {e}")
        print(result)

try:
    print("=" * 80)
    print(f"动态分析器简化测试 - {code_filename}")
    print("=" * 80)

    print(f"\n测试代码文件: {CODE_FILE}")
    print(f"测试用例文件: {XML_FILE}")
    print("\n测试代码:")
    print(code)
    print("=" * 80)

    # 第一步：静态分析
    print("\n" + "=" * 80)
    print("第一步：静态分析")
    print("=" * 80)

    static_analyzer = StaticAnalyzer(source_code=code)
    print("[OK] 静态分析器创建成功")

    # 执行静态分析
    cfg = static_analyzer.build_control_flow_graph()
    predicates = static_analyzer.extract_predicates_and_constraints()
    data_deps = static_analyzer.build_data_dependency_graph()
    var_types = static_analyzer.get_variable_types()
    constants = static_analyzer.extract_constants_and_comparisons()

    static_result = {
        'cfg': cfg,
        'predicates': predicates,
        'data_dependencies': data_deps,
        'variable_types': var_types,
        'constants': constants
    }

    # 第二步：创建动态分析器
    print("\n" + "=" * 80)
    print("第二步：创建动态分析器")
    print("=" * 80)

    analyzer = DynamicAnalyzer(code, static_result)
    print("[OK] 动态分析器创建成功")

    # 第三步：从XML解析测试用例
    print("\n" + "=" * 80)
    print("第三步：从XML解析测试用例")
    print("=" * 80)

    test_cases = parse_test_cases_from_xml(XML_FILE)
    print(f"解析的测试用例 ({len(test_cases)} 个):")
    for tc in test_cases:
        print(f"  用例{tc['id']}: 输入={tc.get('input_display', tc['input'])}, 期望输出={tc['expected_output']}")

    # 第四步：执行测试
    print("\n" + "=" * 80)
    print("第四步：执行测试")
    print("=" * 80)

    results = []
    for tc in test_cases:
        print(f"\n执行测试用例 {tc['id']}: {tc.get('input_display', tc['input'])}")
        result = analyzer.run_with_input(tc['input'])
        results.append({
            'test_case': tc,
            'result': result
        })
        
        # 打印结果
        print_result(
            f"测试用例 {tc['id']} 的动态分析结果",
            f"run_with_input(['{tc['input']}'])",
            "dict - 包含execution_info、trace、coverage、variable_values、branch_distances、type_validation",
            result
        )
        
        # 验证输出
        actual_output = result['execution_info']['output'].strip()
        expected_output = tc['expected_output']
        match = actual_output == expected_output
        print(f"\n验证: 期望={expected_output}, 实际={actual_output}, 匹配={match}")


    # 第五步：聚合分析
    print("\n" + "=" * 80)
    print("第五步：聚合分析")
    print("=" * 80)

    aggregated = analyzer.aggregate_coverage([r['result'] for r in results])
    print_result(
        "聚合覆盖率分析",
        "aggregate_coverage(results)",
        "dict - 包含total_edges、total_blocks、edge_count、block_count",
        aggregated
    )

    # 增量覆盖专项测试（与总结中边/块覆盖率一致，标注静态分析 B1/B2 标识）
    print("\n" + "=" * 80)
    print("增量覆盖统计验证")
    print("=" * 80)
    print("\n--- 各轮次增量覆盖明细 ---")
    for r in results:
        tc  = r["test_case"]
        cov = r["result"].get("coverage", {})
        print("\n  [用例 {}] 输入: {}".format(tc["id"], tc.get("input_display", tc["input"])))
        print("    总边数/块数    : {} / {}".format(
            cov.get("total_edges", 0), cov.get("total_blocks", 0)))
        print("    累计边覆盖率   : {}/{} ({:.2%})".format(
            cov.get("covered_edge_count", 0), cov.get("total_edges", 0), cov.get("edge_coverage_rate", 0)))
        print("    累计块覆盖率   : {}/{} ({:.2%})".format(
            cov.get("covered_block_count", 0), cov.get("total_blocks", 0), cov.get("block_coverage_rate", 0)))
        newly_e = cov.get("newly_covered_edges", [])
        if newly_e:
            print("    本轮新增覆盖边 ({} 条):".format(len(newly_e)))
            for e in newly_e:
                bid = "[{}]".format(e["branch_id"]) if e["branch_id"] else ""
                print("      + {} -[{}]-> {}  line={}  {}".format(
                    e["from"], e["label"], e["to"], e["lineno"], bid))
        else:
            print("    本轮新增覆盖边 : 无")
        uncov_e = cov.get("uncovered_edges", [])
        if uncov_e:
            print("    仍未覆盖边    ({} 条):".format(len(uncov_e)))
            for e in uncov_e:
                bid = "[{}]".format(e["branch_id"]) if e["branch_id"] else ""
                print("      - {} -[{}]-> {}  line={}  {}".format(
                    e["from"], e["label"], e["to"], e["lineno"], bid))
        else:
            print("    仍未覆盖边    : 全部已覆盖")
    print("\n--- 聚合后最终覆盖状态 ---")
    print("  边覆盖率 : {}/{} ({:.2%})".format(
        aggregated.get("covered_edge_count", 0), aggregated.get("total_edges", 0),
        aggregated.get("edge_coverage_rate", 0)))
    print("  块覆盖率 : {}/{} ({:.2%})".format(
        aggregated.get("covered_block_count", 0), aggregated.get("total_blocks", 0),
        aggregated.get("block_coverage_rate", 0)))
    print("  已覆盖边 :")
    for e in aggregated.get("covered_edges", []):
        bid = "[{}]".format(e["branch_id"]) if e["branch_id"] else ""
        print("    + {} -[{}]-> {}  line={}  {}".format(
            e["from"], e["label"], e["to"], e["lineno"], bid))
    print("  未覆盖边 :")
    for e in aggregated.get("uncovered_edges", []):
        bid = "[{}]".format(e["branch_id"]) if e["branch_id"] else ""
        print("    - {} -[{}]-> {}  line={}  {}".format(
            e["from"], e["label"], e["to"], e["lineno"], bid))
    if not aggregated.get("uncovered_edges"):
        print("    （无）")

    # 第六步：总结
    print("\n" + "=" * 80)
    print("第六步：总结")
    print("=" * 80)
    successful_tests = sum(1 for r in results if r["result"]["execution_info"]["success"])
    correct_outputs  = sum(1 for r in results if r["result"]["execution_info"]["output"].strip() == r["test_case"]["expected_output"])
    total_cfg_edges  = cfg.get("metrics", {}).get("num_edges", 0)
    total_cfg_nodes  = cfg.get("metrics", {}).get("num_nodes", 0)
    edge_coverage_rate  = aggregated.get("edge_coverage_rate",  0) * 100
    block_coverage_rate = aggregated.get("block_coverage_rate", 0) * 100
    summary = {
        "静态分析": {
            "CFG节点数": cfg.get("metrics", {}).get("num_nodes", 0),
            "CFG边数": cfg.get("metrics", {}).get("num_edges", 0),
            "谓词数量": len(predicates),
            "推断变量类型数": len(var_types)
        },
        "测试执行": {
            "总测试用例": len(test_cases),
            "成功执行数": successful_tests,
            "输出正确数": correct_outputs,
            "成功率": "{:.1f}%".format(successful_tests/len(test_cases)*100) if test_cases else "0%",
            "正确率": "{:.1f}%".format(correct_outputs/len(test_cases)*100) if test_cases else "0%"
        },
        "动态分析": {
            "总执行步数": sum(r["result"]["execution_info"]["execution_steps"] for r in results),
            "平均执行步数": sum(r["result"]["execution_info"]["execution_steps"] for r in results) / len(results) if results else 0,
            "已覆盖边数": aggregated.get("covered_edge_count", 0),
            "总CFG边数": aggregated.get("total_edges", 0),
            "边覆盖率": "{:.2f}%".format(edge_coverage_rate),
            "已覆盖块数": aggregated.get("covered_block_count", 0),
            "总CFG块数": aggregated.get("total_blocks", 0),
            "块覆盖率": "{:.2f}%".format(block_coverage_rate)
        }
    }
    print_result("最终总结", "summary", "dict - 包含静态分析、测试执行、动态分析的统计信息", summary)
    print("\n" + "=" * 80)
    print("测试完成！")
    print("结果已保存到: {}".format(output_file))
    print("=" * 80)

finally:
    sys.stdout = original_stdout
    output_handle.close()
    print("[OK] 测试完成！结果已保存到: {}".format(output_file))
    print("[INFO] 测试代码: {}".format(CODE_FILE))
    print("[INFO] 测试用例: {}".format(XML_FILE))
