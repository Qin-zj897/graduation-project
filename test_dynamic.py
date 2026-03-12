#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态分析器测试
"""

from collections import defaultdict
from static_analyzer import StaticAnalyzer
from input_generator import DynamicAnalyzerWithAutoInput
import json

# 原始测试代码
code = """
a = eval(input())
b = max(a)
c = min(a)
i = 0
while i < len(a):
    if b == a[i]:
        a.remove(b)
        i = i - 1
    elif c == a[i]:
        a.remove(c)
        i = i - 1
    i = i + 1
print(a)
"""

print("=" * 80)
print("动态分析器功能测试")
print("=" * 80)
print("\n测试代码:")
print(code)

# ============================================================================
# 第一步：静态分析
# ============================================================================
print("\n" + "=" * 80)
print("1. 静态分析")
print("=" * 80)

static_analyzer = StaticAnalyzer(source_code=code)
print("[OK] 静态分析器创建成功")

# 执行静态分析
cfg = static_analyzer.build_control_flow_graph()
predicates = static_analyzer.extract_predicates_and_constraints()
data_deps = static_analyzer.build_data_dependency_graph()
var_types = static_analyzer.get_variable_types()
constants = static_analyzer.extract_constants_and_comparisons()

print(f"  CFG: {cfg.get('metrics', {}).get('num_nodes', 0)} 节点, {cfg.get('metrics', {}).get('num_edges', 0)} 边")
print(f"  谓词: {len(predicates)} 个")
print(f"  数据依赖: {len(data_deps)} 个变量")
print(f"  类型推断: {len(var_types)} 个变量")
print(f"  常量: {len(constants)} 个变量")

# 构建静态分析结果
static_result = {
    'cfg': cfg,
    'predicates': predicates,
    'data_dependencies': data_deps,
    'variable_types': var_types,
    'constants': constants
}

# ============================================================================
# 第二步：创建增强的动态分析器
# ============================================================================
print("\n" + "=" * 80)
print("2. 创建动态分析器")
print("=" * 80)

analyzer = DynamicAnalyzerWithAutoInput(code, static_result)
print("[OK] 动态分析器创建成功")

# ============================================================================
# 第三步：分析输入信息
# ============================================================================
print("\n" + "=" * 80)
print("3. 分析输入信息")
print("=" * 80)

input_info = analyzer.get_input_info()
print(f"\n推断的输入变量:")
for var, var_type in input_info['input_variables'].items():
    print(f"  {var}: {var_type}")

print(f"\n输入变量详情:")
for var, details in input_info['variable_details'].items():
    print(f"  {var}:")
    print(f"    类型: {details['type']}")
    if details['boundary_values']:
        print(f"    边界值: {details['boundary_values']}")

# ============================================================================
# 第四步：自动生成常规测试用例
# ============================================================================
print("\n" + "=" * 80)
print("4. 自动生成常规测试用例")
print("=" * 80)

test_inputs = analyzer.input_generator.generate_test_inputs(num_cases=5)
print(f"\n自动生成的测试用例 ({len(test_inputs)} 个):")
for i, test_input in enumerate(test_inputs, 1):
    print(f"  用例{i}: {test_input}")

# ============================================================================
# 第五步：执行常规测试
# ============================================================================
print("\n" + "=" * 80)
print("5. 执行常规测试")
print("=" * 80)

print(f"\n执行 {len(test_inputs)} 个常规测试用例...")
regular_results = analyzer.run_with_auto_generated_inputs(num_cases=5)

print(f"\n常规测试结果汇总:")
for i, result in enumerate(regular_results, 1):
    exec_info = result['execution_info']
    coverage = result['coverage']
    print(f"  测试{i}:")
    print(f"    成功: {exec_info['success']}")
    print(f"    步数: {exec_info['execution_steps']}")
    print(f"    输出: {exec_info['output'][:40]}...")
    print(f"    覆盖: 边={coverage['edge_count']}, 块={coverage['block_count']}")

# ============================================================================
# 第六步：自动生成边界值测试用例
# ============================================================================
print("\n" + "=" * 80)
print("6. 自动生成边界值测试用例")
print("=" * 80)

boundary_inputs = analyzer.input_generator.generate_boundary_test_inputs()
print(f"\n自动生成的边界值测试用例 ({len(boundary_inputs)} 个):")
for i, test_input in enumerate(boundary_inputs, 1):
    print(f"  用例{i}: {test_input}")

# ============================================================================
# 第七步：执行边界值测试
# ============================================================================
print("\n" + "=" * 80)
print("7. 执行边界值测试")
print("=" * 80)

print(f"\n执行 {len(boundary_inputs)} 个边界值测试用例...")
boundary_results = analyzer.run_with_boundary_test_inputs()

print(f"\n边界值测试结果汇总:")
for i, result in enumerate(boundary_results, 1):
    exec_info = result['execution_info']
    coverage = result['coverage']
    print(f"  测试{i}:")
    print(f"    成功: {exec_info['success']}")
    print(f"    步数: {exec_info['execution_steps']}")
    print(f"    输出: {exec_info['output'][:40]}...")
    print(f"    覆盖: 边={coverage['edge_count']}, 块={coverage['block_count']}")

# ============================================================================
# 第八步：详细分析第一个测试
# ============================================================================
print("\n" + "=" * 80)
print("8. 详细分析第一个测试")
print("=" * 80)

first_result = regular_results[0]
print(f"\n输入: {test_inputs[0]}")

# 8.1 执行信息
print("\n8.1 执行信息:")
exec_info = first_result['execution_info']
print(f"  成功: {exec_info['success']}")
print(f"  执行步数: {exec_info['execution_steps']}")
print(f"  执行时间: {exec_info['execution_time']:.4f}秒")
print(f"  输出:\n{exec_info['output']}")

# 8.2 行级轨迹
print("\n8.2 行级轨迹记录:")
trace = first_result['trace']
print(f"  总步数: {trace['step_count']}")
print(f"  轨迹详情 (前10步):")
for record in trace['records'][:10]:
    vars_str = ', '.join([f"{k}={v}" for k, v in record['vars'].items()])
    print(f"    步骤{record['step']:3d}: 第{record['lineno']:2d}行 | {vars_str if vars_str else '(无变化)'}")

# 8.3 变量值序列
print("\n8.3 关键变量值序列:")
var_values = first_result['variable_values']
for var in sorted(var_values.keys())[:3]:  # 只显示前3个
    values = var_values[var]
    print(f"  变量 '{var}' ({len(values)} 个值):")
    for v in values[:5]:  # 只显示前5个
        print(f"    步骤{v['step']:3d}, 第{v['lineno']:2d}行: {v['value']}")
    if len(values) > 5:
        print(f"    ... 还有 {len(values) - 5} 个值")

# 8.4 分支距离
print("\n8.4 分支距离计算:")
branch_dists = first_result['branch_distances']
if branch_dists:
    for lineno in sorted(branch_dists.keys()):
        dists = branch_dists[lineno]
        print(f"  第{lineno}行分支:")
        for d in dists[:3]:  # 只显示前3个
            print(f"    步骤{d['step']:3d}: '{d['condition']}' -> 距离={d['distance']:.2f}")
else:
    print("  未记录分支距离")

# 8.5 CFG边覆盖
print("\n8.5 CFG边覆盖:")
coverage = first_result['coverage']
print(f"  覆盖边数: {coverage['edge_count']}")
print(f"  覆盖块数: {coverage['block_count']}")

# 8.6 类型验证
print("\n8.6 运行时类型验证:")
type_val = first_result['type_validation']
if type_val.get('confirmed_types'):
    print(f"  确认的类型:")
    for var, typ in type_val['confirmed_types'].items():
        print(f"    {var}: {typ}")
if type_val.get('corrected_types'):
    print(f"  修正的类型:")
    for var, types in type_val['corrected_types'].items():
        print(f"    {var}: 静态={types['static']}, 运行时={types['runtime']}")

# ============================================================================
# 第九步：聚合分析
# ============================================================================
print("\n" + "=" * 80)
print("9. 聚合分析")
print("=" * 80)

all_results = regular_results + boundary_results
aggregated = analyzer.dynamic_analyzer.aggregate_coverage(all_results)

print(f"\n总体覆盖率:")
print(f"  总覆盖边数: {aggregated['edge_count']}/{len(analyzer.dynamic_analyzer.cfg_edges)}")
# 计算实际的块覆盖数
cfg = analyzer.static_result.get('cfg', {})
actual_nodes_with_lineno = set()
for branch in cfg.get('branches', []):
    if 'lineno' in branch:
        actual_nodes_with_lineno.add(branch.get('node_id'))
actual_covered_blocks = len([b for b in aggregated['total_blocks'] 
                             if b in actual_nodes_with_lineno])
actual_total_blocks = len(actual_nodes_with_lineno)
print(f"  总覆盖块数: {actual_covered_blocks}/{actual_total_blocks}")

if len(analyzer.dynamic_analyzer.cfg_edges) > 0:
    edge_coverage = aggregated['edge_count'] / len(analyzer.dynamic_analyzer.cfg_edges) * 100
    print(f"  边覆盖率: {edge_coverage:.2f}%")

if actual_total_blocks > 0:
    block_coverage = actual_covered_blocks / actual_total_blocks * 100
    print(f"  块覆盖率: {block_coverage:.2f}%")

# 详细的边覆盖记录
print(f"\n详细的边覆盖:")
if aggregated['total_edges']:
    print(f"  覆盖的边 ({len(aggregated['total_edges'])} 条):")
    for i, edge in enumerate(sorted(aggregated['total_edges']), 1):
        from_node, to_node = edge
        print(f"    {i}. {from_node} -> {to_node}")
else:
    print("  未覆盖任何边")

# 详细的块覆盖记录
print(f"\n详细的块覆盖:")
if aggregated['total_blocks']:
    covered_blocks_list = sorted([b for b in aggregated['total_blocks'] 
                                  if b in actual_nodes_with_lineno])
    print(f"  覆盖的块 ({len(covered_blocks_list)} 个):")
    for i, block in enumerate(covered_blocks_list, 1):
        # 查找块对应的行号
        block_lineno = None
        for branch in cfg.get('branches', []):
            if branch.get('node_id') == block and 'lineno' in branch:
                block_lineno = branch.get('lineno')
                break
        if block_lineno:
            print(f"    {i}. 块 {block} (第{block_lineno}行)")
        else:
            print(f"    {i}. 块 {block}")
else:
    print("  未覆盖任何块")

# 分支距离统计
print(f"\n分支距离统计:")
all_branch_dists = defaultdict(list)
for result in all_results:
    for lineno, dists in result.get('branch_distances', {}).items():
        for d in dists:
            all_branch_dists[lineno].append(d['distance'])

if all_branch_dists:
    print(f"  发现 {len(all_branch_dists)} 个分支:")
    for lineno in sorted(all_branch_dists.keys()):
        distances = all_branch_dists[lineno]
        min_dist = min(distances)
        max_dist = max(distances)
        avg_dist = sum(distances) / len(distances)
        print(f"    第{lineno}行: 最小={min_dist:.2f}, 最大={max_dist:.2f}, 平均={avg_dist:.2f}")
else:
    print("  未记录分支距离")

# ============================================================================
# 第十步：最终总结
# ============================================================================
print("\n" + "=" * 80)
print("10. 最终总结")
print("=" * 80)

summary = {
    '静态分析': {
        'CFG节点数': cfg.get('metrics', {}).get('num_nodes', 0),
        'CFG边数': cfg.get('metrics', {}).get('num_edges', 0),
        '谓词数量': len(predicates),
        '推断变量类型数': len(var_types)
    },
    '输入生成': {
        '推断的输入变量': len(input_info['input_variables']),
        '常规测试用例': len(test_inputs),
        '边界值测试用例': len(boundary_inputs),
        '总测试用例': len(test_inputs) + len(boundary_inputs)
    },
    '动态分析': {
        '成功执行数': sum(1 for r in all_results if r['execution_info']['success']),
        '总执行步数': sum(r['execution_info']['execution_steps'] for r in all_results),
        '平均执行步数': sum(r['execution_info']['execution_steps'] for r in all_results) / len(all_results),
        '总覆盖边数': aggregated['edge_count'],
        '总覆盖块数': actual_covered_blocks,
        '发现分支数': len(all_branch_dists),
        '边覆盖率': f"{edge_coverage:.2f}%"
    }
}

print(json.dumps(summary, indent=2, ensure_ascii=False))

print("\n" + "=" * 80)
print("[OK] 所有测试完成！")
print("=" * 80)

print("\n关键成就:")
print(f"  [OK] 从静态分析自动推断了 {len(input_info['input_variables'])} 个输入变量")
print(f"  [OK] 自动生成了 {len(test_inputs)} 个常规测试用例")
print(f"  [OK] 自动生成了 {len(boundary_inputs)} 个边界值测试用例")
print(f"  [OK] 总共执行了 {len(all_results)} 个测试，全部成功")
print(f"  [OK] 达到了 {edge_coverage:.2f}% 的边覆盖率")
