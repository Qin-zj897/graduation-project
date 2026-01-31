#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
静态分析器简化测试 - 统一格式
直接打印每个方法的返回值，使用 JSON 格式化输出
"""

import json
from static_analyzer import StaticAnalyzer

# 测试代码
code = """
h=eval(input())
n=eval(input())
x=input().split()
alp=0.5
sum1=h
if h % 11 == 0:
    pass
if h and n:
    pass
if n > 10:
    print("n>10")
if 20 > n > 10:
    print("yes")
for i in range(1,n):
    sum1+=h*alp*2
    h=h*alp
print("%.2f" %sum1)
"""

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
                # 检查是否是元组列表（如 CFG 的 nodes/edges）
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

print("=" * 80)
print("静态分析器简化测试 - 所有方法返回值")
print("=" * 80)

print("\n测试代码:")
print(code)
print("=" * 80)

# 创建分析器实例
analyzer = StaticAnalyzer(code)

# 1. 提取常量与比较值
result = analyzer.extract_constants_and_comparisons()
print_result(
    "1. 提取常量与比较值",
    "extract_constants_and_comparisons()",
    "dict - 变量名到常量值列表的映射",
    result
)

# 2. 构建控制流图
result = analyzer.build_control_flow_graph()
print_result(
    "2. 构建控制流图",
    "build_control_flow_graph()",
    "dict - 包含nodes, edges, graph_metrics等",
    result
)

# 3. 谓词挖掘
result = analyzer.predicate_mining()
print_result(
    "3. 谓词挖掘",
    "predicate_mining()",
    "list - 谓词信息列表",
    result
)

# 4. 链式比较分析
result = analyzer.analyze_chained_comparisons()
print_result(
    "4. 链式比较分析",
    "analyze_chained_comparisons()",
    "list - 链式比较信息列表",
    result
)

# 5. 数据依赖分析
result = analyzer.build_data_dependency_graph()
print_result(
    "5. 数据依赖分析",
    "build_data_dependency_graph()",
    "dict - 变量名到依赖变量列表的映射",
    result
)

# 6. 变量类型推断
result = analyzer.get_variable_types()
print_result(
    "6. 变量类型推断",
    "get_variable_types()",
    "dict - 变量名到类型字符串的映射",
    result
)

# 7. 输入结构推断
result = analyzer.infer_input_structure()
print_result(
    "7. 输入结构推断",
    "infer_input_structure()",
    "dict - 变量名到类型的映射（排除循环变量）",
    result
)

# 8. 后向切片
result = analyzer.backward_slice('sum1', 17)
print_result(
    "8. 后向切片",
    "backward_slice('sum1', 17)",
    "dict - 包含boundary_variables和data_flow_paths",
    result
)

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
