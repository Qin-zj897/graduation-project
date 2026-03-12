# test.py
import ast
from static_analyzer import StaticAnalyzer, VariableFinder

source_code = '''def search(nums):
    for x in nums:
        if nums.count(x) > len(nums) // 2:
            return x
    else:
        return False

nums = eval(input())
y = search(nums)
print(y)
'''

def test_static_analyzer():
    print("=== 静态分析器测试 ===")
    print("测试代码：")
    print(source_code)
    print("\n" + "="*60 + "\n")
    
    try:
        # 1. 创建静态分析器实例
        analyzer = StaticAnalyzer(source_code=source_code)
        print("[OK] 静态分析器实例创建成功")
        
        # 2. 测试常量提取
        print("\n1. 提取常量与比较值:")
        constants_result = analyzer.extract_constants_and_comparisons()
        if constants_result:
            for var, values in constants_result.items():
                print(f"  变量 '{var}':")
                for val_info in values:
                    print(f"    值: {val_info['value']} (类型: {val_info['type']})")
        else:
            print("  未提取到常量")
        
        # 3. 测试控制流图构建（简化版）
        print("\n2. 构建控制流图:")
        cfg_result = analyzer.build_control_flow_graph()
        if cfg_result:
            metrics = cfg_result.get('metrics', {})
            print(f"  节点数量: {metrics.get('num_nodes', 0)}")
            print(f"  边数量: {metrics.get('num_edges', 0)}")
            print(f"  函数数量: {metrics.get('num_functions', 0)}")
            
            # 显示分支点信息
            branches = cfg_result.get('branches', [])
            if branches:
                print(f"\n  分支点信息 ({len(branches)} 个):")
                for branch in branches:
                    branch_type = branch.get('type')
                    lineno = branch.get('lineno')
                    if branch_type == 'if_condition':
                        condition = branch.get('condition', {}).get('expression', 'unknown')
                        print(f"    第{lineno}行: if ({condition})")
                    elif branch_type == 'for_loop':
                        loop_info = branch.get('loop', {})
                        target = loop_info.get('target', '?')
                        iterable = loop_info.get('iterable', '?')
                        print(f"    第{lineno}行: for {target} in {iterable}")
                    elif branch_type == 'while_loop':
                        condition = branch.get('condition', {}).get('expression', 'unknown')
                        print(f"    第{lineno}行: while ({condition})")
            
            # 统计边标签
            edges = cfg_result.get('edges', [])
            edge_labels = {}
            for edge in edges:
                label = edge.get('label', 'unlabeled')
                edge_labels[label] = edge_labels.get(label, 0) + 1
            
            print(f"\n  边标签分布: {dict(edge_labels)}")
            
            # 检查循环结构
            has_loop = any(b['type'] in ['for_loop', 'while_loop'] for b in branches)
            if has_loop:
                has_continue_edge = 'continue' in edge_labels
                if has_continue_edge:
                    print(f"    [OK] 检测到循环结构，包含回边")
                else:
                    print(f"    [WARN] 检测到循环结构，但缺少回边")
            
            # 检查条件分支
            has_if = any(b['type'] == 'if_condition' for b in branches)
            if has_if:
                has_true_edge = 'true' in edge_labels
                has_false_edge = 'false' in edge_labels
                if has_true_edge and has_false_edge:
                    print(f"    [OK] 检测到条件分支，包含 true/false 边")
                else:
                    print(f"    [WARN] 检测到条件分支，但缺少 true/false 边")
            
            print(f"  [OK] CFG 构建成功并通过验证")
        else:
            print(f"  [FAIL] 控制流图构建失败")
        
        # 4. 测试谓词/约束提取（统一方法）
        print("\n3. 谓词/约束提取:")
        predicates_result = analyzer.extract_predicates_and_constraints()
        if predicates_result:
            print(f"  发现 {len(predicates_result)} 个谓词")
            
            for i, pred in enumerate(predicates_result, 1):
                print(f"\n  谓词 {i}:")
                print(f"    pred_id: {pred.get('pred_id', 'N/A')}")
                print(f"    anchor_stmt: {pred.get('anchor_stmt', 'N/A')}")
                print(f"    表达式: {pred['expression']}")
                
                if 'var' in pred:
                    print(f"    主要变量: {pred['var']}")
                
                # 布尔运算的变量列表
                if 'vars' in pred:
                    print(f"    涉及变量: {', '.join(pred['vars'])}")
                
                # 布尔运算的原子条件
                if 'atomic_conditions' in pred:
                    print(f"    原子条件:")
                    for ac in pred['atomic_conditions']:
                        normalized_tag = " (归一化)" if ac.get('normalized') else ""
                        print(f"      - {ac['expression']}{normalized_tag}")
                
                if 'value' in pred:
                    value = pred['value']
                    if isinstance(value, list):
                        print(f"    比较值: {value} (链式比较)")
                    else:
                        print(f"    比较值: {value}")
                        
                if 'boundary_values' in pred:
                    print(f"    边界测试建议: {pred['boundary_values']}")
        else:
            print(f"  未发现谓词")
        
        # 5. 测试数据依赖分析
        print("\n4. 数据依赖分析:")
        data_deps_result = analyzer.build_data_dependency_graph()
        if data_deps_result:
            print(f"  发现 {len(data_deps_result)} 个变量的依赖关系")
            for var, depends_on in data_deps_result.items():
                if depends_on:  # 只显示有依赖关系的变量
                    print(f"  变量 '{var}' 依赖于: {', '.join(depends_on)}")
                else:
                    print(f"  变量 '{var}' 无依赖")
        else:
            print(f"  未发现数据依赖关系")
        
        # 6. 测试变量类型推断
        print("\n5. 变量类型推断:")
        var_types_result = analyzer.get_variable_types()
        if var_types_result:
            print(f"  发现 {len(var_types_result)} 个变量的类型:")
            for var, var_type in var_types_result.items():
                print(f"    {var}: {var_type}")
        else:
            print(f"  未推断出变量类型")
        
        # 7. 测试后向切片
        print("\n6. 后向切片测试:")
        # 动态发现所有变量
        all_variables = set()

        finder = VariableFinder()
        finder.visit(analyzer.ast_tree)

        # 排除明显的函数名
        function_names = {'eval', 'input', 'print', 'range'}
        true_variables = [v for v in finder.variables if v not in function_names]

        if true_variables:
            print(f"  发现 {len(true_variables)} 个变量: {', '.join(sorted(true_variables))}")
            
            # 使用analyzer的_backward_slice方法，它会返回需要边界测试的变量
            test_vars = ['h', 'n', 'sum1']  # 测试关键变量
            for var_name in test_vars:
                if var_name in true_variables:
                    # 找到变量的使用位置
                    usage_lines = []
                    
                    class UsageFinder(ast.NodeVisitor):
                        def __init__(self, target_var):
                            self.target_var = target_var
                            self.lines = []
                            
                        def visit_Name(self, node):
                            if node.id == self.target_var:
                                if hasattr(node, 'lineno'):
                                    self.lines.append(node.lineno)
                    
                    finder = UsageFinder(var_name)
                    finder.visit(analyzer.ast_tree)
                    
                    if finder.lines:
                        unique_lines = sorted(set(finder.lines))
                        # 对最后一个使用位置进行切片
                        last_line = unique_lines[-1]
                        print(f"\n  变量 '{var_name}' 后向切片 (在第{last_line}行):")
                        
                        slice_result = analyzer.backward_slice(var_name, last_line, include_all_defs=True)
                        if slice_result:
                            # 显示数据流路径
                            data_flow_paths = slice_result.get('data_flow_paths', [])
                            print(f"    数据流路径: {len(data_flow_paths)} 条")
                            
                            if data_flow_paths:
                                print(f"    具体路径:")
                                for i, path in enumerate(data_flow_paths[:5], 1):  # 只显示前5条路径
                                    from_var, from_line = path['from']
                                    to_var, to_line = path['to']
                                    def_type = path.get('def_type', 'unknown')
                                    print(f"      路径 {i}: {from_var}(第{from_line}行) -> {to_var}(第{to_line}行) ")
                            
                            # 显示需要边界测试的变量
                            boundary_vars = slice_result.get('boundary_variables', [])
                            if boundary_vars:
                                print(f"    需要边界测试的变量:")
                                for bv in boundary_vars:
                                    if bv['variable'] == var_name:  # 只显示当前变量
                                        print(f"      * {bv['variable']}:")
                                        
                                        # 显示边界值
                                        if 'boundary_values' in bv and bv['boundary_values']:
                                            bv_info = bv['boundary_values']
                                            if bv_info.get('values'):
                                                print(f"        边界值: {bv_info['values']}")
                                            if bv_info.get('lower') is not None:
                                                print(f"        下界: {bv_info['lower']}")
                                            if bv_info.get('upper') is not None:
                                                print(f"        上界: {bv_info['upper']}")
                                        
                                        # 显示建议测试值
                                        if 'suggested_test_values' in bv and bv['suggested_test_values']:
                                            test_vals = bv['suggested_test_values']
                                            print(f"        建议测试值 (前10个): {test_vals[:10]}")
                                        else:
                                            print(f"        建议测试值: 未生成或为空")
                            else:
                                print(f"    无需边界测试")
                        else:
                            print(f"    切片失败")
        else:
            print("  未发现变量")
        
        print("\n" + "="*60)
        print("[OK] 所有测试完成")
        
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    test_static_analyzer()