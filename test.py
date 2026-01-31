# test.py
import ast
from static_analyzer import StaticAnalyzer, VariableFinder

source_code = '''h=eval(input())
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
print(\"%.2f\" %sum1)
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
        
        # 3. 测试控制流图构建
        print("\n2. 构建控制流图:")
        cfg_result = analyzer.build_control_flow_graph()
        if cfg_result:
            print(f"  节点数量: {cfg_result['graph_metrics']['num_nodes']}")
            print(f"  边数量: {cfg_result['graph_metrics']['num_edges']}")
            print(f"  函数数量: {cfg_result['graph_metrics']['num_functions']}")
            
            # CFG 详细验证
            print("\n  CFG 结构验证:")
            nodes = cfg_result['nodes']
            edges = cfg_result['edges']
            
            # 统计节点类型
            node_types = {}
            for node_id, node_attrs in nodes:
                node_type = node_attrs.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            print(f"    节点类型分布: {dict(node_types)}")
            
            # 检查函数定义是否重复
            func_entry_count = node_types.get('func_entry', 0)
            func_exit_count = node_types.get('func_exit', 0)
            if func_entry_count > 0:
                print(f"    函数入口块: {func_entry_count} 个")
                print(f"    函数出口块: {func_exit_count} 个")
                
                # 验证函数块数量是否匹配
                expected_func_count = cfg_result['graph_metrics']['num_functions']
                if func_entry_count == expected_func_count and func_exit_count == expected_func_count:
                    print(f"    [OK] 函数块数量正确，无重复")
                else:
                    print(f"    [WARN] 函数块数量异常: 期望{expected_func_count}个，实际入口{func_entry_count}个，出口{func_exit_count}个")
            
            # 统计边类型
            edge_types = {}
            edge_labels = {}
            for from_node, to_node, edge_attrs in edges:
                edge_type = edge_attrs.get('type', 'normal')
                edge_label = edge_attrs.get('label', 'unlabeled')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
                edge_labels[edge_label] = edge_labels.get(edge_label, 0) + 1
            
            print(f"    边类型分布: {dict(edge_types)}")
            
            # 检查 return 边
            return_edge_count = edge_labels.get('return', 0)
            if return_edge_count > 0:
                print(f"    return 边: {return_edge_count} 个")
                
                # 检查 return 边是否正确连接到函数出口
                return_edges_valid = True
                for from_node, to_node, edge_attrs in edges:
                    if edge_attrs.get('label') == 'return':
                        # 检查目标节点是否是函数出口块
                        to_node_type = None
                        for node_id, node_attrs in nodes:
                            if node_id == to_node:
                                to_node_type = node_attrs.get('type')
                                break
                        
                        if to_node_type and 'func_exit' not in to_node_type and 'main_entry' not in to_node_type:
                            return_edges_valid = False
                            break
                
                if return_edges_valid:
                    print(f"    [OK] return 边正确连接")
                else:
                    print(f"    [WARN] 部分 return 边连接异常")
            
            # 检查函数调用边和返回边
            call_edge_count = edge_types.get('call_edge', 0)
            return_edge_type_count = edge_types.get('return_edge', 0)
            if call_edge_count > 0 or return_edge_type_count > 0:
                print(f"    函数调用边: {call_edge_count} 个")
                print(f"    函数返回边: {return_edge_type_count} 个")
                
                if call_edge_count == return_edge_type_count:
                    print(f"    [OK] 调用边和返回边数量匹配")
                else:
                    print(f"    [WARN] 调用边({call_edge_count})和返回边({return_edge_type_count})数量不匹配")
            
            # 检查循环结构
            has_loop = any('while' in t or 'for' in t for t in node_types.keys())
            if has_loop:
                has_continue_edge = 'continue' in edge_labels
                if has_continue_edge:
                    print(f"    [OK] 检测到循环结构，包含回边")
                else:
                    print(f"    [WARN] 检测到循环结构，但缺少回边")
            
            # 检查条件分支
            has_if = any('if' in t for t in node_types.keys())
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
        
        # 4. 测试谓词挖掘
        print("\n3. 谓词挖掘:")
        predicates_result = analyzer.predicate_mining()
        if predicates_result:
            print(f"  发现 {len(predicates_result)} 个谓词")
            
            for i, pred in enumerate(predicates_result, 1):
                print(f"\n  谓词 {i}:")
                print(f"    表达式: {pred['expression']}")
                
                if 'var' in pred:
                    print(f"    主要变量: {pred['var']}")
                
                if 'value' in pred:
                    print(f"    比较值: {pred['value']}")
                        
                if 'boundary_values' in pred:
                    print(f"    边界测试建议: {pred['boundary_values']}")
        else:
            print(f"  未发现谓词")
        
        # 5. 测试链式比较分析
        print("\n4. 链式比较分析:")
        chained_result = analyzer.analyze_chained_comparisons()
        if chained_result:
            print(f"  发现 {len(chained_result)} 个链式比较")
            for i, comp in enumerate(chained_result, 1):
                print(f"\n  链式比较 {i}:")
                print(f"    表达式: {comp['expression']}")
                print(f"    主变量: {comp['main_variable']}")
                print(f"    边界: {comp['boundaries']}")
                if comp.get('test_values'):
                    print(f"    测试值建议: {comp['test_values']}")
        else:
            print(f"  未发现链式比较")
        
        # 6. 测试数据依赖分析
        print("\n5. 数据依赖分析:")
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
        
        # 7. 测试变量类型推断
        print("\n6. 变量类型推断:")
        var_types_result = analyzer.get_variable_types()
        if var_types_result:
            print(f"  发现 {len(var_types_result)} 个变量的类型:")
            for var, var_type in var_types_result.items():
                print(f"    {var}: {var_type}")
        else:
            print(f"  未推断出变量类型")
        
        # 8. 测试输入结构推断
        print("\n7. 输入结构推断:")
        input_result = analyzer.infer_input_structure()
        if input_result:
            print(f"  发现 {len(input_result)} 个需要变异的数据:")
            for var, var_type in input_result.items():
                print(f"    {var}: {var_type}")
        else:
            print(f"  未发现需要变异的数据")
        
        # 8. 测试后向切片
        print("\n8. 后向切片测试:")
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