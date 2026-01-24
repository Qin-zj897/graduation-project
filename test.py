# test.py
import ast
from static_analyzer import StaticAnalyzer

source_code = '''
h=eval(input())
n=eval(input())
alp=0.5
sum1=h
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
        print("✓ 静态分析器实例创建成功")
        
        # 2. 测试常量提取
        print("\n1. 提取常量与比较值:")
        constants = analyzer.extract_constants_and_comparisons()
        for var, values in constants.items():
            print(f"  变量 '{var}':")
            for val_info in values:
                print(f"    值: {val_info['value']} (类型: {val_info['type']}, 来源: {val_info['source']})")
        
        # 3. 测试控制流图构建
        print("\n2. 构建控制流图:")
        cfg = analyzer.build_control_flow_graph()
        print(f"  节点数量: {len(cfg.nodes())}")
        print(f"  边数量: {len(cfg.edges())}")
        print("  节点列表:")
        for node in sorted(cfg.nodes()):
            node_type = cfg.nodes[node].get('type', 'unknown')
            print(f"    - {node} (类型: {node_type})")
        
        # 4. 测试谓词挖掘
        print("\n3. 谓词挖掘:")
        predicates = analyzer.predicate_mining()
        for i, pred in enumerate(predicates, 1):
            print(f"  谓词 {i}:")
            print(f"    表达式: {pred['expression']}")
            print(f"    类型: {pred['type']}")
            print(f"    位置: 第{pred['lineno']}行")
            if 'boundary_info' in pred and pred['boundary_info']:
                print(f"    边界信息: {pred['boundary_info']}")
        
        # 5. 测试数据依赖分析
        print("\n4. 数据依赖分析:")
        data_deps = analyzer.build_data_dependency_graph()
        if data_deps:
            for var, deps in data_deps.items():
                if deps:  # 只显示有依赖关系的变量
                    print(f"  变量 '{var}' 的依赖:")
                    for dep in deps:
                        if dep['type'] == 'def-use':
                            print(f"    定义在第{dep['from_line']}行，使用在第{dep['to_line']}行")
                        elif dep['type'] == 'data_dependency':
                            print(f"    依赖变量: {dep['depends_on']} (在第{dep['line']}行)")
        else:
            print("  未发现数据依赖关系")
        
        # 6. 测试输入结构推断
        print("\n5. 输入结构推断:")
        try:
            input_struct = analyzer.infer_input_structure()
            
            has_info = False
            
            # 显示函数信息
            if input_struct['functions']:
                has_info = True
                print("  函数信息:")
                for func in input_struct['functions']:
                    print(f"    函数: {func['name']}")
                    if func['args']:
                        print("      参数:")
                        for arg in func['args']:
                            type_info = arg['type'] if arg['type'] else "未知类型"
                            print(f"        - {arg['name']}: {type_info}")
                    
                    if func['returns']:
                        print(f"      返回值: {func['returns']}")
            
            # 显示全局输入（没有函数定义时）
            if input_struct.get('global_inputs'):
                has_info = True
                print("  输入分析:")
                for inp in input_struct['global_inputs']:
                    print(f"    第{inp['line']}行: {inp.get('description', '输入')}")
                    if inp.get('variable'):
                        print(f"      赋值给变量: {inp['variable']}")
            
            # 显示检测到的变量类型
            if input_struct.get('detected_types'):
                has_info = True
                print("  检测到的变量类型:")
                for var, types in input_struct['detected_types'].items():
                    print(f"    {var}: {', '.join(types)}")
            
            if not has_info:
                print("  未发现明显的输入结构模式")
                
        except Exception as e:
            print(f"  输入结构推断出错: {e}")
        
        # 7. 测试变量类型推断
        print("\n6. 变量类型推断:")
        try:
            var_types = analyzer.get_variable_types()
            if var_types:
                print("  所有变量类型推断:")
                
                # 收集所有变量
                all_vars = list(var_types.keys())
                
                if all_vars:
                    # 按字母排序显示
                    for var in sorted(all_vars):
                        var_type = var_types[var]
                        
                        # 提供更友好的类型描述
                        type_desc = {
                            'Unknown': '未知类型',
                            'Any': '任意类型',
                            'list': '列表',
                            'list_element': '列表元素',
                            'int': '整数',
                            'str': '字符串',
                            'bool': '布尔值',
                            'float': '浮点数'
                        }.get(var_type, var_type)
                        # 查找变量在代码中的使用位置
                        usage_lines = []
                        for node in ast.walk(analyzer.ast_tree):
                            if isinstance(node, ast.Name) and node.id == var:
                                if hasattr(node, 'lineno'):
                                    usage_lines.append(node.lineno)
                        
                        usage_info = ""
                        if usage_lines:
                            unique_lines = sorted(set(usage_lines))
                            if len(unique_lines) <= 3:
                                usage_info = f" (使用于第{', '.join(map(str, unique_lines))}行)"
                            else:
                                usage_info = f" (使用于第{unique_lines[0]}, {unique_lines[1]}, ... 行)"
                        
                        print(f"    {var:10} : {type_desc:15}{usage_info}")
                else:
                    print("  未发现变量")
            else:
                print("  未推断出变量类型")
        except Exception as e:
            print(f"  变量类型推断失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 8. 测试后向切片
        print("\n7. 后向切片测试:")
        try:
            # 动态发现所有变量
            all_variables = set()
            
            class VariableFinder(ast.NodeVisitor):
                def __init__(self):
                    self.variables = set()
                    
                def visit_Assign(self, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.variables.add(target.id)
                    
                def visit_For(self, node):
                    if isinstance(node.target, ast.Name):
                        self.variables.add(node.target.id)
                    
                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Store):  # 只关心被赋值的变量
                        self.variables.add(node.id)
            
            finder = VariableFinder()
            finder.visit(analyzer.ast_tree)
            
            # 排除明显的函数名
            function_names = {'eval', 'input', 'print', 'range'}
            true_variables = [v for v in finder.variables if v not in function_names]
            
            if true_variables:
                print(f"  发现 {len(true_variables)} 个变量: {', '.join(sorted(true_variables))}")
                
                for var_name in sorted(true_variables):
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
                        print(f"\n  变量 '{var_name}' 分析:")
                        print(f"    使用位置: 第{', '.join(map(str, unique_lines))}行")
                        
                        # 对重要的使用位置进行切片（选择第一个和最后一个）
                        if len(unique_lines) > 0:
                            # 第一个使用位置
                            first_line = unique_lines[0]
                            try:
                                slice_result = analyzer.backward_slice(var_name, first_line)
                                if slice_result:
                                    affected = sorted(slice_result)
                                    if len(affected) > 1:
                                        print(f"    在第{first_line}行定义的后向切片: 影响 {len(affected)} 行")
                                        # 显示关键的几行
                                        source_lines = analyzer.source_code.split('\n')
                                        for line_no in affected[:3]:  # 只显示前3行
                                            if 1 <= line_no <= len(source_lines):
                                                code = source_lines[line_no-1].strip()
                                                if code:
                                                    print(f"      第{line_no:2d}行: {code}")
                                        if len(affected) > 3:
                                            print(f"      ... 还有 {len(affected)-3} 行")
                                    else:
                                        print(f"    在第{first_line}行: 仅影响当前行")
                            except Exception as e:
                                print(f"    切片分析失败: {e}")
                    else:
                        print(f"\n  变量 '{var_name}' 未找到使用位置")
            else:
                print("  未发现变量")
                
            # 分析主要的数据流依赖
            print("\n  数据流依赖分析:")
            
            # 查找输入变量
            input_vars = []
            for node in ast.walk(analyzer.ast_tree):
                if isinstance(node, ast.Assign):
                    if isinstance(node.value, ast.Call):
                        if isinstance(node.value.func, ast.Name):
                            if node.value.func.id == 'eval':
                                for target in node.targets:
                                    if isinstance(target, ast.Name):
                                        input_vars.append(target.id)
            
            if input_vars:
                print(f"    输入变量: {', '.join(input_vars)}")
            
            # 查找输出
            output_lines = []
            for node in ast.walk(analyzer.ast_tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id == 'print':
                            output_lines.append(node.lineno)
            
            if output_lines:
                print(f"    输出位置: 第{', '.join(map(str, output_lines))}行")
                
        except Exception as e:
            print(f"  切片测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60)
        print("✓ 所有测试完成")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    test_static_analyzer()