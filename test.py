# test.py
import ast
from static_analyzer import StaticAnalyzer

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
        print("✓ 静态分析器实例创建成功")
        
        # 2. 测试常量提取
        print("\n1. 提取常量与比较值:")
        constants_result = analyzer.extract_constants_and_comparisons()
        if constants_result['success']:
            constants_data = constants_result['data']['constants']
            for var, values in constants_data.items():
                print(f"  变量 '{var}':")
                for val_info in values:
                    print(f"    值: {val_info['value']} (类型: {val_info['type']}, 来源: {val_info['source']})")
        else:
            print(f"  ✗ 常量提取失败: {constants_result['message']}")
        
        # 3. 测试控制流图构建
        print("\n2. 构建控制流图:")
        cfg_result = analyzer.build_control_flow_graph()
        if cfg_result['success']:
            cfg_data = cfg_result['data']
            print(f"  节点数量: {cfg_data['graph_metrics']['num_nodes']}")
            print(f"  边数量: {cfg_data['graph_metrics']['num_edges']}")
            print(f"  函数数量: {cfg_data['graph_metrics']['num_functions']}")
        else:
            print(f"  ✗ 控制流图构建失败: {cfg_result['message']}")
        
        # 4. 测试谓词挖掘
        print("\n3. 谓词挖掘:")
        predicates_result = analyzer.predicate_mining()
        if predicates_result['success']:
            predicates_data = predicates_result['data']['predicates']
            stats = predicates_result['data']['statistics']
            print(f"  发现 {stats['total_predicates']} 个谓词")
            
            for i, pred in enumerate(predicates_data, 1):
                print(f"\n  谓词 {i}:")
                print(f"    表达式: {pred['expression']}")
                print(f"    类型: {pred.get('type', 'unknown')}")
                
                if 'var' in pred:
                    print(f"    主要变量: {pred['var']}")
                
                if 'value' in pred:
                    print(f"    比较值: {pred['value']}")
                    
                if 'mod' in pred and 'rem' in pred:
                    print(f"    取模条件: {pred['var']} % {pred['mod']} == {pred['rem']}")
                    
                if 'lineno' in pred:
                    print(f"    位置: 第{pred['lineno']}行")
                    
                if 'boundary_info' in pred:
                    binfo = pred['boundary_info']
                    if 'suggested_values' in binfo:
                        print(f"    边界测试建议: {binfo['suggested_values']}")
                        
                if pred.get('is_chained', False):
                    print(f"    链式比较长度: {pred.get('chain_length', 0)}")
        else:
            print(f"  ✗ 谓词挖掘失败: {predicates_result['message']}")
        
        # 5. 测试链式比较分析
        print("\n4. 链式比较分析:")
        chained_result = analyzer.analyze_chained_comparisons()
        if chained_result['success']:
            chained_data = chained_result['data']['chained_comparisons']
            for i, comp in enumerate(chained_data, 1):
                print(f"  链式比较 {i}:")
                print(f"    表达式: {comp['expression']}")
                print(f"    主变量: {comp['main_variable']}")
                print(f"    边界: {comp['boundaries']}")
                if 'test_values' in comp:
                    print(f"    测试值建议: {comp['test_values']}")
        else:
            print(f"  ✗ 链式比较分析失败: {chained_result['message']}")
        
        # 6. 测试数据依赖分析
        print("\n5. 数据依赖分析:")
        data_deps_result = analyzer.build_data_dependency_graph()
        if data_deps_result['success']:
            data_deps = data_deps_result['data']['dependencies']
            for var, deps in data_deps.items():
                if deps:  # 只显示有依赖关系的变量
                    print(f"  变量 '{var}' 的依赖:")
                    for dep in deps:
                        if dep['type'] == 'def-use':
                            print(f"    定义在第{dep['from_line']}行，使用在第{dep['to_line']}行")
                        elif dep['type'] == 'data_dependency':
                            print(f"    依赖变量: {dep['depends_on']} (在第{dep['line']}行)")
        else:
            print(f"  ✗ 数据依赖分析失败: {data_deps_result['message']}")
        
        # 7. 测试输入结构推断
        print("\n6. 输入结构推断:")
        input_result = analyzer.infer_input_structure()
        if input_result['success']:
            input_data = input_result['data']['input_structure']
            
            # 显示输入模式
            if input_data['input_patterns']:
                print("  输入模式:")
                for pattern in input_data['input_patterns']:
                    print(f"    变量: {pattern['variable']}")
                    print(f"      类型: {pattern['input_type']}")
                    print(f"      转换链: {pattern['conversion_chain']}")
                    print(f"      推断类型: {pattern['inferred_type']}")
                    print(f"      位置: 第{pattern['lineno']}行")
            else:
                print("  未发现输入模式")
                
            # 显示检测到的变量类型
            if input_data['detected_types']:
                print("\n  检测到的变量类型:")
                for var, types in input_data['detected_types'].items():
                    print(f"    {var}: {', '.join(types)}")
        else:
            print(f"  ✗ 输入结构推断失败: {input_result['message']}")
        
        # 8. 测试输入模式分析
        print("\n7. 输入模式分析:")
        patterns_result = analyzer.analyze_input_patterns()
        if patterns_result['success']:
            patterns_data = patterns_result['data']['patterns']
            for pattern_type, patterns in patterns_data.items():
                if patterns:
                    print(f"  {pattern_type}: {len(patterns)} 个模式")
                    for pattern in patterns[:2]:  # 只显示前2个
                        print(f"    表达式: {pattern['expression']}")
        else:
            print(f"  ✗ 输入模式分析失败: {patterns_result['message']}")
        
        # 9. 测试变量类型推断
        print("\n8. 变量类型推断:")
        var_types_result = analyzer.get_variable_types()
        if var_types_result['success']:
            var_types = var_types_result['data']['variable_types']
            if var_types:
                print("  变量类型推断:")
                for var, type_info in sorted(var_types.items()):
                    var_type = type_info.get('type', 'Unknown')
                    confidence = type_info.get('confidence', 'low')
                    sources = type_info.get('sources', [])
                    
                    source_desc = []
                    for source in sources[:2]:  # 只显示前2个来源
                        source_desc.append(f"{source['source']}:{source['type']}")
                    
                    print(f"    {var:10} : {var_type:15} (置信度: {confidence})")
                    if source_desc:
                        print(f"              来源: {', '.join(source_desc)}")
            else:
                print("  未推断出变量类型")
        else:
            print(f"  ✗ 变量类型推断失败: {var_types_result['message']}")
        
        # 10. 测试后向切片
        print("\n9. 后向切片测试:")
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
                        if slice_result['success']:
                            slice_data = slice_result['data']
                            affected_lines = slice_data['affected_lines']
                            print(f"    影响 {len(affected_lines)} 行代码: {affected_lines}")
                            
                            # 显示需要边界测试的变量
                            boundary_vars = slice_data.get('boundary_variables', [])
                            if boundary_vars:
                                print(f"    需要边界测试的变量:")
                                for bv in boundary_vars:
                                    if bv['variable'] == var_name:  # 只显示当前变量
                                        print(f"      * {bv['variable']}:")
                                        
                                        # 显示原因
                                        reasons = []
                                        if bv['reasons']['in_boundary_condition']:
                                            reasons.append("在边界条件中使用")
                                        if bv['reasons']['in_loop']:
                                            reasons.append("在循环中使用")
                                        if bv['reasons']['is_input_related']:
                                            reasons.append("与输入相关")
                                        
                                        if reasons:
                                            print(f"        原因: {', '.join(reasons)}")
                                        
                                        # 显示边界值
                                        if 'boundary_values' in bv:
                                            bv_info = bv['boundary_values']
                                            if bv_info['values']:
                                                print(f"        边界值: {bv_info['values']}")
                                            if bv_info['lower'] is not None:
                                                print(f"        下界: {bv_info['lower']}")
                                            if bv_info['upper'] is not None:
                                                print(f"        上界: {bv_info['upper']}")
                                        
                                        # 显示建议测试值 - 确保这个字段存在
                                        if 'suggested_test_values' in bv and bv['suggested_test_values']:
                                            test_vals = bv['suggested_test_values']
                                            print(f"        建议测试值 (前10个): {test_vals[:10]}")
                                        else:
                                            print(f"        建议测试值: 未生成或为空")
                                        
                                        # 显示相关谓词
                                        if bv['boundary_predicates']:
                                            print(f"        相关谓词:")
                                            for pred in bv['boundary_predicates']:
                                                expr = pred.get('expression', '无表达式')
                                                print(f"          - {expr}")
                            else:
                                print(f"    无需边界测试")
                        else:
                            print(f"    切片失败: {slice_result['message']}")
        else:
            print("  未发现变量")
        
        print("\n" + "="*60)
        print("✓ 所有测试完成")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    test_static_analyzer()
