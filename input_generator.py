#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
输入生成器 - 从静态分析结果自动推断输入类型和生成测试数据
"""

from typing import Dict, List, Any


class InputGenerator:
    """从静态分析结果自动生成测试输入"""
    
    def __init__(self, static_analysis_result: Dict[str, Any]):
        """
        初始化输入生成器
        
        Args:
            static_analysis_result: 静态分析结果
        """
        self.static_result = static_analysis_result
        self.variable_types = static_analysis_result.get('variable_types', {})
        self.constants = static_analysis_result.get('constants', {})
        self.predicates = static_analysis_result.get('predicates', [])
        self.data_deps = static_analysis_result.get('data_dependencies', {})
    
    def infer_input_variables(self) -> Dict[str, str]:
        """推断输入变量及其类型"""
        input_vars = {}
        
        def is_complex_type(var_type):
            """检查是否是复杂类型"""
            return 'list' in var_type or 'dict' in var_type or var_type in ['list', 'dict']
        
        # 优先级1：查找复杂类型且无依赖的变量
        # 这些最可能是输入变量
        for var, var_type in self.variable_types.items():
            if is_complex_type(var_type):
                deps = self.data_deps.get(var, [])
                if len(deps) == 0:
                    input_vars[var] = var_type
                    return input_vars
        
        # 优先级2：查找任何无依赖的变量
        for var, var_type in self.variable_types.items():
            if var_type in ['str', 'int', 'float'] or is_complex_type(var_type):
                deps = self.data_deps.get(var, [])
                if len(deps) == 0:
                    input_vars[var] = var_type
                    return input_vars
        
        # 优先级3：如果所有变量都有依赖，选择依赖最少的复杂类型
        min_deps = float('inf')
        best_var = None
        for var, var_type in self.variable_types.items():
            if is_complex_type(var_type):
                deps = self.data_deps.get(var, [])
                if len(deps) < min_deps:
                    min_deps = len(deps)
                    best_var = var
        
        if best_var:
            input_vars[best_var] = self.variable_types[best_var]
            return input_vars
        
        # 优先级4：选择依赖最少的任何变量
        min_deps = float('inf')
        best_var = None
        for var, var_type in self.variable_types.items():
            if var_type in ['str', 'int', 'float'] or is_complex_type(var_type):
                deps = self.data_deps.get(var, [])
                if len(deps) < min_deps:
                    min_deps = len(deps)
                    best_var = var
        
        if best_var:
            input_vars[best_var] = self.variable_types[best_var]
        
        return input_vars
    
    def get_boundary_values(self, var_name: str) -> List[Any]:
        """获取变量的边界值"""
        values = set()
        
        # 从常量中提取
        if var_name in self.constants:
            for const_info in self.constants[var_name]:
                values.add(const_info.get('value'))
        
        # 从谓词中提取
        for pred in self.predicates:
            if 'var' in pred and pred['var'] == var_name:
                if 'value' in pred:
                    val = pred['value']
                    if isinstance(val, list):
                        values.update(val)
                    else:
                        values.add(val)
                
                if 'boundary_values' in pred:
                    values.update(pred['boundary_values'])
        
        return sorted(list(values))
    
    def generate_test_inputs(self, num_cases: int = 5) -> List[List[str]]:
        """生成测试输入"""
        input_vars = self.infer_input_variables()
        
        if not input_vars:
            return []
        
        # 为每个输入变量收集可能的值
        var_values = {}
        for var, var_type in input_vars.items():
            boundary_vals = self.get_boundary_values(var)
            
            if var_type == 'int':
                values = boundary_vals if boundary_vals else [0, 1, -1, 10, 100]
            elif var_type == 'float':
                values = boundary_vals if boundary_vals else [0.0, 1.0, -1.0, 0.5, 3.14]
            elif var_type == 'str':
                values = ['', 'a', 'test', '123', 'hello']
            elif 'list' in var_type:
                # 处理list和list[...]类型
                values = ['[]', '[1]', '[1, 2, 3]', '[1, 2, 3, 4, 5]', '["a", "b", "c"]']
            elif 'dict' in var_type:
                # 处理dict和dict[...]类型
                values = ['{}', '{"a": 1}', '{"a": 1, "b": 2}', '{"x": 10, "y": 20}']
            else:
                values = ['0', '1', 'test']
            
            var_values[var] = values
        
        # 生成测试用例
        test_cases = []
        var_list = list(input_vars.keys())
        
        for i in range(num_cases):
            case = []
            for var in var_list:
                values = var_values[var]
                idx = i % len(values)
                case.append(str(values[idx]))
            test_cases.append(case)
        
        return test_cases
    
    def generate_boundary_test_inputs(self) -> List[List[str]]:
        """生成边界值测试输入"""
        input_vars = self.infer_input_variables()
        
        if not input_vars:
            return []
        
        test_cases = []
        var_list = list(input_vars.keys())
        
        for var in var_list:
            boundary_vals = self.get_boundary_values(var)
            
            if boundary_vals:
                for val in boundary_vals:
                    case = []
                    for v in var_list:
                        if v == var:
                            case.append(str(val))
                        else:
                            case.append('0')
                    test_cases.append(case)
        
        return test_cases
    
    def get_input_description(self) -> Dict[str, Any]:
        """获取输入的描述信息"""
        input_vars = self.infer_input_variables()
        
        description = {
            'input_variables': input_vars,
            'variable_details': {}
        }
        
        for var, var_type in input_vars.items():
            boundary_vals = self.get_boundary_values(var)
            description['variable_details'][var] = {
                'type': var_type,
                'boundary_values': boundary_vals
            }
        
        return description


class DynamicAnalyzerWithAutoInput:
    """增强的动态分析器 - 自动生成输入"""
    
    def __init__(self, source_code: str, static_analysis_result: Dict[str, Any]):
        """初始化"""
        from dynamic_analyzer import DynamicAnalyzer
        
        self.source_code = source_code
        self.static_result = static_analysis_result
        self.dynamic_analyzer = DynamicAnalyzer(source_code, static_analysis_result)
        self.input_generator = InputGenerator(static_analysis_result)
    
    def run_with_auto_generated_inputs(self, num_cases: int = 5) -> List[Dict[str, Any]]:
        """使用自动生成的输入执行程序"""
        test_inputs = self.input_generator.generate_test_inputs(num_cases)
        
        if not test_inputs:
            print("[WARN] 无法自动生成输入")
            return []
        
        print(f"[INFO] 自动生成了 {len(test_inputs)} 个测试用例")
        
        results = []
        for i, test_input in enumerate(test_inputs, 1):
            print(f"[INFO] 执行测试用例 {i}: {test_input}")
            result = self.dynamic_analyzer.run_with_input(test_input)
            results.append(result)
        
        return results
    
    def run_with_boundary_test_inputs(self) -> List[Dict[str, Any]]:
        """使用边界值测试输入执行程序"""
        test_inputs = self.input_generator.generate_boundary_test_inputs()
        
        if not test_inputs:
            print("[WARN] 无法生成边界值测试输入")
            return []
        
        print(f"[INFO] 生成了 {len(test_inputs)} 个边界值测试用例")
        
        results = []
        for i, test_input in enumerate(test_inputs, 1):
            print(f"[INFO] 执行边界值测试用例 {i}: {test_input}")
            result = self.dynamic_analyzer.run_with_input(test_input)
            results.append(result)
        
        return results
    
    def get_input_info(self) -> Dict[str, Any]:
        """获取输入信息"""
        return self.input_generator.get_input_description()
