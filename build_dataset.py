# -*- coding: utf-8 -*-
# 伪正确代码生成器

import os
import json
import ast
import random
import copy
import subprocess
import tempfile
import shutil
import sys
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict
import math

class StaticAnalyzer:
    """静态分析器 - 分析代码结构和复杂性"""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tree = None
        
        try:
            self.tree = ast.parse(source_code)
        except SyntaxError as e:
            print(f"语法解析错误: {e}")
            self.tree = None
    
    def analyze_complexity(self) -> Dict[str, Any]:
        """分析代码复杂度"""
        if self.tree is None:
            return {"error": "无法解析代码"}
        
        analysis = {
            "cyclomatic_complexity": self._calculate_cyclomatic_complexity(),
            "halstead_metrics": self._calculate_halstead_metrics(),
            "control_flow_depth": self._calculate_control_flow_depth(),
            "variable_usage": self._analyze_variable_usage(),
            "operation_patterns": self._analyze_operation_patterns(),
            "code_structure": self._analyze_code_structure(),
            "risk_factors": self._identify_risk_factors()
        }
        
        return analysis
    
    def _calculate_cyclomatic_complexity(self) -> int:
        """计算圈复杂度"""
        complexity = 1  # 基础复杂度
        
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # 每个and/or增加复杂度
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_halstead_metrics(self) -> Dict[str, float]:
        """计算Halstead软件科学度量"""
        operators = set()
        operands = set()
        
        operator_nodes = [
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
            ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
            ast.FloorDiv, ast.Invert, ast.Not, ast.UAdd, ast.USub,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Is, ast.IsNot, ast.In, ast.NotIn,
            ast.And, ast.Or
        ]
        
        for node in ast.walk(self.tree):
            # 统计操作符
            if any(isinstance(node, op_type) for op_type in operator_nodes):
                operators.add(type(node).__name__)
            
            # 统计操作数（变量和常量）
            if isinstance(node, ast.Name):
                operands.add(node.id)
            elif isinstance(node, ast.Constant):
                value = node.value
                if isinstance(value, (int, float)):
                    operands.add(str(value))
                elif isinstance(value, str):
                    operands.add(repr(value)[:20])
                else:
                    operands.add(str(value)[:20])
            elif hasattr(ast, 'Num') and isinstance(node, ast.Num):
                # 向后兼容 Python 3.7
                operands.add(str(node.n))
            elif hasattr(ast, 'Str') and isinstance(node, ast.Str):
                # 向后兼容 Python 3.7
                operands.add(repr(node.s)[:20])
        
        n1 = len(operators)  # 不同操作符数量
        n2 = len(operands)   # 不同操作数数量
        N1 = sum(1 for node in ast.walk(self.tree) 
                if any(isinstance(node, op_type) for op_type in operator_nodes))  # 操作符总数
        N2 = sum(1 for node in ast.walk(self.tree) 
                if isinstance(node, (ast.Name, ast.Constant)) or 
                (hasattr(ast, 'Num') and isinstance(node, ast.Num)) or
                (hasattr(ast, 'Str') and isinstance(node, ast.Str)))  # 操作数总数
        
        vocabulary = n1 + n2 if (n1 + n2) > 0 else 1
        length = N1 + N2 if (N1 + N2) > 0 else 1
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = volume * difficulty
        
        return {
            "unique_operators": n1,
            "unique_operands": n2,
            "total_operators": N1,
            "total_operands": N2,
            "vocabulary": vocabulary,
            "length": length,
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort
        }
    
    def _calculate_control_flow_depth(self) -> int:
        """计算控制流最大深度"""
        max_depth = 0
        current_depth = 0
        
        # 使用AST遍历计算嵌套深度
        def traverse(node, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            # 增加深度的节点类型
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                new_depth = depth + 1
                for child in ast.iter_child_nodes(node):
                    traverse(child, new_depth)
            else:
                for child in ast.iter_child_nodes(node):
                    traverse(child, depth)
        
        if self.tree:
            traverse(self.tree, 0)
        
        return max_depth
    
    def _analyze_variable_usage(self) -> Dict[str, Any]:
        """分析变量使用模式"""
        variables = defaultdict(lambda: {"reads": 0, "writes": 0, "scopes": set()})
        current_scope = []
        
        def track_variable(name, action="read"):
            if name in variables:
                if action == "read":
                    variables[name]["reads"] += 1
                elif action == "write":
                    variables[name]["writes"] += 1
                variables[name]["scopes"].add(tuple(current_scope))
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        track_variable(target.id, "write")
            
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                track_variable(node.id, "read")
        
        # 计算变量使用统计
        total_vars = len(variables)
        read_write_ratio = 0
        if total_vars > 0:
            total_reads = sum(v["reads"] for v in variables.values())
            total_writes = sum(v["writes"] for v in variables.values())
            if total_writes > 0:
                read_write_ratio = total_reads / total_writes
        
        return {
            "total_variables": total_vars,
            "variable_details": dict(variables),
            "read_write_ratio": read_write_ratio,
            "scoped_variables": sum(1 for v in variables.values() if len(v["scopes"]) > 1)
        }
    
    def _analyze_operation_patterns(self) -> Dict[str, Any]:
        """分析操作模式"""
        patterns = {
            "arithmetic_operations": 0,
            "comparison_operations": 0,
            "logical_operations": 0,
            "bitwise_operations": 0,
            "function_calls": 0,
            "list_operations": 0,
            "string_operations": 0
        }
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv)):
                    patterns["arithmetic_operations"] += 1
                elif isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
                    patterns["bitwise_operations"] += 1
            
            elif isinstance(node, ast.Compare):
                patterns["comparison_operations"] += 1
            
            elif isinstance(node, ast.BoolOp):
                patterns["logical_operations"] += 1
            
            elif isinstance(node, ast.Call):
                patterns["function_calls"] += 1
                
                # 检查特定类型的函数调用
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ['append', 'remove', 'pop', 'extend', 'insert']:
                        patterns["list_operations"] += 1
                    elif func_name in ['find', 'replace', 'split', 'join', 'strip']:
                        patterns["string_operations"] += 1
        
        return patterns
    
    def _analyze_code_structure(self) -> Dict[str, Any]:
        """分析代码结构"""
        structure = {
            "has_loops": False,
            "has_conditionals": False,
            "has_functions": False,
            "has_exceptions": False,
            "has_nesting": False,
            "statement_types": defaultdict(int)
        }
        
        for node in ast.walk(self.tree):
            node_type = type(node).__name__
            structure["statement_types"][node_type] += 1
            
            if isinstance(node, (ast.For, ast.While)):
                structure["has_loops"] = True
            
            if isinstance(node, ast.If):
                structure["has_conditionals"] = True
            
            if isinstance(node, ast.FunctionDef):
                structure["has_functions"] = True
            
            if isinstance(node, (ast.Try, ast.ExceptHandler)):
                structure["has_exceptions"] = True
        
        # 检查嵌套结构
        def check_nesting(node, depth=0):
            if depth > 1:
                structure["has_nesting"] = True
                return
            
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                        check_nesting(child, depth + 1)
        
        if self.tree:
            for node in self.tree.body:
                check_nesting(node)
        
        return structure
    
    def _identify_risk_factors(self) -> List[str]:
        """识别风险因素"""
        risks = []
        
        # 检查常见的潜在问题模式
        code_lines = self.source_code.split('\n')
        
        # 1. 检查硬编码值
        hardcoded_numbers = re.findall(r'\b\d+(\.\d+)?\b', self.source_code)
        if len(hardcoded_numbers) > 5:  # 过多硬编码数字
            risks.append("excessive_hardcoded_values")
        
        # 2. 检查魔法字符串
        string_literals = re.findall(r'["\']([^"\']+)["\']', self.source_code)
        if len(string_literals) > 3:
            risks.append("magic_strings")
        
        # 3. 检查复杂的条件表达式
        for node in ast.walk(self.tree):
            if isinstance(node, ast.BoolOp):
                if len(node.values) > 3:  # 过多条件的布尔表达式
                    risks.append("complex_boolean_expression")
                    break
        
        # 4. 检查深度嵌套
        if self._calculate_control_flow_depth() > 3:
            risks.append("deep_nesting")
        
        # 5. 检查过长的函数/代码块
        if len(code_lines) > 30:
            risks.append("long_code_block")
        
        # 6. 检查变量重用
        var_usage = self._analyze_variable_usage()
        if var_usage["read_write_ratio"] > 10:  # 读远多于写
            risks.append("variable_reuse_pattern")
        
        return risks
    
    def calculate_mutation_energy(self) -> Dict[str, float]:
        """计算变异能量分配"""
        analysis = self.analyze_complexity()
        
        if "error" in analysis:
            return {"total_energy": 0, "distribution": {}}
        
        # 基础能量计算
        base_energy = 100
        
        # 基于复杂度调整能量
        complexity_factor = min(analysis["cyclomatic_complexity"] / 5, 3.0)
        
        # 基于Halstead工作量调整
        effort_factor = min(analysis["halstead_metrics"]["effort"] / 1000, 2.0)
        
        # 基于风险因素调整
        risk_factor = 1.0 + (len(analysis["risk_factors"]) * 0.2)
        
        total_energy = base_energy * complexity_factor * effort_factor * risk_factor
        
        # 能量分配策略
        distribution = {
            "loop_mutations": 0.25,  # 循环变异
            "condition_mutations": 0.20,  # 条件变异
            "operator_mutations": 0.15,  # 操作符变异
            "constant_mutations": 0.15,  # 常量变异
            "boundary_mutations": 0.15,  # 边界变异
            "logic_mutations": 0.10,  # 逻辑变异
        }
        
        # 根据代码特征调整分配
        if analysis["code_structure"]["has_loops"]:
            distribution["loop_mutations"] += 0.10
        
        if analysis["code_structure"]["has_conditionals"]:
            distribution["condition_mutations"] += 0.10
        
        if analysis["operation_patterns"]["arithmetic_operations"] > 5:
            distribution["operator_mutations"] += 0.05
        
        if analysis["risk_factors"]:
            if "deep_nesting" in analysis["risk_factors"]:
                distribution["logic_mutations"] += 0.05
            if "complex_boolean_expression" in analysis["risk_factors"]:
                distribution["condition_mutations"] += 0.05
        
        # 归一化
        total = sum(distribution.values())
        distribution = {k: v/total for k, v in distribution.items()}
        
        return {
            "total_energy": total_energy,
            "distribution": distribution,
            "analysis_summary": {
                "complexity": analysis["cyclomatic_complexity"],
                "risk_factors": analysis["risk_factors"],
                "structure": analysis["code_structure"]
            }
        }


# 用于正则表达式匹配
import re


class IntelligentMutator:
    """智能变异器 - 分配变异能量"""
    
    def __init__(self, source_code: str, problem_type: str):
        self.source_code = source_code
        self.problem_type = problem_type  # '2910' or '3039'
        self.tree = None
        self.original_tree = None
        
        try:
            self.tree = ast.parse(source_code)
            self.original_tree = copy.deepcopy(self.tree)
        except SyntaxError as e:
            print(f"代码解析错误: {e}")
            self.tree = None
        
        # 初始化分析器
        self.analyzer = StaticAnalyzer(source_code)
        self.energy_info = self.analyzer.calculate_mutation_energy()
        
        # 变异计数器
        self.mutation_count = 0
        self.successful_mutations = 0
    
    def allocate_mutation_budget(self) -> Dict[str, int]:
        """基于能量分配变异预算"""
        total_energy = self.energy_info["total_energy"]
        distribution = self.energy_info["distribution"]
        
        budget = {}
        for mutation_type, ratio in distribution.items():
            budget[mutation_type] = int(total_energy * ratio)
        
        # 确保最小预算
        for key in budget:
            if budget[key] < 5:
                budget[key] = 5
        
        print(f"变异能量分配: 总计={total_energy:.1f}")
        for k, v in budget.items():
            print(f"  {k}: {v}")
        
        return budget
    
    def generate_pseudo_correct_code(self, max_attempts: int = 50) -> List[Dict[str, Any]]:
        """生成伪正确代码（通过全部测试但包含潜在缺陷）"""
        print(f"\n生成伪正确代码 (问题: {self.problem_type})")
        print("=" * 60)
        
        # 分配变异预算
        budget = self.allocate_mutation_budget()
        
        results = []
        attempt = 0
        
        while attempt < max_attempts and len(results) < 10:  # 最多生成10个伪正确代码
            attempt += 1
            print(f"\n尝试 #{attempt}:")
            
            # 选择变异类型（基于预算权重）
            mutation_types = []
            for mt, count in budget.items():
                mutation_types.extend([mt] * count)
            
            if not mutation_types:
                continue
            
            # 随机选择变异类型
            mutation_type = random.choice(mutation_types)
            
            # 执行变异
            mutated_code = self._apply_mutation(mutation_type)
            
            if mutated_code == self.source_code:
                print("  无变化，跳过")
                continue
            
            # 检查代码有效性
            if not self._validate_code_syntax(mutated_code):
                print("  语法无效，跳过")
                continue
            
            # 创建记录
            record = {
                "attempt": attempt,
                "mutation_type": mutation_type,
                "code": mutated_code,
                "energy_used": budget.get(mutation_type, 0),
                "original_code_hash": hash(self.source_code)
            }
            
            results.append(record)
            
            # 减少该类型的预算（模拟能量消耗）
            if budget[mutation_type] > 1:
                budget[mutation_type] -= 1
            
            print(f"  成功应用 {mutation_type} 变异")
        
        print(f"\n生成完成: {len(results)} 个变异代码")
        return results
    
    def _apply_mutation(self, mutation_type: str) -> str:
        """应用特定类型的变异"""
        self.mutation_count += 1
        
        # 恢复原始AST
        self.tree = copy.deepcopy(self.original_tree)
        
        if self.tree is None:
            return self.source_code
        
        try:
            if mutation_type == "loop_mutations":
                return self._mutate_loops()
            elif mutation_type == "condition_mutations":
                return self._mutate_conditions()
            elif mutation_type == "operator_mutations":
                return self._mutate_operators()
            elif mutation_type == "constant_mutations":
                return self._mutate_constants()
            elif mutation_type == "boundary_mutations":
                return self._mutate_boundaries()
            elif mutation_type == "logic_mutations":
                return self._mutate_logic()
            else:
                return self._mutate_random()
        except Exception as e:
            print(f"变异失败: {e}")
            return self.source_code
    
    def _mutate_loops(self) -> str:
        """变异循环结构 - 生成看似正确的循环变异"""
        mutations_applied = 0
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.For):
                # 变异1: 修改循环范围（边界情况）
                if isinstance(node.iter, ast.Call):
                    if hasattr(node.iter.func, 'id') and node.iter.func.id == 'range':
                        args = node.iter.args
                        
                        # 只对某些边界情况进行变异
                        if random.random() < 0.3:
                            if len(args) == 2:
                                # 轻微调整边界
                                if isinstance(args[1], ast.Constant) and isinstance(args[1].value, (int, float)):
                                    # 增加或减少1（边界情况）
                                    adjustment = random.choice([-1, 1])
                                    new_value = args[1].n + adjustment
                                    args[1] = ast.Constant(n=new_value)
                                    mutations_applied += 1
                        
                        # 变异2: 修改步长（如果存在）
                        if len(args) == 3:
                            if isinstance(args[2], ast.Constant) and isinstance(args[2].value, (int, float)):
                                # 轻微修改步长
                                new_step = args[2].n + random.choice([-1, 1, 0])
                                if new_step != 0:  # 避免除零
                                    args[2] = ast.Constant(n=new_step)
                                    mutations_applied += 1
            
            elif isinstance(node, ast.While):
                # 变异3: 轻微修改循环条件
                if random.random() < 0.2:
                    # 在条件中添加微小的偏移
                    if isinstance(node.test, ast.Compare):
                        if len(node.test.ops) == 1 and len(node.test.comparators) == 1:
                            op = node.test.ops[0]
                            comparator = node.test.comparators[0]
                            
                            if isinstance(op, ast.Lt) and isinstance(comparator, ast.Constant) and isinstance(comparator.value, (int, float)):
                                # i < n -> i < n + 1
                                node.test.comparators[0] = ast.Constant(n=comparator.n + 1)
                                mutations_applied += 1
        
        if mutations_applied > 0:
            self.successful_mutations += 1
            return self._ast_to_code()
        
        return self.source_code
    
    def _mutate_conditions(self) -> str:
        """变异条件判断 - 生成微妙的条件错误"""
        mutations_applied = 0
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.If):
                # 变异1: 添加冗余条件
                if random.random() < 0.1:
                    # 添加一个总是为True的冗余条件
                    if not node.orelse:
                        node.orelse = [ast.Pass()]
                    mutations_applied += 1
                
                # 变异2: 轻微修改条件边界
                if isinstance(node.test, ast.Compare):
                    if random.random() < 0.2:
                        if len(node.test.ops) == 1 and len(node.test.comparators) == 1:
                            op = node.test.ops[0]
                            comparator = node.test.comparators[0]
                            
                            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, (int, float)):
                                # 轻微调整边界值
                                adjustment = random.choice([-0.1, 0.1, -0.01, 0.01])
                                new_value = comparator.n + adjustment
                                
                                # 根据问题类型调整
                                if self.problem_type == "2910":
                                    # 对于高度问题，保持正值
                                    if new_value > 0:
                                        node.test.comparators[0] = ast.Constant(n=round(new_value, 2))
                                        mutations_applied += 1
                                else:
                                    node.test.comparators[0] = ast.Constant(n=new_value)
                                    mutations_applied += 1

            elif isinstance(node, ast.Compare):
                # 变异3: 修改比较运算符（微妙变化）
                if random.random() < 0.15:
                    if len(node.ops) == 1:
                        op = node.ops[0]
                        
                        # 只在某些情况下修改，避免明显错误
                        if isinstance(op, ast.Lt):
                            # < 改为 <= （微妙边界变化）
                            node.ops[0] = ast.LtE()
                            mutations_applied += 1
                        elif isinstance(op, ast.Gt):
                            # > 改为 >=
                            node.ops[0] = ast.GtE()
                            mutations_applied += 1
        
        if mutations_applied > 0:
            self.successful_mutations += 1
            return self._ast_to_code()
        
        return self.source_code
    
    def _mutate_operators(self) -> str:
        """变异操作符 - 生成微妙的操作错误"""
        mutations_applied = 0
        
        for node in ast.walk(self.tree):
            # 变异算术操作符
            if isinstance(node, ast.BinOp):
                if random.random() < 0.2:
                    # 只对某些操作符进行微妙变化
                    if isinstance(node.op, ast.Add):
                        # + 改为 - （在某些情况下可能不明显）
                        if random.random() < 0.3:
                            node.op = ast.Sub()
                            mutations_applied += 1
                    elif isinstance(node.op, ast.Mult):
                        # * 改为 / （需要检查除数）
                        if random.random() < 0.2:
                            # 确保除数不为零
                            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) and node.right.n != 0:
                                node.op = ast.Div()
                                mutations_applied += 1
            
            # 变异赋值操作
            elif isinstance(node, ast.AugAssign):
                if random.random() < 0.15:
                    # 轻微修改增量
                    if isinstance(node.op, ast.Add):
                        if isinstance(node.value, (int, float)):
                            # i += 1 -> i += 2 （微妙变化）
                            if node.value.n == 1:
                                node.value = ast.Constant(n=2)
                                mutations_applied += 1
        
        if mutations_applied > 0:
            self.successful_mutations += 1
            return self._ast_to_code()
        
        return self.source_code
    
    def _mutate_constants(self) -> str:
        """变异常量值 - 生成微妙的数值错误"""
        mutations_applied = 0
        
        for node in ast.walk(self.tree):
            # 变异数字常量
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if random.random() < 0.25:
                    original = node.value
                    
                    # 根据问题类型进行不同的变异
                    if self.problem_type == "2910":
                        # 对于高度反弹问题
                        if original == 0.5:
                            # 衰减系数轻微变化
                            new_value = original + random.choice([-0.01, 0.01, -0.02, 0.02])
                            new_value = max(0.1, min(0.9, new_value))  # 保持合理范围
                            node.value = round(new_value, 2)
                            mutations_applied += 1
                        elif original == 2:
                            # 乘法因子变化
                            new_value = original + random.choice([-0.1, 0.1, -0.2, 0.2])
                            node.value = new_value
                            mutations_applied += 1
                    
                    elif self.problem_type == "3039":
                        # 对于列表删除问题
                        if original == 0 or original == 1:
                            # 索引调整值变化
                            new_value = original + random.choice([-1, 1])
                            if new_value >= -5:  # 保持合理范围
                                node.value = new_value
                                mutations_applied += 1
            
            # 向后兼容 Python 3.7 的 ast.Num
            elif hasattr(ast, 'Num') and isinstance(node, ast.Num):
                if random.random() < 0.25:
                    original = node.n
                    
                    # 根据问题类型进行不同的变异
                    if self.problem_type == "2910":
                        # 对于高度反弹问题
                        if original == 0.5:
                            # 衰减系数轻微变化
                            new_value = original + random.choice([-0.01, 0.01, -0.02, 0.02])
                            new_value = max(0.1, min(0.9, new_value))  # 保持合理范围
                            node.n = round(new_value, 2)
                            mutations_applied += 1
                        elif original == 2:
                            # 乘法因子变化
                            new_value = original + random.choice([-0.1, 0.1, -0.2, 0.2])
                            node.n = new_value
                            mutations_applied += 1
                    
                    elif self.problem_type == "3039":
                        # 对于列表删除问题
                        if original == 0 or original == 1:
                            # 索引调整值变化
                            new_value = original + random.choice([-1, 1])
                            if new_value >= -5:  # 保持合理范围
                                node.n = new_value
                                mutations_applied += 1

            # 变异字符串常量（如果有）
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if random.random() < 0.1:
                    # 轻微修改格式字符串
                    if "%.2f" in node.value:
                        # 修改浮点数格式
                        new_format = random.choice(["%.1f", "%.3f"])
                        node.value = node.value.replace("%.2f", new_format)
                        mutations_applied += 1
            # 向后兼容 Python 3.7 的 ast.Str
            elif hasattr(ast, 'Str') and isinstance(node, ast.Str):
                if random.random() < 0.1:
                    # 轻微修改格式字符串
                    if "%.2f" in node.s:
                        # 修改浮点数格式
                        new_format = random.choice(["%.1f", "%.3f"])
                        node.s = node.s.replace("%.2f", new_format)
                        mutations_applied += 1
        
        if mutations_applied > 0:
            self.successful_mutations += 1
            return self._ast_to_code()
        
        return self.source_code
    
    def _mutate_boundaries(self) -> str:
        """变异边界条件 - 生成边界相关错误"""
        mutations_applied = 0
        
        for node in ast.walk(self.tree):
            # 变异循环边界
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if hasattr(node.iter.func, 'id') and node.iter.func.id == 'range':
                        args = node.iter.args
                        
                        if len(args) == 1 and isinstance(args[0], ast.Constant):
                            # range(n) -> range(n ± 1) （边界变化）
                            if random.random() < 0.3:
                                adjustment = random.choice([-1, 1])
                                new_n = max(1, args[0].n + adjustment)  # 保持至少1
                                args[0] = ast.Constant(n=new_n)
                                mutations_applied += 1
            
            # 变异列表/字符串索引
            elif isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Index):
                    if isinstance(node.slice.value, ast.Constant):
                        # 修改索引值
                        if random.random() < 0.2:
                            new_index = node.slice.value.n + random.choice([-1, 1])
                            if new_index >= 0:  # 保持有效索引
                                node.slice.value = ast.Constant(n=new_index)
                                mutations_applied += 1
        
        if mutations_applied > 0:
            self.successful_mutations += 1
            return self._ast_to_code()
        
        return self.source_code
    
    def _mutate_logic(self) -> str:
        """变异逻辑结构 - 生成微妙的逻辑错误"""
        mutations_applied = 0
        
        for node in ast.walk(self.tree):
            # 变异布尔逻辑
            if isinstance(node, ast.BoolOp):
                if random.random() < 0.2:
                    if isinstance(node.op, ast.And):
                        # 添加冗余条件
                        if len(node.values) < 4:  # 避免过长
                            # 添加一个总是为True的条件
                            true_condition = ast.Compare(
                                left=ast.Constant(n=1),
                                ops=[ast.Eq()],
                                comparators=[ast.Constant(n=1)]
                            )
                            node.values.append(true_condition)
                            mutations_applied += 1
            
            # 变异条件分支
            elif isinstance(node, ast.If):
                if random.random() < 0.1:
                    # 添加冗余的elif或else
                    if not node.orelse:
                        # 添加一个简单的else分支
                        node.orelse = [ast.Pass()]
                        mutations_applied += 1
        
        if mutations_applied > 0:
            self.successful_mutations += 1
            return self._ast_to_code()
        
        return self.source_code
    
    def _mutate_random(self) -> str:
        """随机变异 - 组合多种变异策略"""
        mutation_types = [
            self._mutate_loops,
            self._mutate_conditions,
            self._mutate_operators,
            self._mutate_constants,
            self._mutate_boundaries,
            self._mutate_logic
        ]
        
        # 尝试多种变异，直到成功
        for mutation_func in random.sample(mutation_types, len(mutation_types)):
            result = mutation_func()
            if result != self.source_code:
                return result
        
        return self.source_code
    
    def _ast_to_code(self) -> str:
        """将AST转换回代码"""
        try:
            # 使用ast.unparse
            if hasattr(ast, 'unparse'):
                return ast.unparse(self.tree)
            else:
                # 简化的AST转代码
                return self._simple_ast_to_code(self.tree)
        except:
            # 回退到原始代码
            return self.source_code
    
    def _simple_ast_to_code(self, node: ast.AST, indent: int = 0) -> str:
        """简化的AST转代码函数"""
        indent_str = " " * indent
        
        if isinstance(node, ast.Module):
            return "\n".join(self._simple_ast_to_code(stmt, indent) for stmt in node.body)
        
        elif isinstance(node, ast.Expr):
            return indent_str + self._simple_ast_to_code(node.value, indent)
        
        elif isinstance(node, ast.Assign):
            targets = ", ".join(self._simple_ast_to_code(t, indent) for t in node.targets)
            return f"{indent_str}{targets} = {self._simple_ast_to_code(node.value, indent)}"
        
        elif isinstance(node, ast.AugAssign):
            op_map = {ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/="}
            op_str = op_map.get(type(node.op), "+=")
            return f"{indent_str}{self._simple_ast_to_code(node.target, indent)} {op_str} {self._simple_ast_to_code(node.value, indent)}"
        
        elif isinstance(node, ast.For):
            target = self._simple_ast_to_code(node.target, indent)
            iter_src = self._simple_ast_to_code(node.iter, indent)
            body = "\n".join(self._simple_ast_to_code(stmt, indent + 4) for stmt in node.body)
            return f"{indent_str}for {target} in {iter_src}:\n{body}"
        
        elif isinstance(node, ast.While):
            test = self._simple_ast_to_code(node.test, indent)
            body = "\n".join(self._simple_ast_to_code(stmt, indent + 4) for stmt in node.body)
            return f"{indent_str}while {test}:\n{body}"
        
        elif isinstance(node, ast.If):
            test = self._simple_ast_to_code(node.test, indent)
            body = "\n".join(self._simple_ast_to_code(stmt, indent + 4) for stmt in node.body)
            result = f"{indent_str}if {test}:\n{body}"
            
            if node.orelse:
                orelse = "\n".join(self._simple_ast_to_code(stmt, indent + 4) for stmt in node.orelse)
                result += f"\n{indent_str}else:\n{orelse}"
            
            return result
        
        elif isinstance(node, ast.Name):
            return node.id
        
        elif isinstance(node, ast.Num):
            return str(node.n)
        
        elif isinstance(node, ast.Str):
            return repr(node.s)
        
        elif isinstance(node, ast.Call):
            func = self._simple_ast_to_code(node.func, indent)
            args = ", ".join(self._simple_ast_to_code(arg, indent) for arg in node.args)
            return f"{func}({args})"
        
        elif isinstance(node, ast.BinOp):
            op_map = {
                ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                ast.Mod: "%", ast.Pow: "**", ast.FloorDiv: "//"
            }
            left = self._simple_ast_to_code(node.left, indent)
            right = self._simple_ast_to_code(node.right, indent)
            op_str = op_map.get(type(node.op), "+")
            return f"({left} {op_str} {right})"
        
        elif isinstance(node, ast.Compare):
            left = self._simple_ast_to_code(node.left, indent)
            ops_map = {
                ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
                ast.Gt: ">", ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not",
                ast.In: "in", ast.NotIn: "not in"
            }
            comparators = [self._simple_ast_to_code(c, indent) for c in node.comparators]
            result = left
            for op, comparator in zip(node.ops, comparators):
                op_str = ops_map.get(type(op), "==")
                result += f" {op_str} {comparator}"
            return result
        
        elif isinstance(node, ast.Return):
            if node.value:
                return f"{indent_str}return {self._simple_ast_to_code(node.value, indent)}"
            return f"{indent_str}return"
        
        elif isinstance(node, ast.Pass):
            return f"{indent_str}pass"
        
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return repr(node.value)
            else:
                return str(node.value)
        
        else:
            # 对于未处理的节点类型，返回占位符
            return f"{indent_str}# <{type(node).__name__}>"
    
    def _validate_code_syntax(self, code: str) -> bool:
        """验证代码语法"""
        try:
            ast.parse(code)
            return True
        except:
            return False


class TestOracle:
    """测试预言 - 验证代码是否通过所有测试用例"""
    
    def __init__(self, test_cases: List[Dict[str, str]]):
        self.test_cases = test_cases
    
    def test_code(self, code: str) -> Tuple[bool, Dict[str, Any]]:
        """测试代码是否通过所有测试用例"""
        results = {
            "total": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for i, test_case in enumerate(self.test_cases):
            success, output = self._execute_code(code, test_case['input'])
            
            test_result = {
                "test_id": i,
                "input": test_case['input'],
                "expected": test_case['expected'],
                "success": success,
                "output": output,
                "passed": False
            }
            
            if success:
                # 检查输出是否匹配
                if self._compare_output(output, test_case['expected']):
                    results["passed"] += 1
                    test_result["passed"] = True
                else:
                    results["failed"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append(test_result)
        
        # 判断是否通过所有测试
        all_passed = (results["passed"] == results["total"])
        
        return all_passed, results
    
    def _execute_code(self, code: str, input_str: str) -> Tuple[bool, str]:
        """执行代码"""
        temp_file = None
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                f.flush()
                temp_file = f.name
            
            # 执行代码
            result = subprocess.run(
                [sys.executable, temp_file],
                input=input_str,
                capture_output=True,
                text=True,
                timeout=5,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
                
        except subprocess.TimeoutExpired:
            return False, "执行超时"
        except Exception as e:
            return False, str(e)
        finally:
            # 确保删除临时文件
            if temp_file and os.path.exists(temp_file):
                try:
                    # 尝试多次删除，避免文件占用
                    import time
                    for _ in range(3):
                        try:
                            os.unlink(temp_file)
                            break
                        except (PermissionError, OSError):
                            time.sleep(0.1)
                except Exception:
                    pass  # 如果删除失败，至少我们尝试过了
    
    def _compare_output(self, actual: str, expected: str) -> bool:
        """比较输出（处理浮点数精度等）"""
        actual = actual.strip()
        expected = expected.strip()
        
        # 处理浮点数
        if '.' in expected:
            try:
                actual_num = float(actual)
                expected_num = float(expected)
                return abs(actual_num - expected_num) < 0.01
            except:
                pass
        
        # 处理列表
        if actual.startswith('[') and expected.startswith('['):
            # 移除空格和引号差异
            import re
            actual_clean = re.sub(r'\s+', '', actual).replace("'", "").replace('"', "")
            expected_clean = re.sub(r'\s+', '', expected).replace("'", "").replace('"', "")
            return actual_clean == expected_clean
        
        return actual == expected


class PseudoCorrectGenerator:
    """伪正确代码生成器主类"""
    
    def __init__(self, testcase_files: Dict[str, str] = None):
        print("初始化伪正确代码生成器...")
        
        if testcase_files is None:
            testcase_files = {
                "2910": "2910.xml",
                "3039": "3039.xml"
            }
        
        # 加载测试用例
        self.testcases_2910 = self._load_testcases(testcase_files['2910'])
        self.testcases_3039 = self._load_testcases(testcase_files['3039'])
        
        print(f"测试用例加载完成:")
        print(f"  2910题: {len(self.testcases_2910)} 个测试用例")
        print(f"  3039题: {len(self.testcases_3039)} 个测试用例")
    
    def _load_testcases(self, xml_file: str) -> List[Dict[str, str]]:
        """从XML文件加载测试用例"""
        if not os.path.exists(xml_file):
            print(f"警告: 文件 {xml_file} 不存在")
            return []
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            testcases = []
            for elem in root:
                if elem.tag.startswith('testData'):
                    input_elem = elem.find('input')
                    output_elem = elem.find('output')
                    
                    if input_elem is not None and output_elem is not None:
                        input_str = input_elem.text.strip() if input_elem.text else ''
                        output_str = output_elem.text.strip() if output_elem.text else ''
                        
                        testcases.append({
                            'input': input_str,
                            'expected': output_str
                        })
            
            return testcases
        except Exception as e:
            print(f"加载测试用例失败 {xml_file}: {e}")
            return []
    
    def generate_for_problem(self, problem_id: str, correct_code: str) -> List[Dict[str, Any]]:
        """为特定问题生成伪正确代码"""
        print(f"\n{'='*60}")
        print(f"为问题 {problem_id} 生成伪正确代码")
        print(f"{'='*60}")
        
        # 获取测试用例
        test_cases = self.testcases_2910 if problem_id == "2910" else self.testcases_3039
        if not test_cases:
            print("错误: 没有测试用例")
            return []
        
        # 创建测试预言
        test_oracle = TestOracle(test_cases)
        
        # 首先验证原始代码是否正确
        print("验证原始代码...")
        original_passed, original_results = test_oracle.test_code(correct_code)
        
        if not original_passed:
            print("错误: 原始代码未能通过所有测试")
            print(f"通过率: {original_results['passed']}/{original_results['total']}")
            return []
        
        print("✓ 原始代码通过所有测试")
        
        # 创建智能变异器
        mutator = IntelligentMutator(correct_code, problem_id)
        
        # 生成变异代码
        mutation_records = mutator.generate_pseudo_correct_code(max_attempts=100)
        
        # 筛选伪正确代码（通过全部测试的变异代码）
        pseudo_correct_codes = []
        
        print(f"\n测试 {len(mutation_records)} 个变异代码...")
        
        for i, record in enumerate(mutation_records):
            code = record["code"]
            
            # 跳过与原始代码相同的变异
            if code == correct_code:
                continue
            
            # 测试变异代码
            passed, test_results = test_oracle.test_code(code)
            
            if passed:
                # 这是伪正确代码！
                pseudo_record = {
                    "filename": f"pseudo_{problem_id}_{i:03d}.py",
                    "code_content": code,
                    "is_pseudo_correct": True,  # 通过全部测试
                    "problem_id": problem_id,
                    "mutation_type": record["mutation_type"],
                    "energy_used": record["energy_used"],
                    "test_results": {
                        "total_tests": test_results["total"],
                        "passed_tests": test_results["passed"],
                        "all_passed": True
                    },
                    "static_analysis": mutator.energy_info,
                    "mutation_details": {
                        "type": record["mutation_type"],
                        "attempt": record["attempt"]
                    }
                }
                
                pseudo_correct_codes.append(pseudo_record)
                print(f"  ✓ 发现伪正确代码 #{len(pseudo_correct_codes)}: {record['mutation_type']}")
        
        print(f"\n生成完成: 找到 {len(pseudo_correct_codes)} 个伪正确代码")
        return pseudo_correct_codes
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_dir: str = "pseudo_correct_dataset"):
        """保存数据集"""
        if output_dir is None:
            output_dir = "pseudo_correct_dataset"
        
        print(f"\n保存数据集到: {output_dir}")
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        # 保存每个伪正确代码
        for record in dataset:
            # 使用问题ID和原始文件名创建更清晰的文件名
            original_name = os.path.splitext(record["original_filename"])[0]
            filename = f"pseudo_{record['problem_id']}_{original_name}_{record['mutation_type']}.json"
            
            # 确保文件名有效
            import re
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)  # 替换非法字符
            
            filepath = os.path.join(output_dir, filename)
            
            # 只保存必要字段
            simplified_record = {
                "filename": record["filename"],
                "code_content": record["code_content"],
                "is_pseudo_correct": record["is_pseudo_correct"],
                "problem_id": record["problem_id"],
                "mutation_type": record["mutation_type"],
                "original_filename": record.get("original_filename", ""),
                "original_size": record.get("original_size", 0)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(simplified_record, f, indent=2, ensure_ascii=False)
        
        # 创建汇总文件
        summary = {
            "total_files": len(dataset),
            "problem_distribution": {
                "2910": sum(1 for r in dataset if r["problem_id"] == "2910"),
                "3039": sum(1 for r in dataset if r["problem_id"] == "3039")
            },
            "mutation_type_distribution": {},
            "original_file_stats": {},
            "files": [
                {
                    "filename": r["filename"],
                    "problem_id": r["problem_id"],
                    "mutation_type": r["mutation_type"],
                    "original_filename": r.get("original_filename", ""),
                    "original_size": r.get("original_size", 0)
                }
                for r in dataset
            ]
        }
        
        # 统计变异类型分布
        for record in dataset:
            mt = record["mutation_type"]
            summary["mutation_type_distribution"][mt] = summary["mutation_type_distribution"].get(mt, 0) + 1
        
        # 统计原始文件贡献
        for record in dataset:
            original_file = record.get("original_filename", "unknown")
            summary["original_file_stats"][original_file] = summary["original_file_stats"].get(original_file, 0) + 1
        
        summary_path = os.path.join(output_dir, "dataset_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"保存完成: {summary['total_files']} 个文件")
        print(f"汇总文件: {summary_path}")
        return summary


def main():
    """主函数"""
    print("基于AFL++思想的伪正确代码生成器")
    print("=" * 60)
    print("说明: 生成通过全部测试但包含潜在缺陷的代码")
    print("=" * 60)
    
    # 获取当前脚本所在目录（最灵活的方式）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前目录: {current_dir}")

    # 定义正确代码目录（相对路径）
    success_dirs = {
        "2910": os.path.join(current_dir, "2910", "success"),
        "3039": os.path.join(current_dir, "3039", "success")
    }

    print(f"2910正确代码目录: {success_dirs['2910']}")
    print(f"3039正确代码目录: {success_dirs['3039']}")

    # 检查测试用例文件
    testcase_files = {
        "2910": os.path.join(current_dir, "2910.xml"),
        "3039": os.path.join(current_dir, "3039.xml")
    }

    for problem_id, filepath in testcase_files.items():
        if not os.path.exists(filepath):
            print(f"错误: 测试用例文件不存在 - {filepath}")
            return
        else:
            print(f"找到测试用例文件: {filepath}")

    # 检查正确代码目录是否存在
    for problem_id, dir_path in success_dirs.items():
        if not os.path.exists(dir_path):
            print(f"错误: 目录不存在 - {dir_path}")
            return
        else:
            print(f"找到正确代码目录: {dir_path}")
    
    # 加载所有正确代码
    print("\n加载正确代码文件...")
    correct_codes = {}

    for problem_id, dir_path in success_dirs.items():
        print(f"\n扫描目录: {dir_path}")
        
        if not os.path.exists(dir_path):
            print(f"  目录不存在，跳过")
            continue
        
        problem_codes = []
        py_files = [f for f in os.listdir(dir_path) if f.endswith('.py')]
        
        if not py_files:
            print(f"  警告: 没有找到.py文件")
            continue
        
        for filename in sorted(py_files):  # 排序以便按顺序处理
            filepath = os.path.join(dir_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code_content = f.read().strip()
                
                if code_content:
                    problem_codes.append({
                        "filename": filename,
                        "filepath": filepath,
                        "code_content": code_content,
                        "size": len(code_content)
                    })
                    print(f"  ✓ {filename} ({len(code_content)} 字符)")
                    
            except Exception as e:
                print(f"  ✗ 加载失败 {filename}: {e}")
        
        if problem_codes:
            correct_codes[problem_id] = problem_codes
            print(f"成功加载 {len(problem_codes)} 个正确代码文件")
        else:
            print(f"问题 {problem_id} 没有可用的正确代码")
    
    if not correct_codes:
        print("错误: 没有加载到任何正确代码")
        return
    
    # 创建生成器
    generator = PseudoCorrectGenerator()
    
    # 为每个问题生成伪正确代码
    all_pseudo_codes = []
    
    for problem_id in ["2910", "3039"]:
        if problem_id not in correct_codes:
            print(f"\n跳过问题 {problem_id}: 没有正确代码")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理问题: {problem_id} (共{len(correct_codes[problem_id])}个正确代码)")
        print(f"{'='*60}")
        
        problem_correct_codes = correct_codes[problem_id]
        
        for i, code_info in enumerate(problem_correct_codes):
            print(f"\n[{i+1}/{len(problem_correct_codes)}] 处理: {code_info['filename']}")
            print("-" * 40)
            
            # 为每个正确代码生成伪正确代码
            pseudo_codes = generator.generate_for_problem(
                problem_id=problem_id,
                correct_code=code_info["code_content"]
            )
            
            # 添加原始文件信息
            for pseudo_code in pseudo_codes:
                pseudo_code.update({
                    "original_filename": code_info["filename"],
                    "original_filepath": code_info["filepath"],
                    "original_size": code_info["size"]
                })
            
            all_pseudo_codes.extend(pseudo_codes)
            
            if pseudo_codes:
                print(f"  生成了 {len(pseudo_codes)} 个伪正确代码")
            else:
                print(f"  未生成伪正确代码")
    
    # 保存数据集
    if all_pseudo_codes:
        # 修改保存函数，让它能接收自定义输出目录
        output_dir = os.path.join(current_dir, "pseudo_correct_dataset")
        summary = generator.save_dataset(all_pseudo_codes, output_dir)
        
        print(f"\n{'='*60}")
        print("数据集生成完成!")
        print(f"{'='*60}")
        print(f"总计: {summary['total_files']} 个伪正确代码")
        
        # 详细统计
        print(f"\n详细统计:")
        for problem_id in ["2910", "3039"]:
            if problem_id in correct_codes:
                problem_pseudo = [r for r in all_pseudo_codes if r["problem_id"] == problem_id]
                print(f"  {problem_id}题:")
                print(f"    - 原始正确代码: {len(correct_codes[problem_id])} 个")
                print(f"    - 生成伪正确代码: {len(problem_pseudo)} 个")
                
                # 按原始文件统计
                if problem_pseudo:
                    file_stats = {}
                    for record in problem_pseudo:
                        filename = record["original_filename"]
                        file_stats[filename] = file_stats.get(filename, 0) + 1
                    
                    for filename, count in file_stats.items():
                        print(f"      {filename}: {count} 个")
                else:
                    print(f"      (无生成)")
        
        print(f"\n变异类型分布:")
        for mt, count in summary['mutation_type_distribution'].items():
            print(f"  {mt}: {count} 个")
        
        print(f"\n所有文件已保存到: {output_dir}")
        print(f"汇总文件: {os.path.join(output_dir, 'dataset_summary.json')}")
    else:
        print("\n未生成任何伪正确代码")

if __name__ == "__main__":
    main()