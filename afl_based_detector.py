import ast
import os
import random
import math
import re
from collections import defaultdict
from advanced_executor import execute_code, check_multiple_testcases
from testcase_parser import parse_testcases

class AFLBasedDetector:
    """基于AFL思想的伪正确代码检测器"""
    
    def __init__(self, problem_id):
        self.problem_id = problem_id
        self.reference_code = self.load_reference_code()
        self.standard_testcases = parse_testcases(f'{problem_id}.xml')
        
        # 数据结构
        self.coverage_map = {}  # 存储代码的覆盖信息
        self.interesting_inputs = []  # 发现的有趣输入
        self.mutation_strategies = self.get_mutation_strategies()
        
        # 为不同题目定制的输入生成器
        self.input_generators = {
            '2910': self.generate_2910_input,
            '3039': self.generate_3039_input
        }
    
    def load_reference_code(self):
        """加载参考代码"""
        if self.problem_id == '2910':
            return '''h=eval(input())
n=eval(input())
alp=0.5
sum1=h
for i in range(1,n):
    sum1+=h*alp*2
    h=h*alp
print("%.2f" %sum1)'''
        else:
            return '''a = eval(input())
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
print(a)'''
    
    def get_mutation_strategies(self):
        """AFL的变异策略"""
        strategies = [
            ('bit_flip', self.mutate_bit_flip),      # 位翻转
            ('arithmetic', self.mutate_arithmetic),  # 算术变异
            ('interesting', self.mutate_interesting), # 插入特殊值
            ('havoc', self.mutate_havoc),            # 随机组合变异
        ]
        return strategies
    
    # ========== AFL风格的变异函数 ==========
    
    def mutate_bit_flip(self, value):
        """位翻转变异（针对整数）"""
        if isinstance(value, int):
            # 随机翻转一个位
            bit_pos = random.randint(0, 31)
            return value ^ (1 << bit_pos)
        return value
    
    def mutate_arithmetic(self, value):
        """算术变异"""
        if isinstance(value, (int, float)):
            delta = random.choice([-10, -5, -2, -1, 1, 2, 5, 10])
            # 加/减一个小值
            if random.random() < 0.5:
                return value + delta
            else:
                return value - delta
        return value
    
    def mutate_interesting(self, value):
        """插入特殊值 类似AFL的字典"""
        interesting_values = [
            0, 1, -1, 255, 256, 1024, 
            float('inf'), -float('inf'),
            float('nan')
        ]
        
        if isinstance(value, (int, float)):
            if random.random() < 0.3:
                return random.choice(interesting_values)
        
        return value
    
    def mutate_havoc(self, value, depth=3):
        """随机组合变异 AFL的havoc阶段"""
        result = value
        
        for _ in range(depth):
            strategy = random.choice(self.mutation_strategies)
            if strategy[0] != 'havoc':  # 避免无限递归
                result = strategy[1](result)
        
        return result
    
    # ========== 题目特定的输入生成 ==========
    
    def generate_2910_input(self, base_input=None):
        """生成2910题的测试输入"""
        if base_input:
            # 解析现有输入
            lines = base_input.strip().split('\n')
            try:
                h = float(lines[0]) if len(lines) > 0 and lines[0].strip() else random.randint(1, 100)
                n = float(lines[1]) if len(lines) > 1 and lines[1].strip() else random.randint(1, 100)
            except (ValueError, IndexError):
                h = random.randint(1, 100)
                n = random.randint(1, 100)
        else:
            h = random.randint(1, 100)
            n = random.randint(1, 100)
        
        # 应用变异
        if random.random() < 0.7:
            strategy = random.choice(self.mutation_strategies)
            # 确保变异后的值是有效的
            original_h = h
            h = strategy[1](h)
            
            # 检查变异后的值是否有效
            try:
                h_float = float(h)
                if math.isinf(h_float) or math.isnan(h_float):
                    h = original_h  # 恢复原值
            except (ValueError, TypeError):
                h = original_h
        
        if random.random() < 0.7:
            strategy = random.choice(self.mutation_strategies)
            original_n = n
            n = strategy[1](n)
            
            # 检查变异后的值是否有效
            try:
                n_float = float(n)
                if math.isinf(n_float) or math.isnan(n_float):
                    n = original_n
            except (ValueError, TypeError):
                n = original_n
        
        # 确保n是正整数 - 正确缩进版本
        try:
            n_int = int(float(n))
            # 限制n的范围在合理范围内
            if n_int <= 0 or n_int > 1000:
                n_int = random.randint(1, 100)
            n = n_int
        except (ValueError, TypeError, OverflowError):
            n = random.randint(1, 100)
        
        # 限制h的范围 - 正确缩进版本
        try:
            h_float = float(h)
            if not (-1000 <= h_float <= 1000):
                h = random.randint(1, 100)
        except (ValueError, TypeError):
            h = random.randint(1, 100)
        
        return f"{h}\n{n}"
    
    def generate_3039_input(self, base_input=None):
        """生成3039题的测试输入"""
        if base_input:
            # 尝试解析列表
            try:
                lst = eval(base_input.strip())
                if not isinstance(lst, list):
                    lst = []
            except:
                lst = []
        else:
            lst = []
        
        # 随机决定列表长度
        target_len = random.randint(0, 20)
        
        # 如果现有列表太短，扩展它
        while len(lst) < target_len:
            lst.append(random.randint(-100, 100))
        
        # 如果太长，随机删除一些
        if len(lst) > target_len:
            lst = lst[:target_len]
        
        # 应用变异：随机修改一些元素
        for i in range(len(lst)):
            if random.random() < 0.3:  # 30%的概率变异每个元素
                strategy = random.choice(self.mutation_strategies)
                lst[i] = strategy[1](lst[i])
        
        # 确保是有效的Python列表表达式
        return str(lst)
    
    # ========== 覆盖率收集 ==========
    
    def collect_coverage(self, code_str, test_input):
        """收集代码执行时的分支覆盖"""
        # 使用AST注入覆盖点
        try:
            tree = ast.parse(code_str)
            line_coverage = set()
            
            # 简化版：记录执行的行号
            # 在实际实现中，这里应该用sys.settrace
            # 为了演示，我们假设能获取到行号信息
            
            # 执行代码并记录输出
            output = execute_code(code_str, test_input)
            
            # 这里应该返回更详细的覆盖信息
            # 比如：执行了哪些分支、哪些条件
            return {
                'output': output,
                'line_coverage': line_coverage,  # 实际应该是执行的行号
                'success': 'ERROR' not in output
            }
            
        except Exception as e:
            return {
                'output': f"ERROR: {str(e)}",
                'line_coverage': set(),
                'success': False
            }
    
    # ========== 核心检测算法 ==========
    
    def fuzz_student_code(self, student_code, max_tests=100):
        """
        AFL风格的模糊测试学生代码
        返回是否发现bug,发现bug的测试用例
        """
        print(f"  开始模糊测试，最多{max_tests}个测试...")
        
        found_bugs = []
        seed_queue = []  # 种子队列（AFL核心概念）
        
        # 1. 初始种子：标准测试用例
        for testcase in self.standard_testcases:
            seed_queue.append(testcase['input'])
        
        # 2. 生成一些随机种子
        generator = self.input_generators[self.problem_id]
        for _ in range(5):
            seed_queue.append(generator(None))
        
        total_coverage = set()
        
        # 3. 主模糊测试循环
        for iteration in range(max_tests):
            if not seed_queue:
                break
            
            # AFL调度：选择种子
            seed = random.choice(seed_queue)
            
            # 为每个种子分配"能量"
            energy = random.randint(1, 8)
            
            for _ in range(energy):
                # 生成变异输入
                generator = self.input_generators[self.problem_id]
                test_input = generator(seed)
                
                # 执行学生代码
                student_result = self.collect_coverage(student_code, test_input)
                
                # 执行参考代码
                reference_result = self.collect_coverage(self.reference_code, test_input)
                
                # 检查输出是否不同
                if (student_result['success'] and reference_result['success'] and 
                    student_result['output'] != reference_result['output']):
                    
                    # 发现bug！
                    bug_info = {
                        'test_input': test_input,
                        'student_output': student_result['output'],
                        'reference_output': reference_result['output'],
                        'iteration': iteration
                    }
                    
                    found_bugs.append(bug_info)
                    print(f"    发现bug! 输入: {repr(test_input)}")
                    print(f"    学生输出: {student_result['output']}")
                    print(f"    参考输出: {reference_result['output']}")
                
                # 检查是否发现新覆盖（简化版）
                # 实际应该对比覆盖率位图
                new_coverage = False  # 这里需要实际计算
            
            # 如果发现bug，提前结束
            if found_bugs and len(found_bugs) >= 3:
                break
        
        return len(found_bugs) > 0, found_bugs
    
    def detect_pseudo_correct(self, student_code):
        """
        检测伪正确代码
        步骤：
        1. 首先检查是否能通过所有标准测试
        2. 如果能通过，进行模糊测试
        3. 如果模糊测试发现bug,就是伪正确代码
        """
        print("  检查标准测试用例...")
        
        # 1. 检查标准测试用例
        passes_standard = True
        for testcase in self.standard_testcases:
            student_output = execute_code(student_code, testcase['input'])
            expected_output = testcase['expected']
            
            # 对于2910题，需要处理浮点精度
            if self.problem_id == '2910':
                try:
                    student_float = float(student_output)
                    expected_float = float(expected_output)
                    if abs(student_float - expected_float) > 0.01:
                        passes_standard = False
                        break
                except:
                    if student_output != expected_output:
                        passes_standard = False
                        break
            else:
                if student_output != expected_output:
                    passes_standard = False
                    break
        
        if not passes_standard:
            return False, "未通过标准测试", []
        
        print("  标准测试通过，开始模糊测试...")
        
        # 2. 进行模糊测试
        has_bug, bug_cases = self.fuzz_student_code(student_code, max_tests=50)
        
        if has_bug:
            return True, f"发现{len(bug_cases)}个bug", bug_cases
        else:
            return False, "未发现bug", []
    
    def check_standard_tests_with_enhanced_executor(self, code_str, testcases, timeout=5):
        """使用增强执行器检查标准测试"""
        # 先检测危险循环模式
        dangerous_patterns = self._detect_dangerous_loop_patterns(code_str)
        if dangerous_patterns:
            print(f"  警告: 检测到危险循环模式: {dangerous_patterns}")
        
        # 使用更严格的超时
        result = check_multiple_testcases(code_str, testcases, timeout=timeout)
        
        if result['all_passed']:
            return True, "通过所有标准测试"
        else:
            failed = [r for r in result['results'] if not r['success']]
            reasons = [f"用例{f['testcase']}: {f['reason']}" for f in failed[:3]]
            return False, f"未通过测试: {'; '.join(reasons)}"
    
    def _detect_dangerous_loop_patterns(self, code_str):
        """检测代码中的危险循环模式"""
        patterns = [
            # 模式: for x in list_var: list_var.append(...)
            (r'for\s+\w+\s+in\s+(\w+)\s*:.*\1\.append\(', 
             "在遍历列表时向同一列表添加元素"),
            
            # 模式: while True: 或 while 1:
            (r'while\s+(True|1|\(\))\s*:', 
             "无限循环"),
            
            # 模式: 循环条件变量在循环内未修改
            (r'while\s+(\w+)\s*(>|<|>=|<=|==|!=)\s*[\w\.]+\s*:.*?(?!\1\s*=)', 
             "循环条件可能永远为真"),
        ]
        
        issues = []
        for pattern, description in patterns:
            if re.search(pattern, code_str, re.DOTALL | re.IGNORECASE):
                issues.append(description)
        
        return issues
    
    def build_dataset_with_fuzzing(self, student_dir):
        """
        使用模糊测试构建伪正确代码数据集
        """
        pseudo_codes = []
        
        success_files = [f for f in os.listdir(student_dir) 
                        if f.endswith('.py')]
        
        print(f"检查 {len(success_files)} 个成功通过的代码...")
        
        for i, filename in enumerate(success_files, 1):
            print(f"\n[{i}/{len(success_files)}] 处理: {filename}")
            
            with open(os.path.join(student_dir, filename), 'r', encoding='utf-8') as f:
                code = f.read()
            
            # 使用模糊测试检测
            is_pseudo, reason, bug_cases = self.detect_pseudo_correct(code)
            
            if is_pseudo:
                # AST分析找出可能的问题模式
                ast_issues = self.analyze_with_ast(code)
                
                pseudo_codes.append({
                    'filename': filename,
                    'code': code,
                    'detection_reason': reason,
                    'bug_cases': bug_cases[:3],  # 最多保存3个发现的bug用例
                    'ast_issues': ast_issues,
                    'problem_id': self.problem_id
                })
                
                print(f"  ✓ 标记为伪正确代码: {reason}")
            else:
                print(f"  ✗ 不是伪正确代码: {reason}")
        
        return pseudo_codes
    
    # ========== AST分析部分 ==========
    
    def analyze_with_ast(self, code_str):
        """结合AFL思想的AST分析 - 针对具体题目的深度分析"""
        issues = []
        
        try:
            tree = ast.parse(code_str)
            
            if self.problem_id == '2910':
                issues.extend(self._analyze_2910_ast(tree))
            elif self.problem_id == '3039':
                issues.extend(self._analyze_3039_ast(tree))
            
        except SyntaxError as e:
            issues.append(f"语法错误: {e}")
        
        return issues

    def _analyze_2910_ast(self, tree):
        """深度分析2910题的AST"""
        issues = []
        
        # 1. 查找硬编码输出值
        hardcoded_values = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    # 检查是否是标准测试用例的输出值
                    std_outputs = [2.88, 26.96, 21.00, 18.00, 6.00]
                    if abs(node.value - round(node.value, 2)) < 0.001:  # 保留两位小数
                        rounded = round(node.value, 2)
                        if rounded in std_outputs:
                            hardcoded_values.add(rounded)
        
        if hardcoded_values:
            issues.append(f"发现硬编码标准测试输出: {sorted(hardcoded_values)}")
        
        # 2. 分析循环结构 - 检查循环次数偏差
        loop_analysis = self._analyze_2910_loops(tree)
        if loop_analysis:
            issues.extend(loop_analysis)
        
        # 3. 检查边界条件处理
        boundary_analysis = self._analyze_2910_boundaries(tree)
        if boundary_analysis:
            issues.extend(boundary_analysis)
        
        # 4. 检查浮点数精度问题
        precision_issues = self._analyze_2910_precision(tree)
        if precision_issues:
            issues.extend(precision_issues)
        
        return issues

    def _analyze_2910_loops(self, tree):
        """分析循环次数问题"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # 检查range的参数
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                    if node.iter.func.id == 'range':
                        args = node.iter.args
                        
                        if len(args) == 1:  # range(n)
                            issues.append("使用range(n)而不是range(1,n) - 多循环一次")
                        elif len(args) == 2:  # range(start, n)
                            start = args[0]
                            end = args[1]
                            
                            # 检查起始值
                            if isinstance(start, ast.Constant):
                                if start.value != 1:
                                    issues.append(f"循环起始值应为1，实际为{start.value}")
                            else:
                                # 如果是变量，标记为需要检查
                                issues.append("循环起始值使用变量，可能不是1")
                        
                        # 检查循环体内是否有条件提前退出
                        has_break = False
                        for child in ast.walk(node):
                            if isinstance(child, ast.Break):
                                has_break = True
                                break
                        
                        if has_break:
                            issues.append("循环中有break语句，可能提前结束循环")
        
        return issues

    def _analyze_2910_boundaries(self, tree):
        """分析边界条件"""
        issues = []
        
        # 检查是否有对n的特殊处理
        has_n_special_case = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # 检查条件中是否包含对n的边界检查
                test = node.test
                
                # 提取条件中的变量和常量
                conditions = []
                
                # 简单提取比较表达式
                if isinstance(test, ast.Compare):
                    left = test.left
                    ops = test.ops
                    comparators = test.comparators
                    
                    # 检查是否比较了n
                    left_str = ast.unparse(left) if hasattr(ast, 'unparse') else str(left)
                    if 'n' in left_str:
                        for comparator in comparators:
                            if isinstance(comparator, ast.Constant):
                                val = comparator.value
                                if val in [0, 1, 2]:
                                    conditions.append(f"n {type(ops[0]).__name__} {val}")
                
                if conditions:
                    has_n_special_case = True
                    issues.append(f"发现边界条件检查: {conditions}")
        
        if not has_n_special_case:
            issues.append("未发现对n=0,1,2等边界情况的特殊处理")
        
        # 检查输出格式化
        has_formatting = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'format':
                        has_formatting = True
                elif isinstance(node.func, ast.Name):
                    if node.func.id == 'print':
                        # 检查print参数是否有格式控制
                        for arg in node.args:
                            if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
                                has_formatting = True
        
        if not has_formatting:
            issues.append("可能未使用格式化输出，存在浮点数精度问题风险")
        
        return issues

    def _analyze_2910_precision(self, tree):
        """分析浮点数精度问题"""
        issues = []
        
        # 检查是否使用了整数除法
        has_int_division = False
        has_float_conversion = False
        
        for node in ast.walk(tree):
            # 检查整数除法
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Div) or isinstance(node.op, ast.FloorDiv):
                    # 检查操作数类型
                    left_type = self._infer_expr_type(node.left)
                    right_type = self._infer_expr_type(node.right)
                    
                    if left_type == 'int' and right_type == 'int':
                        if isinstance(node.op, ast.Div):
                            has_int_division = True
                            issues.append("发现整数除法，可能导致精度丢失")
        
        # 检查是否显式转换了类型
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['float', 'int']:
                        has_float_conversion = True
        
        if not has_float_conversion:
            issues.append("未发现显式的float转换，可能存在整数运算精度问题")
        
        return issues

    def _infer_expr_type(self, node):
        """推断表达式类型（简化版）"""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return 'int'
            elif isinstance(node.value, float):
                return 'float'
            elif isinstance(node.value, str):
                return 'str'
        
        # 检查变量名
        if isinstance(node, ast.Name):
            name = node.id
            # 简单推断：h通常是浮点数，n通常是整数
            if name == 'h':
                return 'float'
            elif name == 'n':
                return 'int'
        
        # 检查二元运算
        if isinstance(node, ast.BinOp):
            left_type = self._infer_expr_type(node.left)
            right_type = self._infer_expr_type(node.right)
            
            # 简单类型提升规则
            if 'float' in [left_type, right_type]:
                return 'float'
            elif 'int' in [left_type, right_type]:
                return 'int'
        
        return 'unknown'

    def _analyze_3039_ast(self, tree):
        """深度分析3039题的AST"""
        issues = []
        
        # 1. 检查最大最小值查找策略
        issues.extend(self._check_3039_minmax_strategy(tree))
        
        # 2. 检查初始化问题
        issues.extend(self._check_3039_initialization(tree))
        
        # 3. 检查删除策略
        issues.extend(self._check_3039_deletion_strategy(tree))
        
        # 4. 检查边界情况
        issues.extend(self._check_3039_boundary_cases(tree))
        
        return issues

    def _check_3039_minmax_strategy(self, tree):
        """检查3039题的最大最小值查找策略"""
        issues = []
        
        # 检查是否使用内置函数
        uses_builtin_max = False
        uses_builtin_min = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == 'max':
                        uses_builtin_max = True
                    elif node.func.id == 'min':
                        uses_builtin_min = True
        
        if not uses_builtin_max or not uses_builtin_min:
            issues.append("未使用内置max/min函数，自己实现可能不准确")
        
        return issues

    def _check_3039_initialization(self, tree):
        """检查3039题的初始化问题"""
        issues = []
        
        # 查找最大最小值变量的初始化
        max_vars = []
        min_vars = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        var_name_lower = var_name.lower()
                        
                        # 检查是否可能是最大值变量
                        if 'max' in var_name_lower:
                            init_value = self._get_constant_value(node.value)
                            if init_value is not None:
                                max_vars.append((var_name, init_value))
                        
                        # 检查是否可能是最小值变量
                        elif 'min' in var_name_lower:
                            init_value = self._get_constant_value(node.value)
                            if init_value is not None:
                                min_vars.append((var_name, init_value))
        
        # 分析初始化值的问题
        for var_name, init_value in max_vars:
            if isinstance(init_value, (int, float)):
                if init_value == 0:
                    issues.append(f"最大值变量{var_name}初始化为0，如果列表值都小于0，将无法正确找到最大值")
                elif init_value < 100:
                    issues.append(f"最大值变量{var_name}初始化为{init_value}，可能小于实际最大值")
        
        for var_name, init_value in min_vars:
            if isinstance(init_value, (int, float)):
                if init_value == 0:
                    issues.append(f"最小值变量{var_name}初始化为0，如果列表值都大于0，将无法正确找到最小值")
                elif init_value > -100:
                    issues.append(f"最小值变量{var_name}初始化为{init_value}，可能大于实际最小值")
        
        return issues

    def _check_3039_deletion_strategy(self, tree):
        """检查3039题的删除策略"""
        issues = []
        
        # 检查删除操作类型
        deletion_methods = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['remove', 'pop']:
                        deletion_methods.add(node.func.attr)
        
        if deletion_methods:
            methods = ', '.join(deletion_methods)
            issues.append(f"使用{methods}方法删除元素")
        
        # 检查是否删除所有匹配值
        has_while_loop = False
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                has_while_loop = True
                issues.append("使用while循环，可能用于删除所有相同值")
        
        if not has_while_loop and len(deletion_methods) > 0:
            issues.append("可能未删除所有相同的最大最小值")
        
        return issues

    def _check_3039_boundary_cases(self, tree):
        """检查3039题的边界情况"""
        issues = []
        
        # 检查空列表处理
        has_empty_check = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test_str = self._node_to_string(node.test)
                if 'len' in test_str or 'empty' in test_str.lower():
                    has_empty_check = True
        
        if not has_empty_check:
            issues.append("未发现对空列表的显式检查")
        
        return issues
    
    def _get_constant_value(self, node):
        """获取常量节点的值"""
        if isinstance(node, ast.Constant) and hasattr(node, 'value'):
            return node.value
        return None
    
    def _node_to_string(self, node):
        """将AST节点转换为字符串（兼容版本）"""
        try:
            # 尝试使用ast.unparse（Python 3.9+）
            import ast
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
        except:
            pass
        
        # 回退方法
        if isinstance(node, ast.Compare):
            # 简化处理比较节点
            left_str = self._node_to_string(node.left)
            ops_str = ''.join([type(op).__name__ for op in node.ops])
            comparators_str = ' '.join([self._node_to_string(c) for c in node.comparators])
            return f"{left_str} {ops_str} {comparators_str}"
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant) and hasattr(node, 'value'):
            return str(node.value)
        elif isinstance(node, ast.BinOp):
            left_str = self._node_to_string(node.left)
            right_str = self._node_to_string(node.right)
            op_str = type(node.op).__name__
            return f"{left_str} {op_str} {right_str}"
        
        return str(node)


# ========== 主程序 ==========

def main():
    """主函数：构建伪正确代码数据集 - 简化文件保存版本"""
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from datetime import datetime
    from advanced_executor import check_multiple_testcases
    
    results = {}
    
    # 超时设置（秒）
    TIMEOUT = 5
    
    # 创建输出目录
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for problem_id in ['2910', '3039']:
        print(f"\n{'='*60}")
        print(f"处理题目: {problem_id}")
        print(f"{'='*60}")
        
        # 创建题目特定目录
        problem_output_dir = os.path.join(output_dir, problem_id)
        os.makedirs(problem_output_dir, exist_ok=True)
        
        # 初始化统计
        stats = {
            'success_dir_total': 0,
            'failure_dir_total': 0,
            'total_files': 0,
            'success_dir_stats': {
                'syntax_error': 0,
                'standard_test_failed': 0,
                'timeout_error': 0,
                'execution_error': 0,
                'pseudo_correct': 0,
                'truly_correct': 0
            },
            'failure_dir_stats': {
                'syntax_error': 0,
                'standard_test_passed': 0,
                'standard_test_failed': 0
            }
        }
        
        detector = AFLBasedDetector(problem_id)
        
        # 目录结构
        success_dir = f'{problem_id}/success'
        failure_dir = f'{problem_id}/failure'
        
        pseudo_codes = []
        failed_standard_codes = []
        truly_correct_codes = []
        
        # ========== 处理成功目录 ==========
        if os.path.exists(success_dir):
            success_files = [f for f in os.listdir(success_dir) if f.endswith('.py')]
            stats['success_dir_total'] = len(success_files)
            stats['total_files'] += len(success_files)
            
            print(f"成功目录: {success_dir} ({len(success_files)}个文件)")
            
            for i, filename in enumerate(success_files, 1):
                print(f"\n[{i}/{len(success_files)}] 检查成功代码: {filename}")
                
                try:
                    with open(os.path.join(success_dir, filename), 'r', encoding='utf-8') as f:
                        code = f.read()
                except Exception as e:
                    print(f"  ✗ 文件读取错误: {e}")
                    continue
                
                # 1. 检查语法
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    print(f"  ✗ 语法错误: {e}")
                    stats['success_dir_stats']['syntax_error'] += 1
                    continue
                
                # 2. 使用增强执行器检查标准测试
                print(f"  检查标准测试...")
                passes_standard, reason = detector.check_standard_tests_with_enhanced_executor(
                    code, detector.standard_testcases, timeout=TIMEOUT
                )
                
                if not passes_standard:
                    print(f"  ✗ {reason}")
                    stats['success_dir_stats']['standard_test_failed'] += 1
                    
                    # 保存未通过标准测试的代码信息
                    failed_info = {
                        'filename': filename,
                        'code': code,
                        'failed_reason': reason,
                        'problem_id': problem_id
                    }
                    failed_standard_codes.append(failed_info)
                    continue
                
                print(f"  ✓ 通过所有标准测试")
                
                # 3. 模糊测试检测伪正确代码
                print(f"  进行模糊测试...")
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(detector.detect_pseudo_correct, code)
                        is_pseudo, reason, bug_cases = future.result(timeout=10)
                except TimeoutError:
                    print(f"  ⚠ 模糊测试超时，跳过")
                    stats['success_dir_stats']['timeout_error'] += 1
                    continue
                except Exception as e:
                    print(f"  ⚠ 模糊测试错误: {e}，跳过")
                    stats['success_dir_stats']['execution_error'] += 1
                    continue
                
                if is_pseudo:
                    print(f"  ✗ 伪正确代码: {reason}")
                    stats['success_dir_stats']['pseudo_correct'] += 1
                    
                    # AST分析
                    ast_issues = detector.analyze_with_ast(code)
                    
                    pseudo_info = {
                        'filename': filename,
                        'code': code,
                        'detection_reason': reason,
                        'bug_cases': bug_cases[:3],
                        'ast_issues': ast_issues,
                        'problem_id': problem_id,
                        'source': 'success'
                    }
                    pseudo_codes.append(pseudo_info)
                    
                else:
                    print(f"  ✓ 真正正确代码")
                    stats['success_dir_stats']['truly_correct'] += 1
                    
                    truly_info = {
                        'filename': filename,
                        'code': code,
                        'problem_id': problem_id
                    }
                    truly_correct_codes.append(truly_info)
        
        # ========== 处理失败目录 ==========
        if os.path.exists(failure_dir):
            failure_files = [f for f in os.listdir(failure_dir) if f.endswith('.py')]
            stats['failure_dir_total'] = len(failure_files)
            stats['total_files'] += len(failure_files)
            
            print(f"\n失败目录: {failure_dir} ({len(failure_files)}个文件)")
            
            for i, filename in enumerate(failure_files, 1):
                print(f"\n[{i}/{len(failure_files)}] 检查失败代码: {filename}")
                
                try:
                    with open(os.path.join(failure_dir, filename), 'r', encoding='utf-8') as f:
                        code = f.read()
                except Exception as e:
                    print(f"  ✗ 文件读取错误: {e}")
                    continue
                
                # 1. 检查语法
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    print(f"  ✗ 语法错误: {e}")
                    stats['failure_dir_stats']['syntax_error'] += 1
                    continue
                
                # 2. 检查标准测试
                print(f"  检查标准测试...")
                test_result = check_multiple_testcases(code, detector.standard_testcases, timeout=TIMEOUT)
                
                if test_result['all_passed']:
                    print(f"  ⚠ 意外通过所有标准测试（可能标签错误）")
                    stats['failure_dir_stats']['standard_test_passed'] += 1
                else:
                    failed_count = test_result['failed']
                    reason = test_result['results'][0]['reason'] if test_result['results'] else "未知原因"
                    print(f"  ✗ 未通过标准测试 ({failed_count}/{test_result['total']} 失败): {reason}")
                    stats['failure_dir_stats']['standard_test_failed'] += 1
        
        # ========== 保存JSON文件 ==========
        print(f"\n正在保存结果文件...")
        
        # 1. 保存伪正确代码
        if pseudo_codes:
            pseudo_file = os.path.join(output_dir, f"pseudo_correct_{problem_id}.json")
            with open(pseudo_file, 'w', encoding='utf-8') as f:
                json.dump(pseudo_codes, f, indent=2, ensure_ascii=False, default=str)
            print(f"  伪正确代码保存到: {pseudo_file}")
        
        # 2. 保存未通过标准测试的代码
        if failed_standard_codes:
            failed_file = os.path.join(output_dir, f"failed_standard_{problem_id}.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_standard_codes, f, indent=2, ensure_ascii=False, default=str)
            print(f"  未通过标准测试代码保存到: {failed_file}")
        
        # 3. 保存真正正确代码（可选）
        if truly_correct_codes and len(truly_correct_codes) < 20:  # 如果数量不多就保存
            truly_file = os.path.join(output_dir, f"truly_correct_{problem_id}.json")
            with open(truly_file, 'w', encoding='utf-8') as f:
                json.dump(truly_correct_codes, f, indent=2, ensure_ascii=False, default=str)
            print(f"  真正正确代码保存到: {truly_file}")
        
        # ========== 计算汇总统计 ==========
        success_stats = stats['success_dir_stats']
        success_valid = (stats['success_dir_total'] - 
                        success_stats['syntax_error'] - 
                        success_stats['timeout_error'] - 
                        success_stats['execution_error'])
        
        failure_stats = stats['failure_dir_stats']
        total_correct = success_stats['truly_correct'] + success_stats['pseudo_correct']
        total_tested = success_valid
        
        # 保存结果
        results[problem_id] = {
            'stats': stats,
            'failed_standard_count': len(failed_standard_codes),
            'pseudo_correct_count': len(pseudo_codes),
            'truly_correct_count': success_stats['truly_correct'],
            'summary': {
                'success_dir_files': stats['success_dir_total'],
                'failure_dir_files': stats['failure_dir_total'],
                'total_files': stats['total_files'],
                'success_dir_valid': success_valid,
                'failed_standard_tests': len(failed_standard_codes),
                'pseudo_correct': len(pseudo_codes),
                'truly_correct': success_stats['truly_correct'],
                'success_dir_pseudo_ratio': (
                    success_stats['pseudo_correct'] / total_tested * 100 
                    if total_tested > 0 else 0
                ),
                'overall_pseudo_ratio': (
                    success_stats['pseudo_correct'] / total_correct * 100 
                    if total_correct > 0 else 0
                )
            }
        }
        
        # ========== 打印详细统计 ==========
        print(f"\n{'='*60}")
        print(f"题目 {problem_id} 详细统计:")
        print(f"{'='*60}")
        
        if stats['success_dir_total'] > 0:
            print(f"\n成功目录 ({stats['success_dir_total']}个文件):")
            print(f"  语法错误: {success_stats['syntax_error']}")
            print(f"  标准测试失败: {success_stats['standard_test_failed']}")
            print(f"  执行超时: {success_stats['timeout_error']}")
            print(f"  执行错误: {success_stats['execution_error']}")
            print(f"  有效测试文件: {success_valid}")
            print(f"  真正正确代码: {success_stats['truly_correct']}")
            print(f"  伪正确代码: {success_stats['pseudo_correct']}")
            
            if success_valid > 0:
                pseudo_ratio = success_stats['pseudo_correct'] / success_valid * 100
                print(f"  伪正确比例(基于有效测试): {pseudo_ratio:.1f}%")
            
            if total_correct > 0:
                overall_ratio = success_stats['pseudo_correct'] / total_correct * 100
                print(f"  伪正确比例(基于正确代码): {overall_ratio:.1f}%")
        
        if stats['failure_dir_total'] > 0:
            print(f"\n失败目录 ({stats['failure_dir_total']}个文件):")
            print(f"  语法错误: {failure_stats['syntax_error']}")
            print(f"  通过标准测试: {failure_stats['standard_test_passed']}")
            print(f"  未通过标准测试: {failure_stats['standard_test_failed']}")
        
        print(f"\n总计: {stats['total_files']} 个文件")
    
    # ========== 整体统计 ==========
    print(f"\n{'='*60}")
    print("整体统计结果:")
    print(f"{'='*60}")
    
    total_success_files = sum(r['stats']['success_dir_total'] for r in results.values())
    total_failure_files = sum(r['stats']['failure_dir_total'] for r in results.values())
    total_files = sum(r['stats']['total_files'] for r in results.values())
    
    total_pseudo = sum(r['stats']['success_dir_stats']['pseudo_correct'] for r in results.values())
    total_truly = sum(r['stats']['success_dir_stats']['truly_correct'] for r in results.values())
    total_failed_standard = sum(r['failed_standard_count'] for r in results.values())
    total_correct_overall = total_pseudo + total_truly
    
    print(f"总文件数: {total_files}")
    print(f"  成功目录: {total_success_files} 个文件")
    print(f"  失败目录: {total_failure_files} 个文件")
    print(f"\n成功目录详细结果:")
    print(f"  标准测试失败: {total_failed_standard} 个")
    print(f"  真正正确代码: {total_truly} 个")
    print(f"  伪正确代码: {total_pseudo} 个")
    print(f"  总计正确代码: {total_correct_overall}")
    
    if total_correct_overall > 0:
        total_ratio = total_pseudo / total_correct_overall * 100
        print(f"  总体伪正确比例: {total_ratio:.1f}%")
    
    print(f"\n所有JSON文件保存在: {output_dir}")
    
    # 保存总结
    summary = {
        'problems': results,
        'overall_stats': {
            'total_files': total_files,
            'success_dir_files': total_success_files,
            'failure_dir_files': total_failure_files,
            'failed_standard_tests': total_failed_standard,
            'truly_correct': total_truly,
            'pseudo_correct': total_pseudo,
            'total_correct': total_correct_overall,
            'pseudo_ratio': total_ratio if total_correct_overall > 0 else 0
        },
        'output_files': [
            f"pseudo_correct_2910.json",
            f"pseudo_correct_3039.json", 
            f"failed_standard_2910.json",
            f"failed_standard_3039.json"
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(output_dir, 'pseudo_correct_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n详细统计已保存到: {summary_file}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    main()