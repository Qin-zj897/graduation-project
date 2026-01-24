import ast
import inspect
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import networkx as nx


class StaticAnalyzer:
    """静态分析工具类，基于AST进行代码分析"""
    
    def __init__(self, source_code: str = None, file_path: str = None):
        """
        初始化静态分析器
        
        Args:
            source_code: 源代码字符串
            file_path: 源代码文件路径（二选一）
        """
        if source_code:
            self.source_code = source_code
        elif file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.source_code = f.read()
        else:
            raise ValueError("必须提供source_code或file_path参数")
        
        try:
            self.ast_tree = ast.parse(self.source_code)
        except SyntaxError as e:
            raise ValueError(f"代码语法错误: {e}")
        
        # 分析结果存储
        self._constants = None
        self._cfg = None
        self._predicates = None
        self._data_deps = None
        self._variable_types = None
        self._def_use_chains = None
        self._input_structure = None
        self._mutation_rules = None
        
    def extract_constants_and_comparisons(self) -> Dict[str, List[Any]]:
        """
        常量与比较值提取
        - 形成变异字典候选值：从代码中提取所有常量值和比较表达式的值，用于变异测试
        
        Returns:
            字典：{变量名: [常量值列表]}
        """
        self._constants = defaultdict(list)
        
        class ConstantExtractor(ast.NodeVisitor):
            def __init__(self, constants_dict):
                self.constants = constants_dict
                self._current_function = None
                
            def visit_FunctionDef(self, node):
                """记录当前函数"""
                old_function = self._current_function
                self._current_function = node.name
                self.generic_visit(node)
                self._current_function = old_function
                
            def visit_Assign(self, node):
                """处理赋值语句，提取常量值"""
                # 获取常量值
                if isinstance(node.value, ast.Constant):
                    const_value = node.value.value
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.constants[target.id].append({
                                'value': const_value,
                                'type': type(const_value).__name__,
                                'source': 'assignment',
                                'lineno': node.lineno,
                                'function': self._current_function
                            })
                
                # 处理二元运算中的常量（如 x = y + 10）
                elif isinstance(node.value, ast.BinOp):
                    self._extract_binop_constants(node.value, node.targets, node.lineno)
                
                self.generic_visit(node)
            
            def _extract_binop_constants(self, binop_node, targets, lineno):
                """提取二元运算中的常量"""
                # 检查左侧常量
                if isinstance(binop_node.left, ast.Constant):
                    const_value = binop_node.left.value
                    for target in targets:
                        if isinstance(target, ast.Name):
                            self.constants[target.id].append({
                                'value': const_value,
                                'type': type(const_value).__name__,
                                'source': 'binary_op_left',
                                'lineno': lineno,
                                'function': self._current_function
                            })
                
                # 检查右侧常量
                if isinstance(binop_node.right, ast.Constant):
                    const_value = binop_node.right.value
                    for target in targets:
                        if isinstance(target, ast.Name):
                            self.constants[target.id].append({
                                'value': const_value,
                                'type': type(const_value).__name__,
                                'source': 'binary_op_right',
                                'lineno': lineno,
                                'function': self._current_function
                            })
                
                # 递归检查嵌套的二元运算
                if isinstance(binop_node.left, ast.BinOp):
                    self._extract_binop_constants(binop_node.left, targets, lineno)
                if isinstance(binop_node.right, ast.BinOp):
                    self._extract_binop_constants(binop_node.right, targets, lineno)
                
            def visit_Compare(self, node):
                """处理比较表达式，提取比较值"""
                # 提取左侧变量与右侧常量的比较
                if isinstance(node.left, ast.Name):
                    var_name = node.left.id
                    for comparator in node.comparators:
                        if isinstance(comparator, ast.Constant):
                            self.constants[var_name].append({
                                'value': comparator.value,
                                'type': type(comparator.value).__name__,
                                'source': 'comparison',
                                'lineno': node.lineno,
                                'function': self._current_function,
                                'op': type(node.ops[0]).__name__ if node.ops else None
                            })
                
                # 提取左侧常量与右侧变量的比较（如 10 < x）
                elif isinstance(node.left, ast.Constant):
                    const_value = node.left.value
                    for i, comparator in enumerate(node.comparators):
                        if isinstance(comparator, ast.Name):
                            var_name = comparator.id
                            self.constants[var_name].append({
                                'value': const_value,
                                'type': type(const_value).__name__,
                                'source': 'comparison_reverse',
                                'lineno': node.lineno,
                                'function': self._current_function,
                                'op': type(node.ops[i]).__name__ if i < len(node.ops) else None
                            })
                
                self.generic_visit(node)
            
            def visit_AugAssign(self, node):
                """处理增强赋值（如 x += 1）"""
                if isinstance(node.value, ast.Constant) and isinstance(node.target, ast.Name):
                    self.constants[node.target.id].append({
                        'value': node.value.value,
                        'type': type(node.value.value).__name__,
                        'source': 'augmented_assignment',
                        'lineno': node.lineno,
                        'function': self._current_function,
                        'op': type(node.op).__name__
                    })
                self.generic_visit(node)
        
        extractor = ConstantExtractor(self._constants)
        extractor.visit(self.ast_tree)
        
        # 去重并排序
        for var in self._constants:
            unique_values = []
            seen = set()
            for item in self._constants[var]:
                key = (item['value'], item['source'])
                if key not in seen:
                    seen.add(key)
                    unique_values.append(item)
            self._constants[var] = unique_values
        
        return dict(self._constants)
    
    def build_control_flow_graph(self) -> nx.DiGraph:
        """
        构建控制流图(CFG)
        - 识别出各个分支，确保变异后的用例覆盖所有分支
        
        Returns:
            控制流图 (NetworkX有向图)
        """
        self._cfg = nx.DiGraph()
        
        class CFGBuilder(ast.NodeVisitor):
            def __init__(self, cfg_graph):
                self.cfg = cfg_graph
                self.node_counter = 0
                self.current_block = None
                self.block_stack = []
                self.function_blocks = defaultdict(list)
                
            def _new_block(self, label="block"):
                """创建新的基本块"""
                block_id = f"{label}_{self.node_counter}"
                self.node_counter += 1
                self.cfg.add_node(block_id, type=label, statements=[])
                return block_id
            
            def _add_statement(self, node, stmt_type):
                """添加语句到当前基本块"""
                if self.current_block:
                    stmt_info = {
                        'type': stmt_type,
                        'lineno': node.lineno,
                        'col_offset': node.col_offset,
                        'code': ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                    }
                    self.cfg.nodes[self.current_block]['statements'].append(stmt_info)
            
            def visit_FunctionDef(self, node):
                """处理函数定义"""
                # 创建函数入口基本块
                entry_block = self._new_block(f"func_{node.name}_entry")
                exit_block = self._new_block(f"func_{node.name}_exit")
                
                # 保存旧的基本块
                old_block = self.current_block
                self.current_block = entry_block
                
                # 添加函数信息
                self.cfg.nodes[entry_block]['function'] = node.name
                self.cfg.nodes[exit_block]['function'] = node.name
                
                # 处理函数体
                self.generic_visit(node)
                
                # 连接当前块到退出块
                if self.current_block:
                    self.cfg.add_edge(self.current_block, exit_block)
                
                # 恢复旧的基本块
                self.current_block = old_block
                
                # 记录函数的基本块
                self.function_blocks[node.name] = (entry_block, exit_block)
            
            def visit_If(self, node):
                """处理if语句，创建分支"""
                # 为if条件创建基本块
                if_block = self._new_block("if_condition")
                true_block = self._new_block("if_true")
                false_block = self._new_block("if_false")
                merge_block = self._new_block("if_merge")
                
                # 记录条件信息
                condition_info = {
                    'expression': ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test),
                    'lineno': node.lineno
                }
                self.cfg.nodes[if_block]['condition'] = condition_info
                
                # 连接当前块到if条件块
                if self.current_block:
                    self.cfg.add_edge(self.current_block, if_block)
                
                # 保存当前块并处理true分支
                old_block = self.current_block
                self.current_block = true_block
                for stmt in node.body:
                    self.visit(stmt)
                # 连接true分支到合并块
                if self.current_block:
                    self.cfg.add_edge(self.current_block, merge_block)
                
                # 处理false分支
                self.current_block = false_block
                for stmt in node.orelse:
                    self.visit(stmt)
                # 连接false分支到合并块
                if self.current_block:
                    self.cfg.add_edge(self.current_block, merge_block)
                
                # 恢复并设置当前块为合并块
                self.current_block = merge_block
                
                # 连接if条件块到两个分支
                self.cfg.add_edge(if_block, true_block, label="true")
                self.cfg.add_edge(if_block, false_block, label="false")
            
            def visit_While(self, node):
                """处理while循环"""
                # 创建循环基本块
                loop_condition = self._new_block("while_condition")
                loop_body = self._new_block("while_body")
                loop_exit = self._new_block("while_exit")
                
                # 记录循环信息
                loop_info = {
                    'expression': ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test),
                    'lineno': node.lineno
                }
                self.cfg.nodes[loop_condition]['loop'] = loop_info
                
                # 连接当前块到循环条件
                if self.current_block:
                    self.cfg.add_edge(self.current_block, loop_condition)
                
                # 连接循环条件到循环体（true分支）
                self.cfg.add_edge(loop_condition, loop_body, label="true")
                # 连接循环条件到退出（false分支）
                self.cfg.add_edge(loop_condition, loop_exit, label="false")
                
                # 处理循环体
                old_block = self.current_block
                self.current_block = loop_body
                for stmt in node.body:
                    self.visit(stmt)
                # 连接循环体回到循环条件
                if self.current_block:
                    self.cfg.add_edge(self.current_block, loop_condition)
                
                # 恢复并设置当前块为退出块
                self.current_block = loop_exit
            
            def visit_For(self, node):
                """处理for循环"""
                # 创建循环基本块
                loop_header = self._new_block("for_header")
                loop_body = self._new_block("for_body")
                loop_exit = self._new_block("for_exit")
                
                # 连接当前块到循环头
                if self.current_block:
                    self.cfg.add_edge(self.current_block, loop_header)
                
                # 连接循环头到循环体
                self.cfg.add_edge(loop_header, loop_body)
                # 连接循环头到退出
                self.cfg.add_edge(loop_header, loop_exit)
                
                # 处理循环体
                old_block = self.current_block
                self.current_block = loop_body
                for stmt in node.body:
                    self.visit(stmt)
                # 连接循环体回到循环头
                if self.current_block:
                    self.cfg.add_edge(self.current_block, loop_header)
                
                # 恢复并设置当前块为退出块
                self.current_block = loop_exit
            
            def visit_Return(self, node):
                """处理return语句"""
                self._add_statement(node, 'return')
                # return语句没有后续语句
                self.current_block = None
            
            def generic_visit(self, node):
                """处理其他语句"""
                if isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign)):
                    self._add_statement(node, type(node).__name__.lower())
                super().generic_visit(node)
        
        builder = CFGBuilder(self._cfg)
        builder.visit(self.ast_tree)
        
        return self._cfg
    
    def predicate_mining(self) -> List[Dict[str, Any]]:
        """
        谓词挖掘
        - 识别边界条件和特殊值，通过谓词组合覆盖，而不是路径覆盖，减少测试数量
        
        Returns:
            谓词列表，每个谓词包含表达式和位置信息
        """
        self._predicates = []
        
        class PredicateVisitor(ast.NodeVisitor):
            def __init__(self, predicates_list):
                self.predicates = predicates_list
                self._current_function = None
                
            def visit_FunctionDef(self, node):
                """记录当前函数"""
                old_function = self._current_function
                self._current_function = node.name
                self.generic_visit(node)
                self._current_function = old_function
            
            def visit_If(self, node):
                """提取if语句的条件谓词"""
                self._extract_predicate(node.test, 'if_condition', node.lineno)
                self.generic_visit(node)
                
            def visit_While(self, node):
                """提取while循环的条件谓词"""
                self._extract_predicate(node.test, 'while_condition', node.lineno)
                self.generic_visit(node)
                
            def visit_Assert(self, node):
                """提取断言语句"""
                self._extract_predicate(node.test, 'assert', node.lineno)
                self.generic_visit(node)
            
            def _extract_predicate(self, test_node, pred_type, lineno):
                """提取谓词信息"""
                # 解构比较表达式
                if isinstance(test_node, ast.Compare):
                    for i, (op, comparator) in enumerate(zip(test_node.ops, test_node.comparators)):
                        predicate_info = {
                            'type': pred_type,
                            'expression': ast.unparse(test_node) if hasattr(ast, 'unparse') else str(test_node),
                            'full_expression': ast.unparse(test_node) if hasattr(ast, 'unparse') else str(test_node),
                            'lineno': lineno,
                            'col_offset': test_node.col_offset,
                            'op': type(op).__name__,
                            'left': ast.unparse(test_node.left) if hasattr(ast, 'unparse') else str(test_node.left),
                            'right': ast.unparse(comparator) if hasattr(ast, 'unparse') else str(comparator),
                            'function': self._current_function,
                            'boundary_info': self._analyze_boundary(test_node.left, op, comparator)
                        }
                        self.predicates.append(predicate_info)
                else:
                    # 处理其他类型的布尔表达式
                    predicate_info = {
                        'type': pred_type,
                        'expression': ast.unparse(test_node) if hasattr(ast, 'unparse') else str(test_node),
                        'full_expression': ast.unparse(test_node) if hasattr(ast, 'unparse') else str(test_node),
                        'lineno': lineno,
                        'col_offset': test_node.col_offset,
                        'function': self._current_function
                    }
                    self.predicates.append(predicate_info)
            
            def _analyze_boundary(self, left, op, right):
                """分析边界条件"""
                boundary = {}
                
                # 检查是否为边界比较
                if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    # 尝试提取常数值作为边界
                    if isinstance(right, ast.Constant):
                        boundary['value'] = right.value
                        boundary['type'] = type(right.value).__name__
                        
                        if isinstance(op, ast.Lt):
                            boundary['condition'] = 'less_than'
                            boundary['suggested_values'] = [
                                right.value - 1 if isinstance(right.value, (int, float)) else None,
                                right.value,
                                right.value + 1 if isinstance(right.value, (int, float)) else None
                            ]
                        elif isinstance(op, ast.LtE):
                            boundary['condition'] = 'less_than_or_equal'
                            boundary['suggested_values'] = [
                                right.value - 1 if isinstance(right.value, (int, float)) else None,
                                right.value,
                                right.value + 1 if isinstance(right.value, (int, float)) else None
                            ]
                        elif isinstance(op, ast.Gt):
                            boundary['condition'] = 'greater_than'
                            boundary['suggested_values'] = [
                                right.value - 1 if isinstance(right.value, (int, float)) else None,
                                right.value,
                                right.value + 1 if isinstance(right.value, (int, float)) else None
                            ]
                        elif isinstance(op, ast.GtE):
                            boundary['condition'] = 'greater_than_or_equal'
                            boundary['suggested_values'] = [
                                right.value - 1 if isinstance(right.value, (int, float)) else None,
                                right.value,
                                right.value + 1 if isinstance(right.value, (int, float)) else None
                            ]
                
                return boundary if boundary else None
        
        visitor = PredicateVisitor(self._predicates)
        visitor.visit(self.ast_tree)
        
        return self._predicates
    
    def backward_slice(self, variable_name: str, line_number: int) -> Set[str]:
        """
        后向切片
        - 识别哪些变量需要边界测试
        - 识别功能影响的输入，避免重复测试
        
        Args:
            variable_name: 目标变量名
            line_number: 目标行号
            
        Returns:
            影响目标变量的所有语句集合（行号）
        """
        # 构建def-use链
        def_use_chains = self._build_def_use_chains()
        
        # 获取目标变量的定义和使用信息
        affected_lines = set()
        work_list = [(variable_name, line_number)]
        visited = set()
        
        while work_list:
            var, line = work_list.pop()
            if (var, line) in visited:
                continue
            visited.add((var, line))
            
            # 添加当前行
            affected_lines.add(line)
            
            # 查找定义该变量的语句
            if var in def_use_chains:
                for def_info in def_use_chains[var]['definitions']:
                    def_line = def_info['lineno']
                    if def_line < line:  # 只考虑之前的定义
                        work_list.append((var, def_line))
                        
                        # 查找定义语句中使用的变量
                        for used_var in def_info['uses']:
                            work_list.append((used_var, def_line))
        
        return affected_lines
    
    def _build_def_use_chains(self) -> Dict[str, Dict[str, List]]:
        """构建def-use链（内部方法）"""
        if self._def_use_chains is not None:
            return self._def_use_chains
        
        self._def_use_chains = defaultdict(lambda: {
            'definitions': [],
            'uses': []
        })
        
        class DefUseVisitor(ast.NodeVisitor):
            def __init__(self, chains):
                self.chains = chains
                self.current_scope = []
                
            def visit_FunctionDef(self, node):
                """进入函数作用域"""
                self.current_scope.append(node.name)
                self.generic_visit(node)
                self.current_scope.pop()
                
            def visit_Assign(self, node):
                """处理赋值语句（定义）"""
                # 获取使用的变量
                uses = self._extract_uses(node.value)
                
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        self.chains[var_name]['definitions'].append({
                            'lineno': node.lineno,
                            'scope': self.current_scope.copy(),
                            'uses': uses,
                            'node': node
                        })
                
                self.generic_visit(node)
            
            def visit_Name(self, node):
                """处理变量使用"""
                if isinstance(node.ctx, ast.Load):  # 读取变量
                    var_name = node.id
                    self.chains[var_name]['uses'].append({
                        'lineno': node.lineno,
                        'scope': self.current_scope.copy(),
                        'node': node
                    })
                            
            def _extract_uses(self, node):
                """从表达式提取使用的变量"""
                uses = []
                
                class UseExtractor(ast.NodeVisitor):
                    def __init__(self, use_list):
                        self.uses = use_list
                    
                    def visit_Name(self, node):
                        if isinstance(node.ctx, ast.Load):
                            self.uses.append(node.id)
                
                extractor = UseExtractor(uses)
                if node:
                    extractor.visit(node)
                
                return uses
        
        visitor = DefUseVisitor(self._def_use_chains)
        visitor.visit(self.ast_tree)
        
        return self._def_use_chains
    
    def build_data_dependency_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        构建Def-Use/数据依赖图
        - 记录数据从哪里产生，到哪里被使用
        - 识别哪些变量修改后可能影响输出，即应该优先变异哪些变量
        
        Returns:
            数据依赖关系字典
        """
        # 先构建def-use链
        def_use_chains = self._build_def_use_chains()
        
        self._data_deps = defaultdict(list)
        
        # 构建数据依赖关系
        for var_name, info in def_use_chains.items():
            for definition in info['definitions']:
                def_line = definition['lineno']
                # 找到使用这个定义的所有地方
                for use in info['uses']:
                    use_line = use['lineno']
                    if use_line > def_line:  # 使用在定义之后
                        # 检查是否在同一作用域
                        if self._same_scope(definition['scope'], use['scope']):
                            self._data_deps[var_name].append({
                                'from_line': def_line,
                                'to_line': use_line,
                                'type': 'def-use'
                            })
        
        # 构建变量间的依赖关系
        for var_name, info in def_use_chains.items():
            for definition in info['definitions']:
                for used_var in definition['uses']:
                    if used_var in def_use_chains:
                        self._data_deps[var_name].append({
                            'depends_on': used_var,
                            'line': definition['lineno'],
                            'type': 'data_dependency'
                        })
        
        return dict(self._data_deps)
    
    def _same_scope(self, scope1, scope2):
        """检查两个作用域是否相同"""
        return scope1 == scope2
    
    def infer_input_structure(self) -> Dict[str, Any]:
        """
        输入结构/格式推断
        - 自动推导出程序期望的输入数据格式、结构和约束
        - 生成符合规则的测试数据
        - 确定变异算子规则，对什么数据类型变异
        
        Returns:
            输入结构推断结果
        """
        self._input_structure = {
            'functions': [],
            'parameters': [],
            'expected_types': {},
            'constraints': [],
            'input_patterns': [],
            'input_calls': [],
            'main_inputs': [],
            'detected_types': {} ,
            'global_inputs': [] 
        }
        
        class InputStructureVisitor(ast.NodeVisitor):
            def __init__(self, structure):
                self.structure = structure
                self._current_function = None
                self._variable_usage = {}  # 跟踪变量使用方式
                
            def visit_FunctionDef(self, node):
                """分析函数定义"""
                func_info = {
                    'name': node.name,
                    'args': [],
                    'returns': None,
                    'docstring': ast.get_docstring(node),
                    'lineno': node.lineno
                }
                
                # 分析参数
                if node.args:
                    # 位置参数
                    if hasattr(node.args, 'args'):
                        for arg in node.args.args:
                            if arg:
                                arg_info = self._analyze_argument(arg)
                                if arg_info:
                                    arg_info['position'] = len(func_info['args'])
                                    func_info['args'].append(arg_info)
                                    
                                    # 记录到参数列表
                                    self.structure['parameters'].append({
                                        'function': node.name,
                                        **arg_info
                                    })
                    
                    # 关键字参数
                    if hasattr(node.args, 'kwonlyargs'):
                        for arg in node.args.kwonlyargs:
                            if arg:
                                arg_info = self._analyze_argument(arg)
                                if arg_info:
                                    arg_info['position'] = 'keyword'
                                    func_info['args'].append(arg_info)
                                    
                                    self.structure['parameters'].append({
                                        'function': node.name,
                                        **arg_info
                                    })
                
                # 分析返回值类型
                if node.returns:
                    try:
                        func_info['returns'] = ast.unparse(node.returns)
                    except:
                        func_info['returns'] = str(node.returns)
                
                self.structure['functions'].append(func_info)
                
                # 分析函数体中的约束和变量使用
                old_function = self._current_function
                self._current_function = node.name
                self.generic_visit(node)
                self._current_function = old_function
            
            def _analyze_argument(self, arg):
                """分析单个参数"""
                if not arg:
                    return None
                    
                try:
                    arg_info = {
                        'name': arg.arg if hasattr(arg, 'arg') else 'unknown',
                        'annotation': None,
                        'type': None,
                        'default': None
                    }
                    
                    # 获取注解
                    if hasattr(arg, 'annotation') and arg.annotation:
                        try:
                            arg_info['annotation'] = ast.unparse(arg.annotation)
                            arg_info['type'] = self._infer_type_from_annotation(arg.annotation)
                        except:
                            arg_info['annotation'] = str(arg.annotation)
                            arg_info['type'] = str(arg.annotation)
                    
                    return arg_info
                except Exception:
                    return None
            
            def _infer_type_from_annotation(self, annotation):
                """从类型注解推断类型"""
                if not annotation:
                    return None
                    
                try:
                    if isinstance(annotation, ast.Name):
                        return annotation.id
                    elif isinstance(annotation, ast.Subscript):
                        # 处理泛型如 List[int]
                        return ast.unparse(annotation)
                    elif isinstance(annotation, ast.Constant):
                        return annotation.value
                    elif hasattr(annotation, '_fields'):
                        return str(annotation)
                    else:
                        return None
                except:
                    return str(annotation) if annotation else None
            
            def visit_Assign(self, node):
                """分析赋值语句，识别输入"""
                # 分析输入赋值
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                        
                        if func_name == 'eval':
                            # 检查 eval 的参数
                            if node.value.args and isinstance(node.value.args[0], ast.Call):
                                inner_call = node.value.args[0]
                                if isinstance(inner_call.func, ast.Name) and inner_call.func.id == 'input':
                                    # 找到输入赋值
                                    input_info = {
                                        'type': 'eval_input_assignment',
                                        'variable': None,
                                        'line': node.lineno,
                                        'description': f'eval(input()) 赋值'
                                    }
                                    
                                    # 获取赋值的目标变量
                                    for target in node.targets:
                                        if isinstance(target, ast.Name):
                                            input_info['variable'] = target.id
                                            # 记录变量类型
                                            if target.id not in self.structure['detected_types']:
                                                self.structure['detected_types'][target.id] = set()
                                            self.structure['detected_types'][target.id].add('numeric')  # eval通常返回数值
                                            break
                                    
                                    if input_info['variable']:
                                        self.structure['global_inputs'].append(input_info)
                                        self.structure['main_inputs'].append({
                                            'type': 'main_eval_input',
                                            'function': 'eval(input())',
                                            'line': node.lineno,
                                            'variable': input_info['variable'],
                                            'description': f'读取并求值输入，赋值给变量 {input_info["variable"]}'
                                        })
                
                self.generic_visit(node)
            
            def visit_Call(self, node):
                """分析函数调用"""
                if not node or not hasattr(node, 'func'):
                    self.generic_visit(node)
                    return
                
                # 分析 eval(input()) 调用
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    if func_name == 'eval':
                        # 检查 eval 的参数是否是 input()
                        if node.args and isinstance(node.args[0], ast.Call):
                            inner_call = node.args[0]
                            if isinstance(inner_call.func, ast.Name) and inner_call.func.id == 'input':
                                # 记录输入调用
                                input_call = {
                                    'type': 'eval_input',
                                    'function': 'eval',
                                    'line': node.lineno,
                                    'inner_function': 'input',
                                    'returns_type': 'evaluated_expression'
                                }
                                self.structure['input_patterns'].append(input_call)
                                self.structure['input_calls'].append(input_call)
                
                self.generic_visit(node)
        
        visitor = InputStructureVisitor(self._input_structure)
        
        try:
            visitor.visit(self.ast_tree)
            
            # 后处理：更新参数类型
            for func in self._input_structure['functions']:
                for arg in func['args']:
                    arg_name = arg['name']
                    # 如果参数有检测到的类型，更新它
                    if arg_name in self._input_structure['detected_types']:
                        detected_types = self._input_structure['detected_types'][arg_name]
                        # 如果有明确的类型检测，更新参数类型
                        if detected_types:
                            # 选择最具体的类型
                            if 'list' in detected_types:
                                arg['type'] = 'list'
                            elif 'dict' in detected_types:
                                arg['type'] = 'dict'
                            elif 'str' in detected_types:
                                arg['type'] = 'str'
                            elif 'int' in detected_types:
                                arg['type'] = 'int'
                            else:
                                # 如果有多个类型，显示第一个
                                arg['type'] = next(iter(detected_types))
        except Exception as e:
            self._input_structure['error'] = str(e)
        
        return self._input_structure
    
    def get_variable_types(self) -> Dict[str, str]:
        """
        推断变量类型
        - 用于确定变异算子规则
        
        Returns:
            变量类型字典
        """
        if self._variable_types is not None:
            return self._variable_types
        
        self._variable_types = {}
        
        # 排除的函数名列表
        BUILTIN_FUNCTIONS = {
            'eval', 'input', 'print', 'len', 'range', 'str', 'int', 'float', 
            'list', 'dict', 'set', 'tuple', 'bool', 'type', 'isinstance'
        }
        
        # 收集真正的变量（排除函数调用和内置函数）
        class VariableCollector(ast.NodeVisitor):
            def __init__(self):
                self.variables = set()
                self._current_function = None
                self.function_calls = set()  # 函数调用
                
            def visit_FunctionDef(self, node):
                old_function = self._current_function
                self._current_function = node.name
                # 收集参数（真正的变量）
                for arg in node.args.args:
                    if isinstance(arg, ast.arg):
                        self.variables.add(arg.arg)
                self.generic_visit(node)
                self._current_function = old_function
                
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # 检查是否是真正的变量（不是函数调用）
                        if target.id not in BUILTIN_FUNCTIONS:
                            self.variables.add(target.id)
                self.generic_visit(node)
                
            def visit_For(self, node):
                if isinstance(node.target, ast.Name):
                    if node.target.id not in BUILTIN_FUNCTIONS:
                        self.variables.add(node.target.id)
                self.generic_visit(node)
                
            def visit_Call(self, node):
                # 记录函数调用，但不作为变量
                if isinstance(node.func, ast.Name):
                    self.function_calls.add(node.func.id)
                self.generic_visit(node)
                
            def visit_Name(self, node):
                # 只收集作为存储（赋值目标）或读取的变量，且不是内置函数
                if isinstance(node.ctx, (ast.Load, ast.Store)):
                    if node.id not in BUILTIN_FUNCTIONS and node.id not in self.function_calls:
                        self.variables.add(node.id)
        
        collector = VariableCollector()
        collector.visit(self.ast_tree)
        
        # 真正的变量（排除函数名）
        true_variables = collector.variables - collector.function_calls - BUILTIN_FUNCTIONS
        
        class TypeInferencer(ast.NodeVisitor):
            def __init__(self, type_dict, variables_set):
                self.types = type_dict
                self.variables = variables_set
                self._current_function = None
                
            def visit_FunctionDef(self, node):
                old_function = self._current_function
                self._current_function = node.name
                
                # 记录函数参数类型
                for arg in node.args.args:
                    if isinstance(arg, ast.arg) and arg.arg in self.variables:
                        self.types[arg.arg] = 'Any'  # 默认参数类型
                
                self.generic_visit(node)
                self._current_function = old_function
                
            def visit_For(self, node):
                if isinstance(node.target, ast.Name) and node.target.id in self.variables:
                    loop_var = node.target.id
                    # 设置循环变量类型
                    self.types[loop_var] = 'element'  # 循环变量通常是元素
                    
                    # 分析迭代对象
                    if isinstance(node.iter, ast.Name) and node.iter.id in self.variables:
                        iter_var = node.iter.id
                        # 被迭代的对象通常是列表
                        self.types[iter_var] = 'list'
                
                self.generic_visit(node)
                
            def visit_Call(self, node):
                # 分析 nums.count(x) 中的 nums 类型
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        obj_name = node.func.value.id
                        method_name = node.func.attr
                        
                        if obj_name in self.variables:
                            # 根据方法推断类型
                            if method_name == 'count':
                                self.types[obj_name] = 'list'
                            elif method_name == 'len':
                                # len() 的参数通常是序列
                                self.types[obj_name] = 'list'
                
                self.generic_visit(node)
                
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in self.variables:
                        # 分析赋值右侧
                        if isinstance(node.value, ast.Call):
                            # nums = eval(input())
                            if isinstance(node.value.func, ast.Name):
                                if node.value.func.id == 'eval':
                                    self.types[target.id] = 'list'  # 通常eval返回列表
                        elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                            # y = search(nums)
                            if node.value.func.id == 'search':
                                self.types[target.id] = 'Any'  # search可能返回元素或False
                
                self.generic_visit(node)
        
        inferencer = TypeInferencer(self._variable_types, true_variables)
        
        try:
            inferencer.visit(self.ast_tree)
        except Exception as e:
            print(f"类型推断警告: {e}")
        
        # 确保所有变量都有类型
        for var in true_variables:
            if var not in self._variable_types:
                self._variable_types[var] = 'Unknown'
        
        return self._variable_types