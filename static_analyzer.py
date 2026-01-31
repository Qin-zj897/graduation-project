import ast
import inspect
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import networkx as nx


class StaticAnalyzer:
    """静态分析工具类,基于AST进行代码分析"""
    
    def __init__(self, source_code: str = None, file_path: str = None):
        """
        初始化静态分析器
        
        Args:
            source_code: 源代码字符串
            file_path: 源代码文件路径(二选一)
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
        

        
    def extract_constants_and_comparisons(self) -> Dict[str, Any]:
        """
        常量与比较值提取
        - 提取所有常量值和比较表达式的值
        
        Returns:
            dict, 变量名到常量值列表的映射, 每个常量包含value(值)和type(类型)
            例如: {'h': [{'value': 0, 'type': 'int'}, {'value': 11, 'type': 'int'}], 'n': [{'value': 10, 'type': 'int'}]}
            失败时返回空字典 {}
        """
        try:
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
                    """处理赋值语句,提取常量值"""
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
                    
                    # 处理二元运算中的常量(如 x = y + 10)
                    elif isinstance(node.value, ast.BinOp):
                        self._extract_binop_constants(node.value, node.targets, node.lineno)
                    
                    # 处理函数调用中的常量(如 x = len(y))
                    elif isinstance(node.value, ast.Call):
                        self._extract_call_constants(node.value, node.targets, node.lineno)
                    
                    # 处理下标访问中的常量(如 x = y[0])
                    elif isinstance(node.value, ast.Subscript):
                        self._extract_subscript_constants(node.value, node.targets, node.lineno)
                    
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
                
                def _extract_call_constants(self, call_node, targets, lineno):
                    """提取函数调用中的常量"""
                    if isinstance(call_node.func, ast.Name):
                        func_name = call_node.func.id
                        
                        # 处理 len(x) == 0 这样的比较
                        if func_name == 'len':
                            for target in targets:
                                if isinstance(target, ast.Name):
                                    self.constants[target.id].append({
                                        'value': 0,
                                        'type': 'int',
                                        'source': 'len_function_result',
                                        'lineno': lineno,
                                        'function': self._current_function,
                                        'call_info': f'len({ast.unparse(call_node.args[0]) if call_node.args else "unknown"})'
                                    })
                        
                        # 处理其他函数调用中的常量参数
                        for arg in call_node.args:
                            if isinstance(arg, ast.Constant):
                                const_value = arg.value
                                for target in targets:
                                    if isinstance(target, ast.Name):
                                        self.constants[target.id].append({
                                            'value': const_value,
                                            'type': type(const_value).__name__,
                                            'source': f'function_arg_{func_name}',
                                            'lineno': lineno,
                                            'function': self._current_function,
                                            'call_info': f'{func_name}({ast.unparse(arg)})'
                                        })
                
                def _extract_subscript_constants(self, subscript_node, targets, lineno):
                    """提取下标访问中的常量(增强版)"""
                    # 处理下标索引中的常量(如 a[0])
                    if isinstance(subscript_node.slice, ast.Constant):
                        const_value = subscript_node.slice.value
                        for target in targets:
                            if isinstance(target, ast.Name):
                                self.constants[target.id].append({
                                    'value': const_value,
                                    'type': type(const_value).__name__,
                                    'source': 'subscript_index',
                                    'lineno': lineno,
                                    'function': self._current_function,
                                    'subscript_info': f'{ast.unparse(subscript_node.value)}[{ast.unparse(subscript_node.slice)}]'
                                })
                    
                    # 处理切片操作中的常量(如 a[1:5:2])
                    elif isinstance(subscript_node.slice, ast.Slice):
                        slice_info = subscript_node.slice
                        slice_constants = []
                        
                        if slice_info.lower and isinstance(slice_info.lower, ast.Constant):
                            slice_constants.append(('lower', slice_info.lower.value))
                        if slice_info.upper and isinstance(slice_info.upper, ast.Constant):
                            slice_constants.append(('upper', slice_info.upper.value))
                        if slice_info.step and isinstance(slice_info.step, ast.Constant):
                            slice_constants.append(('step', slice_info.step.value))
                        
                        for const_type, const_value in slice_constants:
                            for target in targets:
                                if isinstance(target, ast.Name):
                                    self.constants[target.id].append({
                                        'value': const_value,
                                        'type': type(const_value).__name__,
                                        'source': f'slice_{const_type}',
                                        'lineno': lineno,
                                        'function': self._current_function,
                                        'subscript_info': f'{ast.unparse(subscript_node.value)}[{ast.unparse(subscript_node.slice)}]'
                                    })
                    
                    # 处理扩展切片(如 a[1:5, 2])
                    elif isinstance(subscript_node.slice, ast.Tuple):
                        for idx, elt in enumerate(subscript_node.slice.elts):
                            if isinstance(elt, ast.Constant):
                                const_value = elt.value
                                for target in targets:
                                    if isinstance(target, ast.Name):
                                        self.constants[target.id].append({
                                            'value': const_value,
                                            'type': type(const_value).__name__,
                                            'source': f'multi_dim_index_{idx}',
                                            'lineno': lineno,
                                            'function': self._current_function,
                                            'subscript_info': f'{ast.unparse(subscript_node.value)}[{ast.unparse(subscript_node.slice)}]'
                                        })
                        
                def visit_Compare(self, node):
                    """处理比较表达式,提取比较值(增强版)"""
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
                    
                    # 提取左侧常量与右侧变量的比较(如 10 < x)
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
                    
                    # 处理函数调用与常量的比较(如 len(x) == 0)
                    elif isinstance(node.left, ast.Call):
                        self._extract_call_comparison(node.left, node.comparators, node.ops, node.lineno)
                    
                    # 处理二元运算与常量的比较(如 x % 2 == 1, x + y == 10)
                    elif isinstance(node.left, ast.BinOp):
                        self._extract_binop_comparison(node.left, node.comparators, node.ops, node.lineno)
                    
                    # 处理下标访问与常量的比较(如 a[0] == 5)
                    elif isinstance(node.left, ast.Subscript):
                        self._extract_subscript_comparison(node.left, node.comparators, node.ops, node.lineno)
                    
                    self.generic_visit(node)
                
                def _extract_call_comparison(self, call_node, comparators, ops, lineno):
                    """提取函数调用与常量的比较(如 len(x) == 0)"""
                    if isinstance(call_node.func, ast.Name):
                        func_name = call_node.func.id
                        
                        for i, comparator in enumerate(comparators):
                            if isinstance(comparator, ast.Constant):
                                const_value = comparator.value
                                op_name = type(ops[i]).__name__ if i < len(ops) else None
                                
                                # 特殊处理 len() 函数
                                if func_name == 'len':
                                    for arg in call_node.args:
                                        if isinstance(arg, ast.Name):
                                            var_name = arg.id
                                            self.constants[var_name].append({
                                                'value': const_value,
                                                'type': type(const_value).__name__,
                                                'source': f'len_comparison',
                                                'lineno': lineno,
                                                'function': self._current_function,
                                                'op': op_name,
                                                'call_info': f'len({var_name}) {op_name} {const_value}',
                                                'len_comparison': True
                                            })
                                
                                # 处理其他函数调用
                                else:
                                    for arg in call_node.args:
                                        if isinstance(arg, ast.Name):
                                            var_name = arg.id
                                            self.constants[var_name].append({
                                                'value': const_value,
                                                'type': type(const_value).__name__,
                                                'source': f'call_comparison_{func_name}',
                                                'lineno': lineno,
                                                'function': self._current_function,
                                                'op': op_name,
                                                'call_info': f'{func_name}({var_name}) {op_name} {const_value}'
                                            })
                
                def _extract_binop_comparison(self, binop_node, comparators, ops, lineno):
                    """提取二元运算与常量的比较(如 x % 2 == 1, x + y == 10)"""
                    for i, comparator in enumerate(comparators):
                        if isinstance(comparator, ast.Constant):
                            const_value = comparator.value
                            op_name = type(ops[i]).__name__ if i < len(ops) else None
                            
                            # 提取二元运算中的变量
                            self._extract_binop_vars_for_comparison(binop_node, const_value, op_name, lineno)
                
                def _extract_binop_vars_for_comparison(self, binop_node, const_value, op_name, lineno):
                    """从二元运算中提取变量用于比较"""
                    # 检查左侧是否为变量
                    if isinstance(binop_node.left, ast.Name):
                        var_name = binop_node.left.id
                        self.constants[var_name].append({
                            'value': const_value,
                            'type': type(const_value).__name__,
                            'source': 'binop_comparison',
                            'lineno': lineno,
                            'function': self._current_function,
                            'op': op_name,
                            'binop_info': f'{var_name} {type(binop_node.op).__name__} {ast.unparse(binop_node.right)} {op_name} {const_value}'
                        })
                    
                    # 检查右侧是否为变量
                    if isinstance(binop_node.right, ast.Name):
                        var_name = binop_node.right.id
                        self.constants[var_name].append({
                            'value': const_value,
                            'type': type(const_value).__name__,
                            'source': 'binop_comparison',
                            'lineno': lineno,
                            'function': self._current_function,
                            'op': op_name,
                            'binop_info': f'{ast.unparse(binop_node.left)} {type(binop_node.op).__name__} {var_name} {op_name} {const_value}'
                        })
                    
                    # 特殊处理取模运算(x % 2 == 1)
                    if isinstance(binop_node.op, ast.Mod):
                        # 提取模数
                        if isinstance(binop_node.right, ast.Constant):
                            mod_value = binop_node.right.value
                            if isinstance(binop_node.left, ast.Name):
                                var_name = binop_node.left.id
                                self.constants[var_name].append({
                                    'value': mod_value,
                                    'type': type(mod_value).__name__,
                                    'source': 'modulo_value',
                                    'lineno': lineno,
                                    'function': self._current_function,
                                    'op': 'Mod',
                                    'modulo_info': f'{var_name} % {mod_value} == {const_value}'
                                })
                    
                    # 递归处理嵌套的二元运算
                    if isinstance(binop_node.left, ast.BinOp):
                        self._extract_binop_vars_for_comparison(binop_node.left, const_value, op_name, lineno)
                    if isinstance(binop_node.right, ast.BinOp):
                        self._extract_binop_vars_for_comparison(binop_node.right, const_value, op_name, lineno)
                
                def _extract_subscript_comparison(self, subscript_node, comparators, ops, lineno):
                    """提取下标访问与常量的比较(如 a[0] == 5)"""
                    for i, comparator in enumerate(comparators):
                        if isinstance(comparator, ast.Constant):
                            const_value = comparator.value
                            op_name = type(ops[i]).__name__ if i < len(ops) else None
                            
                            # 提取下标访问中的变量
                            if isinstance(subscript_node.value, ast.Name):
                                var_name = subscript_node.value.id
                                self.constants[var_name].append({
                                    'value': const_value,
                                    'type': type(const_value).__name__,
                                    'source': 'subscript_comparison',
                                    'lineno': lineno,
                                    'function': self._current_function,
                                    'op': op_name,
                                    'subscript_info': f'{var_name}[{ast.unparse(subscript_node.slice)}] {op_name} {const_value}'
                                })
                            
                            # 提取下标索引中的常量(如果存在)
                            if isinstance(subscript_node.slice, ast.Constant):
                                index_value = subscript_node.slice.value
                                self.constants[f'{var_name}[{index_value}]'].append({
                                    'value': const_value,
                                    'type': type(const_value).__name__,
                                    'source': 'subscript_element_comparison',
                                    'lineno': lineno,
                                    'function': self._current_function,
                                    'op': op_name,
                                    'full_info': f'{var_name}[{index_value}] {op_name} {const_value}'
                                })
                
                def visit_AugAssign(self, node):
                    """处理增强赋值(如 x += 1)"""
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
                
                def visit_Subscript(self, node):
                    """单独处理下标表达式"""
                    # 为了捕获类似 a[0] 这样的直接使用
                    if isinstance(node.slice, ast.Constant) and isinstance(node.value, ast.Name):
                        var_name = node.value.id
                        index_value = node.slice.value
                        key = f'{var_name}[{index_value}]'
                        
                        self.constants[key].append({
                            'value': index_value,
                            'type': type(index_value).__name__,
                            'source': 'direct_subscript',
                            'lineno': node.lineno,
                            'function': self._current_function,
                            'full_expr': f'{var_name}[{index_value}]'
                        })
                    
                    self.generic_visit(node)
            
            extractor = ConstantExtractor(self._constants)
            extractor.visit(self.ast_tree)
            
            # 去重并简化，只保留value和type
            simplified_constants = {}
            for var in self._constants:
                unique_values = []
                seen = set()
                for item in self._constants[var]:
                    key = (item['value'], item['type'])
                    if key not in seen:
                        seen.add(key)
                        unique_values.append({
                            'value': item['value'],
                            'type': item['type']
                        })
                simplified_constants[var] = unique_values
            
            return simplified_constants
            
        except Exception as e:
            return {}
    
    def build_control_flow_graph(self) -> Dict[str, Any]:
        """
        构建控制流图(CFG) 
        
        Returns:
            {
                'nodes': list, 节点列表, 每个节点包含节点ID和属性(type, statements, lineno等)
                'edges': list, 边列表, 每个边包含起点,终点和属性(label, type等)
                'graph_metrics': dict, 图的度量信息, 包含num_nodes(节点数), num_edges(边数), num_functions(函数数), is_directed(是否有向图)
                'functions': dict, 函数名到(入口块, 出口块)的映射
                'call_edges': list, 函数调用边列表, 记录调用关系
            }
            失败时返回空字典 {}
        """
        try:
            self._cfg = nx.DiGraph()
            
            class CFGBuilder(ast.NodeVisitor):
                def __init__(self, cfg_graph):
                    self.cfg = cfg_graph
                    self.node_counter = 0
                    self.current_block = None
                    self.block_stack = []  # 用于嵌套结构
                    self.loop_stack = []   # 专门用于循环的栈:(header, exit)
                    self.function_blocks = {}  # 函数名 -> (entry, exit)
                    self.call_edges = []       # 函数调用边
                    
                def _new_block(self, label="block", **attrs):
                    """创建新的基本块"""
                    block_id = f"{label}_{self.node_counter}"
                    self.node_counter += 1
                    attrs_default = {
                        'type': label,
                        'statements': [],
                        'lineno': None,
                        'col_offset': None
                    }
                    attrs_default.update(attrs)
                    self.cfg.add_node(block_id, **attrs_default)
                    return block_id
                
                def _add_statement(self, node, stmt_type):
                    """添加语句到当前基本块"""
                    if self.current_block:
                        stmt_info = {
                            'type': stmt_type,
                            'lineno': node.lineno,
                            'col_offset': node.col_offset,
                            'code': ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                            'node': node
                        }
                        self.cfg.nodes[self.current_block]['statements'].append(stmt_info)
                        
                        # 记录行号信息
                        if self.cfg.nodes[self.current_block]['lineno'] is None:
                            self.cfg.nodes[self.current_block]['lineno'] = node.lineno
                
                def _connect_blocks(self, from_block, to_block, label=None):
                    """连接两个基本块"""
                    if from_block and to_block:
                        edge_attrs = {}
                        if label:
                            edge_attrs['label'] = label
                        self.cfg.add_edge(from_block, to_block, **edge_attrs)
                
                def visit_FunctionDef(self, node):
                    """处理函数定义"""
                    # 创建函数入口和出口块
                    entry_block = self._new_block(
                        label="func_entry",
                        function=node.name,
                        args=[arg.arg for arg in node.args.args],
                        has_return=False
                    )
                    exit_block = self._new_block(
                        label="func_exit",
                        function=node.name,
                        is_exit=True
                    )
                    
                    # 记录函数块映射
                    self.function_blocks[node.name] = (entry_block, exit_block)
                    
                    # 保存旧的当前块,进入函数体
                    old_block = self.current_block
                    self.current_block = entry_block
                    
                    # 推入函数上下文
                    self.block_stack.append(('function', exit_block))
                    
                    # 处理函数体
                    for stmt in node.body:
                        self.visit(stmt)
                    
                    # 如果函数体末尾没有显式return,连接到退出块
                    if self.current_block and self.current_block != exit_block:
                        # 检查最后一个块是否有return语句
                        last_statements = self.cfg.nodes[self.current_block].get('statements', [])
                        has_return = any(stmt['type'] == 'return' for stmt in last_statements)
                        if not has_return:
                            self._connect_blocks(self.current_block, exit_block, label="implicit_return")
                    
                    # 弹出函数上下文
                    self.block_stack.pop()
                    
                    # 恢复旧的当前块
                    self.current_block = old_block
                    
                    # 不返回任何值,避免在 generic_visit 中被重复处理
                
                def visit_If(self, node):
                    """处理if语句"""
                    # 为if条件创建基本块
                    condition_block = self._new_block("if_condition")
                    true_block = self._new_block("if_true")
                    false_block = self._new_block("if_false")
                    merge_block = self._new_block("if_merge")
                    
                    # 记录条件信息
                    self.cfg.nodes[condition_block]['condition'] = {
                        'expression': ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test),
                        'lineno': node.lineno
                    }
                    
                    # 连接当前块到条件块
                    self._connect_blocks(self.current_block, condition_block)
                    
                    # 保存当前块
                    old_block = self.current_block
                    
                    # 处理true分支
                    self.current_block = true_block
                    self.block_stack.append(('if', true_block, false_block, merge_block))
                    for stmt in node.body:
                        self.visit(stmt)
                    
                    # true分支处理完后,如果没有break/continue/return,连接到合并块
                    if self.current_block and self._can_fall_through():
                        self._connect_blocks(self.current_block, merge_block, label="true_end")
                    
                    # 处理false分支(else或elif)
                    self.current_block = false_block
                    self.block_stack[-1] = ('if', true_block, false_block, merge_block)  # 更新栈
                    for stmt in node.orelse:
                        self.visit(stmt)
                    
                    # false分支处理完后,连接到合并块
                    if self.current_block and self._can_fall_through():
                        self._connect_blocks(self.current_block, merge_block, label="false_end")
                    
                    # 恢复并设置当前块为合并块
                    self.block_stack.pop()
                    self.current_block = merge_block
                    
                    # 连接条件块到两个分支
                    self._connect_blocks(condition_block, true_block, label="true")
                    self._connect_blocks(condition_block, false_block, label="false")
                
                def visit_While(self, node):
                    """处理while循环 """
                    # 创建循环块
                    loop_header = self._new_block("while_header")
                    loop_body_start = self._new_block("while_body_start")
                    loop_exit = self._new_block("while_exit")
                    
                    # 如果有else子句
                    if node.orelse:
                        loop_else = self._new_block("while_else")
                        loop_exit_after_else = self._new_block("while_exit_after_else")
                    else:
                        loop_else = None
                        loop_exit_after_else = loop_exit
                    
                    # 记录循环信息
                    loop_info = {
                        'type': 'while',
                        'condition': ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test),
                        'lineno': node.lineno,
                        'has_else': bool(node.orelse)
                    }
                    self.cfg.nodes[loop_header]['loop'] = loop_info
                    
                    # 连接当前块到循环头
                    self._connect_blocks(self.current_block, loop_header)
                    
                    # 连接循环头到循环体(条件为真)
                    self._connect_blocks(loop_header, loop_body_start, label="true")
                    
                    # 连接循环头到退出或else(条件为假)
                    if loop_else:
                        self._connect_blocks(loop_header, loop_else, label="false")
                    else:
                        self._connect_blocks(loop_header, loop_exit, label="false")
                    
                    # 保存旧的当前块
                    old_block = self.current_block
                    
                    # 处理循环体
                    self.current_block = loop_body_start
                    self.loop_stack.append((loop_header, loop_exit_after_else))
                    self.block_stack.append(('loop', loop_header, loop_exit_after_else))
                    
                    for stmt in node.body:
                        self.visit(stmt)
                    
                    # 循环体结束后,如果没有break/return,连接到循环头(继续迭代)
                    if self.current_block and self._can_fall_through():
                        self._connect_blocks(self.current_block, loop_header, label="continue")
                    
                    # 处理else子句
                    if loop_else:
                        # 连接else子句到最终退出块
                        self._connect_blocks(loop_else, loop_exit_after_else)
                        
                        # 处理else子句中的语句
                        self.current_block = loop_else
                        for stmt in node.orelse:
                            self.visit(stmt)
                        
                        # else子句结束后连接到退出块
                        if self.current_block and self._can_fall_through():
                            self._connect_blocks(self.current_block, loop_exit_after_else)
                    
                    # 弹出循环栈
                    self.loop_stack.pop()
                    self.block_stack.pop()
                    
                    # 恢复并设置当前块为退出块
                    self.current_block = loop_exit_after_else
                
                def visit_For(self, node):
                    """处理for循环 """
                    # 创建循环块
                    loop_header = self._new_block("for_header")
                    loop_body_start = self._new_block("for_body_start")
                    loop_exit = self._new_block("for_exit")
                    
                    # 如果有else子句
                    if node.orelse:
                        loop_else = self._new_block("for_else")
                        loop_exit_after_else = self._new_block("for_exit_after_else")
                    else:
                        loop_else = None
                        loop_exit_after_else = loop_exit
                    
                    # 记录循环信息
                    loop_info = {
                        'type': 'for',
                        'target': ast.unparse(node.target) if hasattr(ast, 'unparse') else str(node.target),
                        'iterable': ast.unparse(node.iter) if hasattr(ast, 'unparse') else str(node.iter),
                        'lineno': node.lineno,
                        'has_else': bool(node.orelse)
                    }
                    self.cfg.nodes[loop_header]['loop'] = loop_info
                    
                    # 连接当前块到循环头
                    self._connect_blocks(self.current_block, loop_header)
                    
                    # 连接循环头到循环体(有下一个元素)
                    self._connect_blocks(loop_header, loop_body_start, label="iterate")
                    
                    # 连接循环头到退出或else(迭代结束)
                    if loop_else:
                        self._connect_blocks(loop_header, loop_else, label="exhausted")
                    else:
                        self._connect_blocks(loop_header, loop_exit, label="exhausted")
                    
                    # 保存旧的当前块
                    old_block = self.current_block
                    
                    # 处理循环体
                    self.current_block = loop_body_start
                    self.loop_stack.append((loop_header, loop_exit_after_else))
                    self.block_stack.append(('loop', loop_header, loop_exit_after_else))
                    
                    for stmt in node.body:
                        self.visit(stmt)
                    
                    # 循环体结束后,如果没有break/return,连接到循环头(继续迭代)
                    if self.current_block and self._can_fall_through():
                        self._connect_blocks(self.current_block, loop_header, label="continue")
                    
                    # 处理else子句
                    if loop_else:
                        # 连接else子句到最终退出块
                        self._connect_blocks(loop_else, loop_exit_after_else)
                        
                        # 处理else子句中的语句
                        self.current_block = loop_else
                        for stmt in node.orelse:
                            self.visit(stmt)
                        
                        # else子句结束后连接到退出块
                        if self.current_block and self._can_fall_through():
                            self._connect_blocks(self.current_block, loop_exit_after_else)
                    
                    # 弹出循环栈
                    self.loop_stack.pop()
                    self.block_stack.pop()
                    
                    # 恢复并设置当前块为退出块
                    self.current_block = loop_exit_after_else
                
                def visit_Return(self, node):
                    """处理return语句"""
                    self._add_statement(node, 'return')
                    
                    # 找到函数的退出块
                    exit_block = None
                    for block_type, *block_info in reversed(self.block_stack):
                        if block_type == 'function':
                            exit_block = block_info[0]  # 函数退出块
                            break
                    
                    if exit_block and self.current_block:
                        # 连接到函数退出块
                        self._connect_blocks(self.current_block, exit_block, label="return")
                        
                        # 标记当前块有return语句
                        self.cfg.nodes[self.current_block]['has_return'] = True
                        
                        # return语句终止当前执行流
                        self.current_block = None
                    else:
                        # 如果没有找到函数上下文,创建临时的退出块
                        return_block = self._new_block("return")
                        self._connect_blocks(self.current_block, return_block, label="return")
                        self.current_block = None
                
                def visit_Break(self, node):
                    """处理break语句"""
                    self._add_statement(node, 'break')
                    
                    # 找到最近的循环退出块
                    exit_block = None
                    for block_type, header, exit in reversed(self.block_stack):
                        if block_type == 'loop':
                            exit_block = exit
                            break
                    
                    if exit_block and self.current_block:
                        self._connect_blocks(self.current_block, exit_block, label="break")
                        
                        # break语句终止当前执行流
                        self.current_block = None
                
                def visit_Continue(self, node):
                    """处理continue语句"""
                    self._add_statement(node, 'continue')
                    
                    # 找到最近的循环头
                    loop_header = None
                    for block_type, header, exit in reversed(self.block_stack):
                        if block_type == 'loop':
                            loop_header = header
                            break
                    
                    if loop_header and self.current_block:
                        self._connect_blocks(self.current_block, loop_header, label="continue")
                        
                        # continue语句终止当前执行流
                        self.current_block = None
                
                def visit_Try(self, node):
                    """处理try/except/finally语句"""
                    # 创建try块
                    try_block = self._new_block("try_block")
                    try_exit = self._new_block("try_exit")
                    
                    # 连接当前块到try块
                    self._connect_blocks(self.current_block, try_block)
                    
                    # 保存旧的当前块
                    old_block = self.current_block
                    
                    # 处理try块
                    self.current_block = try_block
                    self.block_stack.append(('try', try_exit))
                    
                    for stmt in node.body:
                        self.visit(stmt)
                    
                    # try块结束后,如果没有异常,连接到退出块
                    if self.current_block and self._can_fall_through():
                        self._connect_blocks(self.current_block, try_exit, label="no_exception")
                    
                    # 处理except块
                    for handler in node.handlers:
                        except_block = self._new_block("except_block")
                        # 连接try块到except块(异常发生)
                        self._connect_blocks(try_block, except_block, label="exception")
                        
                        self.current_block = except_block
                        for stmt in handler.body:
                            self.visit(stmt)
                        
                        # except块结束后连接到退出块
                        if self.current_block and self._can_fall_through():
                            self._connect_blocks(self.current_block, try_exit, label="handled")
                    
                    # 处理finally块
                    if node.finalbody:
                        finally_block = self._new_block("finally_block")
                        
                        # 从try块和所有except块连接到finally块
                        for pred in self.cfg.predecessors(try_exit):
                            self._connect_blocks(pred, finally_block, label="finally")
                        
                        # 处理finally块
                        self.current_block = finally_block
                        for stmt in node.finalbody:
                            self.visit(stmt)
                        
                        # finally块结束后连接到最终退出块
                        finally_exit = self._new_block("finally_exit")
                        if self.current_block and self._can_fall_through():
                            self._connect_blocks(self.current_block, finally_exit)
                        
                        self.current_block = finally_exit
                    
                    # 弹出try上下文
                    self.block_stack.pop()
                    
                    # 恢复当前块
                    self.current_block = old_block
                
                def visit_Call(self, node):
                    """处理函数调用,建立调用关系"""
                    self._add_statement(node, 'call')
                    
                    # 提取函数名
                    func_name = None
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    
                    if func_name and self.current_block:
                        # 记录调用信息
                        call_info = {
                            'function': func_name,
                            'lineno': node.lineno,
                            'args': [ast.unparse(arg) for arg in node.args] if hasattr(ast, 'unparse') else [],
                            'keywords': [ast.unparse(kw) for kw in node.keywords] if hasattr(ast, 'unparse') else []
                        }
                        
                        # 添加到当前块的调用列表
                        if 'calls' not in self.cfg.nodes[self.current_block]:
                            self.cfg.nodes[self.current_block]['calls'] = []
                        self.cfg.nodes[self.current_block]['calls'].append(call_info)
                        
                        # 如果被调用的函数也在同一代码中,建立调用边
                        if func_name in self.function_blocks:
                            callee_entry, callee_exit = self.function_blocks[func_name]
                            self.call_edges.append({
                                'caller': self.current_block,
                                'callee': callee_entry,
                                'function_name': func_name,  # 添加函数名
                                'lineno': node.lineno,
                                'return_to': self.current_block  # 返回到调用点
                            })
                    
                    self.generic_visit(node)
                
                def visit_Expr(self, node):
                    """处理表达式语句"""
                    self._add_statement(node, 'expr')
                    self.generic_visit(node)
                
                def visit_Assign(self, node):
                    """处理赋值语句"""
                    self._add_statement(node, 'assign')
                    self.generic_visit(node)
                
                def visit_AugAssign(self, node):
                    """处理增强赋值"""
                    self._add_statement(node, 'augassign')
                    self.generic_visit(node)
                
                def visit_Assert(self, node):
                    """处理断言语句"""
                    self._add_statement(node, 'assert')
                    self.generic_visit(node)
                
                def visit_Raise(self, node):
                    """处理raise语句"""
                    self._add_statement(node, 'raise')
                    
                    # raise语句通常终止当前执行流
                    self.current_block = None
                
                def _can_fall_through(self):
                    """检查当前块是否可以fall through(没有终止语句)"""
                    if not self.current_block:
                        return False
                    
                    statements = self.cfg.nodes[self.current_block].get('statements', [])
                    if not statements:
                        return True
                    
                    last_stmt = statements[-1]
                    # 以下语句会终止执行流
                    terminating_types = {'return', 'break', 'continue', 'raise'}
                    return last_stmt['type'] not in terminating_types
                
                def generic_visit(self, node):
                    """默认的节点访问方法"""
                    # 跳过不需要特殊处理的节点
                    super().generic_visit(node)
            
            # 创建构建器并遍历AST
            builder = CFGBuilder(self._cfg)
            
            # 创建主程序入口块
            main_entry = builder._new_block("main_entry", is_main=True)
            builder.current_block = main_entry
            
            # 只遍历一次,处理所有语句
            for node in self.ast_tree.body:
                builder.visit(node)
            
            # 添加函数调用边到图中
            for call_edge in builder.call_edges:
                caller_block = call_edge['caller']
                callee_entry = call_edge['callee']
                return_to_block = call_edge['return_to']
                
                # 添加调用边:从调用点到被调用函数入口
                self._cfg.add_edge(
                    caller_block,
                    callee_entry,
                    label='call',
                    type='call_edge',
                    lineno=call_edge['lineno']
                )
                
                # 添加返回边:从被调用函数退出块返回到调用点
                # 从 call_edge 中获取函数名
                func_name = call_edge.get('function_name')
                if func_name and func_name in builder.function_blocks:
                    callee_entry_block, callee_exit_block = builder.function_blocks[func_name]
                    self._cfg.add_edge(
                        callee_exit_block,
                        return_to_block,
                        label='return',
                        type='return_edge',
                        lineno=call_edge['lineno']
                    )
            
            # 生成可序列化的CFG信息
            cfg_info = {
                'nodes': list(self._cfg.nodes(data=True)),
                'edges': list(self._cfg.edges(data=True)),
                'graph_metrics': {
                    'num_nodes': self._cfg.number_of_nodes(),
                    'num_edges': self._cfg.number_of_edges(),
                    'is_directed': self._cfg.is_directed(),
                    'functions': list(builder.function_blocks.keys()),
                    'num_functions': len(builder.function_blocks),
                    'num_call_edges': len(builder.call_edges)
                },
                'functions': builder.function_blocks,
                'call_edges': builder.call_edges
            }
            
            return cfg_info
            
        except Exception as e:
            return {}
    
    def predicate_mining(self) -> Dict[str, Any]:
        """
        谓词挖掘
        - 识别边界条件和特殊值, 通过谓词组合覆盖, 而不是路径覆盖, 减少测试数量
        
        Returns:
            list, 谓词列表, 每个谓词包含:
                - expression: 表达式
                - var: 变量名(如果有)
                - value: 比较值(如果有)
                - boundary_values: 边界测试建议值(如果有)
            例如: [{'expression': 'n > 10', 'var': 'n', 'value': 10, 'boundary_values': [9, 10, 11]}, ...]
            失败时返回空列表 []
        """
        try:
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
                    try:
                        # 获取表达式字符串
                        expression = ast.unparse(test_node) if hasattr(ast, 'unparse') else str(test_node)
                        
                        predicate_info = {
                            'type': pred_type,
                            'expression': expression,
                            'lineno': lineno,
                            'col_offset': test_node.col_offset,
                            'function': self._current_function,
                            'node_type': type(test_node).__name__
                        }
                        
                        # 如果是比较表达式,提取详细信息
                        if isinstance(test_node, ast.Compare):
                            self._extract_comparison_details(test_node, predicate_info)
                        
                        # 如果是逻辑表达式
                        elif isinstance(test_node, ast.BoolOp):
                            predicate_info['bool_op'] = type(test_node.op).__name__
                        
                        self.predicates.append(predicate_info)
                        
                    except Exception as e:
                        # 如果提取失败,至少记录基本信息
                        self.predicates.append({
                            'type': pred_type,
                            'expression': '提取失败',
                            'lineno': lineno,
                            'function': self._current_function,
                            'error': str(e)
                        })
                
                def _extract_comparison_details(self, compare_node, predicate_info):
                    """提取比较表达式的详细信息"""
                    try:
                        # 链式比较
                        if len(compare_node.comparators) > 1:
                            predicate_info['is_chained'] = True
                            predicate_info['chain_length'] = len(compare_node.comparators)
                            predicate_info['ops'] = [type(op).__name__ for op in compare_node.ops]
                            
                            # 分析边界
                            boundaries = self._analyze_chained_boundaries(compare_node)
                            if boundaries:
                                predicate_info['boundary_info'] = boundaries
                        else:
                            # 单个比较
                            predicate_info['is_chained'] = False
                            predicate_info['op'] = type(compare_node.ops[0]).__name__
                            
                            # 尝试提取变量和值
                            self._extract_single_comparison(compare_node, predicate_info)

                            # 确保边界信息被提取
                            if 'boundary_info' not in predicate_info:
                                # 尝试从单个比较中提取边界
                                left_expr = compare_node.left
                                right_expr = compare_node.comparators[0]
                                op = compare_node.ops[0]
                                boundary = self._analyze_single_boundary(left_expr, op, right_expr)
                                if boundary:
                                    predicate_info['boundary_info'] = boundary
                            
                    except Exception:
                        pass
                
                def _extract_single_comparison(self, compare_node, predicate_info):
                    """提取单个比较的详细信息"""
                    try:
                        left_expr = compare_node.left
                        right_expr = compare_node.comparators[0]
                        op = compare_node.ops[0]
                        
                        # 尝试提取变量名
                        left_var = self._extract_variable_name(left_expr)
                        right_var = self._extract_variable_name(right_expr)
                        
                        # 尝试提取常量值
                        left_const = self._extract_constant_value(left_expr)
                        right_const = self._extract_constant_value(right_expr)
                        
                        # 确定变量和值的组合
                        if left_var and right_const is not None:
                            predicate_info['var'] = left_var
                            predicate_info['value'] = right_const
                            predicate_info['comparison_type'] = 'variable_op_constant'
                        elif left_const is not None and right_var:
                            predicate_info['var'] = right_var
                            predicate_info['value'] = left_const
                            predicate_info['comparison_type'] = 'constant_op_variable'
                        elif left_var and right_var:
                            predicate_info['var_left'] = left_var
                            predicate_info['var_right'] = right_var
                            predicate_info['comparison_type'] = 'variable_op_variable'
                        
                        # 特殊处理取模运算
                        if isinstance(left_expr, ast.BinOp) and isinstance(left_expr.op, ast.Mod):
                            if isinstance(left_expr.left, ast.Name) and isinstance(left_expr.right, ast.Constant):
                                var_name = left_expr.left.id
                                mod_value = left_expr.right.value
                                if isinstance(op, ast.Eq) and isinstance(right_expr, ast.Constant):
                                    predicate_info['var'] = var_name
                                    predicate_info['mod'] = mod_value
                                    predicate_info['rem'] = right_expr.value
                                    predicate_info['comparison_type'] = 'modulo_equality'
                        
                        # 提取边界信息
                        boundary = self._analyze_single_boundary(left_expr, op, right_expr)
                        if boundary:
                            predicate_info['boundary_info'] = boundary
                            
                    except Exception:
                        pass
                
                def _analyze_chained_boundaries(self, compare_node):
                    """分析链式比较的边界"""
                    try:
                        boundaries = {
                            'lower': None,
                            'upper': None,
                            'lower_inclusive': False,
                            'upper_inclusive': False,
                            'values': []
                        }
                        
                        left_expr = compare_node.left
                        
                        for i, (op, right_expr) in enumerate(zip(compare_node.ops, compare_node.comparators)):
                            # 提取常量值
                            left_const = self._extract_constant_value(left_expr)
                            right_const = self._extract_constant_value(right_expr)
                            
                            if left_const is not None or right_const is not None:
                                # 分析边界条件
                                if isinstance(op, ast.Lt):  # a < b
                                    if left_const is not None:  # const < var
                                        boundaries['lower'] = left_const
                                        boundaries['lower_inclusive'] = False
                                    elif right_const is not None:  # var < const
                                        boundaries['upper'] = right_const
                                        boundaries['upper_inclusive'] = False
                                        
                                elif isinstance(op, ast.LtE):  # a <= b
                                    if left_const is not None:  # const <= var
                                        boundaries['lower'] = left_const
                                        boundaries['lower_inclusive'] = True
                                    elif right_const is not None:  # var <= const
                                        boundaries['upper'] = right_const
                                        boundaries['upper_inclusive'] = True
                                        
                                elif isinstance(op, ast.Gt):  # a > b
                                    if left_const is not None:  # const > var
                                        boundaries['upper'] = left_const
                                        boundaries['upper_inclusive'] = False
                                    elif right_const is not None:  # var > const
                                        boundaries['lower'] = right_const
                                        boundaries['lower_inclusive'] = False
                                        
                                elif isinstance(op, ast.GtE):  # a >= b
                                    if left_const is not None:  # const >= var
                                        boundaries['upper'] = left_const
                                        boundaries['upper_inclusive'] = True
                                    elif right_const is not None:  # var >= const
                                        boundaries['lower'] = right_const
                                        boundaries['lower_inclusive'] = True
                            
                            left_expr = right_expr
                        
                        # 生成建议测试值
                        if boundaries['lower'] is not None or boundaries['upper'] is not None:
                            boundaries['suggested_values'] = self._generate_boundary_test_values(
                                boundaries['lower'], boundaries['upper'],
                                boundaries['lower_inclusive'], boundaries['upper_inclusive']
                            )
                        
                        return boundaries
                        
                    except Exception:
                        return None
                
                def _analyze_single_boundary(self, left_expr, op, right_expr):
                    """分析单个比较的边界"""
                    try:
                        boundary = {}
                        
                        # 检查是否为边界比较
                        if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                            # 获取值
                            left_value = self._extract_constant_value(left_expr)
                            right_value = self._extract_constant_value(right_expr)
                            
                            if isinstance(op, ast.Lt):  # a < b
                                boundary['condition'] = 'less_than'
                                if right_value is not None:
                                    boundary['value'] = right_value
                                    boundary['suggested_values'] = self._generate_single_boundary_values(
                                        right_value, 'less_than'
                                    )
                                    
                            elif isinstance(op, ast.LtE):  # a <= b
                                boundary['condition'] = 'less_than_or_equal'
                                if right_value is not None:
                                    boundary['value'] = right_value
                                    boundary['suggested_values'] = self._generate_single_boundary_values(
                                        right_value, 'less_than_or_equal'
                                    )
                                    
                            elif isinstance(op, ast.Gt):  # a > b
                                boundary['condition'] = 'greater_than'
                                if left_value is not None:
                                    boundary['value'] = left_value
                                    boundary['suggested_values'] = self._generate_single_boundary_values(
                                        left_value, 'greater_than'
                                    )
                                    
                            elif isinstance(op, ast.GtE):  # a >= b
                                boundary['condition'] = 'greater_than_or_equal'
                                if left_value is not None:
                                    boundary['value'] = left_value
                                    boundary['suggested_values'] = self._generate_single_boundary_values(
                                        left_value, 'greater_than_or_equal'
                                    )
                            
                            return boundary
                            
                    except Exception:
                        return None
                
                def _extract_variable_name(self, expr):
                    """从表达式中提取变量名"""
                    if isinstance(expr, ast.Name):
                        return expr.id
                    elif isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
                        return expr.value.id
                    elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == 'len':
                        if expr.args and isinstance(expr.args[0], ast.Name):
                            return expr.args[0].id
                    return None
                
                def _extract_constant_value(self, expr):
                    """从表达式中提取常量值"""
                    if isinstance(expr, ast.Constant):
                        return expr.value
                    elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.USub, ast.UAdd)):
                        if isinstance(expr.operand, ast.Constant):
                            value = expr.operand.value
                            if isinstance(expr.op, ast.USub):
                                return -value
                            else:
                                return value
                    return None
                
                def _generate_boundary_test_values(self, lower, upper, lower_inc, upper_inc):
                    """为边界生成测试值"""
                    test_values = set()
                    
                    if lower is not None and isinstance(lower, (int, float)):
                        # 下边界相关
                        if lower_inc:
                            test_values.update([lower - 1, lower, lower + 1])
                        else:
                            test_values.update([lower - 1, lower - 0.5, lower, lower + 1])
                    
                    if upper is not None and isinstance(upper, (int, float)):
                        # 上边界相关
                        if upper_inc:
                            test_values.update([upper - 1, upper, upper + 1])
                        else:
                            test_values.update([upper - 1, upper, upper + 0.5, upper + 1])
                    
                    # 区间内值
                    if lower is not None and upper is not None:
                        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                            middle = (lower + upper) / 2
                            test_values.add(middle)
                            
                            lower_near = lower + (upper - lower) * 0.25
                            upper_near = lower + (upper - lower) * 0.75
                            test_values.update([lower_near, upper_near])
                    
                    return sorted(list(test_values))
                
                def _generate_single_boundary_values(self, value, condition):
                    """为单个边界生成测试值"""
                    if not isinstance(value, (int, float)):
                        return []
                    
                    if condition == 'less_than':
                        return [value - 1, value, value + 1]
                    elif condition == 'less_than_or_equal':
                        return [value - 1, value, value + 1]
                    elif condition == 'greater_than':
                        return [value - 1, value, value + 1]
                    elif condition == 'greater_than_or_equal':
                        return [value - 1, value, value + 1]
                    else:
                        return [value - 1, value, value + 1]
            
            visitor = PredicateVisitor(self._predicates)
            visitor.visit(self.ast_tree)
            
            # 统计信息
            stats = {
                'total_predicates': len(self._predicates),
                'predicates_by_type': defaultdict(int),
                'predicates_by_function': defaultdict(int),
                'variables_in_predicates': set(),
                'chained_comparisons': 0,
                'boundary_predicates': 0
            }
            
            for pred in self._predicates:
                stats['predicates_by_type'][pred.get('type', 'unknown')] += 1
                if 'function' in pred:
                    stats['predicates_by_function'][pred['function']] += 1
                
                # 收集变量
                if 'var' in pred:
                    stats['variables_in_predicates'].add(pred['var'])
                if 'var_left' in pred:
                    stats['variables_in_predicates'].add(pred['var_left'])
                if 'var_right' in pred:
                    stats['variables_in_predicates'].add(pred['var_right'])
                
                # 统计链式比较和边界谓词
                if pred.get('is_chained', False):
                    stats['chained_comparisons'] += 1
                if 'boundary_info' in pred:
                    stats['boundary_predicates'] += 1
            
            # 简化谓词信息，只保留必要字段
            simplified_predicates = []
            for pred in self._predicates:
                simplified = {
                    'expression': pred.get('expression', '')
                }
                
                # 添加变量（如果有）
                if 'var' in pred:
                    simplified['var'] = pred['var']
                
                # 添加比较值（如果有）
                if 'value' in pred:
                    simplified['value'] = pred['value']
                
                # 添加边界测试建议（如果有）
                if 'boundary_info' in pred and 'suggested_values' in pred['boundary_info']:
                    simplified['boundary_values'] = pred['boundary_info']['suggested_values']
                
                simplified_predicates.append(simplified)
            
            return simplified_predicates
            
        except Exception as e:
            return []
    
    def analyze_chained_comparisons(self) -> Dict[str, Any]:
        """
        专门分析链式比较, 提取完整的边界信息
        
        Returns:
            list, 链式比较列表, 每个包含:
                - expression: 表达式
                - main_variable: 主变量
                - boundaries: 边界信息(lower, upper等)
                - test_values: 测试值建议
            例如: [{'expression': '20 > n > 10', 'main_variable': 'n', 'boundaries': {...}, 'test_values': [9, 10, 11, ...]}, ...]
            失败时返回空列表 []
        """
        
        try:
            chained_predicates = []
            
            class ChainedComparisonVisitor(ast.NodeVisitor):
                def __init__(self, result):
                    self.result = result
                    self._current_function = None
                
                def visit_FunctionDef(self, node):
                    old_function = self._current_function
                    self._current_function = node.name
                    self.generic_visit(node)
                    self._current_function = old_function
                
                def visit_Compare(self, node):
                    if len(node.comparators) > 1:  # 链式比较
                        self._analyze_chained_comparison(node)
                    self.generic_visit(node)
                
                def _analyze_chained_comparison(self, node):
                    """Analyze chained comparison in detail"""
                    # 提取变量名和所有常量
                    variables = set()
                    constants = []
                    positions = []  # 每个常量的位置信息
                    
                    # 分析左侧
                    left_info = self._analyze_expression(node.left)
                    if left_info['is_variable']:
                        variables.add(left_info['name'])
                    elif left_info['is_constant']:
                        constants.append((left_info['value'], 'left'))
                        positions.append({
                            'value': left_info['value'],
                            'side': 'left',
                            'expr': ast.unparse(node.left) if hasattr(ast, 'unparse') else str(node.left)
                        })
                    
                    # 分析所有比较器和操作符
                    for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
                        comp_info = self._analyze_expression(comparator)
                        if comp_info['is_variable']:
                            variables.add(comp_info['name'])
                        elif comp_info['is_constant']:
                            side = f'comparator_{i}'
                            constants.append((comp_info['value'], side))
                            positions.append({
                                'value': comp_info['value'],
                                'side': side,
                                'position': i,
                                'op': type(op).__name__,
                                'expr': ast.unparse(comparator) if hasattr(ast, 'unparse') else str(comparator)
                            })
                    
                    # 如果只有一个主要变量,进行边界分析
                    if len(variables) == 1:
                        main_var = next(iter(variables))
                        boundaries = self._calculate_chained_boundaries(node, main_var)
                        
                        chained_info = {
                            'type': 'chained_comparison',
                            'expression': ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                            'main_variable': main_var,
                            'variables': list(variables),
                            'constants': constants,
                            'positions': positions,
                            'boundaries': boundaries,
                            'lineno': node.lineno,
                            'function': self._current_function,
                            'ops': [type(op).__name__ for op in node.ops]
                        }
                        
                        # 生成测试值建议
                        chained_info['test_values'] = self._generate_chained_test_values(boundaries)
                        
                        self.result.append(chained_info)
                
                def _analyze_expression(self, expr):
                    """分析表达式,提取信息"""
                    result = {
                        'is_variable': False,
                        'is_constant': False,
                        'name': None,
                        'value': None
                    }
                    
                    if isinstance(expr, ast.Name):
                        result['is_variable'] = True
                        result['name'] = expr.id
                    elif isinstance(expr, ast.Constant):
                        result['is_constant'] = True
                        result['value'] = expr.value
                    elif isinstance(expr, ast.Num):  # Python 3.7及之前
                        result['is_constant'] = True
                        result['value'] = expr.n
                    elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.USub, ast.UAdd)):
                        # 处理带符号的数字
                        operand = expr.operand
                        if isinstance(operand, (ast.Constant, ast.Num)):
                            value = operand.value if isinstance(operand, ast.Constant) else operand.n
                            if isinstance(expr.op, ast.USub):
                                result['is_constant'] = True
                                result['value'] = -value
                            else:
                                result['is_constant'] = True
                                result['value'] = value
                    
                    return result
                
                def _calculate_chained_boundaries(self, node, main_var):
                    """计算链式比较的边界"""
                    boundaries = {
                        'lower': None,  # 下界
                        'upper': None,  # 上界
                        'lower_inclusive': False,  # 下界是否包含
                        'upper_inclusive': False,  # 上界是否包含
                        'intervals': []  # 所有区间
                    }
                    
                    # 分析每个比较
                    left_expr = node.left
                    left_is_var = self._is_variable(left_expr, main_var)
                    
                    for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
                        right_is_var = self._is_variable(comparator, main_var)
                        
                        if left_is_var and not right_is_var:
                            # 变量在左边,常量在右边
                            const_value = self._extract_constant_value(comparator)
                            if const_value is not None:
                                if isinstance(op, ast.Lt):  # var < const
                                    boundaries['upper'] = const_value
                                    boundaries['upper_inclusive'] = False
                                elif isinstance(op, ast.LtE):  # var <= const
                                    boundaries['upper'] = const_value
                                    boundaries['upper_inclusive'] = True
                                elif isinstance(op, ast.Gt):  # var > const
                                    boundaries['lower'] = const_value
                                    boundaries['lower_inclusive'] = False
                                elif isinstance(op, ast.GtE):  # var >= const
                                    boundaries['lower'] = const_value
                                    boundaries['lower_inclusive'] = True
                        
                        elif not left_is_var and right_is_var:
                            # 常量在左边,变量在右边
                            const_value = self._extract_constant_value(left_expr)
                            if const_value is not None:
                                if isinstance(op, ast.Lt):  # const < var
                                    boundaries['lower'] = const_value
                                    boundaries['lower_inclusive'] = False
                                elif isinstance(op, ast.LtE):  # const <= var
                                    boundaries['lower'] = const_value
                                    boundaries['lower_inclusive'] = True
                                elif isinstance(op, ast.Gt):  # const > var
                                    boundaries['upper'] = const_value
                                    boundaries['upper_inclusive'] = False
                                elif isinstance(op, ast.GtE):  # const >= var
                                    boundaries['upper'] = const_value
                                    boundaries['upper_inclusive'] = True
                        
                        # 更新left_expr和left_is_var
                        left_expr = comparator
                        left_is_var = right_is_var
                    
                    return boundaries
                
                def _is_variable(self, expr, target_var):
                    """检查表达式是否是目标变量"""
                    if isinstance(expr, ast.Name):
                        return expr.id == target_var
                    return False
                
                def _extract_constant_value(self, expr):
                    """提取常量值"""
                    if isinstance(expr, ast.Constant):
                        return expr.value
                    elif isinstance(expr, ast.Num):
                        return expr.n
                    elif isinstance(expr, ast.UnaryOp):
                        return self._extract_constant_value(expr.operand)
                    return None
                
                def _generate_chained_test_values(self, boundaries):
                    """为链式比较生成测试值"""
                    test_values = set()
                    
                    lower = boundaries.get('lower')
                    upper = boundaries.get('upper')
                    lower_inc = boundaries.get('lower_inclusive', False)
                    upper_inc = boundaries.get('upper_inclusive', False)
                    
                    if lower is not None and isinstance(lower, (int, float)):
                        # 下边界测试值
                        if lower_inc:
                            test_values.update([lower - 1, lower, lower + 1])
                        else:
                            test_values.update([lower - 1, lower - 0.5, lower, lower + 1])
                    
                    if upper is not None and isinstance(upper, (int, float)):
                        # 上边界测试值
                        if upper_inc:
                            test_values.update([upper - 1, upper, upper + 1])
                        else:
                            test_values.update([upper - 1, upper, upper + 0.5, upper + 1])
                    
                    # 区间内测试值(如果上下界都存在)
                    if lower is not None and upper is not None:
                        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                            # 中间值
                            middle = (lower + upper) / 2
                            test_values.add(middle)
                            
                            # 靠近边界的值
                            lower_near = lower + (upper - lower) * 0.25
                            upper_near = lower + (upper - lower) * 0.75
                            test_values.update([lower_near, upper_near])
                    
                    # 转换为排序列表
                    return sorted(list(test_values))
            
            visitor = ChainedComparisonVisitor(chained_predicates)
            visitor.visit(self.ast_tree)
            
            # 统计信息
            stats = {
                'total_chained_comparisons': len(chained_predicates),
                'variables_in_chained': set(),
                'test_value_suggestions': 0
            }
            
            for comp in chained_predicates:
                if 'main_variable' in comp:
                    stats['variables_in_chained'].add(comp['main_variable'])
                if 'test_values' in comp:
                    stats['test_value_suggestions'] += len(comp['test_values'])
            
            stats['variables_in_chained'] = list(stats['variables_in_chained'])
            
            # 简化链式比较信息，只保留必要字段
            simplified_comparisons = []
            for comp in chained_predicates:
                simplified = {
                    'expression': comp.get('expression', ''),
                    'main_variable': comp.get('main_variable', ''),
                    'boundaries': comp.get('boundaries', {}),
                    'test_values': comp.get('test_values', [])
                }
                simplified_comparisons.append(simplified)
            
            return simplified_comparisons
            
        except Exception as e:
            return []
    
    def backward_slice(self, variable_name: str, line_number: int, 
                    include_all_defs: bool = False) -> Dict[str, Any]:
        """
        后向切片 - 增强版
        - 识别需要边界测试的变量
        - 支持多种定义类型
        - 返回需要边界测试的变量
        
        Args:
            variable_name: 目标变量名
            line_number: 目标行号
            include_all_defs: 是否包含所有定义(包括不在数据流路径上的)
            
        Returns:
            {
                'boundary_variables': list, 需要边界测试的变量列表, 每个包含variable(变量名), boundary_values(边界值), suggested_test_values(建议测试值)
                'data_flow_paths': list, 数据流路径列表, 记录变量的定义-使用关系
            }
            例如: {'boundary_variables': [{'variable': 'n', 'boundary_values': {...}, 'suggested_test_values': [9, 10, 11]}], 'data_flow_paths': [...]}
            失败时返回空字典 {}
        """
        try:
            # 构建def-use链
            def_use_chains = self._build_def_use_chains()
            
            if variable_name not in def_use_chains:
                return {}
            
            # 找到目标变量在指定行的使用
            target_use = None
            for use in def_use_chains[variable_name].get('uses', []):
                if use['lineno'] == line_number:
                    target_use = use
                    break
            
            # 执行后向切片
            affected_lines = set()
            affected_vars = set([variable_name])
            data_flow_paths = []
            work_list = [(variable_name, line_number, [])]  # (变量, 行号, 路径)
            visited = set()
            
            while work_list:
                var, line, path = work_list.pop()
                
                if (var, line) in visited:
                    continue
                visited.add((var, line))
                
                # 添加当前行到影响行集合
                affected_lines.add(line)
                
                # 记录当前路径
                current_path = path + [(var, line)]
                
                # 查找定义该变量的语句
                if var in def_use_chains:
                    for def_info in def_use_chains[var].get('definitions', []):
                        def_line = def_info['lineno']
                        
                        # 检查定义是否在数据流路径上
                        if def_line < line:  # 只考虑之前的定义
                            if include_all_defs or self._is_on_dataflow_path(def_info, var, line):
                                
                                # 添加定义行
                                affected_lines.add(def_line)
                                affected_vars.add(var)
                                
                                # 记录数据流路径
                                data_flow_paths.append({
                                    'from': (var, def_line),
                                    'to': (var, line),
                                    'path': current_path,
                                    'def_type': def_info.get('type', 'unknown')
                                })
                                
                                # 查找定义语句中使用的变量
                                for used_var in def_info.get('uses', []):
                                    work_list.append((used_var, def_line, current_path))
            
            # 识别需要边界测试的变量
            boundary_variables = self._identify_boundary_variables(
                affected_vars, def_use_chains, line_number
            )
            
            # 统计信息
            stats = {
                'affected_lines_count': len(affected_lines),
                'affected_variables_count': len(affected_vars),
                'boundary_variables_count': len(boundary_variables),
                'data_flow_paths_count': len(data_flow_paths),
                'slice_depth': max((line - min(affected_lines)) for line in affected_lines) if affected_lines else 0
            }
            
            # 简化边界变量信息，只保留必要字段
            simplified_boundary_vars = []
            for bv in boundary_variables:
                simplified = {
                    'variable': bv['variable'],
                    'boundary_values': bv.get('boundary_values', {}),
                    'suggested_test_values': bv.get('suggested_test_values', [])
                }
                simplified_boundary_vars.append(simplified)
            
            return {
                'boundary_variables': simplified_boundary_vars,
                'data_flow_paths': data_flow_paths
            }
            
        except Exception as e:
            return {}
        
    def _is_on_dataflow_path(self, def_info, var, use_line):
        """检查定义是否在数据流路径上"""
        def_line = def_info['lineno']
        
        # 基本检查:定义在使用之前
        if def_line > use_line:
            return False
        
        # 检查作用域是否匹配
        def_scope = def_info.get('scope', [])
        
        return True

    def _identify_boundary_variables(self, variables, def_use_chains, target_line):
        """识别需要边界测试的变量"""
        boundary_vars = []
        
        # 获取谓词信息
        predicates_result = self.predicate_mining()
        # predicates_result 现在直接是列表
        predicates = predicates_result if predicates_result else []
        
        for var in variables:
            var_info = def_use_chains.get(var, {})
            uses = var_info.get('uses', [])
            definitions = var_info.get('definitions', [])
            
            # 检查变量是否在边界条件中使用
            in_boundary_condition = False
            boundary_predicates = []
            
            for pred in predicates:
                # 检查谓词中是否包含该变量
                if self._variable_in_predicate(var, pred):
                    in_boundary_condition = True
                    boundary_predicates.append(pred)
            
            # 检查变量是否在循环中使用
            in_loop = False
            loop_contexts = []
            for use in uses:
                if use.get('context', {}).get('in_loop', False):
                    in_loop = True
                    loop_contexts.append({
                        'line': use['lineno'],
                        'function': use.get('function')
                    })
            
            # 检查变量是否输入相关
            is_input_related = self._is_input_related(var, definitions)
            
            # 确定是否需要边界测试
            needs_boundary_test = (
                in_boundary_condition or 
                in_loop or 
                is_input_related
            )
            
            if needs_boundary_test:
                # 获取变量的边界值信息
                boundary_values = self._get_variable_boundary_values(var, predicates)

                # 生成建议测试值
                suggested_values = self._suggest_test_values(var, boundary_values, boundary_predicates)

                # 收集变量定义信息
                var_definitions = []
                for d in definitions:
                    if d['lineno'] <= target_line:
                        def_info = {
                            'line': d['lineno'],
                            'type': d.get('type', 'unknown'),
                            'source': d.get('value_expr', 'unknown')
                        }
                        var_definitions.append(def_info)
                
                boundary_vars.append({
                    'variable': var,
                    'needs_boundary_test': needs_boundary_test,
                    'reasons': {
                        'in_boundary_condition': in_boundary_condition,
                        'in_loop': in_loop,
                        'is_input_related': is_input_related
                    },
                    'boundary_predicates': boundary_predicates,
                    'loop_contexts': loop_contexts,
                    'boundary_values': boundary_values,
                    'suggested_test_values': suggested_values,
                    'definitions': var_definitions
                })
        
        # 按重要性排序
        boundary_vars.sort(key=lambda x: self._calculate_variable_importance(x))
        
        return boundary_vars

    def _variable_in_predicate(self, var, predicate):
        """检查变量是否在谓词中"""
        if 'var' in predicate and predicate['var'] == var:
            return True
        
        if 'var_left' in predicate and predicate['var_left'] == var:
            return True
        
        if 'var_right' in predicate and predicate['var_right'] == var:
            return True
        
        # 检查表达式
        expr = predicate.get('expression', '')
        if expr and var in expr:
            # 使用正则表达式确保是完整的单词
            import re
            pattern = rf'\b{re.escape(var)}\b'
            return bool(re.search(pattern, expr))

    def _is_input_related(self, var, definitions):
        """检查变量是否与输入相关"""
        for def_info in definitions:
            def_type = def_info.get('type', '')
            value_expr = def_info.get('value_expr', '')
            
            # 检查是否是输入转换的结果
            input_patterns = [
                'input()',
                'int(input())',
                'float(input())',
                'eval(input())'
            ]
            
            if any(pattern in str(value_expr) for pattern in input_patterns):
                return True
            
            # 检查是否是函数参数
            if def_type == 'parameter':
                return True
            
            # 检查是否来自输入相关的变量
            uses = def_info.get('uses', [])
            for used_var in uses:
                # 递归检查使用的变量是否输入相关
                if self._is_input_related(used_var, []):
                    return True
        
        return False

    def _get_variable_boundary_values(self, var, predicates):
        """获取变量的边界值"""
        boundary_values = {
            'lower': None,
            'upper': None,
            'values': set(),
            'comparisons': []
        }
        
        for pred in predicates:
            if self._variable_in_predicate(var, pred):
                boundary_info = pred.get('boundary_info', {})
                
                # 收集比较值
                if 'value' in boundary_info:
                    boundary_values['values'].add(boundary_info['value'])
                
                if 'left_value' in boundary_info:
                    boundary_values['values'].add(boundary_info['left_value'])
                
                if 'right_value' in boundary_info:
                    boundary_values['values'].add(boundary_info['right_value'])
                
                # 收集建议值
                if 'suggested_values' in boundary_info:
                    for val in boundary_info['suggested_values']:
                        boundary_values['values'].add(val)
                
                # 记录比较信息
                boundary_values['comparisons'].append({
                    'predicate': pred.get('expression', ''),
                    'kind': pred.get('type', 'unknown'),
                    'info': boundary_info
                })

                # 从边界信息中直接提取上下界
                if 'lower' in boundary_info and boundary_info['lower'] is not None:
                    boundary_values['lower'] = boundary_info['lower']
                if 'upper' in boundary_info and boundary_info['upper'] is not None:
                    boundary_values['upper'] = boundary_info['upper']

        # 如果边界信息中没有明确的上下界,尝试从值集合中推断
        numeric_values = [v for v in boundary_values['values'] if isinstance(v, (int, float))]
        if numeric_values:
            if boundary_values['lower'] is None:
                boundary_values['lower'] = min(numeric_values)
            if boundary_values['upper'] is None:
                boundary_values['upper'] = max(numeric_values)
        
        # 转换为排序列表
        boundary_values['values'] = sorted(list(boundary_values['values']))
        
        return boundary_values

    def _suggest_test_values(self, var, boundary_values, boundary_predicates=None):
        """为变量生成测试值建议"""
        test_values = set()
        
        # 添加边界值
        if boundary_values['values']:
            for val in boundary_values['values']:
                if isinstance(val, (int, float, str, bool)):
                    test_values.add(val)

        # 2. 从谓词中获取建议值
        if boundary_predicates:
            for pred in boundary_predicates:
                if 'boundary_info' in pred and 'suggested_values' in pred['boundary_info']:
                    for val in pred['boundary_info']['suggested_values']:
                        if isinstance(val, (int, float)):
                            test_values.add(val)
        
        # 添加边界附近的典型测试值
        lower = boundary_values['lower']
        upper = boundary_values['upper']
        
        if lower is not None and isinstance(lower, (int, float)):
            test_values.update([lower - 1, lower, lower + 1])
        
        if upper is not None and isinstance(upper, (int, float)):
            test_values.update([upper - 1, upper, upper + 1])
        
        # 如果只有单个值,添加一些典型变异
        if len(boundary_values['values']) == 1:
            single_value = next(iter(boundary_values['values']))
            if isinstance(single_value, (int, float)):
                test_values.update([single_value - 10, single_value + 10])
            elif isinstance(single_value, str):
                test_values.update(['', 'long_string', 'special_chars'])
        
        # 转换为排序列表
        return sorted(list(test_values))

    def _calculate_variable_importance(self, var_info):
        """计算变量的重要性分数"""
        score = 0
        
        # 在边界条件中使用的变量很重要
        if var_info['reasons']['in_boundary_condition']:
            score += 10
        
        # 在循环中使用的变量较重要
        if var_info['reasons']['in_loop']:
            score += 5
        
        # 输入相关的变量重要
        if var_info['reasons']['is_input_related']:
            score += 8
        
        # 有多个定义或使用的变量可能更重要
        definitions = var_info.get('definitions', [])
        score += min(len(definitions), 5)  # 最多加5分
        
        return -score  # 负分用于排序(分数越高越靠前)
    
    def _build_def_use_chains(self) -> Dict[str, Dict[str, List]]:
        """构建完整的def-use链(内部方法)"""
        if self._def_use_chains is not None:
            return self._def_use_chains
        
        self._def_use_chains = {}
        
        class DefUseVisitor(ast.NodeVisitor):
            def __init__(self, chains):
                self.chains = chains
                self.current_scope = []
                self._current_function = None
                self._variable_context = {}  # 跟踪变量上下文
                
            def _ensure_var_entry(self, var_name):
                """确保变量在字典中有正确的结构"""
                if var_name not in self.chains:
                    self.chains[var_name] = {
                        'definitions': [],
                        'uses': [],
                        'type': None,
                        'source': None
                    }
                
            def visit_FunctionDef(self, node):
                """进入函数作用域"""
                old_function = self._current_function
                self._current_function = node.name
                self.current_scope.append(node.name)
                
                # 处理函数参数(也是定义)
                if node.args:
                    # 位置参数
                    if hasattr(node.args, 'args'):
                        for arg in node.args.args:
                            if arg:
                                var_name = arg.arg
                                self._ensure_var_entry(var_name)
                                self.chains[var_name]['definitions'].append({
                                    'lineno': node.lineno,
                                    'scope': self.current_scope.copy(),
                                    'type': 'parameter',
                                    'function': node.name,
                                    'position': 'arg',
                                    'uses': []  # 参数定义时不使用其他变量
                                })
                                self.chains[var_name]['source'] = 'parameter'
                    
                    # 关键字参数
                    if hasattr(node.args, 'kwonlyargs'):
                        for arg in node.args.kwonlyargs:
                            if arg:
                                var_name = arg.arg
                                self._ensure_var_entry(var_name)
                                self.chains[var_name]['definitions'].append({
                                    'lineno': node.lineno,
                                    'scope': self.current_scope.copy(),
                                    'type': 'parameter',
                                    'function': node.name,
                                    'position': 'kwarg',
                                    'uses': []
                                })
                                self.chains[var_name]['source'] = 'parameter'
                    
                    # 可变参数 *args
                    if hasattr(node.args, 'vararg') and node.args.vararg:
                        var_name = node.args.vararg.arg
                        self._ensure_var_entry(var_name)
                        self.chains[var_name]['definitions'].append({
                            'lineno': node.lineno,
                            'scope': self.current_scope.copy(),
                            'type': 'vararg',
                            'function': node.name,
                            'position': '*args',
                            'uses': []
                        })
                        self.chains[var_name]['source'] = 'parameter'
                    
                    # 关键字参数字典 **kwargs
                    if hasattr(node.args, 'kwarg') and node.args.kwarg:
                        var_name = node.args.kwarg.arg
                        self._ensure_var_entry(var_name)
                        self.chains[var_name]['definitions'].append({
                            'lineno': node.lineno,
                            'scope': self.current_scope.copy(),
                            'type': 'kwarg',
                            'function': node.name,
                            'position': '**kwargs',
                            'uses': []
                        })
                        self.chains[var_name]['source'] = 'parameter'
                
                self.generic_visit(node)
                self.current_scope.pop()
                self._current_function = old_function
                
            def visit_Assign(self, node):
                """处理赋值语句(定义)"""
                # 获取使用的变量和操作类型
                uses_info = self._extract_uses_with_context(node.value)
                uses = [info['var'] for info in uses_info]
                
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        self._ensure_var_entry(var_name)
                        
                        # 记录赋值表达式
                        try:
                            value_expr = ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                        except:
                            value_expr = str(node.value)
                        
                        self.chains[var_name]['definitions'].append({
                            'lineno': node.lineno,
                            'scope': self.current_scope.copy(),
                            'function': self._current_function,
                            'uses': uses,
                            'uses_with_context': uses_info,  
                            'type': 'assignment',
                            'node_type': 'Assign',
                            'value_expr': value_expr,
                            'value_node': node.value  
                        })
                        self.chains[var_name]['source'] = 'assignment'
                    
                self.generic_visit(node)
                
            def visit_AnnAssign(self, node):
                """处理带类型注解的赋值语句"""
                uses = self._extract_uses(node.value) if node.value else []
                
                if isinstance(node.target, ast.Name):
                    var_name = node.target.id
                    self._ensure_var_entry(var_name)
                    
                    annotation = None
                    if node.annotation:
                        try:
                            annotation = ast.unparse(node.annotation)
                        except:
                            annotation = str(node.annotation)
                    
                    self.chains[var_name]['definitions'].append({
                        'lineno': node.lineno,
                        'scope': self.current_scope.copy(),
                        'function': self._current_function,
                        'uses': uses,
                        'type': 'annotated_assignment',
                        'annotation': annotation,
                        'node_type': 'AnnAssign',
                        'value_expr': ast.unparse(node.value) if node.value else None
                    })
                    self.chains[var_name]['source'] = 'assignment'
                
                self.generic_visit(node)
                
            def visit_AugAssign(self, node):
                """处理增强赋值(如 x += 1)"""
                # 增强赋值既是使用也是定义
                uses = self._extract_uses(node.value)
                
                # 目标变量本身也被使用
                if isinstance(node.target, ast.Name):
                    uses.append(node.target.id)
                
                if isinstance(node.target, ast.Name):
                    var_name = node.target.id
                    self._ensure_var_entry(var_name)
                    self.chains[var_name]['definitions'].append({
                        'lineno': node.lineno,
                        'scope': self.current_scope.copy(),
                        'function': self._current_function,
                        'uses': uses,
                        'type': 'augmented_assignment',
                        'op': type(node.op).__name__,
                        'node_type': 'AugAssign',
                        'value_expr': ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                    })
                    self.chains[var_name]['source'] = 'assignment'
                
                self.generic_visit(node)
                
            def visit_For(self, node):
                """处理for循环(循环变量定义)"""
                # 获取迭代对象中使用的变量
                iter_uses = self._extract_uses(node.iter)
                
                if isinstance(node.target, ast.Name):
                    var_name = node.target.id
                    self._ensure_var_entry(var_name)
                    self.chains[var_name]['definitions'].append({
                        'lineno': node.lineno,
                        'scope': self.current_scope.copy(),
                        'function': self._current_function,
                        'uses': iter_uses,
                        'type': 'for_loop_variable',
                        'node_type': 'For',
                        'iter_expr': ast.unparse(node.iter) if hasattr(ast, 'unparse') else str(node.iter)
                    })
                    self.chains[var_name]['source'] = 'loop_variable'
                
                # 处理循环体
                self.generic_visit(node)
                
            def visit_With(self, node):
                """处理with语句(上下文管理器变量)"""
                for item in node.items:
                    # 处理 as 子句中的变量
                    if item.optional_vars:
                        if isinstance(item.optional_vars, ast.Name):
                            var_name = item.optional_vars.id
                            self._ensure_var_entry(var_name)
                            
                            # 获取上下文表达式中的使用
                            context_uses = self._extract_uses(item.context_expr)
                            
                            self.chains[var_name]['definitions'].append({
                                'lineno': node.lineno,
                                'scope': self.current_scope.copy(),
                                'function': self._current_function,
                                'uses': context_uses,
                                'type': 'context_manager_variable',
                                'node_type': 'With',
                                'context_expr': ast.unparse(item.context_expr) if hasattr(ast, 'unparse') else str(item.context_expr)
                            })
                            self.chains[var_name]['source'] = 'context_manager'
                
                self.generic_visit(node)
                
            def visit_ExceptHandler(self, node):
                """处理异常处理器中的变量"""
                if node.name:
                    var_name = node.name
                    self._ensure_var_entry(var_name)
                    self.chains[var_name]['definitions'].append({
                        'lineno': node.lineno,
                        'scope': self.current_scope.copy(),
                        'function': self._current_function,
                        'uses': [],  # 异常变量不依赖于其他变量
                        'type': 'exception_variable',
                        'node_type': 'ExceptHandler',
                        'exception_type': ast.unparse(node.type) if node.type else None
                    })
                    self.chains[var_name]['source'] = 'exception'
                
                self.generic_visit(node)
                
            def visit_ListComp(self, node):
                """处理列表推导式中的生成器变量"""
                self._handle_comprehension_vars(node, 'list_comprehension')
                self.generic_visit(node)
                
            def visit_SetComp(self, node):
                """处理集合推导式中的生成器变量"""
                self._handle_comprehension_vars(node, 'set_comprehension')
                self.generic_visit(node)
                
            def visit_DictComp(self, node):
                """处理字典推导式中的生成器变量"""
                self._handle_comprehension_vars(node, 'dict_comprehension')
                self.generic_visit(node)
                
            def visit_GeneratorExp(self, node):
                """处理生成器表达式中的生成器变量"""
                self._handle_comprehension_vars(node, 'generator_expression')
                self.generic_visit(node)
                
            def _handle_comprehension_vars(self, node, comp_type):
                """处理推导式中的变量定义"""
                for generator in node.generators:
                    # 生成器中的目标变量
                    if isinstance(generator.target, ast.Name):
                        var_name = generator.target.id
                        self._ensure_var_entry(var_name)
                        
                        # 获取迭代器中的使用
                        iter_uses = self._extract_uses(generator.iter)
                        
                        self.chains[var_name]['definitions'].append({
                            'lineno': node.lineno,
                            'scope': self.current_scope.copy(),
                            'function': self._current_function,
                            'uses': iter_uses,
                            'type': f'{comp_type}_generator',
                            'node_type': type(node).__name__,
                            'iter_expr': ast.unparse(generator.iter) if hasattr(ast, 'unparse') else str(generator.iter)
                        })
                        self.chains[var_name]['source'] = 'comprehension'
                
            def visit_Name(self, node):
                """处理变量使用"""
                if isinstance(node.ctx, ast.Load):  # 读取变量
                    var_name = node.id
                    self._ensure_var_entry(var_name)
                    self.chains[var_name]['uses'].append({
                        'lineno': node.lineno,
                        'scope': self.current_scope.copy(),
                        'function': self._current_function,
                        'context': self._get_usage_context(node),
                        'node_type': 'Name'
                    })
                            
            def _extract_uses_with_context(self, node):
                """从表达式提取使用的变量及上下文"""
                uses_info = []
                
                class UseExtractor(ast.NodeVisitor):
                    def __init__(self, use_list):
                        self.uses = use_list
                        self.current_context = []
                    
                    def visit_BinOp(self, node):
                        # 记录操作符类型
                        op_type = type(node.op).__name__
                        self.current_context.append(f'binop_{op_type}')
                        self.generic_visit(node)
                        self.current_context.pop()
                    
                    def visit_Compare(self, node):
                        self.current_context.append('compare')
                        self.generic_visit(node)
                        self.current_context.pop()
                    
                    def visit_Call(self, node):
                        # 记录函数名
                        func_name = None
                        if isinstance(node.func, ast.Name):
                            func_name = node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            func_name = node.func.attr
                        
                        if func_name:
                            self.current_context.append(f'call_{func_name}')
                        
                        self.generic_visit(node)
                        
                        if func_name:
                            self.current_context.pop()
                    
                    def visit_Name(self, node):
                        if isinstance(node.ctx, ast.Load):
                            context_info = {
                                'var': node.id,
                                'context': self.current_context.copy() if self.current_context else [],
                                'lineno': node.lineno
                            }
                            self.uses.append(context_info)
                
                extractor = UseExtractor(uses_info)
                if node:
                    extractor.visit(node)
                
                return uses_info
                
            def _get_usage_context(self, node):
                """获取变量使用的上下文信息"""
                context = {
                    'parent_type': None,
                    'in_condition': False,
                    'in_loop': False,
                    'in_call': False,
                    'in_assignment': False
                }
                
                # 检查父节点类型
                if hasattr(node, 'parent'):
                    parent = node.parent
                    if parent:
                        context['parent_type'] = type(parent).__name__
                
                return context
            
            def _extract_uses(self, node):
                """从表达式中提取使用的变量名列表"""
                uses = []
                
                class UseExtractor(ast.NodeVisitor):
                    def __init__(self, use_list):
                        self.uses = use_list
                    
                    def visit_Name(self, node):
                        if isinstance(node.ctx, ast.Load):
                            self.uses.append(node.id)
                        self.generic_visit(node)
                
                if node:
                    extractor = UseExtractor(uses)
                    extractor.visit(node)
                
                return uses
        
        visitor = DefUseVisitor(self._def_use_chains)
        visitor.visit(self.ast_tree)
        
        return self._def_use_chains
    
    def build_data_dependency_graph(self) -> Dict[str, Any]:
        """
        构建Def-Use/数据依赖图
        - 记录数据从哪里产生, 到哪里被使用
        - 识别哪些变量修改后可能影响输出, 即应该优先变异哪些变量
        
        Returns:
            dict, 变量名到依赖变量列表的映射
            例如: {'sum1': ['h', 'alp'], 'h': [], 'n': []}
            表示sum1依赖于h和alp, h和n不依赖其他变量
            失败时返回空字典 {}
        """
        try:
            # 先构建def-use链
            def_use_chains = self._build_def_use_chains()
            
            self._data_deps = defaultdict(list)
            
            # 构建数据依赖关系
            for var_name, info in def_use_chains.items():
                for definition in info.get('definitions', []):
                    def_line = definition['lineno']
                    # 找到使用这个定义的所有地方
                    for use in info.get('uses', []):
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
                for definition in info.get('definitions', []):
                    for used_var in definition.get('uses', []):
                        if used_var in def_use_chains:
                            self._data_deps[var_name].append({
                                'depends_on': used_var,
                                'line': definition['lineno'],
                                'type': 'data_dependency'
                            })
            
            # 内置函数列表(需要过滤)
            builtin_functions = {
                'eval', 'input', 'print', 'range', 'int', 'float', 'str', 'len',
                'abs', 'round', 'sum', 'max', 'min', 'type', 'isinstance',
                'issubclass', 'open', 'close', 'read', 'write', 'search',
                'map', 'filter', 'zip', 'enumerate', 'sorted', 'reversed',
                'list', 'dict', 'set', 'tuple', 'bool', 'bytes', 'bytearray',
                'complex', 'frozenset', 'memoryview', 'object', 'slice',
                'staticmethod', 'classmethod', 'property', 'super',
                'all', 'any', 'ascii', 'bin', 'callable', 'chr', 'compile',
                'delattr', 'dir', 'divmod', 'format', 'getattr', 'globals',
                'hasattr', 'hash', 'help', 'hex', 'id', 'iter', 'locals',
                'next', 'oct', 'ord', 'pow', 'repr', 'setattr', 'vars'
            }
            
            # 简化返回结构: 只返回变量名到依赖变量列表的映射
            result = {}
            for var, deps in self._data_deps.items():
                # 过滤掉内置函数名
                if var in builtin_functions:
                    continue
                    
                # 提取依赖的变量名列表，同时过滤掉函数名和自依赖
                depends_on = []
                for dep in deps:
                    if 'depends_on' in dep:
                        dep_var = dep['depends_on']
                        # 过滤掉内置函数名、自依赖和重复项
                        if (dep_var not in builtin_functions and 
                            dep_var != var and 
                            dep_var not in depends_on):
                            depends_on.append(dep_var)
                result[var] = depends_on
            
            return result
            
        except Exception as e:
            return {}
    
    def _same_scope(self, scope1, scope2):
        """检查两个作用域是否相同"""
        return scope1 == scope2
    
    def infer_input_structure(self) -> Dict[str, Any]:
        """
        输入结构/格式推断
        - 支持各种input()使用模式
        - 考虑上下文推断类型
        - 识别循环,条件中的输入
        - 智能推断数据结构
        
        Returns:
            dict, 变量名到类型的映射, 包含输入变量和常量
            例如: {'h': 'int', 'n': 'int', 'alp': 'float', 'x': 'list[str]'}
            表示h和n是整数输入, alp是浮点常量, x是字符串列表
            不包含循环变量(如for循环中的i)
            失败时返回空字典 {}
        """
        
        try:
            # 获取完整的变量类型信息（内部方法，包含来源信息）
            full_result = self._get_variable_types_full()
            if not full_result or 'variable_types' not in full_result:
                return {}
            
            variable_types = full_result['variable_types']
            
            # 识别循环变量（通过检查变量的来源）
            loop_variables = set()
            for var_name, type_info in variable_types.items():
                if isinstance(type_info, dict):
                    sources = type_info.get('sources', [])
                    # 如果变量的来源是 for_loop，则认为是循环变量
                    for source in sources:
                        if source.get('source') == 'for_loop':
                            loop_variables.add(var_name)
                            break
            
            # 构建结果，排除循环变量
            result = {}
            for var_name, type_info in variable_types.items():
                # 跳过循环变量
                if var_name in loop_variables:
                    continue
                    
                if isinstance(type_info, dict):
                    var_type = type_info.get('type', 'Any')
                    result[var_name] = var_type
                else:
                    result[var_name] = str(type_info)
            
            return result
            
        except Exception as e:
            return {}
    
    def get_variable_types(self) -> Dict[str, str]:
        """
        简化的变量类型推断 - 只返回变量名到类型的映射
        
        Returns:
            dict: 变量名到类型字符串的映射，例如 {'n': 'int', 'arr': 'list', 's': 'str'}
            失败时返回空字典 {}
        """
        try:
            # 调用完整的内部方法获取详细信息
            full_result = self._get_variable_types_full()
            
            if not full_result or 'variable_types' not in full_result:
                return {}
            
            # 只提取变量名和类型
            simple_types = {}
            for var_name, var_info in full_result['variable_types'].items():
                simple_types[var_name] = var_info.get('type', 'Any')
            
            return simple_types
            
        except Exception as e:
            return {}
    
    def _get_variable_types_full(self) -> Dict[str, Any]:
        """
        完整的变量类型推断（内部使用）
        - 结合多种推断方法
        - 支持复杂的类型推断
        - 增强对输入表达式的类型推断
        
        Returns:
            {
                'variable_types': dict, 变量名到类型信息的映射, 每个包含type(类型), confidence(置信度), sources(来源列表)等
                'statistics': dict, 统计信息, 包含total_variables(变量总数), types_found(发现的类型列表), confidence_distribution(置信度分布), sources_distribution(来源分布), variables_by_confidence(按置信度分组的变量)
                'metadata': dict, 元数据, 包含has_type_inference(是否有类型推断), type_variety(类型多样性), inference_quality(推断质量)
            }
            失败时返回空字典 {}
        """
        try:
            if self._variable_types is not None:
                return {
                        'variable_types': self._variable_types,
                        'statistics': {
                            'total_variables': len(self._variable_types),
                            'types_found': set(v['type'] for v in self._variable_types.values()),
                            'confidence_distribution': defaultdict(int)
                        }
                    }
            
            self._variable_types = {}
            
            # 获取其他分析结果作为上下文
            input_structure_result = self.infer_input_structure()
            def_use_chains = self._build_def_use_chains()
            constants_result = self.extract_constants_and_comparisons()
            
            # constants_result 现在直接是字典，不需要再取 'constants' 键
            constants = constants_result if constants_result else {}
            
            # 1. 创建类型推断器
            inferencer = TypeInferencer(
                variable_types=self._variable_types,
                input_structure=input_structure_result.get('input_structure', {}) if input_structure_result else {},
                def_use_chains=def_use_chains,
                constants=constants,
                ast_tree=self.ast_tree
            )
            
            # 2. 执行多轮类型推断
            inferencer.visit(self.ast_tree)
            
            # 3. 传播和解析类型
            self._propagate_types()
            self._resolve_type_conflicts()
            self._infer_missing_types()
            self._infer_from_usage_context()

            # 过滤掉内置函数和明显不是变量的项
            self._filter_non_variables()
            
            # 同步更新输入结构中的变量类型
            self._sync_input_structure_types()
            
            # 统计信息
            stats = {
                'total_variables': len(self._variable_types),
                'types_found': set(),
                'confidence_distribution': defaultdict(int),
                'sources_distribution': defaultdict(int),
                'variables_by_confidence': defaultdict(list)
            }
            
            for var_name, var_info in self._variable_types.items():
                stats['types_found'].add(var_info['type'])
                stats['confidence_distribution'][var_info.get('confidence', 'unknown')] += 1
                
                # 统计来源
                for source in var_info.get('sources', []):
                    stats['sources_distribution'][source['source']] += 1
                
                # 按置信度分组
                confidence = var_info.get('confidence', 'unknown')
                stats['variables_by_confidence'][confidence].append(var_name)
            
            stats['types_found'] = list(stats['types_found'])
            stats['confidence_distribution'] = dict(stats['confidence_distribution'])
            stats['sources_distribution'] = dict(stats['sources_distribution'])
            
            return {
                    'variable_types': self._variable_types,
                    'statistics': stats,
                    'metadata': {
                        'has_type_inference': len(self._variable_types) > 0,
                        'type_variety': len(stats['types_found']),
                        'inference_quality': stats['confidence_distribution'].get('high', 0) / max(1, len(self._variable_types))
                    }
                }
            
        except Exception as e:
            return {}
        
    def _filter_non_variables(self):
        """过滤掉非变量项"""
        # 内置函数列表
        builtin_functions = {
            'eval', 'input', 'print', 'range', 'int', 'float', 'str', 'len',
            'abs', 'round', 'sum', 'max', 'min', 'type', 'isinstance',
            'issubclass', 'open', 'close', 'read', 'write', 'search',
            'map', 'filter', 'zip', 'enumerate', 'sorted', 'reversed'
        }
        
        # 要移除的项
        to_remove = []
        
        for var_name in list(self._variable_types.keys()):
            # 如果是内置函数
            if var_name in builtin_functions:
                to_remove.append(var_name)
            # 如果以特殊符号开头或包含特殊字符
            elif var_name.startswith('_') or '.' in var_name:
                to_remove.append(var_name)
            # 如果类型信息过于模糊
            elif self._variable_types[var_name].get('confidence', 'low') == 'low' and \
                self._variable_types[var_name].get('type', '') == 'Any':
                # 可以考虑移除或保留
                pass
        
        for var_name in to_remove:
            del self._variable_types[var_name]
    
    def _sync_input_structure_types(self):
        """将变量类型推断结果同步更新到输入结构中"""
        if self._input_structure is None:
            return
        
        # 更新 input_patterns 中的类型
        if 'input_patterns' in self._input_structure:
            for pattern in self._input_structure['input_patterns']:
                var_name = pattern.get('variable')
                if var_name and var_name in self._variable_types:
                    var_type_info = self._variable_types[var_name]
                    # 更新推断类型
                    pattern['inferred_type'] = var_type_info['type']
                    pattern['type_confidence'] = var_type_info.get('confidence', 'unknown')
                    pattern['type_sources'] = [s['source'] for s in var_type_info.get('sources', [])]
        
        # 更新 detected_types 字典
        if 'detected_types' in self._input_structure:
            for var_name, var_type_info in self._variable_types.items():
                # 检查这个变量是否是输入变量
                is_input_var = any(
                    pattern.get('variable') == var_name 
                    for pattern in self._input_structure.get('input_patterns', [])
                )
                if is_input_var:
                    self._input_structure['detected_types'][var_name] = {
                        'type': var_type_info['type'],
                        'confidence': var_type_info.get('confidence', 'unknown'),
                        'sources': [s['source'] for s in var_type_info.get('sources', [])]
                    }
        
        # 更新 expected_types 字典
        if 'expected_types' in self._input_structure:
            for var_name, var_type_info in self._variable_types.items():
                # 检查这个变量是否是输入变量
                is_input_var = any(
                    pattern.get('variable') == var_name 
                    for pattern in self._input_structure.get('input_patterns', [])
                )
                if is_input_var:
                    self._input_structure['expected_types'][var_name] = var_type_info['type']
        
        # 更新 main_inputs 中的类型信息
        if 'main_inputs' in self._input_structure:
            for main_input in self._input_structure['main_inputs']:
                var_name = main_input.get('variable')
                if var_name and var_name in self._variable_types:
                    var_type_info = self._variable_types[var_name]
                    main_input['detected_type'] = var_type_info['type']
                    main_input['type_confidence'] = var_type_info.get('confidence', 'unknown')
        
        # 更新 input_calls 中的类型信息
        if 'input_calls' in self._input_structure:
            for input_call in self._input_structure['input_calls']:
                var_name = input_call.get('variable')
                if var_name and var_name in self._variable_types:
                    var_type_info = self._variable_types[var_name]
                    input_call['inferred_type'] = var_type_info['type']
                    input_call['type_confidence'] = var_type_info.get('confidence', 'unknown')
        
        # 更新 constraints 中的类型信息
        if 'constraints' in self._input_structure:
            for constraint in self._input_structure['constraints']:
                var_name = constraint.get('variable')
                if var_name and var_name in self._variable_types:
                    var_type_info = self._variable_types[var_name]
                    constraint['type'] = var_type_info['type']
                    constraint['type_confidence'] = var_type_info.get('confidence', 'unknown')
                    constraint['type_sources'] = [s['source'] for s in var_type_info.get('sources', [])]
        
        # 更新 input_transformations 中的结束类型
        if 'input_transformations' in self._input_structure:
            for transformation in self._input_structure['input_transformations']:
                var_name = transformation.get('variable')
                if var_name and var_name in self._variable_types:
                    var_type_info = self._variable_types[var_name]
                    transformation['end_type'] = var_type_info['type']
    
    def _propagate_types(self):
        """传播类型信息"""

        # 推断eval(input())的类型
        self._infer_eval_input_types()

        # 进行赋值类型推断
        self._infer_assignment_types()

        # 推断循环中的变量类型
        self._infer_loop_variable_types()

        # 分析列表使用模式
        self._analyze_list_usage_patterns()

        # 特殊处理增强赋值
        self._propagate_augmented_assignment_types()
        
        # 进行通用数值类型推断
        self._infer_numeric_types()

        # 基于def-use链传播类型
        def_use_chains = self._build_def_use_chains()
        
        changed = True
        max_iterations = 20 
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for var_name, var_info in self._variable_types.items():
                # 如果变量类型不明确,尝试从使用处推断
                if var_info.get('confidence', 'low') in ['low', 'medium']:
                    if var_name in def_use_chains:
                        uses = def_use_chains[var_name].get('uses', [])
                        
                        # 1. 检查变量是否用于取模运算(%)
                        for use in uses:
                            if use.get('op_type') == 'Mod' or use.get('context', {}).get('operation') == '%':
                                # 取模运算通常用于整数
                                if var_info['type'] in ['Any', 'unknown', 'numeric']:
                                    self._update_variable_type(var_name, 'int', 'modulo_usage','high')
                                    changed = True
                                break
                        
                        # 2. 检查变量是否用于range()函数
                        for use in uses:
                            context = use.get('context', {})
                            if context.get('call_func') == 'range':
                                # range()的参数通常是整数
                                if var_info['type'] in ['Any', 'unknown', 'numeric']:
                                    self._update_variable_type(var_name, 'int', 'range_usage','high')
                                    changed = True
                                break
                        
                        # 3. 检查变量是否在比较运算中使用
                        if self._is_used_in_comparison(var_name, uses):
                            if var_info['type'] in ['Any', 'unknown']:
                                # 如果比较中使用了数值常量,推断为数值类型
                                self._update_variable_type(var_name, 'numeric', 'comparison_usage','high')
                                changed = True
                        
                        # 4. 检查变量是否在数值运算中使用
                        if self._is_used_in_numeric_context(var_name, uses):
                            if var_info['type'] in ['Any', 'unknown']:
                                self._update_variable_type(var_name, 'numeric', 'numeric_operation_usage','high')
                                changed = True
                        
                        # 5. 检查变量是否在循环中使用
                        if self._is_used_in_loop(var_name, uses):
                            if var_info['type'] in ['Any', 'unknown', 'numeric']:
                                # 循环索引或范围通常是整数
                                self._update_variable_type(var_name, 'int', 'loop_usage','high')
                                changed = True

                        if self._is_used_in_numeric_loop(var_name, def_use_chains):
                            if var_info['type'] in ['Any', 'unknown']:
                                self._update_variable_type(var_name, 'numeric', 'numeric_loop_usage', 'high')
                                changed = True
            
            # 6. 传播类型到相关变量(包括赋值关系)
            changed = changed or self._propagate_related_types(def_use_chains)
            
            # 7. 额外的赋值传播
            changed = changed or self._propagate_direct_assignments(def_use_chains)

    def _is_used_in_numeric_loop(self, var_name, def_use_chains):
        """检查变量是否在数值循环中使用"""
        if var_name not in def_use_chains:
            return False
        
        uses = def_use_chains[var_name].get('uses', [])
        
        for use in uses:
            # 检查是否在循环的数值运算中使用
            context = use.get('context', {})
            
            # 检查循环上下文
            if context.get('in_loop', False):
                # 检查是否在数值运算中
                if 'context' in use:
                    for ctx in use['context']:
                        if ctx.startswith('binop_'):
                            return True
                if context.get('parent_type') in ['BinOp', 'AugAssign']:
                    return True
        
        return False
    
    def _analyze_list_usage_patterns(self):
        """分析列表使用模式"""
        # 分析 max()/min() 使用的列表
        class ListUsageAnalyzer(ast.NodeVisitor):
            def __init__(self, variable_types):
                self.var_types = variable_types
                self.list_evidence = defaultdict(list)  # var -> [(evidence_type, confidence)]
                
            def visit_Call(self, node):
                # 检查 max()/min() 调用
                if isinstance(node.func, ast.Name) and node.func.id in ['max', 'min']:
                    if node.args:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.Name):
                            var_name = first_arg.id
                            self.list_evidence[var_name].append(('used_in_max_min', 'high'))
                            
                            # 检查 max()/min() 的返回值是否被赋值
                            if isinstance(node.parent, ast.Assign):
                                # max/min 通常用于数值比较,推断列表元素为数值
                                self.list_evidence[var_name].append(('returns_numeric_from_max_min', 'high'))
                
                # 检查 len() 调用
                if isinstance(node.func, ast.Name) and node.func.id == 'len':
                    if node.args:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.Name):
                            var_name = first_arg.id
                            self.list_evidence[var_name].append(('used_in_len', 'medium'))
                
                self.generic_visit(node)
                
            def visit_Subscript(self, node):
                # 检查下标访问
                if isinstance(node.value, ast.Name):
                    var_name = node.value.id
                    self.list_evidence[var_name].append(('subscript_access', 'medium'))
                
                self.generic_visit(node)
                
            def visit_Compare(self, node):
                # 检查列表元素的比较
                left = node.left
                for comparator in node.comparators:
                    # 检查是否为列表元素与常量的比较
                    if isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name):
                        var_name = left.value.id
                        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, (int, float)):
                            self.list_evidence[var_name].append(
                                (f'element_comparison_{type(comparator.value).__name__}', 'high')
                            )
                    
                    left = comparator
                
                self.generic_visit(node)
        
        analyzer = ListUsageAnalyzer(self._variable_types)
        
        # 为节点添加 parent 属性以便追踪
        for node in ast.walk(self.ast_tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
        
        analyzer.visit(self.ast_tree)
        
        # 基于证据更新类型
        for var_name, evidence_list in analyzer.list_evidence.items():
            if var_name in self._variable_types:
                # 检查是否有强证据表明是数值列表
                has_max_min = any(ev[0] == 'used_in_max_min' for ev in evidence_list)
                has_numeric_return = any(ev[0] == 'returns_numeric_from_max_min' for ev in evidence_list)
                has_int_comparison = any(ev[0] == 'element_comparison_int' for ev in evidence_list)
                has_float_comparison = any(ev[0] == 'element_comparison_float' for ev in evidence_list)
                
                if has_max_min or has_numeric_return:
                    if has_int_comparison and not has_float_comparison:
                        self._update_variable_type(var_name, 'list[int]', 'max_min_with_int_comparison', 'high')
                    elif has_float_comparison:
                        self._update_variable_type(var_name, 'list[float]', 'max_min_with_float_comparison', 'high')
                    else:
                        self._update_variable_type(var_name, 'list[numeric]', 'used_in_max_min', 'medium')

    def _propagate_augmented_assignment_types(self):
        """传播增强赋值的类型"""
        # 分析增强赋值语句
        class AugAssignAnalyzer(ast.NodeVisitor):
            def __init__(self, variable_types):
                self.var_types = variable_types
                self.assignments = []  # (target, value_expr, op, lineno)
            
            def visit_AugAssign(self, node):
                if isinstance(node.target, ast.Name):
                    target_var = node.target.id
                    op_type = type(node.op).__name__
                    
                    try:
                        value_expr = ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                    except:
                        value_expr = str(node.value)
                    
                    self.assignments.append((target_var, value_expr, op_type, node.lineno))
                
                self.generic_visit(node)
        
        # 运行分析器
        analyzer = AugAssignAnalyzer(self._variable_types)
        analyzer.visit(self.ast_tree)
        
        # 处理增强赋值
        for target_var, value_expr, op_type, lineno in analyzer.assignments:
            if target_var in self._variable_types:
                # 分析右侧表达式中的变量类型
                var_types_in_expr = []
                
                # 简单解析表达式中的变量
                import re
                var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
                variables_in_expr = re.findall(var_pattern, value_expr)
                
                for var in variables_in_expr:
                    if var in self._variable_types:
                        var_types_in_expr.append(self._variable_types[var]['type'])
                
                # 如果有数值类型的变量,推断为数值类型
                numeric_types = ['int', 'float', 'numeric']
                if any(t in numeric_types for t in var_types_in_expr):
                    # 检查是否有浮点数运算
                    has_float = any(t == 'float' for t in var_types_in_expr)
                    
                    current_type = self._variable_types[target_var]['type']
                    current_conf = self._variable_types[target_var].get('confidence', 'low')
                    
                    # 提升类型:从 Any -> numeric -> int/float
                    if current_type == 'Any':
                        new_type = 'float' if has_float else 'int'
                        self._update_variable_type(
                            target_var, 
                            new_type, 
                            f'augmented_assignment_{op_type}', 
                            'medium'
                        )
                    elif current_type == 'numeric':
                        new_type = 'float' if has_float else 'int'
                        if new_type != 'numeric':  # 更具体化
                            self._update_variable_type(
                                target_var, 
                                new_type, 
                                f'augmented_assignment_specific_{op_type}', 
                                'medium'
                            )

    def _propagate_direct_assignments(self, def_use_chains):
        """传播直接赋值关系"""
        changed = False
        
        # 查找所有直接赋值:target = source
        for var_name, var_info in self._variable_types.items():
            if var_name in def_use_chains:
                for def_info in def_use_chains[var_name].get('definitions', []):
                    if def_info.get('type') == 'assignment':
                        value_expr = def_info.get('value_expr', '')
                        
                        # 检查是否为简单变量赋值:var1 = var2
                        if value_expr in self._variable_types:
                            source_var = value_expr
                            source_info = self._variable_types[source_var]
                            
                            # 源变量有明确类型
                            if source_info['confidence'] in ['high', 'medium'] and source_info['type'] != 'Any':
                                current_info = self._variable_types[var_name]
                                
                                # 如果目标变量类型不明确
                                if current_info['type'] in ['Any', 'unknown'] or current_info['confidence'] == 'low':
                                    self._update_variable_type(
                                        var_name,
                                        source_info['type'],
                                        f'direct_assignment_from_{source_var}',
                                        source_info['confidence']
                                    )
                                    changed = True
        
        return changed

    def _is_used_in_numeric_context(self, var_name, uses):
        """检查变量是否在数值上下文中使用"""
        for use in uses:
            context = use.get('context', {})
            
            # 从def-use链中获取更详细的信息
            if 'context' in use:
                for ctx in use['context']:
                    if ctx.startswith('binop_'):
                        op_type = ctx.replace('binop_', '')
                        if op_type in ['Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Mod', 'Pow']:
                            return True
            
            parent_type = context.get('parent_type')
            
            # 在二元运算中使用
            if parent_type in ['BinOp']:
                op_type = context.get('op_type', '')
                if op_type in ['Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Mod', 'Pow']:
                    return True
            
            # 在赋值中使用
            if parent_type in ['AugAssign']:
                op_type = context.get('op_type', '')
                if op_type in ['Add', 'Sub', 'Mult', 'Div']:
                    return True
        
        return False

    def _is_used_in_comparison(self, var_name, uses):
        """检查变量是否在比较运算中使用"""
        for use in uses:
            context = use.get('context', {})
            
            # 从def-use链中检查
            if 'context' in use:
                for ctx in use['context']:
                    if ctx == 'compare':
                        return True
            
            parent_type = context.get('parent_type')
            
            if parent_type in ['Compare']:
                return True
        
        return False

    def _is_used_in_loop(self, var_name, uses):
        """检查变量是否在循环中使用"""
        for use in uses:
            context = use.get('context', {})
            
            if context.get('in_loop', False):
                return True
            
            # 检查是否为range参数
            if 'context' in use:
                for ctx in use['context']:
                    if ctx == 'call_range':
                        return True
            
            if context.get('call_func') == 'range':
                return True
        
        return False
    
    def _infer_loop_variable_types(self):
        """专门推断循环中变量的类型"""
        # 分析循环中的变量赋值
        class LoopVariableAnalyzer(ast.NodeVisitor):
            def __init__(self, variable_types):
                self.var_types = variable_types
                self.loop_contexts = []  # 当前循环上下文栈
                self.variable_modifications = defaultdict(list)  # 变量在循环中的修改记录
                
            def visit_For(self, node):
                # 进入循环
                self.loop_contexts.append(('for', node.lineno))
                
                # 分析循环变量
                if isinstance(node.target, ast.Name):
                    var_name = node.target.id
                    # 循环变量通常是整数
                    if var_name in self.var_types:
                        self._update_type(var_name, 'int', 'for_loop_index', 'high')
                
                # 分析循环体
                self.generic_visit(node)
                
                self.loop_contexts.pop()
                
            def visit_While(self, node):
                # 进入循环
                self.loop_contexts.append(('while', node.lineno))
                
                self.generic_visit(node)
                
                self.loop_contexts.pop()
                
            def visit_Assign(self, node):
                # 在循环中的赋值
                if self.loop_contexts:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            # 分析赋值表达式
                            expr_type = self._infer_expression_type(node.value)
                            if expr_type:
                                self.variable_modifications[var_name].append({
                                    'type': expr_type,
                                    'context': 'loop_assignment',
                                    'lineno': node.lineno
                                })
                                # 如果表达式中有浮点数运算,变量应该是浮点数
                                if self._has_float_operation(node.value):
                                    self._update_type(var_name, 'float', 'float_assignment_in_loop', 'high')
                                
                self.generic_visit(node)
                
            def visit_AugAssign(self, node):
                # 在循环中的增强赋值
                if self.loop_contexts and isinstance(node.target, ast.Name):
                    var_name = node.target.id
                    # 分析增强赋值
                    expr_type = self._infer_expression_type(node.value)
                    if expr_type:
                        self.variable_modifications[var_name].append({
                            'type': expr_type,
                            'context': 'aug_assignment_in_loop',
                            'lineno': node.lineno,
                            'op': type(node.op).__name__
                        })
                        
                        # 特殊处理:如果有浮点数运算
                        if self._has_float_operation(node.value):
                            self._update_type(var_name, 'float', 'float_aug_assignment_in_loop', 'high')
                        elif expr_type in ['int', 'float', 'numeric']:
                            # 数值类型的增强赋值
                            self._update_type(var_name, 'numeric', 'numeric_aug_assignment_in_loop', 'medium')
                
                self.generic_visit(node)
                
            def _infer_expression_type(self, node):
                """推断表达式类型"""
                if isinstance(node, ast.Constant):
                    value = node.value
                    if isinstance(value, int):
                        return 'int'
                    elif isinstance(value, float):
                        return 'float'
                    elif isinstance(value, str):
                        return 'str'
                    elif isinstance(value, bool):
                        return 'bool'
                        
                elif isinstance(node, ast.Name):
                    var_name = node.id
                    if var_name in self.var_types:
                        return self.var_types[var_name]['type']
                        
                elif isinstance(node, ast.BinOp):
                    left_type = self._infer_expression_type(node.left)
                    right_type = self._infer_expression_type(node.right)
                    
                    if left_type and right_type:
                        # 数值运算
                        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
                            # 如果有浮点数,结果为浮点数
                            if left_type == 'float' or right_type == 'float':
                                return 'float'
                            elif left_type == 'int' and right_type == 'int':
                                return 'int'
                            else:
                                return 'numeric'
                
                return None
                
            def _has_float_operation(self, node):
                """检查表达式是否包含浮点数运算"""
                if isinstance(node, ast.Constant):
                    return isinstance(node.value, float)
                elif isinstance(node, ast.Name):
                    var_name = node.id
                    # 检查变量名是否可能是浮点数
                    float_patterns = ['alp', 'ratio', 'rate', 'percent', 'float']
                    if any(pattern in var_name.lower() for pattern in float_patterns):
                        return True
                    # 检查变量类型
                    if var_name in self.var_types:
                        return self.var_types[var_name]['type'] == 'float'
                elif isinstance(node, ast.BinOp):
                    return (self._has_float_operation(node.left) or 
                            self._has_float_operation(node.right) or
                            isinstance(node.op, ast.Div))
                
                return False
                
            def _update_type(self, var_name, new_type, source, confidence):
                """更新变量类型"""
                if var_name in self.var_types:
                    current_info = self.var_types[var_name]
                    current_type = current_info['type']
                    current_conf = current_info.get('confidence', 'low')
                    
                    # 类型提升规则
                    type_hierarchy = {'Any': 0, 'numeric': 1, 'int': 2, 'float': 3}
                    current_rank = type_hierarchy.get(current_type, 0)
                    new_rank = type_hierarchy.get(new_type, 0)
                    
                    # 如果新类型更具体,则更新
                    if new_rank > current_rank or (new_rank == current_rank and confidence > current_conf):
                        current_info['type'] = new_type
                        current_info['confidence'] = confidence
                        
                        if 'sources' not in current_info:
                            current_info['sources'] = []
                        
                        current_info['sources'].append({
                            'source': source,
                            'type': new_type,
                            'confidence': confidence
                        })
        
        # 运行分析器
        analyzer = LoopVariableAnalyzer(self._variable_types)
        analyzer.visit(self.ast_tree)

    def _infer_from_usage_context(self):
        """从使用上下文推断类型"""
        # 分析整个AST树,收集变量的使用信息
        class UsageAnalyzer(ast.NodeVisitor):
            def __init__(self, variable_types, def_use_chains):
                self.var_types = variable_types
                self.def_use_chains = def_use_chains
                self._current_context = []
                self.usage_info = defaultdict(lambda: {
                    'in_numeric_ops': False,
                    'in_comparisons': False,
                    'in_loops': False,
                    'in_range_args': False,
                    'with_constants': [],
                    'operations': [],
                    'has_float_operation': False
                })
            
            def visit_BinOp(self, node):
                # 分析二元运算中的变量
                self._analyze_expression(node.left)
                self._analyze_expression(node.right)
                
                # 记录操作符类型
                op_name = type(node.op).__name__
                
                if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                    # 数值运算
                    if isinstance(node.left, ast.Name):
                        var_name = node.left.id
                        self.usage_info[var_name]['in_numeric_ops'] = True
                        self.usage_info[var_name]['operations'].append({
                            'type': 'binop',
                            'op': op_name
                        })
                        if isinstance(node.op, ast.Div):  # 除法可能产生浮点数
                            self.usage_info[var_name]['has_float_operation'] = True
                    
                    if isinstance(node.right, ast.Name):
                        var_name = node.right.id
                        self.usage_info[var_name]['in_numeric_ops'] = True
                        self.usage_info[var_name]['operations'].append({
                            'type': 'binop',
                            'op': op_name
                        })
                        if isinstance(node.op, ast.Div):  # 除法可能产生浮点数
                            self.usage_info[var_name]['has_float_operation'] = True
                    
                    # 检查是否与常量一起运算
                    if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Constant):
                        var_name = node.left.id
                        self.usage_info[var_name]['with_constants'].append({
                            'value': node.right.value,
                            'op': op_name,
                            'side': 'right'
                        })
                    elif isinstance(node.right, ast.Name) and isinstance(node.left, ast.Constant):
                        var_name = node.right.id
                        self.usage_info[var_name]['with_constants'].append({
                            'value': node.left.value,
                            'op': op_name,
                            'side': 'left'
                        })
                
                self.generic_visit(node)
        
            def visit_Compare(self, node):
                # 分析比较运算中的变量
                self._analyze_expression(node.left)
                for comparator in node.comparators:
                    self._analyze_expression(comparator)
                
                # 记录变量在比较中的使用
                if isinstance(node.left, ast.Name):
                    var_name = node.left.id
                    self.usage_info[var_name]['in_comparisons'] = True
                    self.usage_info[var_name]['operations'].append({
                        'type': 'compare',
                        'ops': [type(op).__name__ for op in node.ops]
                    })
                
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Name):
                        var_name = comparator.id
                        self.usage_info[var_name]['in_comparisons'] = True
                        self.usage_info[var_name]['operations'].append({
                            'type': 'compare',
                            'ops': [type(op).__name__ for op in node.ops]
                        })
                
                # 检查是否与数值常量比较
                for i, comparator in enumerate(node.comparators):
                    if isinstance(comparator, ast.Constant) and isinstance(node.left, ast.Name):
                        var_name = node.left.id
                        self.usage_info[var_name]['with_constants'].append({
                            'value': comparator.value,
                            'op': type(node.ops[i]).__name__ if i < len(node.ops) else None,
                            'side': 'right'
                        })
                
                self.generic_visit(node)

            def visit_Call(self, node):
                # 检查range函数调用
                if isinstance(node.func, ast.Name) and node.func.id == 'range':
                    for arg in node.args:
                        if isinstance(arg, ast.Name):
                            var_name = arg.id
                            self.usage_info[var_name]['in_range_args'] = True
                            self.usage_info[var_name]['in_loops'] = True
                
                self.generic_visit(node)
            
            def visit_For(self, node):
                # 记录循环上下文
                old_context = self._current_context.copy()
                self._current_context.append('for_loop')
                
                if isinstance(node.target, ast.Name):
                    var_name = node.target.id
                    self.usage_info[var_name]['in_loops'] = True
                
                # 分析循环中的变量使用
                self.generic_visit(node)
                
                self._current_context = old_context
            
            def visit_While(self, node):
                # 记录循环上下文
                old_context = self._current_context.copy()
                self._current_context.append('while_loop')
                
                self.generic_visit(node)
                
                self._current_context = old_context
            
            def _analyze_expression(self, expr):
                """分析表达式中的变量使用"""
                if isinstance(expr, ast.Name):
                    var_name = expr.id
                    # 记录变量在数值上下文中使用
                    if self._current_context and self._current_context[-1] in ['for_loop', 'while_loop']:
                        self.usage_info[var_name]['in_loops'] = True

        # 专门处理增强赋值中的变量
        class AugAssignContextAnalyzer(ast.NodeVisitor):
            def __init__(self, variable_types):
                self.var_types = variable_types
                self.aug_assignments = defaultdict(list)  # var -> [(op, expr, lineno)]
            
            def visit_AugAssign(self, node):
                if isinstance(node.target, ast.Name):
                    var_name = node.target.id
                    op_type = type(node.op).__name__
                    
                    try:
                        expr_str = ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                    except:
                        expr_str = str(node.value)
                    
                    self.aug_assignments[var_name].append({
                        'op': op_type,
                        'expr': expr_str,
                        'lineno': node.lineno,
                        'has_float_operation': self._has_float_operation(node.value)
                    })
                
                self.generic_visit(node)
            
            def _has_float_operation(self, expr_node):
                """检查表达式是否包含浮点数运算"""
                if isinstance(expr_node, ast.Constant):
                    return isinstance(expr_node.value, float)
                elif isinstance(expr_node, ast.Name):
                    var_name = expr_node.id
                    return var_name in ['alp']  # 已知的浮点变量
                elif isinstance(expr_node, ast.BinOp):
                    return (self._has_float_operation(expr_node.left) or 
                            self._has_float_operation(expr_node.right) or
                            isinstance(expr_node.op, ast.Div))  # 除法产生浮点数
                return False

        # 专门分析循环中的变量使用
        class LoopUsageAnalyzer(ast.NodeVisitor):
            def __init__(self, variable_types, ast_tree):
                self.var_types = variable_types
                self.ast_tree = ast_tree
                self.loop_depth = 0
                self.loop_variable_usage = defaultdict(lambda: {
                    'in_numeric_ops': False,
                    'in_float_ops': False,
                    'in_comparisons': False,
                    'as_loop_index': False,
                    'modified_in_loop': False,
                    'modification_count': 0
                })
                
            def visit_For(self, node):
                self.loop_depth += 1
                
                # 记录循环变量
                if isinstance(node.target, ast.Name):
                    var_name = node.target.id
                    self.loop_variable_usage[var_name]['as_loop_index'] = True
                
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_While(self, node):
                self.loop_depth += 1
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_Assign(self, node):
                if self.loop_depth > 0:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            self.loop_variable_usage[var_name]['modified_in_loop'] = True
                            self.loop_variable_usage[var_name]['modification_count'] += 1
                
                self.generic_visit(node)
                
            def visit_AugAssign(self, node):
                if self.loop_depth > 0 and isinstance(node.target, ast.Name):
                    var_name = node.target.id
                    self.loop_variable_usage[var_name]['modified_in_loop'] = True
                    self.loop_variable_usage[var_name]['modification_count'] += 1
                    
                    # 检查是否是数值运算
                    if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                        self.loop_variable_usage[var_name]['in_numeric_ops'] = True
                        
                        # 检查是否有浮点数运算
                        if self._has_float_in_expression(node.value):
                            self.loop_variable_usage[var_name]['in_float_ops'] = True
                
                self.generic_visit(node)
                
            def visit_BinOp(self, node):
                if self.loop_depth > 0:
                    # 分析二元运算中的变量
                    if isinstance(node.left, ast.Name):
                        var_name = node.left.id
                        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
                            self.loop_variable_usage[var_name]['in_numeric_ops'] = True
                            if self._has_float_in_expression(node.right):
                                self.loop_variable_usage[var_name]['in_float_ops'] = True
                    
                    if isinstance(node.right, ast.Name):
                        var_name = node.right.id
                        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
                            self.loop_variable_usage[var_name]['in_numeric_ops'] = True
                            if self._has_float_in_expression(node.left):
                                self.loop_variable_usage[var_name]['in_float_ops'] = True
                
                self.generic_visit(node)
                
            def _has_float_in_expression(self, node):
                """检查表达式中是否有浮点数"""
                if isinstance(node, ast.Constant):
                    return isinstance(node.value, float)
                elif isinstance(node, ast.Name):
                    var_name = node.id
                    return var_name in ['alp'] or (var_name in self.var_types and 
                                                self.var_types[var_name]['type'] == 'float')
                elif isinstance(node, ast.BinOp):
                    return (self._has_float_in_expression(node.left) or 
                            self._has_float_in_expression(node.right))
                return False
        
        # 运行循环使用分析器
        loop_analyzer = LoopUsageAnalyzer(self._variable_types, self.ast_tree)
        loop_analyzer.visit(self.ast_tree)
        
        # 基于循环使用信息更新类型
        for var_name, usage_info in loop_analyzer.loop_variable_usage.items():
            if var_name in self._variable_types:
                var_info = self._variable_types[var_name]
                
                # 如果变量在循环的数值运算中使用
                if usage_info['in_numeric_ops']:
                    if var_info['type'] in ['Any', 'unknown']:
                        if usage_info['in_float_ops']:
                            self._update_variable_type(var_name, 'float', 'float_operation_in_loop', 'high')
                        else:
                            self._update_variable_type(var_name, 'numeric', 'numeric_operation_in_loop', 'high')
                    elif var_info['type'] == 'numeric' and usage_info['in_float_ops']:
                        # 从 numeric 提升到 float
                        self._update_variable_type(var_name, 'float', 'numeric_to_float_in_loop', 'high')
                
                # 如果变量在循环中被多次修改
                if usage_info['modification_count'] > 0 and usage_info['in_numeric_ops']:
                    if var_info['type'] in ['Any', 'unknown']:
                        self._update_variable_type(var_name, 'numeric', 'modified_in_numeric_loop', 'high')
        
        analyzer = AugAssignContextAnalyzer(self._variable_types)
        analyzer.visit(self.ast_tree)
        
        # 基于增强赋值推断类型
        for var_name, assignments in analyzer.aug_assignments.items():
            if var_name in self._variable_types:
                var_info = self._variable_types[var_name]
                
                # 检查是否有涉及浮点数的增强赋值
                has_float_assign = any(assign['has_float_operation'] for assign in assignments)
                
                if has_float_assign and var_info['type'] in ['Any', 'int', 'numeric']:
                    # 提升到 float 类型
                    self._update_variable_type(
                        var_name, 
                        'float', 
                        'augmented_float_assignment', 
                        'high'
                    )
        
        # 执行使用分析
        analyzer = UsageAnalyzer(self._variable_types, self._build_def_use_chains())
        analyzer.visit(self.ast_tree)
        
        # 基于使用信息更新类型
        for var_name, usage_info in analyzer.usage_info.items():
            if var_name in self._variable_types:
                var_info = self._variable_types[var_name]
                
                # 如果变量在数值运算中使用,推断为数值类型
                if usage_info['in_numeric_ops']:
                    if var_info['type'] in ['Any', 'unknown']:
                        self._add_variable_type(var_name, 'numeric', 'numeric_usage', 'high')
                    elif var_info['type'] == 'int' and usage_info.get('has_float_operation', False):
                        self._add_variable_type(var_name, 'float', 'float_usage', 'medium')
                
                # 如果变量在比较中使用,推断为可比较类型(通常是数值)
                if usage_info['in_comparisons']:
                    if var_info['type'] in ['Any', 'unknown']:
                        self._add_variable_type(var_name, 'numeric', 'comparison_usage', 'high')
                
                # 如果变量在循环中使用,可能是整数(如索引)
                if usage_info['in_loops']:
                    if var_info['type'] in ['Any', 'unknown']:
                        self._add_variable_type(var_name, 'int', 'loop_usage', 'medium')

    def _infer_type_from_expression(self, expr_str):
        """从表达式字符串推断类型"""
        if not expr_str:
            return None
        
        # 处理 eval(input())
        if 'eval(input())' in expr_str:
            return 'Any'  # eval可以是任何类型
        
        # 处理 int(input()), float(input())
        if 'int(input())' in expr_str:
            return 'int'
        elif 'float(input())' in expr_str:
            return 'float'
        elif 'str(input())' in expr_str:
            return 'str'
        elif 'bool(input())' in expr_str:
            return 'bool'
        
        # 处理 input().split()
        if 'input().split()' in expr_str:
            return 'list[str]'
        
        # 处理 map(int, input().split())
        if 'map(int, input().split())' in expr_str:
            return 'list[int]'
        elif 'map(float, input().split())' in expr_str:
            return 'list[float]'
        
        # 处理 list(input().split())
        if 'list(input().split())' in expr_str:
            return 'list[str]'
        
        # 处理 list(map(int, input().split()))
        if 'list(map(int, input().split()))' in expr_str:
            return 'list[int]'
        elif 'list(map(float, input().split()))' in expr_str:
            return 'list[float]'
        
        # 处理数值常量
        if expr_str.replace('.', '').replace('-', '').isdigit():
            if '.' in expr_str:
                return 'float'
            else:
                return 'int'
        
        # 处理字符串常量
        if (expr_str.startswith("'") and expr_str.endswith("'")) or \
        (expr_str.startswith('"') and expr_str.endswith('"')):
            return 'str'
        
        # 处理布尔常量
        if expr_str in ['True', 'False']:
            return 'bool'
        
        # 处理列表常量
        if expr_str.startswith('[') and expr_str.endswith(']'):
            return 'list'
        
        # 处理字典常量
        if expr_str.startswith('{') and expr_str.endswith('}'):
            return 'dict'
        
        return None
    
    def _infer_assignment_types(self):
        """专门推断赋值关系的类型"""
        # 构建赋值关系图
        class AssignmentAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.assignments = []  # (target, source, lineno, in_loop)
                self.in_loop = False
                
            def visit_For(self, node):
                old_in_loop = self.in_loop
                self.in_loop = True
                self.generic_visit(node)
                self.in_loop = old_in_loop
                
            def visit_While(self, node):
                old_in_loop = self.in_loop
                self.in_loop = True
                self.generic_visit(node)
                self.in_loop = old_in_loop
                
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        target_var = target.id
                        
                        # 检查右侧是否为变量
                        if isinstance(node.value, ast.Name):
                            source_var = node.value.id
                            self.assignments.append((target_var, source_var, node.lineno, self.in_loop))
                        # 检查右侧是否为常量
                        elif isinstance(node.value, ast.Constant):
                            value = node.value.value
                            if isinstance(value, (int, float)):
                                self.assignments.append((target_var, str(value), node.lineno, self.in_loop))
                        # 检查右侧是否为复杂表达式
                        elif isinstance(node.value, ast.BinOp):
                            # 标记为数值表达式
                            self.assignments.append((target_var, 'numeric_expr', node.lineno, self.in_loop))
                
                self.generic_visit(node)
        
        # 分析赋值关系
        analyzer = AssignmentAnalyzer()
        analyzer.visit(self.ast_tree)
        
        # 传播赋值类型
        for target_var, source, lineno, in_loop in analyzer.assignments:
            if target_var in self._variable_types:
                target_info = self._variable_types[target_var]
                target_type = target_info['type']
                target_conf = target_info['confidence']
                
                # 如果目标类型不明确
                if target_type in ['Any', 'unknown'] or target_conf == 'low':
                    # 源是变量
                    if source in self._variable_types:
                        source_info = self._variable_types[source]
                        source_type = source_info['type']
                        source_conf = source_info['confidence']
                        
                        # 如果是循环中的赋值,增加置信度
                        if in_loop and source_type in ['int', 'float', 'numeric']:
                            source_conf = 'high' if source_conf == 'medium' else source_conf
                        
                        # 传播类型(如果源类型更明确)
                        if source_conf in ['high', 'medium'] and source_type not in ['Any', 'unknown']:
                            if target_type in ['Any', 'unknown'] or source_conf > target_conf:
                                self._update_variable_type(
                                    target_var,
                                    source_type,
                                    f'assignment_from_{source}' + ('_in_loop' if in_loop else ''),
                                    source_conf
                                )
                    
                    # 源是数值常量
                    elif source.replace('.', '').replace('-', '').isdigit():
                        # 确定常量类型
                        if '.' in source:
                            const_type = 'float'
                            const_conf = 'high'
                        else:
                            const_type = 'int'
                            const_conf = 'high'
                        
                        # 如果是循环中的数值常量赋值,变量应该是数值类型
                        if in_loop:
                            self._update_variable_type(
                                target_var,
                                const_type,
                                f'loop_assignment_from_constant_{source}',
                                const_conf
                            )
                    
                    # 源是数值表达式
                    elif source == 'numeric_expr' and in_loop:
                        self._update_variable_type(
                            target_var,
                            'numeric',
                            'numeric_expression_in_loop',
                            'medium'
                        )
    
    def _infer_numeric_types(self):
        """专门推断数值类型"""
        # 分析整个代码中的数值使用模式
        class NumericAnalyzer(ast.NodeVisitor):
            def __init__(self, variable_types, ast_tree):
                self.var_types = variable_types
                self.ast_tree = ast_tree
                self.numeric_usage = defaultdict(lambda: {
                    'modulo': [],
                    'range_args': [],
                    'comparisons': [],
                    'assignments': [],
                    'arithmetic': [],
                    'function_calls': []
                })
                self.var_definitions = {}
                
            def _find_definition(self, var_name, node):
                """查找变量的定义"""
                for child in ast.walk(self.ast_tree):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name) and target.id == var_name:
                                return child
                    elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name) and child.target.id == var_name:
                        return child
                return None
                
            def visit_Assign(self, node):
                # 记录赋值关系
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        self.var_definitions[var_name] = node
                        
                        # 检查是否为其他变量的赋值
                        if isinstance(node.value, ast.Name):
                            source_var = node.value.id
                            self.numeric_usage[var_name]['assignments'].append(('var_assignment', source_var))
                
                self.generic_visit(node)
                
            def visit_BinOp(self, node):
                # 检查取模运算
                if isinstance(node.op, ast.Mod):
                    left = node.left
                    right = node.right
                    
                    # 检查左侧是否为变量,右侧是否为整数常量
                    if isinstance(left, ast.Name) and isinstance(right, ast.Constant) and isinstance(right.value, int):
                        var_name = left.id
                        self.numeric_usage[var_name]['modulo'].append({
                            'divisor': right.value,
                            'lineno': node.lineno
                        })
                    
                    # 检查右侧是否为变量,左侧是否为整数常量
                    elif isinstance(right, ast.Name) and isinstance(left, ast.Constant) and isinstance(left.value, int):
                        var_name = right.id
                        self.numeric_usage[var_name]['modulo'].append({
                            'divisor': left.value,
                            'lineno': node.lineno
                        })
                
                # 检查算术运算
                if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)):
                    # 记录参与算术运算的变量
                    if isinstance(node.left, ast.Name):
                        var_name = node.left.id
                        self.numeric_usage[var_name]['arithmetic'].append({
                            'operator': type(node.op).__name__,
                            'right_type': 'variable' if isinstance(node.right, ast.Name) else 'constant',
                            'lineno': node.lineno
                        })
                    
                    if isinstance(node.right, ast.Name):
                        var_name = node.right.id
                        self.numeric_usage[var_name]['arithmetic'].append({
                            'operator': type(node.op).__name__,
                            'left_type': 'variable' if isinstance(node.left, ast.Name) else 'constant',
                            'lineno': node.lineno
                        })
                
                self.generic_visit(node)
                
            def visit_Compare(self, node):
                # 检查数值比较
                left = node.left
                for i, (comparator, op) in enumerate(zip(node.comparators, node.ops)):
                    # 检查是否为变量与数值的比较
                    if isinstance(left, ast.Name) and isinstance(comparator, ast.Constant):
                        var_name = left.id
                        if isinstance(comparator.value, (int, float)):
                            self.numeric_usage[var_name]['comparisons'].append({
                                'value': comparator.value,
                                'operator': type(op).__name__,
                                'side': 'right',
                                'lineno': node.lineno
                            })
                    
                    # 检查是否为数值与变量的比较
                    elif isinstance(comparator, ast.Name) and isinstance(left, ast.Constant):
                        var_name = comparator.id
                        if isinstance(left.value, (int, float)):
                            self.numeric_usage[var_name]['comparisons'].append({
                                'value': left.value,
                                'operator': type(op).__name__,
                                'side': 'left',
                                'lineno': node.lineno
                            })
                    
                    left = comparator
                
                self.generic_visit(node)
                
            def visit_Call(self, node):
                # 检查range函数参数
                if isinstance(node.func, ast.Name) and node.func.id == 'range':
                    for i, arg in enumerate(node.args):
                        if isinstance(arg, ast.Name):
                            var_name = arg.id
                            self.numeric_usage[var_name]['range_args'].append({
                                'position': i,
                                'lineno': node.lineno
                            })
                            self.numeric_usage[var_name]['function_calls'].append({
                                'function': 'range',
                                'position': i,
                                'lineno': node.lineno
                            })
                
                self.generic_visit(node)
        
        # 运行分析器
        analyzer = NumericAnalyzer(self._variable_types, self.ast_tree)
        analyzer.visit(self.ast_tree)
        
        # 根据分析结果更新类型
        for var_name, usages in analyzer.numeric_usage.items():
            if var_name in self._variable_types:
                var_info = self._variable_types[var_name]
                
                # 收集证据
                evidence = []
                
                # 1. 取模运算 -> 必须是整数
                if usages['modulo']:
                    evidence.append(('modulo_operation', 'int', 'high'))
                
                # 2. range函数参数 -> 必须是整数
                if usages['range_args']:
                    evidence.append(('range_argument', 'int', 'high'))
                
                # 3. 数值比较
                if usages['comparisons']:
                    # 检查是否有整数比较
                    int_comparisons = [c for c in usages['comparisons'] if isinstance(c['value'], int)]
                    if int_comparisons:
                        evidence.append(('integer_comparison', 'int', 'medium'))
                    
                    # 检查是否有浮点数比较
                    float_comparisons = [c for c in usages['comparisons'] if isinstance(c['value'], float)]
                    if float_comparisons and not usages['modulo'] and not usages['range_args']:
                        # 如果没有取模或range,可能是浮点数
                        evidence.append(('float_comparison', 'float', 'medium'))
                
                # 4. 算术运算
                if usages['arithmetic']:
                    # 检查是否包含除法
                    has_division = any(a['operator'] in ['Div', 'FloorDiv'] for a in usages['arithmetic'])
                    if has_division and not usages['modulo'] and not usages['range_args']:
                        evidence.append(('division_operation', 'float', 'medium'))
                    elif usages['arithmetic']:
                        evidence.append(('arithmetic_operation', 'numeric', 'medium'))
                
                # 5. 变量赋值链 - 增强这部分
                if usages['assignments']:
                    for assignment in usages['assignments']:
                        if assignment[0] == 'var_assignment':
                            source_var = assignment[1]
                            if source_var in self._variable_types:
                                source_info = self._variable_types[source_var]
                                source_type = source_info['type']
                                source_conf = source_info['confidence']
                                
                                # 如果源变量有明确类型,证据权重更高
                                if source_conf in ['high', 'medium'] and source_type != 'Any':
                                    evidence.append((f'strong_assignment_from_{source_var}', source_type, source_conf))
                                else:
                                    evidence.append((f'assignment_from_{source_var}', source_type, source_conf))
                
                # 根据证据推断类型
                if evidence:
                    # 按置信度和类型具体程度排序
                    def get_evidence_score(ev):
                        type_hierarchy = {'numeric': 0, 'int': 1, 'float': 2}
                        conf_score = {'high': 2, 'medium': 1, 'low': 0}
                        return (conf_score[ev[2]], type_hierarchy.get(ev[1], -1))
                    
                    evidence.sort(key=get_evidence_score, reverse=True)
                    
                    # 取最好的证据
                    best_evidence = evidence[0]
                    best_type = best_evidence[1]
                    best_confidence = best_evidence[2]
                    best_source = best_evidence[0]
                    
                    current_info = self._variable_types[var_name]
                    current_type = current_info['type']
                    current_conf = current_info.get('confidence', 'low')
                    
                    # 如果当前类型不明确,或者新证据更好
                    if (current_type in ['Any', 'unknown'] or 
                        best_confidence > current_conf or
                        (best_confidence == current_conf and best_type != 'Any')):
                        
                        self._update_variable_type(
                            var_name, 
                            best_type, 
                            f'numeric_inference_{best_source}', 
                            best_confidence
                        )

    def _infer_eval_input_types(self):
        """专门推断eval(input())的类型"""
        # 查找所有eval(input())调用
        class EvalInputFinder(ast.NodeVisitor):
            def __init__(self):
                self.eval_inputs = []  # (var_name, assign_node)
                
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # 检查右侧是否为eval(input())
                        if isinstance(node.value, ast.Call):
                            call = node.value
                            if isinstance(call.func, ast.Name) and call.func.id == 'eval':
                                if call.args and isinstance(call.args[0], ast.Call):
                                    inner_call = call.args[0]
                                    if isinstance(inner_call.func, ast.Name) and inner_call.func.id == 'input':
                                        self.eval_inputs.append((target.id, node))
                
                self.generic_visit(node)
        
        finder = EvalInputFinder()
        finder.visit(self.ast_tree)
        
        # 为每个eval(input())变量分析使用上下文
        for var_name, assign_node in finder.eval_inputs:
            # 查找变量的所有使用
            class UsageAnalyzer(ast.NodeVisitor):
                def __init__(self, target_var):
                    self.target_var = target_var
                    self.usages = []
                    
                def visit_Call(self, node):
                    # 检查max()/min()函数调用
                    if isinstance(node.func, ast.Name) and node.func.id in ['max', 'min']:
                        for arg in node.args:
                            if isinstance(arg, ast.Name) and arg.id == self.target_var:
                                self.usages.append((node.func.id, None, node.lineno))
                    
                    # 检查len()函数调用
                    if isinstance(node.func, ast.Name) and node.func.id == 'len':
                        for arg in node.args:
                            if isinstance(arg, ast.Name) and arg.id == self.target_var:
                                self.usages.append(('len', None, node.lineno))
                    
                    # 检查下标访问
                    if isinstance(node.func, ast.Name) and node.func.id in ['list', 'tuple', 'set', 'dict']:
                        for arg in node.args:
                            self._check_nested_usage(arg)
                    
                    self.generic_visit(node)
                
                def _check_nested_usage(self, expr):
                    """检查嵌套使用"""
                    if isinstance(expr, ast.Name) and expr.id == self.target_var:
                        self.usages.append(('iterable_conversion', None, getattr(expr, 'lineno', 0)))
                    elif isinstance(expr, ast.Call):
                        for arg in expr.args:
                            self._check_nested_usage(arg)
                
                def visit_Subscript(self, node):
                    # 检查下标访问
                    if isinstance(node.value, ast.Name) and node.value.id == self.target_var:
                        self.usages.append(('subscript_access', None, node.lineno))
                    
                    self.generic_visit(node)
                
                def visit_BinOp(self, node):
                    # 检查取模运算
                    if isinstance(node.op, ast.Mod):
                        if isinstance(node.left, ast.Name) and node.left.id == self.target_var:
                            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                                self.usages.append(('modulo', node.right.value, node.lineno))
                    
                    self.generic_visit(node)
                    
                def visit_Compare(self, node):
                    # 检查数值比较
                    left = node.left
                    for i, (comparator, op) in enumerate(zip(node.comparators, node.ops)):
                        if isinstance(left, ast.Name) and left.id == self.target_var:
                            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, (int, float)):
                                self.usages.append(('comparison', comparator.value, node.lineno))
                        
                        # 检查是否为数值与变量的比较
                        elif isinstance(comparator, ast.Name) and comparator.id == self.target_var:
                            if isinstance(left, ast.Constant) and isinstance(left.value, (int, float)):
                                self.usages.append(('comparison', left.value, node.lineno))
                        
                        left = comparator
                    
                    self.generic_visit(node)
            
            analyzer = UsageAnalyzer(var_name)
            analyzer.visit(self.ast_tree)
            
            # 根据使用推断类型
            inferred_type = 'Any'
            confidence = 'low'
            source = 'eval_input_context'
            
            if analyzer.usages:
                # 检查是否有max/min函数调用
                has_max_min = any(usage[0] in ['max', 'min'] for usage in analyzer.usages)
                has_len = any(usage[0] == 'len' for usage in analyzer.usages)
                has_subscript = any(usage[0] == 'subscript_access' for usage in analyzer.usages)
                has_iterable_conversion = any(usage[0] == 'iterable_conversion' for usage in analyzer.usages)
                has_list_method = any(usage[0] == 'list_method' for usage in analyzer.usages)
                
                # 检查是否有取模运算
                has_modulo = any(usage[0] == 'modulo' for usage in analyzer.usages)
                has_range_arg = any(usage[0] == 'range_arg' for usage in analyzer.usages)
                has_int_comparison = any(
                    usage[0] == 'comparison' and isinstance(usage[1], int) 
                    for usage in analyzer.usages
                )
                has_float_comparison = any(
                    usage[0] == 'comparison' and isinstance(usage[1], float)
                    for usage in analyzer.usages
                )
                
                if has_max_min:
                    # max/min要求可迭代且元素可比较,通常是数值列表
                    if has_len or has_subscript or has_iterable_conversion or has_list_method:
                        # 进一步检查比较类型
                        if has_int_comparison and not has_float_comparison:
                            inferred_type = 'list[int]'
                            confidence = 'high'
                            source = 'eval_input_with_max_min_int_context'
                        elif has_float_comparison:
                            inferred_type = 'list[float]'
                            confidence = 'high'
                            source = 'eval_input_with_max_min_float_context'
                        else:
                            # 检查是否有元素访问
                            if has_subscript:
                                # 如果有下标访问但没有具体数值比较,可能是泛型数值列表
                                inferred_type = 'list[numeric]'
                                confidence = 'high'  # 提升置信度
                                source = 'eval_input_with_max_min_and_subscript'
                            else:
                                inferred_type = 'list[numeric]'
                                confidence = 'medium'
                                source = 'eval_input_with_max_min_generic'
                
                elif has_modulo or has_range_arg:
                    inferred_type = 'int'
                    confidence = 'high'
                    source = 'eval_input_with_modulo_or_range'
                elif has_int_comparison and not has_float_comparison:
                    inferred_type = 'int'
                    confidence = 'medium'
                    source = 'eval_input_with_int_comparison'
                elif has_float_comparison:
                    inferred_type = 'float'
                    confidence = 'medium'
                    source = 'eval_input_with_float_comparison'
                elif analyzer.usages:
                    # 有其他使用但没有具体类型证据
                    inferred_type = 'numeric'
                    confidence = 'low'
                    source = 'eval_input_in_numeric_context'
            
            # 更新类型
            if var_name in self._variable_types:
                self._update_variable_type(var_name, inferred_type, source, confidence)
            else:
                self._add_variable_type(var_name, inferred_type, source, confidence)
    
    def _infer_type_from_name_pattern(self, var_name):
        """根据变量名模式推断类型"""
        patterns = {
            # 数值类型
            r'(num|count|total|sum|avg|score|price|amount|value|h|n|height|width|length|size|age|year)$': 'numeric',
            r'^(i|j|k|index|idx|n|m|p|q|r|s|t|u|v|x|y|z)$': 'int',
            r'(alp|alpha|ratio|rate|percent|percentage|probability|prob|chance)$': 'float',
            r'^h$': 'numeric',  # 常见表示高度
            r'^n$': 'int',      # 常见表示数量
            
            # 字符串类型
            r'(name|str|text|msg|description|title|content|string|s_|txt|str_|word|char)$': 'str',
            
            # 布尔类型
            r'(flag|is_|has_|can_|should_|enable|disable|valid|active|enabled|disabled)$': 'bool',
            
            # 列表类型
            r'(list|array|items|elements|values|results|vec|arr|collection|seq)$': 'list',
            r'^lst_': 'list',
            r'^arr_': 'list',
            r'^x$': 'list',     # 常见表示列表(如坐标)
            r'^a$': 'list',     # 常见表示列表
            r'^b$': 'list',     # 常见表示列表
            r'^nums$': 'list[int]',  # 常见表示整数列表
            r'^arr$': 'list',   # 常见表示数组
            r'^values$': 'list', # 常见表示值列表
            
            # 字典类型
            r'(dict|map|hash|table|hashmap|mapping)$': 'dict',
            r'^d_': 'dict',
            r'^map_': 'dict',
            
            # 集合类型
            r'(set|collection|group)$': 'set',
            
            # 文件类型
            r'(file|f|fp|handle|stream|io)$': 'file',
            
            # 特殊模式
            r'(sum\d*|total\d*|result\d*|output\d*)$': 'numeric',
            r'(temp|tmp|temporary|swap|buffer)$': 'Any',
        }
        
        import re
        for pattern, var_type in patterns.items():
            if re.search(pattern, var_name, re.IGNORECASE):
                return var_type
        
        return None

    def _propagate_related_types(self, def_use_chains):
        """传播相关变量的类型"""
        changed = False
        
        for var_name, var_info in self._variable_types.items():
            if var_info.get('confidence', 'low') != 'high' and var_name in def_use_chains:
                # 从定义中使用的变量推断类型
                for def_info in def_use_chains[var_name].get('definitions', []):
                    if def_info.get('type') == 'assignment':
                        value_expr = def_info.get('value_expr', '')
                        
                        # 检查是否直接赋值为其他变量
                        if value_expr in self._variable_types:
                            source_var = value_expr
                            source_info = self._variable_types[source_var]
                            
                            # 如果源变量有明确类型且置信度高
                            if source_info['confidence'] in ['high', 'medium'] and source_info['type'] != 'Any':
                                # 传播类型到目标变量
                                current_conf = var_info.get('confidence', 'low')
                                current_type = var_info['type']
                                
                                # 更新条件:
                                # 1. 当前类型不明确
                                # 2. 源类型置信度更高
                                # 3. 源类型更具体
                                if (current_type in ['Any', 'unknown'] or 
                                    source_info['confidence'] > current_conf or
                                    (source_info['type'] != 'Any' and current_type == 'Any')):
                                    
                                    self._update_variable_type(
                                        var_name,
                                        source_info['type'],
                                        f'assignment_from_{source_var}',
                                        source_info['confidence']
                                    )
                                    changed = True
                    
                    # 处理增强赋值
                    elif def_info.get('type') == 'augmented_assignment':
                        # 获取使用的变量
                        uses = def_info.get('uses', [])
                        if uses:
                            # 检查使用的变量是否有明确类型
                            for used_var in uses:
                                if used_var in self._variable_types:
                                    source_info = self._variable_types[used_var]
                                    if source_info['confidence'] in ['high', 'medium'] and source_info['type'] in ['int', 'float', 'numeric']:
                                        # 增强赋值通常保持数值类型
                                        current_conf = var_info.get('confidence', 'low')
                                        if current_conf in ['low', 'medium']:
                                            self._update_variable_type(
                                                var_name,
                                                source_info['type'],
                                                f'augmented_with_{used_var}',
                                                'medium'
                                            )
                                            changed = True
                                            break
        
        return changed


    def _update_variable_type(self, var_name, new_type, source, confidence='medium'):
        """更新变量类型"""
        if var_name not in self._variable_types:
            self._add_variable_type(var_name, new_type, source, confidence)
            return True
        else:
            current_info = self._variable_types[var_name]
            
            # 类型提升规则 - 数字越大表示类型越具体
            type_hierarchy = {
                'unknown': 0,
                'Any': 1,
                'numeric': 2,  # 泛型数值类型
                'bool': 3,
                'str': 4,
                'int': 5,      # int 比 numeric 更具体
                'float': 6,    # float 比 int 更具体(精度更高)
                'list': 7,
                'list[str]': 8,
                'list[numeric]': 9,
                'list[int]': 10,
                'list[float]': 11,
                'dict': 12,
                'tuple': 13
            }
            
            current_rank = type_hierarchy.get(current_info['type'], 0)
            new_rank = type_hierarchy.get(new_type, 0)
            
            # 置信度层次
            conf_hierarchy = {'low': 0, 'medium': 1, 'high': 2}
            current_conf = conf_hierarchy.get(current_info.get('confidence', 'low'), 0)
            new_conf = conf_hierarchy.get(confidence, 0)
            
            should_update = False
            
            # 更新规则:
            # 1. 新类型更具体且置信度不低于当前
            if new_rank > current_rank and new_conf >= current_conf:
                should_update = True
            # 2. 相同类型但置信度更高
            elif new_rank == current_rank and new_conf > current_conf:
                should_update = True
            # 3. 特殊来源(如取模,range使用)优先 - 即使置信度相同也更新
            elif source in ['modulo_usage', 'range_usage', 'numeric_context_inference', 
                        'eval_input_context', 'eval_input_with_modulo_or_range']:
                if new_type != 'Any' and new_conf >= current_conf:
                    should_update = True
            # 4. 从Any到具体类型 - 降低置信度要求
            elif current_info['type'] == 'Any' and new_type != 'Any':
                should_update = True
            # 5. 从numeric到int/float的提升
            elif current_info['type'] == 'numeric' and new_type in ['int', 'float']:
                should_update = True
            
            if should_update:
                current_info['type'] = new_type
                current_info['confidence'] = confidence
                
                # 记录来源
                if 'sources' not in current_info:
                    current_info['sources'] = []
                
                # 检查是否已有相同来源的类型
                source_exists = any(s['source'] == source for s in current_info['sources'])
                if not source_exists:
                    current_info['sources'].append({
                        'source': source,
                        'type': new_type,
                        'confidence': confidence
                    })
                
                return True
            else:
                # 即使不更新主类型,也记录来源
                if 'sources' not in current_info:
                    current_info['sources'] = []
                
                source_exists = any(s['source'] == source for s in current_info['sources'])
                if not source_exists:
                    current_info['sources'].append({
                        'source': source,
                        'type': new_type,
                        'confidence': confidence
                    })
            
            return False

    def _resolve_type_conflicts(self):
        """解决类型冲突"""
        for var_name, var_info in self._variable_types.items():
            sources = var_info.get('sources', [])
            
            if len(sources) > 1:
                # 收集所有推断的类型
                type_counts = {}
                type_scores = {}
                
                for source in sources:
                    t = source['type']
                    conf = source.get('confidence', 'medium')
                    source_type = source.get('source', '')
                    
                    # 跳过 Any 类型,除非它是唯一的类型
                    if t == 'Any':
                        continue
                    
                    # 根据来源类型分配权重
                    weight = 1
                    if conf == 'high':
                        weight = 3
                    elif conf == 'medium':
                        weight = 2
                    
                    # 特殊来源的额外权重
                    if source_type.endswith('_usage'):  # 使用上下文
                        weight += 2
                    elif source_type in ['assignment', 'parameter']:  # 直接定义
                        weight += 1
                    elif source_type == 'annotation':  # 类型注解
                        weight += 3
                    elif source_type in ['range_usage', 'modulo_usage', 'numeric_inference_range_argument']:
                        weight += 3  # range 和 modulo 是强类型指示器
                    
                    type_counts[t] = type_counts.get(t, 0) + 1
                    type_scores[t] = type_scores.get(t, 0) + weight
                
                # 选择权重最高的类型
                if type_scores:
                    best_type = max(type_scores.items(), key=lambda x: x[1])[0]
                    var_info['type'] = best_type
                    
                    # 更新置信度
                    max_weight = max(type_scores.values())
                    total_weight = sum(type_scores.values())
                    
                    if max_weight / total_weight > 0.7:
                        var_info['confidence'] = 'high'
                    elif max_weight / total_weight > 0.5:
                        var_info['confidence'] = 'medium'
                    else:
                        var_info['confidence'] = 'low'

    def _infer_missing_types(self):
        """推断缺失的类型"""
        # 获取所有变量
        def_use_chains = self._build_def_use_chains()
        
        for var_name in def_use_chains:
            if var_name not in self._variable_types:
                # 尝试根据变量名推断类型
                inferred_type = self._infer_type_from_name(var_name)
                if inferred_type:
                    self._add_variable_type(var_name, inferred_type, 'name_pattern', 'low')
                else:
                    # 默认类型
                    self._add_variable_type(var_name, 'Any', 'default', 'low')

    def _infer_type_from_name(self, var_name):
        """根据变量名推断类型"""
        # 常见的命名约定
        type_patterns = {
            # 数值类型
            r'(num|count|total|sum|avg|score|price|amount|value)$': 'numeric',
            r'^(i|j|k|index|idx)$': 'int',
            
            # 字符串类型
            r'(name|str|text|msg|description|title|content)$': 'str',
            r'^s_': 'str',
            
            # 布尔类型
            r'(flag|is_|has_|can_|should_|enable|disable)$': 'bool',
            
            # 列表类型
            r'(list|array|items|elements|values|results)$': 'list',
            r'^lst_': 'list',
            
            # 字典类型
            r'(dict|map|hash|table)$': 'dict',
            r'^d_': 'dict',
            
            # 文件类型
            r'(file|f|fp|handle)$': 'file',
        }
        
        import re
        for pattern, var_type in type_patterns.items():
            if re.search(pattern, var_name, re.IGNORECASE):
                return var_type
        
        return None

    def _add_variable_type(self, var_name, var_type, source, confidence):
        """添加变量类型信息"""
        if var_name not in self._variable_types:
            self._variable_types[var_name] = {
                'type': var_type,
                'sources': [{'source': source, 'type': var_type, 'confidence': confidence}],
                'confidence': confidence,
                'evidence': []
            }
        else:
            # 添加新的来源
            current_info = self._variable_types[var_name]
            current_info['sources'].append({
                'source': source,
                'type': var_type,
                'confidence': confidence
            })

class TypeInferencer(ast.NodeVisitor):
    """完整的类型推断器"""
    
    def __init__(self, variable_types, input_structure, def_use_chains, constants, ast_tree=None):
        self.types = variable_types
        self.input_structure = input_structure
        self.def_use_chains = def_use_chains
        self.constants = constants
        self.ast_tree = ast_tree
        self._current_function = None
        self._current_class = None
        self._scope_stack = []  # 作用域栈
        
        # 增强的内置函数返回类型映射
        self.builtin_return_types = {
            # 类型转换
            'int': 'int',
            'float': 'float',
            'str': 'str',
            'bool': 'bool',
            'bytes': 'bytes',
            'bytearray': 'bytearray',
            'list': 'list',
            'tuple': 'tuple',
            'dict': 'dict',
            'set': 'set',
            'frozenset': 'frozenset',
            
            # 数学函数
            'abs': 'numeric',
            'round': 'numeric',
            'len': 'int',
            'sum': 'numeric',
            'max': 'numeric',
            'min': 'numeric',
            'pow': 'numeric',
            
            # 输入输出
            'input': 'str',
            'open': '_io.TextIOWrapper',
            'print': 'None',
            
            # 类型检查
            'type': 'type',
            'isinstance': 'bool',
            'issubclass': 'bool',
            
            # 序列操作
            'range': 'range',
            'enumerate': 'enumerate',
            'zip': 'zip',
            'reversed': 'reversed',
            'sorted': 'list',
            
            # eval 和 exec
            'eval': 'Any',  # eval可以是任何类型
            'exec': 'None',
            
            # 其他
            'iter': 'iterator',
            'next': 'Any',
            'filter': 'filter',
            'map': 'map',
            'all': 'bool',
            'any': 'bool'
        }
        
        # 字符串方法返回类型
        self.str_method_returns = {
            'split': 'list[str]',
            'strip': 'str',
            'rstrip': 'str',
            'lstrip': 'str',
            'upper': 'str',
            'lower': 'str',
            'title': 'str',
            'capitalize': 'str',
            'replace': 'str',
            'find': 'int',
            'index': 'int',
            'count': 'int',
            'startswith': 'bool',
            'endswith': 'bool',
            'isdigit': 'bool',
            'isalpha': 'bool',
            'isalnum': 'bool',
            'islower': 'bool',
            'isupper': 'bool',
            'isspace': 'bool'
        }
        
        # 运算符返回类型
        self.operator_return_types = {
            ast.Add: self._infer_add_type,
            ast.Sub: 'numeric',
            ast.Mult: 'numeric',
            ast.Div: 'float',
            ast.FloorDiv: 'int',
            ast.Mod: 'numeric',
            ast.Pow: 'numeric',
            ast.LShift: 'int',
            ast.RShift: 'int',
            ast.BitOr: 'int',
            ast.BitAnd: 'int',
            ast.BitXor: 'int',
            ast.MatMult: 'numeric',
            
            # 比较运算符
            ast.Eq: 'bool',
            ast.NotEq: 'bool',
            ast.Lt: 'bool',
            ast.LtE: 'bool',
            ast.Gt: 'bool',
            ast.GtE: 'bool',
            ast.Is: 'bool',
            ast.IsNot: 'bool',
            ast.In: 'bool',
            ast.NotIn: 'bool',
            
            # 逻辑运算符
            ast.And: 'bool',
            ast.Or: 'bool',
            ast.Not: 'bool'
        }
    
    def _add_type(self, var_name, type_info, source, confidence='medium'):
        """添加类型信息"""
        if var_name not in self.types:
            self.types[var_name] = {
                'type': type_info,
                'sources': [],
                'confidence': confidence,
                'evidence': []
            }
        
        # 检查是否已有相同来源的类型
        existing_sources = [s['source'] for s in self.types[var_name]['sources']]
        if source not in existing_sources:
            self.types[var_name]['sources'].append({
                'source': source,
                'type': type_info,
                'confidence': confidence
            })
        
        # 更新证据
        self.types[var_name]['evidence'].append({
            'source': source,
            'type': type_info
        })
    
    def visit_FunctionDef(self, node):
        """处理函数定义中的类型推断"""
        old_function = self._current_function
        self._current_function = node.name
        self._scope_stack.append(('function', node.name))
        
        # 推断参数类型
        self._infer_parameter_types(node)
        
        # 分析函数体
        self.generic_visit(node)
        
        # 推断返回值类型
        self._infer_return_type(node)
        
        self._scope_stack.pop()
        self._current_function = old_function
    
    def _infer_parameter_types(self, node):
        """推断函数参数类型"""
        if not node.args:
            return
        
        # 位置参数
        for arg in node.args.args:
            if isinstance(arg, ast.arg):
                var_name = arg.arg
                
                # 1. 从注解推断
                if arg.annotation:
                    type_info = self._infer_type_from_annotation(arg.annotation)
                    if type_info:
                        self._add_type(var_name, type_info, 'annotation', 'high')
                
                # 2. 从默认值推断
                # 注意:这里无法获取默认值,需要在调用时推断
                else:
                    # 根据参数名推断
                    inferred_type = self._infer_type_from_name_pattern(var_name)
                    if inferred_type:
                        self._add_type(var_name, inferred_type, 'parameter_name', 'medium')
                    else:
                        self._add_type(var_name, 'Any', 'parameter_default', 'low')
        
        # 关键字参数
        if hasattr(node.args, 'kwonlyargs'):
            for arg in node.args.kwonlyargs:
                if isinstance(arg, ast.arg):
                    var_name = arg.arg
                    
                    if arg.annotation:
                        type_info = self._infer_type_from_annotation(arg.annotation)
                        if type_info:
                            self._add_type(var_name, type_info, 'annotation', 'high')
                    else:
                        inferred_type = self._infer_type_from_name_pattern(var_name)
                        if inferred_type:
                            self._add_type(var_name, inferred_type, 'parameter_name', 'medium')
    
    def _infer_return_type(self, node):
        """推断函数返回值类型"""
        # 1. 从返回类型注解推断
        if node.returns:
            return_type = self._infer_type_from_annotation(node.returns)
            if return_type:
                self._add_type(f'{node.name}_return', return_type, 'return_annotation', 'high')
                return
        
        # 2. 从函数体中的 return 语句推断
        return_types = []
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                ret_type = self._enhanced_infer_expression_type(stmt.value)
                if ret_type and ret_type['type'] != 'Any':
                    return_types.append(ret_type['type'])
        
        # 3. 合并所有返回类型
        if return_types:
            # 如果所有返回类型相同,使用该类型
            unique_types = list(set(return_types))
            if len(unique_types) == 1:
                self._add_type(f'{node.name}_return', unique_types[0], 'return_inference', 'medium')
            else:
                # 多种返回类型,使用联合类型或更通用的类型
                # 检查是否都是数值类型
                if all(t in ['int', 'float', 'numeric'] for t in unique_types):
                    self._add_type(f'{node.name}_return', 'numeric', 'return_inference', 'medium')
                # 检查是否包含 bool 和数值类型(常见模式)
                elif set(unique_types) <= {'int', 'float', 'numeric', 'bool'}:
                    # 使用联合类型表示
                    union_type = ' | '.join(sorted(unique_types))
                    self._add_type(f'{node.name}_return', union_type, 'return_inference', 'medium')
                else:
                    # 其他情况,记录所有可能的类型
                    union_type = ' | '.join(sorted(unique_types))
                    self._add_type(f'{node.name}_return', union_type, 'return_inference', 'low')
    
    def visit_ClassDef(self, node):
        """处理类定义"""
        old_class = self._current_class
        self._current_class = node.name
        self._scope_stack.append(('class', node.name))
        
        # 记录类名
        self._add_type(node.name, 'class', 'class_definition', 'high')
        
        self.generic_visit(node)
        
        self._scope_stack.pop()
        self._current_class = old_class
    
    def visit_Assign(self, node):
        """从赋值推断类型 - 增强版"""
        # 增强的右侧表达式类型推断
        rhs_type = self._enhanced_infer_expression_type(node.value)
        
        if rhs_type:
            # 处理多个目标
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    self._add_type(var_name, rhs_type['type'], 'assignment', rhs_type.get('confidence', 'medium'))
                    
                    # 记录赋值表达式供后续分析
                    if var_name in self.types:
                        if 'definitions' not in self.types[var_name]:
                            self.types[var_name]['definitions'] = []
                        
                        try:
                            expr_str = ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                        except:
                            expr_str = str(node.value)
                            
                        self.types[var_name]['definitions'].append({
                            'type': 'assignment',
                            'value_expr': expr_str,
                            'rhs_type': rhs_type['type'],
                            'lineno': node.lineno
                        })
                
                # 处理解包赋值:a, b = [1, 2]
                elif isinstance(target, (ast.Tuple, ast.List)):
                    self._infer_unpacking_type(target, node.value, rhs_type)
        
        self.generic_visit(node)
    
    def _enhanced_infer_expression_type(self, node):
        """增强的表达式类型推断"""
        if node is None:
            return None
        
        # 1. 处理复杂输入表达式
        if isinstance(node, ast.Call):
            return self._analyze_complex_input_expression(node)
        
        # 2. 处理常量
        if isinstance(node, ast.Constant):
            value = node.value
            # 注意:bool 是 int 的子类,所以必须先检查 bool
            if isinstance(value, bool):
                return {'type': 'bool', 'confidence': 'high', 'value': value}
            elif isinstance(value, int):
                return {'type': 'int', 'confidence': 'high', 'value': value}
            elif isinstance(value, float):
                return {'type': 'float', 'confidence': 'high', 'value': value}
            elif isinstance(value, str):
                return {'type': 'str', 'confidence': 'high', 'value': value}
            elif value is None:
                return {'type': 'None', 'confidence': 'high'}
            else:
                return {'type': type(value).__name__, 'confidence': 'high', 'value': value}
        
        # 3. 处理变量
        if isinstance(node, ast.Name):
            var_name = node.id
            if var_name in self.types:
                return {
                    'type': self.types[var_name]['type'],
                    'confidence': self.types[var_name]['confidence'],
                    'source': 'variable'
                }
            else:
                # 检查是否为内置常量
                builtin_constants = ['True', 'False', 'None']
                if var_name in builtin_constants:
                    return {'type': var_name if var_name != 'None' else 'None', 'confidence': 'high'}
                return {'type': 'Any', 'confidence': 'low', 'source': 'unknown_variable'}
        
        # 4. 处理二元运算
        if isinstance(node, ast.BinOp):
                left_type = self._enhanced_infer_expression_type(node.left)
                right_type = self._enhanced_infer_expression_type(node.right)
                
                if left_type and right_type:
                    # 根据运算符推断
                    if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                        return self._infer_numeric_operation_type(left_type, right_type, node.op)
                
                # 其他情况返回默认
                return {'type': 'Any', 'confidence': 'low'}
        
        # 5. 处理比较运算
        if isinstance(node, ast.Compare):
            # 比较总是返回布尔值
            return {'type': 'bool', 'confidence': 'high'}
        
        # 6. 处理列表
        if isinstance(node, ast.List):
            if node.elts:
                element_types = []
                for elt in node.elts:
                    elt_type = self._enhanced_infer_expression_type(elt)
                    if elt_type:
                        element_types.append(elt_type['type'])
                
                if element_types:
                    if len(set(element_types)) == 1:
                        return {'type': f'list[{element_types[0]}]', 'confidence': 'medium'}
                    else:
                        return {'type': 'list', 'confidence': 'medium'}
            
            return {'type': 'list', 'confidence': 'medium'}
        
        # 7. 处理字典
        if isinstance(node, ast.Dict):
            return {'type': 'dict', 'confidence': 'medium'}
        
        # 8. 处理元组
        if isinstance(node, ast.Tuple):
            return {'type': 'tuple', 'confidence': 'medium'}
        
        # 9. 处理下标访问
        if isinstance(node, ast.Subscript):
            value_type = self._enhanced_infer_expression_type(node.value)
            if value_type and value_type['type'].startswith('list['):
                # list[T] 的下标访问返回 T
                return {'type': value_type['type'][5:-1], 'confidence': 'medium'}
            elif value_type and value_type['type'] in ['list', 'tuple', 'dict']:
                return {'type': 'Any', 'confidence': 'low'}
        
        # 10. 处理属性访问
        if isinstance(node, ast.Attribute):
            # 常见字符串方法
            if isinstance(node.value, ast.Name):
                obj_type = self._enhanced_infer_expression_type(node.value)
                if obj_type and obj_type['type'] == 'str' and node.attr in self.str_method_returns:
                    return {'type': self.str_method_returns[node.attr], 'confidence': 'high'}
        
        # 默认返回
        return {'type': 'Any', 'confidence': 'low', 'source': 'unknown_expression'}
    
    def _infer_numeric_operation_type(self, left_type, right_type, op):
        """推断数值运算的类型 - 增强版"""
        left_t = left_type['type']
        right_t = right_type['type']
        
        # 数值类型优先级:int < float < numeric
        type_priority = {
            'int': 1,
            'float': 2,
            'numeric': 3,
            'Any': 0
        }
        
        left_priority = type_priority.get(left_t, 0)
        right_priority = type_priority.get(right_t, 0)
        
        # 取优先级更高的类型
        if left_priority > right_priority:
            result_type = left_t
        elif right_priority > left_priority:
            result_type = right_t
        else:
            result_type = left_t  # 优先级相同,取左边
        
        # 特殊情况处理
        if left_t == 'Any' or right_t == 'Any':
            # 如果任一操作数是 Any,尝试推断
            if left_t != 'Any':
                result_type = left_t
            elif right_t != 'Any':
                result_type = right_t
            else:
                result_type = 'Any'
        
        # 特殊处理除法:除法和真除法产生浮点数
        if isinstance(op, ast.Div):
            result_type = 'float'
        
        # 增强的数值推断规则
        if result_type == 'numeric' and (left_t == 'float' or right_t == 'float'):
            result_type = 'float'
        elif result_type == 'numeric':
            result_type = 'int'  # 默认数值为整数
        
        # 计算置信度
        confidence = min(left_type.get('confidence', 'low'), right_type.get('confidence', 'low'))
        
        # 特殊规则:如果运算是增强赋值的一部分,需要考虑之前的类型
        if isinstance(op, (ast.Add, ast.Sub, ast.Mult)):
            # 如果是增强赋值,需要保持类型一致性
            if left_t == right_t and left_t in ['int', 'float']:
                result_type = left_t
        
        return {'type': result_type, 'confidence': confidence}
    
    def _analyze_complex_input_expression(self, node):
        """分析复杂的输入表达式"""
        if not isinstance(node, ast.Call):
            return None
        
        # 1. eval(input())
        if isinstance(node.func, ast.Name) and node.func.id == 'eval':
            # 检查eval的参数
            if node.args and isinstance(node.args[0], ast.Call):
                arg_call = node.args[0]
                if isinstance(arg_call.func, ast.Name) and arg_call.func.id == 'input':
                    # eval(input())
                    return {
                        'type': 'Any',  # 暂时保持Any
                        'confidence': 'low',
                        'source': 'eval_input',
                        'inference_hints': {
                            'can_be_numeric': True,
                            'can_be_int': True,
                            'can_be_float': True,
                            'can_be_str': True,
                            'can_be_list': True,
                            'can_be_iterable': True
                        },
                        'needs_context_inference': True  # 标记需要上下文推断
                    }
        
        # 2. int(input()), float(input())
        if isinstance(node.func, ast.Name) and node.func.id in ['int', 'float']:
            if node.args and isinstance(node.args[0], ast.Call):
                arg_call = node.args[0]
                if isinstance(arg_call.func, ast.Name) and arg_call.func.id == 'input':
                    # int(input()) 或 float(input())
                    return {
                        'type': node.func.id, 
                        'confidence': 'high', 
                        'source': f'{node.func.id}_input'
                    }
        
        # 3. input().split()
        if isinstance(node.func, ast.Attribute):
            # 获取对象类型
            obj_type = self._enhanced_infer_expression_type(node.func.value)
            if obj_type and obj_type['type'] == 'str':
                method_name = node.func.attr
                if method_name == 'split':
                    return {'type': 'list[str]', 'confidence': 'high', 'source': 'split_input'}
                elif method_name in self.str_method_returns:
                    return {'type': self.str_method_returns[method_name], 'confidence': 'high', 'source': 'str_method'}
        
        # 4. map(int, input().split())
        if isinstance(node.func, ast.Name) and node.func.id == 'map':
            if len(node.args) >= 2:
                # 第一个参数是转换函数
                func_arg = node.args[0]
                # 第二个参数通常是 input().split()
                iter_arg = node.args[1]
                
                if isinstance(func_arg, ast.Name) and func_arg.id in ['int', 'float']:
                    iter_type = self._enhanced_infer_expression_type(iter_arg)
                    if iter_type and iter_type['type'] == 'list[str]':
                        return {
                            'type': f'list[{func_arg.id}]',
                            'confidence': 'high',
                            'source': 'mapped_input'
                        }
        
        # 5. 普通的input()
        if isinstance(node.func, ast.Name) and node.func.id == 'input':
            return {'type': 'str', 'confidence': 'high', 'source': 'input'}
        
        # 6. 其他函数调用
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.builtin_return_types:
                return {
                    'type': self.builtin_return_types[func_name],
                    'confidence': 'high',
                    'source': f'builtin_{func_name}'
                }
        
        # 7. 方法调用
        if isinstance(node.func, ast.Attribute):
            obj_type = self._enhanced_infer_expression_type(node.func.value)
            if obj_type:
                method_name = node.func.attr
                # 字符串方法
                if obj_type['type'] == 'str' and method_name in self.str_method_returns:
                    return {
                        'type': self.str_method_returns[method_name],
                        'confidence': 'high',
                        'source': 'str_method'
                    }
                # 列表方法
                elif obj_type['type'].startswith('list'):
                    if method_name in ['append', 'extend', 'insert', 'remove', 'sort', 'reverse']:
                        return {'type': 'None', 'confidence': 'medium', 'source': 'list_method'}
                    elif method_name == 'pop':
                        if obj_type['type'].startswith('list['):
                            return {'type': obj_type['type'][5:-1], 'confidence': 'medium', 'source': 'list_method'}
                        else:
                            return {'type': 'Any', 'confidence': 'low', 'source': 'list_method'}
                        
        # 8. max()/min() 函数
        if isinstance(node.func, ast.Name) and node.func.id in ['max', 'min']:
            if node.args:
                # 分析参数的第一个元素
                first_arg = node.args[0]
                arg_type = self._enhanced_infer_expression_type(first_arg)
                
                if arg_type:
                    # 如果参数是列表类型
                    if arg_type['type'].startswith('list['):
                        # max(list[T]) 返回 T
                        return {
                            'type': arg_type['type'][5:-1],  # 提取T
                            'confidence': arg_type.get('confidence', 'medium'),
                            'source': f'{node.func.id}_function'
                        }
                    elif arg_type['type'] in ['list', 'tuple', 'set', 'range']:
                        # 通用可迭代对象
                        return {
                            'type': 'numeric',  # 通常是数值
                            'confidence': 'medium',
                            'source': f'{node.func.id}_function'
                        }
                    else:
                        # 其他类型
                        return {
                            'type': arg_type['type'],
                            'confidence': arg_type.get('confidence', 'low'),
                            'source': f'{node.func.id}_function'
                        }
        
        # 9. 用户自定义函数调用 - 查找函数返回类型
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            return_type_key = f'{func_name}_return'
            
            # 查找该函数的返回类型
            if return_type_key in self.types:
                func_return_type = self.types[return_type_key].get('type', 'Any')
                func_confidence = self.types[return_type_key].get('confidence', 'low')
                return {
                    'type': func_return_type,
                    'confidence': func_confidence,
                    'source': f'user_function_{func_name}'
                }
        
        return {'type': 'Any', 'confidence': 'low', 'source': 'unknown_call'}
        
    
    def _infer_unpacking_type(self, target, value, rhs_type):
        """推断解包赋值的类型"""
        if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
            for i, (tgt, elt) in enumerate(zip(target.elts, value.elts)):
                if isinstance(tgt, ast.Name):
                    elt_type = self._enhanced_infer_expression_type(elt)
                    if elt_type:
                        self._add_type(tgt.id, elt_type['type'], 'unpacking_assignment', elt_type.get('confidence', 'medium'))
        
        # 处理 *args 解包
        elif isinstance(value, ast.Starred):
            # 星号解包,类型为列表元素类型
            inner_type = self._enhanced_infer_expression_type(value.value)
            if inner_type and inner_type['type'].startswith('list['):
                elem_type = inner_type['type'][5:-1]
                for tgt in target.elts:
                    if isinstance(tgt, ast.Name):
                        self._add_type(tgt.id, elem_type, 'starred_unpacking', 'low')
    
    def visit_AnnAssign(self, node):
        """处理带注解的赋值"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            
            # 从注解推断
            if node.annotation:
                type_info = self._infer_type_from_annotation(node.annotation)
                if type_info:
                    self._add_type(var_name, type_info, 'type_annotation', 'high')
            
            # 从值推断
            if node.value:
                rhs_type = self._enhanced_infer_expression_type(node.value)
                if rhs_type:
                    self._add_type(var_name, rhs_type['type'], 'annotated_assignment', rhs_type.get('confidence', 'medium'))
        
        self.generic_visit(node)
    
    def visit_AugAssign(self, node):
        """处理增强赋值"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            
            # 获取变量当前类型
            current_type = self.types.get(var_name, {}).get('type')
            
            # 根据操作符推断新类型
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                # 数值运算,保持数值类型
                new_type = 'numeric' if current_type in ['int', 'float', 'numeric'] else 'Any'
                self._add_type(var_name, new_type, 'augmented_assignment', 'medium')
            
            elif isinstance(node.op, (ast.BitOr, ast.BitAnd, ast.BitXor, ast.LShift, ast.RShift)):
                # 位运算,保持整数类型
                self._add_type(var_name, 'int', 'augmented_assignment', 'medium')
        
        self.generic_visit(node)
    
    def visit_For(self, node):
        """推断循环变量类型"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            
            # 推断迭代对象的类型
            iter_type = self._enhanced_infer_expression_type(node.iter)
            
            if iter_type:
                iter_type_str = iter_type['type']
                
                # 根据迭代类型推断循环变量类型
                if iter_type_str == 'range':
                    loop_var_type = 'int'
                elif iter_type_str.startswith('list['):
                    # list[T] 的循环变量类型为 T
                    loop_var_type = iter_type_str[5:-1]
                elif iter_type_str in ['list', 'tuple', 'set']:
                    loop_var_type = 'Any'
                elif iter_type_str == 'str':
                    loop_var_type = 'str'
                elif iter_type_str == 'dict':
                    loop_var_type = 'tuple'  # dict.items() 返回 (key, value)
                else:
                    loop_var_type = 'Any'
                
                self._add_type(var_name, loop_var_type, 'for_loop', iter_type.get('confidence', 'medium'))
        
        # 处理循环体
        self.generic_visit(node)
    
    def visit_With(self, node):
        """推断上下文管理器变量类型"""
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                var_name = item.optional_vars.id
                
                # 推断上下文表达式类型
                context_type = self._enhanced_infer_expression_type(item.context_expr)
                
                if context_type:
                    # 常见的上下文管理器类型
                    ctx_type = context_type['type']
                    if ctx_type == '_io.TextIOWrapper':
                        var_type = '_io.TextIOWrapper'
                    elif 'open' in str(item.context_expr):
                        var_type = '_io.TextIOWrapper'
                    else:
                        var_type = 'Any'
                    
                    self._add_type(var_name, var_type, 'with_statement', context_type.get('confidence', 'medium'))
        
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        """推断异常变量类型"""
        if node.name:
            var_name = node.name
            
            # 异常类型
            if node.type:
                type_info = self._infer_type_from_annotation(node.type)
                if type_info:
                    self._add_type(var_name, type_info, 'exception_handler', 'high')
                else:
                    self._add_type(var_name, 'BaseException', 'exception_handler', 'medium')
            else:
                self._add_type(var_name, 'BaseException', 'exception_handler', 'medium')
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """推断函数调用返回类型"""
        # 增强的函数调用类型推断
        return_type = self._analyze_complex_input_expression(node)
        
        if return_type and return_type['type'] != 'Any':
            # 记录函数调用的返回类型,但这里不直接赋值给变量
            # 变量类型会在Assign节点中处理
            pass
        
        self.generic_visit(node)
        return return_type
    
    def _infer_add_type(self, left_type, right_type):
        """推断加法运算的类型"""
        left_t = left_type['type']
        right_t = right_type['type']
        
        # 字符串连接
        if left_t == 'str' and right_t == 'str':
            return {'type': 'str', 'confidence': 'high'}
        
        # 数值相加
        if left_t in ['int', 'float', 'numeric'] and right_t in ['int', 'float', 'numeric']:
            # int + int = int, float + 任何数值 = float
            if left_t == 'float' or right_t == 'float':
                return {'type': 'float', 'confidence': 'high'}
            else:
                return {'type': 'int', 'confidence': 'high'}
        
        # 列表连接
        if (left_t.startswith('list[') or left_t == 'list') and (right_t.startswith('list[') or right_t == 'list'):
            # 如果列表元素类型相同
            if left_t == right_t:
                return {'type': left_t, 'confidence': 'medium'}
            else:
                return {'type': 'list', 'confidence': 'low'}
        
        return {'type': 'Any', 'confidence': 'low'}
    
    def _infer_type_from_name_pattern(self, var_name):
        """根据变量名模式推断类型"""
        patterns = {
            # 数值类型
            r'(num|count|total|sum|avg|score|price|amount|value|h|n|height|width|length|size|age|year)$': 'numeric',
            r'^(i|j|k|index|idx|n|m|p|q|r|s|t|u|v|x|y|z)$': 'int',
            r'(alp|alpha|ratio|rate|percent|percentage|probability|prob|chance)$': 'float',
            r'^h$': 'numeric',  # 常见表示高度
            r'^n$': 'int',      # 常见表示数量
            
            # 字符串类型
            r'(name|str|text|msg|description|title|content|string|s_|txt|str_|word|char)$': 'str',
            
            # 布尔类型
            r'(flag|is_|has_|can_|should_|enable|disable|valid|active|enabled|disabled)$': 'bool',
            
            # 列表类型
            r'(list|array|items|elements|values|results|vec|arr|collection|seq)$': 'list',
            r'^lst_': 'list',
            r'^arr_': 'list',
            r'^x$': 'list',     # 常见表示列表(如坐标)
            r'^a$': 'list',     # 常见表示列表
            r'^b$': 'list',     # 常见表示列表
            r'^nums$': 'list[int]',  # 常见表示整数列表
            r'^arr$': 'list',   # 常见表示数组
            r'^values$': 'list', # 常见表示值列表
            
            # 字典类型
            r'(dict|map|hash|table|hashmap|mapping)$': 'dict',
            r'^d_': 'dict',
            r'^map_': 'dict',
            
            # 集合类型
            r'(set|collection|group)$': 'set',
            
            # 文件类型
            r'(file|f|fp|handle|stream|io)$': 'file',
            
            # 特殊模式
            r'(sum\d*|total\d*|result\d*|output\d*)$': 'numeric',
            r'(temp|tmp|temporary|swap|buffer)$': 'Any',
        }
        
        import re
        for pattern, var_type in patterns.items():
            if re.search(pattern, var_name, re.IGNORECASE):
                return var_type
        
        return None
    
    def _infer_type_from_annotation(self, annotation):
        """从类型注解推断类型"""
        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Subscript):
                # 处理泛型
                base = self._infer_type_from_annotation(annotation.value)
                slice_info = annotation.slice
                
                if isinstance(slice_info, ast.Name):
                    inner = slice_info.id
                elif isinstance(slice_info, ast.Constant):
                    inner = str(slice_info.value)
                elif isinstance(slice_info, ast.Tuple):
                    # 多参数泛型,如 Dict[str, int]
                    elts = []
                    for elt in slice_info.elts:
                        elt_type = self._infer_type_from_annotation(elt)
                        if elt_type:
                            elts.append(elt_type)
                    inner = ', '.join(elts)
                else:
                    inner = 'Any'
                
                return f"{base}[{inner}]"
            elif isinstance(annotation, ast.Constant):
                return str(annotation.value)
            elif hasattr(annotation, '_fields'):
                # 尝试解构
                try:
                    return ast.unparse(annotation)
                except:
                    return str(annotation)
            else:
                return None
        except Exception:
            return None
        
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