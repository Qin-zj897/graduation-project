import ast
import inspect
import json
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
        
    def _format_response(self, success: bool, message: str, data: Any = None) -> Dict[str, Any]:
        """统一格式化响应为JSON格式"""
        return {
            "success": success,
            "message": message,
            "data": data if data is not None else {}
        }
        
    def extract_constants_and_comparisons(self) -> Dict[str, Any]:
        """
        常量与比较值提取（增强版）
        - 提取所有常量值和比较表达式的值
        
        Returns:
            JSON格式响应: {"message": "...", "data": {...}}
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
                    
                    # 处理函数调用中的常量（如 x = len(y)）
                    elif isinstance(node.value, ast.Call):
                        self._extract_call_constants(node.value, node.targets, node.lineno)
                    
                    # 处理下标访问中的常量（如 x = y[0]）
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
                    """提取下标访问中的常量（增强版）"""
                    # 处理下标索引中的常量（如 a[0]）
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
                    
                    # 处理切片操作中的常量（如 a[1:5:2]）
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
                    
                    # 处理扩展切片（如 a[1:5, 2]）
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
                    """处理比较表达式，提取比较值（增强版）"""
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
                    
                    # 处理函数调用与常量的比较（如 len(x) == 0）
                    elif isinstance(node.left, ast.Call):
                        self._extract_call_comparison(node.left, node.comparators, node.ops, node.lineno)
                    
                    # 处理二元运算与常量的比较（如 x % 2 == 1, x + y == 10）
                    elif isinstance(node.left, ast.BinOp):
                        self._extract_binop_comparison(node.left, node.comparators, node.ops, node.lineno)
                    
                    # 处理下标访问与常量的比较（如 a[0] == 5）
                    elif isinstance(node.left, ast.Subscript):
                        self._extract_subscript_comparison(node.left, node.comparators, node.ops, node.lineno)
                    
                    self.generic_visit(node)
                
                def _extract_call_comparison(self, call_node, comparators, ops, lineno):
                    """提取函数调用与常量的比较（如 len(x) == 0）"""
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
                    """提取二元运算与常量的比较（如 x % 2 == 1, x + y == 10）"""
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
                    
                    # 特殊处理取模运算（x % 2 == 1）
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
                    """提取下标访问与常量的比较（如 a[0] == 5）"""
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
                            
                            # 提取下标索引中的常量（如果存在）
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
            
            # 去重并排序
            for var in self._constants:
                unique_values = []
                seen = set()
                for item in self._constants[var]:
                    key = (item['value'], item['source'], item.get('op', ''))
                    if key not in seen:
                        seen.add(key)
                        unique_values.append(item)
                self._constants[var] = unique_values
            
            # 统计信息
            stats = {
                'total_variables': len(self._constants),
                'total_constants': sum(len(v) for v in self._constants.values()),
                'variables_by_type': {},
                'most_common_constants': []
            }
            
            # 按类型统计
            type_counts = defaultdict(int)
            for var, const_list in self._constants.items():
                for const in const_list:
                    type_counts[const['type']] += 1
            
            stats['variables_by_type'] = dict(type_counts)
            
            # 找出最常见的常量值
            value_counts = defaultdict(int)
            for var, const_list in self._constants.items():
                for const in const_list:
                    if isinstance(const['value'], (int, float, str)):
                        value_counts[const['value']] += 1
            
            if value_counts:
                most_common = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                stats['most_common_constants'] = most_common
            
            return self._format_response(
                success=True,
                message="常量提取完成",
                data={
                    'constants': dict(self._constants),
                    'statistics': stats,
                    'metadata': {
                        'source_code_length': len(self.source_code),
                        'variables_count': len(self._constants)
                    }
                }
            )
            
        except Exception as e:
            return self._format_response(
                success=False,
                message=f"常量提取失败: {str(e)}",
                data={'error': str(e)}
            )
    
    def build_control_flow_graph(self) -> Dict[str, Any]:
        """
        构建控制流图(CFG) 
        
        Returns:
            JSON格式响应: {"message": "...", "data": {...}}
        """
        try:
            self._cfg = nx.DiGraph()
            
            class CFGBuilder(ast.NodeVisitor):
                def __init__(self, cfg_graph):
                    self.cfg = cfg_graph
                    self.node_counter = 0
                    self.current_block = None
                    self.block_stack = []  # 用于嵌套结构
                    self.loop_stack = []   # 专门用于循环的栈：(header, exit)
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
                    
                    # 保存旧的当前块，进入函数体
                    old_block = self.current_block
                    self.current_block = entry_block
                    
                    # 推入函数上下文
                    self.block_stack.append(('function', exit_block))
                    
                    # 处理函数体
                    for stmt in node.body:
                        self.visit(stmt)
                    
                    # 如果函数体末尾没有显式return，连接到退出块
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
                    
                    return entry_block
                
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
                    
                    # true分支处理完后，如果没有break/continue/return，连接到合并块
                    if self.current_block and self._can_fall_through():
                        self._connect_blocks(self.current_block, merge_block, label="true_end")
                    
                    # 处理false分支（else或elif）
                    self.current_block = false_block
                    self.block_stack[-1] = ('if', true_block, false_block, merge_block)  # 更新栈
                    for stmt in node.orelse:
                        self.visit(stmt)
                    
                    # false分支处理完后，连接到合并块
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
                    
                    # 连接循环头到循环体（条件为真）
                    self._connect_blocks(loop_header, loop_body_start, label="true")
                    
                    # 连接循环头到退出或else（条件为假）
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
                    
                    # 循环体结束后，如果没有break/return，连接到循环头（继续迭代）
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
                    
                    # 连接循环头到循环体（有下一个元素）
                    self._connect_blocks(loop_header, loop_body_start, label="iterate")
                    
                    # 连接循环头到退出或else（迭代结束）
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
                    
                    # 循环体结束后，如果没有break/return，连接到循环头（继续迭代）
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
                        # 如果没有找到函数上下文，创建临时的退出块
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
                    
                    # try块结束后，如果没有异常，连接到退出块
                    if self.current_block and self._can_fall_through():
                        self._connect_blocks(self.current_block, try_exit, label="no_exception")
                    
                    # 处理except块
                    for handler in node.handlers:
                        except_block = self._new_block("except_block")
                        # 连接try块到except块（异常发生）
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
                    """处理函数调用，建立调用关系"""
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
                        
                        # 如果被调用的函数也在同一代码中，建立调用边
                        if func_name in self.function_blocks:
                            callee_entry, callee_exit = self.function_blocks[func_name]
                            self.call_edges.append({
                                'caller': self.current_block,
                                'callee': callee_entry,
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
                    """检查当前块是否可以fall through（没有终止语句）"""
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
            
            # 先遍历所有函数定义，建立函数块映射
            for node in ast.walk(self.ast_tree):
                if isinstance(node, ast.FunctionDef):
                    builder.visit_FunctionDef(node)
            
            # 再次遍历，建立主程序的CFG
            builder.current_block = None
            builder.visit(self.ast_tree)
            
            # 添加函数调用边到图中
            for call_edge in builder.call_edges:
                self._cfg.add_edge(
                    call_edge['caller'],
                    call_edge['callee'],
                    label='call',
                    type='call_edge',
                    lineno=call_edge['lineno']
                )
                # 添加返回边（从被调用函数退出块返回到调用点）
                callee_exit = builder.function_blocks.get(call_edge['callee'].split('_')[0], (None, None))[1]
                if callee_exit:
                    self._cfg.add_edge(
                        callee_exit,
                        call_edge['return_to'],
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
            
            return self._format_response(
                success=True,
                message="控制流图构建完成",
                data=cfg_info
            )
            
        except Exception as e:
            return self._format_response(
                success=False,
                message=f"控制流图构建失败: {str(e)}",
                data={'error': str(e)}
            )
    
    def predicate_mining(self) -> Dict[str, Any]:
        """
        谓词挖掘
        - 识别边界条件和特殊值，通过谓词组合覆盖，而不是路径覆盖，减少测试数量
        
        Returns:
            JSON格式响应: {"message": "...", "data": {...}}
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
                        
                        # 如果是比较表达式，提取详细信息
                        if isinstance(test_node, ast.Compare):
                            self._extract_comparison_details(test_node, predicate_info)
                        
                        # 如果是逻辑表达式
                        elif isinstance(test_node, ast.BoolOp):
                            predicate_info['bool_op'] = type(test_node.op).__name__
                        
                        self.predicates.append(predicate_info)
                        
                    except Exception as e:
                        # 如果提取失败，至少记录基本信息
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
            
            stats['variables_in_predicates'] = list(stats['variables_in_predicates'])
            stats['predicates_by_type'] = dict(stats['predicates_by_type'])
            stats['predicates_by_function'] = dict(stats['predicates_by_function'])
            
            return self._format_response(
                success=True,
                message=f"谓词挖掘完成，发现 {len(self._predicates)} 个谓词",
                data={
                    'predicates': self._predicates,  # 直接返回原始谓词，不再标准化
                    'statistics': stats,
                    'metadata': {
                        'total_predicates': len(self._predicates),
                        'contains_boundary_values': stats['boundary_predicates'] > 0,
                        'contains_chained_comparisons': stats['chained_comparisons'] > 0,
                        'unique_variables': len(stats['variables_in_predicates'])
                    }
                }
            )
            
        except Exception as e:
            return self._format_response(
                success=False,
                message=f"谓词挖掘失败: {str(e)}",
                data={'error': str(e)}
            )
    
    def analyze_chained_comparisons(self) -> Dict[str, Any]:
        """
        专门分析链式比较，提取完整的边界信息
        
        Returns:
            JSON格式响应: {"message": "...", "data": {...}}
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
                    """详细分析链式比较"""
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
                    
                    # 如果只有一个主要变量，进行边界分析
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
                    """分析表达式，提取信息"""
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
                            # 变量在左边，常量在右边
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
                            # 常量在左边，变量在右边
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
                    
                    # 区间内测试值（如果上下界都存在）
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
            
            return self._format_response(
                success=True,
                message="链式比较分析完成",
                data={
                    'chained_comparisons': chained_predicates,
                    'statistics': stats,
                    'metadata': {
                        'found_chained_comparisons': len(chained_predicates) > 0,
                        'total_analyzed': len(chained_predicates),
                        'variables_analyzed': len(stats['variables_in_chained'])
                    }
                }
            )
            
        except Exception as e:
            return self._format_response(
                success=False,
                message=f"链式比较分析失败: {str(e)}",
                data={'error': str(e)}
            )
    
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
            include_all_defs: 是否包含所有定义（包括不在数据流路径上的）
            
        Returns:
            JSON格式响应: {"message": "...", "data": {...}}
        """
        try:
            # 构建def-use链
            def_use_chains = self._build_def_use_chains()
            
            if variable_name not in def_use_chains:
                return self._format_response(
                    success=False,
                    message=f"变量 '{variable_name}' 未找到",
                    data={
                        'target_variable': variable_name,
                        'target_line': line_number,
                        'affected_lines': set(),
                        'boundary_variables': [],
                        'data_flow_paths': [],
                        'error': f"Variable '{variable_name}' not found"
                    }
                )
            
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
            
            result = {
                'target_variable': variable_name,
                'target_line': line_number,
                'target_use_context': target_use,
                'affected_lines': sorted(list(affected_lines)),
                'affected_variables': sorted(list(affected_vars)),
                'boundary_variables': boundary_variables,
                'data_flow_paths': data_flow_paths,
                'statistics': stats
            }
            
            return self._format_response(
                success=True,
                message=f"后向切片分析完成，影响 {len(affected_lines)} 行代码",
                data=result
            )
            
        except Exception as e:
            return self._format_response(
                success=False,
                message=f"后向切片失败: {str(e)}",
                data={'error': str(e)}
            )
        
    def _is_on_dataflow_path(self, def_info, var, use_line):
        """检查定义是否在数据流路径上"""
        def_line = def_info['lineno']
        
        # 基本检查：定义在使用之前
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
        if predicates_result['success']:
            predicates = predicates_result['data']['predicates']
        else:
            predicates = []
        
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

        # 如果边界信息中没有明确的上下界，尝试从值集合中推断
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
        
        # 如果只有单个值，添加一些典型变异
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
        
        return -score  # 负分用于排序（分数越高越靠前）
    
    def _build_def_use_chains(self) -> Dict[str, Dict[str, List]]:
        """构建完整的def-use链（内部方法）"""
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
                
                # 处理函数参数（也是定义）
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
                """处理赋值语句（定义）"""
                # 获取使用的变量
                uses = self._extract_uses(node.value)
                
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        self._ensure_var_entry(var_name)
                        self.chains[var_name]['definitions'].append({
                            'lineno': node.lineno,
                            'scope': self.current_scope.copy(),
                            'function': self._current_function,
                            'uses': uses,
                            'type': 'assignment',
                            'node_type': 'Assign',
                            'value_expr': ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                        })
                        self.chains[var_name]['source'] = 'assignment'
                    
                    # 处理解包赋值：a, b = [1, 2]
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                var_name = elt.id
                                self._ensure_var_entry(var_name)
                                self.chains[var_name]['definitions'].append({
                                    'lineno': node.lineno,
                                    'scope': self.current_scope.copy(),
                                    'function': self._current_function,
                                    'uses': uses,
                                    'type': 'unpacking_assignment',
                                    'node_type': 'Assign',
                                    'parent_target': ast.unparse(target)
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
                """处理增强赋值（如 x += 1）"""
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
                """处理for循环（循环变量定义）"""
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
                """处理with语句（上下文管理器变量）"""
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
                            
            def _extract_uses(self, node):
                """从表达式提取使用的变量"""
                uses = []
                
                class UseExtractor(ast.NodeVisitor):
                    def __init__(self, use_list):
                        self.uses = use_list
                        self.context = []
                    
                    def visit_Name(self, node):
                        if isinstance(node.ctx, ast.Load):
                            self.uses.append(node.id)
                    
                    def visit_Call(self, node):
                        # 记录函数调用，但不将其视为变量使用（除非是变量作为函数）
                        if isinstance(node.func, ast.Name):
                            # 函数名本身可能是一个变量
                            self.uses.append(node.func.id)
                        self.generic_visit(node)
                
                extractor = UseExtractor(uses)
                if node:
                    extractor.visit(node)
                
                return uses
                
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
        
        visitor = DefUseVisitor(self._def_use_chains)
        visitor.visit(self.ast_tree)
        
        return self._def_use_chains
    
    def build_data_dependency_graph(self) -> Dict[str, Any]:
        """
        构建Def-Use/数据依赖图
        - 记录数据从哪里产生，到哪里被使用
        - 识别哪些变量修改后可能影响输出，即应该优先变异哪些变量
        
        Returns:
            JSON格式响应: {"message": "...", "data": {...}}
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
            
            # 统计信息
            stats = {
                'total_variables': len(self._data_deps),
                'total_dependencies': sum(len(deps) for deps in self._data_deps.values()),
                'variables_with_dependencies': [],
                'most_dependent_variables': [],
                'dependency_types': defaultdict(int)
            }
            
            for var, deps in self._data_deps.items():
                if deps:
                    stats['variables_with_dependencies'].append(var)
                    stats['most_dependent_variables'].append((var, len(deps)))
                    for dep in deps:
                        stats['dependency_types'][dep['type']] += 1
            
            # 排序
            stats['most_dependent_variables'].sort(key=lambda x: x[1], reverse=True)
            stats['most_dependent_variables'] = stats['most_dependent_variables'][:10]
            stats['dependency_types'] = dict(stats['dependency_types'])
            
            return self._format_response(
                success=True,
                message="数据依赖图构建完成",
                data={
                    'dependencies': dict(self._data_deps),
                    'statistics': stats,
                    'metadata': {
                        'total_variables_analyzed': len(def_use_chains),
                        'has_dependencies': len(self._data_deps) > 0,
                        'dependency_graph_complexity': stats['total_dependencies']
                    }
                }
            )
            
        except Exception as e:
            return self._format_response(
                success=False,
                message=f"数据依赖图构建失败: {str(e)}",
                data={'error': str(e)}
            )
    
    def _same_scope(self, scope1, scope2):
        """检查两个作用域是否相同"""
        return scope1 == scope2
    
    def infer_input_structure(self) -> Dict[str, Any]:
        """
        输入结构/格式推断
        - 支持各种input()使用模式
        - 考虑上下文推断类型
        - 识别循环、条件中的输入
        - 智能推断数据结构
        
        Returns:
            JSON格式响应: {"message": "...", "data": {...}}
        """
        try:
            self._input_structure = {
                'functions': [],
                'parameters': [],
                'expected_types': {},
                'constraints': [],
                'input_patterns': [],
                'input_calls': [],
                'main_inputs': [],
                'detected_types': {},
                'global_inputs': [],
                'input_contexts': {},  # 输入调用的上下文信息
                'input_transformations': []  # 输入转换链
            }
            
            class InputStructureVisitor(ast.NodeVisitor):
                def __init__(self, structure):
                    self.structure = structure
                    self._current_function = None
                    self._current_context = []  # 当前上下文栈
                    self._variable_usage = {}   # 变量使用跟踪
                    self._input_variables = set()  # 输入相关的变量
                    
                def visit_FunctionDef(self, node):
                    """分析函数定义"""
                    old_function = self._current_function
                    old_context = self._current_context.copy()
                    
                    self._current_function = node.name
                    self._current_context.append(('function', node.name))
                    
                    func_info = self._analyze_function_def(node)
                    self.structure['functions'].append(func_info)
                    
                    self.generic_visit(node)
                    
                    self._current_function = old_function
                    self._current_context = old_context
                
                def _analyze_function_def(self, node):
                    """分析函数定义详情"""
                    func_info = {
                        'name': node.name,
                        'args': [],
                        'returns': None,
                        'docstring': ast.get_docstring(node),
                        'lineno': node.lineno,
                        'has_input': False,
                        'input_contexts': []
                    }
                    
                    # 分析参数
                    if node.args:
                        self._analyze_arguments(node.args, func_info)
                    
                    # 分析返回值类型
                    if node.returns:
                        try:
                            func_info['returns'] = ast.unparse(node.returns)
                        except:
                            func_info['returns'] = str(node.returns)
                    
                    return func_info
                
                def _analyze_arguments(self, args_node, func_info):
                    """分析函数参数"""
                    # 位置参数
                    if hasattr(args_node, 'args'):
                        for i, arg in enumerate(args_node.args):
                            arg_info = self._analyze_argument(arg, i, 'positional')
                            if arg_info:
                                func_info['args'].append(arg_info)
                                self.structure['parameters'].append({
                                    'function': func_info['name'],
                                    **arg_info
                                })
                    
                    # 关键字参数
                    if hasattr(args_node, 'kwonlyargs'):
                        for arg in args_node.kwonlyargs:
                            arg_info = self._analyze_argument(arg, 'keyword', 'keyword')
                            if arg_info:
                                func_info['args'].append(arg_info)
                                self.structure['parameters'].append({
                                    'function': func_info['name'],
                                    **arg_info
                                })
                    
                    # 可变参数
                    if hasattr(args_node, 'vararg') and args_node.vararg:
                        arg_info = self._analyze_argument(args_node.vararg, '*', 'vararg')
                        if arg_info:
                            func_info['args'].append(arg_info)
                    
                    # 关键字参数字典
                    if hasattr(args_node, 'kwarg') and args_node.kwarg:
                        arg_info = self._analyze_argument(args_node.kwarg, '**', 'kwarg')
                        if arg_info:
                            func_info['args'].append(arg_info)
                
                def _analyze_argument(self, arg, position, arg_type):
                    """分析单个参数"""
                    try:
                        arg_info = {
                            'name': arg.arg if hasattr(arg, 'arg') else 'unknown',
                            'position': position,
                            'type': arg_type,
                            'annotation': None,
                            'inferred_type': None,
                            'default': None,
                            'is_input_related': False
                        }
                        
                        # 获取类型注解
                        if hasattr(arg, 'annotation') and arg.annotation:
                            try:
                                arg_info['annotation'] = ast.unparse(arg.annotation)
                                arg_info['inferred_type'] = self._infer_type_from_annotation(arg.annotation)
                            except:
                                arg_info['annotation'] = str(arg.annotation)
                        
                        return arg_info
                    except Exception:
                        return None
                
                def visit_Assign(self, node):
                    """分析赋值语句中的输入"""
                    # 处理赋值给变量的输入
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            
                            # 分析赋值值
                            input_info = self._analyze_input_assignment(node.value, var_name)
                            if input_info:
                                self._record_input_pattern(input_info, var_name, node.lineno)
                                
                                # 标记变量为输入相关
                                self._input_variables.add(var_name)
                                
                                # 记录变量类型
                                if var_name not in self.structure['detected_types']:
                                    self.structure['detected_types'][var_name] = set()
                                self.structure['detected_types'][var_name].add(input_info.get('inferred_type', 'unknown'))
                    
                    self.generic_visit(node)
                
                def visit_AnnAssign(self, node):
                    """处理带类型注解的赋值"""
                    if isinstance(node.target, ast.Name):
                        var_name = node.target.id
                        
                        if node.value:
                            input_info = self._analyze_input_assignment(node.value, var_name)
                            if input_info:
                                self._record_input_pattern(input_info, var_name, node.lineno)
                                self._input_variables.add(var_name)
                        
                        # 记录类型注解
                        if var_name not in self.structure['detected_types']:
                            self.structure['detected_types'][var_name] = set()
                        if node.annotation:
                            try:
                                type_str = ast.unparse(node.annotation)
                                self.structure['detected_types'][var_name].add(type_str)
                            except:
                                pass
                    
                    self.generic_visit(node)
                
                def _analyze_input_assignment(self, value_node, target_var):
                    """分析赋值值是否为输入"""
                    if not value_node:
                        return None
                    
                    # 深度分析输入调用链
                    return self._analyze_input_expression(value_node, target_var)
                
                def _analyze_input_expression(self, expr, context_var=None, depth=0):
                    """分析表达式中的输入调用链"""
                    if depth > 5:  # 防止无限递归
                        return None
                    
                    # 1. 直接input()调用
                    if isinstance(expr, ast.Call):
                        input_info = self._analyze_direct_input_call(expr, context_var)
                        if input_info:
                            return input_info
                    
                    # 2. input() 在转换函数中
                    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
                        func_name = expr.func.id
                        
                        # 检查参数中是否有input调用
                        for i, arg in enumerate(expr.args):
                            arg_input = self._analyze_input_expression(arg, context_var, depth+1)
                            if arg_input:
                                # 构建转换链
                                return self._build_transformation_chain(arg_input, func_name, expr)
                        
                        # 检查关键字参数
                        for kw in expr.keywords:
                            kw_input = self._analyze_input_expression(kw.value, context_var, depth+1)
                            if kw_input:
                                return self._build_transformation_chain(kw_input, func_name, expr)
                    
                    # 3. input() 在列表、字典等结构中
                    elif isinstance(expr, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
                        return self._analyze_input_in_container(expr, context_var, depth)
                    
                    # 4. input() 在切片、下标中
                    elif isinstance(expr, ast.Subscript):
                        if isinstance(expr.value, ast.Call):
                            input_info = self._analyze_input_expression(expr.value, context_var, depth+1)
                            if input_info:
                                # 添加下标访问信息
                                input_info['has_subscript'] = True
                                input_info['subscript'] = ast.unparse(expr.slice) if hasattr(ast, 'unparse') else str(expr.slice)
                                return input_info
                    
                    # 5. input() 在二元运算中
                    elif isinstance(expr, ast.BinOp):
                        left_input = self._analyze_input_expression(expr.left, context_var, depth+1)
                        right_input = self._analyze_input_expression(expr.right, context_var, depth+1)
                        
                        if left_input or right_input:
                            # 合并输入信息
                            input_info = left_input or right_input
                            input_info['in_binary_op'] = True
                            input_info['operator'] = type(expr.op).__name__
                            return input_info
                    
                    # 6. input() 在比较中
                    elif isinstance(expr, ast.Compare):
                        for comparator in [expr.left] + expr.comparators:
                            comp_input = self._analyze_input_expression(comparator, context_var, depth+1)
                            if comp_input:
                                comp_input['in_comparison'] = True
                                return comp_input
                    
                    return None
                
                def _analyze_direct_input_call(self, call_node, context_var):
                    """分析直接的input()调用"""
                    if not isinstance(call_node.func, ast.Name):
                        return None
                    
                    func_name = call_node.func.id
                    
                    # input() 函数
                    if func_name == 'input':
                        prompt = None
                        if call_node.args and isinstance(call_node.args[0], ast.Constant):
                            prompt = call_node.args[0].value
                        
                        return {
                            'input_type': 'raw_input',
                            'input_function': 'input()',
                            'prompt': prompt,
                            'inferred_type': 'str',  # input() 默认返回字符串
                            'conversion_chain': ['input'],
                            'context_var': context_var,
                            'lineno': call_node.lineno,
                            'context': self._get_current_context()
                        }
                    
                    return None
                
                def _build_transformation_chain(self, base_input, transformation_func, call_node):
                    """构建输入转换链"""
                    if not base_input:
                        return None
                    
                    # 复制基础输入信息
                    input_info = base_input.copy()
                    
                    # 添加转换函数
                    input_info['conversion_chain'].append(transformation_func)
                    input_info['input_function'] = f"{transformation_func}({input_info['input_function']})"
                    
                    # 更新推断类型
                    new_type = self._infer_type_from_transformation(
                        input_info['inferred_type'], 
                        transformation_func,
                        call_node
                    )
                    input_info['inferred_type'] = new_type
                    
                    # 更新输入类型
                    input_info['input_type'] = f"{transformation_func}_input"
                    
                    return input_info
                
                def _analyze_input_in_container(self, container_node, context_var, depth):
                    """分析容器中的输入"""
                    input_items = []
                    
                    if isinstance(container_node, ast.List):
                        for i, elt in enumerate(container_node.elts):
                            elt_input = self._analyze_input_expression(elt, f"{context_var}[{i}]", depth+1)
                            if elt_input:
                                input_items.append(elt_input)
                    
                    elif isinstance(container_node, ast.Tuple):
                        for i, elt in enumerate(container_node.elts):
                            elt_input = self._analyze_input_expression(elt, f"{context_var}[{i}]", depth+1)
                            if elt_input:
                                input_items.append(elt_input)
                    
                    elif isinstance(container_node, ast.Dict):
                        # 检查键和值
                        for i, (key, value) in enumerate(zip(container_node.keys, container_node.values)):
                            if key:
                                key_input = self._analyze_input_expression(key, f"{context_var}_key_{i}", depth+1)
                                if key_input:
                                    input_items.append(key_input)
                            
                            if value:
                                value_input = self._analyze_input_expression(value, f"{context_var}_value_{i}", depth+1)
                                if value_input:
                                    input_items.append(value_input)
                    
                    if input_items:
                        # 取第一个输入项作为代表，记录容器信息
                        main_input = input_items[0].copy()
                        main_input['in_container'] = True
                        main_input['container_type'] = type(container_node).__name__.lower()
                        main_input['num_inputs_in_container'] = len(input_items)
                        return main_input
                    
                    return None
                
                def _infer_type_from_transformation(self, base_type, transformation, call_node):
                    """根据转换函数推断类型"""
                    type_mappings = {
                        'int': {
                            'str': 'int',
                            'float': 'int',
                            'Any': 'int'
                        },
                        'float': {
                            'str': 'float',
                            'int': 'float',
                            'Any': 'float'
                        },
                        'str': {
                            'Any': 'str'
                        },
                        'list': {
                            'str': 'list[str]',  # str.split() -> list[str]
                            'Any': 'list'
                        },
                        'eval': {
                            'str': 'Any'  # eval可以返回任何类型
                        },
                        'split': {
                            'str': 'list[str]'
                        },
                        'strip': {
                            'str': 'str'
                        },
                        'map': self._infer_map_type,
                        'filter': self._infer_filter_type,
                        'sorted': self._infer_sorted_type
                    }
                    
                    # 特殊处理 map, filter 等函数
                    if transformation in ['map', 'filter', 'sorted']:
                        infer_func = type_mappings[transformation]
                        return infer_func(call_node, base_type)
                    
                    # 普通类型转换
                    if transformation in type_mappings:
                        mapping = type_mappings[transformation]
                        return mapping.get(base_type, 'Any')
                    
                    # 默认返回 Any
                    return 'Any'
                
                def _infer_map_type(self, call_node, base_type):
                    """推断map函数的返回类型"""
                    if len(call_node.args) >= 2:
                        # 检查第一个参数（函数）
                        func_arg = call_node.args[0]
                        if isinstance(func_arg, ast.Name):
                            func_name = func_arg.id
                            if func_name == 'int':
                                return 'list[int]'
                            elif func_name == 'float':
                                return 'list[float]'
                            elif func_name == 'str':
                                return 'list[str]'
                    
                    return 'list[Any]'
                
                def _infer_filter_type(self, call_node, base_type):
                    """推断filter函数的返回类型"""
                    return base_type  # filter保持输入类型
                
                def _infer_sorted_type(self, call_node, base_type):
                    """推断sorted函数的返回类型"""
                    if base_type.startswith('list['):
                        return base_type  # sorted返回相同类型的列表
                    elif base_type == 'str':
                        return 'list[str]'
                    else:
                        return 'list[Any]'
                
                def _record_input_pattern(self, input_info, var_name, lineno):
                    """记录输入模式"""
                    pattern_info = {
                        'variable': var_name,
                        'input_type': input_info['input_type'],
                        'input_function': input_info['input_function'],
                        'inferred_type': input_info['inferred_type'],
                        'conversion_chain': input_info['conversion_chain'],
                        'lineno': lineno,
                        'function': self._current_function,
                        'context': input_info.get('context', []),
                        'prompt': input_info.get('prompt'),
                        'has_subscript': input_info.get('has_subscript', False),
                        'subscript': input_info.get('subscript'),
                        'in_container': input_info.get('in_container', False),
                        'container_type': input_info.get('container_type'),
                        'in_binary_op': input_info.get('in_binary_op', False),
                        'operator': input_info.get('operator'),
                        'in_comparison': input_info.get('in_comparison', False)
                    }
                    
                    self.structure['input_patterns'].append(pattern_info)
                    self.structure['input_calls'].append(pattern_info)
                    
                    # 如果是全局输入（不在函数内）
                    if not self._current_function:
                        self.structure['global_inputs'].append(pattern_info)
                        self.structure['main_inputs'].append({
                            'type': f'main_{input_info["input_type"]}_input',
                            'function': input_info['input_function'],
                            'line': lineno,
                            'variable': var_name,
                            'description': f'{input_info["input_function"]} 赋值给变量 {var_name}'
                        })
                    
                    # 记录输入上下文
                    context_key = f"{self._current_function or 'global'}:{lineno}"
                    if context_key not in self.structure['input_contexts']:
                        self.structure['input_contexts'][context_key] = []
                    self.structure['input_contexts'][context_key].append(pattern_info)
                
                def visit_For(self, node):
                    """记录for循环上下文"""
                    old_context = self._current_context.copy()
                    self._current_context.append(('for_loop', node.lineno))
                    
                    # 分析循环体中的输入
                    self.generic_visit(node)
                    
                    self._current_context = old_context
                
                def visit_While(self, node):
                    """记录while循环上下文"""
                    old_context = self._current_context.copy()
                    self._current_context.append(('while_loop', node.lineno))
                    
                    self.generic_visit(node)
                    
                    self._current_context = old_context
                
                def visit_If(self, node):
                    """记录if条件上下文"""
                    old_context = self._current_context.copy()
                    self._current_context.append(('if_branch', node.lineno))
                    
                    self.generic_visit(node)
                    
                    self._current_context = old_context
                
                def visit_Try(self, node):
                    """记录try块上下文"""
                    old_context = self._current_context.copy()
                    self._current_context.append(('try_block', node.lineno))
                    
                    self.generic_visit(node)
                    
                    self._current_context = old_context
                
                def visit_With(self, node):
                    """记录with块上下文"""
                    old_context = self._current_context.copy()
                    self._current_context.append(('with_block', node.lineno))
                    
                    self.generic_visit(node)
                    
                    self._current_context = old_context
                
                def _get_current_context(self):
                    """获取当前上下文信息"""
                    contexts = []
                    for ctx_type, ctx_info in self._current_context:
                        if ctx_type == 'function':
                            contexts.append(f"function:{ctx_info}")
                        elif ctx_type in ['for_loop', 'while_loop', 'if_branch', 'try_block', 'with_block']:
                            contexts.append(f"{ctx_type}:{ctx_info}")
                    return contexts
                
                def _infer_type_from_annotation(self, annotation):
                    """从类型注解推断类型"""
                    if not annotation:
                        return None
                    
                    try:
                        if isinstance(annotation, ast.Name):
                            return annotation.id
                        elif isinstance(annotation, ast.Subscript):
                            # 处理泛型如 List[int]
                            base = self._infer_type_from_annotation(annotation.value)
                            if isinstance(annotation.slice, ast.Name):
                                inner = annotation.slice.id
                                return f"{base}[{inner}]"
                            elif isinstance(annotation.slice, ast.Constant):
                                inner = annotation.slice.value
                                return f"{base}[{inner}]"
                            else:
                                return base
                        elif isinstance(annotation, ast.Constant):
                            return str(annotation.value)
                        else:
                            return str(annotation)
                    except:
                        return str(annotation) if annotation else None
                
                def visit_Call(self, node):
                    """分析函数调用中的输入（不在赋值中的）"""
                    # 检查是否有input调用
                    input_info = self._analyze_input_expression(node, None)
                    if input_info:
                        # 记录独立的输入调用
                        standalone_input = {
                            'type': 'standalone_input_call',
                            'input_function': input_info['input_function'],
                            'inferred_type': input_info['inferred_type'],
                            'lineno': node.lineno,
                            'function': self._current_function,
                            'context': input_info.get('context', []),
                            'conversion_chain': input_info['conversion_chain']
                        }
                        
                        self.structure['input_calls'].append(standalone_input)
                    
                    self.generic_visit(node)
            
            visitor = InputStructureVisitor(self._input_structure)
            
            visitor.visit(self.ast_tree)
            
            # 后处理：完善类型推断
            self._post_process_input_inference()
            
            # 统计信息
            stats = {
                'total_input_patterns': len(self._input_structure['input_patterns']),
                'total_input_calls': len(self._input_structure['input_calls']),
                'total_functions': len(self._input_structure['functions']),
                'input_types_found': set(),
                'input_contexts_found': len(self._input_structure['input_contexts']),
                'variables_with_input': len(self._input_structure['detected_types'])
            }
            
            for pattern in self._input_structure['input_patterns']:
                stats['input_types_found'].add(pattern['input_type'])
            
            stats['input_types_found'] = list(stats['input_types_found'])
            
            return self._format_response(
                success=True,
                message="输入结构推断完成",
                data={
                    'input_structure': self._input_structure,
                    'statistics': stats,
                    'metadata': {
                        'has_input': len(self._input_structure['input_patterns']) > 0,
                        'input_complexity': len(self._input_structure['input_transformations']),
                        'unique_input_types': len(stats['input_types_found'])
                    }
                }
            )
            
        except Exception as e:
            return self._format_response(
                success=False,
                message=f"输入结构推断失败: {str(e)}",
                data={'error': str(e)}
            )

    def _post_process_input_inference(self):
        """后处理输入推断结果"""
        # 1. 合并相同变量的类型信息
        for var, types in self._input_structure['detected_types'].items():
            if types:
                # 选择最具体的类型
                if 'list[str]' in types:
                    self._input_structure['expected_types'][var] = 'list[str]'
                elif 'list[int]' in types:
                    self._input_structure['expected_types'][var] = 'list[int]'
                elif 'list[float]' in types:
                    self._input_structure['expected_types'][var] = 'list[float]'
                elif 'list' in types:
                    self._input_structure['expected_types'][var] = 'list'
                elif 'dict' in types:
                    self._input_structure['expected_types'][var] = 'dict'
                elif 'int' in types:
                    self._input_structure['expected_types'][var] = 'int'
                elif 'float' in types:
                    self._input_structure['expected_types'][var] = 'float'
                elif 'str' in types:
                    self._input_structure['expected_types'][var] = 'str'
                else:
                    self._input_structure['expected_types'][var] = next(iter(types))
        
        # 2. 分析输入转换链
        self._analyze_input_transformations()
        
        # 3. 识别输入约束
        self._identify_input_constraints()

    def _analyze_input_transformations(self):
        """分析输入转换链"""
        transformations = []
        
        for pattern in self._input_structure['input_patterns']:
            chain = pattern.get('conversion_chain', [])
            if len(chain) > 1:
                transformation = {
                    'variable': pattern['variable'],
                    'input_type': pattern['input_type'],
                    'conversion_chain': chain,
                    'start_type': 'str',  # input()总是返回str
                    'end_type': pattern['inferred_type'],
                    'lineno': pattern['lineno'],
                    'context': pattern['context']
                }
                transformations.append(transformation)
        
        self._input_structure['input_transformations'] = transformations

    def _identify_input_constraints(self):
        """识别输入约束"""
        constraints = []
        
        # 结合谓词分析
        predicates_result = self.predicate_mining()
        if predicates_result['success']:
            predicates = predicates_result['data'].get('predicates', [])
        else:
            predicates = []
        
        for pattern in self._input_structure['input_patterns']:
            var = pattern['variable']
            inferred_type = pattern['inferred_type']
            
            # 查找与该变量相关的谓词
            var_predicates = []
            for pred in predicates:
                expr = pred.get('expression', '')
                if var in expr:
                    var_predicates.append(pred)
            
            if var_predicates:
                constraint = {
                    'variable': var,
                    'type': inferred_type,
                    'predicates': var_predicates,
                    'lineno': pattern['lineno'],
                    'context': pattern['context']
                }
                constraints.append(constraint)
        
        self._input_structure['constraints'] = constraints

    def analyze_input_patterns(self) -> Dict[str, Any]:
        """
        专门分析输入模式，支持各种常见用法
        
        Returns:
            JSON格式响应: {"message": "...", "data": {...}}
        """
        try:
            patterns = {
                'direct_input': [],      # input()
                'converted_input': [],   # int(input()), float(input())
                'eval_input': [],        # eval(input())
                'split_input': [],       # input().split()
                'stripped_input': [],    # input().strip()
                'mapped_input': [],      # map(int, input().split())
                'list_input': [],        # list(input().split())
                'container_input': [],   # 容器中的输入
                'conditional_input': [], # 条件中的输入
                'loop_input': [],        # 循环中的输入
                'error_handled_input': [] # 错误处理的输入
            }
            
            class InputPatternVisitor(ast.NodeVisitor):
                def __init__(self, patterns_dict):
                    self.patterns = patterns_dict
                    self._current_context = []
                    
                def visit_Call(self, node):
                    # 分析各种input()模式
                    self._analyze_input_pattern(node)
                    self.generic_visit(node)
                
                def _analyze_input_pattern(self, node):
                    """分析input()使用模式"""
                    # 检查是否是input()调用
                    input_expr = self._find_input_in_expression(node)
                    if not input_expr:
                        return
                    
                    # 分析模式
                    pattern = self._classify_input_pattern(node, input_expr)
                    if pattern:
                        pattern_type = pattern['pattern_type']
                        self.patterns[pattern_type].append(pattern)
                
                def _find_input_in_expression(self, expr):
                    """在表达式中查找input()调用"""
                    if isinstance(expr, ast.Call):
                        if isinstance(expr.func, ast.Name) and expr.func.id == 'input':
                            return expr
                        
                        # 递归检查参数
                        for arg in expr.args:
                            result = self._find_input_in_expression(arg)
                            if result:
                                return result
                        
                        for kw in expr.keywords:
                            result = self._find_input_in_expression(kw.value)
                            if result:
                                return result
                    
                    return None
                
                def _classify_input_pattern(self, node, input_expr):
                    """分类输入模式"""
                    pattern_info = {
                        'expression': ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                        'input_line': input_expr.lineno,
                        'context': self._current_context.copy(),
                        'pattern_type': 'direct_input'
                    }
                    
                    # 1. 直接input()
                    if node == input_expr:
                        pattern_info['pattern_type'] = 'direct_input'
                        return pattern_info
                    
                    # 2. 类型转换
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        if func_name in ['int', 'float', 'str', 'bool']:
                            pattern_info['pattern_type'] = 'converted_input'
                            pattern_info['conversion'] = func_name
                            return pattern_info
                        
                        elif func_name == 'eval':
                            pattern_info['pattern_type'] = 'eval_input'
                            return pattern_info
                        
                        elif func_name == 'list':
                            # 检查是否是 list(input().split()) 模式
                            if node.args and isinstance(node.args[0], ast.Call):
                                inner = node.args[0]
                                if isinstance(inner.func, ast.Attribute) and inner.func.attr == 'split':
                                    pattern_info['pattern_type'] = 'list_input'
                                    return pattern_info
                        
                        elif func_name == 'map':
                            # map(int, input().split()) 模式
                            pattern_info['pattern_type'] = 'mapped_input'
                            return pattern_info
                    
                    # 3. 字符串方法
                    if isinstance(node.func, ast.Attribute):
                        method_name = node.func.attr
                        
                        if method_name == 'split':
                            pattern_info['pattern_type'] = 'split_input'
                            return pattern_info
                        
                        elif method_name in ['strip', 'rstrip', 'lstrip']:
                            pattern_info['pattern_type'] = 'stripped_input'
                            pattern_info['strip_type'] = method_name
                            return pattern_info
                    
                    return None
            
            visitor = InputPatternVisitor(patterns)
            visitor.visit(self.ast_tree)
            
            # 统计信息
            stats = {
                'total_patterns': sum(len(p) for p in patterns.values()),
                'patterns_by_type': {k: len(v) for k, v in patterns.items() if v},
                'most_common_pattern': None,
                'unique_input_expressions': set()
            }
            
            for pattern_type, pattern_list in patterns.items():
                if pattern_list:
                    for pattern in pattern_list:
                        if 'expression' in pattern:
                            stats['unique_input_expressions'].add(pattern['expression'])
            
            # 找出最常见的模式
            if stats['patterns_by_type']:
                stats['most_common_pattern'] = max(stats['patterns_by_type'].items(), key=lambda x: x[1])
            
            stats['unique_input_expressions'] = list(stats['unique_input_expressions'])
            
            return self._format_response(
                success=True,
                message="输入模式分析完成",
                data={
                    'patterns': patterns,
                    'statistics': stats,
                    'metadata': {
                        'has_input_patterns': stats['total_patterns'] > 0,
                        'pattern_variety': len(stats['patterns_by_type']),
                        'input_diversity': len(stats['unique_input_expressions'])
                    }
                }
            )
            
        except Exception as e:
            return self._format_response(
                success=False,
                message=f"输入模式分析失败: {str(e)}",
                data={'error': str(e)}
            )
    
    def get_variable_types(self) -> Dict[str, Any]:
        """
        完整的变量类型推断
        - 结合多种推断方法
        - 支持复杂的类型推断
        
        Returns:
            JSON格式响应: {"message": "...", "data": {...}}
        """
        try:
            if self._variable_types is not None:
                return self._format_response(
                    success=True,
                    message="变量类型推断完成（缓存）",
                    data={
                        'variable_types': self._variable_types,
                        'statistics': {
                            'total_variables': len(self._variable_types),
                            'types_found': set(v['type'] for v in self._variable_types.values()),
                            'confidence_distribution': defaultdict(int)
                        }
                    }
                )
            
            self._variable_types = {}
            
            # 获取其他分析结果作为上下文
            input_structure_result = self.infer_input_structure()
            def_use_chains = self._build_def_use_chains()
            constants_result = self.extract_constants_and_comparisons()
            
            if constants_result['success']:
                constants = constants_result['data'].get('constants', {})
            else:
                constants = {}
            
            # 1. 创建类型推断器
            inferencer = TypeInferencer(
                variable_types=self._variable_types,
                input_structure=input_structure_result['data']['input_structure'] if input_structure_result['success'] else {},
                def_use_chains=def_use_chains,
                constants=constants
            )
            
            # 2. 执行多轮类型推断
            inferencer.visit(self.ast_tree)
            
            # 3. 传播和解析类型
            self._propagate_types()
            self._resolve_type_conflicts()
            self._infer_missing_types()

            # 过滤掉内置函数和明显不是变量的项
            self._filter_non_variables()
            
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
            
            return self._format_response(
                success=True,
                message=f"变量类型推断完成，分析了 {len(self._variable_types)} 个变量",
                data={
                    'variable_types': self._variable_types,
                    'statistics': stats,
                    'metadata': {
                        'has_type_inference': len(self._variable_types) > 0,
                        'type_variety': len(stats['types_found']),
                        'inference_quality': stats['confidence_distribution'].get('high', 0) / max(1, len(self._variable_types))
                    }
                }
            )
            
        except Exception as e:
            return self._format_response(
                success=False,
                message=f"变量类型推断失败: {str(e)}",
                data={'error': str(e)}
            )
        
    def _filter_non_variables(self):
        """过滤掉非变量项"""
        # 内置函数列表
        builtin_functions = {
            'eval', 'input', 'print', 'range', 'int', 'float', 'str', 'len',
            'abs', 'round', 'sum', 'max', 'min', 'type', 'isinstance',
            'issubclass', 'open', 'close', 'read', 'write','search'
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
    
    def _propagate_types(self):
        """传播类型信息"""
        # 基于def-use链传播类型
        def_use_chains = self._build_def_use_chains()
        
        changed = True
        max_iterations = 10
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for var_name, var_info in self._variable_types.items():
                # 如果变量类型不明确，尝试从使用处推断
                if var_info.get('confidence', 'low') in ['low', 'medium']:
                    # 查找变量的定义
                    if var_name in def_use_chains:
                        defs = def_use_chains[var_name].get('definitions', [])
                        uses = def_use_chains[var_name].get('uses', [])
                        
                        # 1. 从定义推断
                        for def_info in defs:
                            new_type = self._infer_type_from_definition(var_name, def_info)
                            if new_type and new_type != var_info['type']:
                                self._update_variable_type(var_name, new_type, 'propagation_from_definition')
                                changed = True
                        
                        # 2. 从使用推断
                        for use_info in uses:
                            new_type = self._infer_type_from_usage(var_name, use_info)
                            if new_type and new_type != var_info['type']:
                                self._update_variable_type(var_name, new_type, 'propagation_from_usage')
                                changed = True

    def _infer_type_from_definition(self, var_name, def_info):
        """从变量定义推断类型"""
        def_type = def_info.get('type', '')
        
        if def_type == 'parameter':
            # 函数参数，需要从调用处推断（这里简化处理）
            return None
        
        elif def_type == 'assignment':
            # 从赋值表达式推断
            value_expr = def_info.get('value_expr', '')
            if value_expr:
                # 这里需要解析表达式，简化处理
                if 'input()' in value_expr:
                    if 'int(' in value_expr:
                        return 'int'
                    elif 'float(' in value_expr:
                        return 'float'
                    elif 'list(' in value_expr:
                        return 'list'
                    else:
                        return 'str'
        
        return None

    def _infer_type_from_usage(self, var_name, use_info):
        """从变量使用推断类型"""
        # 检查使用上下文
        context = use_info.get('context', {})
        
        # 如果在数值运算中使用
        if context.get('in_numeric_operation', False):
            return 'numeric'
        
        # 如果在字符串操作中使用
        if context.get('in_string_operation', False):
            return 'str'
        
        # 如果在列表操作中使用
        if context.get('in_list_operation', False):
            return 'list'
        
        return None

    def _update_variable_type(self, var_name, new_type, source):
        """更新变量类型"""
        if var_name in self._variable_types:
            current_info = self._variable_types[var_name]
            
            # 只在不明确时更新
            if current_info.get('confidence', 'low') in ['low', 'medium']:
                current_info['type'] = new_type
                current_info['sources'].append({
                    'source': source,
                    'type': new_type,
                    'confidence': 'medium'
                })

    def _resolve_type_conflicts(self):
        """解决类型冲突"""
        for var_name, var_info in self._variable_types.items():
            sources = var_info.get('sources', [])
            
            if len(sources) > 1:
                # 收集所有推断的类型
                type_counts = {}
                for source in sources:
                    t = source['type']
                    conf = source.get('confidence', 'medium')
                    weight = 3 if conf == 'high' else 2 if conf == 'medium' else 1
                    type_counts[t] = type_counts.get(t, 0) + weight
                
                # 选择权重最高的类型
                if type_counts:
                    best_type = max(type_counts.items(), key=lambda x: x[1])[0]
                    var_info['type'] = best_type
                    
                    # 更新置信度
                    max_weight = max(type_counts.values())
                    total_weight = sum(type_counts.values())
                    
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

class TypeInferencer(ast.NodeVisitor):
    """完整的类型推断器"""
    
    def __init__(self, variable_types, input_structure, def_use_chains, constants):
        self.types = variable_types
        self.input_structure = input_structure
        self.def_use_chains = def_use_chains
        self.constants = constants
        self._current_function = None
        self._current_class = None
        self._scope_stack = []  # 作用域栈
        
        # 内置函数返回类型映射
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
            'max': 'Any',
            'min': 'Any',
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
            
            # 其他
            'eval': 'Any',
            'exec': 'None',
            'iter': 'iterator',
            'next': 'Any',
            'filter': 'filter',
            'map': 'map',
            'all': 'bool',
            'any': 'bool'
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
            ast.MatMult: 'numeric',  # 矩阵乘法
            
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
                # 注意：这里无法获取默认值，需要在调用时推断
        
        # 关键字参数
        if hasattr(node.args, 'kwonlyargs'):
            for arg in node.args.kwonlyargs:
                if isinstance(arg, ast.arg):
                    var_name = arg.arg
                    
                    if arg.annotation:
                        type_info = self._infer_type_from_annotation(arg.annotation)
                        if type_info:
                            self._add_type(var_name, type_info, 'annotation', 'high')
    
    def _infer_return_type(self, node):
        """推断函数返回值类型"""
        # 收集所有return语句的返回类型
        return_types = []
        
        class ReturnCollector(ast.NodeVisitor):
            def __init__(self):
                self.types = []
            
            def visit_Return(self, return_node):
                if return_node.value:
                    # 这里需要推断返回值类型
                    # 简化的实现：记录表达式
                    self.types.append(ast.unparse(return_node.value) 
                                    if hasattr(ast, 'unparse') else str(return_node.value))
        
        collector = ReturnCollector()
        collector.visit(node)
        
        # 如果有明确的返回值注解，使用它
        if node.returns:
            return_type = self._infer_type_from_annotation(node.returns)
            if return_type:
                self._add_type(f'{node.name}_return', return_type, 'return_annotation', 'high')
    
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
        """从赋值推断类型"""
        # 推断右侧表达式的类型
        rhs_type = self._infer_expression_type(node.value)
        
        if rhs_type:
            # 处理多个目标
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    self._add_type(var_name, rhs_type['type'], 'assignment', rhs_type.get('confidence', 'medium'))
                
                # 处理解包赋值
                elif isinstance(target, (ast.Tuple, ast.List)):
                    self._infer_unpacking_type(target, node.value, rhs_type)
        
        self.generic_visit(node)
    
    def _infer_unpacking_type(self, target, value, rhs_type):
        """推断解包赋值的类型"""
        if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
            for i, (tgt, elt) in enumerate(zip(target.elts, value.elts)):
                if isinstance(tgt, ast.Name):
                    elt_type = self._infer_expression_type(elt)
                    if elt_type:
                        self._add_type(tgt.id, elt_type['type'], 'unpacking_assignment', elt_type.get('confidence', 'medium'))
        
        # 处理 *args 解包
        elif isinstance(value, ast.Starred):
            # 星号解包，类型为列表元素类型
            inner_type = self._infer_expression_type(value.value)
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
                rhs_type = self._infer_expression_type(node.value)
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
                # 数值运算，保持数值类型
                new_type = 'numeric' if current_type in ['int', 'float', 'numeric'] else 'Any'
                self._add_type(var_name, new_type, 'augmented_assignment', 'medium')
            
            elif isinstance(node.op, (ast.BitOr, ast.BitAnd, ast.BitXor, ast.LShift, ast.RShift)):
                # 位运算，保持整数类型
                self._add_type(var_name, 'int', 'augmented_assignment', 'medium')
        
        self.generic_visit(node)
    
    def visit_For(self, node):
        """推断循环变量类型"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            
            # 推断迭代对象的类型
            iter_type = self._infer_expression_type(node.iter)
            
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
                context_type = self._infer_expression_type(item.context_expr)
                
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
        # 首先推断被调用函数的返回类型
        return_type = self._infer_call_return_type(node)
        
        # 如果调用是赋值的一部分，类型会在Assign中处理
        # 这里主要处理独立函数调用
        
        self.generic_visit(node)
        return return_type
    
    def _infer_call_return_type(self, node):
        """推断函数调用的返回类型"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # 内置函数
            if func_name in self.builtin_return_types:
                return {
                    'type': self.builtin_return_types[func_name],
                    'confidence': 'high',
                    'source': f'builtin_{func_name}'
                }
            
        elif isinstance(node.func, ast.Attribute):
            # 方法调用
            obj_type = self._infer_expression_type(node.func.value)
            if obj_type:
                # 常见方法返回类型
                method_name = node.func.attr
                return self._infer_method_return_type(obj_type['type'], method_name, node)
        
        return {'type': 'Any', 'confidence': 'low', 'source': 'unknown_call'}
    
    def _infer_method_return_type(self, obj_type, method_name, call_node):
        """推断方法调用的返回类型"""
        # 字符串方法
        if obj_type == 'str':
            str_methods = {
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
            if method_name in str_methods:
                return {'type': str_methods[method_name], 'confidence': 'high', 'source': 'str_method'}
        
        # 列表方法
        elif obj_type.startswith('list[') or obj_type == 'list':
            list_methods = {
                'append': 'None',
                'extend': 'None',
                'insert': 'None',
                'remove': 'None',
                'pop': obj_type[5:-1] if obj_type.startswith('list[') else 'Any',
                'index': 'int',
                'count': 'int',
                'sort': 'None',
                'reverse': 'None',
                'copy': obj_type
            }
            if method_name in list_methods:
                return {'type': list_methods[method_name], 'confidence': 'high', 'source': 'list_method'}
        
        # 字典方法
        elif obj_type.startswith('dict[') or obj_type == 'dict':
            dict_methods = {
                'get': 'Any',
                'keys': 'dict_keys',
                'values': 'dict_values',
                'items': 'dict_items',
                'pop': 'Any',
                'popitem': 'tuple',
                'update': 'None',
                'clear': 'None',
                'copy': obj_type
            }
            if method_name in dict_methods:
                return {'type': dict_methods[method_name], 'confidence': 'high', 'source': 'dict_method'}
        
        return {'type': 'Any', 'confidence': 'low', 'source': 'unknown_method'}
    
    def _infer_expression_type(self, node):
        """推断表达式的类型"""
        if node is None:
            return None
        
        # 常量
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, int):
                return {'type': 'int', 'confidence': 'high', 'value': value}
            elif isinstance(value, float):
                return {'type': 'float', 'confidence': 'high', 'value': value}
            elif isinstance(value, str):
                return {'type': 'str', 'confidence': 'high', 'value': value}
            elif isinstance(value, bool):
                return {'type': 'bool', 'confidence': 'high', 'value': value}
            elif value is None:
                return {'type': 'None', 'confidence': 'high'}
            else:
                return {'type': type(value).__name__, 'confidence': 'high', 'value': value}
        
        # 名称（变量）
        elif isinstance(node, ast.Name):
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
        
        # 列表
        elif isinstance(node, ast.List):
            if node.elts:
                # 推断所有元素的类型
                element_types = []
                for elt in node.elts:
                    elt_type = self._infer_expression_type(elt)
                    if elt_type:
                        element_types.append(elt_type['type'])
                
                if element_types:
                    # 检查是否所有元素类型相同
                    if len(set(element_types)) == 1:
                        return {'type': f'list[{element_types[0]}]', 'confidence': 'medium'}
                    else:
                        # 混合类型
                        return {'type': 'list', 'confidence': 'medium'}
            
            return {'type': 'list', 'confidence': 'medium'}
        
        # 字典
        elif isinstance(node, ast.Dict):
            if node.keys and node.values:
                # 推断键和值的类型
                key_types = []
                value_types = []
                
                for key, value in zip(node.keys, node.values):
                    if key:
                        key_type = self._infer_expression_type(key)
                        if key_type:
                            key_types.append(key_type['type'])
                    
                    if value:
                        value_type = self._infer_expression_type(value)
                        if value_type:
                            value_types.append(value_type['type'])
                
                if key_types and value_types:
                    # 简化的字典类型推断
                    if len(set(key_types)) == 1 and len(set(value_types)) == 1:
                        return {'type': f'dict[{key_types[0]}, {value_types[0]}]', 'confidence': 'low'}
            
            return {'type': 'dict', 'confidence': 'medium'}
        
        # 元组
        elif isinstance(node, ast.Tuple):
            if node.elts:
                element_types = []
                for elt in node.elts:
                    elt_type = self._infer_expression_type(elt)
                    if elt_type:
                        element_types.append(elt_type['type'])
                
                if element_types:
                    return {'type': f'tuple[{", ".join(element_types)}]', 'confidence': 'medium'}
            
            return {'type': 'tuple', 'confidence': 'medium'}
        
        # 集合
        elif isinstance(node, ast.Set):
            return {'type': 'set', 'confidence': 'medium'}
        
        # 函数调用
        elif isinstance(node, ast.Call):
            return self._infer_call_return_type(node)
        
        # 二元运算
        elif isinstance(node, ast.BinOp):
            left_type = self._infer_expression_type(node.left)
            right_type = self._infer_expression_type(node.right)
            
            if left_type and right_type:
                # 根据运算符推断类型
                op_type = type(node.op)
                if op_type in self.operator_return_types:
                    handler = self.operator_return_types[op_type]
                    if callable(handler):
                        return handler(left_type, right_type)
                    else:
                        return {'type': handler, 'confidence': 'medium'}
            
            return {'type': 'Any', 'confidence': 'low'}
        
        # 比较
        elif isinstance(node, ast.Compare):
            return {'type': 'bool', 'confidence': 'high'}
        
        # 布尔运算
        elif isinstance(node, ast.BoolOp):
            return {'type': 'bool', 'confidence': 'high'}
        
        # 一元运算
        elif isinstance(node, ast.UnaryOp):
            operand_type = self._infer_expression_type(node.operand)
            if operand_type:
                if isinstance(node.op, ast.Not):
                    return {'type': 'bool', 'confidence': 'high'}
                elif isinstance(node.op, ast.USub):
                    # 负号，保持数值类型
                    if operand_type['type'] in ['int', 'float', 'numeric']:
                        return {'type': operand_type['type'], 'confidence': 'medium'}
            
            return {'type': 'Any', 'confidence': 'low'}
        
        # 下标访问
        elif isinstance(node, ast.Subscript):
            value_type = self._infer_expression_type(node.value)
            if value_type:
                type_str = value_type['type']
                if type_str.startswith('list['):
                    # list[T] 的下标访问返回 T
                    return {'type': type_str[5:-1], 'confidence': 'medium'}
                elif type_str.startswith('dict['):
                    # dict[K, V] 的下标访问返回 V
                    parts = type_str[5:-1].split(', ')
                    if len(parts) == 2:
                        return {'type': parts[1].strip(), 'confidence': 'low'}
                elif type_str in ['list', 'dict', 'tuple']:
                    return {'type': 'Any', 'confidence': 'low'}
            
            return {'type': 'Any', 'confidence': 'low'}
        
        # 属性访问
        elif isinstance(node, ast.Attribute):
            # 简化处理
            return {'type': 'Any', 'confidence': 'low'}
        
        # 推导式
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comp_type = type(node).__name__
            if comp_type == 'ListComp':
                return {'type': 'list', 'confidence': 'medium'}
            elif comp_type == 'SetComp':
                return {'type': 'set', 'confidence': 'medium'}
            elif comp_type == 'DictComp':
                return {'type': 'dict', 'confidence': 'medium'}
            elif comp_type == 'GeneratorExp':
                return {'type': 'generator', 'confidence': 'medium'}
        
        # 条件表达式
        elif isinstance(node, ast.IfExp):
            true_type = self._infer_expression_type(node.body)
            false_type = self._infer_expression_type(node.orelse)
            
            if true_type and false_type and true_type['type'] == false_type['type']:
                return {'type': true_type['type'], 'confidence': 'medium'}
            
            return {'type': 'Any', 'confidence': 'low'}
        
        # 默认返回未知类型
        return {'type': 'Any', 'confidence': 'low', 'source': 'unknown_expression'}
    
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
                    # 多参数泛型，如 Dict[str, int]
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
