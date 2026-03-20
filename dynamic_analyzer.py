#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态分析器 - 基于静态分析结果的动态执行分析
输入：源代码 + 静态分析结果
输出：执行轨迹、边覆盖、变量值序列、分支距离、类型验证
"""

import sys
import builtins
import traceback
import linecache
import time
import re
import ast
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple


class DynamicAnalyzer:
    """
    动态分析器
    """
    
    def __init__(self, 
                source_code: str,
                static_analysis_result: Dict[str, Any],
                timeout: float = 5.0,
                max_execution_steps: int = 10000):
        """
        初始化动态分析器
        
        Args:
            source_code: 源代码字符串
            static_analysis_result: 静态分析结果，包含：
                - cfg: 控制流图
                - predicates: 谓词列表
                - data_dependencies: 数据依赖
                - variable_types: 变量类型
            timeout: 执行超时时间（秒）
            max_execution_steps: 最大执行步数
        """
        self.source_code = source_code
        self.static_result = static_analysis_result
        self.timeout = timeout
        self.max_execution_steps = max_execution_steps
        
        # 从静态分析结果中提取信息
        self._extract_static_info()
        
        # 动态分析结果存储
        self._reset_results()
    
    def _extract_static_info(self):
        """从静态分析结果中提取关键信息"""
        # CFG信息
        cfg = self.static_result.get('cfg', {})
        self.cfg_nodes = set()
        self.cfg_edges = set()
        self.cfg_line_to_nodes = defaultdict(list)  # 行号到节点的映射
        self.cfg_node_to_line = {}  # 节点到行号的映射
        
        if cfg:
            # 提取节点
            for branch in cfg.get('branches', []):
                node_id = branch.get('node_id')
                lineno = branch.get('lineno')
                if node_id:
                    self.cfg_nodes.add(node_id)
                    if lineno:
                        self.cfg_line_to_nodes[lineno].append(node_id)
                        self.cfg_node_to_line[node_id] = lineno
            
            # 提取边，并为边的起止节点建立行号映射
            for edge in cfg.get('edges', []):
                from_node = edge.get('from')
                to_node = edge.get('to')
                if from_node and to_node:
                    self.cfg_edges.add((from_node, to_node))
                    # 所有端点都加入 cfg_nodes
                    self.cfg_nodes.add(from_node)
                    self.cfg_nodes.add(to_node)
                    # 尝试从节点ID中提取行号
                    if from_node not in self.cfg_node_to_line:
                        from_line = self._extract_line_from_node(from_node)
                        if from_line:
                            self.cfg_node_to_line[from_node] = from_line
                            self.cfg_line_to_nodes[from_line].append(from_node)
                    if to_node not in self.cfg_node_to_line:
                        to_line = self._extract_line_from_node(to_node)
                        if to_line:
                            self.cfg_node_to_line[to_node] = to_line
                            self.cfg_line_to_nodes[to_line].append(to_node)
        
        # 关键变量
        self.key_variables = set()
        
        # 从谓词中提取
        predicates = self.static_result.get('predicates', [])
        for pred in predicates:
            if 'var' in pred:
                self.key_variables.add(pred['var'])
            if 'vars' in pred and isinstance(pred['vars'], list):
                self.key_variables.update(pred['vars'])
        
        # 从数据依赖中提取
        data_deps = self.static_result.get('data_dependencies', {})
        self.key_variables.update(data_deps.keys())
        
        # 变量类型
        self.static_types = self.static_result.get('variable_types', {})
        
        # 谓词信息（用于分支距离计算）
        self.predicates_by_line = {}
        for pred in predicates:
            anchor = pred.get('anchor_stmt', '')
            if anchor:
                match = re.search(r'L(\d+)', anchor)
                if match:
                    lineno = int(match.group(1))
                    self.predicates_by_line[lineno] = pred

        # 行号对应的谓词ID
        self.line_to_pred_id = {}
        for pred in predicates:
            pid = pred.get('pred_id', '')
            if not pid:
                continue
            # 优先用 lineno 字段
            ln = pred.get('lineno')
            if isinstance(ln, int):
                self.line_to_pred_id[ln] = pid
            # 兼容从 anchor_stmt 提取
            anchor = pred.get('anchor_stmt', '')
            if anchor:
                m = re.search(r'L(\d+)', anchor)
                if m:
                    self.line_to_pred_id[int(m.group(1))] = pid

        # 构建输入变量 -> 分支谓词 的传递链路
        # 基于静态数据依赖做反向 BFS：从谓词条件变量出发，逆向追踪到输入变量
        input_vars = set()
        try:
            from static_analyzer import StaticAnalyzer as _SA
            _sa = _SA(source_code=self.source_code)
            _inputs = _sa.identify_input_structure().get('inputs', [])
            input_vars = {inp['variable'] for inp in _inputs if inp.get('variable')}
        except Exception:
            pass

        # data_deps: {var -> [依赖的变量列表]}，即 var 的值来自哪些变量
        # 构建正向依赖图 forward_deps: {src -> set(dst)}
        forward_deps: Dict[str, set] = defaultdict(set)
        for var, deps_list in data_deps.items():
            for dep in deps_list:
                forward_deps[dep].add(var)

        for pred in predicates:
            pid = pred.get('pred_id', '')
            raw_var = pred.get('var', '')
            expr    = pred.get('expression', '')
            cond_tokens = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', raw_var + ' ' + expr))
            # 对 key_variables 中非输入变量，若它在当前谓词的表达式中依赖输入变量
            # 则建立：输入变量 -> 该非输入变量 的正向依赖
            for kv in list(self.key_variables):
                if kv in cond_tokens and kv not in input_vars:
                    for iv in input_vars:
                        # 检查 kv 是否能从 iv 到达（已有依赖链）
                        # 用BFS 检查
                        from collections import deque as _dq
                        visited_check = {iv}
                        q_check = _dq([iv])
                        reachable = False
                        while q_check:
                            cur = q_check.popleft()
                            if cur == kv:
                                reachable = True
                                break
                            for nxt in forward_deps.get(cur, set()):
                                if nxt not in visited_check:
                                    visited_check.add(nxt)
                                    q_check.append(nxt)
                        if reachable:
                            # kv 已可达，跳过
                            pass

        # 对每个谓词，从其条件变量反向追踪到输入变量，记录链路
        self.pred_input_chains: Dict[str, List[Dict]] = {}
        self.input_vars = input_vars  # 供运行时动态追踪使用
        self._static_forward_deps = forward_deps  # 供运行时补充

        for pred in predicates:
            pid = pred.get('pred_id', '')
            if not pid:
                continue
            cond_vars = set()
            raw_var = pred.get('var', '')
            for token in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', raw_var):
                cond_vars.add(token)
            for token in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*',
                                    pred.get('expression', '')):
                cond_vars.add(token)

            chains = []
            for inp_var in input_vars:
                from collections import deque
                queue = deque([(inp_var, [inp_var])])
                visited = {inp_var}
                found_chains = []
                while queue:
                    cur, path = queue.popleft()
                    if cur in cond_vars and cur != inp_var:
                        found_chains.append(path[:])
                        continue
                    for nxt in forward_deps.get(cur, set()):
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append((nxt, path + [nxt]))
                if inp_var in cond_vars:
                    found_chains.append([inp_var])
                for chain in found_chains:
                    chains.append({
                        'input_var': inp_var,
                        'chain'    : chain,
                        'affects'  : chain[-1],
                    })

            self.pred_input_chains[pid] = chains

        # 所有分支节点集合用于增量统计
        self.all_branch_nodes = {}
        for branch in cfg.get('branches', []):
            node_id = branch.get('node_id')
            if node_id:
                self.all_branch_nodes[node_id] = {
                    'pred_id': branch.get('pred_id', ''),
                    'lineno' : branch.get('lineno'),
                    'type'   : branch.get('type', ''),
                    'branch_id': '',  # 由下方 branch_constraint_map 填充
                }

        # 从静态分析 branch_constraint_map 获取统一的 B1/B2/B3 标识
        # 建立 cfg_node_id -> branch_id 映射
        self.node_to_branch_id = {}  # node_id -> 'B1'/'B2'...
        branch_map = self.static_result.get('branch_constraint_map', {})
        if not branch_map:
            # 若调用方未提前传入，尝试从静态结果中获取
            try:
                from static_analyzer import StaticAnalyzer
                sa = StaticAnalyzer(source_code=self.source_code)
                branch_map = sa.get_branch_constraint_map()
            except Exception:
                branch_map = {}
        for bid, binfo in branch_map.items():
            nid = binfo.get('cfg_node_id', '')
            if nid:
                self.node_to_branch_id[nid] = bid
                if nid in self.all_branch_nodes:
                    self.all_branch_nodes[nid]['branch_id'] = bid

        # 全部 CFG 边
        self.all_cfg_edges = {}
        for edge in cfg.get('edges', []):
            fn    = edge.get('from', '')
            tn    = edge.get('to', '')
            label = edge.get('label', '')
            if fn and tn:
                self.all_cfg_edges[(fn, tn)] = {
                    'label'  : label,
                    'from'   : fn,
                    'to'     : tn,
                }

        # 全部 CFG 节点（用于增量块覆盖统计，与总结中块覆盖率一致）
        self.all_cfg_nodes = set(self.cfg_nodes)  # cfg_nodes 已含所有节点

        # 跨轮次增量覆盖状态
        self._global_covered_edges  = set()   # set of (from, to)
        self._global_covered_blocks = set()   # set of node_id
        # 兼容旧字段
        self._global_covered_branches     = set()
        self._global_covered_branch_edges = set()
    
    def _reset_results(self):
        """重置动态分析结果"""
        self._trace_records = []
        self._covered_edges = set()
        self._covered_blocks = set()
        self._variable_value_sequences = defaultdict(list)
        self._branch_distances = defaultdict(list)
        self._runtime_types = defaultdict(set)
        self._output_buffer = []
        self._execution_steps = 0
        self._last_line = None
    
    def run_with_input(self, input_data: Any = None) -> Dict[str, Any]:
        """
        使用指定输入执行程序并收集动态分析信息
        
        Args:
            input_data: 输入数据（单个值或列表）
            
        Returns:
            包含所有动态分析结果的字典
        """
        self._reset_results()
        
        # 将源代码加载到linecache，以便在追踪时能读取代码行
        filename = '<dynamic>'
        lines = self.source_code.splitlines(keepends=True)
        linecache.cache[filename] = (
            len(self.source_code),
            None,
            lines,
            filename
        )
        
        # 创建安全命名空间
        safe_builtins = self._create_safe_builtins(input_data)
        global_ns = {'__builtins__': safe_builtins, '__name__': '__main__'}
        local_ns = {}
        
        # 编译代码
        try:
            compiled_code = compile(self.source_code, filename, 'exec')
        except SyntaxError as e:
            linecache.clearcache()
            return self._error_result(f"语法错误: {e}")
        
        # 追踪函数
        analyzer = self
        def trace_function(frame, event, arg):
            if event == 'line':
                analyzer._trace_line(frame, local_ns, global_ns)
            return trace_function
        
        # 执行
        sys.settrace(trace_function)
        start_time = time.time()
        success = False
        error = None
        
        try:
            exec(compiled_code, global_ns, local_ns)
            success = True
        except Exception as e:
            error = traceback.format_exc()
        finally:
            sys.settrace(None)
            linecache.clearcache()
        
        execution_time = time.time() - start_time
        type_validation = self._validate_types()
        
        # 增量覆盖统计
        # 边级：本轮覆盖的边和全局已覆盖边
        this_covered_edges  = self._covered_edges  & set(self.all_cfg_edges.keys())
        newly_covered_edges = this_covered_edges - self._global_covered_edges
        self._global_covered_edges |= this_covered_edges

        # 块级：本轮覆盖的块和全局已覆盖块
        this_covered_blocks  = self._covered_blocks & self.all_cfg_nodes
        newly_covered_blocks = this_covered_blocks - self._global_covered_blocks
        self._global_covered_blocks |= this_covered_blocks

        total_edge_count    = len(self.all_cfg_edges)
        covered_edge_count  = len(self._global_covered_edges)
        total_block_count   = len(self.all_cfg_nodes)
        covered_block_count = len(self._global_covered_blocks)
        edge_coverage_rate  = round(covered_edge_count  / total_edge_count,  4) if total_edge_count  else 1.0
        block_coverage_rate = round(covered_block_count / total_block_count, 4) if total_block_count else 1.0

        uncovered_edges  = set(self.all_cfg_edges.keys()) - self._global_covered_edges
        uncovered_blocks = self.all_cfg_nodes - self._global_covered_blocks

        def _edge_detail(edge_set):
            detail = []
            for (fn, tn) in sorted(edge_set):
                info = self.all_cfg_edges.get((fn, tn), {})
                is_branch = fn in self.all_branch_nodes
                detail.append({
                    'from'      : fn,
                    'to'        : tn,
                    'label'     : info.get('label', ''),
                    'is_branch' : is_branch,
                    'branch_id' : self.node_to_branch_id.get(fn, '') if is_branch else '',
                    'lineno'    : self.cfg_node_to_line.get(fn) or 'virtual',
                })
            return detail

        def _block_detail(node_set):
            detail = []
            for nid in sorted(node_set):
                is_branch = nid in self.all_branch_nodes
                detail.append({
                    'node_id'   : nid,
                    'is_branch' : is_branch,
                    'branch_id' : self.node_to_branch_id.get(nid, '') if is_branch else '',
                    'lineno'    : self.cfg_node_to_line.get(nid) or 'virtual',
                })
            return detail

        # 运行时动态补充 pred_input_chains
        # 利用执行轨迹中变量值序列，推断哪些输入变量在谓词求值前发生了变化
        runtime_chains: Dict[str, List[Dict]] = {}
        for pred in self.static_result.get('predicates', []):
            pid = pred.get('pred_id', '')
            if not pid:
                continue
            static_chains = self.pred_input_chains.get(pid, [])
            cond_vars = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*',
                            pred.get('var', '') + ' ' + pred.get('expression', '')))
            pred_steps = [r['step'] for r in self._branch_distances.get(pid, [])]
            extra_chains = []
            for iv in self.input_vars:
                already = any(c['input_var'] == iv for c in static_chains)
                if already:
                    continue
                iv_steps = [r['step'] for r in self._variable_value_sequences.get(iv, [])]
                if not iv_steps or not pred_steps:
                    continue
                if min(iv_steps) <= max(pred_steps):
                    mid_chain = [iv]
                    for cv in sorted(cond_vars):
                        if cv == iv:
                            continue
                        cv_steps = [r['step'] for r in self._variable_value_sequences.get(cv, [])]
                        if cv_steps and min(iv_steps) <= min(cv_steps) <= max(pred_steps):
                            mid_chain.append(cv)
                    extra_chains.append({
                        'input_var': iv,
                        'chain'    : mid_chain,
                        'affects'  : mid_chain[-1],
                        'source'   : 'runtime',
                    })
            runtime_chains[pid] = static_chains + extra_chains

        #动态分析补充后变量的传递链路
        merged_chains = {
            pid: runtime_chains.get(pid, self.pred_input_chains.get(pid, []))
            for pid in set(list(self.pred_input_chains.keys()) + list(runtime_chains.keys()))
        }

        return {
            'execution_info': {
                'success'        : success,
                'output'         : '\n'.join(self._output_buffer),
                'error'          : error,
                'execution_steps': self._execution_steps,
                'execution_time' : execution_time
            },
            'trace': {
                'records'   : self._trace_records,
                'step_count': len(self._trace_records)
            },
            'coverage': {
                # 本轮原始覆盖
                'covered_edges'      : list(self._covered_edges),
                'covered_blocks'     : list(self._covered_blocks),
                'edge_count'         : len(self._covered_edges),
                'block_count'        : len(self._covered_blocks),
                # 增量统计
                'total_edges'        : total_edge_count,
                'total_blocks'       : total_block_count,
                'covered_edge_count' : covered_edge_count,
                'covered_block_count': covered_block_count,
                'edge_coverage_rate' : edge_coverage_rate,
                'block_coverage_rate': block_coverage_rate,
                'newly_covered_edges'  : _edge_detail(newly_covered_edges),
                'newly_covered_blocks' : _block_detail(newly_covered_blocks),
                'uncovered_edges'      : _edge_detail(uncovered_edges),
                'uncovered_blocks'     : _block_detail(uncovered_blocks),
            },
            'variable_values'    : dict(self._variable_value_sequences),
            'branch_distances'  : dict(self._branch_distances),
            'pred_input_chains' : merged_chains,    #变量传递的链路
            'type_validation' : type_validation
        }
    
    def _error_result(self, error_msg):
        """返回错误结果"""
        return {
            'execution_info': {'success': False, 'error': error_msg, 'execution_steps': 0},
            'trace': {'records': [], 'step_count': 0},
            'coverage': {'covered_edges': [], 'covered_blocks': [], 'edge_count': 0, 'block_count': 0},
            'variable_values': {},
            'branch_distances': {},
            'type_validation': {}
        }
    
    def _trace_line(self, frame, local_ns, global_ns):
        """追踪行执行"""
        self._execution_steps += 1
        if self._execution_steps > self.max_execution_steps:
            raise RuntimeError(f"执行步骤超过最大限制")
        
        lineno = frame.f_lineno
        filename = frame.f_code.co_filename
        
        if '<dynamic>' not in filename:
            return
        
        # 捕获关键变量
        vars_snapshot = {}
        for var in self.key_variables:
            if var in frame.f_locals:
                vars_snapshot[var] = self._safe_value(frame.f_locals[var])
            elif var in frame.f_globals:
                vars_snapshot[var] = self._safe_value(frame.f_globals[var])
        
        # 记录轨迹（只记录有变量变化的步骤）
        changed_vars = {
            var: val for var, val in vars_snapshot.items()
            if not self._variable_value_sequences[var]
            or self._variable_value_sequences[var][-1]['value'] != val
        }
        if changed_vars:
            self._trace_records.append({
                'lineno': lineno,
                'step'  : self._execution_steps,
                'vars'  : changed_vars
            })

        # 记录变量值序列（只在值发生变化时追加）
        for var, val in vars_snapshot.items():
            seq = self._variable_value_sequences[var]
            if not seq or seq[-1]['value'] != val:
                seq.append({
                    'step'  : self._execution_steps,
                    'lineno': lineno,
                    'value' : val
                })
            if val is not None:
                self._runtime_types[var].add(type(val).__name__)

        # 更新CFG覆盖
        self._update_cfg_coverage(lineno)
        
        # 检查分支条件并计算分支距离
        self._check_branch_condition(lineno, frame)
        
        self._last_line = lineno
    
    def _update_cfg_coverage(self, current_line):
        """更新CFG边覆盖"""
        # 记录当前行覆盖的节点
        if current_line in self.cfg_line_to_nodes:
            for node in self.cfg_line_to_nodes[current_line]:
                self._covered_blocks.add(node)
        
        if self._last_line is None:
            return
        
        # 策略1: 精确匹配 - 查找从上一行到当前行的边
        last_nodes = self.cfg_line_to_nodes.get(self._last_line, [])
        current_nodes = self.cfg_line_to_nodes.get(current_line, [])
        
        matched = False
        for from_node in last_nodes:
            for to_node in current_nodes:
                if (from_node, to_node) in self.cfg_edges:
                    self._covered_edges.add((from_node, to_node))
                    self._covered_blocks.add(from_node)
                    self._covered_blocks.add(to_node)
                    matched = True
        
        if matched:
            return
        
        # 策略2: 间接匹配 - 查找通过中间节点连接的边
        # 例如: last_line(3) -> virtual_node -> current_line(4)
        for from_node in last_nodes:
            # 查找从from_node出发的所有边
            for edge_from, edge_to in self.cfg_edges:
                if edge_from == from_node:
                    # 检查edge_to是否能到达current_nodes
                    if edge_to in current_nodes:
                        self._covered_edges.add((edge_from, edge_to))
                        self._covered_blocks.add(edge_from)
                        self._covered_blocks.add(edge_to)
                        matched = True
                    else:
                        # 检查edge_to是否是虚拟节点，且能到达current_line
                        for edge_from2, edge_to2 in self.cfg_edges:
                            if edge_from2 == edge_to and edge_to2 in current_nodes:
                                # 找到了路径: from_node -> edge_to -> edge_to2
                                self._covered_edges.add((edge_from, edge_to))
                                self._covered_edges.add((edge_from2, edge_to2))
                                self._covered_blocks.add(edge_from)
                                self._covered_blocks.add(edge_to)
                                self._covered_blocks.add(edge_to2)
                                matched = True
        
        if matched:
            return
        
        # 策略3: 全局搜索 - 记录所有可能的边
        # 这是最后的兜底策略，标记所有边为潜在覆盖
        for edge_from, edge_to in self.cfg_edges:
            from_line = self.cfg_node_to_line.get(edge_from)
            to_line = self.cfg_node_to_line.get(edge_to)
            
            # 如果边的起点或终点匹配执行的行号，标记为覆盖
            if from_line == self._last_line or to_line == current_line:
                self._covered_edges.add((edge_from, edge_to))
                self._covered_blocks.add(edge_from)
                self._covered_blocks.add(edge_to)
    
    def _check_branch_condition(self, lineno, frame):
        """检查分支条件并计算分支距离"""
        try:
            line = linecache.getline(frame.f_code.co_filename, lineno).strip()

            is_while = line.startswith('while ')
            is_for   = line.startswith('for ')
            is_if    = line.startswith(('if ', 'elif '))

            if not (is_while or is_for or is_if):
                return

            condition = self._extract_condition(line)
            if not condition:
                return

            if is_for:
                distance  = self._calculate_for_loop_distance(line, frame)
                loop_type = 'for'
                # for 循环条件显示为 iterable 表达式
                import re as _re2
                _m = _re2.match(r'for\s+.+\s+in\s+(.+?)\s*:', line)
                condition = _m.group(1).strip() if _m else condition
            else:
                distance  = self._calculate_branch_distance(condition, frame)
                loop_type = 'while' if is_while else 'if'

            if distance is not None:
                key = self.line_to_pred_id.get(lineno, lineno)
                self._branch_distances[key].append({
                    'step'     : self._execution_steps,
                    'pred_id'  : self.line_to_pred_id.get(lineno, ''),
                    'lineno'   : lineno,
                    'condition': condition,
                    'distance' : distance,
                    'evaluated': distance == 0.0,
                    'loop_type': loop_type,
                })
        except Exception:
            pass

    def _calculate_for_loop_distance(self, line, frame):
        """
        计算 for 循环的分支距离。
        for var in iterable:
            - iterable 非空（还有元素可迭代）：距离 = 0
            - iterable 耗尽：距离 = 1
        """
        try:
            import re as _re
            m = _re.match(r'for\s+.+\s+in\s+(.+)\s*:', line)
            if not m:
                return None
            iterable_expr = m.group(1).strip()
            local_vars  = frame.f_locals
            global_vars = frame.f_globals
            iterable_val = eval(iterable_expr, global_vars, local_vars)
            try:
                remaining = len(iterable_val)
            except TypeError:
                remaining = sum(1 for _ in iterable_val)
            return 0.0 if remaining > 0 else 1.0
        except Exception:
            return None
    
    def _extract_condition(self, line):
        """从代码行中提取条件表达式"""
        line = line.strip()
        
        # 移除关键字
        for keyword in ['if ', 'elif ', 'while ']:
            if line.startswith(keyword):
                line = line[len(keyword):]
                break
        
        # 移除冒号及之后的内容
        if ':' in line:
            line = line.split(':', 1)[0]
        
        return line.strip()
    
    def _calculate_branch_distance(self, condition, frame):
        """
        计算分支距离
        
        分支距离定义：
        - 条件为真：距离 = 0
        - 条件为假：根据条件类型计算距离
        """
        try:
            # 获取局部变量
            local_vars = {k: v for k, v in frame.f_locals.items() if not k.startswith('__')}
            global_vars = frame.f_globals
            
            # 评估条件
            try:
                result = eval(condition, global_vars, local_vars)
            except:
                return None
            
            # 如果条件为真，距离为0
            if result:
                return 0.0
            
            # 条件为假，计算距离
            return self._compute_false_branch_distance(condition, local_vars, global_vars)
            
        except Exception as e:
            return None
    
    def _compute_false_branch_distance(self, condition, local_vars, global_vars):
        """计算条件为假时的分支距离"""
        # 处理简单比较表达式
        # 模式1: var op const (如 x > 10)
        # 模式2: const op var (如 10 < x)
        # 模式3: const1 op var op const2 (如 10 < x < 20)
        
        # 尝试匹配链式比较 (如 10 < x < 20)
        chain_pattern = r'([\d.]+)\s*([<>]=?)\s*(\w+)\s*([<>]=?)\s*([\d.]+)'
        match = re.search(chain_pattern, condition)
        if match:
            const1, op1, var_name, op2, const2 = match.groups()
            const1, const2 = float(const1), float(const2)
            
            if var_name in local_vars:
                var_val = local_vars[var_name]
                if isinstance(var_val, (int, float)):
                    # 计算到满足条件的最小距离
                    dist1 = self._calc_single_comparison_distance(var_val, op1, const1, reversed=True)
                    dist2 = self._calc_single_comparison_distance(var_val, op2, const2, reversed=False)
                    return min(dist1, dist2) if dist1 is not None and dist2 is not None else None
        
        # 尝试匹配简单比较
        patterns = [
            (r'(\w+)\s*([<>]=?|==|!=)\s*([\d.]+)', False),  # var op const
            (r'([\d.]+)\s*([<>]=?|==|!=)\s*(\w+)', True)    # const op var
        ]
        
        for pattern, reversed_order in patterns:
            match = re.search(pattern, condition)
            if match:
                groups = match.groups()
                if reversed_order:
                    const, op, var_name = float(groups[0]), groups[1], groups[2]
                else:
                    var_name, op, const = groups[0], groups[1], float(groups[2])
                
                if var_name in local_vars:
                    var_val = local_vars[var_name]
                    if isinstance(var_val, (int, float)):
                        return self._calc_single_comparison_distance(var_val, op, const, reversed_order)
        
        # 处理布尔变量
        bool_pattern = r'^(\w+)$'
        match = re.match(bool_pattern, condition.strip())
        if match:
            var_name = match.group(1)
            if var_name in local_vars:
                val = local_vars[var_name]
                # 如果是布尔值或可转换为布尔值
                return 0.0 if val else 1.0
        
        # 处理取反 (not x)
        not_pattern = r'^not\s+(\w+)$'
        match = re.match(not_pattern, condition.strip())
        if match:
            var_name = match.group(1)
            if var_name in local_vars:
                val = local_vars[var_name]
                return 1.0 if val else 0.0
        
        # 处理模运算 (如 h % 11 == 0)
        mod_pattern = r'(\w+)\s*%\s*([\d.]+)\s*==\s*([\d.]+)'
        match = re.search(mod_pattern, condition)
        if match:
            var_name, divisor, remainder = match.groups()
            divisor, remainder = float(divisor), float(remainder)
            if var_name in local_vars:
                var_val = local_vars[var_name]
                if isinstance(var_val, (int, float)):
                    actual_remainder = var_val % divisor
                    return abs(actual_remainder - remainder)

        # 处理复合条件 and / or（适用于 while 和 if）
        # and：距离 = 各子条件距离之和（所有子条件都要为真）
        # or ：距离 = 各子条件距离的最小值（任一为真即可）
        cond_stripped = condition.strip()
        if ' and ' in cond_stripped:
            parts = cond_stripped.split(' and ')
            dists = [self._compute_false_branch_distance(p.strip(), local_vars, global_vars)
                    for p in parts]
            valid = [d for d in dists if d is not None]
            if valid:
                return sum(valid)
        if ' or ' in cond_stripped:
            parts = cond_stripped.split(' or ')
            dists = [self._compute_false_branch_distance(p.strip(), local_vars, global_vars)
                    for p in parts]
            valid = [d for d in dists if d is not None]
            if valid:
                return min(valid)

        # 默认距离
        return 1.0
    
    def _calc_single_comparison_distance(self, var_val, op, const, reversed_order=False):
        """计算单个比较表达式的距离"""
        if reversed_order:
            # const op var，需要反转操作符
            if op == '<':
                op = '>'
            elif op == '>':
                op = '<'
            elif op == '<=':
                op = '>='
            elif op == '>=':
                op = '<='
        
        if op == '<':
            # var < const，当前为假，说明 var >= const
            # 需要 var 减小到 < const，距离 = var - const + 1
            return max(0, var_val - const + 1)
        elif op == '<=':
            # var <= const，当前为假，说明 var > const
            # 需要 var 减小到 <= const，距离 = var - const
            return max(0, var_val - const)
        elif op == '>':
            # var > const，当前为假，说明 var <= const
            # 需要 var 增大到 > const，距离 = const - var + 1
            return max(0, const - var_val + 1)
        elif op == '>=':
            # var >= const，当前为假，说明 var < const
            # 需要 var 增大到 >= const，距离 = const - var
            return max(0, const - var_val)
        elif op == '==':
            # var == const，当前为假
            # 距离 = |var - const|
            return abs(var_val - const)
        elif op == '!=':
            # var != const，当前为假，说明 var == const
            # 距离 = 0（已经相等，只需要改变条件）
            return 1.0
        
        return None
    
    def _safe_value(self, val):
        """安全获取值的表示"""
        try:
            if isinstance(val, (int, float, str, bool, type(None))):
                return val
            elif isinstance(val, (list, tuple)):
                return f"{type(val).__name__}[{len(val)}]" if len(val) > 10 else val
            elif isinstance(val, dict):
                return f"dict[{len(val)}]" if len(val) > 10 else val
            else:
                return f"<{type(val).__name__}>"
        except:
            return "<unprintable>"
    
    def _extract_line_from_node(self, node_id):
        """从节点ID提取行号"""
        match = re.search(r'L(\d+)', str(node_id))
        return int(match.group(1)) if match else None
    
    def _validate_types(self):
        """验证运行时类型"""
        confirmed, corrected, new_types = {}, {}, {}
        
        for var, runtime_type_set in self._runtime_types.items():
            runtime_types = list(runtime_type_set)
            static_type = self.static_types.get(var)
            
            if not static_type:
                new_types[var] = runtime_types[0] if len(runtime_types) == 1 else runtime_types
            elif len(runtime_types) == 1 and runtime_types[0] == static_type:
                confirmed[var] = static_type
            else:
                corrected[var] = {
                    'static': static_type,
                    'runtime': runtime_types[0] if len(runtime_types) == 1 else runtime_types
                }
        
        return {
            'confirmed_types': confirmed,
            'corrected_types': corrected,
            'new_types': new_types
        }
    
    def _create_safe_builtins(self, input_data):
        """创建安全的builtins"""
        safe_builtins = {}
        SAFE_BUILTINS = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'divmod',
            'enumerate', 'filter', 'float', 'format', 'hex', 'int', 'iter',
            'len', 'list', 'map', 'max', 'min', 'oct', 'ord', 'pow', 'range',
            'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str',
            'sum', 'tuple', 'type', 'zip', 'eval'
        }
        
        for name in SAFE_BUILTINS:
            if name in builtins.__dict__:
                safe_builtins[name] = builtins.__dict__[name]
        
        def safe_print(*args, **kwargs):
            self._output_buffer.append(' '.join(str(a) for a in args))
        safe_builtins['print'] = safe_print
        
        if input_data is not None:
            input_iter = iter(input_data) if isinstance(input_data, list) else iter([input_data])
            def mock_input(prompt=""):
                try:
                    return str(next(input_iter))
                except StopIteration:
                    raise RuntimeError("输入数据不足")
            safe_builtins['input'] = mock_input
        
        return safe_builtins
    
    def run_multiple_inputs(self, inputs_list: List[Any]) -> List[Dict[str, Any]]:
        """使用多个输入执行程序"""
        return [self.run_with_input(input_data) for input_data in inputs_list]
    
    def aggregate_coverage(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多次执行的覆盖信息"""
        total_edge_count    = len(self.all_cfg_edges)
        total_block_count   = len(self.all_cfg_nodes)
        covered_edges       = self._global_covered_edges
        covered_blocks      = self._global_covered_blocks
        uncovered_edges     = set(self.all_cfg_edges.keys()) - covered_edges
        uncovered_blocks    = self.all_cfg_nodes - covered_blocks

        def _edge_detail(edge_set):
            return [
                {
                    'from'      : fn,
                    'to'        : tn,
                    'label'     : self.all_cfg_edges.get((fn, tn), {}).get('label', ''),
                    'is_branch' : fn in self.all_branch_nodes,
                    'branch_id' : self.node_to_branch_id.get(fn, '') if fn in self.all_branch_nodes else '',
                    'lineno'    : self.cfg_node_to_line.get(fn) or 'virtual',
                }
                for fn, tn in sorted(edge_set)
            ]

        def _block_detail(node_set):
            return [
                {
                    'node_id'   : nid,
                    'is_branch' : nid in self.all_branch_nodes,
                    'branch_id' : self.node_to_branch_id.get(nid, '') if nid in self.all_branch_nodes else '',
                    'lineno'    : self.cfg_node_to_line.get(nid) or 'virtual',
                }
                for nid in sorted(node_set)
            ]


        return {
            'total_edges'        : total_edge_count,
            'total_blocks'       : total_block_count,
            'covered_edge_count' : len(covered_edges),
            'covered_block_count': len(covered_blocks),
            'edge_coverage_rate' : round(len(covered_edges)  / total_edge_count,  4) if total_edge_count  else 1.0,
            'block_coverage_rate': round(len(covered_blocks) / total_block_count, 4) if total_block_count else 1.0,
            'covered_edges'      : _edge_detail(covered_edges),
            'uncovered_edges'    : _edge_detail(uncovered_edges),
            'covered_blocks'     : _block_detail(covered_blocks),
            'uncovered_blocks'   : _block_detail(uncovered_blocks),
        }
