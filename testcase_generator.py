#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试用例生成器

算法流程：
1. 静态分析 -> 获取分支列表、输入变量、边界条件
2. 生成初始种子（边界值种子 + 模板种子）
3. 初始化种群（每个个体 = 一个测试用例）
4. 循环直到满足终止条件：
    a. 执行所有测试用例，收集覆盖信息和分支距离
    b. 计算每个个体的适应度
    c. 选择父代（锦标赛选择）
    d. 交叉产生子代（序列交叉）
    e. 变异产生新个体（引导变异）
    f. 合并种群，保留精英
    g. 若某分支长期未覆盖，调用约束求解
5. 精简测试集
6. 返回测试用例集
"""

import os
import re
import ast
import random
import copy
import xml.etree.ElementTree as ET
from itertools import product
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set

from static_analyzer import StaticAnalyzer
from dynamic_analyzer import DynamicAnalyzer
from advanced_executor import AdvancedExecutor


# utils

def _parse_xml(xml_file: str) -> List[Dict]:
    """从 XML 文件解析测试用例"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    test_cases = []
    for i, node in enumerate(root, 1):
        if node.tag.startswith('testData'):
            inp = node.find('input')
            outp = node.find('output')
            if inp is not None and outp is not None:
                input_str = (inp.text or '').strip()
                output_str = (outp.text or '').strip()
                test_cases.append({
                    'id': i,
                    'input': input_str.split('\n') if input_str else [],
                    'input_display': input_str,
                    'expected_output': output_str,
                })
    return test_cases


def _value_to_input_line(value: Any, fmt: str) -> str:
    """根据输入格式将 Python 值转为字符串"""
    if fmt == 'evaluated':
        return repr(value)
    if fmt == 'single_value':
        return str(value)
    if fmt in ('list', 'multi_value', 'split', 'iterator', 'split_string'):
        if isinstance(value, (list, tuple)):
            return ' '.join(str(v) for v in value)
        return str(value)
    return repr(value)


def _build_input_lines(value: Any, input_structure: Dict) -> List[str]:
    """将候选值转换为 run_with_input 所需的字符串列表"""
    inputs = input_structure.get('inputs', [])
    if not inputs:
        return [repr(value)]
    if len(inputs) == 1:
        return [_value_to_input_line(value, inputs[0].get('format', 'evaluated'))]
    if isinstance(value, (list, tuple)) and len(value) == len(inputs):
        grouped_lines: List[str] = []
        idx = 0
        while idx < len(inputs):
            inp = inputs[idx]
            fmt = inp.get('format', 'evaluated')
            entry_point = inp.get('entry_point')
            line_no = inp.get('line')
            if fmt in ('iterator', 'multi_value', 'split', 'split_string'):
                parts = [str(value[idx])]
                j = idx + 1
                while j < len(inputs):
                    other = inputs[j]
                    other_fmt = other.get('format', 'evaluated')
                    if other_fmt != fmt:
                        break
                    if other.get('entry_point') != entry_point or other.get('line') != line_no:
                        break
                    parts.append(str(value[j]))
                    j += 1
                grouped_lines.append(' '.join(parts))
                idx = j
                continue
            grouped_lines.append(_value_to_input_line(value[idx], fmt))
            idx += 1
        return grouped_lines
    return [_value_to_input_line(value, inputs[0].get('format', 'evaluated'))]


def _run_one(da: DynamicAnalyzer, inp: Any, input_structure: Dict = None) -> Dict:
    """执行单个输入，返回动态分析结果"""
    if isinstance(inp, list) and inp and isinstance(inp[0], str):
        inp_lines = inp
    elif input_structure:
        inp_lines = _build_input_lines(inp, input_structure)
    else:
        inp_lines = [repr(inp)]
    return da.run_with_input(inp_lines)


def _normalize_distance(d: float) -> float:
    """将分支距离标准化到 [0,1]，d=0 返回 0"""
    if d is None:
        return 1.0
    if d <= 0.0:
        return 0.0
    return d / (d + 1.0)


class TestcaseGenerator:
    """
    基于遗传算法 + 分支距离的测试用例生成器。
    每个「个体」是一个测试输入值（单值或 tuple 对应多个 input()）。
    """

    def __init__(
        self,
        source_code: str = None,
        *,
        file_path: str = None,
        xml_file: str = None,
        population_size: int = 30,
        max_generations: int = 50,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.3,
        elite_ratio: float = 0.1,
        stagnation_limit: int = 10,
        timeout: float = 10.0,
        random_seed: int = None,
    ):
        if source_code is not None:
            self.source_code = source_code
        elif file_path is not None:
            with open(file_path, 'r', encoding='utf-8') as _f:
                self.source_code = _f.read()
        else:
            raise ValueError("必须提供 source_code 或 file_path 参数")
        self.xml_file = xml_file
        self.pop_size = population_size
        self.max_gen = max_generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = max(1, int(population_size * elite_ratio))
        self.stagnation_limit = stagnation_limit
        self.timeout = timeout

        if random_seed is not None:
            random.seed(random_seed)

        self._sa = StaticAnalyzer(source_code=self.source_code)
        self._static_result: Dict = {}
        self._branches: List[Dict] = []
        self._input_structure: Dict = {}
        self._input_vars: List[Dict] = []
        self._predicates: List[Dict] = []
        self._mutation_candidates: Dict = {}
        self._pred_map: Dict[str, Dict] = {}
        self._da: Optional[DynamicAnalyzer] = None
        self._cov_key_to_branch_id: Dict[str, str] = {}
        self._covered_branches: Set[str] = set()
        self._stagnation_counter: Dict[str, int] = defaultdict(int)
        self._best_distance: Dict[str, float] = {}

        self._base_dir = os.path.dirname(os.path.abspath(__file__))
        self._problem_id = self._detect_problem_id(file_path, xml_file)
        self._reference_code = self._load_reference_code(self._problem_id)
        self._reference_executor = AdvancedExecutor(timeout=self.timeout, use_subprocess=False) if self._reference_code else None
        self._reference_result_cache: Dict[str, Dict[str, Any]] = {}
        self._expected_output_cache: Dict[str, str] = {}
        self._legal_input_cache: Dict[str, bool] = {}
        self._output_tolerance = 1e-3
        self._failure_bonus = 2.5
        self._failure_diff_weight = 1.0

        self._input_mutation_budget = max(6, population_size // 2)
        self._hard_min_valid_cases = 5
        self._supplement_retry_budget = max(40, population_size * 3)

    def _problem_numeric_bounds(self, inp: Dict, idx: int) -> Optional[tuple]:
        """按题目为数值输入提供轻量范围约束。"""
        if self._problem_id == '03039' and inp.get('type', 'Any') == 'int':
            if idx in (0, 1):
                return (1, 40)
            if idx == 2:
                return (2, 1600)
        return None

    def _clamp_problem_value(self, value: Any, inp: Dict, idx: int) -> Any:
        """将数值限制在题目感知范围内。"""
        bounds = self._problem_numeric_bounds(inp, idx)
        if bounds is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            return value
        low, high = bounds
        value = max(low, min(high, value))
        if inp.get('type', 'Any') == 'int':
            return int(round(value))
        return value

    def _normalize_individual(self, individual: Any) -> Any:
        """对个体应用题目感知范围约束。"""
        if not self._input_vars:
            return individual
        if isinstance(individual, tuple):
            values = list(individual)
            for idx, inp in enumerate(self._input_vars[:len(values)]):
                values[idx] = self._clamp_problem_value(values[idx], inp, idx)
            if self._problem_id == '03039' and len(values) >= 3:
                n = max(1, int(values[0]))
                m = max(1, int(values[1]))
                max_k = max(2, n * m)
                values[2] = max(2, min(max_k, int(values[2])))
            return tuple(values)
        if len(self._input_vars) == 1:
            return self._clamp_problem_value(individual, self._input_vars[0], 0)
        return individual

    def generate(self) -> List[Dict]:
        """
        主入口：执行完整测试用例生成流程。

        Returns:
            精简后的测试用例列表，每个元素：
            {
                'input'           : List[str],
                'input_display'   : str,
                'covered_branches': List[str],
                'output'          : str,
                'expected_output' : str,
            }
        """
        # 步骤1：静态分析
        self._run_static_analysis()

        # 步骤2：从 XML 解析初始测试用例（若提供）
        all_test_cases: List[Dict] = []
        xml_individuals: List[Any] = []
        if self.xml_file:
            xml_raw = _parse_xml(self.xml_file)
            for xtc in xml_raw:
                # 直接把原始输入行列表作为「个体」加入种群种子
                inp_lines = xtc.get('input', [])
                if inp_lines:
                    xml_individuals.append(inp_lines)
                # 用动态分析执行一次，收集覆盖信息
                try:
                    if self._da is None:
                        continue
                    res = _run_one(self._da, inp_lines, self._input_structure)
                    expected_output = self._compute_expected_output(inp_lines) or xtc.get('expected_output', '')
                    actual_output = self._actual_output_from_result(res)
                    self._update_covered_branches(res)
                    tc = self._make_test_case(inp_lines, res)
                    if tc:
                        all_test_cases.append(tc)
                    else:
                        all_test_cases.append({
                            'input'           : inp_lines,
                            'input_display'   : xtc.get('input_display', '\n'.join(inp_lines)),
                            'covered_branches': [],
                            'output'          : actual_output,
                            'expected_output' : expected_output,
                            'kills_reference' : bool(expected_output) and not self._is_output_match(actual_output, expected_output),
                            'output_difference_score': self._output_difference_score(actual_output, expected_output),
                        })
                except Exception:
                    pass

        # 步骤3：生成初始种群（含 XML 种子）
        population = self._init_population(xml_individuals)

        # 步骤4：GA 主循环
        for _gen in range(self.max_gen):
            #执行所有测试用例，收集覆盖信息和分支距离
            results = self._evaluate_population(population)

            # 收集本代测试用例
            for ind, res in zip(population, results):
                tc = self._make_test_case(ind, res)
                if tc:
                    all_test_cases.append(tc)

            # 计算适应度
            fitness_scores = [self._compute_fitness(res) for res in results]

            # 终止条件：所有分支已覆盖
            uncovered = self._uncovered_branches()
            if not uncovered:
                break

            # 更新停滞计数器
            self._update_stagnation(results)

            #选择父代
            parents = self._tournament_select(population, fitness_scores)

            #交叉 + 引导变异
            offspring = self._crossover_and_mutate(parents, results)
            offspring.extend(self._generate_testcase_mutants(population, results))

            #合并
            population = self._merge_population(population, offspring, fitness_scores)

            #约束求解
            stagnant = [
                bid for bid in uncovered
                if self._stagnation_counter.get(bid, 0) >= self.stagnation_limit
            ]
            if stagnant:
                extra = self._constraint_solve(stagnant)
                if extra:
                    worst_idx = sorted(
                        range(len(fitness_scores)),
                        key=lambda i: -fitness_scores[i]
                    )[:len(extra)]
                    for idx, new_ind in zip(worst_idx, extra):
                        population[idx] = new_ind

        # 最终评估
        final_results = self._evaluate_population(population)
        for ind, res in zip(population, final_results):
            tc = self._make_test_case(ind, res)
            if tc:
                all_test_cases.append(tc)

        # 步骤5：若唯一有效样例不足，则继续定向补生成（关注输入合法性）
        all_test_cases = self._supplement_unique_valid_cases(all_test_cases, population, final_results)

        # 步骤6：按覆盖率 + 边界多样性精简测试集
        minimized = self._greedy_minimize(all_test_cases)
        return minimized

    # 步骤1：静态分析

    def _detect_problem_id(self, file_path: Optional[str], xml_file: Optional[str]) -> Optional[str]:
        """从文件路径中识别题目编号"""
        candidates = [file_path, xml_file]
        for path in candidates:
            if not path:
                continue
            m = re.search(r'(02882|03025|03039|2910|3039|3226)', str(path))
            if m:
                return m.group(1)
        return None

    def _load_reference_code(self, problem_id: Optional[str]) -> Optional[str]:
        """加载题目参考实现代码。"""
        if not problem_id:
            return None
        ref_path = os.path.join(self._base_dir, problem_id, 'reference.py')
        if not os.path.isfile(ref_path):
            return None
        try:
            with open(ref_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

    def _run_reference(self, inp_lines: List[str]) -> Dict:
        """运行参考代码并返回原始结果。"""
        if not self._reference_executor or not self._reference_code:
            return {'success': False, 'output': ''}
        normalized = [str(x) for x in inp_lines]
        input_str = '\n'.join(normalized)
        cached = self._reference_result_cache.get(input_str)
        if cached is not None:
            return dict(cached)
        result = self._reference_executor.execute_with_timeout(self._reference_code, input_str, timeout=self.timeout)
        self._reference_result_cache[input_str] = dict(result)
        return dict(result)

    def _compute_expected_output(self, inp_lines: List[str]) -> str:
        """运行参考代码得到期望输出。"""
        normalized = [str(x) for x in inp_lines]
        key = '\n'.join(normalized)
        cached = self._expected_output_cache.get(key)
        if cached is not None:
            return cached
        res = self._run_reference(normalized)
        output = ''
        if res.get('success', False):
            output = str(res.get('output', '')).strip()
        self._expected_output_cache[key] = output
        return output

    def _is_legal_input(self, inp_lines: List[str]) -> bool:
        """输入是否合法：优先以参考实现可成功执行为准。"""
        normalized = [str(x) for x in inp_lines]
        key = '\n'.join(normalized)
        cached = self._legal_input_cache.get(key)
        if cached is not None:
            return cached
        if self._reference_executor and self._reference_code:
            ref_res = self._run_reference(normalized)
            result = ref_res.get('success', False)
            self._legal_input_cache[key] = result
            return result
        try:
            probe = _run_one(self._da, normalized, self._input_structure)
            result = probe.get('execution_info', {}).get('success', False)
            self._legal_input_cache[key] = result
            return result
        except Exception:
            self._legal_input_cache[key] = False
            return False

    def _collect_unique_valid_cases(self, test_cases: List[Dict]) -> List[Dict]:
        """按输入去重，并只保留合法且执行成功的样例。"""
        unique: Dict[str, Dict] = {}
        for tc in test_cases:
            inp_lines = [str(x) for x in tc.get('input', [])]
            if not inp_lines:
                continue
            if not self._is_legal_input(inp_lines):
                continue
            key = '\n'.join(inp_lines)
            current = unique.get(key)
            if current is None or tc.get('output_difference_score', 0.0) > current.get('output_difference_score', 0.0):
                copied = dict(tc)
                copied['input'] = inp_lines
                copied['input_display'] = '\n'.join(inp_lines)
                unique[key] = copied
        return list(unique.values())

    def _candidate_seed_pool(self, population: List[Any], results: List[Dict]) -> List[Any]:
        """为补生成阶段构造高质量种子池。"""
        seeds: List[Any] = []
        for ind, res in zip(population, results):
            if res.get('execution_info', {}).get('success', False):
                seeds.append(copy.deepcopy(ind))
        seeds.extend(self._boundary_seeds())
        seeds.extend(self._template_seeds())
        return seeds if seeds else [self._random_individual()]

    def _is_small_scalar_input_structure(self) -> bool:
        """判断是否属于少量标量输入结构。"""
        if not self._input_vars or len(self._input_vars) > 4:
            return False
        scalar_types = {'int', 'float', 'str', 'Any', 'numeric'}
        scalar_formats = {'single_value', 'evaluated', 'multi_value', 'iterator'}
        for inp in self._input_vars:
            vtype = inp.get('type', 'Any')
            fmt = inp.get('format', 'single_value')
            if 'list' in vtype:
                return False
            if fmt in {'list', 'split', 'split_string'}:
                return False
            if vtype not in scalar_types and fmt not in scalar_formats:
                return False
        return True

    def _individual_to_scalar_vector(self, individual: Any) -> Optional[List[Any]]:
        """将个体转换为按输入位置排列的标量向量。"""
        if not self._input_vars:
            return [individual]
        input_lines = individual if isinstance(individual, list) and individual and all(isinstance(x, str) for x in individual) else _build_input_lines(individual, self._input_structure)
        normalized = [str(x).strip() for x in input_lines]

        expanded: List[str] = []
        line_idx = 0
        idx = 0
        while idx < len(self._input_vars):
            if line_idx >= len(normalized):
                return None
            inp = self._input_vars[idx]
            fmt = inp.get('format', 'single_value')
            entry_point = inp.get('entry_point')
            line_no = inp.get('line')
            current_line = normalized[line_idx]
            if fmt in {'iterator', 'multi_value', 'split', 'split_string'}:
                parts = current_line.split()
                group_size = 1
                j = idx + 1
                while j < len(self._input_vars):
                    other = self._input_vars[j]
                    if other.get('format', 'single_value') != fmt:
                        break
                    if other.get('entry_point') != entry_point or other.get('line') != line_no:
                        break
                    group_size += 1
                    j += 1
                if len(parts) != group_size:
                    return None
                expanded.extend(parts)
                line_idx += 1
                idx = j
                continue
            expanded.append(current_line)
            line_idx += 1
            idx += 1

        if line_idx != len(normalized) or len(expanded) != len(self._input_vars):
            return None

        values: List[Any] = []
        for idx, inp in enumerate(self._input_vars):
            parsed = self._safe_parse_value(expanded[idx])
            if isinstance(parsed, (list, tuple, dict, set)):
                return None
            fmt = inp.get('format', 'single_value')
            vtype = inp.get('type', 'Any')
            if fmt in {'iterator', 'multi_value', 'split', 'split_string'} and 'list' not in vtype:
                values.append(parsed)
            else:
                values.append(self._coerce_value(parsed, inp))
        return values

    def _position_candidate_pools(self, seed_pool: List[Any]) -> List[List[Any]]:
        """为少量标量输入构造按位置分组的候选值池。"""
        pools: List[List[Any]] = [[] for _ in self._input_vars]
        seen_tags = [set() for _ in self._input_vars]

        def add_value(idx: int, val: Any):
            if idx >= len(self._input_vars):
                return
            coerced = self._coerce_value(val, self._input_vars[idx])
            if isinstance(coerced, (list, tuple, dict, set)):
                return
            tag = repr(coerced)
            if tag in seen_tags[idx]:
                return
            seen_tags[idx].add(tag)
            pools[idx].append(coerced)

        generic_numeric = [-1, 0, 1, 2, 3, 5, 10]
        generic_string = ['', 'a', '0', '1', 'test']
        boundary_values = self._collect_boundary_values()[:8]

        for idx, inp in enumerate(self._input_vars):
            vtype = inp.get('type', 'Any')
            if vtype in {'int', 'float', 'Any', 'numeric'}:
                for val in generic_numeric:
                    add_value(idx, val)
                for val in boundary_values:
                    add_value(idx, val)
                    add_value(idx, val - 1)
                    add_value(idx, val + 1)
            if vtype == 'str':
                for val in generic_string:
                    add_value(idx, val)

        for seed in seed_pool:
            vector = self._individual_to_scalar_vector(seed)
            if not vector:
                continue
            for idx, val in enumerate(vector):
                add_value(idx, val)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    add_value(idx, val - 1)
                    add_value(idx, val + 1)

        for branch_entry in self._mutation_candidates.values():
            for branch in branch_entry.get('branches', []):
                for val in branch.get('candidates', []):
                    for idx in range(len(self._input_vars)):
                        add_value(idx, val)

        return [pool[:8] for pool in pools]

    def _scalar_combination_candidates(self, existing_keys: set, seed_pool: List[Any]) -> List[Any]:
        """针对少量标量输入，生成通用位置组合候选。"""
        if not self._is_small_scalar_input_structure():
            return []
        pools = self._position_candidate_pools(seed_pool)
        if not pools or any(not pool for pool in pools):
            return []

        candidates: List[Any] = []
        local_seen = set(existing_keys)
        max_combinations = max(self._supplement_retry_budget * 2, 32)

        for values in product(*pools):
            if len(candidates) >= max_combinations:
                break
            if len(self._input_vars) == 1:
                cand = values[0]
            else:
                cand = tuple(values)
            inp_lines = [str(x) for x in _build_input_lines(cand, self._input_structure)]
            key = '\n'.join(inp_lines)
            if key in local_seen:
                continue
            if not self._is_legal_input(inp_lines):
                continue
            local_seen.add(key)
            candidates.append(cand)
        return candidates

    def _generate_legal_candidate_inputs(self, seed_pool: List[Any], existing_keys: set) -> List[Any]:
        """围绕已有种子生成候选输入，并优先保留新的合法输入。"""
        target_branch = next((b for b in self._branches if b['branch_id'] in set(self._uncovered_branches())), None)
        boundary_values = self._collect_boundary_values()
        candidates: List[Any] = []
        local_seen = set(existing_keys)

        scalar_candidates = self._scalar_combination_candidates(local_seen, seed_pool)
        for cand in scalar_candidates:
            inp_lines = [str(x) for x in _build_input_lines(cand, self._input_structure)]
            key = '\n'.join(inp_lines)
            if key in local_seen:
                continue
            local_seen.add(key)
            candidates.append(cand)
            if len(candidates) >= self._supplement_retry_budget:
                return candidates

        for seed in seed_pool:
            trial_inputs = [copy.deepcopy(seed), self._random_perturb(seed)]
            if target_branch is not None:
                trial_inputs.append(self._guided_perturb(seed, target_branch))
            for val in boundary_values[:10]:
                trial_inputs.append(self._replace_first_numeric(seed, val))
                trial_inputs.append(self._replace_first_numeric(seed, val - 1))
                trial_inputs.append(self._replace_first_numeric(seed, val + 1))
            trial_inputs.extend(self._mutate_list_like(seed))

            for cand in trial_inputs:
                if isinstance(cand, list) and cand and all(isinstance(x, str) for x in cand):
                    inp_lines = [str(x) for x in cand]
                else:
                    inp_lines = [str(x) for x in _build_input_lines(cand, self._input_structure)]
                key = '\n'.join(inp_lines)
                if key in local_seen:
                    continue
                if not self._is_legal_input(inp_lines):
                    continue
                local_seen.add(key)
                candidates.append(cand)
                if len(candidates) >= self._supplement_retry_budget:
                    return candidates
        return candidates

    def _supplement_unique_valid_cases(self, test_cases: List[Dict], population: List[Any], results: List[Dict]) -> List[Dict]:
        """若唯一合法有效样例不足 5 个，则继续定向补生成。"""
        enriched = list(test_cases)
        unique_valid = self._collect_unique_valid_cases(enriched)
        if len(unique_valid) >= self._hard_min_valid_cases:
            return enriched

        existing_keys = {'\n'.join(str(x) for x in tc.get('input', [])) for tc in unique_valid}
        seed_pool = self._candidate_seed_pool(population, results)
        candidates = self._generate_legal_candidate_inputs(seed_pool, existing_keys)

        for cand in candidates:
            try:
                res = _run_one(self._da, cand, self._input_structure)
            except Exception:
                continue
            tc = self._make_test_case(cand, res)
            if not tc:
                continue
            inp_lines = [str(x) for x in tc.get('input', [])]
            key = '\n'.join(inp_lines)
            if key in existing_keys:
                continue
            if not self._is_legal_input(inp_lines):
                continue
            enriched.append(tc)
            existing_keys.add(key)
            if len(self._collect_unique_valid_cases(enriched)) >= self._hard_min_valid_cases:
                break

        return enriched

    def _actual_output_from_result(self, res: Dict) -> str:
        """从执行结果中提取用于展示和比较的实际输出。"""
        execution_info = res.get('execution_info', {})
        output = str(execution_info.get('output', '') or '').strip()
        if output:
            return output
        if execution_info.get('success', False):
            return ''
        error = str(execution_info.get('error', '') or '').strip()
        if not error:
            return ''
        error_line = error.splitlines()[-1].strip()
        return f"ERROR: {error_line}" if error_line else 'ERROR'

    def _is_output_match(self, actual: str, expected: str) -> bool:
        """输出比较：先比较字符串，再尝试浮点容差比较"""
        a = (actual or '').strip()
        e = (expected or '').strip()
        if a == e:
            return True
        try:
            return abs(float(a) - float(e)) < self._output_tolerance
        except Exception:
            return False

    def _output_difference_score(self, actual: str, expected: str) -> float:
        """衡量输出与参考实现的差异，差异越大分数越高。"""
        a = (actual or '').strip()
        e = (expected or '').strip()
        if not e:
            return 0.0
        if self._is_output_match(a, e):
            return 0.0
        try:
            return abs(float(a) - float(e))
        except Exception:
            max_len = max(len(a), len(e), 1)
            mismatch = sum(1 for x, y in zip(a, e) if x != y) + abs(len(a) - len(e))
            return mismatch / max_len

    def _collect_boundary_values(self) -> List[float]:
        """收集静态分析得到的数值边界，用于输入变异和测试集保留"""
        vals: List[float] = []
        for pred in self._predicates:
            for v in pred.get('boundary_values', []):
                if isinstance(v, (int, float)):
                    vals.append(float(v))
        for branch in self._branches:
            for key in ('true_constraint', 'false_constraint'):
                constraint = branch.get(key, {})
                value = constraint.get('value')
                if isinstance(value, (int, float)):
                    vals.append(float(value))
                elif isinstance(value, list):
                    for x in value:
                        if isinstance(x, (int, float)):
                            vals.append(float(x))
        seen = set()
        ordered = []
        for v in vals:
            tag = round(float(v), 8)
            if tag not in seen:
                seen.add(tag)
                ordered.append(v)
        return ordered

    def _safe_parse_value(self, text: str) -> Any:
        """尽量把输入行解析为 Python 值"""
        try:
            return ast.literal_eval(text)
        except Exception:
            pass
        try:
            if '.' in text or 'e' in text.lower():
                return float(text)
            return int(text)
        except Exception:
            return text

    def _flatten_numeric_atoms(self, value: Any) -> List[float]:
        """递归提取输入中的所有数值原子。"""
        out: List[float] = []
        if isinstance(value, bool) or value is None:
            return out
        if isinstance(value, (int, float)):
            out.append(float(value))
            return out
        if isinstance(value, dict):
            for v in value.values():
                out.extend(self._flatten_numeric_atoms(v))
            return out
        if isinstance(value, (list, tuple, set)):
            for item in value:
                out.extend(self._flatten_numeric_atoms(item))
        return out

    def _coerce_like(self, original: Any, new_val: Any) -> Any:
        if isinstance(original, bool):
            return bool(new_val)
        if isinstance(original, int) and not isinstance(original, bool):
            return int(round(float(new_val)))
        if isinstance(original, float):
            return float(new_val)
        return new_val

    def _replace_first_numeric(self, individual: Any, new_val: Any) -> Any:
        """将个体中的第一个数值位置替换为 new_val。"""
        if isinstance(individual, tuple):
            items = list(individual)
            for i, item in enumerate(items):
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    items[i] = self._coerce_like(item, new_val)
                    return tuple(items)
                if isinstance(item, list) and item and isinstance(item[0], (int, float)):
                    copied = list(item)
                    copied[0] = self._coerce_like(copied[0], new_val)
                    items[i] = copied
                    return tuple(items)
            return tuple(items)
        if isinstance(individual, list):
            if individual and all(isinstance(x, str) for x in individual):
                lines = list(individual)
                for i, line in enumerate(lines):
                    parsed = self._safe_parse_value(line)
                    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
                        lines[i] = str(self._coerce_like(parsed, new_val))
                        return lines
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], (int, float)):
                        parsed = list(parsed)
                        parsed[0] = self._coerce_like(parsed[0], new_val)
                        lines[i] = repr(parsed)
                        return lines
                return lines
            copied = list(individual)
            if copied and isinstance(copied[0], (int, float)) and not isinstance(copied[0], bool):
                copied[0] = self._coerce_like(copied[0], new_val)
            return copied
        if isinstance(individual, (int, float)) and not isinstance(individual, bool):
            return self._coerce_like(individual, new_val)
        return copy.deepcopy(individual)

    def _mutate_list_like(self, individual: Any) -> List[Any]:
        """对列表形输入做结构性扰动。"""
        variants: List[Any] = []

        def mutate_list(lst: List[Any]) -> List[List[Any]]:
            out: List[List[Any]] = []
            if not lst:
                return [[0], [1], [-1]]
            out.append(lst + [lst[-1]])
            out.append(lst[:-1] if len(lst) > 1 else [])
            out.append(list(reversed(lst)))
            if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in lst):
                out.append(lst + [0])
                out.append(lst + [max(lst)])
                out.append(lst + [min(lst)])
            return out

        if isinstance(individual, list) and individual and all(isinstance(x, (int, float)) for x in individual):
            variants.extend(mutate_list(list(individual)))
        elif isinstance(individual, list) and individual and all(isinstance(x, str) for x in individual):
            lines = list(individual)
            for i, line in enumerate(lines):
                parsed = self._safe_parse_value(line)
                if isinstance(parsed, list):
                    for mutated in mutate_list(list(parsed)):
                        new_lines = list(lines)
                        new_lines[i] = repr(mutated)
                        variants.append(new_lines)
                    break
        elif isinstance(individual, tuple):
            for i, item in enumerate(individual):
                if isinstance(item, list):
                    for mutated in mutate_list(list(item)):
                        parts = list(individual)
                        parts[i] = mutated
                        variants.append(tuple(parts))
                    break
        return variants

    def _mutate_seed_individual(self, individual: Any, target_branch: Optional[Dict]) -> List[Any]:
        """围绕已有测试输入生成若干变异输入。"""
        candidates: List[Any] = [self._random_perturb(individual)]
        if target_branch is not None:
            candidates.append(self._guided_perturb(individual, target_branch))
            constraint = target_branch.get('true_constraint', {})
            var = constraint.get('var')
            for cand in self._constraint_candidates(constraint.get('op'), constraint.get('value'))[:4]:
                made = self._make_individual_with_var(var, cand)
                if made is not None:
                    candidates.append(made)
                candidates.append(self._replace_first_numeric(individual, cand))
        for boundary in self._collect_boundary_values()[:8]:
            candidates.append(self._replace_first_numeric(individual, boundary))
            candidates.append(self._replace_first_numeric(individual, boundary - 1))
            candidates.append(self._replace_first_numeric(individual, boundary + 1))
        candidates.extend(self._mutate_list_like(individual))

        deduped: List[Any] = []
        seen = set()
        for cand in candidates:
            key = repr(cand)
            if key not in seen:
                seen.add(key)
                deduped.append(cand)
        return deduped

    def _generate_testcase_mutants(self, population: List[Any], results: List[Dict]) -> List[Any]:
        """在正常生成流程中，对测试输入做变异。"""
        if not population:
            return []
        ranked = list(zip(population, results))
        ranked.sort(key=lambda pair: self._compute_fitness(pair[1]))
        seed_count = max(2, min(len(ranked), self._input_mutation_budget // 2))
        seeds = [ind for ind, _ in ranked[:seed_count]]
        target_branch = self._find_closest_uncovered_branch(results)

        mutants: List[Any] = []
        seen = {repr(x) for x in population}
        for seed in seeds:
            for cand in self._mutate_seed_individual(seed, target_branch):
                key = repr(cand)
                if key in seen:
                    continue
                seen.add(key)
                mutants.append(cand)
                if len(mutants) >= self._input_mutation_budget:
                    return mutants
        return mutants

    def _run_static_analysis(self):
        """运行静态分析，获取分支列表、输入变量、边界条件。"""
        cfg            = self._sa.build_control_flow_graph()
        predicates     = self._sa.extract_predicates_and_constraints()
        variable_types = self._sa.get_variable_types()
        data_deps      = self._sa.build_data_dependency_graph()
        input_struct   = self._sa.identify_input_structure()
        branch_map     = self._sa.get_branch_constraint_map()
        mutation_cands = self._sa.aggregate_mutation_candidates()

        self._static_result = {
            'cfg'                   : cfg,
            'predicates'            : predicates,
            'variable_types'        : variable_types,
            'data_dependencies'     : data_deps,
            'branch_constraint_map' : branch_map,
        }
        self._input_structure     = input_struct
        self._input_vars          = input_struct.get('inputs', [])
        self._predicates          = predicates
        self._mutation_candidates = mutation_cands
        self._branches            = list(branch_map.values())
        self._pred_map = {p['pred_id']: p for p in predicates if p.get('pred_id')}

        for b in self._branches:
            self._best_distance[b['branch_id']] = float('inf')

        # 构建 coverage_key -> branch_id 反查表
        # coverage_key 与 branch_distances 的 key 一致：
        #   有 pred_id 的分支 → pred_id（如 P1）
        #   for 循环分支      → str(lineno)（如 '5'）
        self._cov_key_to_branch_id: Dict[str, str] = {}
        for b in self._branches:
            pred_id = b.get('pred_id', '')
            lineno  = b.get('lineno', 0)
            cov_key = pred_id if pred_id else (str(lineno) if lineno else '')
            if cov_key:
                self._cov_key_to_branch_id[cov_key] = b['branch_id']

        self._da = DynamicAnalyzer(
            source_code=self.source_code,
            static_analysis_result=self._static_result,
            timeout=self.timeout,
        )

    # 步骤2：种群初始化

    def _init_population(self, xml_individuals: List[Any] = None) -> List[Any]:
        """生成初始种群"""
        seeds = []
        # XML 解析出的个体优先放入（直接是 input_lines 列表）
        if xml_individuals:
            seeds.extend(xml_individuals)
        seeds.extend(self._boundary_seeds())
        seeds.extend(self._template_seeds())
        while len(seeds) < self.pop_size:
            seeds.append(self._random_individual())
        seen, unique = set(), []
        for s in seeds:
            key = repr(s)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        while len(unique) < self.pop_size:
            unique.append(self._random_individual())
        return unique[:self.pop_size]

    def _boundary_seeds(self) -> List[Any]:
        """从谓词边界值生成种子。"""
        seeds = []
        for pred in self._predicates:
            for v in pred.get('boundary_values', []):
                if isinstance(v, (int, float)):
                    ind = self._make_individual_from_scalar(v)
                    if ind is not None:
                        seeds.append(ind)
        return seeds

    def _template_seeds(self) -> List[Any]:
        """从变异候选值生成模板种子。"""
        seeds = []
        for var_info in self._mutation_candidates.values():
            for branch_entry in var_info.get('branches', []):
                for v in branch_entry.get('candidates', []):
                    if isinstance(v, (int, float)):
                        ind = self._make_individual_from_scalar(v)
                        if ind is not None:
                            seeds.append(ind)
                    if len(seeds) >= self.pop_size:
                        return seeds
        return seeds

    def _make_individual_from_scalar(self, scalar: Any) -> Any:
        """将标量值包装成与 input_structure 匹配的个体。"""
        if not self._input_vars:
            return scalar
        if len(self._input_vars) == 1:
            return self._normalize_individual(self._coerce_value(scalar, self._input_vars[0]))
        parts = []
        for i, inp in enumerate(self._input_vars):
            parts.append(self._coerce_value(scalar, inp) if i == 0
                        else self._random_value_for_input(inp, i))
        return self._normalize_individual(tuple(parts))

    def _random_individual(self) -> Any:
        """生成完全随机的个体。"""
        if not self._input_vars:
            return random.randint(-100, 100)
        if len(self._input_vars) == 1:
            return self._normalize_individual(self._random_value_for_input(self._input_vars[0], 0))
        return self._normalize_individual(tuple(self._random_value_for_input(inp, idx) for idx, inp in enumerate(self._input_vars)))

    def _random_value_for_input(self, inp: Dict, idx: int = 0) -> Any:
        """根据输入变量的类型生成随机值。"""
        vtype = inp.get('type', 'Any')
        fmt   = inp.get('format', 'single_value')
        scalar_split_like = fmt in ('multi_value', 'split', 'iterator') and 'list' not in vtype
        bounds = self._problem_numeric_bounds(inp, idx)
        if vtype == 'int':
            if bounds is not None:
                low, high = bounds
                return random.randint(low, high)
            return random.randint(-100, 1000)
        if vtype == 'float':
            if bounds is not None:
                low, high = bounds
                return round(random.uniform(low, high), 3)
            return round(random.uniform(-100.0, 1000.0), 3)
        if vtype == 'str':
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=random.randint(0, 20)))
        if 'list' in vtype or fmt in ('list', 'split_string'):
            n = random.randint(0, 10)
            if 'float' in vtype:
                return [round(random.uniform(-100, 100), 2) for _ in range(n)]
            return [random.randint(-100, 100) for _ in range(n)]
        if scalar_split_like:
            if bounds is not None:
                low, high = bounds
                return random.randint(low, high)
            return random.randint(-100, 1000)
        if bounds is not None:
            low, high = bounds
            return random.randint(low, high)
        return random.randint(-100, 1000)

    def _coerce_value(self, v: Any, inp: Dict) -> Any:
        """将值强制转换为目标输入类型。"""
        vtype = inp.get('type', 'Any')
        fmt   = inp.get('format', 'single_value')
        scalar_split_like = fmt in ('multi_value', 'split', 'iterator') and 'list' not in vtype
        try:
            if vtype == 'int':
                return int(round(v)) if isinstance(v, (int, float)) else v
            if vtype == 'float':
                return float(v) if isinstance(v, (int, float)) else v
            if 'list' in vtype or fmt in ('list', 'split_string'):
                if isinstance(v, (list, tuple)):
                    return list(v)
                return [int(round(v))] if isinstance(v, (int, float)) else [v]
            if scalar_split_like and isinstance(v, (list, tuple)):
                return v[0] if v else 0
        except Exception:
            pass
        return v

    # 执行种群中的个体，收集覆盖信息

    def _evaluate_population(self, population: List[Any]) -> List[Dict]:
        """执行种群中所有个体，返回动态分析结果列表。"""
        results = []
        for ind in population:
            try:
                res = _run_one(self._da, ind, self._input_structure)
            except Exception:
                res = {'execution_info': {'success': False},
                    'coverage': {'covered_edges': [], 'covered_blocks': []},
                    'branch_distances': {}}
            actual_output = self._actual_output_from_result(res)
            input_lines = ind if isinstance(ind, list) and ind and isinstance(ind[0], str) else _build_input_lines(ind, self._input_structure)
            input_lines = [str(x) for x in input_lines]
            expected_output = self._compute_expected_output(input_lines)
            res['expected_output'] = expected_output
            res['output_difference_score'] = self._output_difference_score(actual_output, expected_output)
            res['killed_reference'] = bool(expected_output) and not self._is_output_match(actual_output, expected_output)
            results.append(res)
            # 更新全局已覆盖分支
            self._update_covered_branches(res)
        return results

    def _update_covered_branches(self, res: Dict):
        """从执行结果中提取已覆盖分支，更新全局集合。"""
        branch_dists = res.get('branch_distances', {})
        for key, dist_list in branch_dists.items():
            for entry in (dist_list if isinstance(dist_list, list) else []):
                if entry.get('distance', 1.0) == 0.0:
                    self._covered_branches.add(str(key))

    def _uncovered_branches(self) -> List[str]:
        """返回尚未覆盖的 branch_id 列表。"""
        all_bids = {b['branch_id'] for b in self._branches}
        return list(all_bids - self._covered_branches)

    # 步骤4b：计算适应度
    def _compute_fitness(self, res: Dict) -> float:
        """
        计算个体适应度（越小越好）。

        在分支距离之外，额外将“与参考实现输出差异”作为检错能力目标：
        - 若能打出与参考实现不同的输出，给予额外奖励；
        - 输出差异越大，适应度越低。
        """
        total_fitness = 0.0
        branch_dists = res.get('branch_distances', {})

        for branch in self._branches:
            bid = branch['branch_id']
            pid = branch.get('pred_id', '')

            dist_list = (branch_dists.get(pid) or
                        branch_dists.get(bid) or [])

            if not dist_list:
                total_fitness += 1.0
                continue

            min_dist = min(
                entry.get('distance', float('inf'))
                for entry in dist_list
                if isinstance(entry, dict)
            )

            if min_dist == 0.0:
                total_fitness += 0.0
            else:
                total_fitness += _normalize_distance(min_dist)

        if res.get('killed_reference'):
            total_fitness -= self._failure_bonus
        total_fitness -= self._failure_diff_weight * res.get('output_difference_score', 0.0)

        return total_fitness

    # 辅助：更新停滞计数器

    def _update_stagnation(self, results: List[Dict]):
        """根据当代结果更新每个分支的停滞计数。"""
        branch_dists_all: Dict[str, float] = {}
        for res in results:
            for key, dist_list in res.get('branch_distances', {}).items():
                if not isinstance(dist_list, list):
                    continue
                min_d = min(
                    (e.get('distance', float('inf')) for e in dist_list if isinstance(e, dict)),
                    default=float('inf')
                )
                cur = branch_dists_all.get(str(key), float('inf'))
                branch_dists_all[str(key)] = min(cur, min_d)

        for branch in self._branches:
            bid = branch['branch_id']
            pid = branch.get('pred_id', '')
            best_this_gen = min(
                branch_dists_all.get(pid, float('inf')),
                branch_dists_all.get(bid, float('inf'))
            )
            prev_best = self._best_distance.get(bid, float('inf'))
            if best_this_gen < prev_best:
                self._best_distance[bid] = best_this_gen
                self._stagnation_counter[bid] = 0
            else:
                self._stagnation_counter[bid] = self._stagnation_counter.get(bid, 0) + 1

    # 步骤4c：锦标赛选择父代

    def _tournament_select(self, population: List[Any],
                        fitness_scores: List[float]) -> List[Any]:
        """锦标赛选择，返回与种群等大的父代列表。"""
        parents = []
        n = len(population)
        for _ in range(n):
            candidates = random.sample(range(n), min(self.tournament_size, n))
            winner = min(candidates, key=lambda i: fitness_scores[i])
            parents.append(copy.deepcopy(population[winner]))
        return parents

    # 序列交叉

    def _crossover(self, p1: Any, p2: Any) -> Any:
        """
        序列交叉（针对列表/tuple 个体）。
        若个体是标量，则直接取两者之一或算术平均。
        """
        if random.random() > self.crossover_rate:
            return copy.deepcopy(p1)

        # 多输入 tuple
        if isinstance(p1, tuple) and isinstance(p2, tuple) and len(p1) == len(p2):
            child = tuple(self._crossover_scalar(a, b) for a, b in zip(p1, p2))
            return child

        # 列表
        if isinstance(p1, list) and isinstance(p2, list):
            return self._crossover_list(p1, p2)

        # 标量
        return self._crossover_scalar(p1, p2)

    def _crossover_scalar(self, a: Any, b: Any) -> Any:
        """标量交叉：随机选取一方或算术平均。"""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            r = random.random()
            if r < 0.4:
                return a
            elif r < 0.8:
                return b
            else:
                avg = (a + b) / 2.0
                return int(round(avg)) if isinstance(a, int) and isinstance(b, int) else avg
        return random.choice([a, b])

    def _crossover_list(self, a: list, b: list) -> list:
        """单点交叉（列表）。"""
        if not a:
            return copy.deepcopy(b) if b else []
        if not b:
            return copy.deepcopy(a)
        pt = random.randint(0, min(len(a), len(b)))
        return list(a[:pt]) + list(b[pt:])

    # 步骤4e：引导变异

    def _mutate(self, individual: Any, results: List[Dict]) -> Any:
        """
        引导变异：
            1. 找出当前测试中分支距离最小的未覆盖分支
            2. 获取该分支的条件表达式
            3. 根据条件类型和当前值，计算目标调整方向
            4. 对输入值应用调整
        """
        if random.random() > self.mutation_rate:
            return copy.deepcopy(individual)

        # 找未覆盖的、距离最小的分支
        target_branch = self._find_closest_uncovered_branch(results)

        if target_branch is None:
            # 无目标，纯随机扰动
            return self._random_perturb(individual)

        # 对个体施加引导变异
        return self._guided_perturb(individual, target_branch)

    def _find_closest_uncovered_branch(self, results: List[Dict]) -> Optional[Dict]:
        """找距离最小的未覆盖分支信息。"""
        uncovered = set(self._uncovered_branches())
        if not uncovered:
            return None

        best_bid, best_dist = None, float('inf')
        for res in results:
            for key, dist_list in res.get('branch_distances', {}).items():
                if not isinstance(dist_list, list):
                    continue
                # 找到对应的 branch_id
                bid = str(key)
                if bid not in uncovered:
                    continue
                min_d = min(
                    (e.get('distance', float('inf')) for e in dist_list
                        if isinstance(e, dict) and e.get('distance', 1) > 0),
                    default=float('inf')
                )
                if min_d < best_dist:
                    best_dist = min_d
                    best_bid = bid

        if best_bid is None:
            # fallback：随机选一个未覆盖分支
            best_bid = random.choice(list(uncovered))

        # 找该分支的约束信息
        for b in self._branches:
            if b['branch_id'] == best_bid:
                return b
        return None

    def _guided_perturb(self, individual: Any, branch: Dict) -> Any:
        """
        根据分支约束对个体施加引导变异。

        根据条件类型和当前值，计算调整方向：
            - 若需要 x 变大：x = x + step
            - 若需要 x 变小：x = x - step
            - 若需要 x 等于某值：x = target
        """
        ind = copy.deepcopy(individual)

        # 从 true_constraint 中提取目标变量和操作符
        true_c = branch.get('true_constraint', {})
        op    = true_c.get('op')
        var   = true_c.get('var')
        value = true_c.get('value')

        if value is None or not isinstance(value, (int, float)):
            # 无法引导，随机扰动
            return self._random_perturb(ind)

        # 确定当前变量值和要调整的分量索引
        current, idx = self._get_value_and_index(ind, var)
        if current is None or not isinstance(current, (int, float)):
            return self._random_perturb(ind)

        # 动态步长：与目标值距离越大，步长越大
        diff = abs(current - value)
        if diff > 100:
            step = max(1, int(diff * 0.3))
        elif diff > 10:
            step = max(1, int(diff * 0.5))
        elif diff > 1:
            step = 1
        else:
            step = 0.1

        # 根据操作符决定调整方向
        new_val = current
        if op in ('Lt', 'LtE'):
            # 需要 current < value 或 current <= value → 减小
            new_val = current - step
        elif op in ('Gt', 'GtE'):
            # 需要 current > value 或 current >= value → 增大
            new_val = current + step
        elif op == 'Eq':
            # 需要 current == value → 直接设为目标
            new_val = value
        elif op == 'NotEq':
            new_val = value + step
        else:
            # 未知操作符：随机靠近
            new_val = current + random.choice([-1, 1]) * step

        # 保持整数类型
        if isinstance(current, int):
            new_val = int(round(new_val))

        return self._normalize_individual(self._set_value_at_index(ind, idx, new_val))

    def _get_value_and_index(self, individual: Any, var: Optional[str]):
        """从个体中获取目标变量的当前值和位置索引。"""
        if isinstance(individual, tuple):
            # 多输入 tuple：按变量名匹配
            for i, inp in enumerate(self._input_vars):
                if inp.get('variable') == var and i < len(individual):
                    return individual[i], i
            # fallback：返回第一个数值分量
            for i, v in enumerate(individual):
                if isinstance(v, (int, float)):
                    return v, i
            return None, None
        if isinstance(individual, list):
            if individual and isinstance(individual[0], (int, float)):
                return individual[0], 0
            return None, None
        if isinstance(individual, (int, float)):
            return individual, 0
        return None, None

    def _set_value_at_index(self, individual: Any, idx: Any, new_val: Any) -> Any:
        """将个体指定位置的值替换为 new_val。"""
        if isinstance(individual, tuple):
            lst = list(individual)
            if idx is not None and 0 <= idx < len(lst):
                lst[idx] = new_val
            return tuple(lst)
        if isinstance(individual, list):
            lst = list(individual)
            if idx is not None and 0 <= idx < len(lst):
                lst[idx] = new_val
            return lst
        # 标量
        return new_val

    def _random_perturb(self, individual: Any) -> Any:
        """纯随机扰动：在当前值基础上加减随机步长。"""
        if isinstance(individual, tuple):
            perturbed = tuple(
                self._clamp_problem_value(self._perturb_scalar(v), self._input_vars[idx], idx)
                if idx < len(self._input_vars) else self._perturb_scalar(v)
                for idx, v in enumerate(individual)
            )
            return self._normalize_individual(perturbed)
        if isinstance(individual, list):
            if not individual:
                return [random.randint(-100, 100)]
            return [self._perturb_scalar(v) for v in individual]
        if self._input_vars:
            return self._clamp_problem_value(self._perturb_scalar(individual), self._input_vars[0], 0)
        return self._perturb_scalar(individual)

    # 交叉并变异
    
    def _crossover_and_mutate(self, parents: List[Any],
                            results: List[Dict]) -> List[Any]:
        """
        对父代两两交叉，再对每个子代施加引导变异。
        交叉后验证子代有效性：
            1. 若子代与父代完全相同（无效交叉），重新交叉至多 3 次
            2. 若子代与已见个体重复，直接随机扰动产生新个体
        """
        offspring = []
        seen: set = set()
        # 将父代全部加入 seen，避免子代与父代重复
        for p in parents:
            seen.add(repr(p))

        def _unique_crossover(a: Any, b: Any) -> Any:
            """交叉并验证：重复则重试，最多 3 次，仍重复则随机扰动。"""
            for _ in range(3):
                child = self._crossover(a, b)
                if repr(child) not in seen:
                    return child
                # 交叉结果与已有个体相同，交换父代顺序再试
                a, b = b, a
            # 3 次均重复，随机扰动产生差异个体
            return self._random_perturb(copy.deepcopy(a))

        random.shuffle(parents)
        for i in range(0, len(parents) - 1, 2):
            child1 = _unique_crossover(parents[i], parents[i + 1])
            child2 = _unique_crossover(parents[i + 1], parents[i])
            c1 = self._mutate(child1, results)
            c2 = self._mutate(child2, results)
            seen.add(repr(c1))
            seen.add(repr(c2))
            offspring.append(c1)
            offspring.append(c2)
        if len(parents) % 2 == 1:
            child = _unique_crossover(parents[-1], random.choice(parents))
            c = self._mutate(child, results)
            offspring.append(c)
        return offspring

    # 合并种群

    def _merge_population(self, population: List[Any],
                            offspring: List[Any],
                            fitness_scores: List[float]) -> List[Any]:
        """
        合并当前种群与子代，保留适应度最低的个体），
        其余随机选取填充到种群大小。
        """
        # 精英：当前种群中适应度最低的 elite_size 个
        elite_idx = sorted(range(len(population)),
                        key=lambda i: fitness_scores[i])[:self.elite_size]
        elites = [copy.deepcopy(population[i]) for i in elite_idx]

        pool = offspring + [
            population[i] for i in range(len(population))
            if i not in set(elite_idx)
        ]
        random.shuffle(pool)

        new_pop = elites + pool[:self.pop_size - self.elite_size]
        # 若 pool 不够，补充随机个体
        while len(new_pop) < self.pop_size:
            new_pop.append(self._random_individual())
        return new_pop[:self.pop_size]

    # 步骤4g：约束求解兜底

    def _constraint_solve(self, stagnant_branch_ids: List[str]) -> List[Any]:
        """
        对长期未覆盖的分支，通过分析分支约束直接构造满足条件的输入。

        策略：
            - 取该分支 true_constraint 中的 var 和 value
            - 根据操作符，直接将输入变量设置为恰好满足条件的值
            - 同时在边界附近生成多个候选值
        """
        solved = []
        branch_by_id = {b['branch_id']: b for b in self._branches}

        for bid in stagnant_branch_ids:
            branch = branch_by_id.get(bid)
            if not branch:
                continue

            true_c = branch.get('true_constraint', {})
            op     = true_c.get('op')
            var    = true_c.get('var')
            value  = true_c.get('value')

            # 根据约束生成候选值
            candidates = self._constraint_candidates(op, value)

            for cand in candidates:
                ind = self._make_individual_with_var(var, cand)
                if ind is not None:
                    solved.append(ind)

        return solved

    def _constraint_candidates(self, op: Optional[str],
                            value: Any) -> List[Any]:
        """根据操作符和比较值生成满足约束的候选值列表。"""
        if value is None or not isinstance(value, (int, float)):
            return [0, 1, -1, random.randint(-50, 50)]

        v = value
        if op in ('Lt',):
            return [v - 1, v - 2, v - 10]
        if op in ('LtE',):
            return [v, v - 1, v - 5]
        if op in ('Gt',):
            return [v + 1, v + 2, v + 10]
        if op in ('GtE',):
            return [v, v + 1, v + 5]
        if op in ('Eq',):
            return [v]
        if op in ('NotEq',):
            return [v + 1, v - 1]
        # 链式比较 value 是 list
        if isinstance(value, list) and len(value) == 2:
            lo, hi = value[0], value[1]
            return [lo, hi, (lo + hi) / 2, lo - 1, hi + 1]
        return [v - 1, v, v + 1]

    def _make_individual_with_var(self, var: Optional[str],
                                val: Any) -> Optional[Any]:
        """构造一个让指定输入变量等于 val 的个体。"""
        if not self._input_vars:
            return val
        if len(self._input_vars) == 1:
            return self._coerce_value(val, self._input_vars[0])

        parts = []
        matched = False
        for inp in self._input_vars:
            if inp.get('variable') == var and not matched:
                parts.append(self._coerce_value(val, inp))
                matched = True
            else:
                parts.append(self._random_value_for_input(inp))
        if not matched and parts:
            # 找不到变量名匹配，设置第一个
            inp0 = self._input_vars[0]
            parts[0] = self._coerce_value(val, inp0)
        return tuple(parts)

    # 步骤5：贪心精简测试集

    def _derive_case_features(self, tc: Dict) -> Dict[str, set]:
        """提取测试用例的覆盖、边界、多样性特征。"""
        branches = {f"branch:{b}" for b in tc.get('covered_branches', [])}
        boundary = set()
        shape = set()
        output = set()

        parsed_values = [self._safe_parse_value(line) for line in tc.get('input', [])]
        numeric_atoms: List[float] = []
        for idx, value in enumerate(parsed_values):
            numeric_atoms.extend(self._flatten_numeric_atoms(value))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value < 0:
                    shape.add(f"input{idx}:neg")
                elif value > 0:
                    shape.add(f"input{idx}:pos")
                else:
                    shape.add(f"input{idx}:zero")
                shape.add(f"input{idx}:scalar")
            elif isinstance(value, list):
                n = len(value)
                bucket = 'empty' if n == 0 else 'single' if n == 1 else 'short' if n <= 3 else 'medium' if n <= 6 else 'long'
                shape.add(f"input{idx}:list:{bucket}")
                if len(value) != len(set(repr(x) for x in value)):
                    shape.add(f"input{idx}:list:dup")
                nums = [x for x in value if isinstance(x, (int, float)) and not isinstance(x, bool)]
                if nums:
                    if any(x < 0 for x in nums):
                        shape.add(f"input{idx}:list:neg")
                    if any(x == 0 for x in nums):
                        shape.add(f"input{idx}:list:zero")
                    if any(x > 0 for x in nums):
                        shape.add(f"input{idx}:list:pos")
                    if nums == sorted(nums):
                        shape.add(f"input{idx}:list:sorted")
                    if nums == sorted(nums, reverse=True):
                        shape.add(f"input{idx}:list:revsorted")

        boundaries = self._collect_boundary_values()
        for x in numeric_atoms:
            if x < 0:
                boundary.add('num:neg')
            elif x > 0:
                boundary.add('num:pos')
            else:
                boundary.add('num:zero')
            if boundaries:
                nearest = min(boundaries, key=lambda b: abs(x - b))
                rounded = str(int(nearest)) if float(nearest).is_integer() else f"{nearest:.3f}"
                dist = abs(x - nearest)
                if dist < 1e-4:
                    boundary.add(f"boundary:eq:{rounded}")
                elif dist <= 1:
                    side = 'below' if x < nearest else 'above'
                    boundary.add(f"boundary:near:{side}:{rounded}")
                elif dist <= 5:
                    side = 'below' if x < nearest else 'above'
                    boundary.add(f"boundary:adjacent:{side}:{rounded}")

        expected = self._safe_parse_value((tc.get('expected_output') or tc.get('output') or '').strip())
        if isinstance(expected, bool):
            output.add(f"output:bool:{expected}")
        elif isinstance(expected, (int, float)):
            if expected < 0:
                output.add('output:num:neg')
            elif expected > 0:
                output.add('output:num:pos')
            else:
                output.add('output:num:zero')
        elif isinstance(expected, list):
            bucket = 'empty' if len(expected) == 0 else 'single' if len(expected) == 1 else 'multi'
            output.add(f"output:list:{bucket}")
        elif str(expected) != '':
            output.add(f"output:text:{str(expected)[:16]}")

        return {
            'branches': branches,
            'boundary': boundary,
            'shape': shape,
            'output': output,
        }

    def _infer_case_style(self, test_cases: List[Dict]) -> str:
        """根据输入形态分类"""
        list_hits = 0
        numeric_hits = 0
        for tc in test_cases:
            for line in tc.get('input', []):
                value = self._safe_parse_value(line)
                if isinstance(value, list):
                    list_hits += 1
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_hits += 1
        if list_hits and not numeric_hits:
            return 'list'
        if numeric_hits and not list_hits:
            return 'numeric'
        if list_hits > numeric_hits:
            return 'list'
        return 'numeric' if numeric_hits else 'mixed'

    def _select_essential_targets(self, target: Dict[str, set], style: str) -> Dict[str, set]:
        """按题型选择需要保留的关键特征数量。"""
        if style == 'list':
            boundary_keep = 4
            shape_keep = 8
            output_keep = 5
        elif style == 'numeric':
            boundary_keep = 8
            shape_keep = 4
            output_keep = 4
        else:
            boundary_keep = 6
            shape_keep = 6
            output_keep = 4
        return {
            'branches': set(target['branches']),
            'boundary': set(list(sorted(target['boundary']))[:boundary_keep]),
            'shape': set(list(sorted(target['shape']))[:shape_keep]),
            'output': set(list(sorted(target['output']))[:output_keep]),
        }

    def _pick_style_anchors(self, unique_cases: List[Dict], style: str) -> List[int]:
        """按题型补充少量锚点样本，避免精简过头。"""
        anchors: List[int] = []
        parsed_per_case = [ [self._safe_parse_value(line) for line in tc.get('input', [])] for tc in unique_cases ]

        if style == 'numeric':
            numeric_scores = []
            for idx, parsed in enumerate(parsed_per_case):
                values = []
                for value in parsed:
                    values.extend(self._flatten_numeric_atoms(value))
                if values:
                    numeric_scores.append((min(values), max(values), idx))
            if numeric_scores:
                anchors.append(min(numeric_scores, key=lambda x: x[0])[2])
                anchors.append(max(numeric_scores, key=lambda x: x[1])[2])
        elif style == 'list':
            list_scores = []
            for idx, parsed in enumerate(parsed_per_case):
                for value in parsed:
                    if isinstance(value, list):
                        nums = [x for x in value if isinstance(x, (int, float)) and not isinstance(x, bool)]
                        has_dup = len(value) != len(set(repr(x) for x in value))
                        list_scores.append((len(value), has_dup, any(x == 0 for x in nums), idx))
                        break
            if list_scores:
                anchors.append(min(list_scores, key=lambda x: x[0])[3])
                anchors.append(max(list_scores, key=lambda x: x[0])[3])
                dup_cases = [x[3] for x in list_scores if x[1]]
                if dup_cases:
                    anchors.append(dup_cases[0])
                zero_cases = [x[3] for x in list_scores if x[2]]
                if zero_cases:
                    anchors.append(zero_cases[0])

        deduped = []
        seen = set()
        for idx in anchors:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
        return deduped

    def _classify_input_pattern(self, value: Any) -> str:
        """对输入做粗粒度分类，用于合并同类检错样例。"""
        if isinstance(value, bool):
            return f"bool:{value}"
        if isinstance(value, (int, float)):
            if value < 0:
                return 'num:neg'
            if value > 0:
                return 'num:pos'
            return 'num:zero'
        if isinstance(value, list):
            if not value:
                return 'list:empty'
            nums = [x for x in value if isinstance(x, (int, float)) and not isinstance(x, bool)]
            signs = []
            if any(x < 0 for x in nums):
                signs.append('neg')
            if any(x == 0 for x in nums):
                signs.append('zero')
            if any(x > 0 for x in nums):
                signs.append('pos')
            sign_tag = ','.join(signs) if signs else 'nonnum'
            return f"list:{len(value)}:{sign_tag}"
        return f"type:{type(value).__name__}"

    def _killer_signature(self, tc: Dict) -> tuple:
        """提取检错样例的粗粒度原因签名，避免同因重复保留。"""
        parsed = [self._safe_parse_value(line) for line in tc.get('input', [])]
        input_pattern = tuple(self._classify_input_pattern(value) for value in parsed)
        covered = tuple(sorted(tc.get('covered_branches', [])))
        actual = str(tc.get('output', '')).strip()
        expected = str(tc.get('expected_output', '')).strip()
        try:
            delta_sign = 'pos' if float(actual) - float(expected) > 0 else 'neg'
        except Exception:
            delta_sign = 'text'
        return covered, input_pattern, delta_sign

    def _select_representative_killers(self, unique_cases: List[Dict]) -> List[int]:
        """同类检错原因只保留一个代表样例。"""
        grouped: Dict[tuple, int] = {}
        for idx, tc in enumerate(unique_cases):
            if not tc.get('kills_reference'):
                continue
            signature = self._killer_signature(tc)
            current = grouped.get(signature)
            if current is None or tc.get('output_difference_score', 0.0) > unique_cases[current].get('output_difference_score', 0.0):
                grouped[signature] = idx
        return sorted(grouped.values())

    def _greedy_minimize(self, test_cases: List[Dict]) -> List[Dict]:
        """
        按题型做精简：
            - numeric 题更重视边界值与正负零/极值
            - list 题更重视长度、重复、顺序、零值等结构形态
        """
        if not test_cases:
            return []

        seen_inputs: set = set()
        unique_cases: List[Dict] = []
        for tc in self._collect_unique_valid_cases(test_cases):
            key = tc.get('input_display', repr(tc.get('input', '')))
            if key not in seen_inputs:
                seen_inputs.add(key)
                unique_cases.append(tc)

        if not unique_cases:
            return []

        killer_cases = self._select_representative_killers(unique_cases)

        style = self._infer_case_style(unique_cases)
        features_by_idx = [self._derive_case_features(tc) for tc in unique_cases]
        target = {
            'branches': set().union(*(f['branches'] for f in features_by_idx)) if features_by_idx else set(),
            'boundary': set().union(*(f['boundary'] for f in features_by_idx)) if features_by_idx else set(),
            'shape': set().union(*(f['shape'] for f in features_by_idx)) if features_by_idx else set(),
            'output': set().union(*(f['output'] for f in features_by_idx)) if features_by_idx else set(),
        }
        essential_targets = self._select_essential_targets(target, style)

        if style == 'list':
            weights = {'branches': 100, 'boundary': 10, 'shape': 20, 'output': 10}
        elif style == 'numeric':
            weights = {'branches': 100, 'boundary': 20, 'shape': 8, 'output': 8}
        else:
            weights = {'branches': 100, 'boundary': 15, 'shape': 12, 'output': 8}

        covered = {k: set() for k in essential_targets}
        selected_idx: List[int] = []
        remaining = set(range(len(unique_cases)))

        for idx in killer_cases:
            if idx in remaining:
                selected_idx.append(idx)
                remaining.remove(idx)
                covered['branches'] |= features_by_idx[idx]['branches']
                covered['boundary'] |= features_by_idx[idx]['boundary'] & essential_targets['boundary']
                covered['shape'] |= features_by_idx[idx]['shape'] & essential_targets['shape']
                covered['output'] |= features_by_idx[idx]['output'] & essential_targets['output']

        while remaining:
            best_idx = None
            best_score = -1
            for idx in remaining:
                feats = features_by_idx[idx]
                score = 0
                score += len(feats['branches'] - covered['branches']) * weights['branches']
                score += len(feats['boundary'] & (essential_targets['boundary'] - covered['boundary'])) * weights['boundary']
                score += len(feats['shape'] & (essential_targets['shape'] - covered['shape'])) * weights['shape']
                score += len(feats['output'] & (essential_targets['output'] - covered['output'])) * weights['output']
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None or best_score <= 0:
                break
            selected_idx.append(best_idx)
            remaining.remove(best_idx)
            covered['branches'] |= features_by_idx[best_idx]['branches']
            covered['boundary'] |= features_by_idx[best_idx]['boundary'] & essential_targets['boundary']
            covered['shape'] |= features_by_idx[best_idx]['shape'] & essential_targets['shape']
            covered['output'] |= features_by_idx[best_idx]['output'] & essential_targets['output']
            if all(covered[key] >= essential_targets[key] for key in essential_targets):
                break

        if not selected_idx:
            selected_idx = [0]

        for idx in self._pick_style_anchors(unique_cases, style):
            if idx not in selected_idx:
                selected_idx.append(idx)

        minimized = list(selected_idx)
        protected_idx = set(killer_cases)
        i = 0
        while i < len(minimized):
            if minimized[i] in protected_idx:
                i += 1
                continue
            trial = minimized[:i] + minimized[i + 1:]
            trial_cov = {k: set() for k in essential_targets}
            for idx in trial:
                trial_cov['branches'] |= features_by_idx[idx]['branches']
                trial_cov['boundary'] |= features_by_idx[idx]['boundary'] & essential_targets['boundary']
                trial_cov['shape'] |= features_by_idx[idx]['shape'] & essential_targets['shape']
                trial_cov['output'] |= features_by_idx[idx]['output'] & essential_targets['output']
            if all(trial_cov[key] >= essential_targets[key] for key in essential_targets):
                minimized = trial
            else:
                i += 1

        minimized = sorted(set(minimized))

        min_keep = min(5, len(unique_cases))
        if len(minimized) < min_keep:
            current_cov = {k: set() for k in essential_targets}
            for idx in minimized:
                current_cov['branches'] |= features_by_idx[idx]['branches']
                current_cov['boundary'] |= features_by_idx[idx]['boundary'] & essential_targets['boundary']
                current_cov['shape'] |= features_by_idx[idx]['shape'] & essential_targets['shape']
                current_cov['output'] |= features_by_idx[idx]['output'] & essential_targets['output']

            remaining_candidates = [idx for idx in range(len(unique_cases)) if idx not in set(minimized)]
            while len(minimized) < min_keep and remaining_candidates:
                best_idx = None
                best_gain = -1
                for idx in remaining_candidates:
                    gain = 0
                    gain += len(features_by_idx[idx]['branches'] - current_cov['branches']) * weights['branches']
                    gain += len((features_by_idx[idx]['boundary'] & essential_targets['boundary']) - current_cov['boundary']) * weights['boundary']
                    gain += len((features_by_idx[idx]['shape'] & essential_targets['shape']) - current_cov['shape']) * weights['shape']
                    gain += len((features_by_idx[idx]['output'] & essential_targets['output']) - current_cov['output']) * weights['output']
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx

                if best_idx is None:
                    best_idx = remaining_candidates[0]
                minimized.append(best_idx)
                current_cov['branches'] |= features_by_idx[best_idx]['branches']
                current_cov['boundary'] |= features_by_idx[best_idx]['boundary'] & essential_targets['boundary']
                current_cov['shape'] |= features_by_idx[best_idx]['shape'] & essential_targets['shape']
                current_cov['output'] |= features_by_idx[best_idx]['output'] & essential_targets['output']
                remaining_candidates = [idx for idx in remaining_candidates if idx != best_idx]

        minimized = sorted(set(minimized))
        return [unique_cases[idx] for idx in minimized]

    # 辅助：构建测试用例记录

    def _make_test_case(self, individual: Any, res: Dict) -> Optional[Dict]:
        """将个体和执行结果打包为测试用例字典。"""
        if not res.get('execution_info', {}).get('success', False):
            return None

        if isinstance(individual, list) and all(isinstance(x, str) for x in individual):
            inp_lines = list(individual)
        else:
            inp_lines = _build_input_lines(individual, self._input_structure)
        inp_lines = [str(x) for x in inp_lines]
        display   = '\n'.join(inp_lines)

        # 提取覆盖的分支 id，统一映射为 branch_id（B1/B2 等）
        covered: List[str] = []
        for key, dist_list in res.get('branch_distances', {}).items():
            if not isinstance(dist_list, list):
                continue
            for entry in dist_list:
                if isinstance(entry, dict) and entry.get('distance', 1) == 0.0:
                    # 将 coverage_key（pred_id 或行号）映射为统一的 branch_id
                    branch_id = self._cov_key_to_branch_id.get(str(key), str(key))
                    covered.append(branch_id)
                    break

        expected_output = res.get('expected_output')
        if expected_output is None:
            expected_output = self._compute_expected_output(inp_lines)

        return {
            'input'           : inp_lines,
            'input_display'   : display,
            'covered_branches': list(set(covered)),
            'output'          : self._actual_output_from_result(res),
            'expected_output' : expected_output,
            'kills_reference' : res.get('killed_reference', False),
            'output_difference_score': res.get('output_difference_score', 0.0),
        }

    @staticmethod
    def _perturb_scalar(v: Any) -> Any:
        """对单个标量值施加随机扰动。"""
        if isinstance(v, float):
            return v + random.gauss(0, max(1.0, abs(v) * 0.1))
        if isinstance(v, int):
            step = max(1, abs(v) // 10 + 1)
            return v + random.randint(-step, step)
        return v
