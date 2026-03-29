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
5. 精简测试集（贪心选择最小覆盖集）
6. 返回测试用例集
"""

import random
import copy
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Any, Optional

from static_analyzer import StaticAnalyzer
from dynamic_analyzer import DynamicAnalyzer


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
        return [_value_to_input_line(v, inp.get('format', 'evaluated'))
                for v, inp in zip(value, inputs)]
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


# ---------------------------------------------------------------------------
# TestcaseGenerator
# ---------------------------------------------------------------------------

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

        self._sa = StaticAnalyzer(source_code=source_code)
        self._static_result: Dict = {}
        self._branches: List[Dict] = []
        self._input_structure: Dict = {}
        self._input_vars: List[Dict] = []
        self._predicates: List[Dict] = []
        self._mutation_candidates: Dict = {}
        self._pred_map: Dict[str, Dict] = {}
        self._da: Optional[DynamicAnalyzer] = None
        self._covered_branches: set = set()
        self._stagnation_counter: Dict[str, int] = defaultdict(int)
        self._best_distance: Dict[str, float] = {}

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
                    res = _run_one(self._da, inp_lines, self._input_structure)
                    self._update_covered_branches(res)
                    tc = self._make_test_case(inp_lines, res)
                    if tc:
                        all_test_cases.append(tc)
                    else:
                        # 执行成功但格式化失败时保留原始 XML 信息
                        all_test_cases.append({
                            'input'           : inp_lines,
                            'input_display'   : xtc.get('input_display', '\n'.join(inp_lines)),
                            'covered_branches': [],
                            'output'          : xtc.get('expected_output', ''),
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

        # 步骤5：贪心精简测试集
        return self._greedy_minimize(all_test_cases)

    # 步骤1：静态分析

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
            return self._coerce_value(scalar, self._input_vars[0])
        parts = []
        for i, inp in enumerate(self._input_vars):
            parts.append(self._coerce_value(scalar, inp) if i == 0
                        else self._random_value_for_input(inp))
        return tuple(parts)

    def _random_individual(self) -> Any:
        """生成完全随机的个体。"""
        if not self._input_vars:
            return random.randint(-100, 100)
        if len(self._input_vars) == 1:
            return self._random_value_for_input(self._input_vars[0])
        return tuple(self._random_value_for_input(inp) for inp in self._input_vars)

    def _random_value_for_input(self, inp: Dict) -> Any:
        """根据输入变量的类型生成随机值。"""
        vtype = inp.get('type', 'Any')
        fmt   = inp.get('format', 'single_value')
        if vtype == 'int':
            return random.randint(-100, 1000)
        if vtype == 'float':
            return round(random.uniform(-100.0, 1000.0), 3)
        if vtype == 'str':
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=random.randint(0, 20)))
        if 'list' in vtype or fmt in ('list', 'split', 'split_string', 'iterator'):
            n = random.randint(0, 10)
            if 'float' in vtype:
                return [round(random.uniform(-100, 100), 2) for _ in range(n)]
            return [random.randint(-100, 100) for _ in range(n)]
        return random.randint(-100, 1000)

    def _coerce_value(self, v: Any, inp: Dict) -> Any:
        """将值强制转换为目标输入类型。"""
        vtype = inp.get('type', 'Any')
        fmt   = inp.get('format', 'single_value')
        try:
            if vtype == 'int':
                return int(round(v)) if isinstance(v, (int, float)) else v
            if vtype == 'float':
                return float(v) if isinstance(v, (int, float)) else v
            if 'list' in vtype or fmt in ('list', 'split', 'split_string', 'iterator'):
                if isinstance(v, (list, tuple)):
                    return list(v)
                return [int(round(v))] if isinstance(v, (int, float)) else [v]
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

        算法：
        1. 初始化总适应度 = 0
        2. 遍历所有分支：
            a. 若分支已覆盖（距离 == 0），适应度 += 0
            b. 若分支未覆盖，取该分支在所有执行步骤中的最小距离
            c. 适应度 += normalize(最小距离)
        3. 返回总适应度
        """
        total_fitness = 0.0
        branch_dists = res.get('branch_distances', {})

        for branch in self._branches:
            bid = branch['branch_id']
            pid = branch.get('pred_id', '')

            # 查找该分支对应的距离记录（用 pred_id 或 branch_id 作 key）
            dist_list = (branch_dists.get(pid) or
                        branch_dists.get(bid) or [])

            if not dist_list:
                # 没有执行记录，使用最大距离
                total_fitness += 1.0
                continue

            # 取所有步骤中的最小距离
            min_dist = min(
                entry.get('distance', float('inf'))
                for entry in dist_list
                if isinstance(entry, dict)
            )

            if min_dist == 0.0:
                # 分支已覆盖
                total_fitness += 0.0
            else:
                # 标准化距离
                total_fitness += _normalize_distance(min_dist)

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

        return self._set_value_at_index(ind, idx, new_val)

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
            return tuple(self._perturb_scalar(v) for v in individual)
        if isinstance(individual, list):
            if not individual:
                return [random.randint(-100, 100)]
            return [self._perturb_scalar(v) for v in individual]
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

    def _greedy_minimize(self, test_cases: List[Dict]) -> List[Dict]:
        """
        两阶段最小覆盖集选择：
            阶段一（贪心扩展）：每轮从剩余用例中选覆盖「新分支」最多的用例，
                            直到无新分支可覆盖，得到最大覆盖率下的初始集合。
            阶段二（冗余裁剪）：逐一尝试去除已选用例，若去除后其他用例仍能
                            维持相同覆盖率，则将其去除，保证数量最少。
        目标：在覆盖率尽可能高的前提下，最小化测试用例数量。
        """
        if not test_cases:
            return []

        # 去重（按 input_display）
        seen_inputs: set = set()
        unique_cases: List[Dict] = []
        for tc in test_cases:
            key = tc.get('input_display', repr(tc.get('input', '')))
            if key not in seen_inputs:
                seen_inputs.add(key)
                unique_cases.append(tc)

        # 若所有用例均无覆盖信息，直接返回去重后全集
        if all(not tc.get('covered_branches') for tc in unique_cases):
            return unique_cases

        all_branches = {b['branch_id'] for b in self._branches}

        #最大化覆盖率
        covered_so_far: set = set()
        selected: List[Dict] = []
        remaining = list(unique_cases)

        while remaining:
            best_tc = None
            best_new: set = set()
            for tc in remaining:
                new = set(tc.get('covered_branches', [])) - covered_so_far
                # 主键：新覆盖分支数；次键：总覆盖分支数（tie-break）
                if len(new) > len(best_new) or (
                    len(new) == len(best_new) > 0
                    and len(tc.get('covered_branches', [])) > len(best_tc.get('covered_branches', []))
                ):
                    best_new = new
                    best_tc = tc

            if not best_tc or not best_new:
                break  # 无新分支可覆盖，已达到最大覆盖率

            selected.append(best_tc)
            covered_so_far |= best_new
            remaining.remove(best_tc)

            if all_branches and covered_so_far >= all_branches:
                break  # 已覆盖全部已知分支

        # 若贪心结果为空（所有用例 covered_branches 均为空列表），退回全集
        if not selected:
            return unique_cases

        # 在保持最大覆盖率的前提下最小化数量
        # 从贡献分支数最少的用例开始尝试去除（最不重要的先试）
        max_coverage = set(covered_so_far)  # 阶段一达到的最大覆盖集

        # 按覆盖分支数升序排列（贡献少的优先尝试去除）
        selected.sort(key=lambda tc: len(tc.get('covered_branches', [])))

        minimized: List[Dict] = list(selected)
        i = 0
        while i < len(minimized):
            candidate = minimized[i]
            rest = minimized[:i] + minimized[i + 1:]
            rest_covered = set()
            for tc in rest:
                rest_covered |= set(tc.get('covered_branches', []))
            if rest_covered >= max_coverage:
                # 去除该用例后覆盖率不变，执行去除
                minimized = rest
                # i 不递增，继续检查同位置的下一个元素
            else:
                i += 1

        return minimized

    # 辅助：构建测试用例记录

    def _make_test_case(self, individual: Any, res: Dict) -> Optional[Dict]:
        """将个体和执行结果打包为测试用例字典。"""
        if not res.get('execution_info', {}).get('success', False):
            return None

        inp_lines = _build_input_lines(individual, self._input_structure)
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

        return {
            'input'           : inp_lines,
            'input_display'   : display,
            'covered_branches': list(set(covered)),
            'output'          : res.get('execution_info', {}).get('output', ''),
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
