# TestcaseGenerator 文档

## 概述

`testcase_generator.py` 实现了一个基于**遗传算法（GA）+ 分支距离**的测试用例自动生成器。核心思路是：将每个测试输入视为一个「个体」，以分支距离为适应度函数，通过选择、交叉、变异迭代进化，使测试用例集的分支覆盖率最大化，最后用贪心算法精简到最小覆盖集。

---

## 整体流程

```
静态分析
  └─ 获取分支列表、输入变量结构、谓词边界值、变异候选值
      ↓
XML 种子解析（可选）
  └─ 从 XML 中读取已有测试用例作为初始种子
      ↓
初始种群生成
  └─ 边界值种子 + 模板种子 + 随机补充
      ↓
GA 主循环（最多 max_generations 代）
  ├─ 执行所有个体 → 收集分支距离
  ├─ 计算适应度（分支距离归一化累加）
  ├─ 锦标赛选择父代
  ├─ 交叉 + 引导变异 → 子代
  ├─ 合并种群（保留精英）
  └─ 停滞处理：调用约束求解直接构造满足条件的输入
      ↓
最终评估 → 收集所有生成用例
      ↓
贪心精简（最大覆盖 → 去冗余）
      ↓
返回最小覆盖测试用例集
```

---

## 模块级工具函数

### `_parse_xml(xml_file)`

**作用**：从 XML 文件中解析测试用例。

| 参数 | 类型 | 说明 |
|------|------|------|
| `xml_file` | `str` | XML 文件路径 |

**返回**：`List[Dict]`，每条记录包含 `id`、`input`（按行分割的列表）、`input_display`、`expected_output`。

**实现逻辑**：用 `xml.etree.ElementTree` 解析 XML，遍历所有 `testData*` 节点，提取 `<input>` 和 `<output>` 子元素的文本内容，将输入按换行符分割成列表。

---

### `_value_to_input_line(value, fmt)`

**作用**：将 Python 值按照指定输入格式转换为字符串（单行）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `value` | `Any` | 待转换的值 |
| `fmt` | `str` | 输入格式，取值见下表 |

| `fmt` 取值 | 转换方式 |
|-----------|----------|
| `'evaluated'` | `repr(value)`，适用于 `eval(input())` 读取的输入 |
| `'single_value'` | `str(value)` |
| `'list'` / `'multi_value'` / `'split'` / `'iterator'` / `'split_string'` | 列表/元组元素用空格拼接 |
| 其他 | `repr(value)` |

---

### `_build_input_lines(value, input_structure)`

**作用**：将候选值转换为 `run_with_input` 所需的字符串列表（每行对应一次 `input()` 调用）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `value` | `Any` | 个体值（单值、列表或 tuple） |
| `input_structure` | `Dict` | 静态分析识别的输入结构，包含 `inputs` 列表，每项有 `format`、`type` 等字段 |

**实现逻辑**：
- 若 `inputs` 为空，直接 `repr(value)`
- 若只有 1 个输入变量，调用 `_value_to_input_line`
- 若有多个输入变量且 `value` 是等长 tuple/list，逐一映射各分量

---

### `_run_one(da, inp, input_structure)`

**作用**：执行单个测试输入，返回动态分析结果。

| 参数 | 类型 | 说明 |
|------|------|------|
| `da` | `DynamicAnalyzer` | 动态分析器实例 |
| `inp` | `Any` | 个体值或已构建好的输入行列表 |
| `input_structure` | `Dict` | 输入结构（可选） |

**实现逻辑**：若 `inp` 已是字符串列表则直接传给 `da.run_with_input`；否则先调用 `_build_input_lines` 转换再执行。

---

### `_normalize_distance(d)`

**作用**：将分支距离标准化到 `[0, 1]` 区间。

| 参数 | 类型 | 说明 |
|------|------|------|
| `d` | `float` | 原始分支距离（`0` 表示已覆盖） |

**公式**：
$$
\text{norm}(d) = \begin{cases} 0 & d \leq 0 \\ \dfrac{d}{d+1} & d > 0 \end{cases}
$$

将无界的距离压缩到 `(0, 1)`，使适应度计算更稳定。

---

## `TestcaseGenerator` 类

### 构造函数 `__init__`

```python
TestcaseGenerator(
    source_code=None, *, file_path=None, xml_file=None,
    population_size=30, max_generations=50, tournament_size=3,
    crossover_rate=0.8, mutation_rate=0.3, elite_ratio=0.1,
    stagnation_limit=10, timeout=10.0, random_seed=None
)
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `source_code` | `str` | `None` | 待测源代码字符串（与 `file_path` 二选一） |
| `file_path` | `str` | `None` | 待测源代码文件路径 |
| `xml_file` | `str` | `None` | 已有测试用例的 XML 种子文件路径，为 `None` 则不使用 |
| `population_size` | `int` | `30` | 种群大小（每代同时维护的个体数） |
| `max_generations` | `int` | `50` | GA 最大迭代代数 |
| `tournament_size` | `int` | `3` | 锦标赛选择每次随机取样的个体数，越大选择压力越强 |
| `crossover_rate` | `float` | `0.8` | 交叉概率，低于此值时直接复制父代 |
| `mutation_rate` | `float` | `0.3` | 变异概率，低于此值时跳过变异 |
| `elite_ratio` | `float` | `0.1` | 精英比例，每代保留适应度最好的前 `population_size × elite_ratio` 个个体 |
| `stagnation_limit` | `int` | `10` | 某分支连续多少代距离无改善后触发约束求解兜底 |
| `timeout` | `float` | `10.0` | 单次动态执行超时时间（秒） |
| `random_seed` | `int` | `None` | 随机种子，设置后结果可复现 |

---

### `generate()`

**作用**：主入口，执行完整测试用例生成流程。

**返回**：精简后的测试用例列表，每条记录：
```python
{
    'input'           : List[str],   # 每行输入字符串
    'input_display'   : str,         # 展示用的输入字符串
    'covered_branches': List[str],   # 覆盖的 branch_id 列表（如 ['B1', 'B2']）
    'output'          : str,         # 程序实际输出
}
```

**实现逻辑**：
1. 调用 `_run_static_analysis()` 完成静态分析
2. 若提供 XML 种子文件，解析并动态执行，收集初始覆盖信息
3. 调用 `_init_population()` 生成初始种群
4. 进入 GA 主循环（至多 `max_generations` 代）：
   - 执行种群 → 更新覆盖 → 计算适应度
   - 若所有分支已覆盖则提前终止
   - 锦标赛选择 → 交叉变异 → 合并种群
   - 对停滞分支调用约束求解
5. 最终执行一次完整种群评估
6. 调用 `_greedy_minimize()` 精简测试集并返回

---

### `_run_static_analysis()`

**作用**：调用 `StaticAnalyzer` 完成静态分析，初始化所有内部数据结构。

**实现逻辑**：
1. 依次调用静态分析器的 6 个方法：
   - `build_control_flow_graph()` — 构建 CFG
   - `extract_predicates_and_constraints()` — 提取谓词和约束
   - `get_variable_types()` — 推断变量类型
   - `build_data_dependency_graph()` — 构建数据依赖图
   - `identify_input_structure()` — 识别输入结构
   - `get_branch_constraint_map()` — 获取分支约束映射
2. 填充内部字段：`_branches`、`_input_structure`、`_predicates`、`_pred_map`、`_mutation_candidates`
3. 为每个分支初始化 `_best_distance` 为正无穷
4. 构建 `_cov_key_to_branch_id` 反查表：将动态分析中使用的 coverage key（`pred_id` 或行号字符串）映射回统一的 `branch_id`（B1/B2 等）
5. 创建 `DynamicAnalyzer` 实例

---

### `_init_population(xml_individuals)`

**作用**：生成初始种群。

| 参数 | 类型 | 说明 |
|------|------|------|
| `xml_individuals` | `List[Any]` | 从 XML 解析出的输入行列表，优先加入种群 |

**实现逻辑**：
1. 将 XML 个体、边界值种子（`_boundary_seeds`）、模板种子（`_template_seeds`）依次加入候选池
2. 若候选不足 `pop_size`，随机补充
3. 去重后截取前 `pop_size` 个返回

---

### `_boundary_seeds()`

**作用**：从谓词的边界值生成种子个体。

**实现逻辑**：遍历所有谓词的 `boundary_values` 字段，将每个数值型边界值通过 `_make_individual_from_scalar` 包装成合法个体。

---

### `_template_seeds()`

**作用**：从变异候选值中生成模板种子。

**实现逻辑**：遍历 `_mutation_candidates` 中每个变量的每个分支候选值，同样包装为个体，达到 `pop_size` 后提前返回。

---

### `_make_individual_from_scalar(scalar)`

**作用**：将一个标量值包装成与 `input_structure` 匹配的个体。

| 参数 | 类型 | 说明 |
|------|------|------|
| `scalar` | `Any` | 标量值（通常来自边界值或候选值） |

**实现逻辑**：
- 无输入变量 → 直接返回 `scalar`
- 单输入变量 → 调用 `_coerce_value` 转换类型
- 多输入变量 → 第一个变量用 `scalar`，其余随机生成，打包为 - 多输入变量 → 第一个变量用 scalar，其余随机生成，打包为 	uple

---

### _random_individual()

**作用**：生成一个完全随机的个体。

**实现逻辑**：根据 _input_vars 的数量和类型，调用 _random_value_for_input 生成各分量，多变量时打包为 	uple。

---

### _random_value_for_input(inp)

**作用**：根据输入变量类型描述生成随机值。

| 	ype | 生成策略 |
|--------|----------|
| int | randint(-100, 1000) |
| loat | uniform(-100.0, 1000.0) 保留3位小数 |
| str | 随机小写字母字符串，长度0~20 |
| list系列 | 长度0~10的整数或浮点数列表 |
| 其他 | randint(-100, 1000) |

---

### _evaluate_population(population)

**作用**：执行种群所有个体，返回动态分析结果列表，并更新全局覆盖集合。

**实现逻辑**：对每个个体调用 _run_one，失败时返回空结果；每次调用后更新 _covered_branches。

---

### _update_covered_branches(res)

**作用**：从执行结果中提取距离为0的分支key，加入全局覆盖集合。

**实现逻辑**：遍历 ranch_distances，distance==0.0 的条目对应的key加入 _covered_branches。

---

### _compute_fitness(res)

**作用**：计算单个个体的适应度（越小越好，0最优）。

**公式**：

\text{fitness} = \sum_{b} \text{norm}(\min\_dist_b)

**实现逻辑**：对每个分支取最小分支距离，已覆盖贡献0，未覆盖贡献 
orm(d)，无记录贡献1.0，累加得总适应度。

---

### _update_stagnation(results)

**作用**：更新每个分支的停滞计数。有改善则归零，无改善则+1。

---

### _tournament_select(population, fitness_scores)

**作用**：锦标赛选择父代。每次随机抽取 	ournament_size 个，选适应度最小者，重复 pop_size 次。

---

### _crossover(p1, p2) / _crossover_scalar(a, b) / _crossover_list(a, b)

**作用**：序列交叉产生子代。

- **标量**：40%取a，40%取b，20%取算术平均
- **list**：单点交叉 [:pt] + b[pt:]
- **tuple**：对每个分量独立做标量交叉
- 若随机数 > crossover_rate 则直接复制父代

---

### _mutate(individual, results)

**作用**：引导变异。找距离最近的未覆盖分支，根据其约束条件有方向地调整输入值。

若随机数 > mutation_rate 跳过；无目标分支时做随机扰动。

---

### _guided_perturb(individual, branch)

**作用**：根据 	rue_constraint 的操作符调整输入值方向。

| op | 调整 |
|----|------|
| Lt/LtE | 减小 |
| Gt/GtE | 增大 |
| Eq | 直接设为目标值 |
| NotEq | 目标值+step |

步长与当前值和目标值的差距成比例（差距越大步长越大）。

---

### _crossover_and_mutate(parents, results)

**作用**：两两交叉+变异产生子代。交叉结果若与已有个体重复最多重试3次，仍重复则随机扰动。

---

### _merge_population(population, offspring, fitness_scores)

**作用**：合并种群。保留适应度最低的 elite_size 个精英，其余从子代+非精英父代的混合池中随机填充。

---

### _constraint_solve(stagnant_branch_ids)

**作用**：对停滞分支直接根据约束构造满足条件的输入（兜底）。

**实现逻辑**：从 	rue_constraint 提取操作符和比较值，调用 _constraint_candidates 生成候选值，再调用 _make_individual_with_var 构造完整个体。

---

### _constraint_candidates(op, value)

**作用**：根据操作符生成满足约束的候选值列表。

| op | 候选值 |
|----|--------|
| Lt | [v-1, v-2, v-10] |
| LtE | [v, v-1, v-5] |
| Gt | [v+1, v+2, v+10] |
| GtE | [v, v+1, v+5] |
| Eq | [v] |
| NotEq | [v+1, v-1] |

---

### _make_individual_with_var(var, val)

**作用**：构造使指定变量等于 al 的完整个体。按变量名匹配 _input_vars，其余变量随机生成，找不到匹配则设置第一个变量。

---

### _greedy_minimize(test_cases)

**作用**：两阶段贪心最小化：最大化覆盖率后去冗余，返回最小覆盖测试集。

**实现逻辑**：

**阶段一（贪心扩展）**：
- 去重（按 input_display）
- 每轮从剩余用例中选「新增覆盖分支数最多」的用例加入结果集
- 直到无新分支可覆盖或已覆盖全部已知分支

**阶段二（冗余裁剪）**：
- 按覆盖分支数升序排列（贡献少的优先尝试去除）
- 逐一尝试去除每个用例：若其余用例仍能维持相同覆盖集，则去除
- 保证最终测试集在覆盖率不降的前提下数量最少

---

### _make_test_case(individual, res)

**作用**：将个体和执行结果打包为标准测试用例字典。

**实现逻辑**：
1. 若执行失败则返回 None
2. 调用 _build_input_lines 构建输入行
3. 遍历 ranch_distances，将 distance==0.0 的条目通过 _cov_key_to_branch_id 反查表映射为统一的 ranch_id
4. 返回含 input、input_display、covered_branches、output 的字典

---

### _perturb_scalar(v) （静态方法）

**作用**：对单个标量值施加随机扰动。

- loat：加高斯噪声，标准差为 max(1.0, |v|×0.1)
- int：加减随机步长，步长为 max(1, |v|//10 + 1)

---

## 内部数据结构说明

| 字段 | 类型 | 说明 |
|------|------|------|
| _branches | List[Dict] | 静态分支列表，每项含 ranch_id、pred_id、lineno、cfg_node_id、	rue_constraint、alse_constraint |
| _covered_branches | set | 全局已覆盖的 coverage_key 集合（在所有代中累积） |
| _best_distance | Dict[str, float] | 每个 ranch_id 的历史最优分支距离 |
| _stagnation_counter | Dict[str, int] | 每个 ranch_id 连续无改善的代数 |
| _cov_key_to_branch_id | Dict[str, str] | coverage_key → branch_id 反查表（pred_id 或行号 → B1/B2） |
| _pred_map | Dict[str, Dict] | pred_id → 谓词信息字典 |
