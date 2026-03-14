# 动态分析器 (DynamicAnalyzer) 文档

## 概述

`DynamicAnalyzer` 是一个基于静态分析结果的动态执行分析工具，通过追踪程序执行过程收集运行时信息。

**主要功能：**
- 执行轨迹追踪（行级）
- CFG边和块覆盖统计
- 变量值序列记录
- 分支距离计算
- 运行时类型验证

---

## 初始化

### `__init__(source_code, static_analysis_result, timeout=5.0, max_execution_steps=10000)`

初始化动态分析器。

**参数：**
- `source_code` (str): 源代码字符串
- `static_analysis_result` (dict): 静态分析结果，包含cfg、predicates、data_dependencies、variable_types
- `timeout` (float): 执行超时时间（秒），默认5.0
- `max_execution_steps` (int): 最大执行步数，默认10000

**返回：** 无

---

## 核心方法

### `run_with_input(input_data=None)`

使用指定输入执行程序并收集动态分析信息。

**参数：**
- `input_data` (Any): 输入数据，可以是单个值或列表（每个元素对应一次input()调用）

**返回：** dict
```python
{
    'execution_info': {
        'success': bool,              # 执行是否成功
        'output': str,                # 程序输出
        'error': str or None,         # 错误信息
        'execution_steps': int,       # 总执行步数
        'execution_time': float       # 执行时间（秒）
    },
    'trace': {
        'summary': {
            'step_count': int,        # 记录的轨迹步数
            'total_steps': int        # 总执行步数
        },
        'records': [...]              # 轨迹记录列表
    },
    'coverage': {
        'summary': {
            'edge_count': int,        # 覆盖的边数
            'block_count': int        # 覆盖的块数
        },
        'edges': {
            'total': [...],           # 所有覆盖的边
            'detail': [...]           # 边的详细信息
        },
        'blocks': {
            'total': [...],           # 所有覆盖的块
            'detail': [...]           # 块的详细信息
        }
    },
    'variable_values': {
        'summary': {...},             # 变量统计
        'details': {...}              # 变量值序列
    },
    'branch_distances': {
        'summary': {...},             # 分支统计
        'details': {...}              # 分支距离详情
    },
    'type_validation': {...}          # 类型验证结果
}
```

---

### `run_multiple_inputs(inputs_list)`

使用多个输入执行程序。

**参数：**
- `inputs_list` (list): 输入数据列表

**返回：** list - 每个元素是`run_with_input()`的返回值

---

### `aggregate_coverage(results_list)`

聚合多次执行的覆盖信息。

**参数：**
- `results_list` (list): `run_with_input()`返回值的列表

**返回：** dict
```python
{
    'total_edges': list,      # 所有覆盖的边
    'total_blocks': list,     # 所有覆盖的块
    'edge_count': int,        # 总覆盖边数
    'block_count': int        # 总覆盖块数
}
```

---

## 内部方法

### `_extract_static_info()`

从静态分析结果中提取关键信息（CFG节点、边、关键变量等）。

**返回：** 无（更新内部状态）

---

### `_reset_results()`

重置动态分析结果，为新的执行做准备。

**返回：** 无

---

### `_trace_line(frame, local_ns, global_ns)`

追踪行执行（由sys.settrace调用），记录变量值、更新覆盖、计算分支距离。

**返回：** 无

---

### `_update_cfg_coverage(current_line)`

更新CFG边和块覆盖，支持精确匹配、间接匹配和全局搜索。

**返回：** 无

---

### `_check_branch_condition(lineno, frame)`

检查分支条件并计算分支距离。

**返回：** 无

---

### `_calculate_branch_distance(condition, frame)`

计算分支距离（条件为真时距离=0，为假时计算到满足条件的距离）。

**返回：** float or None

---

### `_compute_false_branch_distance(condition, local_vars, global_vars)`

计算条件为假时的分支距离，支持多种表达式类型。

**支持的表达式：**
- 简单比较：`x > 10`, `y <= 5`
- 链式比较：`10 < x < 20`
- 布尔变量：`flag`, `not flag`
- 模运算：`h % 11 == 0`

**返回：** float - 分支距离值

---

### `_calc_single_comparison_distance(var_val, op, const, reversed_order=False)`

计算单个比较表达式的距离。

**参数：**
- `var_val` (numeric): 变量的当前值
- `op` (str): 比较操作符（<, >, <=, >=, ==, !=）
- `const` (numeric): 比较的常量值
- `reversed_order` (bool): 是否反转操作符（用于处理`const op var`的情况）

**距离计算规则：**
- `x < c` 且为假（x >= c）：距离 = max(0, x - c + 1)
- `x <= c` 且为假（x > c）：距离 = max(0, x - c)
- `x > c` 且为假（x <= c）：距离 = max(0, c - x + 1)
- `x >= c` 且为假（x < c）：距离 = max(0, c - x)
- `x == c` 且为假：距离 = |x - c|
- `x != c` 且为假（x == c）：距离 = 1.0

**返回：** float or None

---

### `_extract_condition(line)`

从代码行中提取条件表达式。

**功能：**
- 移除关键字（if, elif, while）
- 移除冒号及之后的内容
- 返回纯净的条件表达式

**返回：** str - 条件表达式

---

### `_validate_types()`

验证运行时类型，比较静态类型和运行时类型。

**返回：** dict
```python
{
    'confirmed_types': {...},  # 确认的类型
    'corrected_types': {...},  # 修正的类型
    'new_types': {...}         # 新发现的类型
}
```

---

### `_create_safe_builtins(input_data)`

创建安全的builtins环境，限制可用函数，模拟print和input。

**返回：** dict - 安全的builtins

---

## 使用示例

### 基本使用

```python
from static_analyzer import StaticAnalyzer
from dynamic_analyzer import DynamicAnalyzer

# 源代码
code = """
a = float(input())
b = int(input())
print(a + b)
"""

# 静态分析
static_analyzer = StaticAnalyzer(source_code=code)
static_result = {
    'cfg': static_analyzer.build_control_flow_graph(),
    'predicates': static_analyzer.extract_predicates_and_constraints(),
    'data_dependencies': static_analyzer.build_data_dependency_graph(),
    'variable_types': static_analyzer.get_variable_types()
}

# 动态分析
analyzer = DynamicAnalyzer(code, static_result)
result = analyzer.run_with_input(['1.5', '2'])

print(f"成功: {result['execution_info']['success']}")
print(f"输出: {result['execution_info']['output']}")
print(f"边覆盖: {result['coverage']['summary']['edge_count']}")
```

### 多次执行和聚合

```python
# 执行多个测试
results = analyzer.run_multiple_inputs([
    ['1.0', '2'],
    ['3.5', '4'],
    ['5.0', '6']
])

# 聚合覆盖率
aggregated = analyzer.aggregate_coverage(results)
print(f"总覆盖边数: {aggregated['edge_count']}")
print(f"总覆盖块数: {aggregated['block_count']}")
```

---

## 关键特性

### 1. 变量值优化记录
只记录变化的变量值，减少冗余数据。

### 2. 智能CFG覆盖
- 精确匹配：直接查找行号对应的边
- 间接匹配：通过虚拟节点连接
- 全局搜索：兜底策略

### 3. 分支距离计算
支持多种表达式：
- 比较表达式：`<, >, <=, >=, ==, !=`
- 链式比较：`a < b < c`
- 模运算：`a % b == c`
- 布尔表达式：`not x`

### 4. 类型验证
自动比较静态类型和运行时类型，分类报告。

### 5. 安全执行
- 受限的builtins
- 超时控制
- 步数限制

---

## 返回值结构总结

| 方法 | 返回类型 | 主要字段 |
|------|---------|---------|
| `run_with_input()` | dict | execution_info, trace, coverage, variable_values, branch_distances, type_validation |
| `run_multiple_inputs()` | list | 每个元素是run_with_input()的返回值 |
| `aggregate_coverage()` | dict | total_edges, total_blocks, edge_count, block_count |

---

## 注意事项

1. **输入格式**：输入数据应该是列表，每个元素对应一个input()调用
2. **超时处理**：长时间运行的程序会被中断
3. **步数限制**：无限循环会被限制在max_execution_steps内
4. **错误处理**：执行错误会被捕获并记录在error字段中
5. **变量追踪**：只追踪静态分析中识别的关键变量

---

## 相关文件

- `static_analyzer.py` - 静态分析器
- `input_generator.py` - 输入生成器
- `test_dynamic.py` - 测试脚本
- `DYNAMIC_ANALYZER.md` - 详细API文档
