#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, ast, copy, json, os, re, subprocess, sys
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional, Tuple
from dynamic_analyzer import DynamicAnalyzer
from static_analyzer import StaticAnalyzer

class PynguinTestParser(ast.NodeVisitor):
    def __init__(self, module_aliases: Dict[str, str]):
        self.module_aliases = module_aliases
        self.calls: List[Dict[str, Any]] = []
        self.current_test: Optional[str] = None
        self.current_xfail = False
        self.env: Dict[str, Any] = {}
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        prev_test, prev_xfail, prev_env = self.current_test, self.current_xfail, self.env
        self.current_test = node.name
        self.current_xfail = any(isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and isinstance(dec.func.value, ast.Attribute) and isinstance(dec.func.value.value, ast.Name) and dec.func.value.value.id == "pytest" and dec.func.value.attr == "mark" and dec.func.attr == "xfail" for dec in node.decorator_list)
        self.env = {}
        for stmt in node.body: self.visit(stmt)
        self.current_test, self.current_xfail, self.env = prev_test, prev_xfail, prev_env
    def visit_Assign(self, node: ast.Assign) -> Any:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name): return
        target_name = node.targets[0].id
        if isinstance(node.value, ast.Call):
            call_info = self._extract_call(node.value)
            if call_info is not None:
                call_info.update({"result_var": target_name, "expected": None, "test_name": self.current_test, "xfail": self.current_xfail})
                self.calls.append(call_info)
                self.env[target_name] = {"__call_index__": len(self.calls) - 1}
                return
        value = self._safe_eval_expr(node.value)
        if value is not _UNSET: self.env[target_name] = value
    def visit_Assert(self, node: ast.Assert) -> Any:
        if not isinstance(node.test, ast.Compare): return
        if len(node.test.ops) != 1 or len(node.test.comparators) != 1 or not isinstance(node.test.ops[0], ast.Eq): return
        if not isinstance(node.test.left, ast.Name): return
        marker = self.env.get(node.test.left.id)
        if not isinstance(marker, dict) or "__call_index__" not in marker: return
        expected = self._safe_eval_expr(node.test.comparators[0])
        if expected is not _UNSET: self.calls[marker["__call_index__"]]["expected"] = expected
    def visit_Expr(self, node: ast.Expr) -> Any:
        if isinstance(node.value, ast.Call):
            call_info = self._extract_call(node.value)
            if call_info is not None:
                call_info.update({"result_var": None, "expected": None, "test_name": self.current_test, "xfail": self.current_xfail})
                self.calls.append(call_info)
    def _extract_call(self, call: ast.Call) -> Optional[Dict[str, Any]]:
        func = call.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name): return None
        module_alias = func.value.id
        if module_alias not in self.module_aliases: return None
        args: List[Any] = []
        for arg in call.args:
            value = self._safe_eval_expr(arg)
            if value is _UNSET: return None
            args.append(value)
        return {"module_alias": module_alias, "module_name": self.module_aliases[module_alias], "function_name": func.attr, "args": args}
    def _safe_eval_expr(self, node: ast.AST) -> Any:
        try: return ast.literal_eval(node)
        except Exception: pass
        if isinstance(node, ast.Name): return self.env.get(node.id, _UNSET)
        if isinstance(node, ast.List):
            values = [self._safe_eval_expr(elt) for elt in node.elts]
            return _UNSET if any(v is _UNSET for v in values) else values
        if isinstance(node, ast.Tuple):
            values = [self._safe_eval_expr(elt) for elt in node.elts]
            return _UNSET if any(v is _UNSET for v in values) else tuple(values)
        if isinstance(node, ast.Set):
            values = [self._safe_eval_expr(elt) for elt in node.elts]
            return _UNSET if any(v is _UNSET for v in values) else set(values)
        if isinstance(node, ast.Dict):
            keys = [self._safe_eval_expr(k) for k in node.keys]
            values = [self._safe_eval_expr(v) for v in node.values]
            if any(k is _UNSET for k in keys) or any(v is _UNSET for v in values):
                return _UNSET
            return dict(zip(keys, values))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = self._safe_eval_expr(node.operand)
            if operand is _UNSET:
                return _UNSET
            try:
                return -operand if isinstance(node.op, ast.USub) else +operand
            except Exception:
                return _UNSET
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "module_0" and node.func.attr == "object" and not node.args: return object()
        return _UNSET

class _Unset: pass
_UNSET = _Unset()

def parse_three_ints(text: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if text is None:
        return None
    m = re.match(r"\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def parse_four_ints(text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if text is None:
        return None
    m = re.match(r"\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def parse_two_values(text: Optional[str]) -> Optional[Tuple[str, str]]:
    if text is None:
        return None
    if " | " in text:
        left, right = text.split(" | ", 1)
        return left.strip(), right.strip()
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if len(parts) == 2:
        return parts[0], parts[1]
    return None


def is_valid_case(problem_id: Optional[str], input_text: Optional[str]) -> bool:
    values = parse_three_ints(input_text)
    if problem_id == "03025":
        return input_text is not None and bool(str(input_text).strip())
    if problem_id == "03039":
        if values is None:
            return False
        n, m, k = values
        return n >= 1 and m >= 1 and 2 <= k <= n * m
    if problem_id == "2910":
        pair = parse_two_values(input_text)
        if pair is None:
            return False
        try:
            h = float(pair[0])
            n = int(pair[1])
        except Exception:
            return False
        return h >= 0 and n >= 1
    if problem_id in {"3039", "3226"}:
        if input_text is None:
            return False
        try:
            value = ast.literal_eval(input_text.strip())
        except Exception:
            return False
        if not isinstance(value, list) or not value:
            return False
        return all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    if values is None:
        return True
    return True


def infer_label_from_module(module_name: str) -> Optional[str]:
    if module_name.startswith("s_"): return "s"
    if module_name.startswith("f_"): return "f"
    return None

def load_source_code(py_file: str) -> str:
    with open(py_file, "r", encoding="utf-8") as f: return f.read()

def parse_pynguin_test_file(test_file: str) -> Tuple[str, List[Dict[str, Any]]]:
    tree = ast.parse(load_source_code(test_file), filename=test_file)
    module_aliases: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and alias.asname.startswith("module_"): module_aliases[alias.asname] = alias.name
    parser = PynguinTestParser(module_aliases)
    parser.visit(tree)
    target_module = next((m for m in module_aliases.values() if m.startswith(("s_", "f_"))), None)
    if target_module is None: raise ValueError(f"无法从测试文件识别目标模块: {test_file}")
    return target_module, [call for call in parser.calls if call["module_name"] == target_module]

def _is_stdin_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool))


def values_match(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    try:
        return abs(float(actual) - float(expected)) < 1e-3
    except Exception:
        return False


def _safe_clone(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _contains_input_call(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "input" for child in ast.walk(node))


def _contains_any_call(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Call) for child in ast.walk(node))


def _build_direct_exec_source(source: str, py_file: str) -> Any:
    tree = ast.parse(source, filename=py_file)
    safe_body: List[ast.stmt] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            safe_body.append(stmt)
            continue
        if isinstance(stmt, ast.Assign) and not _contains_input_call(stmt) and not _contains_any_call(stmt.value):
            safe_body.append(stmt)
            continue
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None and not _contains_input_call(stmt) and not _contains_any_call(stmt.value):
            safe_body.append(stmt)
            continue
        if isinstance(stmt, ast.AugAssign) and not _contains_input_call(stmt) and not _contains_any_call(stmt.value):
            safe_body.append(stmt)
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            safe_body.append(stmt)
    module = ast.Module(body=safe_body, type_ignores=getattr(tree, "type_ignores", []))
    return ast.fix_missing_locations(module)


def evaluate_direct_call(py_file: str, function_name: str, args: List[Any]) -> Dict[str, Any]:
    source = load_source_code(py_file)
    namespace: Dict[str, Any] = {"__name__": "__pynguin_eval__"}
    module_dir = os.path.dirname(os.path.abspath(py_file))
    inserted_path = False
    if module_dir and module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        inserted_path = True
    try:
        try:
            safe_module = _build_direct_exec_source(source, py_file)
            exec(compile(safe_module, py_file, "exec"), namespace, namespace)
        except Exception as exc:
            return {"success": False, "missing": False, "error_type": type(exc).__name__, "error": str(exc)}
        func = namespace.get(function_name)
        if not callable(func):
            return {"success": False, "missing": True, "error_type": "MissingFunction", "error": function_name}
        try:
            result = func(*[_safe_clone(arg) for arg in args])
            return {"success": True, "missing": False, "return": result}
        except Exception as exc:
            return {"success": False, "missing": False, "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        if inserted_path:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass


def direct_results_match(program_result: Dict[str, Any], reference_result: Dict[str, Any]) -> bool:
    if program_result.get("success") != reference_result.get("success"):
        return False
    if not program_result.get("success"):
        return program_result.get("error_type") == reference_result.get("error_type")
    return values_match(program_result.get("return"), reference_result.get("return"))


def normalize_direct_solve_args(problem_id: Optional[str], args: List[Any]) -> Optional[List[Any]]:
    if problem_id == "03039":
        if len(args) != 3:
            return None
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in args):
            return None
        n, m, k = args
        if n >= 1 and m >= 1 and 2 <= k <= n * m:
            return [n, m, k]
        return None
    if len(args) != 1:
        return None
    value = args[0]
    if problem_id == "3039":
        if isinstance(value, set):
            value = list(value)
        elif isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list) and value:
            if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
                return [value]
        return None
    if problem_id == "3226":
        if isinstance(value, list) and value and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
            return [value]
        return None
    return None


def normalize_lifted_core_args(problem_id: Optional[str], function_name: str, args: List[Any]) -> Optional[List[Any]]:
    if problem_id == "03039":
        if function_name not in {"solve", "cmb", "calc_base", "nCk"}:
            return None
        return normalize_direct_solve_args(problem_id, args)
    if problem_id == "03025" and function_name == "solve_normalized":
        return normalize_03025_normalized_args(args)
    return None


def normalize_03025_normalized_args(args: List[Any]) -> Optional[List[int]]:
    if len(args) != 3:
        return None
    try:
        n, a, b = [int(value) for value in args]
    except Exception:
        return None
    n = abs(n) % 200 + 1
    a = abs(a) % 100
    b = abs(b) % 100
    if a + b == 0:
        a = 1
    if a + b > 100:
        total = a + b
        a = max(1, a * 100 // total)
        b = 100 - a
        if b == 0 and total > 1:
            b = 1
            a = 99
    c = 100 - a - b
    return [n, a, b, c]


def normalize_03025_args(args: List[Any]) -> Optional[List[int]]:
    if len(args) != 4:
        return None
    try:
        n, a, b, c = [int(value) for value in args]
    except Exception:
        return None
    n = max(1, abs(n) % 100000 + 1)
    a = abs(a) % 100
    b = abs(b) % 100
    if a + b == 0:
        a = 1
    if a + b > 100:
        total = a + b
        a = max(1, a * 100 // total)
        b = 100 - a
        if b == 0 and total > 1:
            b = 1
            a = 99
    c = 100 - a - b
    return [n, a, b, c]


def _serialize_stdin_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    return str(value)


def flatten_input_lines(input_lines: List[str]) -> str:
    if len(input_lines) == 1:
        return input_lines[0]
    return "\n".join(input_lines)


def build_input_variants(args: List[Any], problem_id: Optional[str] = None) -> List[List[str]]:
    if problem_id == "03025":
        normalized = normalize_03025_args(args)
        if normalized is None:
            return []
        return [[" ".join(str(value) for value in normalized)]]
    if not args or not all(_is_stdin_scalar(arg) for arg in args):
        return []
    single_line = [" ".join(_serialize_stdin_scalar(arg) for arg in args)]
    multi_line = [_serialize_stdin_scalar(arg) for arg in args]
    variants: List[List[str]] = [single_line]
    if multi_line != single_line:
        variants.append(multi_line)
    return variants

def build_analyzer(py_file: str, timeout: float = 5.0) -> DynamicAnalyzer:
    source_code = load_source_code(py_file)
    sa = StaticAnalyzer(source_code=source_code)
    static_result = {"cfg": sa.build_control_flow_graph(), "predicates": sa.extract_predicates_and_constraints(), "data_dependencies": sa.build_data_dependency_graph(), "variable_types": sa.get_variable_types(), "branch_constraint_map": sa.get_branch_constraint_map()}
    return DynamicAnalyzer(source_code, static_result, timeout=timeout)


def run_analyzer_with_input(analyzer: DynamicAnalyzer, py_file: str, input_lines: List[str]) -> Dict[str, Any]:
    module_dir = os.path.dirname(os.path.abspath(py_file))
    inserted_path = False
    if module_dir and module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        inserted_path = True
    try:
        return analyzer.run_with_input(input_lines)
    finally:
        if inserted_path:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass


def run_program_plain(py_file: str, input_lines: List[str], timeout_seconds: float = 8.0) -> Dict[str, Any]:
    input_text = flatten_input_lines(input_lines)
    if input_text and not input_text.endswith("\n"):
        input_text += "\n"
    try:
        completed = subprocess.run(
            [sys.executable, py_file],
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "TimeoutExpired"}
    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    return {
        "success": completed.returncode == 0 and bool(output),
        "output": output,
        "error": error,
        "returncode": completed.returncode,
    }


def extract_code_id(module_name: str) -> str: return module_name

def get_reference_file(py_file: str) -> str: return os.path.join(os.path.dirname(os.path.dirname(py_file)), "reference.py")

def evaluate_program(py_file: str, test_file: str) -> Dict[str, Any]:
    module_name = os.path.splitext(os.path.basename(py_file))[0]
    label = infer_label_from_module(module_name)
    bucket = "success" if label == "s" else "failure" if label == "f" else None
    problem_id = module_name.split("_")[1] if "_" in module_name else None
    target_module, calls = parse_pynguin_test_file(test_file)
    use_plain_reference = problem_id == "03025"
    max_dynamic_coverage_cases = 1 if problem_id == "03025" else None
    dynamic_coverage_cases = 0
    try:
        analyzer = build_analyzer(py_file, timeout=2.0 if problem_id == "03025" else 5.0)
    except Exception:
        analyzer = None
    ref_file = get_reference_file(py_file)
    ref_analyzer = None
    if not use_plain_reference:
        ref_analyzer = build_analyzer(ref_file)
    candidate_calls = calls if problem_id in {"02882", "03025"} else [call for call in calls if not call.get("xfail")]
    solve_like_functions = {"solve"}
    if problem_id == "03025":
        solve_like_functions.add("solve_normalized")
    solve_calls = [call for call in candidate_calls if call.get("function_name") in solve_like_functions]
    auxiliary_calls = [call for call in candidate_calls if call.get("function_name") not in solve_like_functions]
    raw_case_count = valid_case_count = invalid_input_count = mismatches = 0
    for call in candidate_calls:
        function_name = call.get("function_name")
        args = call.get("args", [])

        variants = []
        if function_name == "solve" and problem_id not in {"03039"}:
            variants = build_input_variants(args, problem_id)
        if variants:
            raw_case_count += 1
            chosen_variant: Optional[List[str]] = None
            ref_exec = {}
            for input_lines in variants:
                input_text = flatten_input_lines(input_lines)
                if not is_valid_case(problem_id, input_text):
                    continue
                if use_plain_reference:
                    maybe_ref_exec = run_program_plain(ref_file, input_lines)
                else:
                    ref_result = run_analyzer_with_input(ref_analyzer, ref_file, input_lines)
                    maybe_ref_exec = ref_result.get("execution_info", {})
                if maybe_ref_exec.get("success", False):
                    chosen_variant = input_lines
                    ref_exec = maybe_ref_exec
                    break
            if chosen_variant is None:
                invalid_input_count += 1
                continue
            valid_case_count += 1
            if use_plain_reference:
                exec_info = run_program_plain(py_file, chosen_variant)
                if analyzer is not None and (max_dynamic_coverage_cases is None or dynamic_coverage_cases < max_dynamic_coverage_cases):
                    try:
                        run_analyzer_with_input(analyzer, py_file, chosen_variant)
                        dynamic_coverage_cases += 1
                    except Exception:
                        pass
            else:
                result = run_analyzer_with_input(analyzer, py_file, chosen_variant)
                exec_info = result.get("execution_info", {})
            if not exec_info.get("success", False):
                mismatches += 1
                continue
            actual_output = exec_info.get("output", "").strip()
            expected_output = ref_exec.get("output", "").strip()
            if not values_match(actual_output, expected_output):
                mismatches += 1
            continue

        normalized_direct_args = None
        if function_name == "solve":
            normalized_direct_args = normalize_direct_solve_args(problem_id, args)
        else:
            normalized_direct_args = normalize_lifted_core_args(problem_id, function_name, args)
        if normalized_direct_args is not None:
            raw_case_count += 1
            direct_function_name = "solve"
            program_result = evaluate_direct_call(py_file, direct_function_name, normalized_direct_args)
            reference_result = evaluate_direct_call(ref_file, direct_function_name, normalized_direct_args)
            if reference_result.get("missing"):
                invalid_input_count += 1
                continue
            valid_case_count += 1
            if analyzer is not None:
                coverage_variants = build_input_variants(normalized_direct_args)
                for coverage_input in coverage_variants[:1]:
                    try:
                        run_analyzer_with_input(analyzer, py_file, coverage_input)
                    except Exception:
                        pass
            if not direct_results_match(program_result, reference_result):
                mismatches += 1
            continue

        raw_case_count += 1
        invalid_input_count += 1
    if analyzer is None:
        edge_pct = None
        block_pct = None
    else:
        aggregated = analyzer.aggregate_coverage([])
        edge_pct = round(aggregated.get("edge_coverage_rate", 0.0) * 100, 2)
        block_pct = round(aggregated.get("block_coverage_rate", 0.0) * 100, 2)
    generation_failed = valid_case_count == 0
    program_correct = False if generation_failed else ((mismatches == 0) if label == "s" else (mismatches > 0 if label == "f" else False))
    return {"module": target_module, "code_id": extract_code_id(target_module), "label": label, "bucket": bucket, "edge_coverage": edge_pct, "block_coverage": block_pct, "program_correct": program_correct, "raw_case_count": raw_case_count, "valid_case_count": valid_case_count, "invalid_input_count": invalid_input_count, "auxiliary_call_count": len(auxiliary_calls), "mismatches": mismatches, "generation_failed": generation_failed}

def average(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 2) if valid else None

def split_items(items: List[Dict]) -> Tuple[List[Dict], List[Dict]]: return [item for item in items if item["bucket"] == "success"], [item for item in items if item["bucket"] == "failure"]

def build_success_section(items: List[Dict]) -> Dict:
    generation_failed_items = [item for item in items if item.get("generation_failed")]
    valid_items = [item for item in items if not item.get("generation_failed")]
    hidden_bug_items = [item for item in valid_items if not item["program_correct"]]
    consistent_items = [item for item in valid_items if item["program_correct"]]
    return {"program_count": len(items), "valid_program_count": len(valid_items), "generation_failed_ids": [item["code_id"] for item in generation_failed_items], "generation_failed_count": len(generation_failed_items), "raw_case_count": sum(item.get("raw_case_count", 0) for item in valid_items), "valid_case_count": sum(item.get("valid_case_count", 0) for item in valid_items), "invalid_input_count": sum(item.get("invalid_input_count", 0) for item in items), "auxiliary_call_count": sum(item.get("auxiliary_call_count", 0) for item in items), "avg_edge_coverage": average([item["edge_coverage"] for item in valid_items]), "avg_block_coverage": average([item["block_coverage"] for item in valid_items]), "hidden_bug_detected_ids": [item["code_id"] for item in hidden_bug_items], "hidden_bug_detected_count": len(hidden_bug_items), "hidden_bug_detection_rate": round(len(hidden_bug_items) * 100.0 / len(valid_items), 2) if valid_items else None, "consistent_code_count": len(consistent_items), "consistent_rate": round(len(consistent_items) * 100.0 / len(valid_items), 2) if valid_items else None}


def build_failure_section(items: List[Dict]) -> Dict:
    generation_failed_items = [item for item in items if item.get("generation_failed")]
    valid_items = [item for item in items if not item.get("generation_failed")]
    detected_items = [item for item in valid_items if item["program_correct"]]
    undetected_items = [item for item in valid_items if not item["program_correct"]]
    return {"program_count": len(items), "valid_program_count": len(valid_items), "generation_failed_ids": [item["code_id"] for item in generation_failed_items], "generation_failed_count": len(generation_failed_items), "raw_case_count": sum(item.get("raw_case_count", 0) for item in valid_items), "valid_case_count": sum(item.get("valid_case_count", 0) for item in valid_items), "invalid_input_count": sum(item.get("invalid_input_count", 0) for item in items), "auxiliary_call_count": sum(item.get("auxiliary_call_count", 0) for item in items), "avg_edge_coverage": average([item["edge_coverage"] for item in valid_items]), "avg_block_coverage": average([item["block_coverage"] for item in valid_items]), "bug_detected_count": len(detected_items), "bug_detection_rate": round(len(detected_items) * 100.0 / len(valid_items), 2) if valid_items else None, "undetected_failure_ids": [item["code_id"] for item in undetected_items], "undetected_failure_count": len(undetected_items)}

def build_output_path(pynguin_dir: str, problem_id: Optional[str] = None) -> str:
    statistics_dir = os.path.join(os.path.dirname(pynguin_dir.rstrip(os.sep)), "statistics")
    os.makedirs(statistics_dir, exist_ok=True)
    prefix = f"pynguin_stats_{problem_id}_" if problem_id else "pynguin_stats_"
    return os.path.join(statistics_dir, f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


def _evaluate_program_entry(program_file: str, test_file: str) -> None:
    item = evaluate_program(program_file, test_file)
    print(json.dumps(item, ensure_ascii=False))


def evaluate_program_with_timeout(program_file: str, test_file: str, timeout_seconds: int = 20) -> Dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, __file__, "--internal-evaluate", program_file, test_file],
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        raise RuntimeError(stderr or stdout or f"subprocess exit={completed.returncode}")
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("No evaluation output produced")
    return json.loads(lines[-1])

def discover_pynguin_pairs(pynguin_root: str, dataset_root: str, problem_id: Optional[str] = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for test_file in glob(os.path.join(pynguin_root, "**", "test_*.py"), recursive=True):
        if not os.path.isfile(test_file) or os.path.basename(os.path.dirname(test_file)) == "_probe": continue
        try: module_name, _ = parse_pynguin_test_file(test_file)
        except Exception: continue
        label = infer_label_from_module(module_name)
        if label is None: continue
        current_problem_id = module_name.split("_")[1]
        if problem_id and current_problem_id != problem_id:
            continue
        subdir = "success" if label == "s" else "failure"
        program_file = os.path.join(dataset_root, current_problem_id, subdir, f"{module_name}.py")
        if os.path.isfile(program_file): pairs.append((program_file, test_file))
    return sorted(set(pairs))

def main() -> None:
    parser = argparse.ArgumentParser(description="统计 Pynguin 结果：success 中隐藏伪正确检出、failure 中错误检出")
    parser.add_argument("--pynguin-dir", default=os.path.join(os.path.dirname(__file__), "testcase_result", "pynguin"), help="Pynguin 结果目录")
    parser.add_argument("--dataset-root", default=os.path.dirname(__file__), help="数据集根目录")
    parser.add_argument("--problem", default=None, help="只统计指定题号，例如 2910 / 03039 / 3039 / 3226")
    parser.add_argument("--limit", type=int, default=None, help="只统计前 N 个程序，便于快速验证")
    parser.add_argument("--module-timeout", type=int, default=20, help="单模块统计超时时间（秒）")
    parser.add_argument("--internal-evaluate", nargs=2, metavar=("PROGRAM_FILE", "TEST_FILE"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.internal_evaluate:
        _evaluate_program_entry(args.internal_evaluate[0], args.internal_evaluate[1])
        return

    pairs = discover_pynguin_pairs(args.pynguin_dir, args.dataset_root, args.problem)
    if args.limit is not None: pairs = pairs[: args.limit]
    total = len(pairs)
    items = []
    for index, (program_file, test_file) in enumerate(pairs, start=1):
        module_name = os.path.splitext(os.path.basename(program_file))[0]
        print(f"[PROGRESS] {index}/{total} START {module_name}", flush=True)
        try:
            item = evaluate_program_with_timeout(program_file, test_file, timeout_seconds=args.module_timeout)
            items.append(item)
            print(f"[PROGRESS] {index}/{total} DONE {module_name}", flush=True)
        except subprocess.TimeoutExpired:
            label = infer_label_from_module(module_name)
            items.append({"module": module_name, "code_id": extract_code_id(module_name), "label": label, "bucket": "success" if label == "s" else "failure" if label == "f" else None, "edge_coverage": None, "block_coverage": None, "program_correct": False, "raw_case_count": 0, "valid_case_count": 0, "invalid_input_count": 0, "auxiliary_call_count": 0, "mismatches": -1, "generation_failed": True, "error": f"Module timeout > {args.module_timeout}s"})
            print(f"[PROGRESS] {index}/{total} TIMEOUT {module_name}", flush=True)
        except Exception as exc:
            label = infer_label_from_module(module_name)
            items.append({"module": module_name, "code_id": extract_code_id(module_name), "label": label, "bucket": "success" if label == "s" else "failure" if label == "f" else None, "edge_coverage": None, "block_coverage": None, "program_correct": False, "raw_case_count": 0, "valid_case_count": 0, "invalid_input_count": 0, "auxiliary_call_count": 0, "mismatches": -1, "generation_failed": True, "error": str(exc)})
            print(f"[PROGRESS] {index}/{total} ERROR {module_name}: {type(exc).__name__}: {exc}", flush=True)
    success_items, failure_items = split_items(items)
    summary = {"algorithm": "pynguin", "problem_id": args.problem, "file_count": len(items), "input_validity_rule": "reference_program_accepts_input; 03025 pynguin args are normalized to legal inputs", "success": build_success_section(success_items), "failure": build_failure_section(failure_items)}
    output_path = build_output_path(args.pynguin_dir, args.problem)
    with open(output_path, "w", encoding="utf-8") as f: json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已写入: {output_path}")

if __name__ == "__main__":
    main()
