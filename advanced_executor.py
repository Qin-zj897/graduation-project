import sys
import io
import traceback
import threading
import time
import ctypes
import inspect
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# ========== 异常定义 ==========

class LoopLimitExceeded(Exception):
    """循环次数超过限制异常"""
    pass

class AppendLimitExceeded(Exception):
    """列表操作超过限制异常"""
    pass

class TimeoutExceeded(Exception):
    """执行超时异常"""
    pass

# ========== 增强版执行器 ==========

class AdvancedExecutor:
    """增强版代码执行器，支持覆盖收集、超时控制、死循环防护"""
    
    def __init__(self, timeout=5, max_iterations=10000, max_appends=10000):
        self.coverage_lines = set()
        self.original_trace = None
        self.timeout = timeout
        self.max_iterations = max_iterations
        self.max_appends = max_appends
        self._stop_execution = False
        self._execution_thread = None
        self._iteration_count = 0
        
    # ========== 线程控制 ==========
    
    def _async_raise(self, tid, exctype):
        """强制中断线程"""
        if not inspect.isclass(exctype):
            raise TypeError("Only types can be raised")
        
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(tid), 
            ctypes.py_object(exctype)
        )
        
        if res == 0:
            raise ValueError("Invalid thread id")
        elif res != 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
    
    # ========== 覆盖收集 ==========
    
    def trace_hook(self, frame, event, arg):
        """覆盖收集钩子"""
        if event == 'line':
            self.coverage_lines.add(frame.f_lineno)
        return self.trace_hook
    
    # ========== 安全数据结构 ==========
    
    def _safe_range(self, max_loops):
        """安全的range函数，限制总循环次数"""
        loop_counter = [0]
        
        def safe_range(*args):
            loop_counter[0] += 1
            if loop_counter[0] > max_loops:
                raise LoopLimitExceeded(f"超过最大循环次数限制: {max_loops}")
            return range(*args)
        
        return safe_range
    
    class SafeList(list):
        """安全的列表类，限制append操作次数"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._append_count = 0
            self._max_appends = 10000
            
        def set_max_appends(self, max_appends):
            """设置最大append次数"""
            self._max_appends = max_appends
            
        def append(self, item):
            self._append_count += 1
            if self._append_count > self._max_appends:
                raise AppendLimitExceeded(f"列表append操作超过限制: {self._max_appends}次")
            super().append(item)
            
        def extend(self, iterable):
            try:
                if hasattr(iterable, '__len__'):
                    length = len(iterable)
                    self._append_count += length
                    if self._append_count > self._max_appends:
                        raise AppendLimitExceeded(f"列表操作超过限制: {self._max_appends}次")
            except:
                pass
            super().extend(iterable)
            
        def remove(self, value):
            # 对于remove操作，也进行计数
            self._append_count += 1
            if self._append_count > self._max_appends:
                raise AppendLimitExceeded(f"列表操作超过限制: {self._max_appends}次")
            super().remove(value)
            
        def pop(self, index=-1):
            self._append_count += 1
            if self._append_count > self._max_appends:
                raise AppendLimitExceeded(f"列表操作超过限制: {self._max_appends}次")
            return super().pop(index)
    
    # ========== 代码注入和保护 ==========
    
    def _inject_list_protection(self, code_str):
        """向代码中注入列表保护"""
        lines = code_str.split('\n')
        protected_lines = []
        
        for line in lines:
            protected_line = line
            
            # 检测列表定义并替换为SafeList
            if '= []' in line or '= list()' in line or '=list()' in line:
                # 简单替换，实际应该用更复杂的方法
                protected_line = line.replace('= []', '= SafeList()')
                protected_line = protected_line.replace('= list()', '= SafeList()')
                protected_line = protected_line.replace('=list()', '=SafeList()')
            
            # 检测list()调用
            if 'list(' in line and 'SafeList' not in line:
                protected_line = line.replace('list(', 'SafeList(')
            
            protected_lines.append(protected_line)
        
        return '\n'.join(protected_lines)
    
    def _detect_dangerous_patterns(self, code_str):
        """检测代码中的危险模式"""
        patterns = [
            # 危险循环模式：在遍历列表时修改同一列表
            (r'for\s+\w+\s+in\s+(\w+)\s*:.*\1\.(append|extend|remove|pop|insert)\s*\(', 
             "危险: 在遍历列表时修改同一列表"),
            
            # 无限循环模式
            (r'while\s+(True|1|\(\))\s*:', 
             "危险: 可能无限循环"),
            
            # while循环条件可能永远为真
            (r'while\s+(\w+)\s*(>|<|>=|<=|==|!=)\s*[\w\.]+\s*:(?!.*\1\s*=)', 
             "警告: 循环条件可能永远成立"),
            
            # 嵌套死循环风险
            (r'for\s+.*:\s*for\s+.*:\s*while\s+.*:', 
             "警告: 多层嵌套循环"),
        ]
        
        issues = []
        for pattern, description in patterns:
            if re.search(pattern, code_str, re.DOTALL | re.IGNORECASE):
                issues.append(description)
        
        return issues
    
    # ========== 核心执行方法 ==========
    
    def _execute_with_protection(self, code_str, input_str):
        """带保护的代码执行"""
        self.coverage_lines.clear()
        
        # 保存原来的状态
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        self.original_trace = sys.gettrace()
        
        try:
            # 设置覆盖收集
            sys.settrace(self.trace_hook)
            
            # 重定向IO
            sys.stdin = io.StringIO(input_str)
            sys.stdout = io.StringIO()
            
            # 注入列表保护
            protected_code = self._inject_list_protection(code_str)
            
            # 创建安全的环境
            safe_builtins = {
                'range': self._safe_range(self.max_iterations),
                'len': len,
                'print': print,
                'input': input,
                'eval': eval,
                'int': int,
                'float': float,
                'str': str,
                'format': format,
                'max': max,
                'min': min,
                'sum': sum,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'list': self.SafeList,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'bool': bool,
                'type': type,
                'isinstance': isinstance,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'reversed': reversed,
                '__import__': __import__
            }
            
            # 设置SafeList的最大append次数
            self.SafeList._max_appends = self.max_appends
            
            safe_globals = {
                '__builtins__': safe_builtins,
                'SafeList': self.SafeList
            }
            
            # 执行代码
            exec(protected_code, safe_globals)
            
            # 获取输出
            output = sys.stdout.getvalue().strip()
            
            return {
                'output': output,
                'coverage': sorted(list(self.coverage_lines)),
                'success': True,
                'timeout': False,
                'interrupted': False
            }
            
        except (LoopLimitExceeded, AppendLimitExceeded) as e:
            return {
                'output': f"LOOP_LIMIT_ERROR: {str(e)}",
                'coverage': sorted(list(self.coverage_lines)),
                'success': False,
                'error': str(e),
                'timeout': False,
                'interrupted': True
            }
        except Exception as e:
            error_msg = str(e)
            return {
                'output': f"ERROR: {error_msg}",
                'coverage': sorted(list(self.coverage_lines)),
                'success': False,
                'error': traceback.format_exc(),
                'timeout': False,
                'interrupted': False
            }
        finally:
            # 恢复状态
            try:
                sys.settrace(self.original_trace)
                sys.stdin = old_stdin
                sys.stdout = old_stdout
            except:
                pass
    
    def execute_with_coverage(self, code_str, input_str):
        """执行代码并收集覆盖（带超时和死循环防护）"""
        self._execution_thread = None
        self._stop_execution = False
        
        def run_in_thread():
            self._execution_thread = threading.current_thread().ident
            return self._execute_with_protection(code_str, input_str)
        
        try:
            # 先检测危险模式
            dangerous_patterns = self._detect_dangerous_patterns(code_str)
            if dangerous_patterns:
                print(f"  检测到危险模式: {dangerous_patterns}")
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                result = future.result(timeout=self.timeout)
                return result
        except FutureTimeoutError:
            # 超时后强制中断
            self._stop_execution = True
            
            if self._execution_thread:
                try:
                    self._async_raise(self._execution_thread, KeyboardInterrupt)
                except:
                    pass
            
            time.sleep(0.1)
            
            return {
                'output': f"TIMEOUT_ERROR: 执行超时（{self.timeout}秒）",
                'coverage': [],
                'success': False,
                'error': f"执行超时（{self.timeout}秒）",
                'timeout': True,
                'interrupted': True
            }
        except Exception as e:
            return {
                'output': f"EXECUTION_ERROR: {str(e)}",
                'coverage': [],
                'success': False,
                'error': str(e),
                'timeout': False,
                'interrupted': False
            }
    
    def execute_with_timeout(self, code_str, input_str, timeout=None):
        """执行代码（带超时，不收集覆盖）"""
        if timeout is None:
            timeout = self.timeout
        
        def run_code():
            try:
                old_stdin = sys.stdin
                old_stdout = sys.stdout
                
                sys.stdin = io.StringIO(input_str)
                sys.stdout = io.StringIO()
                
                # 使用保护执行
                result = self._execute_with_protection(code_str, input_str)
                
                sys.stdin = old_stdin
                sys.stdout = old_stdout
                
                return {
                    'output': result['output'],
                    'success': result['success'],
                    'error': result.get('error', ''),
                    'timeout': False,
                    'interrupted': result.get('interrupted', False)
                }
                
            except Exception as e:
                return {
                    'output': f"ERROR: {str(e)}",
                    'success': False,
                    'error': str(e),
                    'timeout': False,
                    'interrupted': False
                }
            finally:
                try:
                    sys.stdin = sys.__stdin__
                    sys.stdout = sys.__stdout__
                except:
                    pass
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_code)
                result = future.result(timeout=timeout)
                return result
        except FutureTimeoutError:
            return {
                'output': f"TIMEOUT_ERROR: 执行超时（{timeout}秒）",
                'success': False,
                'error': "执行超时",
                'timeout': True,
                'interrupted': True
            }
        except Exception as e:
            return {
                'output': f"EXECUTION_ERROR: {str(e)}",
                'success': False,
                'error': str(e),
                'timeout': False,
                'interrupted': False
            }

# ========== 便捷函数 ==========

# 创建默认执行器实例
default_executor = AdvancedExecutor(timeout=5, max_iterations=10000, max_appends=10000)

def execute_code(code_str, input_str, timeout=None):
    """简化接口（带超时）"""
    if timeout is not None:
        executor = AdvancedExecutor(timeout=timeout)
    else:
        executor = default_executor
    
    result = executor.execute_with_timeout(code_str, input_str)
    return result['output']

def execute_with_coverage(code_str, input_str, timeout=None):
    """执行并收集覆盖（带超时）"""
    if timeout is not None:
        executor = AdvancedExecutor(timeout=timeout)
    else:
        executor = default_executor
    
    return executor.execute_with_coverage(code_str, input_str)

def check_code_output(code_str, input_str, expected_output, timeout=5):
    """检查代码输出是否与期望一致（带超时）"""
    executor = AdvancedExecutor(timeout=timeout)
    result = executor.execute_with_timeout(code_str, input_str)
    
    if result.get('timeout', False):
        return False, f"执行超时（{timeout}秒）"
    if result.get('interrupted', False):
        return False, f"执行被中断: {result['output']}"
    if not result.get('success', False):
        return False, f"执行错误: {result['output']}"
    
    actual_output = result['output']
    
    # 尝试处理浮点数比较
    try:
        actual_float = float(actual_output)
        expected_float = float(expected_output)
        if abs(actual_float - expected_float) < 0.01:  # 容忍0.01的误差
            return True, "输出匹配"
        else:
            return False, f"输出不匹配: {actual_output} != {expected_output}"
    except:
        # 不是浮点数，直接比较字符串
        if actual_output == expected_output:
            return True, "输出匹配"
        else:
            return False, f"输出不匹配: {actual_output} != {expected_output}"

def check_multiple_testcases(code_str, testcases, timeout=5):
    """检查多个测试用例（带死循环保护）"""
    executor = AdvancedExecutor(timeout=timeout)
    results = []
    
    for i, testcase in enumerate(testcases, 1):
        print(f"  测试用例 {i}/{len(testcases)}...", end='')
        
        result = executor.execute_with_timeout(code_str, testcase['input'])
        
        if result.get('timeout', False):
            print(f" 超时")
            results.append({
                'testcase': i,
                'success': False,
                'reason': f"执行超时（{timeout}秒）",
                'input': testcase['input'],
                'expected': testcase['expected'],
                'actual': 'TIMEOUT'
            })
            continue
        
        if result.get('interrupted', False):
            print(f" 中断")
            results.append({
                'testcase': i,
                'success': False,
                'reason': f"执行被中断: {result['output']}",
                'input': testcase['input'],
                'expected': testcase['expected'],
                'actual': 'INTERRUPTED'
            })
            continue
        
        if not result.get('success', False):
            print(f" 错误: {result['output'][:30]}...")
            results.append({
                'testcase': i,
                'success': False,
                'reason': f"执行错误: {result['output']}",
                'input': testcase['input'],
                'expected': testcase['expected'],
                'actual': result['output']
            })
            continue
        
        actual_output = result['output']
        expected_output = testcase['expected']
        
        # 尝试处理浮点数比较
        try:
            actual_float = float(actual_output)
            expected_float = float(expected_output)
            if abs(actual_float - expected_float) < 0.01:
                print(f" ✓ 通过")
                results.append({
                    'testcase': i,
                    'success': True,
                    'reason': "输出匹配",
                    'input': testcase['input'],
                    'expected': expected_output,
                    'actual': actual_output
                })
            else:
                print(f" ✗ 不匹配")
                results.append({
                    'testcase': i,
                    'success': False,
                    'reason': f"输出不匹配（浮点数）",
                    'input': testcase['input'],
                    'expected': expected_output,
                    'actual': actual_output
                })
        except:
            # 不是浮点数，直接比较字符串
            if actual_output == expected_output:
                print(f" ✓ 通过")
                results.append({
                    'testcase': i,
                    'success': True,
                    'reason': "输出匹配",
                    'input': testcase['input'],
                    'expected': expected_output,
                    'actual': actual_output
                })
            else:
                print(f" ✗ 不匹配")
                results.append({
                    'testcase': i,
                    'success': False,
                    'reason': f"输出不匹配",
                    'input': testcase['input'],
                    'expected': expected_output,
                    'actual': actual_output
                })
    
    # 统计结果
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    return {
        'total': total,
        'passed': passed,
        'failed': total - passed,
        'results': results,
        'all_passed': passed == total
    }