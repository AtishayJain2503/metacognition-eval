"""
nemo_eval.tools.repl
====================
Hermetic sandboxed Python REPL execution environment for NVIDIA NeMo evaluation harness.
Features:
- AST Static Security Validator (CodeSecurityValidator)
- Safe Builtins Whitelist and Controlled Import Subsystem
- Dual-Phase AST Compilation (exec statements + eval terminal expression)
- Subprocess ProcessWorker Sandbox with hard wall-clock timeouts
- Standard stream redirection (sys.stdout, sys.stderr) with output bounding
- Stateful session management and ephemeral stateless execution
"""

import ast
import builtins
import difflib
import io
import json
import math
import multiprocessing
from multiprocessing.connection import Connection
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

from nemo_eval.tools.diagnostics import DiagnosticClassifier
from nemo_eval.tools.schemas import DiagnosticError, ToolResult


# ---------------------------------------------------------------------------
# Security Blacklists and Configuration
# ---------------------------------------------------------------------------

FORBIDDEN_MODULES: Set[str] = {
    # System & Process Execution
    "os", "sys", "subprocess", "shutil", "posix", "nt", "pty", "commands",
    "runpy", "multiprocessing", "threading", "_thread", "concurrent",
    "ctypes", "_ctypes", "builtins", "importlib", "signal",
    # Network & IPC
    "socket", "urllib", "requests", "http", "httpx", "aiohttp", "urllib3",
    "ftplib", "smtplib", "poplib", "imaplib", "telnetlib", "asyncio",
    # Code Inspection & Metaprogramming
    "inspect", "gc", "linecache", "trace", "faulthandler", "dis", "code",
    "codeop", "types", "pdb", "cProfile", "profile",
    # Platform / Windows specific
    "winreg", "win32api", "win32con", "msvcrt", "_winapi", "_posixsubprocess"
}

FORBIDDEN_ATTRIBUTES: Set[str] = {
    "__subclasses__", "__bases__", "__base__", "__mro__",
    "__globals__", "__code__", "__builtins__", "__import__",
    "__class__", "__qualname__", "__closure__", "__func__",
    "__self__", "__module__", "__dict__", "__getattribute__",
    "__init_subclass__", "__reduce__", "__reduce_ex__",
    # Frame and Coroutine internal handles
    "gi_frame", "gi_code", "f_globals", "f_locals", "f_builtins",
    "f_code", "f_back", "f_trace", "cr_frame", "cr_code"
}

FORBIDDEN_CALLS: Set[str] = {
    "eval", "exec", "compile", "__import__", "open", "getattr",
    "setattr", "delattr", "hasattr", "breakpoint", "input", "exit", "quit",
    "help", "globals", "locals", "vars", "dir"
}

ALLOWED_IMPORT_MODULES: Set[str] = {
    "math", "random", "json", "re", "datetime", "collections",
    "itertools", "functools", "statistics", "decimal", "fractions",
    "copy", "time", "string", "heapq", "bisect", "typing",
    "pandas", "numpy", "polars", "pyarrow", "scipy", "sqlite3"
}


class SecurityViolationError(Exception):
    """Raised when submitted code violates sandbox AST security rules."""
    def __init__(self, message: str, line_number: Optional[int] = None, column_offset: Optional[int] = None):
        super().__init__(message)
        self.line_number = line_number
        self.column_offset = column_offset


class CodeSecurityValidator(ast.NodeVisitor):
    """
    Static AST analyzer inspecting submitted Python code before compilation.
    Enforces strict isolation by prohibiting forbidden modules, attribute introspection,
    and dangerous builtin invocations.
    """
    def __init__(self):
        self.violations: List[Dict[str, Any]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split('.')[0]
            if base_mod in FORBIDDEN_MODULES:
                self.violations.append({
                    "type": "ForbiddenImport",
                    "message": f"Import of module '{alias.name}' is prohibited.",
                    "line": getattr(node, "lineno", 1),
                    "col": getattr(node, "col_offset", 0) + 1,
                    "token": alias.name
                })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_mod = node.module.split('.')[0]
            if base_mod in FORBIDDEN_MODULES:
                self.violations.append({
                    "type": "ForbiddenImport",
                    "message": f"Import from module '{node.module}' is prohibited.",
                    "line": getattr(node, "lineno", 1),
                    "col": getattr(node, "col_offset", 0) + 1,
                    "token": node.module
                })
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.violations.append({
                "type": "ForbiddenAttribute",
                "message": f"Access to introspection attribute '{node.attr}' is prohibited.",
                "line": getattr(node, "lineno", 1),
                "col": getattr(node, "col_offset", 0) + 1,
                "token": node.attr
            })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            self.violations.append({
                "type": "ForbiddenCall",
                "message": f"Invocation of dangerous function '{node.func.id}()' is prohibited.",
                "line": getattr(node, "lineno", 1),
                "col": getattr(node, "col_offset", 0) + 1,
                "token": node.func.id
            })
        self.generic_visit(node)

    def check_code(self, code: str) -> List[Dict[str, Any]]:
        """Parses source code and returns list of security violations without raising."""
        tree = ast.parse(code, filename="<repl>")
        self.violations.clear()
        self.visit(tree)
        return list(self.violations)

    def validate(self, code: str) -> None:
        """
        Parses source code into AST and runs security verification.
        Raises SyntaxError if parsing fails, or SecurityViolationError on violation.
        """
        tree = ast.parse(code, filename="<repl>")
        self.violations.clear()
        self.visit(tree)

        if self.violations:
            first_v = self.violations[0]
            raise SecurityViolationError(
                message=first_v["message"],
                line_number=first_v["line"],
                column_offset=first_v["col"]
            )


# ---------------------------------------------------------------------------
# Safe Builtins Whitelist and Controlled Import
# ---------------------------------------------------------------------------

def safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    """Controlled import wrapper allowing only vetted analytical and standard modules."""
    root_module = name.split('.')[0]
    if root_module not in ALLOWED_IMPORT_MODULES:
        raise ImportError(f"Import of module '{name}' is restricted in the sandbox environment.")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS: Dict[str, Any] = {
    # Core Types & Constructors
    "int": int, "float": float, "str": str, "bool": bool, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "frozenset": frozenset,
    "bytes": bytes, "bytearray": bytearray, "complex": complex,
    
    # Mathematical & Numerical Primitives
    "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
    "divmod": divmod, "pow": pow,
    
    # Sequence Operations & Iterators
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
    "all": all, "any": any, "slice": slice, "iter": iter, "next": next,
    
    # Formatting & Representation
    "isinstance": isinstance, "issubclass": issubclass, "repr": repr,
    "print": print, "format": format, "chr": chr, "ord": ord,
    "hex": hex, "bin": bin, "oct": oct, "ascii": ascii,
    
    # Standard Exceptions
    "Exception": Exception, "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError, "AttributeError": AttributeError,
    "BufferError": BufferError, "EOFError": EOFError,
    "FloatingPointError": FloatingPointError, "IndexError": IndexError,
    "KeyError": KeyError, "LookupError": LookupError,
    "MemoryError": MemoryError, "NameError": NameError,
    "NotImplementedError": NotImplementedError, "OverflowError": OverflowError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
    "StopAsyncIteration": StopAsyncIteration, "SyntaxError": SyntaxError,
    "IndentationError": IndentationError, "TabError": TabError,
    "TypeError": TypeError, "UnboundLocalError": UnboundLocalError,
    "UnicodeError": UnicodeError, "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError, "Warning": Warning,
    "UserWarning": UserWarning, "DeprecationWarning": DeprecationWarning,
    
    # Safe Constants
    "True": True, "False": False, "None": None, "Ellipsis": ...,
    
    # Safe Import
    "__import__": safe_import,
}


def build_default_namespace() -> Dict[str, Any]:
    """Constructs a clean, safe initial execution namespace with analytical modules."""
    ns: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS.copy(),
        "math": math,
        "json": json,
        "re": re,
    }
    if np is not None:
        ns["np"] = np
        ns["numpy"] = np
    if pd is not None:
        ns["pd"] = pd
        ns["pandas"] = pd
    return ns


# ---------------------------------------------------------------------------
# Dual-Phase AST Compilation & Execution
# ---------------------------------------------------------------------------

def compile_and_execute_ast(code: str, namespace: Dict[str, Any]) -> Any:
    """
    Executes Python code in the provided namespace using dual-phase AST compilation.
    Evaluates body statements as exec and terminal expression as eval to capture return values.
    """
    tree = ast.parse(code, filename="<repl>", mode="exec")
    if not tree.body:
        return None

    last_stmt = tree.body[-1]

    if isinstance(last_stmt, ast.Expr):
        exec_stmts = tree.body[:-1]
        eval_expr = last_stmt.value

        if exec_stmts:
            exec_mod = ast.Module(body=exec_stmts, type_ignores=[])
            compiled_exec = compile(exec_mod, filename="<repl>", mode="exec")
            exec(compiled_exec, namespace)

        eval_mod = ast.Expression(body=eval_expr)
        compiled_eval = compile(eval_mod, filename="<repl>", mode="eval")
        return eval(compiled_eval, namespace)
    else:
        compiled_exec = compile(tree, filename="<repl>", mode="exec")
        exec(compiled_exec, namespace)
        return None


# ---------------------------------------------------------------------------
# Subprocess Worker Process Loop
# ---------------------------------------------------------------------------

def _repl_worker_loop(pipe: Connection, max_output_length: int = 50000) -> None:
    """
    Top-level worker process loop for executing Python code in process isolation.
    Listens on pipe for commands (EXEC, RESET, TERMINATE).
    """
    namespace = build_default_namespace()

    while True:
        try:
            if not pipe.poll(timeout=None):
                continue
            msg = pipe.recv()
        except (EOFError, BrokenPipeError, KeyboardInterrupt):
            break

        cmd = msg.get("command")

        if cmd == "TERMINATE":
            break

        elif cmd == "RESET":
            namespace = build_default_namespace()
            try:
                pipe.send({"status": "reset_complete"})
            except Exception:
                break

        elif cmd == "EXEC":
            code = msg.get("code", "")
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            start_perf = time.perf_counter()

            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                ret_val = compile_and_execute_ast(code, namespace)
                elapsed_ms = (time.perf_counter() - start_perf) * 1000.0

                raw_stdout = stdout_buf.getvalue()
                raw_stderr = stderr_buf.getvalue()

                # Bounded output
                if len(raw_stdout) > max_output_length:
                    raw_stdout = raw_stdout[:max_output_length] + f"\n... [Output truncated. Total chars: {len(raw_stdout)}]"
                if len(raw_stderr) > max_output_length:
                    raw_stderr = raw_stderr[:max_output_length] + f"\n... [Output truncated. Total chars: {len(raw_stderr)}]"

                # Data serialization safety
                safe_data = ret_val
                if pd is not None and isinstance(ret_val, pd.DataFrame):
                    safe_data = ret_val.to_dict(orient="records") if len(ret_val) <= 100 else ret_val.head(50).to_dict(orient="records")
                elif pd is not None and isinstance(ret_val, pd.Series):
                    safe_data = ret_val.to_dict()
                elif np is not None and isinstance(ret_val, (np.ndarray, np.generic)):
                    safe_data = ret_val.tolist()

                resp = {
                    "status": "success",
                    "execution_time_ms": elapsed_ms,
                    "data": safe_data,
                    "stdout": raw_stdout,
                    "stderr": raw_stderr,
                    "error": None,
                }
                pipe.send(resp)

            except BaseException as exc:
                elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
                raw_stdout = stdout_buf.getvalue()
                raw_stderr = stderr_buf.getvalue()

                if len(raw_stdout) > max_output_length:
                    raw_stdout = raw_stdout[:max_output_length] + f"\n... [Output truncated. Total chars: {len(raw_stdout)}]"
                if len(raw_stderr) > max_output_length:
                    raw_stderr = raw_stderr[:max_output_length] + f"\n... [Output truncated. Total chars: {len(raw_stderr)}]"

                session_keys = [k for k in namespace.keys() if not k.startswith("__")]
                diag = DiagnosticClassifier.create_diagnostic_error(
                    exc=exc,
                    source_code=code,
                    context={"session_vars": session_keys}
                )

                resp = {
                    "status": "error",
                    "execution_time_ms": elapsed_ms,
                    "data": None,
                    "stdout": raw_stdout,
                    "stderr": raw_stderr,
                    "error": diag.model_dump(),
                }
                try:
                    pipe.send(resp)
                except Exception:
                    break

            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr


# ---------------------------------------------------------------------------
# Process Worker Sandbox
# ---------------------------------------------------------------------------

class ProcessWorkerSandbox:
    """
    Manages an isolated child process running Python REPL execution blocks.
    Enforces hard wall-clock timeouts via process termination.
    """
    def __init__(self, stateful: bool = True, max_output_length: int = 50000):
        self.stateful = stateful
        self.max_output_length = max_output_length
        self.process: Optional[multiprocessing.Process] = None
        self.parent_conn: Optional[Connection] = None
        self.child_conn: Optional[Connection] = None
        self._is_alive = False

    def is_alive(self) -> bool:
        return self._is_alive and self.process is not None and self.process.is_alive()

    def _ensure_worker_running(self) -> None:
        if self.process is None or not self.process.is_alive():
            self.kill()
            self.parent_conn, self.child_conn = multiprocessing.Pipe(duplex=True)
            self.process = multiprocessing.Process(
                target=_repl_worker_loop,
                args=(self.child_conn, self.max_output_length),
                daemon=True
            )
            self.process.start()
            self._is_alive = True

    def execute(self, code: str, timeout: float = 10.0) -> Dict[str, Any]:
        self._ensure_worker_running()
        start_time = time.perf_counter()

        try:
            self.parent_conn.send({"command": "EXEC", "code": code})
        except Exception as e:
            self.kill()
            raise e

        # Poll with wall-clock timeout
        if self.parent_conn.poll(timeout):
            try:
                response = self.parent_conn.recv()
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                if "execution_time_ms" not in response or response["execution_time_ms"] == 0.0:
                    response["execution_time_ms"] = elapsed_ms
                if not self.stateful:
                    self.kill()
                return response
            except Exception as e:
                self.kill()
                raise e
        else:
            # Wall-clock timeout exceeded: kill process immediately
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.kill()

            diag = DiagnosticError(
                error_type="TimeoutError",
                message=f"Execution timed out after {timeout:.2f} seconds.",
                suggestion=f"Execution exceeded {timeout:.1f}s time limit. Vectorize operations using NumPy/Pandas and avoid non-terminating loops.",
                raw_traceback=f"TimeoutError: Execution exceeded wall-clock limit of {timeout}s"
            )

            return {
                "status": "error",
                "execution_time_ms": elapsed_ms,
                "data": None,
                "stdout": "",
                "stderr": "",
                "error": diag.model_dump(),
            }

    def reset(self) -> None:
        """Resets the state of the active worker."""
        if self.is_alive():
            try:
                self.parent_conn.send({"command": "RESET"})
                if self.parent_conn.poll(2.0):
                    self.parent_conn.recv()
                else:
                    self.kill()
            except Exception:
                self.kill()

    def kill(self) -> None:
        """Force terminates worker process and cleans up resources."""
        if self.process is not None:
            try:
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=0.3)
                    if self.process.is_alive():
                        self.process.kill()
            except Exception:
                pass
            finally:
                self.process = None

        if self.parent_conn:
            try:
                self.parent_conn.close()
            except Exception:
                pass
            self.parent_conn = None

        if self.child_conn:
            try:
                self.child_conn.close()
            except Exception:
                pass
            self.child_conn = None

        self._is_alive = False


# ---------------------------------------------------------------------------
# REPL Session Manager
# ---------------------------------------------------------------------------

class REPLSessionManager:
    """
    Manages active stateful REPL sessions mapped by session_id.
    Handles session creation, routing, reset, and teardown.
    """
    def __init__(self, default_timeout: float = 10.0, max_output_length: int = 50000):
        self._sessions: Dict[str, ProcessWorkerSandbox] = {}
        self.default_timeout = default_timeout
        self.max_output_length = max_output_length

    def get_or_create_session(self, session_id: str) -> ProcessWorkerSandbox:
        if session_id not in self._sessions or not self._sessions[session_id].is_alive():
            self._sessions[session_id] = ProcessWorkerSandbox(
                stateful=True, max_output_length=self.max_output_length
            )
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            self._sessions[session_id].reset()
            return True
        return False

    def close_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            sandbox = self._sessions.pop(session_id)
            sandbox.kill()
            return True
        return False

    def close_all(self) -> None:
        for session_id in list(self._sessions.keys()):
            self.close_session(session_id)


# ---------------------------------------------------------------------------
# PythonREPL Tool Class
# ---------------------------------------------------------------------------

class PythonREPL:
    """
    Main tool interface for sandboxed Python code execution.
    Conforms to OpenAI / NeMo function-calling protocol.
    """
    def __init__(
        self,
        default_timeout: float = 10.0,
        max_output_length: int = 50000,
        enable_ast_security: bool = True
    ):
        self.default_timeout = default_timeout
        self.max_output_length = max_output_length
        self.enable_ast_security = enable_ast_security
        self.validator = CodeSecurityValidator()
        self.session_manager = REPLSessionManager(
            default_timeout=default_timeout,
            max_output_length=max_output_length
        )

    def execute(
        self,
        code: str,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> ToolResult:
        """
        Executes code snippet in either a stateful session or ephemeral stateless sandbox.
        
        Args:
            code: Python code string to execute.
            session_id: Optional session identifier for persistent stateful execution.
            timeout: Optional wall-clock timeout in seconds (defaults to self.default_timeout).
            
        Returns:
            ToolResult containing status, execution_time_ms, data, stdout, stderr, and optional DiagnosticError.
        """
        start_perf = time.perf_counter()
        effective_timeout = timeout if timeout is not None else self.default_timeout

        # Phase 1: AST Security Validation
        if self.enable_ast_security:
            try:
                self.validator.validate(code)
            except SyntaxError as e:
                elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
                diag = DiagnosticClassifier.create_diagnostic_error(exc=e, source_code=code)
                return ToolResult(
                    status="error",
                    execution_time_ms=elapsed_ms,
                    data=None,
                    stdout="",
                    stderr="",
                    error=diag
                )
            except SecurityViolationError as e:
                elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
                diag = DiagnosticClassifier.create_diagnostic_error(
                    exc=e,
                    source_code=code,
                    context={"lineno": e.line_number, "col_offset": e.column_offset, "token": str(e)}
                )
                return ToolResult(
                    status="error",
                    execution_time_ms=elapsed_ms,
                    data=None,
                    stdout="",
                    stderr="",
                    error=diag
                )

        # Phase 2: Process Sandbox Execution
        if session_id:
            sandbox = self.session_manager.get_or_create_session(session_id)
        else:
            sandbox = ProcessWorkerSandbox(stateful=False, max_output_length=self.max_output_length)

        try:
            resp = sandbox.execute(code, timeout=effective_timeout)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            diag = DiagnosticClassifier.create_diagnostic_error(exc=e, source_code=code)
            return ToolResult(
                status="error",
                execution_time_ms=elapsed_ms,
                data=None,
                stdout="",
                stderr="",
                error=diag
            )

        err_obj = None
        if resp.get("error"):
            err_obj = DiagnosticError.model_validate(resp["error"])

        return ToolResult(
            status=resp.get("status", "error"),
            execution_time_ms=resp.get("execution_time_ms", (time.perf_counter() - start_perf) * 1000.0),
            data=resp.get("data"),
            stdout=resp.get("stdout", ""),
            stderr=resp.get("stderr", ""),
            error=err_obj
        )

    def reset_session(self, session_id: str) -> bool:
        """Resets the state of a persistent session."""
        return self.session_manager.reset_session(session_id)

    def close_session(self, session_id: str) -> bool:
        """Closes and tears down a persistent session worker process."""
        return self.session_manager.close_session(session_id)

    def close(self) -> None:
        """Shuts down all active session workers."""
        self.session_manager.close_all()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
