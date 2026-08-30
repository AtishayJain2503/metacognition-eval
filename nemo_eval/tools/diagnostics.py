"""
nemo_eval.tools.diagnostics
---------------------------
Deterministic error classification, visual syntax caret highlighting, 
traceback sanitization, and actionable remediation suggestions for agent self-correction.
"""

import ast
import difflib
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from nemo_eval.tools.schemas import DiagnosticError


class DiagnosticClassifier:
    """Classifies exceptions and builds rich, actionable DiagnosticError envelopes."""

    SAFE_BUILTINS_NAMES = {
        "int", "float", "str", "bool", "list", "dict", "set", "tuple", "frozenset",
        "len", "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
        "all", "any", "abs", "min", "max", "sum", "round", "print", "repr",
        "isinstance", "issubclass", "math", "pd", "np", "df", "sqrt", "pow"
    }

    @staticmethod
    def classify_error_type(exc: BaseException) -> str:
        """Map an exception instance to a canonical error_type string."""
        if isinstance(exc, SyntaxError):
            return "SyntaxError"
        if isinstance(exc, NameError):
            return "NameError"
        if isinstance(exc, KeyError):
            return "KeyError"
        if isinstance(exc, IndexError):
            return "IndexError"
        if isinstance(exc, TimeoutError):
            return "TimeoutError"
        if isinstance(exc, TypeError):
            return "TypeError"
        if isinstance(exc, ValueError):
            return "ValueError"
        if isinstance(exc, ZeroDivisionError):
            return "ZeroDivisionError"
        if isinstance(exc, FileNotFoundError):
            return "FileNotFoundError"

        exc_type_name = type(exc).__name__
        if "SecurityViolation" in exc_type_name:
            return "SecurityViolation"
        if "OperationalError" in exc_type_name or "DatabaseError" in exc_type_name:
            return "OperationalError"
        if "Timeout" in exc_type_name:
            return "TimeoutError"

        return exc_type_name or "Exception"

    @staticmethod
    def format_visual_pointer(line_text: str, column_offset: int, token_len: int = 1) -> str:
        """Generate a caret pointer string matching the visual position in line_text."""
        safe_col = max(0, column_offset - 1)
        safe_len = max(1, token_len)
        return " " * safe_col + "^" * safe_len

    @classmethod
    def highlight_python_syntax_error(
        cls, exc: SyntaxError, source_code: Optional[str] = None
    ) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
        """Extract line, column, code snippet and caret pointer for Python syntax errors."""
        lineno = exc.lineno
        offset = exc.offset or 1
        line_text = exc.text

        if line_text is None and source_code and lineno:
            lines = source_code.splitlines()
            if 0 < lineno <= len(lines):
                line_text = lines[lineno - 1]

        if line_text:
            clean_line = line_text.rstrip("\r\n")
            pointer = cls.format_visual_pointer(clean_line, offset, token_len=1)
            return lineno, offset, clean_line, pointer

        return lineno, offset, None, None

    @classmethod
    def highlight_sql_error(
        cls, query: str, err_msg: str
    ) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
        """Locate offending SQL token and generate caret pointer snippet."""
        if not query:
            return None, None, None, None

        near_match = re.search(r'near ["\x27]([^"\x27]+)["\x27]: syntax error', err_msg, re.IGNORECASE)
        tbl_match = re.search(r'no such table:\s*([^\s,;]+)', err_msg, re.IGNORECASE)
        col_match = re.search(r'no such column:\s*([^\s,;]+)', err_msg, re.IGNORECASE)

        token = None
        if near_match:
            token = near_match.group(1)
        elif tbl_match:
            token = tbl_match.group(1)
        elif col_match:
            token = col_match.group(1)

        if token:
            lines = query.splitlines()
            for idx, line in enumerate(lines):
                pos = line.find(token)
                if pos != -1:
                    lineno = idx + 1
                    col_offset = pos + 1
                    pointer = cls.format_visual_pointer(line, col_offset, token_len=len(token))
                    return lineno, col_offset, line, pointer

        first_line = query.splitlines()[0] if query.splitlines() else query
        return 1, 1, first_line, "^"

    @classmethod
    def suggest_name_remediation(
        cls, undefined_name: str, session_vars: Optional[List[str]] = None
    ) -> str:
        """Generate fuzzy remediation suggestion for undefined variables/functions."""
        candidates = set(cls.SAFE_BUILTINS_NAMES)
        if session_vars:
            candidates.update(session_vars)

        matches = difflib.get_close_matches(undefined_name, list(candidates), n=1, cutoff=0.5)
        if matches:
            return f"Variable or function '{undefined_name}' is not defined. Did you mean '{matches[0]}'?"
        return (
            f"Variable or function '{undefined_name}' is not defined. "
            "Ensure it is defined in the current session or import required safe modules."
        )

    @classmethod
    def suggest_key_remediation(
        cls, missing_key: str, available_keys: Optional[List[str]] = None
    ) -> str:
        """Generate fuzzy remediation suggestion for missing dict keys or DataFrame columns."""
        if available_keys:
            matches = difflib.get_close_matches(missing_key, available_keys, n=1, cutoff=0.5)
            keys_preview = str(available_keys[:8]) + ("..." if len(available_keys) > 8 else "")
            if matches:
                return (
                    f"Key/Column '{missing_key}' not found. Did you mean '{matches[0]}'? "
                    f"Available keys/columns: {keys_preview}."
                )
            return f"Key/Column '{missing_key}' not found. Available keys/columns: {keys_preview}."
        return f"Verify dictionary or DataFrame column '{missing_key}' exists using .columns or .keys()."

    @classmethod
    def suggest_sql_table_remediation(
        cls, missing_table: str, available_tables: Optional[List[str]] = None
    ) -> str:
        """Generate fuzzy remediation suggestion for missing SQLite tables."""
        if available_tables:
            matches = difflib.get_close_matches(missing_table, available_tables, n=1, cutoff=0.5)
            tables_preview = ", ".join(f"'{t}'" for t in available_tables)
            if matches:
                return (
                    f"Table '{missing_table}' does not exist. Did you mean '{matches[0]}'? "
                    f"Available tables: [{tables_preview}]. Call 'sqlite_schema' to inspect structure."
                )
            return f"Table '{missing_table}' does not exist. Available tables: [{tables_preview}]. Call 'sqlite_schema' to inspect structure."
        return f"Table '{missing_table}' does not exist. Call 'sqlite_schema' to retrieve active database tables."

    @classmethod
    def suggest_sql_column_remediation(
        cls, missing_column: str, available_columns: Optional[List[str]] = None, table_name: Optional[str] = None
    ) -> str:
        """Generate fuzzy remediation suggestion for missing SQLite columns."""
        prefix = f" in table '{table_name}'" if table_name else ""
        if available_columns:
            matches = difflib.get_close_matches(missing_column, available_columns, n=1, cutoff=0.5)
            cols_preview = ", ".join(f"'{c}'" for c in available_columns[:10])
            if matches:
                return (
                    f"Column '{missing_column}' does not exist{prefix}. Did you mean '{matches[0]}'? "
                    f"Available columns: [{cols_preview}]."
                )
            return f"Column '{missing_column}' does not exist{prefix}. Available columns: [{cols_preview}]."
        return f"Column '{missing_column}' does not exist{prefix}. Inspect schema using 'sqlite_schema'."

    @classmethod
    def sanitize_traceback(cls, exc: BaseException, limit_frames: int = 4) -> str:
        """Filter internal framework stack frames and return a clean, concise traceback."""
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        cleaned = []
        for line in tb_lines:
            if any(internal in line for internal in ["multiprocessing\\", "multiprocessing/", "unittest\\", "_pytest"]):
                continue
            cleaned.append(line)
        return "".join(cleaned[-limit_frames:]) if cleaned else "".join(tb_lines[-limit_frames:])

    @classmethod
    def create_diagnostic_error(
        cls,
        exc: BaseException,
        source_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticError:
        """Main entry point to transform any exception into a structured DiagnosticError."""
        ctx = context or {}
        error_type = cls.classify_error_type(exc)
        message = str(exc).strip()
        line_number = None
        column_offset = None
        code_snippet = None
        pointer = None
        suggestion = ""
        raw_tb = cls.sanitize_traceback(exc)

        if isinstance(exc, SyntaxError):
            line_number, column_offset, code_snippet, pointer = cls.highlight_python_syntax_error(
                exc, source_code
            )
            suggestion = (
                f"Syntax error on line {line_number}. Check for unclosed brackets, missing colons, "
                "or invalid Python syntax near the indicated pointer."
            )

        elif isinstance(exc, NameError):
            match = re.search(r"name '([^']+)' is not defined", message)
            undefined_name = match.group(1) if match else ""
            session_vars = ctx.get("session_vars", [])
            suggestion = cls.suggest_name_remediation(undefined_name, session_vars)
            if source_code and undefined_name:
                lines = source_code.splitlines()
                for idx, line in enumerate(lines):
                    if undefined_name in line:
                        line_number = idx + 1
                        column_offset = line.find(undefined_name) + 1
                        code_snippet = line
                        pointer = cls.format_visual_pointer(line, column_offset, len(undefined_name))
                        break

        elif isinstance(exc, KeyError):
            match = re.search(r"['\"]?([^'\"]+)['\"]?", message)
            missing_key = match.group(1) if match else message
            available_keys = ctx.get("available_keys") or ctx.get("columns")
            suggestion = cls.suggest_key_remediation(missing_key, available_keys)

        elif isinstance(exc, IndexError):
            suggestion = (
                "Index out of bounds. Verify sequence length with len() or check DataFrame shape "
                "before positional indexing."
            )

        elif isinstance(exc, ZeroDivisionError):
            suggestion = "Check denominator for zero or empty collection before division."

        elif error_type == "SecurityViolation":
            line_number = ctx.get("lineno") or getattr(exc, "line_number", None)
            column_offset = ctx.get("col_offset") or getattr(exc, "column_offset", None)
            token = ctx.get("token") or message
            if source_code and line_number and 0 < line_number <= len(source_code.splitlines()):
                code_snippet = source_code.splitlines()[line_number - 1]
                pointer = cls.format_visual_pointer(code_snippet, column_offset or 1, len(token))
            suggestion = (
                f"Security boundary violation for '{token}'. Prohibited module, attribute, or builtin. "
                "Use safe analytical packages (numpy, pandas, math) and avoid system/introspection calls."
            )

        elif error_type == "TimeoutError":
            timeout_s = ctx.get("timeout_seconds", 5.0)
            suggestion = (
                f"Execution timed out after {timeout_s:.1f}s. Eliminate infinite loops, reduce iteration count, "
                "or vectorize operations using Pandas / NumPy."
            )

        elif error_type == "OperationalError":
            query = ctx.get("query") or source_code or ""
            line_number, column_offset, code_snippet, pointer = cls.highlight_sql_error(query, message)
            
            tbl_match = re.search(r'no such table:\s*([^\s,;]+)', message, re.IGNORECASE)
            col_match = re.search(r'no such column:\s*([^\s,;]+)', message, re.IGNORECASE)
            
            if tbl_match:
                missing_tbl = tbl_match.group(1)
                available_tables = ctx.get("available_tables", [])
                suggestion = cls.suggest_sql_table_remediation(missing_tbl, available_tables)
            elif col_match:
                missing_col = col_match.group(1)
                available_cols = ctx.get("available_columns", [])
                table_name = ctx.get("table_name")
                suggestion = cls.suggest_sql_column_remediation(missing_col, available_cols, table_name)
            elif "readonly" in message.lower() or "read-only" in message.lower():
                suggestion = "Database is operating in strict read-only mode (PRAGMA query_only = ON). Data modification statements are prohibited."
            elif "not authorized" in message.lower() or "unauthorized" in message.lower():
                suggestion = "Database security policy denied this operation. Modifying query_only or administrative PRAGMAs is prohibited."
            elif "interrupted" in message.lower():
                suggestion = "SQL execution aborted by progress handler opcode timeout. Optimize recursive CTEs or reduce join complexity."
            else:
                suggestion = "Check SQL syntax, table/column names, and read-only transaction state."

        elif isinstance(exc, FileNotFoundError):
            suggestion = f"File not found: {message}. Verify the path is correct relative to the workspace."

        else:
            suggestion = f"{error_type} encountered: {message}. Review input parameters and logic."

        return DiagnosticError(
            error_type=error_type,
            message=message,
            line_number=line_number,
            column_offset=column_offset,
            code_snippet=code_snippet,
            pointer=pointer,
            suggestion=suggestion,
            raw_traceback=raw_tb,
        )

    @classmethod
    def format_error(
        cls,
        exc: BaseException,
        source_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticError:
        """Alias for create_diagnostic_error."""
        return cls.create_diagnostic_error(exc, source_code, context)


DiagnosticFormatter = DiagnosticClassifier
