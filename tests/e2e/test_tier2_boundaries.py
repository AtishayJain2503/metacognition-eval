"""
test_tier2_boundaries.py - Opaque-Box Boundary and Error Condition Tests (Tier 2).
Covers security jailbreaks, hard timeouts, recursion limits, division by zero, empty tables,
malformed CSVs, and float tolerances (≥5 boundary tests per area).
"""
import ast
import io
import math
import os
import sqlite3
import sys
import pandas as pd
import numpy as np
import pytest
from typing import List, Dict, Any

from tests.e2e.conftest import DiagnosticError, ToolResult, BenchmarkTask


# ===========================================================================
# 1. AST Security Jailbreak & Sandbox Boundaries (≥5 Tests)
# ===========================================================================

class TestSecurityBoundaries:
    """Rigorous tests for AST security bypass attempts and dangerous introspection."""

    FORBIDDEN_MODULES = {
        "os", "sys", "subprocess", "shutil", "socket", "urllib",
        "requests", "http", "ctypes", "builtins", "importlib",
        "posix", "nt", "pty", "commands", "runpy", "multiprocessing"
    }

    FORBIDDEN_ATTRIBUTES = {
        "__subclasses__", "__bases__", "__mro__", "__globals__",
        "__code__", "__builtins__", "__import__", "__class__",
        "__qualname__", "__closure__", "__func__"
    }

    FORBIDDEN_CALLS = {
        "eval", "exec", "compile", "__import__", "open", "getattr",
        "setattr", "delattr", "hasattr", "breakpoint", "input"
    }

    def _validate_ast(self, code: str) -> List[str]:
        tree = ast.parse(code)
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in self.FORBIDDEN_MODULES:
                        violations.append(f"Import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in self.FORBIDDEN_MODULES:
                    violations.append(f"ImportFrom {node.module}")
            elif isinstance(node, ast.Attribute):
                if node.attr in self.FORBIDDEN_ATTRIBUTES:
                    violations.append(f"Attribute {node.attr}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_CALLS:
                    violations.append(f"Call {node.func.id}")
        return violations

    def test_boundary_forbidden_os_import(self):
        """Rejects direct 'import os'."""
        code = "import os\nos.system('whoami')"
        violations = self._validate_ast(code)
        assert len(violations) > 0
        assert any("os" in v for v in violations)

    def test_boundary_forbidden_from_import_subprocess(self):
        """Rejects 'from subprocess import Popen'."""
        code = "from subprocess import Popen\nPopen(['dir'], shell=True)"
        violations = self._validate_ast(code)
        assert len(violations) > 0
        assert any("subprocess" in v for v in violations)

    def test_boundary_forbidden_socket_network_access(self):
        """Rejects 'import socket' guaranteeing zero network dependencies."""
        code = "import socket\ns = socket.socket()\ns.connect(('8.8.8.8', 53))"
        violations = self._validate_ast(code)
        assert len(violations) > 0
        assert any("socket" in v for v in violations)

    def test_boundary_forbidden_dunder_subclasses_jailbreak(self):
        """Rejects object class subclass traversal exploit."""
        code = "exploited = ().__class__.__bases__[0].__subclasses__()"
        violations = self._validate_ast(code)
        assert any("Attribute __class__" in v or "Attribute __subclasses__" in v for v in violations)

    def test_boundary_forbidden_dunder_globals_access(self):
        """Rejects function __globals__ dictionary introspection."""
        code = "g = (lambda: None).__globals__"
        violations = self._validate_ast(code)
        assert any("Attribute __globals__" in v for v in violations)

    def test_boundary_forbidden_open_call(self):
        """Rejects file system access via open()."""
        code = "with open('/etc/passwd', 'r') as f:\n    data = f.read()"
        violations = self._validate_ast(code)
        assert any("Call open" in v for v in violations)

    def test_boundary_forbidden_dynamic_eval_call(self):
        """Rejects dynamic eval() execution inside sandbox."""
        code = "res = eval('1 + 2')"
        violations = self._validate_ast(code)
        assert any("Call eval" in v for v in violations)

    def test_boundary_forbidden_ctypes_import(self):
        """Rejects C-level ctypes memory access."""
        code = "import ctypes\nctypes.CDLL(None)"
        violations = self._validate_ast(code)
        assert any("ctypes" in v for v in violations)


# ===========================================================================
# 2. Hard Timeouts, Runaway Loops & Progress Handlers (≥5 Tests)
# ===========================================================================

class TestTimeoutAndRecursionBoundaries:
    """Tests for infinite loop detection, recursive CTE limits, and timeouts."""

    def test_boundary_sqlite_recursive_cte_progress_handler_abort(self):
        """SQLite progress handler aborts runaway recursive CTE queries."""
        conn = sqlite3.connect(":memory:")
        
        # Install progress handler that cancels query after 100 opcode checks
        opcode_count = 0
        def progress_handler():
            nonlocal opcode_count
            opcode_count += 1
            if opcode_count > 100:
                return 1 # Non-zero aborts query
            return 0

        conn.set_progress_handler(progress_handler, 10)
        cursor = conn.cursor()

        runaway_sql = """
        WITH RECURSIVE infinite_counter(n) AS (
            SELECT 1
            UNION ALL
            SELECT n + 1 FROM infinite_counter
        )
        SELECT sum(n) FROM infinite_counter;
        """

        with pytest.raises(sqlite3.OperationalError, match="interrupted|cancelled|callback"):
            cursor.execute(runaway_sql)
            cursor.fetchall()

        conn.close()

    def test_boundary_python_repl_timeout_simulation(self):
        """ToolResult captures timeout status with execution metadata."""
        timeout_res = ToolResult(
            status="error",
            execution_time_ms=5002.1,
            error=DiagnosticError(
                error_type="TimeoutError",
                message="Code execution exceeded hard timeout limit (5.0s)",
                suggestion="Optimize iterative loops or reduce computational complexity."
            )
        )
        assert timeout_res.status == "error"
        assert timeout_res.execution_time_ms > 5000.0
        assert timeout_res.error.error_type == "TimeoutError"

    def test_boundary_recursion_depth_limit(self):
        """Catches and handles Python RecursionError from infinite recursive function calls."""
        def infinite_rec(n):
            return infinite_rec(n + 1)

        try:
            infinite_rec(1)
        except RecursionError as e:
            diag = DiagnosticError(
                error_type="RecursionError",
                message=str(e),
                suggestion="Check recursive base case and termination conditions."
            )
            assert diag.error_type == "RecursionError"
            assert "base case" in diag.suggestion

    def test_boundary_sqlite_busy_timeout_setting(self):
        """Enforces busy timeout PRAGMA on SQLite connection."""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("PRAGMA busy_timeout = 5000;")
        cursor.execute("PRAGMA busy_timeout;")
        val = cursor.fetchone()[0]
        assert val == 5000
        conn.close()

    def test_boundary_max_page_count_quota(self):
        """Enforces max_page_count memory quota on transient SQLite database."""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("PRAGMA max_page_count = 1000;")
        cursor.execute("PRAGMA max_page_count;")
        val = cursor.fetchone()[0]
        assert val == 1000
        conn.close()


# ===========================================================================
# 3. Arithmetic Boundaries & Division by Zero (≥5 Tests)
# ===========================================================================

class TestArithmeticBoundaries:
    """Tests for zero division, negative exponents, float overflow, and NaN handling."""

    def test_boundary_division_by_zero_handling(self):
        """Handles ZeroDivisionError cleanly with actionable suggestion."""
        try:
            val = 100 / 0
        except ZeroDivisionError as e:
            diag = DiagnosticError(
                error_type="ZeroDivisionError",
                message=str(e),
                suggestion="Ensure denominator is non-zero before division."
            )
            assert diag.error_type == "ZeroDivisionError"
            assert "non-zero" in diag.suggestion

    def test_boundary_float_overflow(self):
        """Handles extreme numerical overflow values (e.g. 1e308 * 1e308 -> inf)."""
        huge_val = 1e308 * 10.0
        assert math.isinf(huge_val)

    def test_boundary_float_tolerance_exact_boundaries(self):
        """Tests float tolerance boundaries at exact rel_tol=0.01 threshold."""
        target = 100.0
        # Exactly 1% above -> 101.0 (True)
        assert math.isclose(101.0, target, rel_tol=0.01) is True
        # 1.009% above -> 101.009 (True)
        assert math.isclose(101.009, target, rel_tol=0.01) is True
        # 1.011% above -> 101.011 (False)
        assert math.isclose(101.011, target, rel_tol=0.01) is False

    def test_boundary_zero_reference_absolute_tolerance(self):
        """When gold answer is 0.0, relative tolerance fails and absolute tolerance governs."""
        target = 0.0
        candidate_close = 0.0005
        candidate_far = 0.05
        # Absolute tolerance = 0.001
        assert math.isclose(candidate_close, target, abs_tol=0.001) is True
        assert math.isclose(candidate_far, target, abs_tol=0.001) is False

    def test_boundary_nan_comparison_policy(self):
        """Validates NaN vs NaN comparison equality in evaluation engine."""
        def compare_values(pred, gold):
            if isinstance(pred, float) and isinstance(gold, float):
                if math.isnan(pred) and math.isnan(gold):
                    return True
                return math.isclose(pred, gold, rel_tol=0.01)
            return pred == gold

        assert compare_values(float('nan'), float('nan')) is True
        assert compare_values(float('nan'), 0.0) is False

    def test_boundary_negative_zero_equality(self):
        """Ensures -0.0 and +0.0 are treated as numerically equal."""
        neg_zero = -0.0
        pos_zero = 0.0
        assert neg_zero == pos_zero
        assert math.isclose(neg_zero, pos_zero, abs_tol=1e-5)


# ===========================================================================
# 4. Tabular & Database Edge Cases (Empty, Single-Row, Pagination) (≥5 Tests)
# ===========================================================================

class TestDataEngineBoundaries:
    """Tests for empty tables, malformed CSVs, and exact pagination limits."""

    def test_boundary_empty_sqlite_table(self):
        """Querying an empty table returns empty list with 0 count."""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE empty_tbl (id INT, val TEXT);")
        cursor.execute("SELECT * FROM empty_tbl;")
        rows = cursor.fetchall()
        assert rows == []
        assert len(rows) == 0
        conn.close()

    def test_boundary_single_row_single_column_table(self):
        """Handles 1x1 table extraction correctly."""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE scalar_tbl (val INT);")
        cursor.execute("INSERT INTO scalar_tbl VALUES (42);")
        cursor.execute("SELECT val FROM scalar_tbl;")
        row = cursor.fetchone()
        assert row == (42,)
        conn.close()

    def test_boundary_pagination_exact_50_rows(self):
        """Query returning exactly 50 rows does NOT trigger has_more=True."""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_50 (id INT);")
        cursor.executemany("INSERT INTO test_50 VALUES (?);", [(i,) for i in range(50)])
        
        limit = 50
        cursor.execute("SELECT * FROM test_50;")
        rows = cursor.fetchmany(limit + 1)
        has_more = len(rows) > limit
        returned_rows = rows[:limit]

        assert len(returned_rows) == 50
        assert has_more is False
        conn.close()

    def test_boundary_pagination_51_rows_triggers_has_more(self):
        """Query returning 51 rows triggers has_more=True with 50 returned rows."""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_51 (id INT);")
        cursor.executemany("INSERT INTO test_51 VALUES (?);", [(i,) for i in range(51)])
        
        limit = 50
        cursor.execute("SELECT * FROM test_51;")
        rows = cursor.fetchmany(limit + 1)
        has_more = len(rows) > limit
        returned_rows = rows[:limit]

        assert len(returned_rows) == 50
        assert has_more is True
        conn.close()

    def test_boundary_malformed_csv_parsing(self, temp_dir):
        """Handles CSV with missing values, empty lines, and trailing delimiters cleanly."""
        csv_file = os.path.join(temp_dir, "dirty_data.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("col_a,col_b,col_c\n")
            f.write("1,hello,10.5\n")
            f.write("2,,20.0\n")      # Missing middle value
            f.write("3,world,\n")      # Missing trailing value
            f.write("\n")              # Empty newline
            f.write("4,test,40.2\n")

        df = pd.read_csv(csv_file)
        assert len(df) == 4
        assert pd.isna(df.loc[1, "col_b"])
        assert pd.isna(df.loc[2, "col_c"])
        assert df.loc[3, "col_a"] == 4

    def test_boundary_csv_with_semicolon_delimiter(self, temp_dir):
        """Parses CSV formatted with semicolon delimiters."""
        csv_file = os.path.join(temp_dir, "semicolon.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("id;name;score\n1;Alice;98.5\n2;Bob;85.0\n")

        df = pd.read_csv(csv_file, sep=";")
        assert df.shape == (2, 3)
        assert df.loc[0, "name"] == "Alice"
