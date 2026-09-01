"""
tests.unit.test_tools.test_adversarial_m1
-----------------------------------------
Adversarial stress-testing suite for Milestone 1: Hermetic Tool Sandboxes & Auxiliary Engines.
Tests empirical robustness against:
- Sandbox escapes, dunder traversal, forbidden imports/calls, and builtins pollution
- Process timeout termination, recursion bombs, allocation pressure, and session recovery
- SQLite runaway recursive CTEs, Cartesian products, mutation attempts, and PRAGMA escapes
- Tabular malformed files, extreme numbers (inf/NaN/overflow), mixed types, and SQL injection in column names
- Rich diagnostic error classification, visual caret pointers, and fuzzy remediation hints
"""

import io
import math
import os
import sqlite3
import tempfile
import time
import pytest
import pandas as pd
import numpy as np

from nemo_eval.tools.repl import PythonREPL, CodeSecurityValidator, SecurityViolationError, SAFE_BUILTINS
from nemo_eval.tools.sqlite_engine import SQLiteEngine, SQLiteEngineConfig, QueryResult
from nemo_eval.tools.tabular import TabularEngine, TabularSchemaInfo, TabularSummaryInfo, TabularSampleInfo
from nemo_eval.tools.diagnostics import DiagnosticClassifier
from nemo_eval.tools.schemas import ToolResult, DiagnosticError


# ===========================================================================
# 1. ADVERSARIAL PYTHON REPL TESTS
# ===========================================================================

class TestAdversarialREPL:
    """Stress-tests AST security boundaries, execution sandbox, and timeout/resource controls."""

    @pytest.fixture(scope="function")
    def repl(self):
        r = PythonREPL(default_timeout=5.0, max_output_length=50000)
        yield r
        r.close()

    @pytest.mark.parametrize("attack_code,attack_desc", [
        ("().__class__.__bases__[0].__subclasses__()", "Subclasses jailbreak"),
        ("(lambda: None).__globals__", "Function globals access"),
        ("(lambda: None).__code__", "Function code object access"),
        ("int.__builtins__", "Builtins attribute access"),
        ("str.__bases__", "Type bases access"),
        ("dict.__mro__", "MRO hierarchy access"),
        ("(lambda x: lambda: x)(1).__closure__", "Closure cell access"),
        ("def gen(): yield 1\ng = gen()\ng.gi_frame", "Generator frame access"),
        ("int.__dict__", "Type dictionary access"),
        ("int.__getattribute__", "Getattribute access"),
        ("object.__init_subclass__", "Init subclass hook"),
        ("int.__reduce__", "Pickle reduce method"),
        ("int.__reduce_ex__", "Pickle reduce_ex method"),
        ("def f(): pass\nf.__qualname__", "Qualname inspection"),
        ("def f(): pass\nf.__module__", "Module attribute inspection"),
    ])
    def test_dunder_introspection_attacks(self, repl, attack_code, attack_desc):
        """Empirically assert that all dunder traversal attempts are intercepted by AST security."""
        res = repl.execute(attack_code)
        assert res.status == "error", f"Attack succeeded unexpectedly: {attack_desc}"
        assert res.error is not None
        assert res.error.error_type == "SecurityViolation"
        assert "prohibited" in res.error.message.lower()

    @pytest.mark.parametrize("import_code,mod_name", [
        ("import os", "os"),
        ("import sys", "sys"),
        ("import subprocess", "subprocess"),
        ("import ctypes", "ctypes"),
        ("import socket", "socket"),
        ("import threading", "threading"),
        ("import multiprocessing", "multiprocessing"),
        ("import importlib", "importlib"),
        ("import winreg", "winreg"),
        ("import posix", "posix"),
        ("import nt", "nt"),
        ("import shutil", "shutil"),
        ("import requests", "requests"),
        ("import urllib", "urllib"),
        ("import http", "http"),
        ("import asyncio", "asyncio"),
        ("import inspect", "inspect"),
        ("import gc", "gc"),
        ("from os import path", "os.path"),
        ("from subprocess import Popen", "subprocess.Popen"),
    ])
    def test_forbidden_module_imports(self, repl, import_code, mod_name):
        """Empirically assert that forbidden module imports are blocked by AST validator."""
        res = repl.execute(import_code)
        assert res.status == "error", f"Import of {mod_name} was allowed!"
        assert res.error is not None
        assert res.error.error_type == "SecurityViolation"

    @pytest.mark.parametrize("call_code,func_name", [
        ("eval('1+1')", "eval"),
        ("exec('x=1')", "exec"),
        ("open('dummy.txt', 'w')", "open"),
        ("getattr(int, '__doc__')", "getattr"),
        ("setattr(int, 'x', 1)", "setattr"),
        ("delattr(int, 'x')", "delattr"),
        ("hasattr(int, 'x')", "hasattr"),
        ("breakpoint()", "breakpoint"),
        ("globals()", "globals"),
        ("locals()", "locals"),
        ("vars()", "vars"),
        ("dir()", "dir"),
        ("input()", "input"),
        ("exit()", "exit"),
        ("quit()", "quit"),
        ("help()", "help"),
    ])
    def test_forbidden_builtin_calls(self, repl, call_code, func_name):
        """Empirically assert that dangerous builtin calls are blocked by AST or restricted namespace."""
        res = repl.execute(call_code)
        assert res.status == "error", f"Call to {func_name} succeeded!"
        assert res.error is not None
        assert res.error.error_type in ("SecurityViolation", "NameError")

    def test_indirect_eval_and_open_evasions(self, repl):
        """Test calling eval/open indirectly via list indexing or ternary operators."""
        res1 = repl.execute("fn = eval; fn('1+1')")
        assert res1.status == "error"
        
        res2 = repl.execute("f = open; f('test.txt', 'w')")
        assert res2.status == "error"

    def test_timeout_infinite_while_loop(self, repl):
        """Empirically verify hard wall-clock timeout terminates infinite while loop."""
        start_t = time.perf_counter()
        res = repl.execute("while True: pass", timeout=1.0)
        elapsed = time.perf_counter() - start_t
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "TimeoutError"
        assert 0.7 <= elapsed <= 3.5, f"Timeout took {elapsed:.2f}s instead of ~1.0s"

    def test_timeout_infinite_generator_consumption(self, repl):
        """Empirically verify timeout terminates infinite generator expressions."""
        code = """
def infinite_seq():
    n = 0
    while True:
        yield n
        n += 1
sum(infinite_seq())
"""
        res = repl.execute(code, timeout=1.0)
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "TimeoutError"

    def test_recursion_bomb(self, repl):
        """Empirically verify infinite recursion triggers RecursionError or TimeoutError cleanly."""
        code = """
def explode():
    return explode()
explode()
"""
        res = repl.execute(code, timeout=1.5)
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type in ("RecursionError", "TimeoutError")

    def test_memory_allocation_pressure(self, repl):
        """Empirically test memory allocation handling under large arrays."""
        code = """
x = [1] * (20 * 1000 * 1000)
len(x)
"""
        res = repl.execute(code, timeout=3.0)
        assert res.status in ("success", "error")
        if res.status == "success":
            assert res.data == 20000000

    def test_output_buffer_capping_adversarial(self, repl):
        """Verify that output buffer flooding (>200,000 chars) is capped."""
        res = repl.execute("print('X' * 200000)")
        assert res.status == "success"
        assert len(res.stdout) <= 55000
        assert "[Output truncated." in res.stdout

    def test_stateful_session_persistence_and_isolation(self, repl):
        """Verify variables persist in same session but remain strictly isolated across sessions."""
        repl.execute("secret_token = 'ALPHA_OMEGA'", session_id="session_1")
        res1 = repl.execute("secret_token", session_id="session_1")
        assert res1.status == "success"
        assert res1.data == "ALPHA_OMEGA"

        res2 = repl.execute("secret_token", session_id="session_2")
        assert res2.status == "error"
        assert res2.error.error_type == "NameError"

    def test_stateful_session_recovery_after_timeout(self, repl):
        """Verify that a session recovers and accepts subsequent commands after a timeout-induced process kill."""
        res_timeout = repl.execute("while True: pass", session_id="recovery_session", timeout=1.0)
        assert res_timeout.status == "error"
        assert res_timeout.error.error_type == "TimeoutError"

        time.sleep(0.1)
        res_after = repl.execute("100 + 200", session_id="recovery_session", timeout=3.0)
        assert res_after.status == "success"
        assert res_after.data == 300


# ===========================================================================
# 2. ADVERSARIAL SQLITE ENGINE TESTS
# ===========================================================================

class TestAdversarialSQLite:
    """Stress-tests SQLite progress handler timeouts, read-only mode, and query bounding."""

    @pytest.fixture(scope="function")
    def engine(self):
        cfg = SQLiteEngineConfig(
            db_path=":memory:",
            read_only=True,
            max_rows_default=50,
            max_rows_hard_cap=200,
            timeout_seconds=1.0,
            opcode_check_interval=500
        )
        e = SQLiteEngine(cfg)
        # Seed test data
        e.init_from_sql("""
            CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL, stock INTEGER, data BLOB);
            INSERT INTO products VALUES (1, 'Widget', 19.99, 100, X'DEADBEEF');
            INSERT INTO products VALUES (2, 'Gadget', 49.99, 0, X'CAFEBABE');
            INSERT INTO products VALUES (3, 'Doohickey', 9.99, 500, NULL);
        """)
        yield e
        e.close()

    def test_runaway_recursive_cte_timeout(self, engine):
        """Empirically assert opcode progress handler aborts infinite recursive CTE within timeout."""
        query = "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt) SELECT count(*) FROM cnt;"
        t0 = time.perf_counter()
        res = engine.execute_tool(query)
        elapsed = time.perf_counter() - t0
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type in ("OperationalError", "TimeoutError")
        assert "interrupted" in res.error.message.lower() or "timeout" in res.error.message.lower()
        assert elapsed <= 2.5, f"Query ran for {elapsed:.2f}s before progress handler aborted."

    def test_massive_cartesian_cross_join_timeout(self, engine):
        """Empirically assert runaway Cartesian join is aborted by progress handler."""
        query = """
            WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM nums LIMIT 1000)
            SELECT COUNT(*) FROM nums a, nums b, nums c, nums d;
        """
        res = engine.execute_tool(query)
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type in ("OperationalError", "TimeoutError")

    @pytest.mark.parametrize("mutation_sql,desc", [
        ("INSERT INTO products (id, name, price, stock) VALUES (4, 'Evil', 0.0, 0);", "INSERT"),
        ("UPDATE products SET price = 0.0;", "UPDATE"),
        ("DELETE FROM products;", "DELETE"),
        ("DROP TABLE products;", "DROP TABLE"),
        ("CREATE TABLE evil_table (id INT);", "CREATE TABLE"),
        ("ALTER TABLE products ADD COLUMN secret TEXT;", "ALTER TABLE"),
        ("CREATE INDEX idx_products_name ON products(name);", "CREATE INDEX"),
        ("VACUUM;", "VACUUM"),
    ])
    def test_mutating_queries_blocked_in_readonly_mode(self, engine, mutation_sql, desc):
        """Empirically assert all mutation statements are blocked by read_only PRAGMA."""
        res = engine.execute_tool(mutation_sql)
        assert res.status == "error", f"{desc} succeeded in read_only mode!"
        assert res.error is not None
        assert res.error.error_type == "OperationalError"
        assert "readonly" in res.error.message.lower() or "read-only" in res.error.message.lower()

    def test_pragma_query_only_bypass_attempt(self, engine):
        """Attempt to override query_only inside execute_tool and verify strict postcondition state."""
        res1 = engine.execute_tool("PRAGMA query_only = OFF;")
        # In read_only mode, PRAGMA query_only = OFF may execute or be no-op, but subsequent writes must fail
        res2 = engine.execute_tool("INSERT INTO products VALUES (99, 'Hacked', 0, 0, NULL);")
        assert res2.status == "error"
        assert res2.error is not None
        assert res2.error.error_type == "OperationalError"
        assert "readonly" in res2.error.message.lower() or "read-only" in res2.error.message.lower()

        # Verify no data written
        res3 = engine.execute_tool("SELECT * FROM products WHERE id = 99;")
        assert res3.status == "success"
        assert res3.data["count"] == 0

        # Verify connection query_only remains 1
        res4 = engine.execute_tool("PRAGMA query_only;")
        assert res4.status == "success"
        assert res4.data["rows"][0]["query_only"] == 1

    def test_savepoint_creation_blocked_in_readonly_mode(self, engine):
        """Verify savepoint creation fails in read_only mode."""
        with pytest.raises(sqlite3.OperationalError):
            engine.create_savepoint("sp1")

    def test_binary_blob_sanitization_in_query_and_schema(self, engine):
        """Empirically verify binary BLOB data is sanitized to <BLOB len=N> representation."""
        res = engine.execute_tool("SELECT data FROM products WHERE data IS NOT NULL;")
        assert res.status == "success"
        rows = res.data["rows"]
        assert len(rows) == 2
        for r in rows:
            assert r["data"].startswith("<BLOB len=")

        schema_res = engine.schema_tool("products")
        assert schema_res.status == "success"
        sample_rows = schema_res.data["tables"]["products"]["sample_rows"]
        for sr in sample_rows:
            if sr["data"] is not None:
                assert str(sr["data"]).startswith("<BLOB len=")

    def test_query_result_bounding_and_pagination_limits(self, engine):
        """Verify strict result bounding and pagination metadata."""
        # Query 1: default bound 50
        res = engine.execute_tool("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n LIMIT 100) SELECT x FROM n;")
        assert res.status == "success"
        assert res.data["count"] == 50
        assert res.data["has_more"] is True

        # Query 2: custom bound 20
        res2 = engine.execute_tool("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n LIMIT 100) SELECT x FROM n;", max_rows=20)
        assert res2.status == "success"
        assert res2.data["count"] == 20
        assert res2.data["has_more"] is True

        # Query 3: hard cap 200 enforced when max_rows=1000
        res3 = engine.execute_tool("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n LIMIT 500) SELECT x FROM n;", max_rows=1000)
        assert res3.status == "success"
        assert res3.data["count"] == 200
        assert res3.data["has_more"] is True

    def test_schema_introspection_sql_injection_safety(self, engine):
        """Verify SQL injection attempt in table_name parameter does not execute arbitrary SQL."""
        malicious_tbl = 'products"; DROP TABLE products; --'
        res = engine.schema_tool(table_name=malicious_tbl)
        assert res.status == "success"
        # Table products should still exist
        assert "products" in engine.get_known_tables()

    def test_transient_disk_file_isolation(self, tmp_path):
        """Verify seed disk file is protected against modification via transient copy."""
        db_file = tmp_path / "golden.db"
        conn = sqlite3.connect(str(db_file))
        conn.execute("CREATE TABLE original (id INT, val TEXT);")
        conn.execute("INSERT INTO original VALUES (1, 'untouched');")
        conn.commit()
        conn.close()

        cfg = SQLiteEngineConfig(db_path=str(db_file), read_only=False)
        engine = SQLiteEngine(cfg)
        engine.execute_query("INSERT INTO original VALUES (2, 'modified_in_transient');")
        engine.close()

        # Check golden file on disk: must NOT have the new row
        conn_check = sqlite3.connect(str(db_file))
        cur = conn_check.cursor()
        cur.execute("SELECT COUNT(*) FROM original;")
        count = cur.fetchone()[0]
        conn_check.close()
        assert count == 1, "Golden database file on disk was modified directly!"


# ===========================================================================
# 3. ADVERSARIAL TABULAR ENGINE TESTS
# ===========================================================================

class TestAdversarialTabular:
    """Stress-tests tabular parsing of malformed files, extreme floats, and column profiling."""

    def test_nonexistent_file_raises_filenotfound(self):
        """Verify clean diagnostic error on non-existent file path."""
        res = TabularEngine.inspect_tool("non_existent_file_12345.csv", action="schema")
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "FileNotFoundError"

    def test_empty_csv_file(self, tmp_path):
        """Verify error handling on empty CSV file (0 bytes)."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("", encoding="utf-8")
        res = TabularEngine.inspect_tool(str(empty_csv), action="schema")
        assert res.status == "error"

    def test_mixed_delimiters_and_encodings(self, tmp_path):
        """Verify automatic delimiter detection across semicolon, tab, pipe, and comma."""
        delims = [
            ("semicolon.csv", "col1;col2;col3\n1;2;3\n4;5;6\n"),
            ("tab.tsv", "col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6\n"),
            ("pipe.txt", "col1|col2|col3\n1|2|3\n4|5|6\n"),
            ("comma.csv", "col1,col2,col3\n1,2,3\n4,5,6\n"),
        ]
        for fname, content in delims:
            fpath = tmp_path / fname
            fpath.write_text(content, encoding="utf-8")
            schema = TabularEngine.inspect_schema(str(fpath))
            assert schema.shape["rows"] == 2
            assert schema.shape["columns"] == 3

    def test_latin1_encoding_with_accents(self, tmp_path):
        """Verify encoding fallback correctly decodes Latin-1 characters."""
        fpath = tmp_path / "latin1_test.csv"
        content = "name,city\nFrançois,Montréal\nJürgen,München\n"
        with open(str(fpath), "wb") as f:
            f.write(content.encode("latin1"))

        schema = TabularEngine.inspect_schema(str(fpath))
        assert schema.shape["rows"] == 2
        sample = TabularEngine.get_sample(str(fpath), n_rows=2)
        assert sample.records[0]["name"] == "François"
        assert sample.records[1]["city"] == "München"

    def test_utf8_bom_encoding(self, tmp_path):
        """Verify UTF-8 with BOM correctly loads without header corruption."""
        fpath = tmp_path / "bom_test.csv"
        content = "id,val\n1,100\n2,200\n"
        with open(str(fpath), "wb") as f:
            f.write(content.encode("utf-8-sig"))

        schema = TabularEngine.inspect_schema(str(fpath))
        assert schema.columns[0]["name"] == "id"

    def test_extreme_floating_point_numbers(self, tmp_path):
        """Empirically test handling of inf, -inf, NaN, extreme magnitude floats."""
        fpath = tmp_path / "extreme_floats.csv"
        csv_text = """metric,val
inf_pos,inf
inf_neg,-inf
nan_val,nan
large_pos,1e308
tiny_pos,1e-300
neg_zero,-0.0
standard,42.5
"""
        fpath.write_text(csv_text, encoding="utf-8")
        summary = TabularEngine.profile_summary(str(fpath))
        assert "val" in summary.column_summaries
        
        sample = TabularEngine.get_sample(str(fpath), n_rows=10)
        assert sample.n_rows_returned == 7
        # Verify JSON serialization does not crash on NaN/inf in records
        assert sample.records[0]["val"] is None or isinstance(sample.records[0]["val"], (float, str))

    def test_all_null_and_mixed_type_column_profiling(self, tmp_path):
        """Test summary profiler on columns with 100% nulls and mixed string/number types."""
        fpath = tmp_path / "edge_types.csv"
        csv_text = """all_null,all_bool,mixed_col
,True,123
,False,hello
,True,45.67
,False,
"""
        fpath.write_text(csv_text, encoding="utf-8")
        summary = TabularEngine.profile_summary(str(fpath))
        assert summary.column_summaries["all_null"].null_percentage == 100.0
        assert summary.column_summaries["all_bool"].unique_count == 2
        assert summary.column_summaries["mixed_col"].non_null_count == 3

    def test_load_to_sqlite_with_malicious_column_names(self, tmp_path):
        """Verify column sanitization when bridging tabular data to SQLite tables."""
        fpath = tmp_path / "injection_columns.csv"
        csv_text = '"id"; DROP TABLE products; --","col with spaces","valid_col"\n1,2,3\n4,5,6\n'
        fpath.write_text(csv_text, encoding="utf-8")
        
        conn = sqlite3.connect(":memory:")
        rows_loaded = TabularEngine.load_to_sqlite(str(fpath), "imported_data", conn)
        assert rows_loaded == 2
        
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM imported_data;")
        assert cur.fetchone()[0] == 2
        conn.close()

    def test_get_sample_boundary_clamping(self, tmp_path):
        """Verify n_rows bounds clamping in get_sample."""
        fpath = tmp_path / "sample_test.csv"
        fpath.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
        
        # Test negative n_rows clamped to 1
        s1 = TabularEngine.get_sample(str(fpath), n_rows=-5)
        assert s1.n_rows_returned == 1

        # Test n_rows exceeding table length
        s2 = TabularEngine.get_sample(str(fpath), n_rows=1000)
        assert s2.n_rows_returned == 3


# ===========================================================================
# 4. ADVERSARIAL ERROR DIAGNOSTICS TESTS
# ===========================================================================

class TestAdversarialDiagnostics:
    """Stress-tests visual caret pointer precision and fuzzy suggestion algorithms."""

    def test_python_syntax_error_caret_alignment(self):
        """Verify exact column pointer alignment for syntax errors."""
        bad_code = "x = 10 +\ny = 20"
        try:
            compile(bad_code, "<string>", "exec")
        except SyntaxError as e:
            diag = DiagnosticClassifier.create_diagnostic_error(e, source_code=bad_code)
            assert diag.error_type == "SyntaxError"
            assert diag.line_number == 1
            assert diag.pointer is not None
            assert "^" in diag.pointer

    def test_sql_fuzzy_table_remediation(self):
        """Verify table typos are matched against known schema tables."""
        suggestion = DiagnosticClassifier.suggest_sql_table_remediation(
            "custmrs", available_tables=["customers", "orders", "products"]
        )
        assert "Did you mean 'customers'?" in suggestion

    def test_sql_fuzzy_column_remediation(self):
        """Verify column typos are matched against known table columns."""
        suggestion = DiagnosticClassifier.suggest_sql_column_remediation(
            "frst_name", available_columns=["first_name", "last_name", "email"], table_name="users"
        )
        assert "Did you mean 'first_name'?" in suggestion

    def test_traceback_sanitization_filters_internals(self):
        """Verify internal framework frames are sanitized out of raw_traceback."""
        try:
            raise ValueError("Adversarial diagnostic test exception")
        except Exception as e:
            sanitized = DiagnosticClassifier.sanitize_traceback(e)
            assert "multiprocessing" not in sanitized
            assert "Adversarial diagnostic test exception" in sanitized
