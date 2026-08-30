"""
tests.unit.test_eval.test_sql_match
===================================
Unit tests for multiset SQL matching, cell normalization, ORDER BY enforcement, and timeout/safety.
"""

import pytest

from nemo_eval.eval.sql_match import (
    evaluate_sql,
    execute_sql_safely,
    extract_sql_from_text,
    has_order_by_clause,
    normalize_row_tuple,
    normalize_sql_cell,
)


class TestSqlCellAndRowNormalization:
    """Test cell normalization and row tuple hashing."""

    def test_normalize_sql_cell(self):
        assert normalize_sql_cell(None) is None
        assert normalize_sql_cell("NULL") is None
        assert normalize_sql_cell("null") is None
        assert normalize_sql_cell("None") is None
        assert normalize_sql_cell(1.0) == 1
        assert normalize_sql_cell(100.25001) == 100.25
        assert normalize_sql_cell("  California  ") == "california"

    def test_normalize_row_tuple(self):
        raw_row = ("Alice", 95000.0, "NULL")
        norm = normalize_row_tuple(raw_row)
        assert norm == ("alice", 95000, None)


class TestSqlOrderAndExtraction:
    """Test extracting SQL from markdown and detecting ORDER BY."""

    def test_extract_sql_from_text(self):
        text = "```sql\nSELECT name FROM employees WHERE salary > 50000;\n```"
        assert extract_sql_from_text(text) == "SELECT name FROM employees WHERE salary > 50000"

    def test_has_order_by_clause(self):
        assert has_order_by_clause("SELECT name FROM employees ORDER BY salary DESC") is True
        assert has_order_by_clause("SELECT name FROM employees WHERE id = 1") is False


class TestSqlExecutionAndEquivalence:
    """Test SQL execution matching against SQLite database."""

    def test_multiset_sql_matching_unordered(self, eval_sqlite_db):
        cand_sql = "SELECT name, salary FROM employees WHERE dept = 'Engineering';"
        gold_sql = "SELECT name, salary FROM employees WHERE dept = 'Engineering' ORDER BY name DESC;"
        # Unordered multiset matching allows different row orders when check_order=False
        res = evaluate_sql(cand_sql, gold_sql, db_path=eval_sqlite_db, check_order=False)
        assert res.is_correct is True
        assert res.score == 1.0

    def test_strict_ordered_sql_matching(self, eval_sqlite_db):
        cand_sql = "SELECT name FROM employees ORDER BY salary ASC;"
        gold_sql = "SELECT name FROM employees ORDER BY salary DESC;"
        res = evaluate_sql(cand_sql, gold_sql, db_path=eval_sqlite_db, check_order=True)
        assert res.is_correct is False

    def test_multiset_duplicate_row_preservation(self, eval_sqlite_db):
        # Two identical rows must match count
        cand_res = [("Engineering",), ("Engineering",), ("Engineering",)]
        gold_res = [("Engineering",), ("Engineering",), ("Engineering",)]
        res_match = evaluate_sql(cand_res, gold_res)
        assert res_match.is_correct is True

        cand_fewer = [("Engineering",), ("Engineering",)]
        res_mismatch = evaluate_sql(cand_fewer, gold_res)
        assert res_mismatch.is_correct is False

    def test_column_projection_count_mismatch(self, eval_sqlite_db):
        cand_sql = "SELECT id, name, salary FROM employees;"
        gold_sql = "SELECT id, name FROM employees;"
        res = evaluate_sql(cand_sql, gold_sql, db_path=eval_sqlite_db)
        assert res.is_correct is False
        assert "Column count mismatch" in res.diagnostic_message

    def test_sql_syntax_error_returns_diagnostic(self, eval_sqlite_db):
        bad_sql = "SELECT * FROM non_existent_table WHERE abc = 123;"
        gold_sql = "SELECT count(*) FROM employees;"
        res = evaluate_sql(bad_sql, gold_sql, db_path=eval_sqlite_db)
        assert res.is_correct is False
        assert "OperationalError" in res.diagnostic_message

    def test_sql_safety_prohibits_dml(self, eval_sqlite_db):
        drop_sql = "DROP TABLE employees;"
        rows, cols, err = execute_sql_safely(eval_sqlite_db, drop_sql)
        assert err is not None
        assert "Prohibited write statement" in err
