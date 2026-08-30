"""
tests.unit.test_tools.test_sqlite_query
---------------------------------------
Unit tests for SQL query execution, bounding, pagination, and error envelopes.
"""

import pytest

from nemo_eval.tools.sqlite_engine import SQLiteEngine, SQLiteEngineConfig


class TestSQLiteQuery:
    """Tests for SQL query execution, bounding, and pagination flags."""

    def test_query_result_bounding_and_pagination(self):
        engine = SQLiteEngine(SQLiteEngineConfig(read_only=False, max_rows_default=25))
        # Insert 60 rows
        engine.init_from_sql("""
            CREATE TABLE large_seq (id INTEGER PRIMARY KEY, score REAL);
        """)
        insert_sql = "INSERT INTO large_seq VALUES " + ", ".join(f"({i}, {i * 1.5})" for i in range(1, 61)) + ";"
        engine.execute_query(insert_sql)

        # Default limit of 25 rows
        res = engine.execute_query("SELECT * FROM large_seq ORDER BY id ASC;")
        assert res.returned_rows == 25
        assert len(res.rows) == 25
        assert res.has_more is True
        assert res.is_truncated is True
        assert "LIMIT and OFFSET" in res.suggestion
        assert res.rows[0]["id"] == 1
        assert res.rows[24]["id"] == 25

        # Explicit limit of 10 rows
        res10 = engine.execute_query("SELECT * FROM large_seq ORDER BY id ASC;", max_rows=10)
        assert res10.returned_rows == 10
        assert res10.has_more is True

        # Query returning fewer rows than limit
        res_small = engine.execute_query("SELECT * FROM large_seq WHERE id <= 5;")
        assert res_small.returned_rows == 5
        assert res_small.has_more is False
        assert res_small.is_truncated is False
        assert res_small.suggestion is None

        engine.close()

    def test_execute_tool_success(self, sample_sqlite_engine):
        res = sample_sqlite_engine.execute_tool("SELECT name, price FROM products WHERE price > 50 ORDER BY price DESC;")
        assert res.status == "success"
        assert res.data["count"] == 2
        assert res.data["rows"][0]["name"] == "Laptop"
        assert res.data["rows"][0]["price"] == 999.99

    def test_execute_tool_syntax_error_diagnostic(self, sample_sqlite_engine):
        res = sample_sqlite_engine.execute_tool("SLECT * FROM products;")
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "OperationalError"
        assert "^" in res.error.pointer

    def test_execute_tool_missing_table_fuzzy_match(self, sample_sqlite_engine):
        res = sample_sqlite_engine.execute_tool("SELECT * FROM productss;")
        assert res.status == "error"
        assert res.error is not None
        assert "products" in res.error.suggestion
        assert "Did you mean 'products'?" in res.error.suggestion
