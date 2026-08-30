"""
tests.unit.test_tools.test_sqlite_lifecycle
-------------------------------------------
Unit tests for SQLite engine in-memory and transient file lifecycles, and savepoints.
"""

import os
import sqlite3
import tempfile
import pytest

from nemo_eval.tools.sqlite_engine import SQLiteEngine, SQLiteEngineConfig


class TestSQLiteLifecycle:
    """Tests for SQLite database engine lifecycle and connection management."""

    def test_in_memory_lifecycle(self):
        engine = SQLiteEngine(SQLiteEngineConfig(read_only=False))
        engine.init_from_sql("""
            CREATE TABLE test_table (id INT PRIMARY KEY, name TEXT);
            INSERT INTO test_table VALUES (1, 'Alpha'), (2, 'Beta');
        """)
        
        tables = engine.get_known_tables()
        assert "test_table" in tables

        res = engine.execute_query("SELECT COUNT(*) FROM test_table;")
        assert res.rows[0]["COUNT(*)"] == 2

        engine.close()
        with pytest.raises(RuntimeError, match="closed"):
            engine.execute_query("SELECT 1;")

    def test_context_manager_lifecycle(self):
        with SQLiteEngine(SQLiteEngineConfig(read_only=False)) as engine:
            engine.init_from_sql("CREATE TABLE numbers (n INT); INSERT INTO numbers VALUES (10), (20);")
            res = engine.execute_query("SELECT SUM(n) as total FROM numbers;")
            assert res.rows[0]["total"] == 30

        # After exiting context manager, engine is closed
        with pytest.raises(RuntimeError):
            engine.execute_query("SELECT 1;")

    def test_transient_disk_copy_isolation(self, tmp_path):
        # Create a golden database file on disk
        golden_db_path = tmp_path / "golden.db"
        conn = sqlite3.connect(golden_db_path)
        conn.execute("CREATE TABLE source (val TEXT);")
        conn.execute("INSERT INTO source VALUES ('original_data');")
        conn.commit()
        conn.close()

        # Open SQLiteEngine pointing to golden db with read_only=False
        engine = SQLiteEngine(SQLiteEngineConfig(db_path=str(golden_db_path), read_only=False))
        assert engine._temp_file is not None
        assert os.path.exists(engine._temp_file)

        # Mutate transient instance
        engine.execute_query("INSERT INTO source VALUES ('mutated_data');")
        res = engine.execute_query("SELECT count(*) as cnt FROM source;")
        assert res.rows[0]["cnt"] == 2

        # Verify golden DB on disk remains completely unmodified
        verify_conn = sqlite3.connect(golden_db_path)
        golden_cnt = verify_conn.execute("SELECT count(*) FROM source;").fetchone()[0]
        verify_conn.close()
        assert golden_cnt == 1

        # Clean teardown deletes temp file
        temp_path = engine._temp_file
        engine.close()
        assert not os.path.exists(temp_path)

    def test_savepoint_lifecycle(self):
        engine = SQLiteEngine(SQLiteEngineConfig(read_only=False))
        engine.init_from_sql("CREATE TABLE accounts (id INT PRIMARY KEY, balance REAL); INSERT INTO accounts VALUES (1, 500.0);")

        engine.create_savepoint("sp1")
        engine.execute_query("UPDATE accounts SET balance = 1000.0 WHERE id = 1;")
        res1 = engine.execute_query("SELECT balance FROM accounts WHERE id = 1;")
        assert res1.rows[0]["balance"] == 1000.0

        # Rollback to savepoint
        engine.rollback_savepoint("sp1")
        res2 = engine.execute_query("SELECT balance FROM accounts WHERE id = 1;")
        assert res2.rows[0]["balance"] == 500.0

        # Create and release savepoint
        engine.create_savepoint("sp2")
        engine.execute_query("UPDATE accounts SET balance = 750.0 WHERE id = 1;")
        engine.release_savepoint("sp2")
        res3 = engine.execute_query("SELECT balance FROM accounts WHERE id = 1;")
        assert res3.rows[0]["balance"] == 750.0

        engine.close()
