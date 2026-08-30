"""
tests.unit.test_tools.test_sqlite_security_limits
-------------------------------------------------
Unit tests for read-only PRAGMA, opcode progress handler timeouts, and resource quotas.
"""

import sqlite3
import time
import pytest

from nemo_eval.tools.sqlite_engine import SQLiteEngine, SQLiteEngineConfig


class TestSQLiteSecurityLimits:
    """Tests for SQLite query limits, read-only mode, and runaway query termination."""

    def test_readonly_pragma_enforcement(self):
        engine = SQLiteEngine(SQLiteEngineConfig(read_only=True))
        # Initial seeding works via init_from_sql
        engine.init_from_sql("CREATE TABLE items (id INT, val TEXT); INSERT INTO items VALUES (1, 'A');")

        # Read query succeeds
        res = engine.execute_query("SELECT * FROM items;")
        assert len(res.rows) == 1

        # Write operations fail with OperationalError
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            engine.execute_query("INSERT INTO items VALUES (2, 'B');")

        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            engine.execute_query("UPDATE items SET val = 'C' WHERE id = 1;")

        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            engine.execute_query("DROP TABLE items;")

        engine.close()

    @pytest.mark.parametrize("pragma_payload", [
        "PRAGMA query_only = OFF;",
        "PRAGMA query_only = 0;",
        "PRAGMA query_only = false;",
        "PRAGMA query_only = NO;",
        "PRAGMA query_only = '0';",
        "PRAGMA query_only = \"OFF\";",
        "pragma QUERY_ONLY = 0;",
        "   PRAGMA   query_only   =   0  ;  ",
        "/* bypass */ PRAGMA query_only = 0;",
        "-- comment\nPRAGMA query_only = OFF;",
    ])
    def test_pragma_query_only_tamper_resistance(self, pragma_payload):
        engine = SQLiteEngine(SQLiteEngineConfig(read_only=True))
        engine.init_from_sql("CREATE TABLE items (id INT, val TEXT); INSERT INTO items VALUES (1, 'A');")

        # Attempt PRAGMA modification
        try:
            engine.execute_query(pragma_payload)
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            pass  # Active rejection is also valid

        # Immediate write attempt must fail
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            engine.execute_query("INSERT INTO items VALUES (99, 'Hacked');")

        # Connection query_only pragma must remain 1
        res = engine.execute_query("PRAGMA query_only;")
        assert res.rows[0]["query_only"] == 1

        # Original data untouched
        res_check = engine.execute_query("SELECT COUNT(*) as cnt FROM items;")
        assert res_check.rows[0]["cnt"] == 1

        engine.close()

    def test_sql_savepoint_and_transaction_statements_blocked_in_readonly(self):
        engine = SQLiteEngine(SQLiteEngineConfig(read_only=True))
        engine.init_from_sql("CREATE TABLE items (id INT, val TEXT); INSERT INTO items VALUES (1, 'A');")

        # Direct savepoint helper methods must fail in read-only mode
        with pytest.raises(sqlite3.OperationalError, match="read-only"):
            engine.create_savepoint("sp1")

        with pytest.raises(sqlite3.OperationalError, match="read-only"):
            engine.release_savepoint("sp1")

        with pytest.raises(sqlite3.OperationalError, match="read-only"):
            engine.rollback_savepoint("sp1")

        engine.close()

    def test_readonly_false_allows_full_dml_and_ddl(self):
        engine = SQLiteEngine(SQLiteEngineConfig(read_only=False))
        engine.init_from_sql("CREATE TABLE items (id INT, val TEXT);")

        # INSERT
        res_ins = engine.execute_query("INSERT INTO items VALUES (1, 'Alpha');")
        assert res_ins.rowcount == 1

        # UPDATE
        res_upd = engine.execute_query("UPDATE items SET val = 'Beta' WHERE id = 1;")
        assert res_upd.rowcount == 1

        # SELECT
        res_sel = engine.execute_query("SELECT val FROM items WHERE id = 1;")
        assert res_sel.rows[0]["val"] == "Beta"

        # DELETE
        res_del = engine.execute_query("DELETE FROM items WHERE id = 1;")
        assert res_del.rowcount == 1

        # DROP
        engine.execute_query("DROP TABLE items;")
        assert "items" not in engine.get_known_tables()

        engine.close()

    def test_connection_health_and_isolation_after_blocked_query(self):
        engine = SQLiteEngine(SQLiteEngineConfig(read_only=True))
        engine.init_from_sql("CREATE TABLE items (id INT, val TEXT); INSERT INTO items VALUES (1, 'A'), (2, 'B');")

        # Blocked write
        with pytest.raises(sqlite3.OperationalError):
            engine.execute_query("INSERT INTO items VALUES (3, 'C');")

        # Next query executes cleanly without connection failure
        res = engine.execute_query("SELECT * FROM items ORDER BY id ASC;")
        assert len(res.rows) == 2
        assert res.rows[0]["val"] == "A"
        assert res.rows[1]["val"] == "B"

        engine.close()

    def test_opcode_progress_timeout_recursive_cte(self):
        # Configure small timeout (0.5s)
        config = SQLiteEngineConfig(read_only=True, timeout_seconds=0.5, opcode_check_interval=500)
        engine = SQLiteEngine(config)

        runaway_cte = "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt) SELECT count(*) FROM cnt;"
        start_time = time.perf_counter()

        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            engine.execute_query(runaway_cte)

        elapsed = time.perf_counter() - start_time
        assert elapsed < 2.0  # Stopped promptly near 0.5s limit

        engine.close()

    def test_savepoint_prohibited_in_readonly_mode(self):
        engine = SQLiteEngine(SQLiteEngineConfig(read_only=True))
        with pytest.raises(sqlite3.OperationalError, match="read-only"):
            engine.create_savepoint("sp_fail")
        engine.close()

    def test_high_level_execute_tool_catches_timeout_and_readonly(self):
        config = SQLiteEngineConfig(read_only=True, timeout_seconds=0.5, opcode_check_interval=500)
        engine = SQLiteEngine(config)
        engine.init_from_sql("CREATE TABLE data (x INT);")

        # Read-only violation returns error ToolResult
        res_write = engine.execute_tool("INSERT INTO data VALUES (1);")
        assert res_write.status == "error"
        assert res_write.error is not None
        assert "read-only" in res_write.error.suggestion.lower() or "readonly" in res_write.error.suggestion.lower()

        # Runaway query returns error ToolResult with timeout diagnostic
        runaway_cte = "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt) SELECT count(*) FROM cnt;"
        res_timeout = engine.execute_tool(runaway_cte)
        assert res_timeout.status == "error"
        assert res_timeout.error is not None
        assert "opcode timeout" in res_timeout.error.suggestion.lower() or "interrupted" in res_timeout.error.message.lower()

        engine.close()
