"""
nemo_eval.tools.sqlite_engine
=============================
Hermetic SQLite database execution engine with progress handler timeouts, 
read-only PRAGMAs, schema introspection, and bounded result pagination.
"""

from dataclasses import dataclass, field
import difflib
import os
import shutil
import sqlite3
import tempfile
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field

from nemo_eval.tools.diagnostics import DiagnosticClassifier
from nemo_eval.tools.schemas import (
    ColumnInfo,
    ColumnSchema,
    DatabaseSchemaResponse,
    DiagnosticError,
    ForeignKeyInfo,
    ForeignKeySchema,
    TableInfoResponse,
    TableSchema,
    ToolResult,
)


@dataclass
class SQLiteEngineConfig:
    """Configuration options for SQLiteEngine."""
    db_path: Optional[str] = None
    read_only: bool = True
    max_rows_default: int = 50
    max_rows_hard_cap: int = 200
    timeout_seconds: float = 5.0
    opcode_check_interval: int = 1000
    max_page_count: int = 65536
    busy_timeout_ms: int = 5000
    enable_foreign_keys: bool = True
    sample_rows_count: int = 3


class QueryResult(BaseModel):
    """Execution output from a SQL query."""
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    returned_rows: int = 0
    total_rows_estimate: Optional[int] = None
    has_more: bool = False
    is_truncated: bool = False
    rowcount: int = -1
    execution_time_ms: float = 0.0
    warning: Optional[str] = None
    suggestion: Optional[str] = None


class SQLiteEngine:
    """
    Hermetic SQLite execution engine supporting in-memory and transient disk file lifecycles.
    Enforces read-only safety, opcode progress handler timeouts, and structured schema extraction.
    """
    def __init__(self, config: Optional[SQLiteEngineConfig] = None):
        self.config = config or SQLiteEngineConfig()
        self._temp_file: Optional[str] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._is_seeding: bool = False
        self._initialize_connection()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SQLite connection is closed.")
        return self._conn

    def _setup_authorizer(self) -> None:
        """Register authorizer hook to deny security PRAGMA tampering and unauthorized operations."""
        def _authorizer(action_code: int, arg1: Optional[str], arg2: Optional[str], db_name: Optional[str], trigger_name: Optional[str]) -> int:
            if self.config.read_only and not self._is_seeding:
                if action_code == getattr(sqlite3, "SQLITE_PRAGMA", 19):
                    if arg1 and arg1.lower() == "query_only":
                        if arg2 is not None and str(arg2).strip().lower() not in ("on", "1", "yes", "true"):
                            return getattr(sqlite3, "SQLITE_DENY", 1)
                    elif arg1 and arg1.lower() == "writable_schema":
                        if arg2 is not None and str(arg2).strip().lower() in ("on", "1", "yes", "true"):
                            return getattr(sqlite3, "SQLITE_DENY", 1)
            return getattr(sqlite3, "SQLITE_OK", 0)

        if self._conn is not None:
            self._conn.set_authorizer(_authorizer)

    def _initialize_connection(self) -> None:
        if self.config.db_path and self.config.db_path != ":memory:":
            if not os.path.exists(self.config.db_path):
                raise FileNotFoundError(f"Database file not found: {self.config.db_path}")
            # Create transient temporary copy to prevent golden asset mutation
            fd, tmp_path = tempfile.mkstemp(suffix=".db")
            os.close(fd)
            shutil.copyfile(self.config.db_path, tmp_path)
            self._temp_file = tmp_path
            self._conn = sqlite3.connect(self._temp_file, isolation_level=None)
        else:
            self._conn = sqlite3.connect(":memory:", isolation_level=None)

        cursor = self._conn.cursor()
        if self.config.enable_foreign_keys:
            cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute(f"PRAGMA busy_timeout = {self.config.busy_timeout_ms};")
        cursor.execute(f"PRAGMA max_page_count = {self.config.max_page_count};")
        if self.config.read_only:
            cursor.execute("PRAGMA query_only = ON;")
        self._setup_authorizer()

    def init_from_sql(self, sql_script: str) -> None:
        """Populate database from SQL DDL/DML script (temporarily relaxes read_only during seed)."""
        was_readonly = self.config.read_only
        cursor = self.connection.cursor()
        if was_readonly:
            self._is_seeding = True
            cursor.execute("PRAGMA query_only = OFF;")
        try:
            cursor.executescript(sql_script)
        finally:
            if was_readonly:
                try:
                    cursor.execute("PRAGMA query_only = ON;")
                finally:
                    self._is_seeding = False
            else:
                self._is_seeding = False

    def get_known_tables(self) -> List[str]:
        """List all user-defined tables and views in the database."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name ASC;")
        return [r[0] for r in cursor.fetchall()]

    def get_table_columns(self, table_name: str) -> List[str]:
        """List column names for a specific table."""
        cursor = self.connection.cursor()
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        return [r[1] for r in cursor.fetchall()]

    def get_schema(self, table_name: Optional[str] = None) -> DatabaseSchemaResponse:
        """Introspect tables, columns, constraints, foreign keys, and sample rows."""
        cursor = self.connection.cursor()
        query = "SELECT name, type, sql FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
        params = []
        if table_name:
            query += " AND lower(name) = lower(?)"
            params.append(table_name)
        query += " ORDER BY name ASC;"

        cursor.execute(query, params)
        tables_meta = cursor.fetchall()
        
        tables_dict: Dict[str, TableInfoResponse] = {}
        errors: List[str] = []

        for tbl_name, obj_type, ddl in tables_meta:
            try:
                # Column inspection
                cursor.execute(f'PRAGMA table_info("{tbl_name}")')
                col_rows = cursor.fetchall()
                columns = [
                    ColumnInfo(
                        cid=c[0],
                        name=c[1],
                        type=c[2] or "TEXT",
                        nullable=not bool(c[3]),
                        default_value=c[4],
                        primary_key=bool(c[5]),
                        pk_order=c[5]
                    )
                    for c in col_rows
                ]
                primary_keys = [c.name for c in columns if c.primary_key]

                # Foreign keys
                cursor.execute(f'PRAGMA foreign_key_list("{tbl_name}")')
                fk_rows = cursor.fetchall()
                foreign_keys = [
                    ForeignKeyInfo(
                        id=fk[0],
                        seq=fk[1],
                        referenced_table=fk[2],
                        from_column=fk[3],
                        referenced_column=fk[4],
                        on_update=fk[5],
                        on_delete=fk[6],
                        match=fk[7]
                    )
                    for fk in fk_rows
                ]

                # Sample rows
                sample_rows: List[Dict[str, Any]] = []
                try:
                    cursor.execute(f'SELECT * FROM "{tbl_name}" LIMIT {self.config.sample_rows_count}')
                    col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                    for row in cursor.fetchall():
                        sanitized_row = {}
                        for k, v in zip(col_names, row):
                            if isinstance(v, bytes):
                                sanitized_row[k] = f"<BLOB len={len(v)}>"
                            else:
                                sanitized_row[k] = v
                        sample_rows.append(sanitized_row)
                except Exception as e:
                    errors.append(f"Sample row retrieval failed for '{tbl_name}': {str(e)}")

                # Row count
                row_count = None
                try:
                    cursor.execute(f'SELECT COUNT(*) FROM "{tbl_name}"')
                    row_count = cursor.fetchone()[0]
                except Exception:
                    pass

                tables_dict[tbl_name] = TableInfoResponse(
                    name=tbl_name,
                    type=obj_type,
                    ddl=ddl,
                    row_count=row_count,
                    columns=columns,
                    foreign_keys=foreign_keys,
                    primary_keys=primary_keys,
                    sample_rows=sample_rows
                )
            except Exception as e:
                errors.append(f"Schema extraction failed for '{tbl_name}': {str(e)}")

        return DatabaseSchemaResponse(
            database_type="sqlite",
            table_count=len(tables_dict),
            tables=tables_dict,
            errors=errors
        )

    def execute_query(self, query: str, max_rows: Optional[int] = None) -> QueryResult:
        """
        Executes SQL statement with opcode progress handler timeout and row bounding.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        effective_limit = max_rows if max_rows is not None else self.config.max_rows_default
        effective_limit = max(1, min(effective_limit, self.config.max_rows_hard_cap))

        start_perf = time.perf_counter()
        
        def _progress_cb() -> int:
            if (time.perf_counter() - start_perf) > self.config.timeout_seconds:
                return 1
            return 0

        cursor = self.connection.cursor()
        if self.config.read_only:
            cursor.execute("PRAGMA query_only = ON;")
        self.connection.set_progress_handler(_progress_cb, self.config.opcode_check_interval)
        
        try:
            cursor.execute(query)
            
            # Result set query (SELECT / PRAGMA / EXPLAIN)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                raw_rows = cursor.fetchmany(effective_limit + 1)
                
                has_more = len(raw_rows) > effective_limit
                is_truncated = has_more
                output_rows = raw_rows[:effective_limit]
                
                sanitized_rows: List[Dict[str, Any]] = []
                for row in output_rows:
                    record = {}
                    for col_name, val in zip(columns, row):
                        if isinstance(val, bytes):
                            record[col_name] = f"<BLOB len={len(val)}>"
                        else:
                            record[col_name] = val
                    sanitized_rows.append(record)

                suggestion = None
                warning = None
                if is_truncated:
                    suggestion = (
                        f"Result truncated at {effective_limit} rows. "
                        "Use LIMIT and OFFSET, or SQL aggregation functions (COUNT, SUM, AVG) for deeper queries."
                    )
                    warning = f"Output truncated at row limit {effective_limit}."

                elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
                return QueryResult(
                    columns=columns,
                    rows=sanitized_rows,
                    returned_rows=len(sanitized_rows),
                    has_more=has_more,
                    is_truncated=is_truncated,
                    rowcount=cursor.rowcount,
                    execution_time_ms=round(elapsed_ms, 3),
                    warning=warning,
                    suggestion=suggestion
                )
            else:
                # DML / DDL statement
                elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
                return QueryResult(
                    columns=[],
                    rows=[],
                    returned_rows=0,
                    has_more=False,
                    is_truncated=False,
                    rowcount=cursor.rowcount,
                    execution_time_ms=round(elapsed_ms, 3)
                )
        finally:
            self.connection.set_progress_handler(None, 0)
            if self.config.read_only:
                try:
                    cursor.execute("PRAGMA query_only = ON;")
                except Exception:
                    pass

    def execute_tool(self, query: str, max_rows: Optional[int] = None) -> ToolResult:
        """High-level tool wrapper returning standard ToolResult envelope."""
        start_perf = time.perf_counter()
        try:
            res = self.execute_query(query, max_rows=max_rows)
            return ToolResult(
                status="success",
                execution_time_ms=res.execution_time_ms,
                data={
                    "columns": res.columns,
                    "rows": res.rows,
                    "count": res.returned_rows,
                    "has_more": res.has_more,
                    "suggestion": res.suggestion or "",
                },
                stdout="",
                stderr=""
            )
        except BaseException as e:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            known_tables = []
            try:
                known_tables = self.get_known_tables()
            except Exception:
                pass

            diag = DiagnosticClassifier.create_diagnostic_error(
                exc=e,
                source_code=query,
                context={
                    "query": query,
                    "available_tables": known_tables,
                    "timeout_seconds": self.config.timeout_seconds,
                }
            )
            return ToolResult(
                status="error",
                execution_time_ms=round(elapsed_ms, 3),
                data=None,
                stdout="",
                stderr="",
                error=diag
            )

    def schema_tool(self, table_name: Optional[str] = None) -> ToolResult:
        """High-level tool wrapper returning standard ToolResult envelope for schema inspection."""
        start_perf = time.perf_counter()
        try:
            schema_resp = self.get_schema(table_name=table_name)
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            return ToolResult(
                status="success",
                execution_time_ms=round(elapsed_ms, 3),
                data=schema_resp.model_dump(),
                stdout="",
                stderr=""
            )
        except BaseException as e:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            diag = DiagnosticClassifier.create_diagnostic_error(
                exc=e,
                context={"table_name": table_name}
            )
            return ToolResult(
                status="error",
                execution_time_ms=round(elapsed_ms, 3),
                data=None,
                stdout="",
                stderr="",
                error=diag
            )

    def create_savepoint(self, savepoint_name: str) -> None:
        """Create transaction savepoint for reversible speculative modifications."""
        if self.config.read_only:
            raise sqlite3.OperationalError("Cannot create savepoints in read-only mode.")
        self.connection.execute(f"SAVEPOINT {savepoint_name};")

    def release_savepoint(self, savepoint_name: str) -> None:
        """Release transaction savepoint and commit intermediate modifications."""
        if self.config.read_only:
            raise sqlite3.OperationalError("Cannot release savepoints in read-only mode.")
        self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name};")

    def rollback_savepoint(self, savepoint_name: str) -> None:
        """Rollback state to a specified savepoint."""
        if self.config.read_only:
            raise sqlite3.OperationalError("Cannot rollback savepoints in read-only mode.")
        self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name};")

    def close(self) -> None:
        """Cleanly close connection and delete temporary disk files."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        if self._temp_file and os.path.exists(self._temp_file):
            try:
                os.unlink(self._temp_file)
            except Exception:
                pass
            self._temp_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
