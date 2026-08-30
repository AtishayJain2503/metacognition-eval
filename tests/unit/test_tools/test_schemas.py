"""
tests.unit.test_tools.test_schemas
----------------------------------
Unit tests for Pydantic v2 schemas and OpenAI/NeMo tool specifications.
"""

import pytest
from pydantic import ValidationError

from nemo_eval.tools.schemas import (
    ColumnInfo,
    DatabaseSchemaResponse,
    DiagnosticError,
    ForeignKeyInfo,
    REPLInput,
    SQLiteQueryInput,
    SQLiteSchemaInput,
    TableInfoResponse,
    TabularInspectInput,
    ToolResult,
    get_openai_tool_specs,
    get_tool_input_model,
    validate_tool_payload,
)


class TestToolSchemas:
    """Tests for Pydantic input/output schemas and tool definitions."""

    def test_diagnostic_error_model(self):
        diag = DiagnosticError(
            error_type="SyntaxError",
            message="invalid syntax",
            line_number=3,
            column_offset=12,
            code_snippet="x = 10 +",
            pointer="         ^",
            suggestion="Add missing operand.",
            raw_traceback="SyntaxError: invalid syntax"
        )
        assert diag.error_type == "SyntaxError"
        assert diag.line_number == 3
        prompt_str = diag.format_for_prompt()
        assert "[Error: SyntaxError]" in prompt_str
        assert "Line 3: x = 10 +" in prompt_str
        assert "Suggestion: Add missing operand." in prompt_str

    def test_tool_result_model(self):
        res_success = ToolResult(
            status="success",
            execution_time_ms=15.2,
            data={"rows": [1, 2, 3]},
            stdout="Completed",
            stderr=""
        )
        assert res_success.is_success is True
        assert res_success.is_error is False
        obs = res_success.to_agent_observation()
        assert "Completed" in obs
        assert "rows" in obs

        res_error = ToolResult(
            status="error",
            execution_time_ms=5.0,
            error=DiagnosticError(
                error_type="NameError",
                message="name 'foo' is not defined",
                suggestion="Define 'foo' before use."
            )
        )
        assert res_error.is_success is False
        assert res_error.is_error is True
        assert "name 'foo' is not defined" in res_error.to_agent_observation()

    def test_repl_input_validation(self):
        valid = REPLInput(code="print(1 + 1)", session_id="s1")
        assert valid.code == "print(1 + 1)"
        assert valid.session_id == "s1"

        with pytest.raises(ValidationError):
            REPLInput(code="")  # min_length=1

        with pytest.raises(ValidationError):
            REPLInput(code="x = 1", extra_field="forbidden")  # extra="forbid"

    def test_sqlite_query_input_validation(self):
        valid = SQLiteQueryInput(query="SELECT * FROM customers;", max_rows=25)
        assert valid.query == "SELECT * FROM customers;"
        assert valid.max_rows == 25

        with pytest.raises(ValidationError):
            SQLiteQueryInput(query="", max_rows=10)

        with pytest.raises(ValidationError):
            SQLiteQueryInput(query="SELECT 1;", max_rows=0)  # ge=1

    def test_sqlite_schema_input_validation(self):
        valid1 = SQLiteSchemaInput()
        assert valid1.table_name is None

        valid2 = SQLiteSchemaInput(table_name="orders", database_id="db_main")
        assert valid2.table_name == "orders"
        assert valid2.database_id == "db_main"

    def test_tabular_inspect_input_validation(self):
        valid = TabularInspectInput(file_path="data.csv", action="summary", n_rows=10)
        assert valid.file_path == "data.csv"
        assert valid.action == "summary"
        assert valid.n_rows == 10

        with pytest.raises(ValidationError):
            TabularInspectInput(file_path="data.csv", action="invalid_action")

    def test_openai_tool_specs_export(self):
        specs = get_openai_tool_specs()
        assert len(specs) == 4
        tool_names = [s["function"]["name"] for s in specs]
        assert "python_repl" in tool_names
        assert "sqlite_query" in tool_names
        assert "sqlite_schema" in tool_names
        assert "tabular_inspect" in tool_names

        for spec in specs:
            assert spec["type"] == "function"
            assert "name" in spec["function"]
            assert "description" in spec["function"]
            assert "parameters" in spec["function"]

    def test_validate_tool_payload_helper(self):
        payload = {"code": "x = 42", "session_id": "test_sess"}
        validated = validate_tool_payload("python_repl", payload)
        assert isinstance(validated, REPLInput)
        assert validated.code == "x = 42"

        with pytest.raises(ValueError, match="Unknown tool name"):
            validate_tool_payload("unknown_tool", {})

    def test_database_schema_models(self):
        col = ColumnInfo(cid=0, name="id", type="INTEGER", primary_key=True)
        fk = ForeignKeyInfo(id=0, seq=0, from_column="cat_id", referenced_table="categories", referenced_column="id")
        tbl = TableInfoResponse(
            name="products",
            type="table",
            ddl="CREATE TABLE products (id INTEGER PRIMARY KEY, cat_id INTEGER);",
            row_count=10,
            columns=[col],
            foreign_keys=[fk],
            primary_keys=["id"],
            sample_rows=[{"id": 1, "cat_id": 2}]
        )
        db_schema = DatabaseSchemaResponse(
            database_type="sqlite",
            table_count=1,
            tables={"products": tbl},
            errors=[]
        )
        assert db_schema.table_count == 1
        assert "products" in db_schema.tables
        assert db_schema.tables["products"].columns[0].name == "id"
