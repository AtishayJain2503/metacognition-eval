"""
nemo_eval.tools.schemas
-----------------------
Pydantic v2 data models and OpenAI/NeMo function calling JSON schemas 
for hermetic tool execution environments.
"""

import json
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class DiagnosticError(BaseModel):
    """Structured error diagnostic envelope engineered for LLM self-correction."""
    model_config = ConfigDict(extra="ignore")

    error_type: str = Field(..., description="Classification category of the error.")
    message: str = Field(..., description="Human-readable error description.")
    line_number: Optional[int] = Field(default=None, description="1-indexed line where the error occurred.")
    column_offset: Optional[int] = Field(default=None, description="1-indexed column position of the error.")
    code_snippet: Optional[str] = Field(default=None, description="Offending line of source code.")
    pointer: Optional[str] = Field(default=None, description="Visual caret pointer indicating error column.")
    suggestion: str = Field(default="", description="Actionable remediation hint for the LLM agent.")
    raw_traceback: str = Field(default="", description="Sanitized traceback tail.")

    def format_for_prompt(self) -> str:
        """Format diagnostic into a high-signal prompt injection string."""
        lines = [f"[Error: {self.error_type}] {self.message}"]
        if self.code_snippet:
            prefix = f"Line {self.line_number}: " if self.line_number is not None else "Code: "
            lines.append(f"{prefix}{self.code_snippet}")
            if self.pointer:
                indent = " " * len(prefix)
                lines.append(f"{indent}{self.pointer}")
        if self.suggestion:
            lines.append(f"Suggestion: {self.suggestion}")
        return "\n".join(lines)


class ToolResult(BaseModel):
    """Unified result envelope for all hermetic tool invocations."""
    model_config = ConfigDict(extra="ignore")

    status: Literal["success", "error"] = Field(..., description="Execution outcome status.")
    execution_time_ms: float = Field(default=0.0, ge=0.0, description="Wall-clock execution duration in milliseconds.")
    data: Any = Field(default=None, description="Structured payload: query rows, schema dict, or return value.")
    stdout: str = Field(default="", description="Captured standard output string.")
    stderr: str = Field(default="", description="Captured standard error string.")
    error: Optional[DiagnosticError] = Field(default=None, description="Diagnostic details if status is error.")

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    def to_agent_observation(self, max_length: int = 4000) -> str:
        """Convert execution result into standard agent observation string with truncation."""
        if self.is_error and self.error:
            obs = self.error.format_for_prompt()
            if self.stdout:
                obs = f"Stdout:\n{self.stdout}\n\n{obs}"
            return obs

        parts = []
        if self.stdout:
            parts.append(self.stdout.strip())
        if self.data is not None:
            if isinstance(self.data, (dict, list)):
                try:
                    parts.append(json.dumps(self.data, indent=2, default=str))
                except Exception:
                    parts.append(str(self.data))
            else:
                parts.append(str(self.data))

        content = "\n".join(parts) if parts else "Execution completed with no output."
        if len(content) > max_length:
            content = content[:max_length] + f"\n... [Output truncated. Total chars: {len(content)}]"
        return content


class ColumnInfo(BaseModel):
    """Metadata for a database column."""
    model_config = ConfigDict(extra="ignore")

    cid: int = 0
    name: str
    type: str = "TEXT"
    nullable: bool = True
    default_value: Optional[Any] = None
    primary_key: bool = False
    pk_order: int = 0


ColumnSchema = ColumnInfo


class ForeignKeyInfo(BaseModel):
    """Foreign key relationship metadata."""
    model_config = ConfigDict(extra="ignore")

    id: int = 0
    seq: int = 0
    from_column: str
    referenced_table: str
    referenced_column: str
    on_update: str = "NO ACTION"
    on_delete: str = "NO ACTION"
    match: str = "NONE"


ForeignKeySchema = ForeignKeyInfo


class TableInfoResponse(BaseModel):
    """Complete schema definition and context samples for a database table or view."""
    model_config = ConfigDict(extra="ignore")

    name: str
    type: Literal["table", "view"] = "table"
    ddl: Optional[str] = None
    row_count: Optional[int] = None
    columns: List[ColumnInfo] = Field(default_factory=list)
    foreign_keys: List[ForeignKeyInfo] = Field(default_factory=list)
    primary_keys: List[str] = Field(default_factory=list)
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)


TableSchema = TableInfoResponse


class DatabaseSchemaResponse(BaseModel):
    """Response envelope containing full introspected schema for a database."""
    model_config = ConfigDict(extra="ignore")

    database_type: str = "sqlite"
    table_count: int = 0
    tables: Dict[str, TableInfoResponse] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool Input Payload Schemas
# ---------------------------------------------------------------------------

class REPLInput(BaseModel):
    """Input schema for python_repl tool."""
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=1, description="The Python code snippet to execute.")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier for stateful execution.")


class SQLiteQueryInput(BaseModel):
    """Input schema for sqlite_query tool."""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="The SQL query statement to execute.")
    max_rows: int = Field(default=50, ge=1, le=1000, description="Maximum rows to return.")
    database_id: Optional[str] = Field(default=None, description="Optional database identifier.")


class SQLiteSchemaInput(BaseModel):
    """Input schema for sqlite_schema tool."""
    model_config = ConfigDict(extra="forbid")

    table_name: Optional[str] = Field(default=None, description="Optional specific table name to inspect.")
    database_id: Optional[str] = Field(default=None, description="Optional database identifier.")


class TabularInspectInput(BaseModel):
    """Input schema for tabular_inspect tool."""
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(..., min_length=1, description="Relative or absolute path to the local tabular dataset.")
    action: Literal["schema", "summary", "head", "tail"] = Field(default="schema", description="Inspection action.")
    n_rows: int = Field(default=5, ge=1, le=100, description="Number of sample rows to retrieve.")


TOOL_INPUT_MODELS: Dict[str, type[BaseModel]] = {
    "python_repl": REPLInput,
    "sqlite_query": SQLiteQueryInput,
    "sqlite_schema": SQLiteSchemaInput,
    "tabular_inspect": TabularInspectInput,
}


TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "python_repl": {
        "type": "function",
        "function": {
            "name": "python_repl",
            "description": "Execute deterministic Python code in a hermetic sandboxed REPL environment. Captures stdout, return expressions, and structured error diagnostics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code snippet to execute."
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session identifier for stateful execution across sequential turns."
                    }
                },
                "required": ["code"],
                "additionalProperties": False
            }
        }
    },
    "sqlite_query": {
        "type": "function",
        "function": {
            "name": "sqlite_query",
            "description": "Execute a SQL query against the hermetically isolated SQLite database with automatic row bounding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query statement to execute."
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": "Maximum number of rows to return (default: 50, max: 1000).",
                        "default": 50
                    },
                    "database_id": {
                        "type": "string",
                        "description": "Optional identifier for the target SQLite database instance."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    "sqlite_schema": {
        "type": "function",
        "function": {
            "name": "sqlite_schema",
            "description": "Retrieve comprehensive schema details, table DDLs, column types, primary/foreign keys, and sample rows from the active database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Optional specific table name to inspect. If omitted, returns all tables."
                    },
                    "database_id": {
                        "type": "string",
                        "description": "Optional identifier for the target SQLite database instance."
                    }
                },
                "required": [],
                "additionalProperties": False
            }
        }
    },
    "tabular_inspect": {
        "type": "function",
        "function": {
            "name": "tabular_inspect",
            "description": "Inspect schema, column data types, summary statistics, and sample rows from a local tabular dataset (CSV, Parquet, or JSONL).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the local tabular dataset file (CSV, Parquet, or JSONL)."
                    },
                    "action": {
                        "type": "string",
                        "enum": ["schema", "summary", "head", "tail"],
                        "description": "Inspection action: 'schema' for dtypes/shape, 'summary' for statistical profile, 'head'/'tail' for sample rows.",
                        "default": "schema"
                    },
                    "n_rows": {
                        "type": "integer",
                        "description": "Number of rows to retrieve for head/tail actions (default: 5).",
                        "default": 5
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        }
    }
}


def get_openai_tool_specs() -> List[Dict[str, Any]]:
    """Return list of tool specifications formatted for OpenAI / NeMo function calling API."""
    return list(TOOL_SCHEMAS.values())


def get_tool_input_model(tool_name: str) -> type[BaseModel]:
    """Get Pydantic validation model for a tool name."""
    if tool_name not in TOOL_INPUT_MODELS:
        raise ValueError(f"Unknown tool name: {tool_name}. Supported tools: {list(TOOL_INPUT_MODELS.keys())}")
    return TOOL_INPUT_MODELS[tool_name]


def validate_tool_payload(tool_name: str, arguments: Dict[str, Any]) -> BaseModel:
    """Validate dictionary arguments against the tool's Pydantic model."""
    model_cls = get_tool_input_model(tool_name)
    return model_cls.model_validate(arguments)
