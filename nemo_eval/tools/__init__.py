"""
nemo_eval.tools
===============
Hermetic tool sandboxes and auxiliary execution engines:
- PythonREPL: Process-isolated AST-validated Python sandbox with dual-phase compilation.
- SQLiteEngine: Deterministic SQLite database engine with read-only PRAGMAs and progress timeout handlers.
- TabularEngine: Multi-format tabular data processor (CSV, Parquet, JSONL).
- DiagnosticClassifier / DiagnosticFormatter: Structured error classifier and syntax highlighter.
- schemas: Pydantic v2 data models and OpenAI/NeMo function calling JSON schemas.
"""

from nemo_eval.tools.diagnostics import (
    DiagnosticClassifier,
    DiagnosticFormatter,
)
from nemo_eval.tools.repl import (
    ALLOWED_IMPORT_MODULES,
    CodeSecurityValidator,
    FORBIDDEN_ATTRIBUTES,
    FORBIDDEN_CALLS,
    FORBIDDEN_MODULES,
    ProcessWorkerSandbox,
    PythonREPL,
    REPLSessionManager,
    SAFE_BUILTINS,
    SecurityViolationError,
    compile_and_execute_ast,
)
from nemo_eval.tools.schemas import (
    ColumnInfo,
    ColumnSchema,
    DatabaseSchemaResponse,
    DiagnosticError,
    ForeignKeyInfo,
    ForeignKeySchema,
    REPLInput,
    SQLiteQueryInput,
    SQLiteSchemaInput,
    TableInfoResponse,
    TableSchema,
    TabularInspectInput,
    ToolResult,
    TOOL_INPUT_MODELS,
    TOOL_SCHEMAS,
    get_openai_tool_specs,
    get_tool_input_model,
    validate_tool_payload,
)
from nemo_eval.tools.sqlite_engine import (
    QueryResult,
    SQLiteEngine,
    SQLiteEngineConfig,
)
from nemo_eval.tools.tabular import (
    TabularColumnSummary,
    TabularEngine,
    TabularSampleInfo,
    TabularSchemaInfo,
    TabularSummaryInfo,
)

__all__ = [
    # Schemas
    "ToolResult",
    "DiagnosticError",
    "REPLInput",
    "SQLiteQueryInput",
    "SQLiteSchemaInput",
    "TabularInspectInput",
    "ColumnInfo",
    "ColumnSchema",
    "ForeignKeyInfo",
    "ForeignKeySchema",
    "TableInfoResponse",
    "TableSchema",
    "DatabaseSchemaResponse",
    "TOOL_SCHEMAS",
    "TOOL_INPUT_MODELS",
    "get_openai_tool_specs",
    "get_tool_input_model",
    "validate_tool_payload",
    # Diagnostics
    "DiagnosticClassifier",
    "DiagnosticFormatter",
    # REPL
    "PythonREPL",
    "CodeSecurityValidator",
    "SecurityViolationError",
    "REPLSessionManager",
    "ProcessWorkerSandbox",
    "SAFE_BUILTINS",
    "FORBIDDEN_MODULES",
    "FORBIDDEN_ATTRIBUTES",
    "FORBIDDEN_CALLS",
    "ALLOWED_IMPORT_MODULES",
    "compile_and_execute_ast",
    # SQLite
    "SQLiteEngine",
    "SQLiteEngineConfig",
    "QueryResult",
    # Tabular
    "TabularEngine",
    "TabularSchemaInfo",
    "TabularSummaryInfo",
    "TabularSampleInfo",
    "TabularColumnSummary",
]
