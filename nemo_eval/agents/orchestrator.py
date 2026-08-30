"""
nemo_eval.agents.orchestrator
------------------------------
[W.O] Workflow Orchestration Engine.

Handles:
    - Tool selection: matching a sub-goal to the best available tool.
    - Parameter bridging: threading intermediate outputs from prior steps
      as inputs to the current step.
    - Execution chaining: dispatching tool invocations and collecting results.

Metrics:
    - Acc_tool: Tool selection accuracy (fraction of correct tool choices).
    - SPEA (Sub-goal Parameter Edge Accuracy): fraction of parameter bridges
      that successfully transfer data between consecutive sub-goals.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict

from nemo_eval.tools.schemas import ToolResult
from nemo_eval.tools.repl import PythonREPL
from nemo_eval.tools.sqlite_engine import SQLiteEngine, SQLiteEngineConfig
from nemo_eval.tools.tabular import TabularEngine


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """A resolved tool invocation request from the agent."""
    model_config = ConfigDict(extra="ignore")

    tool_name: str = Field(..., description="Tool identifier: 'python_repl', 'sqlite_query', 'sqlite_schema', 'tabular_inspect'.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Parsed arguments for the tool.")
    sub_goal_id: Optional[str] = Field(default=None, description="The sub-goal this tool call is serving.")
    raw_llm_tool_call: Optional[Dict[str, Any]] = Field(default=None)


class ToolDispatchResult(BaseModel):
    """Result of a single tool dispatch with bookkeeping for chaining."""
    model_config = ConfigDict(extra="ignore")

    sub_goal_id: str
    tool_name: str
    arguments: Dict[str, Any]
    result: ToolResult
    execution_time_ms: float
    is_valid_tool: bool = Field(description="True if the tool name was recognized.")
    bridge_applied: bool = Field(default=False, description="True if prior outputs were injected into arguments.")

    @property
    def succeeded(self) -> bool:
        return self.result.is_success


class OrchestratorConfig(BaseModel):
    """Configuration for ToolOrchestrator behavior."""
    model_config = ConfigDict(extra="ignore")

    max_dispatch_time_ms: float = Field(default=30_000.0, description="Per-tool execution wall-clock budget in ms.")
    auto_bridge_outputs: bool = Field(default=True, description="Automatically inject prior tool outputs into python_repl context.")
    sqlite_config: SQLiteEngineConfig = Field(default_factory=SQLiteEngineConfig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_KNOWN_TOOLS = frozenset({"python_repl", "sqlite_query", "sqlite_schema", "tabular_inspect"})

_HINT_TO_TOOL = {
    "python_repl": "python_repl",
    "sqlite_query": "sqlite_query",
    "tabular_inspect": "tabular_inspect",
    None: None,
}


class ToolOrchestrator:
    """
    Dispatches tool calls for each sub-goal and bridges intermediate outputs.

    Usage:
        orchestrator = ToolOrchestrator(config=OrchestratorConfig())
        result = orchestrator.dispatch(tool_call, prior_outputs)
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._repl = PythonREPL(default_timeout=self.config.max_dispatch_time_ms / 1000.0)
        self._tabular = TabularEngine()
        self._sqlite_engines: Dict[str, SQLiteEngine] = {}  # keyed by db_path
        self._tool_call_log: List[ToolDispatchResult] = []

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def dispatch(
        self,
        tool_call: ToolCall,
        prior_outputs: Optional[Dict[str, ToolResult]] = None,
    ) -> ToolDispatchResult:
        """
        Execute a single tool call, optionally bridging prior outputs into context.

        Args:
            tool_call: The resolved tool call to execute.
            prior_outputs: Mapping of sub_goal_id -> ToolResult from completed steps.

        Returns:
            ToolDispatchResult with execution metadata.
        """
        t0 = time.monotonic()
        prior_outputs = prior_outputs or {}
        is_valid = tool_call.tool_name in _KNOWN_TOOLS
        bridge_applied = False

        if not is_valid:
            result = ToolResult(
                status="error",
                execution_time_ms=0.0,
                error={
                    "error_type": "UnknownTool",
                    "message": f"Unknown tool '{tool_call.tool_name}'. Must be one of: {sorted(_KNOWN_TOOLS)}.",
                    "suggestion": "Check the tool_hint field in the sub-goal and use a supported tool name.",
                    "raw_traceback": "",
                },
            )
            dispatch_result = ToolDispatchResult(
                sub_goal_id=tool_call.sub_goal_id or "unknown",
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
                result=result,
                execution_time_ms=(time.monotonic() - t0) * 1000.0,
                is_valid_tool=False,
            )
            self._tool_call_log.append(dispatch_result)
            return dispatch_result

        args = dict(tool_call.arguments)

        # Parameter bridging: inject prior dataframe/scalar outputs into REPL context
        if self.config.auto_bridge_outputs and prior_outputs and tool_call.tool_name == "python_repl":
            args, bridge_applied = self._bridge_repl_context(args, prior_outputs)

        result = self._execute_tool(tool_call.tool_name, args)
        elapsed = (time.monotonic() - t0) * 1000.0

        dispatch_result = ToolDispatchResult(
            sub_goal_id=tool_call.sub_goal_id or "unknown",
            tool_name=tool_call.tool_name,
            arguments=args,
            result=result,
            execution_time_ms=elapsed,
            is_valid_tool=True,
            bridge_applied=bridge_applied,
        )
        self._tool_call_log.append(dispatch_result)
        return dispatch_result

    def select_tool(self, tool_hint: Optional[str], sub_goal_description: str) -> str:
        """
        Resolve a tool name from a hint, falling back to heuristics on description.

        Returns a guaranteed valid tool name from _KNOWN_TOOLS.
        """
        if tool_hint and tool_hint in _KNOWN_TOOLS:
            return tool_hint

        desc_lower = sub_goal_description.lower()
        if any(kw in desc_lower for kw in ("sql", "query", "table", "database", "select", "join", "where")):
            return "sqlite_query"
        if any(kw in desc_lower for kw in ("schema", "columns", "ddl", "structure")):
            return "sqlite_schema"
        if any(kw in desc_lower for kw in ("csv", "parquet", "dataframe", "inspect", "tabular", "column stat")):
            return "tabular_inspect"
        return "python_repl"

    def compute_tool_accuracy(self) -> Tuple[float, int]:
        """
        Compute Acc_tool across all dispatched calls.

        Returns:
            (accuracy, total_calls) — fraction of dispatches that used valid tools.
        """
        total = len(self._tool_call_log)
        if total == 0:
            return 1.0, 0
        valid = sum(1 for r in self._tool_call_log if r.is_valid_tool)
        return valid / total, total

    def compute_spea(self) -> float:
        """
        Compute SPEA (Sub-goal Parameter Edge Accuracy):
        Fraction of tool calls where parameter bridging was both attempted
        and the execution succeeded.

        Returns:
            SPEA score in [0, 1].
        """
        bridged = [r for r in self._tool_call_log if r.bridge_applied]
        if not bridged:
            return 1.0
        successful = sum(1 for r in bridged if r.succeeded)
        return successful / len(bridged)

    def reset(self) -> None:
        """Reset tool call log and REPL state for a new episode."""
        self._tool_call_log.clear()
        self._repl = PythonREPL(default_timeout=self.config.max_dispatch_time_ms / 1000.0)
        self._sqlite_engines.clear()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Route to the appropriate tool and return a ToolResult."""
        try:
            if tool_name == "python_repl":
                code = args.get("code", "")
                return self._repl.execute(code)

            elif tool_name == "sqlite_query":
                db_path = args.get("db_path", ":memory:")
                query = args.get("query", "")
                engine = self._get_sqlite_engine(db_path)
                return engine.execute_query(query)

            elif tool_name == "sqlite_schema":
                db_path = args.get("db_path", ":memory:")
                table_name = args.get("table_name")
                engine = self._get_sqlite_engine(db_path)
                return engine.get_schema(table_name=table_name)

            elif tool_name == "tabular_inspect":
                file_path = args.get("file_path", "")
                return self._tabular.inspect(file_path)

            else:
                return ToolResult(
                    status="error",
                    error={
                        "error_type": "UnknownTool",
                        "message": f"No executor for '{tool_name}'.",
                        "suggestion": "Use a known tool name.",
                        "raw_traceback": "",
                    },
                )
        except Exception as exc:
            return ToolResult(
                status="error",
                error={
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "suggestion": "An unexpected error occurred in tool dispatch.",
                    "raw_traceback": "",
                },
            )

    def _get_sqlite_engine(self, db_path: str) -> SQLiteEngine:
        """Return a cached or new SQLiteEngine for the given db_path."""
        if db_path not in self._sqlite_engines:
            cfg = SQLiteEngineConfig(db_path=db_path if db_path != ":memory:" else None)
            self._sqlite_engines[db_path] = SQLiteEngine(config=cfg)
        return self._sqlite_engines[db_path]

    def _bridge_repl_context(
        self, args: Dict[str, Any], prior_outputs: Dict[str, ToolResult]
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Inject successful prior ToolResult data into the python_repl code as
        pre-assigned Python variables (_sg_1_out, _sg_2_out, ...).

        Returns:
            (updated_args, bridge_applied)
        """
        bridge_snippets: List[str] = []
        for sg_id, prior_result in prior_outputs.items():
            if prior_result.is_success and prior_result.data is not None:
                var_name = f"_{sg_id.replace('-', '_')}_out"
                try:
                    import json as _json
                    repr_val = _json.dumps(prior_result.data, default=str)
                    bridge_snippets.append(f"{var_name} = {repr_val}")
                except Exception:
                    bridge_snippets.append(f"{var_name} = {repr(prior_result.data)}")

        if not bridge_snippets:
            return args, False

        preamble = "\n".join(bridge_snippets) + "\n\n"
        existing_code = args.get("code", "")
        args = {**args, "code": preamble + existing_code}
        return args, True
