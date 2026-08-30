"""
nemo_eval.agents
----------------
Core Agentic and Vanilla Capability Execution Engines (Milestone 3).

Exports:
    - BaseEvaluationEngine: Protocol for evaluation engines.
    - VanillaEngine: Pure zero-shot CoT evaluation engine (0 tools).
    - AgenticEngine: 9-State FSM multi-turn reasoning engine with Python REPL.
    - TaskPlanner: [T.D] Sub-goal DAG generation and topological scoring.
    - ToolOrchestrator: [W.O] Tool selection, parameter bridging, chaining.
    - AgentLoop: Multi-turn reasoning loop integrating model, tools, telemetry.
"""

from nemo_eval.agents.planner import (
    SubGoal,
    TaskPlan,
    PlannerConfig,
    TaskPlanner,
    PlanningMetrics,
)
from nemo_eval.agents.orchestrator import (
    ToolCall as OrchToolCall,
    ToolDispatchResult,
    OrchestratorConfig,
    ToolOrchestrator,
)
from nemo_eval.agents.vanilla import (
    BaseEvaluationEngine,
    VanillaEngine,
)
from nemo_eval.agents.agent_loop import (
    AgentConfig,
    AgentResult,
    AgentLoop,
    AgenticEngine,
)

__all__ = [
    # Engines
    "BaseEvaluationEngine",
    "VanillaEngine",
    "AgenticEngine",
    # Planner
    "SubGoal",
    "TaskPlan",
    "PlannerConfig",
    "TaskPlanner",
    "PlanningMetrics",
    # Orchestrator
    "OrchToolCall",
    "ToolDispatchResult",
    "OrchestratorConfig",
    "ToolOrchestrator",
    # Agent loop
    "AgentConfig",
    "AgentResult",
    "AgentLoop",
]
