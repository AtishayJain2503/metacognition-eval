"""
nemo_eval.agents
----------------
Core Agentic Capability Execution Engine (Milestone 4).

Exports:
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
from nemo_eval.agents.agent_loop import (
    AgentConfig,
    AgentResult,
    AgentLoop,
)

__all__ = [
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
