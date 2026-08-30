"""
nemo_eval.agents.planner
------------------------
[T.D] Task Decomposition and Planning Module.

Decomposes a complex user query into an ordered directed acyclic graph (DAG)
of sub-goals by prompting an LLM, then scores the quality of the decomposition
using structural, topological, and dependency metrics.

Metrics:
    - S_topo (Topological Score): Fraction of edges that respect topological order.
    - P_dep (Dependency Precision): Fraction of declared dependencies that
      are satisfied by prior nodes in the execution order.
    - S_struct (Structural Score): Composite measure of DAG balance, depth
      adequacy, and redundancy avoidance.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class SubGoal(BaseModel):
    """A single decomposed sub-task node in the planning DAG."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique identifier for this sub-goal, e.g. 'sg_1'.")
    description: str = Field(..., description="Natural language description of the sub-task.")
    tool_hint: Optional[str] = Field(
        default=None,
        description="Optional hint for which tool category to use: 'python_repl', 'sqlite_query', 'tabular_inspect', or None.",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of sub-goal IDs that must complete before this sub-goal.",
    )
    expected_output_type: Optional[str] = Field(
        default=None,
        description="Expected data type of the output, e.g. 'dataframe', 'scalar', 'sql_rows', 'string'.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_dependencies(self) -> bool:
        return len(self.depends_on) > 0


class PlanningMetrics(BaseModel):
    """Structural and topological quality metrics for a task plan DAG."""
    model_config = ConfigDict(extra="ignore")

    topological_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="S_topo: Fraction of dependency edges that respect topological order."
    )
    dependency_precision: float = Field(
        ..., ge=0.0, le=1.0,
        description="P_dep: Fraction of declared dependencies pointing to real, prior sub-goals."
    )
    structural_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="S_struct: Composite of DAG balance, depth adequacy, and edge coverage."
    )
    composite_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weighted composite of S_topo, P_dep, S_struct."
    )
    node_count: int = Field(..., ge=0)
    edge_count: int = Field(..., ge=0)
    max_depth: int = Field(..., ge=0)
    has_cycles: bool = Field(...)
    unreachable_nodes: List[str] = Field(default_factory=list)


class TaskPlan(BaseModel):
    """A complete task decomposition plan with DAG of sub-goals and quality metrics."""
    model_config = ConfigDict(extra="ignore")

    task_id: str
    original_query: str
    sub_goals: List[SubGoal] = Field(default_factory=list)
    execution_order: List[str] = Field(
        default_factory=list,
        description="Topologically sorted list of sub-goal IDs for sequential execution.",
    )
    metrics: Optional[PlanningMetrics] = None
    raw_llm_response: Optional[str] = None
    planning_duration_ms: float = 0.0

    @property
    def goal_map(self) -> Dict[str, SubGoal]:
        return {sg.id: sg for sg in self.sub_goals}

    def get_sub_goal(self, sg_id: str) -> Optional[SubGoal]:
        return self.goal_map.get(sg_id)

    def ordered_sub_goals(self) -> List[SubGoal]:
        gm = self.goal_map
        return [gm[sid] for sid in self.execution_order if sid in gm]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class PlannerConfig(BaseModel):
    """Configuration for TaskPlanner behavior."""
    model_config = ConfigDict(extra="ignore")

    max_sub_goals: int = Field(default=12, ge=2, le=30)
    require_tool_hints: bool = Field(default=True)
    system_prompt_override: Optional[str] = None
    topological_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    dependency_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    structural_weight: float = Field(default=0.25, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are a precise task decomposition engine for a data analytics and reasoning benchmark.

Given a complex analytical question, decompose it into a minimal, ordered sequence of
self-contained sub-tasks as a directed acyclic graph (DAG).

Rules:
1. Each sub-goal should map to exactly one atomic analytical operation.
2. Declare ALL dependencies (depends_on) explicitly — do not assume ordering.
3. Use tool_hint to specify: "python_repl", "sqlite_query", "tabular_inspect", or null.
4. Sub-goal IDs must be sequential: "sg_1", "sg_2", ..., "sg_N".
5. Minimize the number of sub-goals — aim for 3-8 for most queries.
6. expected_output_type should be one of: "scalar", "dataframe", "sql_rows", "string", "boolean", "list".

Output ONLY a valid JSON object with this structure:
{
  "sub_goals": [
    {
      "id": "sg_1",
      "description": "...",
      "tool_hint": "sqlite_query",
      "depends_on": [],
      "expected_output_type": "sql_rows"
    },
    ...
  ]
}

Do not include any text outside the JSON object.
"""


class TaskPlanner:
    """
    Decomposes a complex query into a structured sub-goal DAG and scores its quality.

    Workflow:
        1. Prompt the LLM to produce a JSON plan.
        2. Parse and validate the JSON into SubGoal objects.
        3. Perform topological sort and detect cycles.
        4. Compute PlanningMetrics (S_topo, P_dep, S_struct).
    """

    def __init__(self, model_client: Any, config: Optional[PlannerConfig] = None):
        """
        Args:
            model_client: An instance conforming to BaseLLMClient protocol
                          (must implement generate(messages, **kwargs) -> LLMResponse).
            config: Optional PlannerConfig. Defaults to PlannerConfig().
        """
        self.model = model_client
        self.config = config or PlannerConfig()

    def decompose(self, task_id: str, query: str) -> TaskPlan:
        """
        Decompose a query into a TaskPlan with scored sub-goal DAG.

        Args:
            task_id: Unique task identifier for telemetry/reporting.
            query: The complex analytical question to decompose.

        Returns:
            TaskPlan with sub_goals, execution_order, and metrics populated.
        """
        t0 = time.monotonic()
        system = self.config.system_prompt_override or _PLANNER_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Decompose this analytical task:\n\n{query}"},
        ]

        response = self.model.generate(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        raw = response.content or ""
        duration_ms = (time.monotonic() - t0) * 1000.0

        sub_goals = self._parse_sub_goals(raw)
        order, has_cycles, unreachable = self._topological_sort(sub_goals)
        metrics = self._compute_metrics(sub_goals, order, has_cycles, unreachable)

        return TaskPlan(
            task_id=task_id,
            original_query=query,
            sub_goals=sub_goals,
            execution_order=order,
            metrics=metrics,
            raw_llm_response=raw,
            planning_duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _parse_sub_goals(self, raw: str) -> List[SubGoal]:
        """Extract and parse sub_goals JSON from LLM response."""
        # Try to extract JSON object from response
        json_str = raw.strip()

        # Handle markdown code fences
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_str, re.DOTALL)
        if fence_match:
            json_str = fence_match.group(1)
        else:
            # Find the first { ... } block
            obj_match = re.search(r"\{.*\}", json_str, re.DOTALL)
            if obj_match:
                json_str = obj_match.group(0)

        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # Fallback: return a single catch-all sub-goal
            return [SubGoal(id="sg_1", description="Execute the complete task.", tool_hint=None, depends_on=[])]

        raw_goals = data.get("sub_goals", [])
        if not isinstance(raw_goals, list):
            return [SubGoal(id="sg_1", description="Execute the complete task.", tool_hint=None, depends_on=[])]

        goals: List[SubGoal] = []
        seen_ids = set()
        for item in raw_goals[: self.config.max_sub_goals]:
            if not isinstance(item, dict):
                continue
            sg_id = str(item.get("id", f"sg_{len(goals)+1}"))
            if sg_id in seen_ids:
                continue
            seen_ids.add(sg_id)
            goals.append(
                SubGoal(
                    id=sg_id,
                    description=str(item.get("description", "")).strip() or "Unnamed sub-goal",
                    tool_hint=item.get("tool_hint"),
                    depends_on=[str(d) for d in item.get("depends_on", []) if d],
                    expected_output_type=item.get("expected_output_type"),
                    metadata=item.get("metadata", {}),
                )
            )

        return goals if goals else [SubGoal(id="sg_1", description="Execute the complete task.", depends_on=[])]

    def _topological_sort(
        self, sub_goals: List[SubGoal]
    ) -> Tuple[List[str], bool, List[str]]:
        """
        Perform Kahn's algorithm topological sort.

        Returns:
            (order, has_cycles, unreachable_ids)
        """
        ids = {sg.id for sg in sub_goals}
        in_degree: Dict[str, int] = {sg.id: 0 for sg in sub_goals}
        adj: Dict[str, List[str]] = defaultdict(list)

        for sg in sub_goals:
            for dep in sg.depends_on:
                if dep in ids:
                    adj[dep].append(sg.id)
                    in_degree[sg.id] += 1

        queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
        order: List[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        has_cycles = len(order) < len(sub_goals)
        unreachable = [sid for sid in ids if sid not in set(order)]

        # If cycles exist, append unreachable nodes at the end to still have a plan
        if has_cycles:
            order.extend(unreachable)

        return order, has_cycles, unreachable

    def _compute_metrics(
        self,
        sub_goals: List[SubGoal],
        order: List[str],
        has_cycles: bool,
        unreachable: List[str],
    ) -> PlanningMetrics:
        """Compute S_topo, P_dep, S_struct, and composite score."""
        ids = {sg.id for sg in sub_goals}
        n = len(sub_goals)

        # Position map for topological order
        pos_map = {sid: i for i, sid in enumerate(order)}

        # S_topo: fraction of dependency edges where dep appears before node in order
        total_edges = sum(
            1
            for sg in sub_goals
            for dep in sg.depends_on
            if dep in ids
        )
        valid_topo_edges = sum(
            1
            for sg in sub_goals
            for dep in sg.depends_on
            if dep in ids and pos_map.get(dep, 999) < pos_map.get(sg.id, 999)
        )
        s_topo = (valid_topo_edges / total_edges) if total_edges > 0 else 1.0

        # P_dep: fraction of declared deps that reference real IDs
        total_declared = sum(len(sg.depends_on) for sg in sub_goals)
        valid_declared = sum(
            sum(1 for d in sg.depends_on if d in ids)
            for sg in sub_goals
        )
        p_dep = (valid_declared / total_declared) if total_declared > 0 else 1.0

        # S_struct: composite structural quality
        # - node count adequacy (ideal: 3-8)
        node_adequacy = 1.0 if 3 <= n <= 8 else max(0.0, 1.0 - abs(n - 5.5) / 10.0)
        # - depth: BFS from roots
        max_depth = self._compute_max_depth(sub_goals, ids)
        depth_score = min(1.0, max_depth / max(1, n - 1)) if n > 1 else 1.0
        # - no cycles bonus
        cycle_penalty = 0.3 if has_cycles else 0.0
        s_struct = max(0.0, (node_adequacy * 0.5 + depth_score * 0.5) - cycle_penalty)

        composite = (
            self.config.topological_weight * s_topo
            + self.config.dependency_weight * p_dep
            + self.config.structural_weight * s_struct
        )

        return PlanningMetrics(
            topological_score=round(s_topo, 4),
            dependency_precision=round(p_dep, 4),
            structural_score=round(s_struct, 4),
            composite_score=round(composite, 4),
            node_count=n,
            edge_count=total_edges,
            max_depth=max_depth,
            has_cycles=has_cycles,
            unreachable_nodes=unreachable,
        )

    def _compute_max_depth(self, sub_goals: List[SubGoal], ids: set) -> int:
        """BFS to compute maximum depth from any root node."""
        adj: Dict[str, List[str]] = defaultdict(list)
        for sg in sub_goals:
            for dep in sg.depends_on:
                if dep in ids:
                    adj[dep].append(sg.id)

        roots = [sg.id for sg in sub_goals if not sg.depends_on]
        if not roots:
            roots = [sub_goals[0].id] if sub_goals else []

        depth: Dict[str, int] = {r: 0 for r in roots}
        queue = deque(roots)
        max_d = 0

        while queue:
            node = queue.popleft()
            for child in adj[node]:
                if child not in depth:
                    depth[child] = depth[node] + 1
                    max_d = max(max_d, depth[child])
                    queue.append(child)

        return max_d
