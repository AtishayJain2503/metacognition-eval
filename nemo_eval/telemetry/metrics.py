"""
nemo_eval.telemetry.metrics
---------------------------
Plan Adherence Score (PAS) metric calculator.

PAS measures how faithfully the agent's executed tool call sequence
mirrors the intended execution order from the task plan.

Formula:
    PAS = (1 / N) * sum_i[ match(actual_state_i, planned_state_i) ]

Where:
    N = number of steps in the longer sequence (plan vs actual)
    match = 1 if actual sub_goal aligns with planned sub_goal at position i, else 0
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from nemo_eval.telemetry.tracer import EpisodeTrajectory, TrajectoryState


class PlanAdherenceScorer:
    """
    Computes Plan Adherence Score (PAS) from a trajectory and planned execution order.

    PAS = 1.0 → agent executed sub-goals in exactly the planned order.
    PAS = 0.0 → no alignment between plan and actual execution.
    """

    @staticmethod
    def score(
        trajectory: EpisodeTrajectory,
        planned_order: List[str],
    ) -> float:
        """
        Compute PAS for a closed episode trajectory.

        Args:
            trajectory: A finalized EpisodeTrajectory.
            planned_order: Ordered list of sub-goal IDs from TaskPlan.execution_order.

        Returns:
            PAS score in [0, 1].
        """
        if not planned_order:
            return 1.0

        # Extract actual sub_goal IDs executed from TOOL_EXECUTION steps
        actual_order: List[str] = []
        for step in trajectory.steps:
            if step.state == TrajectoryState.TOOL_EXECUTION:
                sg_id = step.input_payload.get("sub_goal_id")
                if sg_id:
                    actual_order.append(sg_id)

        if not actual_order:
            return 0.0

        # Compute longest common subsequence alignment
        matches = PlanAdherenceScorer._lcs_count(planned_order, actual_order)
        # Normalize by length of the planned order (penalize skipped sub-goals)
        return round(matches / len(planned_order), 4)

    @staticmethod
    def _lcs_count(seq_a: List[str], seq_b: List[str]) -> int:
        """Compute LCS length between two sequences."""
        m, n = len(seq_a), len(seq_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    @staticmethod
    def state_distribution(trajectory: EpisodeTrajectory) -> dict:
        """Return a count of time spent in each FSM state."""
        dist: dict = {}
        for step in trajectory.steps:
            key = step.state.value
            dist[key] = dist.get(key, 0) + 1
        return dist
