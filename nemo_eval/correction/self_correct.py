"""
nemo_eval.correction.self_correct
----------------------------------
Self-correction metrics: SCSR, CEI, TOP.

Metrics:
    SCSR (Self-Correction Success Rate):
        Fraction of self-correction attempts that led to a successful recovery
        on the same sub-goal.

    CEI (Correction Efficiency Index):
        Success rate weighted by the inverse of attempts used.
        CEI = successful_corrections / total_attempts
        (penalizes many retries before recovery)

    TOP (Turn Overhead Penalty):
        Extra turns consumed by self-correction as a fraction of max_turns.
        TOP = correction_turns / max_turns
        Lower is better (0.0 = no correction overhead).
"""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, ConfigDict

from nemo_eval.telemetry.tracer import EpisodeTrajectory, TrajectoryState


class CorrectionStats(BaseModel):
    """Per-episode correction statistics."""
    model_config = ConfigDict(extra="ignore")

    total_attempts: int = Field(default=0, ge=0)
    successful_recoveries: int = Field(default=0, ge=0)
    scsr: float = Field(default=1.0, ge=0.0, le=1.0, description="Self-Correction Success Rate.")
    cei: float = Field(default=1.0, ge=0.0, le=1.0, description="Correction Efficiency Index.")
    top: float = Field(default=0.0, ge=0.0, description="Turn Overhead Penalty (fraction of max_turns).")
    correction_turns: int = Field(default=0, ge=0, description="Number of turns spent in SELF_CORRECTION state.")


class SelfCorrectMetrics:
    """
    Computes SCSR, CEI, and TOP from a closed EpisodeTrajectory.

    Usage:
        stats = SelfCorrectMetrics.compute(trajectory, max_turns=25)
    """

    @staticmethod
    def compute(trajectory: EpisodeTrajectory, max_turns: int = 25) -> CorrectionStats:
        """
        Compute self-correction metrics from trajectory steps.

        Args:
            trajectory: A finalized EpisodeTrajectory.
            max_turns: Maximum allowed turns (for TOP normalization).

        Returns:
            CorrectionStats with all metrics populated.
        """
        steps = trajectory.steps
        total_attempts = trajectory.self_correction_attempts
        correction_turns = sum(
            1 for s in steps if s.state == TrajectoryState.SELF_CORRECTION
        )

        # Count successful recoveries:
        # A recovery is successful if a SELF_CORRECTION step is followed eventually
        # by an OBSERVATION step with tool_valid=1.0 before the next SELF_CORRECTION or TERMINAL.
        successful_recoveries = SelfCorrectMetrics._count_successful_recoveries(steps)

        scsr = successful_recoveries / total_attempts if total_attempts > 0 else 1.0
        cei = successful_recoveries / max(correction_turns, 1) if correction_turns > 0 else 1.0
        top = correction_turns / max_turns if max_turns > 0 else 0.0

        return CorrectionStats(
            total_attempts=total_attempts,
            successful_recoveries=successful_recoveries,
            scsr=round(min(scsr, 1.0), 4),
            cei=round(min(cei, 1.0), 4),
            top=round(top, 4),
            correction_turns=correction_turns,
        )

    @staticmethod
    def _count_successful_recoveries(steps: list) -> int:
        """
        Count the number of SELF_CORRECTION windows that ended in a successful OBSERVATION.
        """
        in_correction = False
        count = 0
        for step in steps:
            if step.state == TrajectoryState.SELF_CORRECTION:
                in_correction = True
            elif in_correction:
                if step.state == TrajectoryState.OBSERVATION:
                    tool_valid = step.metrics.get("tool_valid", 0.0)
                    if tool_valid >= 1.0:
                        count += 1
                        in_correction = False
                elif step.state in (TrajectoryState.TERMINAL_SUCCESS, TrajectoryState.TERMINAL_FAILURE):
                    in_correction = False
        return count

    @staticmethod
    def aggregate(stats_list: List[CorrectionStats]) -> CorrectionStats:
        """Aggregate CorrectionStats across multiple episodes."""
        if not stats_list:
            return CorrectionStats()
        n = len(stats_list)
        return CorrectionStats(
            total_attempts=sum(s.total_attempts for s in stats_list),
            successful_recoveries=sum(s.successful_recoveries for s in stats_list),
            scsr=round(sum(s.scsr for s in stats_list) / n, 4),
            cei=round(sum(s.cei for s in stats_list) / n, 4),
            top=round(sum(s.top for s in stats_list) / n, 4),
            correction_turns=sum(s.correction_turns for s in stats_list),
        )
