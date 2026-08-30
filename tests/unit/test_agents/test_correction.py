"""
Unit tests for nemo_eval.correction (Milestone 5 — Verification & Self-Correction).
"""

import pytest
import math

from nemo_eval.correction.verifier import IntermediateVerifier, VerificationResult
from nemo_eval.correction.self_correct import SelfCorrectMetrics, CorrectionStats
from nemo_eval.telemetry.tracer import TrajectoryTracer, TrajectoryState, EpisodeTrajectory, StepEvent
import time


# ---------------------------------------------------------------------------
# IntermediateVerifier
# ---------------------------------------------------------------------------

class TestIntermediateVerifier:
    """Tests for the 5-check IntermediateVerifier."""

    def setup_method(self):
        self.v = IntermediateVerifier()

    def test_non_empty_passes_list(self):
        results = self.v.verify([1, 2, 3])
        assert all(r.passed for r in results)

    def test_non_empty_fails_none(self):
        results = self.v.verify(None)
        assert results[0].passed is False
        assert results[0].check_name == "non_empty"

    def test_non_empty_fails_empty_list(self):
        results = self.v.verify([])
        assert results[0].passed is False

    def test_non_empty_fails_empty_dict(self):
        results = self.v.verify({})
        assert results[0].passed is False

    def test_type_check_list(self):
        results = self.v.verify([1, 2], expected_type="list")
        assert all(r.passed for r in results)

    def test_type_check_scalar_int(self):
        results = self.v.verify(42, expected_type="scalar")
        assert all(r.passed for r in results)

    def test_type_check_fails_wrong_type(self):
        results = self.v.verify("hello", expected_type="list")
        type_result = next(r for r in results if r.check_name == "type_match")
        assert type_result.passed is False

    def test_type_check_dataframe_proxy(self):
        data = [{"col_a": 1, "col_b": 2}, {"col_a": 3, "col_b": 4}]
        results = self.v.verify(data, expected_type="dataframe")
        type_result = next(r for r in results if r.check_name == "type_match")
        assert type_result.passed is True

    def test_schema_check_passes(self):
        data = [{"revenue": 100, "region": "North", "extra": "ignored"}]
        results = self.v.verify(data, expected_schema=["revenue", "region"])
        schema_result = next(r for r in results if r.check_name == "schema_match")
        assert schema_result.passed is True

    def test_schema_check_fails_missing_column(self):
        data = [{"revenue": 100}]
        results = self.v.verify(data, expected_schema=["revenue", "region"])
        schema_result = next(r for r in results if r.check_name == "schema_match")
        assert schema_result.passed is False
        assert "region" in schema_result.detail

    def test_numeric_bounds_pass(self):
        results = self.v.verify([1.0, 2.5, 3.9], numeric_bounds=(0.0, 5.0))
        bounds_result = next(r for r in results if r.check_name == "numeric_bounds")
        assert bounds_result.passed is True

    def test_numeric_bounds_fail(self):
        results = self.v.verify([1.0, 999.0], numeric_bounds=(0.0, 100.0))
        bounds_result = next(r for r in results if r.check_name == "numeric_bounds")
        assert bounds_result.passed is False
        assert "999" in bounds_result.detail

    def test_no_nan_passes_clean_data(self):
        results = self.v.verify([1.0, 2.0, 3.0], strict_no_nan=True)
        nan_result = next(r for r in results if r.check_name == "no_nan")
        assert nan_result.passed is True

    def test_no_nan_fails_on_nan(self):
        results = self.v.verify([1.0, float("nan"), 3.0], strict_no_nan=True)
        nan_result = next(r for r in results if r.check_name == "no_nan")
        assert nan_result.passed is False

    def test_no_nan_fails_on_none_in_list(self):
        results = self.v.verify([1, None, 3], strict_no_nan=True)
        nan_result = next(r for r in results if r.check_name == "no_nan")
        assert nan_result.passed is False

    def test_short_circuit_on_empty(self):
        """When non_empty fails, no other checks should run."""
        results = self.v.verify(None, expected_type="list", strict_no_nan=True)
        assert len(results) == 1  # Only non_empty check ran

    def test_all_passed_utility(self):
        results = self.v.verify([1, 2, 3], expected_type="list")
        assert self.v.all_passed(results) is True

    def test_all_passed_false(self):
        results = self.v.verify([1, 2], expected_type="dict")
        assert self.v.all_passed(results) is False

    def test_nested_dict_nan_detection(self):
        data = {"key": {"nested": float("nan")}}
        results = self.v.verify(data, strict_no_nan=True)
        nan_result = next(r for r in results if r.check_name == "no_nan")
        assert nan_result.passed is False


# ---------------------------------------------------------------------------
# SelfCorrectMetrics
# ---------------------------------------------------------------------------

def _make_trajectory_with_corrections(
    correction_count: int,
    recovery_success: bool,
    max_turns: int = 25,
) -> EpisodeTrajectory:
    """Build a synthetic trajectory with self-correction cycles."""
    steps = []
    steps.append(StepEvent(
        step_id=0, state=TrajectoryState.PLANNING,
        timestamp=time.time(), duration_ms=10.0,
        input_payload={}, output_payload={},
    ))
    for i in range(correction_count):
        # ACTION → TOOL → OBSERVATION (error) → VERIFICATION → SELF_CORRECTION
        steps.append(StepEvent(
            step_id=len(steps), state=TrajectoryState.ACTION_SELECTION,
            timestamp=time.time(), duration_ms=5.0,
            input_payload={}, output_payload={},
        ))
        steps.append(StepEvent(
            step_id=len(steps), state=TrajectoryState.TOOL_EXECUTION,
            timestamp=time.time(), duration_ms=5.0,
            input_payload={}, output_payload={},
        ))
        steps.append(StepEvent(
            step_id=len(steps), state=TrajectoryState.OBSERVATION,
            timestamp=time.time(), duration_ms=2.0,
            input_payload={}, output_payload={},
            metrics={"tool_valid": 0.0},  # failed
        ))
        steps.append(StepEvent(
            step_id=len(steps), state=TrajectoryState.SELF_CORRECTION,
            timestamp=time.time(), duration_ms=3.0,
            input_payload={}, output_payload={},
        ))

    if recovery_success:
        # Final successful OBSERVATION after last self-correction
        steps.append(StepEvent(
            step_id=len(steps), state=TrajectoryState.ACTION_SELECTION,
            timestamp=time.time(), duration_ms=5.0,
            input_payload={}, output_payload={},
        ))
        steps.append(StepEvent(
            step_id=len(steps), state=TrajectoryState.TOOL_EXECUTION,
            timestamp=time.time(), duration_ms=5.0,
            input_payload={}, output_payload={},
        ))
        steps.append(StepEvent(
            step_id=len(steps), state=TrajectoryState.OBSERVATION,
            timestamp=time.time(), duration_ms=2.0,
            input_payload={}, output_payload={},
            metrics={"tool_valid": 1.0},  # success
        ))

    return EpisodeTrajectory(
        task_id="test", model_name="mock",
        steps=steps, status="success" if recovery_success else "failed",
        self_correction_attempts=correction_count,
    )


class TestSelfCorrectMetrics:

    def test_no_corrections_perfect_scores(self):
        traj = EpisodeTrajectory(
            task_id="t", model_name="m",
            steps=[], status="success",
            self_correction_attempts=0,
        )
        stats = SelfCorrectMetrics.compute(traj, max_turns=25)
        assert stats.scsr == 1.0
        assert stats.top == 0.0

    def test_one_successful_correction(self):
        traj = _make_trajectory_with_corrections(1, recovery_success=True)
        stats = SelfCorrectMetrics.compute(traj, max_turns=25)
        assert stats.total_attempts == 1
        assert stats.successful_recoveries == 1
        assert stats.scsr == 1.0

    def test_one_failed_correction(self):
        traj = _make_trajectory_with_corrections(1, recovery_success=False)
        stats = SelfCorrectMetrics.compute(traj, max_turns=25)
        assert stats.total_attempts == 1
        assert stats.successful_recoveries == 0
        assert stats.scsr == 0.0

    def test_correction_turns_counted(self):
        traj = _make_trajectory_with_corrections(2, recovery_success=True)
        stats = SelfCorrectMetrics.compute(traj, max_turns=25)
        assert stats.correction_turns == 2

    def test_top_penalty_proportional(self):
        traj = _make_trajectory_with_corrections(5, recovery_success=False)
        stats = SelfCorrectMetrics.compute(traj, max_turns=25)
        assert stats.top == pytest.approx(5 / 25, abs=1e-4)

    def test_aggregate_empty(self):
        agg = SelfCorrectMetrics.aggregate([])
        assert agg.total_attempts == 0

    def test_aggregate_multiple(self):
        s1 = CorrectionStats(total_attempts=2, successful_recoveries=2, scsr=1.0, cei=1.0, top=0.1, correction_turns=2)
        s2 = CorrectionStats(total_attempts=1, successful_recoveries=0, scsr=0.0, cei=0.0, top=0.2, correction_turns=1)
        agg = SelfCorrectMetrics.aggregate([s1, s2])
        assert agg.total_attempts == 3
        assert agg.scsr == pytest.approx(0.5, abs=1e-4)
