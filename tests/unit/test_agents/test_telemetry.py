"""
Unit tests for nemo_eval.telemetry (Milestone 4 — Trajectory FSM + PAS).
"""

import pytest
import time

from nemo_eval.telemetry.tracer import (
    TrajectoryState,
    StepEvent,
    EpisodeTrajectory,
    TrajectoryTracer,
)
from nemo_eval.telemetry.metrics import PlanAdherenceScorer
from nemo_eval.telemetry.exporters import TelemetryExporter


# ---------------------------------------------------------------------------
# TrajectoryTracer tests
# ---------------------------------------------------------------------------

class TestTrajectoryTracer:

    def _make_tracer(self) -> TrajectoryTracer:
        t = TrajectoryTracer(task_id="test_task", model_name="mock")
        t.begin_episode()
        return t

    def test_begin_episode_clears_state(self):
        t = self._make_tracer()
        assert t.current_state is None
        assert t.step_count == 0

    def test_transition_basic(self):
        t = self._make_tracer()
        event = t.transition(TrajectoryState.PLANNING)
        assert event.state == TrajectoryState.PLANNING
        assert t.current_state == TrajectoryState.PLANNING

    def test_transition_increments_step(self):
        t = self._make_tracer()
        t.transition(TrajectoryState.PLANNING)
        t.transition(TrajectoryState.ACTION_SELECTION)
        assert t.step_count == 2

    def test_legal_transition_not_flagged(self):
        t = self._make_tracer()
        t.transition(TrajectoryState.PLANNING)
        event = t.transition(TrajectoryState.ACTION_SELECTION)
        assert event.invalid_transition is False

    def test_illegal_transition_flagged(self):
        t = self._make_tracer()
        t.transition(TrajectoryState.PLANNING)
        # Jumping from PLANNING directly to TOOL_EXECUTION is illegal
        event = t.transition(TrajectoryState.TOOL_EXECUTION)
        assert event.invalid_transition is True

    def test_illegal_transition_counted(self):
        t = self._make_tracer()
        t.transition(TrajectoryState.PLANNING)
        t.transition(TrajectoryState.TOOL_EXECUTION)  # illegal
        trajectory = t.close_episode(status="failed")
        assert trajectory.invalid_transitions == 1

    def test_self_correction_counted(self):
        t = self._make_tracer()
        t.transition(TrajectoryState.PLANNING)
        t.transition(TrajectoryState.ACTION_SELECTION)
        t.transition(TrajectoryState.TOOL_EXECUTION)
        t.transition(TrajectoryState.OBSERVATION)
        t.transition(TrajectoryState.VERIFICATION)
        t.transition(TrajectoryState.SELF_CORRECTION)
        assert t._self_correction_attempts == 1

    def test_is_terminal_false_initially(self):
        t = self._make_tracer()
        t.transition(TrajectoryState.PLANNING)
        assert t.is_terminal() is False

    def test_is_terminal_success(self):
        t = self._make_tracer()
        t.transition(TrajectoryState.PLANNING)
        t.transition(TrajectoryState.ACTION_SELECTION)
        t.transition(TrajectoryState.FINAL_SYNTHESIS)
        t.transition(TrajectoryState.TERMINAL_SUCCESS)
        assert t.is_terminal() is True

    def test_close_episode_returns_trajectory(self):
        t = self._make_tracer()
        t.transition(TrajectoryState.PLANNING)
        t.transition(TrajectoryState.ACTION_SELECTION)
        t.transition(TrajectoryState.FINAL_SYNTHESIS)
        t.transition(TrajectoryState.TERMINAL_SUCCESS)
        traj = t.close_episode(status="success", final_answer="42")
        assert isinstance(traj, EpisodeTrajectory)
        assert traj.status == "success"
        assert traj.final_answer == "42"

    def test_step_event_payload(self):
        t = self._make_tracer()
        event = t.transition(
            TrajectoryState.PLANNING,
            input_payload={"query": "test"},
            output_payload={"plan_size": 3},
            metrics={"plan_score": 0.9},
        )
        assert event.input_payload["query"] == "test"
        assert event.output_payload["plan_size"] == 3
        assert event.metrics["plan_score"] == 0.9

    def test_total_duration_positive(self):
        t = self._make_tracer()
        t.transition(TrajectoryState.PLANNING)
        traj = t.close_episode(status="failed")
        assert traj.total_duration_ms >= 0.0

    def test_state_sequence(self):
        t = self._make_tracer()
        t.transition(TrajectoryState.PLANNING)
        t.transition(TrajectoryState.ACTION_SELECTION)
        t.transition(TrajectoryState.TOOL_EXECUTION)
        traj = t.close_episode(status="failed")
        assert traj.state_sequence() == ["PLANNING", "ACTION_SELECTION", "TOOL_EXECUTION"]


# ---------------------------------------------------------------------------
# PlanAdherenceScorer tests
# ---------------------------------------------------------------------------

class TestPlanAdherenceScorer:

    def _make_trajectory(self, tool_exec_sg_ids: list) -> EpisodeTrajectory:
        """Build a trajectory with TOOL_EXECUTION steps for given sub_goal_ids."""
        steps = []
        for i, sg_id in enumerate(tool_exec_sg_ids):
            steps.append(StepEvent(
                step_id=i,
                state=TrajectoryState.TOOL_EXECUTION,
                timestamp=time.time(),
                duration_ms=10.0,
                input_payload={"sub_goal_id": sg_id},
            ))
        return EpisodeTrajectory(
            task_id="test", model_name="mock", steps=steps, status="success"
        )

    def test_perfect_adherence(self):
        plan = ["sg_1", "sg_2", "sg_3"]
        traj = self._make_trajectory(["sg_1", "sg_2", "sg_3"])
        pas = PlanAdherenceScorer.score(traj, plan)
        assert pas == 1.0

    def test_partial_adherence(self):
        plan = ["sg_1", "sg_2", "sg_3", "sg_4"]
        traj = self._make_trajectory(["sg_1", "sg_3"])  # skipped sg_2, sg_4
        pas = PlanAdherenceScorer.score(traj, plan)
        assert 0.0 < pas < 1.0

    def test_empty_plan_returns_1(self):
        traj = self._make_trajectory(["sg_1"])
        pas = PlanAdherenceScorer.score(traj, [])
        assert pas == 1.0

    def test_empty_actual_returns_0(self):
        traj = EpisodeTrajectory(task_id="t", model_name="m", steps=[], status="failed")
        pas = PlanAdherenceScorer.score(traj, ["sg_1", "sg_2"])
        assert pas == 0.0

    def test_reordered_gives_lower_score(self):
        plan = ["sg_1", "sg_2", "sg_3"]
        traj_correct = self._make_trajectory(["sg_1", "sg_2", "sg_3"])
        traj_reversed = self._make_trajectory(["sg_3", "sg_2", "sg_1"])
        pas_correct = PlanAdherenceScorer.score(traj_correct, plan)
        pas_reversed = PlanAdherenceScorer.score(traj_reversed, plan)
        # LCS of reversed vs forward chain = 1 (only one element in LCS)
        assert pas_correct > pas_reversed

    def test_state_distribution(self):
        t = TrajectoryTracer(task_id="t", model_name="m")
        t.begin_episode()
        t.transition(TrajectoryState.PLANNING)
        t.transition(TrajectoryState.ACTION_SELECTION)
        t.transition(TrajectoryState.ACTION_SELECTION)
        traj = t.close_episode(status="failed")
        dist = PlanAdherenceScorer.state_distribution(traj)
        assert dist["PLANNING"] == 1
        assert dist["ACTION_SELECTION"] == 2


# ---------------------------------------------------------------------------
# TelemetryExporter tests
# ---------------------------------------------------------------------------

class TestTelemetryExporter:

    def test_append_jsonl(self, tmp_path):
        exporter = TelemetryExporter(output_dir=tmp_path)
        t = TrajectoryTracer(task_id="export_test", model_name="mock")
        t.begin_episode()
        t.transition(TrajectoryState.PLANNING)
        traj = t.close_episode(status="failed")
        out = exporter.append_jsonl(traj, filename="test.jsonl")
        assert out.exists()
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 1
        import json
        data = json.loads(lines[0])
        assert data["task_id"] == "export_test"

    def test_write_markdown_scorecard(self, tmp_path):
        exporter = TelemetryExporter(output_dir=tmp_path)
        t = TrajectoryTracer(task_id="sc_test", model_name="mock")
        t.begin_episode()
        t.transition(TrajectoryState.PLANNING)
        traj = t.close_episode(status="success")
        out = exporter.write_markdown_scorecard([traj], run_label="Test Run")
        assert out.exists()
        content = out.read_text()
        assert "Test Run" in content
        assert "sc_test" in content

    def test_empty_scorecard(self, tmp_path):
        exporter = TelemetryExporter(output_dir=tmp_path)
        out = exporter.write_markdown_scorecard([], filename="empty.md")
        assert out.exists()

    def test_otlp_spans(self, tmp_path):
        exporter = TelemetryExporter(output_dir=tmp_path)
        t = TrajectoryTracer(task_id="span_test", model_name="mock")
        t.begin_episode()
        t.transition(TrajectoryState.PLANNING)
        traj = t.close_episode(status="failed")
        out = exporter.write_otlp_spans(traj)
        assert out.exists()
        import json
        data = json.loads(out.read_text())
        assert "spans" in data
        assert len(data["spans"]) == 1
