"""
tests/unit/test_pipeline/test_reporter.py
-----------------------------------------
Unit tests for MarkdownReporter and PipelineReporter (Milestone 4).
"""

import json
import pytest
from pathlib import Path
from nemo_eval.pipeline.reporter import MarkdownReporter, PipelineReporter
from nemo_eval.pipeline.runner import RunRecord
from nemo_eval.telemetry.tracer import EpisodeTrajectory


class TestMarkdownReporter:
    def test_markdown_scorecard_generation(self):
        trajectories = [
            EpisodeTrajectory(task_id="t1", model_name="Qwen2.5-Math-7B", status="success", ground_truth_score=1.0, total_duration_ms=120.0, peak_ram_mb=45.0, energy_joules=0.5),
            EpisodeTrajectory(task_id="t2", model_name="Qwen2.5-Math-7B", status="failed", ground_truth_score=0.0, total_duration_ms=150.0, peak_ram_mb=46.0, energy_joules=0.6)
        ]
        card = MarkdownReporter.generate_markdown_scorecard(trajectories)
        assert "# Metacognition Evaluation Benchmark Scorecard" in card
        assert "50.00%" in card
        assert "t1" in card
        assert "t2" in card

    def test_dual_mode_comparison_table(self):
        v_traces = [EpisodeTrajectory(task_id="t1", model_name="M", status="success", ground_truth_score=1.0, total_duration_ms=50.0, energy_joules=0.2)]
        a_traces = [EpisodeTrajectory(task_id="t1", model_name="M", status="success", ground_truth_score=1.0, total_duration_ms=200.0, energy_joules=0.8)]
        comp = MarkdownReporter.generate_dual_mode_comparison(v_traces, a_traces)
        assert "Dual-Mode Parity" in comp
        assert "Vanilla (Zero-Shot)" in comp
        assert "Agentic (9-State FSM)" in comp
        assert "Delta (Agentic - Vanilla)" in comp

    def test_write_summary_report_and_json(self, tmp_path):
        reporter = MarkdownReporter(output_dir=tmp_path)
        r1 = RunRecord("Qwen2.5-Math-7B", "math", mode="vanilla")
        r1.add_trajectory(EpisodeTrajectory(task_id="t1", model_name="Qwen2.5-Math-7B", status="success", ground_truth_score=1.0, total_duration_ms=100.0, peak_ram_mb=50.0, energy_joules=0.5))
        r2 = RunRecord("Qwen2.5-Math-7B", "math", mode="agentic")
        r2.add_trajectory(EpisodeTrajectory(task_id="t1", model_name="Qwen2.5-Math-7B", status="success", ground_truth_score=1.0, total_duration_ms=200.0, peak_ram_mb=55.0, energy_joules=1.0))

        scorecard_file = reporter.write_summary_report([r1, r2], filename="summary_scorecard.md")
        json_file = reporter.write_json_summary([r1, r2], filename="summary.json")
        failures_file = reporter.write_failure_traces([r1, r2], filename="failure_traces.md")

        assert scorecard_file.exists()
        assert json_file.exists()
        assert failures_file.exists()

        scorecard_content = scorecard_file.read_text(encoding="utf-8")
        assert "Master Scorecard" in scorecard_content
        assert "Dual-Mode Parity Analysis" in scorecard_content
        assert "Leaderboard" in scorecard_content

        json_data = json.loads(json_file.read_text(encoding="utf-8"))
        assert len(json_data) == 2
        assert json_data[0]["model"] == "Qwen2.5-Math-7B"
