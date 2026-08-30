"""
tests/unit/test_pipeline/test_runner.py
---------------------------------------
Unit tests for BenchmarkRunner and RunRecord (Milestone 4).
"""

import pytest
from pathlib import Path
from nemo_eval.pipeline.config import PipelineConfig, ModelSpec, DatasetSpec, ExecutionMode
from nemo_eval.pipeline.runner import BenchmarkRunner, RunRecord
from nemo_eval.telemetry.tracer import EpisodeTrajectory


class TestBenchmarkRunner:
    def test_run_record_metrics(self):
        record = RunRecord(model_name="Qwen2.5-Math-7B", dataset_name="math", mode="vanilla")
        t1 = EpisodeTrajectory(task_id="t1", model_name="Qwen2.5-Math-7B", status="success", ground_truth_score=1.0, total_duration_ms=100.0, peak_ram_mb=50.0, energy_joules=0.5)
        t2 = EpisodeTrajectory(task_id="t2", model_name="Qwen2.5-Math-7B", status="failed", ground_truth_score=0.0, total_duration_ms=200.0, peak_ram_mb=60.0, energy_joules=1.0)
        record.add_trajectory(t1)
        record.add_trajectory(t2)
        record.elapsed_ms = 300.0

        summary = record.summary()
        assert summary["model"] == "Qwen2.5-Math-7B"
        assert summary["tasks"] == 2
        assert summary["accuracy"] == 50.0
        assert summary["avg_duration_ms"] == 150.0
        assert summary["avg_peak_ram_mb"] == 55.0
        assert summary["avg_energy_joules"] == 0.75

    def test_runner_vanilla_mode_execution(self, tmp_path):
        config = PipelineConfig(
            run_label="unit_test_vanilla",
            output_dir=str(tmp_path / "results"),
            mode="vanilla",
            models=[ModelSpec(name="Qwen2.5-Math-1.5B", provider="mock")],
            datasets=[DatasetSpec(name="math", max_tasks=2)],
            export_jsonl=True
        )
        runner = BenchmarkRunner(config)
        records = runner.run()

        assert len(records) == 1
        rec = records[0]
        assert rec.mode == "vanilla"
        assert len(rec.trajectories) == 2
        for t in rec.trajectories:
            assert t.task_id.startswith("math_")
            assert t.peak_ram_mb > 0.0

    def test_runner_agentic_mode_execution(self, tmp_path):
        config = PipelineConfig(
            run_label="unit_test_agentic",
            output_dir=str(tmp_path / "results"),
            mode="agentic",
            models=[ModelSpec(name="DeepSeek-R1-1.5B", provider="mock")],
            datasets=[DatasetSpec(name="putnam", max_tasks=2)],
            export_jsonl=True
        )
        runner = BenchmarkRunner(config)
        records = runner.run()

        assert len(records) == 1
        rec = records[0]
        assert rec.mode == "agentic"
        assert len(rec.trajectories) == 2

    def test_runner_dual_parity_both_mode(self, tmp_path):
        config = PipelineConfig(
            run_label="unit_test_dual_parity",
            output_dir=str(tmp_path / "results"),
            mode="both",
            models=[ModelSpec(name="Phi4-mini-reasoning", provider="mock")],
            datasets=[DatasetSpec(name="lila", max_tasks=2)],
            export_jsonl=True
        )
        runner = BenchmarkRunner(config)
        records = runner.run()

        # Should produce 2 records: one vanilla, one agentic
        assert len(records) == 2
        modes = {r.mode for r in records}
        assert modes == {"vanilla", "agentic"}

        # Check identical task ids
        v_rec = next(r for r in records if r.mode == "vanilla")
        a_rec = next(r for r in records if r.mode == "agentic")
        assert v_rec.task_ids == a_rec.task_ids
