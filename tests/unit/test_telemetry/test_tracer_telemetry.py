"""
Unit tests for nemo_eval.telemetry.tracer and exporters telemetry integration.
"""

import json
import pytest
from pathlib import Path

from nemo_eval.telemetry.tracer import (
    TrajectoryState,
    StepEvent,
    EpisodeTrajectory,
    TrajectoryTracer,
)
from nemo_eval.telemetry.monitor import HardwareMetrics
from nemo_eval.telemetry.exporters import TelemetryExporter


class TestTracerTelemetryIntegration:
    """Test StepEvent, EpisodeTrajectory, and TrajectoryTracer hardware telemetry."""

    def test_step_event_telemetry_fields(self):
        event = StepEvent(
            step_id=0,
            state=TrajectoryState.PLANNING,
            timestamp=1000.0,
            duration_ms=50.0,
            peak_ram_mb=64.5,
            gpu_vram_mb=1024.0,
            gpu_power_watts=45.0,
            energy_joules=2.25,
        )
        assert event.peak_ram_mb == 64.5
        assert event.gpu_vram_mb == 1024.0
        assert event.gpu_power_watts == 45.0
        assert event.energy_joules == 2.25

    def test_episode_trajectory_telemetry_fields(self):
        traj = EpisodeTrajectory(
            task_id="t1",
            model_name="Qwen2.5-Math-7B",
            status="success",
            peak_ram_mb=128.0,
            gpu_vram_mb=2048.0,
            gpu_power_watts=80.0,
            energy_joules=5.5,
            gpu_available=True,
        )
        assert traj.peak_ram_mb == 128.0
        assert traj.gpu_vram_mb == 2048.0
        assert traj.energy_joules == 5.5
        assert traj.gpu_available is True

    def test_tracer_transition_with_hardware_metrics(self):
        tracer = TrajectoryTracer("task_test", "mock_model", enable_telemetry=False)
        tracer.begin_episode()

        hw_step = HardwareMetrics(
            duration_ms=25.0,
            peak_ram_mb=50.0,
            gpu_vram_mb=500.0,
            gpu_power_watts=30.0,
            energy_joules=0.75,
            gpu_available=True,
        )

        event = tracer.transition(
            TrajectoryState.PLANNING,
            input_payload={"goal": "test"},
            hardware_metrics=hw_step,
        )

        assert event.peak_ram_mb == 50.0
        assert event.gpu_vram_mb == 500.0
        assert event.energy_joules == 0.75

    def test_tracer_close_episode_aggregates_telemetry(self):
        tracer = TrajectoryTracer("task_agg", "mock_model", enable_telemetry=False)
        tracer.begin_episode()

        tracer.transition(
            TrajectoryState.PLANNING,
            hardware_metrics=HardwareMetrics(peak_ram_mb=40.0, energy_joules=1.0, gpu_vram_mb=100.0)
        )
        tracer.transition(
            TrajectoryState.ACTION_SELECTION,
            hardware_metrics=HardwareMetrics(peak_ram_mb=60.0, energy_joules=2.0, gpu_vram_mb=150.0)
        )
        tracer.transition(
            TrajectoryState.FINAL_SYNTHESIS,
            hardware_metrics=HardwareMetrics(peak_ram_mb=55.0, energy_joules=0.5, gpu_vram_mb=120.0)
        )

        traj = tracer.close_episode(status="success", final_answer="42", ground_truth_score=1.0)
        assert traj.peak_ram_mb == 60.0  # max(40, 60, 55)
        assert traj.gpu_vram_mb == 150.0  # max(100, 150, 120)
        assert traj.energy_joules == 3.5  # sum(1.0 + 2.0 + 0.5)

    def test_telemetry_exporter_jsonl_roundtrip(self, tmp_path):
        exporter = TelemetryExporter(output_dir=tmp_path)
        traj = EpisodeTrajectory(
            task_id="jsonl_task",
            model_name="DeepSeek-R1-7B",
            status="success",
            peak_ram_mb=75.5,
            gpu_vram_mb=1200.0,
            energy_joules=3.14,
            gpu_available=True,
        )
        out_file = exporter.append_jsonl(traj, filename="test_traces.jsonl")
        assert out_file.exists()

        line = out_file.read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert data["task_id"] == "jsonl_task"
        assert data["peak_ram_mb"] == 75.5
        assert data["gpu_vram_mb"] == 1200.0
        assert data["energy_joules"] == 3.14

    def test_markdown_scorecard_formats_telemetry_columns(self, tmp_path):
        exporter = TelemetryExporter(output_dir=tmp_path)
        trajectories = [
            EpisodeTrajectory(
                task_id="t1",
                model_name="ModelA",
                status="success",
                ground_truth_score=1.0,
                peak_ram_mb=50.0,
                gpu_vram_mb=100.0,
                energy_joules=0.5,
            ),
            EpisodeTrajectory(
                task_id="t2",
                model_name="ModelB",
                status="failed",
                ground_truth_score=0.0,
                peak_ram_mb=70.0,
                gpu_vram_mb=150.0,
                energy_joules=1.0,
            ),
        ]
        out_file = exporter.write_markdown_scorecard(trajectories, filename="test_scorecard.md")
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "Peak RAM (MB)" in content
        assert "GPU VRAM (MB)" in content
        assert "Energy (J)" in content
        assert "Avg Peak RAM (MB)" in content
        assert "Total Energy (Joules)" in content

    def test_otlp_spans_includes_telemetry_attributes(self, tmp_path):
        exporter = TelemetryExporter(output_dir=tmp_path)
        tracer = TrajectoryTracer("task_span", "mock_model", enable_telemetry=False)
        tracer.begin_episode()
        tracer.transition(
            TrajectoryState.PLANNING,
            hardware_metrics=HardwareMetrics(peak_ram_mb=42.0, gpu_vram_mb=512.0, energy_joules=0.88)
        )
        traj = tracer.close_episode(status="success")

        span_file = exporter.write_otlp_spans(traj)
        assert span_file.exists()
        span_data = json.loads(span_file.read_text(encoding="utf-8"))
        assert "spans" in span_data
        attrs = {a["key"]: a["value"]["doubleValue"] for a in span_data["spans"][0]["attributes"]}
        assert attrs["peak_ram_mb"] == 42.0
        assert attrs["gpu_vram_mb"] == 512.0
        assert attrs["energy_joules"] == 0.88
