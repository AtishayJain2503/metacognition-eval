"""
Unit tests for nemo_eval.eval.trace_audit.
"""

import json
from pathlib import Path
import pytest
from nemo_eval.eval.trace_audit import TrajectoryTraceAuditor


def test_audit_trajectory_success():
    auditor = TrajectoryTraceAuditor(max_tasks=5)
    traj = {
        "task_id": "math_algebra_001",
        "ground_truth_score": 1.0,
        "energy_joules": 500.0,
        "gpu_vram_mb": 7000.0,
        "gpu_power_watts": 20.0,
        "peak_ram_mb": 30.0,
        "steps": [],
    }
    res = auditor.audit_trajectory(traj, dataset_name="math")
    assert res["raw_score"] == 1.0
    assert res["tool_verified"] is True
    assert res["latent_proof"] is True
    assert res["is_timeout"] is False


def test_audit_trajectory_tool_verified():
    auditor = TrajectoryTraceAuditor(max_tasks=5)
    # Simulate a task where raw_score is 0.0 but tool printed the gold answer
    traj = {
        "task_id": "math_algebra_001",
        "ground_truth_score": 0.0,
        "energy_joules": 1200.0,
        "gpu_vram_mb": 7000.0,
        "gpu_power_watts": 20.0,
        "peak_ram_mb": 30.0,
        "steps": [
            {
                "state": "TOOL_EXECUTION",
                "input_payload": {"arguments": {"code": "print(28)"}},
                "output_payload": {"status": "success", "output": "28"},
            },
            {
                "state": "FINAL_SYNTHESIS",
                "input_payload": {},
                "output_payload": {"raw_completion": "The area is \\boxed{28}"},
            }
        ],
    }
    res = auditor.audit_trajectory(traj, dataset_name="math")
    assert res["raw_score"] == 0.0
    assert "tool_verified" in res
    assert "latent_proof" in res


def test_audit_file(tmp_path: Path):
    auditor = TrajectoryTraceAuditor(max_tasks=5)
    jsonl_file = tmp_path / "trajectories_Qwen2.5-Math-7B_math_agentic.jsonl"
    data = [
        {
            "task_id": "math_algebra_001",
            "ground_truth_score": 1.0,
            "energy_joules": 500.0,
            "gpu_vram_mb": 7000.0,
            "gpu_power_watts": 20.0,
            "peak_ram_mb": 30.0,
            "steps": [],
        }
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    res = auditor.audit_file(jsonl_file)
    assert res is not None
    assert res["model"] == "Qwen2.5-Math-7B"
    assert res["dataset"] == "math"
    assert res["raw_correct"] == 1
    assert res["raw_accuracy"] == "100.0%"
