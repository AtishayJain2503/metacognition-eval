"""
nemo_eval.telemetry.exporters
------------------------------
Trajectory export utilities: JSONL stream, Markdown scorecard, OTLP-compatible spans.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Union

from nemo_eval.telemetry.tracer import EpisodeTrajectory


class TelemetryExporter:
    """
    Export EpisodeTrajectory records to JSONL, Markdown, and OTLP span formats.

    Usage:
        exporter = TelemetryExporter(output_dir="./runs/eval_001")
        exporter.append_jsonl(trajectory)
        exporter.write_markdown_scorecard([trajectory_1, trajectory_2])
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./nemo_eval_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # JSONL
    # ------------------------------------------------------------------ #

    def append_jsonl(
        self,
        trajectory: EpisodeTrajectory,
        filename: str = "trajectories.jsonl",
    ) -> Path:
        """Append a trajectory to a JSONL file (one JSON object per line)."""
        out_path = self.output_dir / filename
        with out_path.open("a", encoding="utf-8") as f:
            f.write(trajectory.model_dump_json() + "\n")
        return out_path

    def write_jsonl_batch(
        self,
        trajectories: List[EpisodeTrajectory],
        filename: str = "trajectories.jsonl",
    ) -> Path:
        """Write a batch of trajectories to a JSONL file (overwrites)."""
        out_path = self.output_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(traj.model_dump_json() + "\n")
        return out_path

    # ------------------------------------------------------------------ #
    # Markdown scorecard
    # ------------------------------------------------------------------ #

    def write_markdown_scorecard(
        self,
        trajectories: List[EpisodeTrajectory],
        filename: str = "scorecard.md",
        run_label: str = "Evaluation Run",
    ) -> Path:
        """Generate a Markdown scorecard summarizing all trajectories."""
        if not trajectories:
            return self._write_empty_scorecard(filename, run_label)

        total = len(trajectories)
        successes = sum(1 for t in trajectories if t.status == "success")
        avg_pas = sum(t.plan_adherence_score for t in trajectories) / total
        avg_tool_acc = sum(t.tool_accuracy for t in trajectories) / total
        avg_spea = sum(t.spea for t in trajectories) / total
        avg_gt_score = sum(t.ground_truth_score for t in trajectories) / total
        avg_steps = sum(t.total_steps for t in trajectories) / total
        avg_duration = sum(t.total_duration_ms for t in trajectories) / total
        total_corrections = sum(t.self_correction_attempts for t in trajectories)
        correction_successes = sum(1 for t in trajectories if t.self_correction_success)
        total_invalid_transitions = sum(t.invalid_transitions for t in trajectories)

        lines = [
            f"# {run_label} — Evaluation Scorecard",
            "",
            f"> Generated: {_iso_now()}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Episodes | {total} |",
            f"| Success Rate | {successes}/{total} ({100*successes/total:.1f}%) |",
            f"| Avg Ground Truth Score | {avg_gt_score:.4f} |",
            f"| Avg Plan Adherence (PAS) | {avg_pas:.4f} |",
            f"| Avg Tool Accuracy (Acc_tool) | {avg_tool_acc:.4f} |",
            f"| Avg SPEA | {avg_spea:.4f} |",
            f"| Avg Steps/Episode | {avg_steps:.1f} |",
            f"| Avg Duration (ms) | {avg_duration:.0f} |",
            f"| Total Self-Correction Attempts | {total_corrections} |",
            f"| Episodes with Successful Recovery | {correction_successes} |",
            f"| Total Invalid FSM Transitions | {total_invalid_transitions} |",
            "",
            "## Per-Episode Results",
            "",
            "| Task ID | Model | Status | GT Score | PAS | Acc_tool | SPEA | Steps | Corrections |",
            "|---------|-------|--------|----------|-----|----------|------|-------|-------------|",
        ]

        for t in trajectories:
            status_icon = "✅" if t.status == "success" else "❌"
            lines.append(
                f"| {t.task_id} | {t.model_name} | {status_icon} {t.status} "
                f"| {t.ground_truth_score:.3f} | {t.plan_adherence_score:.3f} "
                f"| {t.tool_accuracy:.3f} | {t.spea:.3f} "
                f"| {t.total_steps} | {t.self_correction_attempts} |"
            )

        lines += [
            "",
            "## State Distribution (All Episodes)",
            "",
            "| FSM State | Total Occurrences |",
            "|-----------|-------------------|",
        ]

        from collections import Counter
        state_counts: Counter = Counter()
        for t in trajectories:
            for step in t.steps:
                state_counts[step.state.value] += 1
        for state, count in sorted(state_counts.items()):
            lines.append(f"| {state} | {count} |")

        out_path = self.output_dir / filename
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------ #
    # OTLP-compatible spans (simplified JSON)
    # ------------------------------------------------------------------ #

    def write_otlp_spans(
        self,
        trajectory: EpisodeTrajectory,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Write OTLP-compatible span records for the trajectory.
        Each StepEvent becomes one span.
        """
        filename = filename or f"spans_{trajectory.task_id}.json"
        spans = []
        trace_id = trajectory.task_id.replace("-", "")[:32].ljust(32, "0")

        for step in trajectory.steps:
            span = {
                "traceId": trace_id,
                "spanId": f"{step.step_id:016x}",
                "name": step.state.value,
                "startTimeUnixNano": int(step.timestamp * 1e9),
                "endTimeUnixNano": int((step.timestamp + step.duration_ms / 1000.0) * 1e9),
                "attributes": [
                    {"key": k, "value": {"doubleValue": v}}
                    for k, v in step.metrics.items()
                ],
                "status": {
                    "code": "STATUS_CODE_ERROR" if step.invalid_transition else "STATUS_CODE_OK"
                },
            }
            spans.append(span)

        out_path = self.output_dir / filename
        out_path.write_text(json.dumps({"spans": spans}, indent=2), encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _write_empty_scorecard(self, filename: str, run_label: str) -> Path:
        out_path = self.output_dir / filename
        out_path.write_text(
            f"# {run_label} — Evaluation Scorecard\n\nNo trajectories recorded.\n",
            encoding="utf-8",
        )
        return out_path


def _iso_now() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
