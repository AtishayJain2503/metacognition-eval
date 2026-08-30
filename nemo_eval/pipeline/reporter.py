"""
nemo_eval.pipeline.reporter
----------------------------
Summary scorecard generator and failure trace analyzer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from nemo_eval.telemetry.exporters import TelemetryExporter


class PipelineReporter:
    """
    Generates comprehensive Markdown and JSON evaluation reports
    from RunRecord summaries.

    Usage:
        reporter = PipelineReporter(output_dir="./nemo_eval_output")
        reporter.write_summary_report(records, run_label="Groq Eval v1")
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_summary_report(
        self,
        records: list,
        run_label: str = "Evaluation Run",
        filename: str = "summary_report.md",
    ) -> Path:
        """Write a Markdown summary report across all RunRecords."""
        summaries = [r.summary() for r in records]

        lines = [
            f"# {run_label} — Summary Report",
            "",
            "> All metrics are offline-hermetic (0% network dependency).",
            "",
            "## Aggregate Results by Model × Dataset",
            "",
            "| Model | Dataset | Tasks | Success% | GT Score | PAS | Acc_tool | SPEA | SCSR | CEI | TOP |",
            "|-------|---------|-------|----------|----------|-----|----------|------|------|-----|-----|",
        ]

        for s in summaries:
            n = s.get("tasks", 0)
            if n == 0:
                continue
            lines.append(
                f"| {s['model']} | {s['dataset']} | {n} "
                f"| {100*s.get('success_rate',0):.1f}% "
                f"| {s.get('avg_gt_score',0):.4f} "
                f"| {s.get('avg_pas',0):.4f} "
                f"| {s.get('avg_tool_accuracy',0):.4f} "
                f"| {s.get('avg_spea',0):.4f} "
                f"| {s.get('avg_scsr',0):.4f} "
                f"| {s.get('avg_cei',0):.4f} "
                f"| {s.get('avg_top',0):.4f} |"
            )

        lines += [
            "",
            "## Metric Definitions",
            "",
            "| Metric | Formula | What It Measures |",
            "|--------|---------|-----------------|",
            "| **GT Score** | Ground truth eval engine | Final answer correctness |",
            "| **PAS** | LCS(planned_order, actual_order) / len(plan) | Plan adherence during execution |",
            "| **Acc_tool** | valid_tool_calls / total_calls | Tool selection correctness |",
            "| **SPEA** | bridged_successes / bridged_calls | Parameter bridging quality |",
            "| **SCSR** | successful_recoveries / attempts | Self-correction success rate |",
            "| **CEI** | recoveries / correction_turns | Correction efficiency |",
            "| **TOP** | correction_turns / max_turns | Turn overhead penalty |",
            "",
        ]

        # Per-record trajectory scorecards
        exporter = TelemetryExporter(output_dir=self.output_dir)
        for record in records:
            if record.trajectories:
                card_file = f"scorecard_{record.model_name}_{record.dataset_name}.md"
                exporter.write_markdown_scorecard(
                    record.trajectories,
                    filename=card_file,
                    run_label=f"{run_label} — {record.model_name} on {record.dataset_name}",
                )
                lines.append(f"- [{record.model_name} × {record.dataset_name}](./{card_file})")

        out_path = self.output_dir / filename
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def write_json_summary(
        self,
        records: list,
        filename: str = "summary.json",
    ) -> Path:
        """Write machine-readable JSON summary."""
        data = [r.summary() for r in records]
        out_path = self.output_dir / filename
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out_path

    def write_failure_traces(
        self,
        records: list,
        filename: str = "failure_traces.md",
    ) -> Path:
        """Extract and report all failed episode traces."""
        lines = ["# Failure Traces", ""]
        total_failures = 0

        for record in records:
            failures = [t for t in record.trajectories if t.status != "success"]
            if not failures:
                continue
            lines.append(f"## {record.model_name} × {record.dataset_name} ({len(failures)} failures)")
            lines.append("")
            for traj in failures:
                total_failures += 1
                lines += [
                    f"### Task: `{traj.task_id}`",
                    f"- **Status**: {traj.status}",
                    f"- **Steps**: {traj.total_steps}",
                    f"- **Self-corrections**: {traj.self_correction_attempts}",
                    f"- **Invalid FSM transitions**: {traj.invalid_transitions}",
                    f"- **State sequence**: `{' → '.join(traj.state_sequence()[-10:])}`",
                    "",
                ]

        if total_failures == 0:
            lines.append("*No failures recorded. All episodes completed successfully.*")

        out_path = self.output_dir / filename
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
