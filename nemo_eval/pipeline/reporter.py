"""
nemo_eval.pipeline.reporter
----------------------------
Comprehensive Markdown scorecard, dual-mode comparison tables, accuracy leaderboards,
and energy efficiency metrics reporting (Milestone 4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nemo_eval.pipeline.runner import RunRecord
from nemo_eval.telemetry.exporters import TelemetryExporter
from nemo_eval.telemetry.tracer import EpisodeTrajectory


class MarkdownReporter:
    """
    Generator for structured Markdown scorecards, dual-mode comparison tables,
    energy/resource usage tables, and accuracy leaderboards.
    """

    def __init__(self, output_dir: Union[str, Path] = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_markdown_scorecard(trajectories: List[EpisodeTrajectory]) -> str:
        """
        Generate top-level benchmark scorecard from a list of EpisodeTrajectory instances.
        Conforms directly to E2E test suite contract.
        """
        if not trajectories:
            return "# Metacognition Evaluation Benchmark Scorecard\n\nNo trajectories provided."

        total = len(trajectories)
        passed = sum(1 for t in trajectories if t.ground_truth_score == 1.0)
        acc = (passed / total) * 100.0 if total > 0 else 0.0
        avg_dur = sum(t.total_duration_ms for t in trajectories) / total
        avg_ram = sum(t.peak_ram_mb for t in trajectories) / total
        avg_vram = sum(t.gpu_vram_mb for t in trajectories) / total
        avg_energy = sum(t.energy_joules for t in trajectories) / total
        total_energy = sum(t.energy_joules for t in trajectories)

        lines = [
            "# Metacognition Evaluation Benchmark Scorecard",
            "",
            "## Summary Metrics",
            f"- **Total Episodes**: {total}",
            f"- **Accuracy**: {acc:.2f}% ({passed}/{total})",
            f"- **Avg Latency**: {avg_dur:.2f} ms",
            f"- **Peak RAM**: {avg_ram:.2f} MB",
            f"- **Peak GPU VRAM**: {avg_vram:.2f} MB",
            f"- **Avg Energy**: {avg_energy:.4f} Joules",
            f"- **Total Energy**: {total_energy:.4f} Joules",
            "",
            "## Trajectory Telemetry Details",
            "",
            "| Task ID | Model | Status | Score | Duration (ms) | Peak RAM (MB) | GPU VRAM (MB) | Energy (J) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t in trajectories:
            lines.append(
                f"| {t.task_id} | {t.model_name} | {t.status} | {t.ground_truth_score} | "
                f"{t.total_duration_ms:.1f} | {t.peak_ram_mb:.1f} | {t.gpu_vram_mb:.1f} | {t.energy_joules:.4f} |"
            )
        return "\n".join(lines)

    @staticmethod
    def generate_dual_mode_comparison(
        vanilla_traces: List[EpisodeTrajectory],
        agentic_traces: List[EpisodeTrajectory]
    ) -> str:
        """
        Generate side-by-side comparison table between Vanilla and Agentic modes.
        Computes Delta Accuracy, Duration Speedup/Overhead, and Energy Ratio.
        """
        v_total = len(vanilla_traces)
        a_total = len(agentic_traces)
        v_pass = sum(1 for t in vanilla_traces if t.ground_truth_score == 1.0)
        a_pass = sum(1 for t in agentic_traces if t.ground_truth_score == 1.0)

        v_acc = (v_pass / v_total) * 100.0 if v_total > 0 else 0.0
        a_acc = (a_pass / a_total) * 100.0 if a_total > 0 else 0.0
        delta_acc = a_acc - v_acc

        v_dur = (sum(t.total_duration_ms for t in vanilla_traces) / v_total) if v_total > 0 else 0.0
        a_dur = (sum(t.total_duration_ms for t in agentic_traces) / a_total) if a_total > 0 else 0.0
        dur_ratio = (a_dur / v_dur) if v_dur > 0 else 1.0

        v_ram = (sum(t.peak_ram_mb for t in vanilla_traces) / v_total) if v_total > 0 else 0.0
        a_ram = (sum(t.peak_ram_mb for t in agentic_traces) / a_total) if a_total > 0 else 0.0

        v_energy = (sum(t.energy_joules for t in vanilla_traces) / v_total) if v_total > 0 else 0.0
        a_energy = (sum(t.energy_joules for t in agentic_traces) / a_total) if a_total > 0 else 0.0
        energy_ratio = (a_energy / v_energy) if v_energy > 0 else 1.0

        lines = [
            "# Dual-Mode Parity & Delta Performance Analysis",
            "",
            "Direct side-by-side evaluation over identical task instances.",
            "",
            "| Metric | Vanilla (Zero-Shot) | Agentic (9-State FSM) | Delta (Agentic - Vanilla) |",
            "|---|---|---|---|",
            f"| **Accuracy (%)** | {v_acc:.2f}% | {a_acc:.2f}% | {delta_acc:+.2f}% |",
            f"| **Total Tasks** | {v_total} | {a_total} | 0 |",
            f"| **Avg Duration (ms)** | {v_dur:.1f} | {a_dur:.1f} | {a_dur - v_dur:+.1f} ms ({dur_ratio:.2f}x) |",
            f"| **Avg Peak RAM (MB)** | {v_ram:.1f} | {a_ram:.1f} | {a_ram - v_ram:+.1f} MB |",
            f"| **Avg Energy (J)** | {v_energy:.4f} | {a_energy:.4f} | {a_energy - v_energy:+.4f} J ({energy_ratio:.2f}x) |",
        ]
        return "\n".join(lines)

    def write_summary_report(
        self,
        records: List[RunRecord],
        run_label: str = "Metacognition Benchmark Evaluation",
        filename: str = "summary_scorecard.md",
    ) -> Path:
        """
        Generate and write the full master Markdown scorecard and leaderboards.
        """
        summaries = [r.summary() for r in records]

        lines = [
            f"# {run_label} — Master Scorecard",
            "",
            "> 100% Offline-Hermetic Evaluation with Real-Time Hardware Resource Telemetry.",
            "",
            "## 1. Aggregate Results by Model × Dataset × Mode",
            "",
            "| Model | Dataset | Mode | Tasks | Accuracy | Avg GT | Duration (ms) | Peak RAM (MB) | Energy (J) | PAS | Tool Acc | SCSR |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        for s in summaries:
            if s.get("tasks", 0) == 0:
                continue
            lines.append(
                f"| **{s['model']}** | {s['dataset']} | `{s['mode']}` | {s['tasks']} | "
                f"**{s.get('accuracy', 0.0):.1f}%** | {s.get('avg_gt_score', 0.0):.3f} | "
                f"{s.get('avg_duration_ms', 0.0):.1f} | {s.get('avg_peak_ram_mb', 0.0):.1f} | "
                f"{s.get('avg_energy_joules', 0.0):.4f} | {s.get('avg_pas', 0.0):.2f} | "
                f"{s.get('avg_tool_accuracy', 0.0):.2f} | {s.get('avg_scsr', 0.0):.2f} |"
            )

        # 2. Dual-Mode Parity Section (if both vanilla and agentic records exist)
        v_traces: List[EpisodeTrajectory] = []
        a_traces: List[EpisodeTrajectory] = []
        for r in records:
            if r.mode == "vanilla":
                v_traces.extend(r.trajectories)
            elif r.mode == "agentic":
                a_traces.extend(r.trajectories)

        if v_traces and a_traces:
            lines.append("")
            lines.append("## 2. Dual-Mode Parity Analysis (Vanilla vs Agentic)")
            lines.append(self.generate_dual_mode_comparison(v_traces, a_traces))

        # 3. Accuracy & Energy Efficiency Leaderboard
        lines.append("")
        lines.append("## 3. Accuracy & Resource Efficiency Leaderboard")
        lines.append("")
        lines.append("| Rank | Model | Mode | Accuracy (%) | Avg Latency (ms) | Peak RAM (MB) | Energy (J/Task) | Efficiency Score (Acc/J) |")
        lines.append("|---|---|---|---|---|---|---|---|")

        sorted_summaries = sorted(summaries, key=lambda x: x.get("accuracy", 0.0), reverse=True)
        for idx, s in enumerate(sorted_summaries, start=1):
            if s.get("tasks", 0) == 0:
                continue
            energy = s.get("avg_energy_joules", 0.0001)
            acc = s.get("accuracy", 0.0)
            eff = acc / max(energy, 0.0001)
            lines.append(
                f"| #{idx} | {s['model']} | {s['mode']} | {acc:.1f}% | "
                f"{s.get('avg_duration_ms', 0.0):.1f} | {s.get('avg_peak_ram_mb', 0.0):.1f} | "
                f"{energy:.4f} | {eff:.1f} |"
            )

        out_path = self.output_dir / filename
        out_path.write_text("\n".join(lines), encoding="utf-8")

        # Also write summary_report.md alias
        if filename != "summary_report.md":
            (self.output_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")

        return out_path

    def write_json_summary(
        self,
        records: List[RunRecord],
        filename: str = "summary.json",
    ) -> Path:
        """Write machine-readable JSON summary."""
        data = [r.summary() for r in records]
        out_path = self.output_dir / filename
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out_path

    def write_failure_traces(
        self,
        records: List[RunRecord],
        filename: str = "failure_traces.md",
    ) -> Path:
        """Extract and report all failed episode traces."""
        lines = ["# Failure Traces & Error Diagnostics", ""]
        total_failures = 0

        for record in records:
            failures = [t for t in record.trajectories if t.status != "success" or t.ground_truth_score < 1.0]
            if not failures:
                continue
            lines.append(f"## {record.model_name} × {record.dataset_name} [{record.mode}] ({len(failures)} unresolved)")
            lines.append("")
            for traj in failures:
                total_failures += 1
                lines += [
                    f"### Task: `{traj.task_id}`",
                    f"- **Status**: `{traj.status}`",
                    f"- **Ground Truth Score**: `{traj.ground_truth_score}`",
                    f"- **Final Answer Candidate**: `{traj.final_answer}`",
                    f"- **Self-Correction Attempts**: {traj.self_correction_attempts}",
                    f"- **Invalid FSM Transitions**: {traj.invalid_transitions}",
                    f"- **State Sequence**: `{' → '.join(traj.state_sequence()[-10:])}`",
                    "",
                ]

        if total_failures == 0:
            lines.append("*No failures recorded. All evaluation episodes completed with 100% accuracy.*")

        out_path = self.output_dir / filename
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path


# Alias for full backward and forward compatibility
PipelineReporter = MarkdownReporter
