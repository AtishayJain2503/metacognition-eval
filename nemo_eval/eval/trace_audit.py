"""
nemo_eval.eval.trace_audit
==========================
Deep latent trace & reasoning span auditor for trajectory evaluations.

Inspects raw trajectory JSONL files, extracting:
1. End-to-end extracted score (ValueExtractor)
2. Tool-verified derivation score (checking Python REPL code & observations)
3. Latent reasoning span derivation score (checking intermediate CoT thoughts & synthesis)
4. Telemetry aggregation (VRAM, RAM, Power Watts, Energy Joules)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from nemo_eval.datasets.lila import LilaLoader
from nemo_eval.datasets.math import MATHLoader
from nemo_eval.datasets.putnam import PutnamBenchLoader


class TrajectoryTraceAuditor:
    """
    Audits trajectory episodes to verify if ground truth answers were derived
    within intermediate reasoning spans or tool executions even if final formatting failed.
    """

    def __init__(self, max_tasks: int = 50):
        self.max_tasks = max_tasks
        self._ground_truths: Dict[str, Dict[str, Any]] = {}
        self._load_ground_truths()

    def _load_ground_truths(self) -> None:
        """Load benchmark ground truth maps."""
        loaders = [
            (MATHLoader, "math"),
            (PutnamBenchLoader, "putnam"),
            (LilaLoader, "lila"),
        ]
        for loader_cls, name in loaders:
            try:
                tasks = loader_cls(max_tasks=self.max_tasks).load_tasks()
                self._ground_truths[name] = {t.task_id: t for t in tasks}
            except TypeError:
                # Handle loaders without max_tasks in __init__
                try:
                    tasks = loader_cls().load_tasks()
                    self._ground_truths[name] = {t.task_id: t for t in tasks}
                except Exception:
                    pass
            except Exception:
                pass

    def audit_trajectory(
        self,
        trajectory_data: Dict[str, Any],
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Audit a single EpisodeTrajectory dictionary.

        Returns:
            Dict containing raw_score, tool_verified, latent_proof, and telemetry.
        """
        tid = trajectory_data.get("task_id", "")
        ds_name = dataset_name or self._infer_dataset(tid)
        task_obj = self._ground_truths.get(ds_name, {}).get(tid)
        gold = str(task_obj.ground_truth).strip() if task_obj and task_obj.ground_truth is not None else None

        raw_score = float(trajectory_data.get("ground_truth_score", 0.0))
        steps = trajectory_data.get("steps", [])
        
        last_step = steps[-1] if steps else {}
        err = str(last_step.get("input_payload", {}).get("error", ""))
        is_timeout = "timed out" in err.lower() or "timeout" in err.lower()

        if raw_score == 1.0:
            return {
                "task_id": tid,
                "gold": gold,
                "raw_score": 1.0,
                "tool_verified": True,
                "latent_proof": True,
                "is_timeout": is_timeout,
                "energy_joules": trajectory_data.get("energy_joules", 0.0),
                "gpu_vram_mb": trajectory_data.get("gpu_vram_mb", 0.0),
                "gpu_power_watts": trajectory_data.get("gpu_power_watts", 0.0),
                "peak_ram_mb": trajectory_data.get("peak_ram_mb", 0.0),
            }

        found_in_tool = False
        found_in_cot = False
        all_text = ""

        for s in steps:
            inp = s.get("input_payload", {})
            out = s.get("output_payload", {})

            code = str(inp.get("arguments", {}).get("code", ""))
            obs = str(out.get("status", "")) + " " + str(out.get("output", ""))

            all_text += f" {code} {out.get('raw_completion', '')} {out.get('final_answer', '')}"

            if gold:
                if (
                    f"print({gold})" in code
                    or f"print({gold}.0)" in code
                    or f"print('{gold}')" in code
                    or f"print(\"{gold}\")" in code
                    or f": {gold}" in obs
                ):
                    found_in_tool = True
                elif f" {gold}" in code and "print(" in code:
                    found_in_tool = True

        if gold and (
            f"\\boxed{{{gold}}}" in all_text
            or f"boxed{{{gold}}}" in all_text
            or f"is {gold}" in all_text
            or f"= {gold}" in all_text
        ):
            found_in_cot = True

        return {
            "task_id": tid,
            "gold": gold,
            "raw_score": raw_score,
            "tool_verified": found_in_tool,
            "latent_proof": found_in_tool or found_in_cot,
            "is_timeout": is_timeout,
            "energy_joules": trajectory_data.get("energy_joules", 0.0),
            "gpu_vram_mb": trajectory_data.get("gpu_vram_mb", 0.0),
            "gpu_power_watts": trajectory_data.get("gpu_power_watts", 0.0),
            "peak_ram_mb": trajectory_data.get("peak_ram_mb", 0.0),
        }

    def audit_file(self, filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Audit an entire trajectory JSONL file."""
        p = Path(filepath)
        if not p.exists():
            return None

        parts = p.stem.replace("trajectories_", "").split("_")
        model_name = parts[0] if len(parts) > 0 else "unknown"
        ds_name = parts[1] if len(parts) > 1 else "math"
        mode = parts[2] if len(parts) > 2 else "agentic"

        trajs = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        trajs.append(json.loads(line.strip()))
                    except Exception:
                        pass

        if not trajs:
            return None

        total = len(trajs)
        raw_correct = 0
        tool_verified_correct = 0
        latent_proof_correct = 0
        timeouts = 0
        total_energy = 0.0
        total_vram = 0.0
        total_ram = 0.0
        total_power = 0.0

        for t in trajs:
            res = self.audit_trajectory(t, dataset_name=ds_name)
            if res["raw_score"] == 1.0:
                raw_correct += 1
            if res["tool_verified"]:
                tool_verified_correct += 1
            if res["latent_proof"]:
                latent_proof_correct += 1
            if res["is_timeout"]:
                timeouts += 1

            total_energy += res["energy_joules"]
            total_vram += res["gpu_vram_mb"]
            total_ram += res["peak_ram_mb"]
            total_power += res["gpu_power_watts"]

        return {
            "file": p.name,
            "model": model_name,
            "dataset": ds_name,
            "mode": mode,
            "total_tasks": total,
            "raw_correct": raw_correct,
            "raw_accuracy": f"{round(raw_correct / total * 100.0, 1)}%",
            "tool_verified_correct": tool_verified_correct,
            "tool_verified_accuracy": f"{round(tool_verified_correct / total * 100.0, 1)}%",
            "latent_proof_correct": latent_proof_correct,
            "latent_proof_accuracy": f"{round(latent_proof_correct / total * 100.0, 1)}%",
            "timeouts": timeouts,
            "completed_task_accuracy": (
                f"{round(raw_correct / (total - timeouts) * 100.0, 1)}%"
                if (total - timeouts) > 0
                else "N/A"
            ),
            "avg_ram_mb": round(total_ram / total, 1),
            "avg_vram_mb": round(total_vram / total, 1),
            "avg_power_watts": round(total_power / total, 1),
            "avg_energy_joules": round(total_energy / total, 1),
            "total_energy_kj": round(total_energy / 1000.0, 2),
        }

    def audit_directory(self, dir_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Audit all trajectory JSONL files in a directory."""
        d = Path(dir_path)
        if not d.exists():
            return []

        results = []
        for f in sorted(d.glob("trajectories_*.jsonl")):
            res = self.audit_file(f)
            if res:
                results.append(res)
        return results

    def _infer_dataset(self, task_id: str) -> str:
        """Infer dataset name from task_id prefix."""
        tid = task_id.lower()
        if tid.startswith("putnam"):
            return "putnam"
        if tid.startswith("lila"):
            return "lila"
        return "math"
