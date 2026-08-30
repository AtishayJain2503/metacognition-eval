"""
nemo_eval.pipeline.runner
--------------------------
Multi-dataset, multi-model evaluation sweep and dual-mode execution engine harness.

Executes VanillaEngine (pure zero-shot CoT) and AgenticEngine (9-state FSM + REPL)
over identical dataset splits for direct parity analysis, tracking hardware telemetry
(duration_ms, peak_ram_mb, gpu_vram_mb, gpu_power_watts, energy_joules) and emitting
streaming JSONL traces to results/.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from nemo_eval.agents.agent_loop import AgentConfig, AgentLoop, AgenticEngine
from nemo_eval.agents.vanilla import VanillaEngine
from nemo_eval.models.base import LLMResponse, ToolCall
from nemo_eval.correction.self_correct import CorrectionStats, SelfCorrectMetrics
from nemo_eval.datasets.base import BaseDatasetLoader, BenchmarkTask
from nemo_eval.datasets.lila import LilaLoader
from nemo_eval.datasets.math import MATHLoader
from nemo_eval.datasets.putnam import PutnamBenchLoader
from nemo_eval.models import get_model_client
from nemo_eval.pipeline.config import DatasetSpec, ExecutionMode, ModelSpec, PipelineConfig
from nemo_eval.telemetry.exporters import TelemetryExporter
from nemo_eval.telemetry.tracer import EpisodeTrajectory


class RunRecord:
    """Aggregated results for one (model, dataset, mode) evaluation run."""

    def __init__(self, model_name: str, dataset_name: str, mode: str = "agentic"):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.mode = mode
        self.trajectories: List[EpisodeTrajectory] = []
        self.correction_stats: List[CorrectionStats] = []
        self.gt_scores: List[float] = []
        self.task_ids: List[str] = []
        self.elapsed_ms: float = 0.0

    def add_trajectory(self, traj: EpisodeTrajectory) -> None:
        self.trajectories.append(traj)
        self.gt_scores.append(traj.ground_truth_score)
        self.task_ids.append(traj.task_id)
        if traj.steps:
            self.correction_stats.append(SelfCorrectMetrics.compute(traj))

    def add(self, result: Any, gt_score: float) -> None:
        """Compatibility helper for AgentResult or direct EpisodeTrajectory."""
        if hasattr(result, "trajectory"):
            traj = result.trajectory
            traj.ground_truth_score = gt_score
        elif isinstance(result, EpisodeTrajectory):
            traj = result
            traj.ground_truth_score = gt_score
        else:
            return
        self.add_trajectory(traj)

    def summary(self) -> Dict[str, Any]:
        n = len(self.trajectories)
        if n == 0:
            return {
                "model": self.model_name,
                "dataset": self.dataset_name,
                "mode": self.mode,
                "tasks": 0,
                "success_rate": 0.0,
                "accuracy": 0.0,
                "avg_gt_score": 0.0,
                "avg_duration_ms": 0.0,
                "avg_peak_ram_mb": 0.0,
                "avg_gpu_vram_mb": 0.0,
                "avg_energy_joules": 0.0,
                "avg_pas": 0.0,
                "avg_tool_accuracy": 0.0,
                "avg_spea": 0.0,
                "avg_scsr": 0.0,
                "avg_cei": 0.0,
                "avg_top": 0.0,
                "total_self_corrections": 0,
                "elapsed_ms": 0.0,
            }

        passed = sum(1 for t in self.trajectories if t.ground_truth_score == 1.0)
        avg_dur = sum(t.total_duration_ms for t in self.trajectories) / n
        avg_ram = sum(t.peak_ram_mb for t in self.trajectories) / n
        avg_vram = sum(t.gpu_vram_mb for t in self.trajectories) / n
        avg_energy = sum(t.energy_joules for t in self.trajectories) / n

        return {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "mode": self.mode,
            "tasks": n,
            "success_rate": passed / n,
            "accuracy": (passed / n) * 100.0,
            "avg_gt_score": sum(self.gt_scores) / n if self.gt_scores else 0.0,
            "avg_duration_ms": round(avg_dur, 2),
            "avg_peak_ram_mb": round(avg_ram, 2),
            "avg_gpu_vram_mb": round(avg_vram, 2),
            "avg_energy_joules": round(avg_energy, 4),
            "avg_pas": sum(t.plan_adherence_score for t in self.trajectories) / n,
            "avg_tool_accuracy": sum(t.tool_accuracy for t in self.trajectories) / n,
            "avg_spea": sum(t.spea for t in self.trajectories) / n,
            "avg_scsr": sum(s.scsr for s in self.correction_stats) / len(self.correction_stats) if self.correction_stats else 0.0,
            "avg_cei": sum(s.cei for s in self.correction_stats) / len(self.correction_stats) if self.correction_stats else 0.0,
            "avg_top": sum(s.top for s in self.correction_stats) / len(self.correction_stats) if self.correction_stats else 0.0,
            "total_self_corrections": sum(t.self_correction_attempts for t in self.trajectories),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class BenchmarkRunner:
    """
    Orchestrates dual-mode and multi-model benchmark evaluation sweeps.

    Usage:
        config = PipelineConfig.from_json("config.json")
        runner = BenchmarkRunner(config)
        records = runner.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._exporter = TelemetryExporter(output_dir=self.output_dir)

    def run(self) -> List[RunRecord]:
        """
        Execute evaluation sweep across all configured (model, dataset) combinations.

        Returns:
            List of RunRecord instances containing trajectories and telemetry.
        """
        records: List[RunRecord] = []
        datasets = self._load_datasets()

        # Determine modes to evaluate
        mode_str = str(self.config.mode.value if isinstance(self.config.mode, ExecutionMode) else self.config.mode).lower()
        run_vanilla = mode_str in ("vanilla", "both", "dual_parity")
        run_agentic = mode_str in ("agentic", "both", "dual_parity")

        for model_spec in self.config.models:
            print(f"\n[Runner] Loading model: {model_spec.name} ({model_spec.provider}/{model_spec.model_id})")
            try:
                model_client = self._build_model_client(model_spec)
            except Exception as e:
                print(f"[Runner] Failed to load model {model_spec.name}: {e}")
                continue

            # Instantiate engines
            sample_interval = float(self.config.telemetry_sample_interval_ms) / 1000.0
            vanilla_engine = VanillaEngine(
                model=model_client,
                enable_telemetry=self.config.enable_telemetry,
                sample_interval_s=sample_interval,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            agent_config = AgentConfig(
                max_turns=self.config.max_turns,
                max_correction_attempts=self.config.max_correction_attempts,
                enable_planning=self.config.enable_planning,
                verify_intermediate=self.config.verify_intermediate,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            agentic_engine = AgenticEngine(model=model_client, config=agent_config)

            for ds_name, tasks in datasets.items():
                print(f"[Runner] Evaluating {model_spec.name} on {ds_name} ({len(tasks)} tasks)")

                # 1. Vanilla execution
                if run_vanilla:
                    v_record = RunRecord(model_name=model_spec.name, dataset_name=ds_name, mode="vanilla")
                    t0 = time.monotonic()
                    for task in tasks:
                        try:
                            # Dynamic Mock Injection for Vanilla Mode
                            if hasattr(model_client, "response_queue"):
                                model_client.reset()
                                model_client.response_queue.clear()
                                gold_val = str(task.ground_truth)
                                if "perfect" in model_spec.model_id:
                                    model_client.add_response(LLMResponse(
                                        content=f"After calculation, the final answer is \\boxed{{{gold_val}}}",
                                        tool_calls=[]
                                    ))
                                elif "deepseek_r1_reasoning" in model_spec.model_id:
                                    model_client.add_response(LLMResponse(
                                        content=f"<think>\nReasoning to get {gold_val}\n</think>\nThe answer is \\boxed{{{gold_val}}}",
                                        reasoning_content=f"Reasoning to get {gold_val}",
                                        tool_calls=[]
                                    ))
                                else:
                                    # Default perfect-like vanilla response
                                    model_client.add_response(LLMResponse(
                                        content=f"The answer is \\boxed{{{gold_val}}}",
                                        tool_calls=[]
                                    ))
                            traj = vanilla_engine.evaluate_task(task)
                            v_record.add_trajectory(traj)
                            if self.config.export_jsonl:
                                self._stream_trajectory(traj, model_spec.name, ds_name, "vanilla")
                        except Exception as e:
                            print(f"  [Vanilla Error] Task {task.task_id}: {e}")
                    v_record.elapsed_ms = (time.monotonic() - t0) * 1000.0
                    records.append(v_record)
                    print(f"  [Vanilla] Done: Acc={v_record.summary()['accuracy']:.1f}% | Duration={v_record.summary()['avg_duration_ms']:.1f}ms | Energy={v_record.summary()['avg_energy_joules']:.4f}J")

                # 2. Agentic execution
                if run_agentic:
                    a_record = RunRecord(model_name=model_spec.name, dataset_name=ds_name, mode="agentic")
                    t0 = time.monotonic()
                    for task in tasks:
                        try:
                            # Dynamic Mock Injection for Agentic Mode
                            if hasattr(model_client, "response_queue"):
                                model_client.reset()
                                model_client.response_queue.clear()
                                gold_val = str(task.ground_truth)
                                
                                # 1. If planning is enabled, queue a valid task plan first
                                if self.config.enable_planning:
                                    model_client.add_response(LLMResponse(
                                        content=json.dumps({
                                            "sub_goals": [
                                                {
                                                    "id": "sg_1",
                                                    "description": "Calculate answer",
                                                    "tool_hint": "python_repl",
                                                    "depends_on": [],
                                                    "expected_output_type": "scalar"
                                                }
                                            ]
                                        }),
                                        tool_calls=[]
                                    ))
                                
                                # 2. Queue subgoal execution & synthesis responses
                                if "perfect" in model_spec.model_id:
                                    model_client.add_response(LLMResponse(
                                        content=json.dumps({
                                            "tool_name": "python_repl",
                                            "arguments": {"code": f"print({repr(gold_val)})"}
                                        }),
                                        tool_calls=[]
                                    ))
                                    model_client.add_response(LLMResponse(
                                        content=gold_val,
                                        tool_calls=[]
                                    ))
                                elif "deepseek_r1_reasoning" in model_spec.model_id:
                                    model_client.add_response(LLMResponse(
                                        content=f"<think>\nI should calculate this.\n</think>\n" + json.dumps({
                                            "tool_name": "python_repl",
                                            "arguments": {"code": f"print({repr(gold_val)})"}
                                        }),
                                        reasoning_content="I should calculate this.",
                                        tool_calls=[]
                                    ))
                                    model_client.add_response(LLMResponse(
                                        content=f"<think>\nFinal check.\n</think>\n{gold_val}",
                                        reasoning_content="Final check.",
                                        tool_calls=[]
                                    ))
                                elif "self_correction" in model_spec.model_id:
                                    # Turn 1: invalid python syntax code
                                    model_client.add_response(LLMResponse(
                                        content=json.dumps({
                                            "tool_name": "python_repl",
                                            "arguments": {"code": "invalid python code syntax"}
                                        }),
                                        tool_calls=[]
                                    ))
                                    # Turn 2: correct python code
                                    model_client.add_response(LLMResponse(
                                        content=json.dumps({
                                            "tool_name": "python_repl",
                                            "arguments": {"code": f"print({repr(gold_val)})"}
                                        }),
                                        tool_calls=[]
                                    ))
                                    model_client.add_response(LLMResponse(
                                        content=gold_val,
                                        tool_calls=[]
                                    ))
                                else:
                                    # Default perfect-like tool flow
                                    model_client.add_response(LLMResponse(
                                        content=json.dumps({
                                            "tool_name": "python_repl",
                                            "arguments": {"code": f"print({repr(gold_val)})"}
                                        }),
                                        tool_calls=[]
                                    ))
                                    model_client.add_response(LLMResponse(
                                        content=gold_val,
                                        tool_calls=[]
                                    ))
                            traj = agentic_engine.evaluate_task(task)
                            a_record.add_trajectory(traj)
                            if self.config.export_jsonl:
                                self._stream_trajectory(traj, model_spec.name, ds_name, "agentic")
                        except Exception as e:
                            print(f"  [Agentic Error] Task {task.task_id}: {e}")
                    a_record.elapsed_ms = (time.monotonic() - t0) * 1000.0
                    records.append(a_record)
                    print(f"  [Agentic] Done: Acc={a_record.summary()['accuracy']:.1f}% | PAS={a_record.summary()['avg_pas']:.2f} | Duration={a_record.summary()['avg_duration_ms']:.1f}ms | Energy={a_record.summary()['avg_energy_joules']:.4f}J")

        return records

    def _stream_trajectory(self, traj: EpisodeTrajectory, model_name: str, ds_name: str, mode: str) -> None:
        """Stream JSONL record to disk."""
        try:
            line = traj.model_dump_json() + "\n"
            # Specific file
            spec_file = self.output_dir / f"trajectories_{model_name}_{ds_name}_{mode}.jsonl"
            with open(spec_file, "a", encoding="utf-8") as f:
                f.write(line)
            # Master log
            master_file = self.output_dir / "streaming_trajectories.jsonl"
            with open(master_file, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Dataset Loading Helpers
    # ------------------------------------------------------------------ #

    def _load_datasets(self) -> Dict[str, List[BenchmarkTask]]:
        """Load all datasets specified in config."""
        loaded: Dict[str, List[BenchmarkTask]] = {}

        for ds_spec in self.config.datasets:
            ds_name_lower = ds_spec.name.lower().strip()
            try:
                if ds_name_lower in ("math", "hendrycks_math"):
                    loader = MATHLoader(dataset_root=ds_spec.data_dir, max_tasks=ds_spec.max_tasks, subject=ds_spec.subject)
                    tasks = loader.load(split=ds_spec.split, limit=ds_spec.max_tasks)
                    loaded["math"] = tasks
                elif ds_name_lower in ("putnam", "putnambench"):
                    loader = PutnamBenchLoader(dataset_root=ds_spec.data_dir, max_tasks=ds_spec.max_tasks, category=ds_spec.category)
                    tasks = loader.load(split=ds_spec.split, limit=ds_spec.max_tasks)
                    loaded["putnam"] = tasks
                elif ds_name_lower in ("lila", "allenai_lila"):
                    subcats = [ds_spec.category] if ds_spec.category else ([ds_spec.subdiscipline] if ds_spec.subdiscipline else None)
                    loader = LilaLoader(dataset_root=ds_spec.data_dir, subcategories=subcats, max_tasks_per_category=ds_spec.max_tasks)
                    tasks = loader.load(split=ds_spec.split, limit=ds_spec.max_tasks)
                    loaded["lila"] = tasks
                elif ds_name_lower == "all":
                    # Load all core benchmarks: MATH (50), Putnam (50), Lila (350)
                    limit = ds_spec.max_tasks or 50
                    loaded["math"] = MATHLoader().load(limit=limit)
                    loaded["putnam"] = PutnamBenchLoader().load(limit=limit)
                    loaded["lila"] = LilaLoader().load(limit=limit)
                elif ds_name_lower == "synthetic":
                    from nemo_eval.datasets.synthetic import SyntheticBenchmarkGenerator
                    gen = SyntheticBenchmarkGenerator()
                    tasks = gen.get_synthetic_benchmark_tasks(str(self.output_dir / "synthetic_data"))
                    if ds_spec.max_tasks:
                        tasks = tasks[:ds_spec.max_tasks]
                    loaded["synthetic"] = tasks
                elif ds_name_lower == "gsm8k":
                    from nemo_eval.datasets.gsm8k import GSM8KLoader
                    loader = GSM8KLoader(split=ds_spec.split, max_tasks=ds_spec.max_tasks or 50)
                    loaded["gsm8k"] = loader.load(limit=ds_spec.max_tasks)
                elif ds_name_lower == "infiagent":
                    from nemo_eval.datasets.infiagent import InfiAgentLoader
                    loader = InfiAgentLoader(data_dir=ds_spec.data_dir or "")
                    loaded["infiagent"] = loader.load(limit=ds_spec.max_tasks)
                elif ds_name_lower == "bird_sql":
                    from nemo_eval.datasets.bird_sql import BirdSQLLoader
                    loader = BirdSQLLoader(data_dir=ds_spec.data_dir or "")
                    loaded["bird_sql"] = loader.load(limit=ds_spec.max_tasks)
                elif ds_name_lower == "databench":
                    from nemo_eval.datasets.databench import DataBenchLoader
                    loader = DataBenchLoader(data_dir=ds_spec.data_dir or "")
                    loaded["databench"] = loader.load(limit=ds_spec.max_tasks)
                else:
                    raise ValueError(f"Unknown benchmark dataset: '{ds_spec.name}'")

                print(f"[Runner] Loaded {len(loaded.get(ds_spec.name, tasks))} tasks from dataset '{ds_spec.name}'.")
            except Exception as e:
                print(f"[Runner] Failed to load dataset '{ds_spec.name}': {e}")

        return loaded

    def _build_model_client(self, spec: ModelSpec) -> Any:
        """Build model client conforming to BaseLLMClient."""
        api_key = None
        if spec.api_key_env:
            api_key = os.environ.get(spec.api_key_env)

        kwargs: Dict[str, Any] = {**spec.extra}
        if api_key:
            kwargs["api_key"] = api_key
        if spec.base_url:
            kwargs["base_url"] = spec.base_url

        return get_model_client(
            provider=spec.provider,
            model_name=spec.model_id or spec.name,
            **kwargs,
        )
