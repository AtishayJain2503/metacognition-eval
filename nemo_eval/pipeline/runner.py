"""
nemo_eval.pipeline.runner
--------------------------
Multi-dataset, multi-model evaluation harness.

Executes AgentLoop across all (model, dataset, task) combinations,
collecting EpisodeTrajectory records and passing them to the evaluator
and reporter.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nemo_eval.agents.agent_loop import AgentLoop, AgentConfig, AgentResult
from nemo_eval.agents.planner import PlannerConfig
from nemo_eval.agents.orchestrator import OrchestratorConfig
from nemo_eval.correction.self_correct import SelfCorrectMetrics, CorrectionStats
from nemo_eval.datasets.base import BenchmarkTask, BaseDatasetLoader
from nemo_eval.datasets.synthetic import SyntheticBenchmarkGenerator
from nemo_eval.eval.engine import evaluate_task_result
from nemo_eval.models import get_model_client
from nemo_eval.pipeline.config import PipelineConfig, ModelSpec, DatasetSpec
from nemo_eval.telemetry.exporters import TelemetryExporter
from nemo_eval.telemetry.tracer import EpisodeTrajectory


class RunRecord:
    """Aggregated results for one (model, dataset) evaluation run."""

    def __init__(self, model_name: str, dataset_name: str):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.trajectories: List[EpisodeTrajectory] = []
        self.correction_stats: List[CorrectionStats] = []
        self.gt_scores: List[float] = []
        self.task_ids: List[str] = []
        self.elapsed_ms: float = 0.0

    def add(self, result: AgentResult, gt_score: float) -> None:
        traj = result.trajectory
        traj.ground_truth_score = gt_score
        self.trajectories.append(traj)
        self.gt_scores.append(gt_score)
        self.task_ids.append(result.task_id)
        self.correction_stats.append(
            SelfCorrectMetrics.compute(traj)
        )

    def summary(self) -> Dict[str, Any]:
        n = len(self.trajectories)
        if n == 0:
            return {"model": self.model_name, "dataset": self.dataset_name, "tasks": 0}
        return {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "tasks": n,
            "success_rate": sum(1 for t in self.trajectories if t.status == "success") / n,
            "avg_gt_score": sum(self.gt_scores) / n,
            "avg_pas": sum(t.plan_adherence_score for t in self.trajectories) / n,
            "avg_tool_accuracy": sum(t.tool_accuracy for t in self.trajectories) / n,
            "avg_spea": sum(t.spea for t in self.trajectories) / n,
            "avg_scsr": sum(s.scsr for s in self.correction_stats) / n,
            "avg_cei": sum(s.cei for s in self.correction_stats) / n,
            "avg_top": sum(s.top for s in self.correction_stats) / n,
            "total_self_corrections": sum(t.self_correction_attempts for t in self.trajectories),
            "elapsed_ms": self.elapsed_ms,
        }


class BenchmarkRunner:
    """
    Orchestrates full evaluation sweeps across all model × dataset combinations.

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
        Execute evaluation for all configured (model, dataset) pairs.

        Returns:
            List of RunRecord, one per (model, dataset) combination.
        """
        records: List[RunRecord] = []
        datasets = self._load_datasets()

        for model_spec in self.config.models:
            print(f"\n[Runner] Loading model: {model_spec.name} ({model_spec.provider}/{model_spec.model_id})")
            try:
                model_client = self._build_model_client(model_spec)
            except Exception as e:
                print(f"[Runner] Failed to load model {model_spec.name}: {e}")
                continue

            agent_config = AgentConfig(
                max_turns=self.config.max_turns,
                max_correction_attempts=self.config.max_correction_attempts,
                enable_planning=self.config.enable_planning,
                verify_intermediate=self.config.verify_intermediate,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            loop = AgentLoop(model_client=model_client, config=agent_config)

            for ds_name, tasks in datasets.items():
                print(f"[Runner] Evaluating {model_spec.name} on {ds_name} ({len(tasks)} tasks)")
                record = RunRecord(model_name=model_spec.name, dataset_name=ds_name)
                t0 = time.monotonic()

                for task in tasks:
                    print(f"  [Task] {task.task_id}: {task.query[:60]}...")
                    try:
                        result = loop.run(
                            task_id=task.task_id,
                            query=task.query,
                            db_path=task.db_path,
                            table_path=task.table_path,
                            model_name=model_spec.name,
                        )

                        # Ground truth evaluation
                        gt_score = 0.0
                        if task.ground_truth is not None and result.final_answer is not None:
                            try:
                                eval_result = evaluate_task_result(
                                    task=task,
                                    candidate_output=str(result.final_answer),
                                )
                                gt_score = eval_result.score
                            except Exception:
                                gt_score = 0.0

                        record.add(result, gt_score)
                        print(f"    -> {result.trajectory.status} | GT={gt_score:.3f} | PAS={result.trajectory.plan_adherence_score:.3f}")

                        # Stream JSONL
                        if self.config.export_jsonl:
                            self._exporter.append_jsonl(
                                result.trajectory,
                                filename=f"trajectories_{model_spec.name}_{ds_name}.jsonl",
                            )

                    except Exception as e:
                        print(f"    -> ERROR: {e}")
                        continue

                record.elapsed_ms = (time.monotonic() - t0) * 1000.0
                records.append(record)

        return records

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_datasets(self) -> Dict[str, List[BenchmarkTask]]:
        """Load all configured datasets."""
        loaded: Dict[str, List[BenchmarkTask]] = {}

        for ds_spec in self.config.datasets:
            try:
                loader = self._build_loader(ds_spec)
                tasks = loader.load(split=ds_spec.split)
                if ds_spec.max_tasks:
                    tasks = tasks[: ds_spec.max_tasks]
                loaded[ds_spec.name] = tasks
                print(f"[Runner] Loaded {len(tasks)} tasks from dataset '{ds_spec.name}'.")
            except Exception as e:
                print(f"[Runner] Failed to load dataset '{ds_spec.name}': {e}")

        return loaded

    def _build_loader(self, ds_spec: DatasetSpec) -> BaseDatasetLoader:
        """Instantiate the appropriate dataset loader."""
        if ds_spec.name == "synthetic":
            from nemo_eval.datasets.synthetic import SyntheticBenchmarkGenerator
            class _SyntheticLoaderWrapper:
                def __init__(self, out_dir):
                    self.out_dir = out_dir
                def load(self, split="test"):
                    gen = SyntheticBenchmarkGenerator()
                    return gen.get_synthetic_benchmark_tasks(str(self.out_dir))
            return _SyntheticLoaderWrapper(self.output_dir / "synthetic_data")

        # GSM8K downloads from HuggingFace — no local data_dir needed
        if ds_spec.name == "gsm8k":
            from nemo_eval.datasets.gsm8k import GSM8KLoader
            return GSM8KLoader(split=ds_spec.split, max_tasks=ds_spec.max_tasks or 50)

        # All other real datasets require a local data_dir
        if not ds_spec.data_dir:
            raise ValueError(f"Dataset '{ds_spec.name}' requires 'data_dir' in config.")

        if ds_spec.name == "infiagent":
            from nemo_eval.datasets.infiagent import InfiAgentLoader
            return InfiAgentLoader(data_dir=ds_spec.data_dir)
        elif ds_spec.name == "bird_sql":
            from nemo_eval.datasets.bird_sql import BirdSQLLoader
            return BirdSQLLoader(data_dir=ds_spec.data_dir)
        elif ds_spec.name == "databench":
            from nemo_eval.datasets.databench import DataBenchLoader
            return DataBenchLoader(data_dir=ds_spec.data_dir)
        else:
            raise ValueError(f"Unknown dataset: {ds_spec.name}")

    def _build_model_client(self, spec: ModelSpec) -> Any:
        """Build a model client from a ModelSpec."""
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
            model_name=spec.model_id,
            **kwargs,
        )
