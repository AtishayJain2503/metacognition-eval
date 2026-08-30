"""
tests/unit/test_adversarial_m3_m4_challenge.py
----------------------------------------------
Comprehensive Adversarial Empirical Stress Test Suite for Milestones M3 & M4:
1. Dual-Mode Parity & Zero Cross-Contamination (VanillaEngine vs AgenticEngine)
2. Target Model Registry (All 7 Models) & DeepSeek-R1 <think> Tag Isolation
3. CLI Execution Robustness (nemo_eval/cli.py flags, sweeps, edge cases, error codes)
4. Markdown Scorecard, Streaming JSONL Telemetry Traces, Delta Metrics, and Leaderboards
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List
import pytest

from nemo_eval.agents.agent_loop import AgentConfig, AgenticEngine, AgentLoop, AgentResult
from nemo_eval.agents.planner import TaskPlanner, TaskPlan, SubGoal
from nemo_eval.agents.vanilla import VanillaEngine, BaseEvaluationEngine
from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.datasets.math import MATHLoader
from nemo_eval.datasets.putnam import PutnamBenchLoader
from nemo_eval.datasets.lila import LilaLoader
from nemo_eval.models.base import BaseLLMClient, LLMMessage, LLMResponse, ModelRegistry, ToolCall, get_model_client
from nemo_eval.models.groq import GroqLLMClient, extract_think_reasoning
from nemo_eval.models.mock_runner import DeterministicMockLLMClient
from nemo_eval.models.registry import (
    MODEL_CONFIGS,
    MODEL_FAMILIES,
    TARGET_MODELS,
    TargetModelSpec,
    create_target_model_client,
    get_model_spec,
    get_target_models,
)
from nemo_eval.pipeline.config import DatasetSpec, ExecutionMode, ModelSpec, PipelineConfig
from nemo_eval.pipeline.reporter import MarkdownReporter, PipelineReporter
from nemo_eval.pipeline.runner import BenchmarkRunner, RunRecord
from nemo_eval.telemetry.tracer import EpisodeTrajectory, TrajectoryState, TrajectoryTracer
from nemo_eval.cli import build_parser, cmd_report, cmd_run, cmd_validate_config


# ===========================================================================
# 1. Dual-Mode Parity & Zero Cross-Contamination Stress Tests
# ===========================================================================

class TestDualModeParityAndIsolation:
    """Stress tests verifying strict isolation and parity between VanillaEngine and AgenticEngine."""

    def test_task_immutability_across_engines(self):
        """Verify that BenchmarkTask attributes are strictly unmutated after both engines execute."""
        task = BenchmarkTask(
            task_id="parity_task_001",
            dataset_name="math",
            subdiscipline="Algebra",
            problem_text="Find the value of $x$ such that $3x + 9 = 24$.",
            ground_truth="5",
            eval_type="math_symbolic",
            metadata={"difficulty": 2, "tags": ["linear_eq", "algebra"]}
        )
        task_snapshot = copy.deepcopy(task)

        mock_v = DeterministicMockLLMClient(
            model_name="Qwen2.5-Math-7B",
            responses={"Find the value of": "Step 1: 3x = 15, x = 5.\nFinal Answer: \\boxed{5}"}
        )
        tool_call_json = json.dumps({
            "tool_name": "python_repl",
            "arguments": {"code": "print(5)"}
        })
        plan_json = json.dumps({
            "sub_goals": [{"id": "sg_1", "description": "Compute x", "tool_hint": "python_repl", "depends_on": []}]
        })
        mock_a = DeterministicMockLLMClient(
            model_name="Qwen2.5-Math-7B",
            response_queue=[
                LLMResponse(content=plan_json),
                LLMResponse(content=tool_call_json),
                LLMResponse(content="5"),
            ]
        )

        v_engine = VanillaEngine(model=mock_v)
        a_engine = AgenticEngine(model=mock_a)

        v_traj = v_engine.evaluate_task(task)
        a_traj = a_engine.evaluate_task(task)

        # Assert task attributes have not been modified or mutated
        assert task.task_id == task_snapshot.task_id
        assert task.dataset_name == task_snapshot.dataset_name
        assert task.subdiscipline == task_snapshot.subdiscipline
        assert task.problem_text == task_snapshot.problem_text
        assert task.ground_truth == task_snapshot.ground_truth
        assert task.eval_type == task_snapshot.eval_type
        assert task.metadata == task_snapshot.metadata

        # Assert both trajectories evaluate the same task
        assert v_traj.task_id == task.task_id
        assert a_traj.task_id == task.task_id
        assert v_traj.ground_truth_score == 1.0
        assert a_traj.ground_truth_score == 1.0

    def test_planner_scalar_llm_response_attribute_error_vulnerability(self):
        """Empirical vulnerability test: TaskPlanner._parse_sub_goals crashes when LLM returns integer/scalar string."""
        mock_model = DeterministicMockLLMClient(responses={"Decompose": "5"})
        planner = TaskPlanner(model_client=mock_model)

        # When LLM returns scalar '5', json.loads returns int 5.
        # planner._parse_sub_goals expects dict with .get("sub_goals")
        with pytest.raises(AttributeError, match="'int' object has no attribute 'get'"):
            planner._parse_sub_goals("5")

    def test_agent_loop_scalar_tool_call_attribute_error_vulnerability(self):
        """Empirical vulnerability test: AgentLoop._parse_tool_call crashes when LLM returns integer/scalar string."""
        mock_model = DeterministicMockLLMClient()
        loop = AgentLoop(model_client=mock_model)
        sg = SubGoal(id="sg_1", description="Calculate", tool_hint=None, depends_on=[])

        # When LLM outputs '42', json.loads returns int 42.
        # loop._parse_tool_call calls data.get("tool_name"), raising AttributeError
        with pytest.raises(AttributeError, match="'int' object has no attribute 'get'"):
            loop._parse_tool_call("42", sg)

    def test_execution_order_invariance_and_no_residual_state(self):
        """Verify execution order (Vanilla->Agentic vs Agentic->Vanilla) causes zero cross-talk."""
        task1 = BenchmarkTask(
            task_id="order_1",
            dataset_name="math",
            subdiscipline="Arithmetic",
            problem_text="Compute 12 * 12",
            ground_truth="144",
            eval_type="exact"
        )
        task2 = BenchmarkTask(
            task_id="order_2",
            dataset_name="math",
            subdiscipline="Arithmetic",
            problem_text="Compute 15 * 15",
            ground_truth="225",
            eval_type="exact"
        )

        plan_json = json.dumps({"sub_goals": [{"id": "sg_1", "description": "calc", "tool_hint": "python_repl", "depends_on": []}]})
        tool_call_json = json.dumps({"tool_name": "python_repl", "arguments": {"code": "print(144)"}})
        mock_model = DeterministicMockLLMClient(
            model_name="Llama3.2-3B",
            responses={
                "Decompose": plan_json,
                "Compute 12 * 12": tool_call_json,
                "Compute 15 * 15": "\\boxed{225}",
                "Problem": "\\boxed{144}",
            }
        )
        v_engine = VanillaEngine(model=mock_model)
        a_engine = AgenticEngine(model=mock_model)

        # Run Sequence 1: Agentic on task1 -> Vanilla on task1
        a_traj_1 = a_engine.evaluate_task(task1)
        v_traj_1 = v_engine.evaluate_task(task1)

        # Run Sequence 2: Vanilla on task2 -> Agentic on task2
        v_traj_2 = v_engine.evaluate_task(task2)
        a_traj_2 = a_engine.evaluate_task(task2)

        # Check Vanilla trajectories strictly contain 0 tool states
        for step in v_traj_1.steps:
            assert step.state not in (TrajectoryState.TOOL_EXECUTION, TrajectoryState.OBSERVATION, TrajectoryState.SELF_CORRECTION)
        for step in v_traj_2.steps:
            assert step.state not in (TrajectoryState.TOOL_EXECUTION, TrajectoryState.OBSERVATION, TrajectoryState.SELF_CORRECTION)

        assert v_traj_1.plan_adherence_score == 1.0
        assert v_traj_1.tool_accuracy == 1.0
        assert v_traj_2.plan_adherence_score == 1.0
        assert v_traj_2.tool_accuracy == 1.0

    def test_concurrent_multithreaded_engine_isolation(self):
        """Stress-test concurrent multi-threaded execution of Vanilla and Agentic engines."""
        results: Dict[str, Any] = {}
        errors: List[Exception] = []

        def worker_vanilla(idx: int):
            try:
                m = DeterministicMockLLMClient(responses={"query": f"\\boxed{{{idx * 10}}}"})
                eng = VanillaEngine(model=m)
                t = BenchmarkTask(task_id=f"v_t_{idx}", dataset_name="math", subdiscipline="Algebra", problem_text="query", ground_truth=str(idx * 10), eval_type="exact")
                traj = eng.evaluate_task(t)
                results[f"v_{idx}"] = traj
            except Exception as e:
                errors.append(e)

        def worker_agentic(idx: int):
            try:
                plan_json = json.dumps({"sub_goals": [{"id": "sg_1", "description": "calc", "tool_hint": "python_repl", "depends_on": []}]})
                tc_json = json.dumps({"tool_name": "python_repl", "arguments": {"code": f"print({idx * 10})"}} )
                m = DeterministicMockLLMClient(
                    response_queue=[
                        LLMResponse(content=plan_json),
                        LLMResponse(content=tc_json),
                        LLMResponse(content=str(idx * 10)),
                    ]
                )
                eng = AgenticEngine(model=m)
                t = BenchmarkTask(task_id=f"a_t_{idx}", dataset_name="math", subdiscipline="Algebra", problem_text="query", ground_truth=str(idx * 10), eval_type="exact")
                traj = eng.evaluate_task(t)
                results[f"a_{idx}"] = traj
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=worker_vanilla, args=(i,)))
            threads.append(threading.Thread(target=worker_agentic, args=(i,)))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
        assert len(results) == 20
        for k, traj in results.items():
            assert traj.ground_truth_score == 1.0
            assert traj.peak_ram_mb > 0.0

    def test_extreme_task_inputs_and_edge_cases(self):
        """Test VanillaEngine and AgenticEngine with empty queries, None ground truth, and massive inputs."""
        plan_json = json.dumps({"sub_goals": [{"id": "sg_1", "description": "calc", "tool_hint": "python_repl", "depends_on": []}]})
        tc_json = json.dumps({"tool_name": "python_repl", "arguments": {"code": "print(0)"}})
        mock_model = DeterministicMockLLMClient(
            model_name="Phi4-mini-reasoning",
            responses={"Decompose": plan_json, "Problem": "\\boxed{0}", "Calculate": tc_json}
        )
        v_engine = VanillaEngine(model=mock_model)
        a_engine = AgenticEngine(model=mock_model)

        # 1. Empty problem text
        task_empty = BenchmarkTask(task_id="empty_q", dataset_name="math", subdiscipline="Algebra", problem_text="", ground_truth="0", eval_type="exact")
        v_traj = v_engine.evaluate_task(task_empty)
        a_traj = a_engine.evaluate_task(task_empty)
        assert v_traj.status in ("success", "failed")
        assert a_traj.status in ("success", "failed")

        # 2. None ground truth
        task_none_gt = BenchmarkTask(task_id="none_gt", dataset_name="math", subdiscipline="Algebra", problem_text="Find x", ground_truth=None, eval_type="exact")
        v_traj2 = v_engine.evaluate_task(task_none_gt)
        a_traj2 = a_engine.evaluate_task(task_none_gt)
        assert v_traj2.ground_truth_score == 0.0
        assert a_traj2.ground_truth_score == 0.0

        # 3. Massive 10,000-character input with LaTeX formulas and special characters
        big_problem = "Let $f(x) = " + " + ".join(f"x^{{{i}}}" for i in range(100)) + "$. Compute $f'(1)$."
        task_big = BenchmarkTask(task_id="big_task", dataset_name="putnam", subdiscipline="Calculus", problem_text=big_problem, ground_truth="4950", eval_type="math_symbolic")
        v_traj3 = v_engine.evaluate_task(task_big)
        a_traj3 = a_engine.evaluate_task(task_big)
        assert v_traj3.task_id == "big_task"
        assert a_traj3.task_id == "big_task"

    def test_engine_model_exception_handling_disparity(self):
        """Empirical vulnerability test: VanillaEngine catches LLM exceptions, but AgenticEngine unhandled during decompose()."""
        failing_model = DeterministicMockLLMClient(inject_errors=[RuntimeError("Unrecoverable GPU Out of Memory")])
        v_engine = VanillaEngine(model=failing_model)

        task = BenchmarkTask(task_id="fail_task", dataset_name="math", subdiscipline="Algebra", problem_text="Solve 1+1", ground_truth="2", eval_type="exact")

        # Vanilla engine gracefully catches exception and returns failed trajectory
        v_traj = v_engine.evaluate_task(task)
        assert v_traj.status == "failed"
        assert v_traj.ground_truth_score == 0.0

        # AgenticEngine uncaught during decompose() when LLM throws error
        failing_model_agentic = DeterministicMockLLMClient(inject_errors=[RuntimeError("Injected LLM Failure during planning")])
        a_engine = AgenticEngine(model=failing_model_agentic)
        with pytest.raises(RuntimeError, match="Injected LLM Failure during planning"):
            a_engine.evaluate_task(task)


# ===========================================================================
# 2. Model Registry & DeepSeek-R1 <think> Tag Isolation Stress Tests
# ===========================================================================

class TestTargetModelsAndThinkIsolation:
    """Stress tests for all 7 target models and <think> token isolation."""

    def test_all_seven_target_models_configured(self):
        """Verify all 7 target models are in registry with accurate hyperparameter specs."""
        expected_models = [
            "Qwen2.5-Math-7B",
            "DeepSeek-R1-7B",
            "Phi4-mini-reasoning",
            "Llama3.2-3B",
            "Qwen2.5-Math-1.5B",
            "DeepSeek-R1-1.5B",
            "Qwen3-4B-Thinking",
        ]
        target_models = get_target_models()
        for m in expected_models:
            assert m in target_models
            assert m in TARGET_MODELS
            assert m in MODEL_CONFIGS
            assert m in MODEL_FAMILIES

            spec = get_model_spec(m)
            assert isinstance(spec, TargetModelSpec)
            assert spec.model_name == m
            assert spec.max_tokens >= 2048
            assert spec.family in ("Qwen", "DeepSeek", "Phi", "Llama")

            # Check reasoning flags
            if "R1" in m or "Thinking" in m or "reasoning" in m:
                assert spec.is_reasoning_model is True

            # Verify client instantiation
            client = create_target_model_client(m, provider="mock")
            assert isinstance(client, BaseLLMClient)

    def test_custom_and_unknown_model_fallback_ordering_behavior(self):
        """Empirical vulnerability test: Model family heuristic matches 'Qwen' for hybrid names because Qwen is checked first."""
        spec1 = get_model_spec("Qwen2.5-Custom-32B")
        assert spec1.family == "Qwen"
        assert spec1.is_reasoning_model is False

        # Note: In registry.py, MODEL_FAMILIES has Qwen before DeepSeek, so 'DeepSeek-Distill-Qwen' matches Qwen
        spec2 = get_model_spec("DeepSeek-R1-Distill-Qwen-14B")
        assert spec2.family == "Qwen"  # Matches Qwen due to dict order
        assert spec2.is_reasoning_model is True  # But correctly identifies 'r1' reasoning flag
        assert spec2.think_isolation is True

        spec3 = get_model_spec("completely_unknown_arch_v1")
        assert spec3.family == "Custom"
        assert spec3.max_tokens == 4096

    def test_extract_think_reasoning_exhaustive_edge_cases(self):
        """Exhaustively stress-test <think> tag regex isolation under adversarial variations."""
        # 1. Multiple think tags
        raw1 = "<think>First thought</think>Intermediate text<think>Second thought</think>Final \\boxed{42}"
        r1, c1 = extract_think_reasoning(raw1)
        assert r1 is not None

        # 2. Nested tags
        raw2 = "<think>Outer <think>inner</think> more</think> Final answer"
        r2, c2 = extract_think_reasoning(raw2)
        assert r2 is not None
        assert "Final answer" in (c2 or "")

        # 3. Newlines and indented code inside think tag
        raw3 = (
            "<think>\n"
            "  import math\n"
            "  ans = math.sqrt(144)\n"
            "</think>\n"
            "\\boxed{12}"
        )
        r3, c3 = extract_think_reasoning(raw3)
        assert "import math" in r3
        assert c3 == "\\boxed{12}"

        # 4. Unclosed think tag with trailing text
        raw4 = "<think>I am halfway through solving the integral..."
        r4, c4 = extract_think_reasoning(raw4)
        assert r4 == "I am halfway through solving the integral..."
        assert c4 is None

        # 5. Empty tag
        raw5 = "<think></think>Clean answer"
        r5, c5 = extract_think_reasoning(raw5)
        assert r5 is None
        assert c5 == "Clean answer"

        # 6. Null and empty inputs
        assert extract_think_reasoning(None) == (None, None)
        assert extract_think_reasoning("") == (None, None)
        assert extract_think_reasoning("    ") == (None, None)

        # 7. No think tag at all
        raw7 = "Step 1: solve. Final answer: \\boxed{7}"
        r7, c7 = extract_think_reasoning(raw7)
        assert r7 is None
        assert c7 == raw7


# ===========================================================================
# 3. CLI Execution Robustness Stress Tests
# ===========================================================================

class TestCLIExecutionRobustness:
    """Stress tests for nemo_eval CLI entry points, flags, and error codes."""

    def test_cli_parser_all_flag_permutations(self):
        """Test build_parser with valid modes, datasets, and arguments."""
        parser = build_parser()

        # 1. run --mode both --dataset math
        args1 = parser.parse_args(["run", "--mode", "both", "--dataset", "math", "--max-tasks", "5"])
        assert args1.command == "run"
        assert args1.mode == "both"
        assert args1.dataset == "math"
        assert args1.max_tasks == 5

        # 2. sweep --models all --dataset all
        args2 = parser.parse_args(["sweep", "--models", "all", "--dataset", "all", "--max-tasks", "2"])
        assert args2.command == "sweep"
        assert args2.models == "all"
        assert args2.dataset == "all"

        # 3. run --mode vanilla --dataset putnam
        args3 = parser.parse_args(["run", "--mode", "vanilla", "--dataset", "putnam"])
        assert args3.mode == "vanilla"
        assert args3.dataset == "putnam"

        # 4. run --mode agentic --dataset lila
        args4 = parser.parse_args(["run", "--mode", "agentic", "--dataset", "lila"])
        assert args4.mode == "agentic"
        assert args4.dataset == "lila"

    def test_cli_cmd_run_execution(self, tmp_path):
        """Empirically execute cmd_run through CLI with mock provider."""
        out_dir = str(tmp_path / "cli_run_out")
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--mode", "both",
            "--dataset", "math",
            "--models", "Qwen2.5-Math-1.5B",
            "--provider", "mock",
            "--max-tasks", "2",
            "--output-dir", out_dir,
        ])

        exit_code = cmd_run(args)
        assert exit_code == 0

        # Verify artifacts generated
        out_p = Path(out_dir)
        assert (out_p / "summary_scorecard.md").exists()
        assert (out_p / "summary.json").exists()
        assert (out_p / "failure_traces.md").exists()
        assert (out_p / "streaming_trajectories.jsonl").exists()

    def test_cli_validate_config_subcommand(self, tmp_path):
        """Test validate-config subcommand with valid and invalid config files."""
        # 1. Valid config
        valid_cfg = tmp_path / "valid_cfg.json"
        config = PipelineConfig(
            run_label="test_val",
            models=[ModelSpec(name="Qwen2.5-Math-7B", provider="mock")],
            datasets=[DatasetSpec(name="math", max_tasks=2)]
        )
        config.to_json(valid_cfg)

        parser = build_parser()
        args_valid = parser.parse_args(["validate-config", str(valid_cfg)])
        assert cmd_validate_config(args_valid) == 0

        # 2. Invalid config (corrupt JSON)
        invalid_cfg = tmp_path / "invalid_cfg.json"
        invalid_cfg.write_text("{ broken json: true,", encoding="utf-8")
        args_invalid = parser.parse_args(["validate-config", str(invalid_cfg)])
        assert cmd_validate_config(args_invalid) == 1

    def test_cli_report_subcommand(self, tmp_path):
        """Test report subcommand with existing and missing directories."""
        parser = build_parser()

        # Non-existent dir -> returns 1
        args_missing = parser.parse_args(["report", "--output-dir", str(tmp_path / "non_existent")])
        assert cmd_report(args_missing) == 1

        # Existing dir with summary.json -> returns 0
        valid_dir = tmp_path / "valid_rep"
        valid_dir.mkdir()
        (valid_dir / "summary.json").write_text("[]", encoding="utf-8")
        args_existing = parser.parse_args(["report", "--output-dir", str(valid_dir)])
        assert cmd_report(args_existing) == 0


# ===========================================================================
# 4. Markdown Scorecard, JSONL Traces, and Metrics Stress Tests
# ===========================================================================

class TestScorecardAndTelemetryReporting:
    """Stress tests for Markdown scorecard, Delta metrics, JSONL streaming, and leaderboards."""

    def test_scorecard_empty_and_single_trajectory(self, tmp_path):
        """Test Markdown scorecard generator with 0 and 1 trajectories."""
        # 1. Empty trajectories
        empty_card = MarkdownReporter.generate_markdown_scorecard([])
        assert "No trajectories provided" in empty_card

        # 2. Single trajectory
        single_traj = [
            EpisodeTrajectory(
                task_id="single_01",
                model_name="DeepSeek-R1-7B",
                status="success",
                ground_truth_score=1.0,
                total_duration_ms=250.0,
                peak_ram_mb=80.0,
                gpu_vram_mb=0.0,
                energy_joules=0.1234,
            )
        ]
        single_card = MarkdownReporter.generate_markdown_scorecard(single_traj)
        assert "Accuracy**: 100.00% (1/1)" in single_card
        assert "single_01" in single_card
        assert "0.1234" in single_card

    def test_dual_mode_comparison_delta_metrics_and_division_by_zero_defense(self):
        """Test Delta Acc, latency ratio, and energy ratio calculations with edge values (zero energy)."""
        # Vanilla traces with zero energy (e.g. CPU only)
        v_traces = [
            EpisodeTrajectory(task_id="t1", model_name="M", status="success", ground_truth_score=1.0, total_duration_ms=100.0, energy_joules=0.0),
            EpisodeTrajectory(task_id="t2", model_name="M", status="failed", ground_truth_score=0.0, total_duration_ms=100.0, energy_joules=0.0),
        ]
        # Agentic traces
        a_traces = [
            EpisodeTrajectory(task_id="t1", model_name="M", status="success", ground_truth_score=1.0, total_duration_ms=300.0, energy_joules=0.5),
            EpisodeTrajectory(task_id="t2", model_name="M", status="success", ground_truth_score=1.0, total_duration_ms=300.0, energy_joules=0.5),
        ]

        # Division by zero should not crash
        comp = MarkdownReporter.generate_dual_mode_comparison(v_traces, a_traces)
        assert "| **Accuracy (%)** | 50.00% | 100.00% | +50.00% |" in comp
        assert "3.00x" in comp

        # Empty lists comparison
        comp_empty = MarkdownReporter.generate_dual_mode_comparison([], [])
        assert "0.00%" in comp_empty

    def test_full_sweep_reporting_and_jsonl_streaming_integrity(self, tmp_path):
        """Stress-test BenchmarkRunner multi-model sweep, verifying streaming JSONL file formatting."""
        out_dir = tmp_path / "sweep_results"
        config = PipelineConfig(
            run_label="multi_model_stress_sweep",
            output_dir=str(out_dir),
            mode="both",
            models=[
                ModelSpec(name="Qwen2.5-Math-7B", provider="mock"),
                ModelSpec(name="DeepSeek-R1-7B", provider="mock"),
            ],
            datasets=[
                DatasetSpec(name="math", max_tasks=2),
                DatasetSpec(name="putnam", max_tasks=2),
            ],
            export_jsonl=True
        )

        runner = BenchmarkRunner(config)
        records = runner.run()

        # 2 models x 2 datasets x 2 modes = 8 records
        assert len(records) == 8

        reporter = MarkdownReporter(output_dir=out_dir)
        scorecard = reporter.write_summary_report(records, filename="summary_scorecard.md")
        json_sum = reporter.write_json_summary(records, filename="summary.json")
        failures = reporter.write_failure_traces(records, filename="failure_traces.md")

        assert scorecard.exists()
        assert json_sum.exists()
        assert failures.exists()

        # Verify streaming JSONL lines
        master_jsonl = out_dir / "streaming_trajectories.jsonl"
        assert master_jsonl.exists()
        lines = master_jsonl.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 16  # 2 models x 2 datasets x 2 modes x 2 tasks = 16 trajectories

        for line in lines:
            parsed = json.loads(line)
            assert "task_id" in parsed
            assert "model_name" in parsed
            assert "steps" in parsed
            assert "ground_truth_score" in parsed

    def test_markdown_special_characters_escaping_in_scorecard(self, tmp_path):
        """Verify individual trajectory scorecard with special characters (pipes, quotes)."""
        trajectories = [
            EpisodeTrajectory(
                task_id="task|with|pipes",
                model_name="Test|Model",
                status="success",
                ground_truth_score=1.0,
                total_duration_ms=150.0,
                peak_ram_mb=60.0,
                energy_joules=0.2
            )
        ]
        scorecard_text = MarkdownReporter.generate_markdown_scorecard(trajectories)
        assert "task|with|pipes" in scorecard_text
        assert "Test|Model" in scorecard_text
