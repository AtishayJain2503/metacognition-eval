"""
tests/unit/test_agents/test_vanilla.py
--------------------------------------
Unit tests for VanillaEngine (Milestone 3).
"""

import pytest
from nemo_eval.agents.vanilla import VanillaEngine, BaseEvaluationEngine
from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.models.mock_runner import DeterministicMockLLMClient
from nemo_eval.telemetry.tracer import EpisodeTrajectory, TrajectoryState


class TestVanillaEngine:
    def test_vanilla_engine_inheritance(self):
        engine = VanillaEngine()
        assert isinstance(engine, BaseEvaluationEngine)

    def test_vanilla_engine_single_turn_execution(self):
        mock_model = DeterministicMockLLMClient(
            model_name="Qwen2.5-Math-7B",
            responses={"Solve for x": "Step 1: Simplify.\nFinal answer: \\boxed{42}"}
        )
        engine = VanillaEngine(model=mock_model)
        task = BenchmarkTask(
            task_id="v_task_1",
            dataset_name="math",
            subdiscipline="Algebra",
            problem_text="Solve for x: 2*x = 84",
            ground_truth="42",
            eval_type="math_symbolic"
        )
        traj = engine.evaluate_task(task)

        assert mock_model.turn_counter == 1
        assert traj.task_id == "v_task_1"
        assert traj.final_answer == "42"
        assert getattr(traj, "raw_completion", None) == "Step 1: Simplify.\nFinal answer: \\boxed{42}"
        assert traj.ground_truth_score == 1.0
        assert traj.status == "success"
        assert traj.plan_adherence_score == 1.0
        assert traj.tool_accuracy == 1.0

    def test_vanilla_engine_zero_tool_steps(self):
        mock_model = DeterministicMockLLMClient(model_name="Llama3.2-3B")
        engine = VanillaEngine(model=mock_model)
        task = BenchmarkTask(
            task_id="v_task_2",
            dataset_name="math",
            subdiscipline="Arithmetic",
            problem_text="Calculate 100 + 200",
            ground_truth="300",
            eval_type="exact"
        )
        traj = engine.evaluate_task(task)

        for step in traj.steps:
            assert step.state != TrajectoryState.TOOL_EXECUTION
            assert step.state != TrajectoryState.SELF_CORRECTION

    def test_vanilla_engine_hardware_telemetry_recording(self):
        mock_model = DeterministicMockLLMClient(model_name="Phi4-mini-reasoning")
        engine = VanillaEngine(model=mock_model, enable_telemetry=True)
        task = BenchmarkTask(
            task_id="v_task_3",
            dataset_name="putnam",
            subdiscipline="Calculus",
            problem_text="Find limit",
            ground_truth="0",
            eval_type="exact"
        )
        traj = engine.evaluate_task(task)

        assert traj.total_duration_ms >= 0.0
        assert traj.peak_ram_mb > 0.0
        assert traj.energy_joules >= 0.0

    def test_vanilla_engine_incorrect_answer_scoring(self):
        mock_model = DeterministicMockLLMClient(
            model_name="Qwen2.5-Math-1.5B",
            responses={"Calculate": "The answer is \\boxed{999}"}
        )
        engine = VanillaEngine(model=mock_model)
        task = BenchmarkTask(
            task_id="v_task_4",
            dataset_name="math",
            subdiscipline="Arithmetic",
            problem_text="Calculate 2 + 2",
            ground_truth="4",
            eval_type="exact"
        )
        traj = engine.evaluate_task(task)

        assert traj.ground_truth_score == 0.0
        assert traj.status == "failed"

    def test_vanilla_engine_exception_handling(self):
        mock_model = DeterministicMockLLMClient(
            model_name="DeepSeek-R1-7B",
            inject_errors=[RuntimeError("API Network Timeout")]
        )
        engine = VanillaEngine(model=mock_model)
        task = BenchmarkTask(
            task_id="v_task_err",
            dataset_name="math",
            subdiscipline="Algebra",
            problem_text="Fail query",
            ground_truth="1",
            eval_type="exact"
        )
        traj = engine.evaluate_task(task)

        assert traj.status == "failed"
        assert traj.ground_truth_score == 0.0
