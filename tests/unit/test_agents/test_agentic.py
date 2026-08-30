"""
tests/unit/test_agents/test_agentic.py
--------------------------------------
Unit tests for AgenticEngine and AgentLoop integration (Milestone 3).
"""

import pytest
from nemo_eval.agents.agent_loop import AgenticEngine, AgentConfig, AgentLoop
from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.models.mock_runner import DeterministicMockLLMClient
from nemo_eval.telemetry.tracer import EpisodeTrajectory, TrajectoryState


class TestAgenticEngine:
    def test_agentic_engine_initialization(self):
        client = DeterministicMockLLMClient("Qwen2.5-Math-7B")
        engine = AgenticEngine(model=client)
        assert engine.model == client
        assert isinstance(engine.config, AgentConfig)

    def test_agentic_engine_evaluate_task_lifecycle(self):
        client = DeterministicMockLLMClient("DeepSeek-R1-7B")
        engine = AgenticEngine(model=client)
        task = BenchmarkTask(
            task_id="ag_task_1",
            dataset_name="math",
            subdiscipline="Arithmetic",
            problem_text="Evaluate 25 * 4",
            ground_truth="100",
            eval_type="exact"
        )
        traj = engine.evaluate_task(task)

        assert isinstance(traj, EpisodeTrajectory)
        assert traj.task_id == "ag_task_1"
        states = [s.state for s in traj.steps]
        assert TrajectoryState.PLANNING in states
        assert TrajectoryState.ACTION_SELECTION in states
        assert len(traj.steps) >= 3

    def test_agentic_engine_max_turns_override(self):
        client = DeterministicMockLLMClient("Llama3.2-3B")
        engine = AgenticEngine(model=client)
        task = BenchmarkTask(
            task_id="ag_task_max",
            dataset_name="putnam",
            subdiscipline="Algebra",
            problem_text="Difficult problem",
            ground_truth="1",
            eval_type="exact"
        )
        traj = engine.evaluate_task(task, max_turns=2)
        assert traj.status in ("failed", "max_turns_exceeded")
        assert traj.total_steps > 0

    def test_agentic_engine_ground_truth_matching(self):
        client = DeterministicMockLLMClient("Phi4-mini-reasoning")
        engine = AgenticEngine(model=client)
        task = BenchmarkTask(
            task_id="ag_task_gt",
            dataset_name="lila",
            subdiscipline="Arithmetic",
            problem_text="Calculate 10 + 10",
            ground_truth="20",
            eval_type="exact"
        )
        traj = engine.evaluate_task(task)
        assert traj.ground_truth_score in (0.0, 1.0)
        assert traj.total_duration_ms >= 0.0
        assert traj.peak_ram_mb > 0.0
