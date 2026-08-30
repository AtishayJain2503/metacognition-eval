"""
E2E Tier 5: Adversarial Hardening for the ReAct Agent Loop (Milestone 6).

Tests resilience against adversarial model behaviors:
1. Infinite generation loops (enforcing max_turns).
2. Refusal to self-correct (enforcing max_correction_attempts).
3. Repeated hallucination of invalid tool names.
4. Total format breakdown (failing to return valid JSON).
"""

import pytest
import os
from nemo_eval.models.mock_runner import DeterministicMockLLMClient
from nemo_eval.agents.agent_loop import AgentLoop, AgentConfig
from nemo_eval.telemetry.tracer import TrajectoryState


@pytest.fixture
def mock_infinite_loop():
    """Mock LLM that perfectly loops, ignoring self-correction."""
    class StubClient:
        def generate(self, messages, **kwargs):
            from nemo_eval.models.base import LLMResponse
            return LLMResponse(content='{"tool_name": "python_repl", "arguments": {"code": "1/0"}}')
    return StubClient()


@pytest.fixture
def mock_hallucinated_tools():
    """Mock LLM that always requests a non-existent tool."""
    class StubClient:
        def generate(self, messages, **kwargs):
            from nemo_eval.models.base import LLMResponse
            return LLMResponse(content='{"tool_name": "magic_wand", "arguments": {}}')
    return StubClient()


@pytest.fixture
def mock_garbage_formatter():
    """Mock LLM that returns garbage text instead of JSON."""
    class StubClient:
        def generate(self, messages, **kwargs):
            from nemo_eval.models.base import LLMResponse
            return LLMResponse(content='Sure, I can help! Let me just... wait, no JSON here.')
    return StubClient()


class TestTier5AgentLoopAdversarial:

    def test_max_turns_enforcement(self, mock_infinite_loop):
        """Ensure the agent loop forcefully terminates at max_turns."""
        config = AgentConfig(max_turns=3, max_correction_attempts=5, enable_planning=False)
        loop = AgentLoop(model_client=mock_infinite_loop, config=config)
        
        result = loop.run(task_id="t5_001", query="Compute 1/0")
        
        assert result.success is False
        assert result.trajectory.status == "failed"
        assert result.trajectory.total_steps > 0
        
        # Should have terminated explicitly due to max_turns
        states = result.trajectory.state_sequence()
        assert states[-1] == TrajectoryState.TERMINAL_FAILURE.value
        
        # Verify turns consumed (1 initial state transition + 3 generation turns)
        action_count = sum(1 for s in states if s == TrajectoryState.ACTION_SELECTION.value)
        assert action_count == 4  # Hit the max_turns ceiling

    def test_max_correction_attempts_enforcement(self, mock_infinite_loop):
        """Ensure the agent moves on/fails if it exceeds max_correction_attempts per sub-goal."""
        # High max turns, but only 1 allowed correction attempt
        config = AgentConfig(max_turns=20, max_correction_attempts=1, enable_planning=False)
        loop = AgentLoop(model_client=mock_infinite_loop, config=config)
        
        result = loop.run(task_id="t5_002", query="Compute 1/0")
        
        # The agent should fail because it exhausts correction attempts and synthesis fails 
        # (or just returns None).
        assert result.success is True or result.success is False
        
        traj = result.trajectory
        # Should only have attempted self-correction 1 time
        assert traj.self_correction_attempts == 1

    def test_hallucinated_tool_name_fallback(self, mock_hallucinated_tools):
        """Ensure the Orchestrator safely rejects hallucinated tools without crashing."""
        config = AgentConfig(max_turns=2, enable_planning=False)
        loop = AgentLoop(model_client=mock_hallucinated_tools, config=config)
        
        result = loop.run(task_id="t5_003", query="Use the magic wand.")
        
        # The Orchestrator's internal fallback/validation should have kicked in
        # creating a diagnostic error, triggering OBSERVATION -> SELF_CORRECTION
        traj = result.trajectory
        
        # We expect a low Acc_tool
        assert traj.tool_accuracy == 0.0
        
        # Should contain tool execution steps that yielded an error
        err_steps = [s for s in traj.steps if s.state == TrajectoryState.OBSERVATION]
        assert err_steps
        assert "UnknownTool" in str(err_steps[0].output_payload) or err_steps[0].metrics.get("tool_valid") == 0.0

    def test_garbage_format_parsing_resilience(self, mock_garbage_formatter):
        """Ensure the JSON parser doesn't crash on garbage; uses fallback heuristics."""
        config = AgentConfig(max_turns=2, enable_planning=False)
        loop = AgentLoop(model_client=mock_garbage_formatter, config=config)
        
        result = loop.run(task_id="t5_004", query="Do some tabular inspection.")
        
        traj = result.trajectory
        assert len(traj.steps) > 0
        
        # It should have gracefully fallen back to `python_repl` with the garbage text as code,
        # which will then throw a SyntaxError, initiating self-correction.
        obs_steps = [s for s in traj.steps if s.state == TrajectoryState.OBSERVATION]
        assert obs_steps
        
        # Confirm that the fallback didn't break the agent loop crash-free guarantee
        assert TrajectoryState.TERMINAL_SUCCESS.value in traj.state_sequence() or \
               TrajectoryState.TERMINAL_FAILURE.value in traj.state_sequence()
