"""
tests.unit.test_models.test_mock_runner
---------------------------------------
Unit tests for DeterministicMockLLMClient, 7 built-in multi-turn rule scenarios,
pattern matching dispatch, fault injection, and trajectory recording.
"""

import pytest

from nemo_eval.models.base import (
    LLMMessage,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    ToolCall,
)
from nemo_eval.models.mock_runner import DeterministicMockLLMClient


class TestDeterministicMockLLMClient:
    """Tests for DeterministicMockLLMClient all 5 modes and recording."""

    def test_mock_perfect_script_mode(self):
        client = DeterministicMockLLMClient(script_mode="perfect")

        # Turn 1: Schema inspection
        r1 = client.generate([LLMMessage(role="user", content="Find categories")])
        assert client.turn_counter == 1
        assert len(r1.tool_calls) == 1
        assert r1.tool_calls[0].name == "sqlite_schema"

        # Turn 2: Query execution
        r2 = client.generate([
            LLMMessage(role="user", content="Find categories"),
            LLMMessage(role="tool", content="categories(category_id, category_name)")
        ])
        assert client.turn_counter == 2
        assert len(r2.tool_calls) == 1
        assert r2.tool_calls[0].name == "sqlite_query"

        # Turn 3: Final Answer
        r3 = client.generate([
            LLMMessage(role="user", content="Find categories"),
            LLMMessage(role="tool", content="[(1, 'Electronics'), (2, 'Books'), (3, 'Home')]")
        ])
        assert client.turn_counter == 3
        assert r3.has_tool_calls is False
        assert "Final Answer: The database contains 3 product categories." in r3.content

    def test_mock_self_correction_script_mode(self):
        client = DeterministicMockLLMClient(script_mode="self_correction")

        # Turn 1: Syntax error
        r1 = client.generate([LLMMessage(role="user", content="Calculate sum")])
        assert r1.tool_calls[0].name == "python_repl"
        assert "test.csv'" in r1.tool_calls[0].arguments["code"]

        # Turn 2: Corrected code
        r2 = client.generate([
            LLMMessage(role="user", content="Calculate sum"),
            LLMMessage(role="tool", content="SyntaxError: unexpected EOF while parsing")
        ])
        assert r2.tool_calls[0].name == "python_repl"
        assert "test.csv')" in r2.tool_calls[0].arguments["code"]

        # Turn 3: Final Answer
        r3 = client.generate([
            LLMMessage(role="tool", content="150")
        ])
        assert r3.content == "Final Answer: 150"

    def test_mock_data_analytics_pipeline_mode(self):
        client = DeterministicMockLLMClient(script_mode="data_analytics_pipeline")

        r1 = client.generate([LLMMessage(role="user", content="Analyze churn")])
        assert r1.tool_calls[0].name == "tabular_inspect"

        r2 = client.generate([LLMMessage(role="tool", content="Columns: [churned, total_charges]")])
        assert r2.tool_calls[0].name == "python_repl"
        assert "fillna(0)" in r2.tool_calls[0].arguments["code"]

        r3 = client.generate([LLMMessage(role="tool", content="Filled nulls")])
        assert r3.tool_calls[0].name == "python_repl"
        assert "mean()" in r3.tool_calls[0].arguments["code"]

        r4 = client.generate([LLMMessage(role="tool", content="Churn rate: 0.375")])
        assert "37.5%" in r4.content

    def test_mock_infinite_loop_thrashing_mode(self):
        client = DeterministicMockLLMClient(script_mode="infinite_loop_thrashing")
        for _ in range(3):
            resp = client.generate([LLMMessage(role="user", content="Retry")])
            assert resp.tool_calls[0].name == "sqlite_query"
            assert "broken_table" in resp.tool_calls[0].arguments["query"]

    def test_mock_schema_mismatch_recovery_mode(self):
        client = DeterministicMockLLMClient(script_mode="schema_mismatch_recovery")

        r1 = client.generate([LLMMessage(role="user", content="Get charges")])
        assert "MonthlyFee" in r1.tool_calls[0].arguments["query"]

        r2 = client.generate([LLMMessage(role="tool", content="Error: no such column: MonthlyFee. Did you mean monthly_charges?")])
        assert "monthly_charges" in r2.tool_calls[0].arguments["query"]

        r3 = client.generate([LLMMessage(role="tool", content="[(77.40,)]")])
        assert "77.40" in r3.content

    def test_mock_parallel_tool_calls_mode(self):
        client = DeterministicMockLLMClient(script_mode="parallel_tool_calls")
        resp = client.generate([LLMMessage(role="user", content="Inspect all")])
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].arguments["table_name"] == "products"
        assert resp.tool_calls[1].arguments["table_name"] == "categories"

    def test_mock_deepseek_r1_reasoning_mode(self):
        client = DeterministicMockLLMClient(script_mode="deepseek_r1_reasoning")
        resp = client.generate([LLMMessage(role="user", content="Solve")])
        assert resp.content == "The result is 42."
        assert "break down the problem" in resp.reasoning_content

    def test_mock_explicit_response_queue(self):
        custom_responses = [
            LLMResponse(content="Custom Step 1"),
            LLMResponse(content="Custom Step 2"),
        ]
        client = DeterministicMockLLMClient(response_queue=custom_responses)

        r1 = client.generate([])
        assert r1.content == "Custom Step 1"
        r2 = client.generate([])
        assert r2.content == "Custom Step 2"

        # Queue exhausted -> default fallback
        r3 = client.generate([])
        assert "Mock response for turn 3" in r3.content

    def test_mock_add_response_helper(self):
        client = DeterministicMockLLMClient(response_queue=[])
        client.add_response({"content": "Dynamic Step"})
        resp = client.generate([])
        assert resp.content == "Dynamic Step"

    def test_mock_pattern_handlers_regex_and_callable(self):
        handlers = {
            r"schema": LLMResponse(content="Matched Schema Pattern", tool_calls=[ToolCall(name="sqlite_schema", arguments={})]),
            r"sum": lambda msgs: LLMResponse(content=f"Matched Sum with {len(msgs)} messages"),
        }
        client = DeterministicMockLLMClient(pattern_handlers=handlers)

        resp1 = client.generate([LLMMessage(role="user", content="Please inspect schema")])
        assert resp1.content == "Matched Schema Pattern"
        assert resp1.has_tool_calls is True

        resp2 = client.generate([LLMMessage(role="user", content="Compute the sum")])
        assert resp2.content == "Matched Sum with 1 messages"

    def test_mock_error_injection(self):
        errors = [
            LLMRateLimitError("Rate limit error", status_code=429),
            None # Success on turn 2
        ]
        client = DeterministicMockLLMClient(
            script_mode="perfect",
            inject_errors=errors
        )

        with pytest.raises(LLMRateLimitError):
            client.generate([LLMMessage(role="user", content="Run")])

        # Second call succeeds
        resp = client.generate([LLMMessage(role="user", content="Run retry")])
        assert resp.has_tool_calls is True
        assert resp.tool_calls[0].name == "sqlite_schema"

    def test_mock_turn_counter_and_recording_lifecycle(self):
        client = DeterministicMockLLMClient(script_mode="perfect")
        client.generate([LLMMessage(role="user", content="Msg 1")])
        client.generate([LLMMessage(role="user", content="Msg 2")])

        client.assert_turn_count(2)
        last_req = client.get_last_request()
        assert last_req is not None
        assert last_req["turn"] == 2
        assert len(client.recorded_requests) == 2

        # Reset
        client.reset()
        assert client.turn_counter == 0
        assert len(client.recorded_requests) == 0
        assert client.get_last_request() is None

    @pytest.mark.anyio
    async def test_mock_async_agenerate(self):
        client = DeterministicMockLLMClient(script_mode="perfect")
        resp = await client.agenerate([LLMMessage(role="user", content="Async turn")])
        assert resp.tool_calls[0].name == "sqlite_schema"
        assert client.turn_counter == 1
