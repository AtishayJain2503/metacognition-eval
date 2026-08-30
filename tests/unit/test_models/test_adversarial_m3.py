"""
tests.unit.test_models.test_adversarial_m3
------------------------------------------
Adversarial stress test suite for Milestone 3 (Model Provider Interfaces):
1. DeterministicMockLLMClient: multi-turn scenarios, response queue depletion, regex pattern fallbacks, error injection.
2. Groq: <think> token isolation regex with malformed/unclosed thinking tags, massive tokens, empty tags, multiple blocks.
3. OpenAI Gateway: text fallback tool extraction with broken XML/markdown, non-tool JSON, nested payloads.
4. NeMo Client: special token parsing (<|begin_of_thought|>, <|tool_call|>), malformed tags, guardrail telemetry.
"""

from __future__ import annotations

import asyncio
import json
import re
import pytest
import httpx

from nemo_eval.models.base import (
    BaseLLMClient,
    FunctionCall,
    LLMAuthenticationError,
    LLMContextLengthExceededError,
    LLMInvalidResponseError,
    LLMMessage,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    ModelConfig,
    ModelRegistry,
    ToolCall,
    get_model_client,
)
from nemo_eval.models.groq import GroqLLMClient, extract_think_reasoning
from nemo_eval.models.mock_runner import DeterministicMockLLMClient
from nemo_eval.models.nemo_client import NeMoClient, extract_nemo_special_tokens
from nemo_eval.models.openai_gateway import OpenAIGatewayClient, extract_text_fallback_tool_calls


# ===========================================================================
# 1. DeterministicMockLLMClient Adversarial Stress Tests
# ===========================================================================

class TestAdversarialMockRunner:
    """Stress tests for DeterministicMockLLMClient."""

    def test_multi_turn_long_horizon_and_reset(self):
        """Stress-test a long horizon execution (50 turns) and complete state reset."""
        client = DeterministicMockLLMClient(script_mode="perfect")

        for turn in range(1, 51):
            msg = [LLMMessage(role="user", content=f"Turn query {turn}")]
            resp = client.generate(msg, test_param=f"val_{turn}")
            assert isinstance(resp, LLMResponse)
            assert client.turn_counter == turn
            assert len(client.recorded_requests) == turn
            assert len(client.history_log) == turn

        client.assert_turn_count(50)
        last_req = client.get_last_request()
        assert last_req is not None
        assert last_req["turn"] == 50
        assert last_req["kwargs"]["test_param"] == "val_50"

        # Test assert_turn_count failure
        with pytest.raises(AssertionError, match="Expected 10 turns, but recorded 50 turns"):
            client.assert_turn_count(10)

        # Test reset
        client.reset()
        assert client.turn_counter == 0
        assert len(client.recorded_requests) == 0
        assert len(client.history_log) == 0
        assert client.get_last_request() is None

        # Post-reset run
        resp = client.generate([LLMMessage(role="user", content="Fresh query")])
        assert client.turn_counter == 1
        assert resp.content == "I will inspect the database schema."

    def test_response_queue_depletion_and_fallback(self):
        """Test behavior when explicit response queue is exhausted, then extended."""
        q = [
            LLMResponse(content="Turn 1 Response", tool_calls=[]),
            {"content": "Turn 2 Dict Response", "tool_calls": []},
        ]
        client = DeterministicMockLLMClient(response_queue=q)

        # Turn 1: pop from queue
        r1 = client.generate([LLMMessage(role="user", content="Q1")])
        assert r1.content == "Turn 1 Response"

        # Turn 2: pop dict from queue
        r2 = client.generate([LLMMessage(role="user", content="Q2")])
        assert r2.content == "Turn 2 Dict Response"

        # Turn 3: Queue is now depleted -> falls back to default
        r3 = client.generate([LLMMessage(role="user", content="Q3")])
        assert r3.content == "Mock response for turn 3"

        # Turn 4: Dynamic addition to queue
        client.add_response(LLMResponse(content="Turn 4 Dynamically Added", tool_calls=[]))
        client.add_response({"content": "Turn 5 Dict Added", "tool_calls": []})

        r4 = client.generate([LLMMessage(role="user", content="Q4")])
        assert r4.content == "Turn 4 Dynamically Added"

        r5 = client.generate([LLMMessage(role="user", content="Q5")])
        assert r5.content == "Turn 5 Dict Added"

        # Turn 6: Depleted again
        r6 = client.generate([LLMMessage(role="user", content="Q6")])
        assert r6.content == "Mock response for turn 6"

    def test_pattern_handlers_multi_turn_context_and_callable(self):
        """Test regex pattern handlers with multi-turn history search, callables, and fallbacks."""
        def custom_handler(messages: list[LLMMessage]) -> LLMResponse:
            last_msg = messages[-1].content or ""
            return LLMResponse(
                content=f"Custom handled: {last_msg.upper()}",
                tool_calls=[ToolCall(name="calculator", arguments={"expr": "1+1"})]
            )

        client = DeterministicMockLLMClient(
            script_mode=None,
            pattern_handlers={
                r"SELECT\s+\*\s+FROM\s+users": LLMResponse(content="Matched SQL query"),
                r"ERROR:\s+missing column": {"content": "Matched recovery hint", "tool_calls": []},
                r"calc:.*": custom_handler,
            }
        )

        # Match Pattern 1
        r1 = client.generate([LLMMessage(role="user", content="Please run: SELECT * FROM users;")])
        assert r1.content == "Matched SQL query"

        # Match Pattern 2
        r2 = client.generate([
            LLMMessage(role="user", content="Run query"),
            LLMMessage(role="assistant", content="Query executed"),
            LLMMessage(role="tool", content="ERROR: missing column 'id'"),
        ])
        assert r2.content == "Matched recovery hint"

        # Match Pattern 3 (callable)
        r3 = client.generate([LLMMessage(role="user", content="calc: 42 * 2")])
        assert r3.content == "Custom handled: CALC: 42 * 2"
        assert len(r3.tool_calls) == 1
        assert r3.tool_calls[0].name == "calculator"

        # Fallback when no pattern matches
        r4 = client.generate([LLMMessage(role="user", content="Unmatched message")])
        assert "Mock response for turn 4" in (r4.content or "")

    def test_empty_messages_and_none_content_handling(self):
        """Test mock runner handling edge-case messages (empty list, None content)."""
        client = DeterministicMockLLMClient(
            script_mode=None,
            pattern_handlers={r"keyword": LLMResponse(content="Found keyword")}
        )

        # 1. Empty message list
        r1 = client.generate([])
        assert r1.content == "Mock response for turn 1"

        # 2. Messages with content=None (e.g. tool call message)
        r2 = client.generate([
            LLMMessage(role="assistant", tool_calls=[ToolCall(name="sqlite_query", arguments={})]),
            LLMMessage(role="tool", content=None, tool_call_id="call_123")
        ])
        assert r2.content == "Mock response for turn 2"

    def test_error_injection_lifecycle(self):
        """Test error injection sequence with None passthroughs and recovery."""
        errors: list[Exception | None] = [
            LLMTimeoutError("Injected Timeout 1"),
            None,
            LLMRateLimitError("Injected 429"),
            ValueError("Custom validation error"),
        ]
        client = DeterministicMockLLMClient(
            inject_errors=errors,
            response_queue=[LLMResponse(content="Passed turn", tool_calls=[])]
        )

        # Turn 1: Timeout error raised
        with pytest.raises(LLMTimeoutError, match="Injected Timeout 1"):
            client.generate([LLMMessage(role="user", content="Q1")])

        # Turn 2: None -> succeeds, pops response queue
        r2 = client.generate([LLMMessage(role="user", content="Q2")])
        assert r2.content == "Passed turn"

        # Turn 3: RateLimit error raised
        with pytest.raises(LLMRateLimitError, match="Injected 429"):
            client.generate([LLMMessage(role="user", content="Q3")])

        # Turn 4: ValueError raised
        with pytest.raises(ValueError, match="Custom validation error"):
            client.generate([LLMMessage(role="user", content="Q4")])

        # Turn 5: No more errors -> succeeds with fallback
        r5 = client.generate([LLMMessage(role="user", content="Q5")])
        assert "Mock response for turn" in (r5.content or "")

    @pytest.mark.anyio
    async def test_async_agenerate_parity(self):
        """Test async agenerate under multi-turn and error injection."""
        client = DeterministicMockLLMClient(
            inject_errors=[LLMAuthenticationError("Async Auth Fail"), None],
            response_queue=[LLMResponse(content="Async Response Success", tool_calls=[])]
        )

        with pytest.raises(LLMAuthenticationError):
            await client.agenerate([LLMMessage(role="user", content="Q1")])

        resp = await client.agenerate([LLMMessage(role="user", content="Q2")])
        assert resp.content == "Async Response Success"


# ===========================================================================
# 2. Groq <think> Token Isolation Adversarial Tests
# ===========================================================================

class TestAdversarialGroqThinkIsolation:
    """Stress tests for Groq <think> token isolation regex."""

    def test_extract_think_unclosed_truncated(self):
        """Test unclosed <think> tag when generation gets cut off mid-thought."""
        raw = "<think>Let me reason step 1: we need to find the max value. Step 2: compute sum"
        reasoning, content = extract_think_reasoning(raw)
        assert reasoning == "Let me reason step 1: we need to find the max value. Step 2: compute sum"
        assert content is None

    def test_extract_think_unclosed_with_prefix_text(self):
        """Test prefix text followed by unclosed <think>."""
        raw = "Beginning of output...\n<think>I am now thinking deeply about the problem"
        reasoning, content = extract_think_reasoning(raw)
        assert reasoning == "I am now thinking deeply about the problem"
        assert content == "Beginning of output..."

    def test_extract_think_empty_and_whitespace_only(self):
        """Test empty and whitespace-only <think> tags."""
        raw = "<think>   \n\t  </think> Final answer: 42"
        reasoning, content = extract_think_reasoning(raw)
        assert reasoning is None
        assert content == "Final answer: 42"

        raw_empty = "<think></think>"
        reasoning, content = extract_think_reasoning(raw_empty)
        assert reasoning is None
        assert content is None

    def test_extract_think_multiline_code_and_special_chars(self):
        """Test <think> block containing markdown, code snippets, regex characters, and unicode."""
        reasoning_body = (
            "```python\n"
            "def solve(query_str):\n"
            "    return [x**2 for x in range(10) if x > 2]\n"
            "```\n"
            "Special chars: $ ^ * + ? { } [ ] \\ | ( ) ! @ # % &\n"
            "Emoji: 🧠 🚀 🔍\n"
            "Chinese: 深度思考推理过程"
        )
        raw = f"<think>\n{reasoning_body}\n</think>\n\nFinal Solution: `[9, 16, 25, 36, 49, 64, 81]`"
        reasoning, content = extract_think_reasoning(raw)
        assert reasoning == reasoning_body
        assert content == "Final Solution: `[9, 16, 25, 36, 49, 64, 81]`"

    def test_extract_think_orphaned_closing_tag(self):
        """Test output containing orphaned </think> without opening tag."""
        raw = "Some reasoning done earlier </think> Final answer."
        reasoning, content = extract_think_reasoning(raw)
        assert reasoning is None
        assert content == raw

    def test_extract_think_massive_payload(self):
        """Stress test with 50,000 characters inside <think>."""
        big_thought = "Thinking step " * 3500  # ~49,000 chars
        raw = f"<think>{big_thought}</think>Result"
        reasoning, content = extract_think_reasoning(raw)
        assert reasoning == big_thought.strip()
        assert content == "Result"

    def test_extract_think_none_and_whitespace_only_strings(self):
        """Test None, empty string, and whitespace string inputs."""
        assert extract_think_reasoning(None) == (None, None)
        assert extract_think_reasoning("") == (None, None)
        assert extract_think_reasoning("    \n\t   ") == (None, None)

    def test_groq_client_unclosed_think_parsing(self, monkeypatch):
        """Test GroqLLMClient handling unclosed <think> from mock HTTP response."""
        mock_response_data = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "deepseek-r1-distill-llama-70b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>Processing interrupted at max_tokens"
                    },
                    "finish_reason": "length"
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        }

        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        client = GroqLLMClient(api_key="mock_key")
        monkeypatch.setattr(
            client._http_client,
            "post",
            lambda *args, **kwargs: httpx.Response(200, json=mock_response_data, request=req)
        )

        resp = client.generate([LLMMessage(role="user", content="Solve hard problem")])
        assert resp.reasoning_content == "Processing interrupted at max_tokens"
        assert resp.content is None
        assert resp.finish_reason == "length"

    def test_groq_client_empty_choices_raises_invalid_response(self, monkeypatch):
        """Test GroqLLMClient raising LLMInvalidResponseError on empty choices payload."""
        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        client = GroqLLMClient(api_key="mock_key")
        monkeypatch.setattr(
            client._http_client,
            "post",
            lambda *args, **kwargs: httpx.Response(200, json={"choices": []}, request=req)
        )

        with pytest.raises(LLMInvalidResponseError, match="Groq response contained no choices"):
            client.generate([LLMMessage(role="user", content="Hello")])


# ===========================================================================
# 3. OpenAI Gateway Text Fallback Tool Extraction Adversarial Tests
# ===========================================================================

class TestAdversarialOpenAITextFallback:
    """Stress tests for OpenAI Gateway markdown and XML text fallback tool calling."""

    def test_extract_fallback_broken_json_in_xml(self):
        """Test broken JSON syntax inside <tool_call> tags (should not crash or extract garbage)."""
        raw = "I will call the tool: <tool_call>{name: 'broken_syntax', unquoted_val: 123}</tool_call> and continue."
        calls, cleaned = extract_text_fallback_tool_calls(raw)
        assert len(calls) == 0
        assert "broken_syntax" in (cleaned or "")

    def test_extract_fallback_broken_json_in_markdown(self):
        """Test broken JSON syntax in markdown code block."""
        raw = "```json\n{ tool = 'calculator', arguments = invalid }\n```\nHere is what happened."
        calls, cleaned = extract_text_fallback_tool_calls(raw)
        assert len(calls) == 0
        assert "calculator" in (cleaned or "")

    def test_extract_fallback_non_tool_json_code_block(self):
        """Test valid JSON in markdown that is NOT a tool call (e.g. data dictionary)."""
        raw = (
            "Here is the summary table:\n"
            "```json\n"
            "{\n"
            '  "status": "success",\n'
            '  "rows_count": 100,\n'
            '  "columns": ["id", "val"]\n'
            "}\n"
            "```\n"
            "Let me know if you need more details."
        )
        calls, cleaned = extract_text_fallback_tool_calls(raw)
        assert len(calls) == 0
        assert "rows_count" in (cleaned or "")

    def test_extract_fallback_code_block_preceded_by_python_script(self):
        """Test extraction when a Python code block precedes a JSON tool call code block."""
        raw = (
            "Here is the script I wrote:\n"
            "```python\n"
            "def compute(x):\n"
            "    return x * 2\n"
            "```\n"
            "Now executing via tool:\n"
            "```json\n"
            "{\"name\": \"python_repl\", \"arguments\": {\"code\": \"compute(10)\"}}\n"
            "```"
        )
        calls, cleaned = extract_text_fallback_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].name == "python_repl"
        assert calls[0].arguments == {"code": "compute(10)"}
        assert "def compute(x):" in (cleaned or "")

    def test_extract_fallback_mixed_xml_and_markdown(self):
        """Test extraction when both XML and Markdown tool calls are present."""
        raw = (
            "Step 1: <tool_call>{\"name\": \"sqlite_schema\", \"arguments\": {\"table\": \"orders\"}}</tool_call>\n"
            "Step 2: ```json\n{\"tool\": \"python_repl\", \"parameters\": {\"code\": \"print('hello')\"}}\n```\n"
            "Step 3: <|tool_call|>{\"function\": \"tabular_inspect\", \"arguments\": {\"file_path\": \"data.csv\"}}<|/tool_call|>"
        )
        calls, cleaned = extract_text_fallback_tool_calls(raw)
        assert len(calls) == 3
        names = [c.name for c in calls]
        assert "sqlite_schema" in names
        assert "python_repl" in names
        assert "tabular_inspect" in names

    def test_extract_fallback_list_of_tool_calls_in_xml(self):
        """Test list of multiple tool call objects inside a single <tool_call> tag."""
        payload = [
            {"name": "tool_a", "arguments": {"x": 1}},
            {"tool": "tool_b", "parameters": {"y": 2}},
        ]
        raw = f"<tool_call>{json.dumps(payload)}</tool_call> Remaining explanation."
        calls, cleaned = extract_text_fallback_tool_calls(raw)
        assert len(calls) == 2
        assert calls[0].name == "tool_a"
        assert calls[0].arguments == {"x": 1}
        assert calls[1].name == "tool_b"
        assert calls[1].arguments == {"y": 2}
        assert cleaned == "Remaining explanation."

    def test_extract_fallback_nested_complex_json_arguments(self):
        """Test tool call with heavily nested structures and special characters in arguments."""
        nested_args = {
            "query": "SELECT * FROM t WHERE str LIKE '%<tool_call>%'",
            "config": {
                "depth": 3,
                "flags": ["a", "b", "c"],
                "metadata": {"source": "test_src", "active": True}
            }
        }
        raw = f"<|tool_call|>{json.dumps({'name': 'complex_tool', 'arguments': nested_args})}<|/tool_call|>"
        calls, cleaned = extract_text_fallback_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].name == "complex_tool"
        assert calls[0].arguments == nested_args
        assert cleaned is None

    def test_openai_gateway_fallback_disabled_flag(self, monkeypatch):
        """Test OpenAI gateway with enable_text_fallback_tool_calling=False."""
        raw_text = "<tool_call>{\"name\": \"sqlite_query\", \"arguments\": {\"query\": \"SELECT 1\"}}</tool_call>"
        mock_response_data = {
            "choices": [{"message": {"role": "assistant", "content": raw_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

        req = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
        client = OpenAIGatewayClient(enable_text_fallback_tool_calling=False)
        monkeypatch.setattr(
            client._http_client,
            "post",
            lambda *args, **kwargs: httpx.Response(200, json=mock_response_data, request=req)
        )
        resp = client.generate([LLMMessage(role="user", content="Run SQL")])
        assert len(resp.tool_calls) == 0
        assert resp.content == raw_text


# ===========================================================================
# 4. NeMo Client Special Token Parsing Adversarial Tests
# ===========================================================================

class TestAdversarialNeMoSpecialTokens:
    """Stress tests for NVIDIA NeMo special token parsing."""

    def test_nemo_unclosed_thought_token(self):
        """Test unclosed <|begin_of_thought|> token."""
        raw = "<|begin_of_thought|>Thinking about the optimal approach for data retrieval..."
        reasoning, tools, cleaned = extract_nemo_special_tokens(raw)
        assert reasoning == "Thinking about the optimal approach for data retrieval..."
        assert len(tools) == 0
        assert cleaned is None

    def test_nemo_interleaved_thought_and_tools_and_text(self):
        """Test combined thought tokens, tool call tokens, and remaining assistant text."""
        raw = (
            "<|begin_of_thought|>\n"
            "1. Need to inspect customers table.\n"
            "2. Issue sqlite_schema tool call.\n"
            "<|end_of_thought|>\n"
            "<|tool_call|>{\"name\": \"sqlite_schema\", \"arguments\": {\"table_name\": \"customers\"}}<|/tool_call|>\n"
            "I have dispatched the schema inspection call."
        )
        reasoning, tools, cleaned = extract_nemo_special_tokens(raw)
        assert reasoning is not None
        assert "Need to inspect customers table" in reasoning
        assert len(tools) == 1
        assert tools[0].name == "sqlite_schema"
        assert tools[0].arguments == {"table_name": "customers"}
        assert cleaned == "I have dispatched the schema inspection call."

    def test_nemo_malformed_tool_call_token(self):
        """Test malformed JSON inside NeMo <|tool_call|>."""
        raw = "<|tool_call|>{not_json: true}<|/tool_call|>Text after error."
        reasoning, tools, cleaned = extract_nemo_special_tokens(raw)
        assert reasoning is None
        assert len(tools) == 0
        assert "Text after error" in (cleaned or "")

    def test_nemo_guardrail_headers_and_moderation_metadata(self, monkeypatch):
        """Test NeMo client extraction of nvcf-reqid, guardrail info, and moderation payloads."""
        mock_response_data = {
            "id": "nemo-cmpl-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<|begin_of_thought|>Safe thinking<|end_of_thought|>Safe content"
                    },
                    "finish_reason": "stop"
                }
            ],
            "guardrails": {
                "input_violation": False,
                "output_violation": False,
                "interventions": []
            },
            "usage": {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55}
        }

        client = NeMoClient(api_key="mock_key", guardrails_enabled=True)
        headers = httpx.Headers({"nvcf-reqid": "req-guid-98765", "content-type": "application/json"})
        req = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")

        monkeypatch.setattr(
            client._http_client,
            "post",
            lambda *args, **kwargs: httpx.Response(200, json=mock_response_data, headers=headers, request=req)
        )

        resp = client.generate([LLMMessage(role="user", content="Safe prompt")])
        assert resp.content == "Safe content"
        assert resp.reasoning_content == "Safe thinking"
        assert resp.raw_response is not None
        assert resp.raw_response["guardrails"]["nvcf_reqid"] == "req-guid-98765"
        assert resp.raw_response["guardrails"]["input_violation"] is False


# ===========================================================================
# 5. Base Models Polymorphic ToolCall Serialization Adversarial Tests
# ===========================================================================

class TestAdversarialBaseDataContracts:
    """Stress tests for ToolCall, LLMMessage, and BaseLLMClient polymorphic behavior."""

    def test_tool_call_deep_nested_and_string_argument_fallbacks(self):
        """Test ToolCall initialization across irregular formats."""
        # 1. Invalid JSON string in arguments -> wraps in raw
        tc1 = ToolCall(name="calc", arguments="not a json string {")
        assert tc1.arguments == {"raw": "not a json string {"}
        assert tc1.function is not None
        assert tc1.function.name == "calc"

        # 2. Nested function object with string JSON arguments
        func = FunctionCall(name="query", arguments='{"sql": "SELECT 1"}')
        tc2 = ToolCall(function=func)
        assert tc2.name == "query"
        assert tc2.arguments == {"sql": "SELECT 1"}

        # 3. to_openai_dict serialization
        openai_d = tc2.to_openai_dict()
        assert openai_d["type"] == "function"
        assert openai_d["function"]["name"] == "query"
        assert json.loads(openai_d["function"]["arguments"]) == {"sql": "SELECT 1"}

    def test_model_registry_custom_class_registration(self):
        """Test ModelRegistry alias registration and dynamic instantiation."""
        class CustomClient(BaseLLMClient):
            def generate(self, messages, **kwargs):
                return LLMResponse(content="Custom Client Response")
            async def agenerate(self, messages, **kwargs):
                return self.generate(messages, **kwargs)

        ModelRegistry.register("custom-provider", CustomClient)
        client = get_model_client("custom-provider")
        resp = client.generate([LLMMessage(role="user", content="Hi")])
        assert resp.content == "Custom Client Response"
