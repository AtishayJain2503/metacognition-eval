"""
tests.unit.test_models.test_groq
--------------------------------
Unit tests for GroqLLMClient, <think> token isolation, rate limiting, and backoff.
"""

import json
import httpx
import pytest

from nemo_eval.models.base import (
    LLMAuthenticationError,
    LLMContextLengthExceededError,
    LLMMessage,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
)
from nemo_eval.models.groq import GroqLLMClient, extract_think_reasoning


class TestExtractThinkReasoning:
    """Tests for DeepSeek-R1 <think> token isolation."""

    def test_extract_think_reasoning_standard(self):
        raw = "<think>\nI should calculate the average.\n</think>\nThe average is 42.0."
        reasoning, content = extract_think_reasoning(raw)
        assert reasoning == "I should calculate the average."
        assert content == "The average is 42.0."

    def test_extract_think_reasoning_multiline_complex(self):
        raw = """<think>
1. First, inspect data.
2. Group by category.
3. Compute sum: 10 + 20 = 30.
</think>
Final Answer: 30"""
        reasoning, content = extract_think_reasoning(raw)
        assert "1. First, inspect data." in reasoning
        assert "3. Compute sum" in reasoning
        assert content == "Final Answer: 30"

    def test_extract_think_reasoning_unclosed_tag(self):
        raw = "<think>Generating partial reasoning that got cut off..."
        reasoning, content = extract_think_reasoning(raw)
        assert reasoning == "Generating partial reasoning that got cut off..."
        assert content is None

    def test_extract_think_reasoning_empty_tag(self):
        raw = "<think></think>Clean output without thinking."
        reasoning, content = extract_think_reasoning(raw)
        assert reasoning is None
        assert content == "Clean output without thinking."

    def test_extract_think_reasoning_no_tag(self):
        raw = "Direct output with no think tokens."
        reasoning, content = extract_think_reasoning(raw)
        assert reasoning is None
        assert content == "Direct output with no think tokens."

    def test_extract_think_reasoning_none_and_empty(self):
        assert extract_think_reasoning(None) == (None, None)
        assert extract_think_reasoning("") == (None, None)
        assert extract_think_reasoning("   ") == (None, None)


class TestGroqLLMClient:
    """Tests for GroqLLMClient request/response lifecycle and error recovery."""

    def test_groq_successful_generation_with_think(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/chat/completions")
            body = json.loads(request.content)
            assert body["model"] == "deepseek-r1-distill-llama-70b"
            assert len(body["messages"]) == 2

            resp_payload = {
                "id": "chatcmpl_test123",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>Counting rows.</think>There are 150 rows in the database."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 45,
                    "completion_tokens": 20,
                    "total_tokens": 65
                }
            }
            return httpx.Response(200, json=resp_payload)

        client = GroqLLMClient(
            model_name="deepseek-r1-distill-llama-70b",
            api_key="test-groq-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.groq.com/openai/v1")
        )

        resp = client.generate(sample_messages)
        assert resp.content == "There are 150 rows in the database."
        assert resp.reasoning_content == "Counting rows."
        assert resp.prompt_tokens == 45
        assert resp.completion_tokens == 20
        assert resp.total_tokens == 65
        assert resp.latency_ms > 0

    def test_groq_tool_calls_parsing(self, sample_messages, sample_tool_specs):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert "tools" in body
            resp_payload = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "sqlite_query",
                                "arguments": json.dumps({"query": "SELECT count(*) FROM tbl;"})
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 50, "completion_tokens": 15}
            }
            return httpx.Response(200, json=resp_payload)

        client = GroqLLMClient(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.groq.com/openai/v1")
        )
        resp = client.generate(sample_messages, tools=sample_tool_specs)
        assert resp.has_tool_calls is True
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "sqlite_query"
        assert resp.tool_calls[0].arguments == {"query": "SELECT count(*) FROM tbl;"}

    def test_groq_authentication_failure_401(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="Invalid API Key")

        client = GroqLLMClient(
            api_key="invalid-key",
            max_retries=3,
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.groq.com/openai/v1")
        )
        with pytest.raises(LLMAuthenticationError) as exc_info:
            client.generate(sample_messages)
        assert exc_info.value.status_code == 401
        assert exc_info.value.provider == "groq"

    def test_groq_rate_limit_backoff_and_recovery(self, sample_messages):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"retry-after": "0.01"}, text="Rate limit hit")
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Recovered"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })

        client = GroqLLMClient(
            max_retries=2,
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.groq.com/openai/v1")
        )
        resp = client.generate(sample_messages)
        assert attempts == 2
        assert resp.content == "Recovered"

    def test_groq_rate_limit_exhausted(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"retry-after": "0.01"}, text="Rate limit persistent")

        client = GroqLLMClient(
            max_retries=2,
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.groq.com/openai/v1")
        )
        with pytest.raises(LLMRateLimitError) as exc_info:
            client.generate(sample_messages)
        assert exc_info.value.status_code == 429

    def test_groq_context_length_error(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="maximum context length is 8192 tokens")

        client = GroqLLMClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.groq.com/openai/v1")
        )
        with pytest.raises(LLMContextLengthExceededError):
            client.generate(sample_messages)

    def test_groq_timeout_error_exhausted(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("Connection timed out")

        client = GroqLLMClient(
            max_retries=1,
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.groq.com/openai/v1")
        )
        with pytest.raises(LLMTimeoutError):
            client.generate(sample_messages)

    @pytest.mark.anyio
    async def test_groq_async_agenerate(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Async Groq Response"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4}
            })

        async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.groq.com/openai/v1"
        )
        client = GroqLLMClient(async_http_client=async_client)
        resp = await client.agenerate(sample_messages)
        assert resp.content == "Async Groq Response"
        assert resp.total_tokens == 16
