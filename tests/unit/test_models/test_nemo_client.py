"""
tests.unit.test_models.test_nemo_client
---------------------------------------
Unit tests for NeMoClient, NVIDIA NIM microservice interface, special token parsing
(<|begin_of_thought|>, <|tool_call|>), and Guardrail metadata ingestion.
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
    LLMTimeoutError,
)
from nemo_eval.models.nemo_client import (
    NeMoClient,
    extract_nemo_special_tokens,
)


class TestNeMoSpecialTokens:
    """Tests for NVIDIA special delimiter token parsing."""

    def test_extract_thought_tokens(self):
        text = "<|begin_of_thought|>\nAnalyze schema first.\n<|end_of_thought|>\nProceed with query."
        reasoning, tools, cleaned = extract_nemo_special_tokens(text)
        assert reasoning == "Analyze schema first."
        assert len(tools) == 0
        assert cleaned == "Proceed with query."

    def test_extract_tool_call_tokens(self):
        text = '<|tool_call|>{"name": "sqlite_query", "arguments": {"query": "SELECT * FROM t;"}}<|/tool_call|>'
        reasoning, tools, cleaned = extract_nemo_special_tokens(text)
        assert reasoning is None
        assert len(tools) == 1
        assert tools[0].name == "sqlite_query"
        assert tools[0].arguments == {"query": "SELECT * FROM t;"}
        assert cleaned is None

    def test_extract_both_thought_and_tool(self):
        text = """<|begin_of_thought|>
Step 1: Check table definition.
<|end_of_thought|>
<|tool_call|>
{"name": "sqlite_schema", "arguments": {"table_name": "products"}}
<|/tool_call|>"""
        reasoning, tools, cleaned = extract_nemo_special_tokens(text)
        assert "Step 1: Check table definition." in reasoning
        assert len(tools) == 1
        assert tools[0].name == "sqlite_schema"
        assert tools[0].arguments == {"table_name": "products"}

    def test_extract_unclosed_thought(self):
        text = "<|begin_of_thought|>Thinking cut off midway"
        reasoning, tools, cleaned = extract_nemo_special_tokens(text)
        assert reasoning == "Thinking cut off midway"
        assert cleaned is None


class TestNeMoClient:
    """Tests for NeMoClient NIM microservice communication and telemetry ingestion."""

    def test_nemo_init_and_headers(self):
        client = NeMoClient(
            model_name="nvidia/llama-3.1-nemotron-70b-instruct",
            api_key="nvapi-test-key",
            guardrails_enabled=True
        )
        assert client.model_name == "nvidia/llama-3.1-nemotron-70b-instruct"
        assert client.guardrails_enabled is True

    def test_nemo_successful_generation_with_guardrails(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get("x-nvidia-guardrails") == "enabled"
            resp_payload = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "<|begin_of_thought|>Finding active categories.<|end_of_thought|>There are 3 categories."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 60, "completion_tokens": 25},
                "guardrails": {"input_moderation": "PASS", "latency_ms": 1.2}
            }
            return httpx.Response(200, json=resp_payload, headers={"nvcf-reqid": "nvcf-req-456"})

        client = NeMoClient(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://integrate.api.nvidia.com/v1")
        )
        resp = client.generate(sample_messages)
        assert resp.content == "There are 3 categories."
        assert resp.reasoning_content == "Finding active categories."
        assert resp.raw_response is not None
        assert "guardrails" in resp.raw_response
        assert resp.raw_response["guardrails"]["nvcf_reqid"] == "nvcf-req-456"
        assert resp.raw_response["guardrails"]["input_moderation"] == "PASS"

    def test_nemo_tool_tokens_in_response(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            resp_payload = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": '<|tool_call|>{"name": "sqlite_schema", "arguments": {}}<|/tool_call|>'
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 50, "completion_tokens": 15}
            }
            return httpx.Response(200, json=resp_payload)

        client = NeMoClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://integrate.api.nvidia.com/v1")
        )
        resp = client.generate(sample_messages)
        assert resp.has_tool_calls is True
        assert resp.tool_calls[0].name == "sqlite_schema"

    def test_nemo_auth_failure(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="NVIDIA API Key Invalid")

        client = NeMoClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://integrate.api.nvidia.com/v1")
        )
        with pytest.raises(LLMAuthenticationError) as exc_info:
            client.generate(sample_messages)
        assert exc_info.value.provider == "nemo"

    @pytest.mark.anyio
    async def test_nemo_async_agenerate(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Async NIM Response"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5}
            })

        async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://integrate.api.nvidia.com/v1"
        )
        client = NeMoClient(async_http_client=async_client)
        resp = await client.agenerate(sample_messages)
        assert resp.content == "Async NIM Response"
        assert resp.total_tokens == 25
