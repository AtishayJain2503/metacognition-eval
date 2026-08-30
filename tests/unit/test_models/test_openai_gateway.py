"""
tests.unit.test_models.test_openai_gateway
------------------------------------------
Unit tests for OpenAIGatewayClient, markdown/XML text fallback tool calling,
and OpenAI-compatible server interfaces (vLLM, Ollama, TGI).
"""

import json
import httpx
import pytest

from nemo_eval.models.base import (
    LLMAuthenticationError,
    LLMContextLengthExceededError,
    LLMInvalidResponseError,
    LLMMessage,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from nemo_eval.models.openai_gateway import (
    OpenAIGatewayClient,
    extract_text_fallback_tool_calls,
)


class TestTextFallbackToolCalling:
    """Tests for extracting tool calls embedded in markdown and XML tags."""

    def test_extract_text_fallback_json_code_block(self):
        text = """I will query the products table.
```json
{
    "name": "sqlite_query",
    "arguments": {"query": "SELECT * FROM products;"}
}
```
Please wait."""
        calls, cleaned = extract_text_fallback_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "sqlite_query"
        assert calls[0].arguments == {"query": "SELECT * FROM products;"}
        assert "I will query the products table." in cleaned
        assert "```json" not in cleaned

    def test_extract_text_fallback_tool_parameters_alias(self):
        text = """Executing REPL:
```json
{
    "tool": "python_repl",
    "parameters": {"code": "result = 2 + 2"}
}
```"""
        calls, cleaned = extract_text_fallback_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "python_repl"
        assert calls[0].arguments == {"code": "result = 2 + 2"}

    def test_extract_text_fallback_xml_tag(self):
        text = 'Here is the call: <tool_call>{"name": "tabular_inspect", "arguments": {"file_path": "data.csv"}}</tool_call>'
        calls, cleaned = extract_text_fallback_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "tabular_inspect"
        assert calls[0].arguments == {"file_path": "data.csv"}
        assert cleaned == "Here is the call:"

    def test_extract_text_fallback_multiple_calls(self):
        text = """Parallel calls:
<tool_call>{"name": "sqlite_schema", "arguments": {"table_name": "users"}}</tool_call>
<tool_call>{"name": "sqlite_schema", "arguments": {"table_name": "orders"}}</tool_call>"""
        calls, cleaned = extract_text_fallback_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].arguments == {"table_name": "users"}
        assert calls[1].arguments == {"table_name": "orders"}

    def test_extract_text_fallback_no_calls(self):
        text = "This is a simple answer with no tool calls."
        calls, cleaned = extract_text_fallback_tool_calls(text)
        assert len(calls) == 0
        assert cleaned == "This is a simple answer with no tool calls."


class TestOpenAIGatewayClient:
    """Tests for OpenAIGatewayClient execution, fallback integration, and error handling."""

    def test_gateway_successful_generation(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/chat/completions")
            body = json.loads(request.content)
            assert body["model"] == "vllm-llama-3"
            return httpx.Response(200, json={
                "id": "chatcmpl-vllm",
                "choices": [{
                    "message": {"role": "assistant", "content": "Database has 500 rows."},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 30, "completion_tokens": 10}
            })

        client = OpenAIGatewayClient(
            model_name="vllm-llama-3",
            base_url="http://localhost:8000/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:8000/v1")
        )
        resp = client.generate(sample_messages)
        assert resp.content == "Database has 500 rows."
        assert resp.total_tokens == 40
        assert resp.has_tool_calls is False

    def test_gateway_native_tool_calls_parsing(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "sqlite_query", "arguments": "{\"query\": \"SELECT 1\"}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10}
            })

        client = OpenAIGatewayClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:8000/v1")
        )
        resp = client.generate(sample_messages)
        assert resp.has_tool_calls is True
        assert resp.tool_calls[0].name == "sqlite_query"
        assert resp.tool_calls[0].arguments == {"query": "SELECT 1"}

    def test_gateway_text_fallback_extraction_in_response(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            raw_msg = """Let me check the schema.
```json
{"name": "sqlite_schema", "arguments": {}}
```"""
            return httpx.Response(200, json={
                "choices": [{
                    "message": {"role": "assistant", "content": raw_msg},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 15}
            })

        client = OpenAIGatewayClient(
            enable_text_fallback_tool_calling=True,
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:8000/v1")
        )
        resp = client.generate(sample_messages)
        assert resp.has_tool_calls is True
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "sqlite_schema"
        assert resp.content == "Let me check the schema."

    def test_gateway_text_fallback_disabled(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            raw_msg = '```json\n{"name": "sqlite_schema", "arguments": {}}\n```'
            return httpx.Response(200, json={
                "choices": [{
                    "message": {"role": "assistant", "content": raw_msg},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 15}
            })

        client = OpenAIGatewayClient(
            enable_text_fallback_tool_calling=False,
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:8000/v1")
        )
        resp = client.generate(sample_messages)
        assert resp.has_tool_calls is False
        assert "sqlite_schema" in resp.content

    def test_gateway_auth_failure(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="Unauthorized token")

        client = OpenAIGatewayClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:8000/v1")
        )
        with pytest.raises(LLMAuthenticationError):
            client.generate(sample_messages)

    def test_gateway_malformed_json_response(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="Not a JSON payload <HTML>502 Bad Gateway</HTML>")

        client = OpenAIGatewayClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:8000/v1")
        )
        with pytest.raises(LLMInvalidResponseError):
            client.generate(sample_messages)

    @pytest.mark.anyio
    async def test_gateway_async_agenerate(self, sample_messages):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Async Ollama Response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })

        async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://localhost:11434/v1"
        )
        client = OpenAIGatewayClient(async_http_client=async_client)
        resp = await client.agenerate(sample_messages)
        assert resp.content == "Async Ollama Response"
        assert resp.total_tokens == 15
