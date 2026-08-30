"""
tests.unit.test_models.test_base
--------------------------------
Unit tests for base data models, protocols, exceptions, and ModelRegistry.
"""

import json
import pytest
from pydantic import ValidationError

from nemo_eval.models.base import (
    BaseLLMClient,
    ChatMessage,
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


class TestBaseDataModels:
    """Tests for ToolCall, FunctionCall, LLMMessage, and LLMResponse."""

    def test_tool_call_direct_initialization(self):
        """Test ToolCall created with top-level name and arguments."""
        tc = ToolCall(id="call_123", name="sqlite_query", arguments={"query": "SELECT 1;"})
        assert tc.id == "call_123"
        assert tc.name == "sqlite_query"
        assert tc.arguments == {"query": "SELECT 1;"}
        assert tc.function is not None
        assert tc.function.name == "sqlite_query"
        assert tc.function.arguments == {"query": "SELECT 1;"}

    def test_tool_call_nested_function_initialization(self):
        """Test ToolCall created with nested FunctionCall object."""
        fn = FunctionCall(name="python_repl", arguments={"code": "x = 42"})
        tc = ToolCall(id="call_456", function=fn)
        assert tc.name == "python_repl"
        assert tc.arguments == {"code": "x = 42"}

    def test_tool_call_json_string_argument_parsing(self):
        """Test ToolCall automatically parses JSON string arguments."""
        data = {
            "id": "call_789",
            "type": "function",
            "function": {
                "name": "tabular_inspect",
                "arguments": json.dumps({"file_path": "data.csv", "action": "schema"})
            }
        }
        tc = ToolCall.model_validate(data)
        assert tc.name == "tabular_inspect"
        assert isinstance(tc.arguments, dict)
        assert tc.arguments["file_path"] == "data.csv"
        assert tc.arguments["action"] == "schema"

    def test_tool_call_invalid_json_string_fallback(self):
        """Test ToolCall gracefully handles non-JSON string arguments."""
        data = {
            "id": "call_999",
            "function": {
                "name": "custom_tool",
                "arguments": "invalid-json-text"
            }
        }
        tc = ToolCall.model_validate(data)
        assert tc.name == "custom_tool"
        assert tc.arguments == {"raw": "invalid-json-text"}

    def test_tool_call_to_openai_dict(self):
        """Test ToolCall serialization into OpenAI function calling wire format."""
        tc = ToolCall(id="call_001", name="sqlite_query", arguments={"query": "SELECT * FROM t;"})
        wire = tc.to_openai_dict()
        assert wire["id"] == "call_001"
        assert wire["type"] == "function"
        assert wire["function"]["name"] == "sqlite_query"
        parsed_args = json.loads(wire["function"]["arguments"])
        assert parsed_args["query"] == "SELECT * FROM t;"

    def test_llm_message_to_wire_dict(self):
        """Test LLMMessage serialization to wire dictionary."""
        tc = ToolCall(id="c1", name="sqlite_query", arguments={"query": "SELECT 1;"})
        msg = LLMMessage(
            role="assistant",
            content="I will run query",
            tool_calls=[tc]
        )
        wire = msg.to_wire_dict()
        assert wire["role"] == "assistant"
        assert wire["content"] == "I will run query"
        assert len(wire["tool_calls"]) == 1
        assert wire["tool_calls"][0]["id"] == "c1"

    def test_llm_message_alias_chat_message(self):
        """Test ChatMessage alias is equivalent to LLMMessage."""
        assert ChatMessage is LLMMessage
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_llm_response_total_tokens_computation(self):
        """Test LLMResponse automatically computes total_tokens if not provided."""
        resp = LLMResponse(content="Answer", prompt_tokens=15, completion_tokens=25)
        assert resp.total_tokens == 40
        assert resp.has_tool_calls is False

    def test_llm_response_with_tool_calls(self):
        """Test LLMResponse has_tool_calls property."""
        tc = ToolCall(name="test_tool", arguments={})
        resp = LLMResponse(content=None, tool_calls=[tc], finish_reason="tool_calls")
        assert resp.has_tool_calls is True
        assert len(resp.tool_calls) == 1

    def test_model_config_validation(self):
        """Test ModelConfig parameter validation."""
        cfg = ModelConfig(
            model_name="test-model",
            temperature=0.7,
            max_tokens=2048,
            timeout=30.0,
            max_retries=3
        )
        assert cfg.model_name == "test-model"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 3

        # Invalid temperature > 2.0
        with pytest.raises(ValidationError):
            ModelConfig(model_name="bad", temperature=2.5)


class TestExceptionHierarchy:
    """Tests for LLMProviderError and its derived exceptions."""

    def test_llm_provider_error_base(self):
        err = LLMProviderError("Base error", provider="groq", status_code=500, raw_body="Internal Error")
        assert "Base error" in str(err)
        assert "[groq]" in str(err)
        assert "(Status 500)" in str(err)
        assert err.provider == "groq"
        assert err.status_code == 500
        assert err.raw_body == "Internal Error"

    def test_llm_authentication_error(self):
        err = LLMAuthenticationError("Invalid API key", provider="openai", status_code=401)
        assert isinstance(err, LLMProviderError)
        assert err.status_code == 401

    def test_llm_rate_limit_error(self):
        err = LLMRateLimitError("Too Many Requests", retry_after=2.5, provider="groq", status_code=429)
        assert isinstance(err, LLMProviderError)
        assert err.retry_after == 2.5
        assert err.status_code == 429

    def test_llm_timeout_error(self):
        err = LLMTimeoutError("Request timed out", provider="nemo")
        assert isinstance(err, LLMProviderError)

    def test_llm_context_length_exceeded_error(self):
        err = LLMContextLengthExceededError("Context window exceeded", provider="openai", status_code=400)
        assert isinstance(err, LLMProviderError)

    def test_llm_invalid_response_error(self):
        err = LLMInvalidResponseError("Malformed JSON response", provider="groq")
        assert isinstance(err, LLMProviderError)


class TestModelRegistry:
    """Tests for ModelRegistry registration, lookup, and instantiation."""

    class DummyClient(BaseLLMClient):
        def generate(self, messages, **kwargs):
            return LLMResponse(content="Dummy sync")

        async def agenerate(self, messages, **kwargs):
            return LLMResponse(content="Dummy async")

    def test_register_and_lookup(self):
        ModelRegistry.register("dummy_test_provider", self.DummyClient)
        cls = ModelRegistry.get_client_class("dummy_test_provider")
        assert cls is self.DummyClient

        # Case insensitivity
        cls_upper = ModelRegistry.get_client_class("DUMMY_TEST_PROVIDER")
        assert cls_upper is self.DummyClient

    def test_create_client(self):
        ModelRegistry.register("dummy_test_factory", self.DummyClient)
        client = ModelRegistry.create_client("dummy_test_factory", model_name="dummy-v1")
        assert isinstance(client, self.DummyClient)
        assert client.model_name == "dummy-v1"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown model provider: 'non_existent_provider'"):
            ModelRegistry.get_client_class("non_existent_provider")

    def test_get_model_client_helper(self):
        ModelRegistry.register("dummy_helper_provider", self.DummyClient)
        client = get_model_client("dummy_helper_provider")
        assert isinstance(client, self.DummyClient)
