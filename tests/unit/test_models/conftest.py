"""
tests.unit.test_models.conftest
-------------------------------
Shared fixtures, mock HTTP transport factories, and sample message payloads
for testing model providers hermetically without network access.
"""

from typing import Any, Callable, Dict, List, Optional
import httpx
import pytest

from nemo_eval.models.base import LLMMessage, ModelConfig, ToolCall


@pytest.fixture
def sample_messages() -> List[LLMMessage]:
    """Sample multi-turn message sequence."""
    return [
        LLMMessage(role="system", content="You are a helpful data analyst."),
        LLMMessage(role="user", content="How many rows are in the database?"),
    ]


@pytest.fixture
def sample_tool_specs() -> List[Dict[str, Any]]:
    """Sample OpenAI-compatible tool specifications."""
    return [
        {
            "type": "function",
            "function": {
                "name": "sqlite_query",
                "description": "Execute a SQL query against the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL query to execute."}
                    },
                    "required": ["query"]
                }
            }
        }
    ]


def make_mock_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    """Helper to create a synchronous mock HTTP transport."""
    return httpx.MockTransport(handler)


def make_async_mock_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    """Helper to create an asynchronous mock HTTP transport."""
    return httpx.MockTransport(handler)
