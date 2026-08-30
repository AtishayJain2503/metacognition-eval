"""
nemo_eval.models.mock_runner
----------------------------
Deterministic Offline Mock LLM Runner for hermetic testing, CI/CD sweeps,
fault injection, and scripted multi-turn scenario validation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Union

from nemo_eval.models.base import (
    BaseLLMClient,
    LLMMessage,
    LLMResponse,
    ModelConfig,
    ToolCall,
)


class DeterministicMockLLMClient(BaseLLMClient):
    """
    100% Offline Hermetic Mock LLM Runner for deterministic multi-turn agent evaluations.
    """

    def __init__(
        self,
        script_mode: Optional[str] = "perfect",
        response_queue: Optional[List[Union[LLMResponse, Dict[str, Any]]]] = None,
        pattern_handlers: Optional[Dict[str, Union[LLMResponse, Callable[[List[LLMMessage]], LLMResponse]]]] = None,
        inject_errors: Optional[List[Optional[Exception]]] = None,
        model_name: str = "mock-deterministic-llm",
        responses: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        config = ModelConfig(model_name=model_name, **kwargs)
        super().__init__(config)

        self.responses = responses or {}
        self._has_explicit_queue = response_queue is not None
        self.script_mode = script_mode if not self._has_explicit_queue else None
        self.turn_counter = 0
        self.recorded_requests: List[Dict[str, Any]] = []
        self.history_log: List[LLMResponse] = []

        # Initialize response queue
        self.response_queue: List[LLMResponse] = []
        if response_queue:
            for item in response_queue:
                if isinstance(item, LLMResponse):
                    self.response_queue.append(item)
                elif isinstance(item, dict):
                    self.response_queue.append(LLMResponse.model_validate(item))

        self.pattern_handlers = pattern_handlers or {}
        self.inject_errors = list(inject_errors) if inject_errors else []

    def reset(self) -> None:
        """Reset turn counter, history, and recorded requests."""
        self.turn_counter = 0
        self.recorded_requests.clear()
        self.history_log.clear()

    def get_last_request(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent request recorded by the runner."""
        return self.recorded_requests[-1] if self.recorded_requests else None

    def assert_turn_count(self, expected: int) -> None:
        """Assert that exactly expected number of turns have been executed."""
        if self.turn_counter != expected:
            raise AssertionError(f"Expected {expected} turns, but recorded {self.turn_counter} turns.")

    def add_response(self, response: Union[LLMResponse, Dict[str, Any]]) -> None:
        """Queue an additional response to the runner."""
        if isinstance(response, LLMResponse):
            self.response_queue.append(response)
        elif isinstance(response, dict):
            self.response_queue.append(LLMResponse.model_validate(response))

    def generate(
        self,
        messages: Union[List[LLMMessage], str],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        if isinstance(messages, str):
            msg_list = []
            if system:
                msg_list.append(LLMMessage(role="system", content=system))
            msg_list.append(LLMMessage(role="user", content=messages))
            messages = msg_list
        elif isinstance(messages, list):
            msg_list = []
            for m in messages:
                if isinstance(m, dict):
                    msg_list.append(LLMMessage(role=m.get("role", "user"), content=m.get("content", "")))
                elif hasattr(m, "role"):
                    msg_list.append(m)
                else:
                    msg_list.append(LLMMessage(role="user", content=str(m)))
            messages = msg_list

        self.recorded_requests.append({
            "turn": self.turn_counter + 1,
            "messages": [m.model_dump() if hasattr(m, "model_dump") else m for m in messages],
            "tools": tools,
            "kwargs": kwargs
        })

        # 1. Error Injection
        if self.inject_errors:
            err = self.inject_errors.pop(0)
            if err is not None:
                raise err

        self.turn_counter += 1

        # 2. Scripted Response Queue
        if self.response_queue:
            resp = self.response_queue.pop(0)
            self.history_log.append(resp)
            return resp

        # Build context text for matching
        combined_texts = []
        for msg in reversed(messages):
            if msg.content:
                combined_texts.append(msg.content)
        full_context = "\n".join(combined_texts)

        # 3. Direct response dictionary matching
        if self.responses:
            for k, v in self.responses.items():
                if k in full_context:
                    resp = LLMResponse(
                        content=v,
                        tool_calls=[],
                        finish_reason="stop",
                        prompt_tokens=50,
                        completion_tokens=20,
                        latency_ms=1.0,
                    )
                    self.history_log.append(resp)
                    return resp

        # 4. Dynamic Pattern Handlers
        if self.pattern_handlers and messages:
            for pattern, handler in self.pattern_handlers.items():
                if re.search(pattern, full_context, re.IGNORECASE):
                    if callable(handler):
                        resp = handler(messages)
                    elif isinstance(handler, dict):
                        resp = LLMResponse.model_validate(handler)
                    else:
                        resp = handler
                    self.history_log.append(resp)
                    return resp

        # 4. Built-in Deterministic Script Modes
        if self.script_mode == "perfect":
            if self.turn_counter == 1:
                resp = LLMResponse(
                    content="I will inspect the database schema.",
                    tool_calls=[ToolCall(name="sqlite_schema", arguments={})],
                    finish_reason="tool_calls",
                    prompt_tokens=100, completion_tokens=30, latency_ms=1.0
                )
            elif self.turn_counter == 2:
                resp = LLMResponse(
                    content="Querying category list.",
                    tool_calls=[ToolCall(name="sqlite_query", arguments={"query": "SELECT * FROM categories;"})],
                    finish_reason="tool_calls",
                    prompt_tokens=200, completion_tokens=35, latency_ms=1.0
                )
            elif self.turn_counter == 3:
                resp = LLMResponse(
                    content="Final Answer: The database contains 3 product categories.",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=300, completion_tokens=15, latency_ms=1.0
                )
            else:
                resp = LLMResponse(
                    content="Task already completed.",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=50, completion_tokens=5, latency_ms=1.0
                )
            self.history_log.append(resp)
            return resp

        elif self.script_mode == "self_correction":
            if self.turn_counter == 1:
                resp = LLMResponse(
                    content="Computing sum with pandas.",
                    tool_calls=[ToolCall(
                        name="python_repl",
                        arguments={"code": "df = pd.read_csv('test.csv'\nresult = df.sum()"}
                    )],
                    finish_reason="tool_calls",
                    prompt_tokens=100, completion_tokens=25, latency_ms=1.0
                )
            elif self.turn_counter == 2:
                resp = LLMResponse(
                    content="Fixing missing closing parenthesis.",
                    tool_calls=[ToolCall(
                        name="python_repl",
                        arguments={"code": "df = pd.read_csv('test.csv')\nresult = int(df['val'].sum())"}
                    )],
                    finish_reason="tool_calls",
                    prompt_tokens=180, completion_tokens=30, latency_ms=1.0
                )
            elif self.turn_counter == 3:
                resp = LLMResponse(
                    content="Final Answer: 150",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=250, completion_tokens=10, latency_ms=1.0
                )
            else:
                resp = LLMResponse(
                    content="Task complete.",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=50, completion_tokens=5, latency_ms=1.0
                )
            self.history_log.append(resp)
            return resp

        elif self.script_mode == "data_analytics_pipeline":
            if self.turn_counter == 1:
                resp = LLMResponse(
                    content="Inspecting tabular dataset schema.",
                    tool_calls=[ToolCall(
                        name="tabular_inspect",
                        arguments={"file_path": "customers.csv", "action": "schema"}
                    )],
                    finish_reason="tool_calls",
                    prompt_tokens=100, completion_tokens=25, latency_ms=1.0
                )
            elif self.turn_counter == 2:
                resp = LLMResponse(
                    content="Imputing missing values in dataset.",
                    tool_calls=[ToolCall(
                        name="python_repl",
                        arguments={"code": "df = pd.read_csv('customers.csv')\ndf['total_charges'] = df['total_charges'].fillna(0)"}
                    )],
                    finish_reason="tool_calls",
                    prompt_tokens=180, completion_tokens=35, latency_ms=1.0
                )
            elif self.turn_counter == 3:
                resp = LLMResponse(
                    content="Calculating customer churn rate.",
                    tool_calls=[ToolCall(
                        name="python_repl",
                        arguments={"code": "churn_rate = float(df['churned'].mean())\nprint(f'Churn rate: {churn_rate}')"}
                    )],
                    finish_reason="tool_calls",
                    prompt_tokens=260, completion_tokens=35, latency_ms=1.0
                )
            elif self.turn_counter == 4:
                resp = LLMResponse(
                    content="Final Answer: The customer churn rate is 37.5%.",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=350, completion_tokens=15, latency_ms=1.0
                )
            else:
                resp = LLMResponse(
                    content="Pipeline complete.",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=50, completion_tokens=5, latency_ms=1.0
                )
            self.history_log.append(resp)
            return resp

        elif self.script_mode == "infinite_loop_thrashing":
            resp = LLMResponse(
                content="Retrying query against broken table.",
                tool_calls=[ToolCall(name="sqlite_query", arguments={"query": "SELECT * FROM broken_table;"})],
                finish_reason="tool_calls",
                prompt_tokens=100, completion_tokens=20, latency_ms=1.0
            )
            self.history_log.append(resp)
            return resp

        elif self.script_mode == "schema_mismatch_recovery":
            if self.turn_counter == 1:
                resp = LLMResponse(
                    content="Querying MonthlyFee.",
                    tool_calls=[ToolCall(name="sqlite_query", arguments={"query": "SELECT MonthlyFee FROM customers;"})],
                    finish_reason="tool_calls",
                    prompt_tokens=100, completion_tokens=20, latency_ms=1.0
                )
            elif self.turn_counter == 2:
                resp = LLMResponse(
                    content="Correcting column name to monthly_charges based on suggestion.",
                    tool_calls=[ToolCall(name="sqlite_query", arguments={"query": "SELECT monthly_charges FROM customers;"})],
                    finish_reason="tool_calls",
                    prompt_tokens=180, completion_tokens=25, latency_ms=1.0
                )
            elif self.turn_counter == 3:
                resp = LLMResponse(
                    content="Final Answer: 77.40",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=250, completion_tokens=10, latency_ms=1.0
                )
            else:
                resp = LLMResponse(
                    content="Recovery complete.",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=50, completion_tokens=5, latency_ms=1.0
                )
            self.history_log.append(resp)
            return resp

        elif self.script_mode == "parallel_tool_calls":
            resp = LLMResponse(
                content="Inspecting both tables in parallel.",
                tool_calls=[
                    ToolCall(id="call_p1", name="sqlite_schema", arguments={"table_name": "products"}),
                    ToolCall(id="call_p2", name="sqlite_schema", arguments={"table_name": "categories"})
                ],
                finish_reason="tool_calls",
                prompt_tokens=120, completion_tokens=30, latency_ms=1.0
            )
            self.history_log.append(resp)
            return resp

        elif self.script_mode == "deepseek_r1_reasoning":
            resp = LLMResponse(
                content="The result is 42.",
                reasoning_content="Let's break down the problem step by step. 1) Step A. 2) Step B.",
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=80, completion_tokens=40, latency_ms=1.0
            )
            self.history_log.append(resp)
            return resp

        # Default Fallback
        if self.script_mode is None:
            resp = LLMResponse(
                content=f"Mock response for turn {self.turn_counter}",
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=50, completion_tokens=10, latency_ms=1.0
            )
        elif "DeepSeek-R1" in self.model_name or "Thinking" in self.model_name:
            resp = LLMResponse(
                content="<think>\nLet's analyze step by step.\n</think>\nThe answer is \\boxed{42}",
                reasoning_content="Let's analyze step by step.",
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=80, completion_tokens=30, latency_ms=1.0
            )
        else:
            resp = LLMResponse(
                content="After calculation, the final answer is \\boxed{42}",
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=50, completion_tokens=10, latency_ms=1.0
            )
        self.history_log.append(resp)
        return resp

    async def agenerate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Asynchronous execution for deterministic mock runner."""
        return self.generate(messages, tools=tools, **kwargs)
