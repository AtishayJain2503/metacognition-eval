"""
nemo_eval.models.openai_gateway
-------------------------------
OpenAI-compatible HTTP gateway client supporting local endpoints (vLLM, Ollama, TGI, SGLang)
and external OpenAI-compatible services with markdown/XML tool calling text fallback.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from nemo_eval.models.base import (
    BaseLLMClient,
    LLMAuthenticationError,
    LLMContextLengthExceededError,
    LLMInvalidResponseError,
    LLMMessage,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    ModelConfig,
    ToolCall,
)


def extract_text_fallback_tool_calls(text: Optional[str]) -> Tuple[List[ToolCall], Optional[str]]:
    """
    Extracts tool calls embedded in markdown code blocks or XML tags from model output.
    Useful for models that do not natively support function calling tokens.

    Patterns supported:
    1. ```json {"name": "...", "arguments": {...}} ```
    2. ```json {"tool": "...", "parameters": {...}} ```
    3. <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    4. <|tool_call|>{"name": "...", "arguments": {...}}<|/tool_call|>

    Returns:
        Tuple[List[ToolCall], Optional[str]]: (extracted_tool_calls, cleaned_text)
    """
    if not text:
        return [], text

    tool_calls: List[ToolCall] = []
    cleaned_text = text

    # Pattern 1: XML / delimiter tool call tags
    xml_patterns = [
        r"<\|tool_call\|>(.*?)<\|/tool_call\|>",
        r"<tool_call>(.*?)</tool_call>",
    ]
    for pattern in xml_patterns:
        matches = list(re.finditer(pattern, cleaned_text, flags=re.DOTALL))
        for m in matches:
            raw_chunk = m.group(1).strip()
            try:
                parsed = json.loads(raw_chunk)
                if isinstance(parsed, dict):
                    t_name = parsed.get("name") or parsed.get("tool") or parsed.get("function")
                    t_args = parsed.get("arguments") or parsed.get("parameters") or {}
                    if t_name:
                        tool_calls.append(ToolCall(name=str(t_name), arguments=t_args if isinstance(t_args, dict) else {"raw": t_args}))
                        cleaned_text = cleaned_text.replace(m.group(0), "")
                elif isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            t_name = item.get("name") or item.get("tool")
                            t_args = item.get("arguments") or item.get("parameters") or {}
                            if t_name:
                                tool_calls.append(ToolCall(name=str(t_name), arguments=t_args if isinstance(t_args, dict) else {"raw": t_args}))
                    cleaned_text = cleaned_text.replace(m.group(0), "")
            except Exception:
                pass

    # Pattern 2: Markdown ```json ... ``` blocks containing tool calls
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    matches = list(re.finditer(code_block_pattern, cleaned_text, flags=re.DOTALL))
    for m in matches:
        raw_json = m.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                t_name = parsed.get("name") or parsed.get("tool")
                t_args = parsed.get("arguments") or parsed.get("parameters") or {}
                if t_name:
                    tool_calls.append(ToolCall(name=str(t_name), arguments=t_args if isinstance(t_args, dict) else {"raw": t_args}))
                    cleaned_text = cleaned_text.replace(m.group(0), "")
        except Exception:
            pass

    cleaned_text = cleaned_text.strip()
    return tool_calls, (cleaned_text if cleaned_text else None)


class OpenAIGatewayClient(BaseLLMClient):
    """
    OpenAI-compatible HTTP gateway client supporting local endpoints (vLLM, Ollama, TGI, SGLang)
    and remote OpenAI-compatible endpoints.
    """
    DEFAULT_BASE_URL = "http://localhost:8000/v1"

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        timeout: float = 60.0,
        max_retries: int = 5,
        extra_headers: Optional[Dict[str, str]] = None,
        enable_text_fallback_tool_calling: bool = True,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        **kwargs
    ):
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        resolved_url = (base_url or os.getenv("OPENAI_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")

        config = ModelConfig(
            model_name=model_name,
            api_key=resolved_key,
            base_url=resolved_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            extra_headers=extra_headers or {},
            **kwargs
        )
        super().__init__(config)

        self.enable_text_fallback = enable_text_fallback_tool_calling
        self._default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "NeMoLongHorizonEval/1.0",
        }
        if self.config.api_key and self.config.api_key != "EMPTY":
            self._default_headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.extra_headers:
            self._default_headers.update(self.config.extra_headers)

        self._http_client = http_client or httpx.Client(
            base_url=self.config.base_url,
            headers=self._default_headers,
            timeout=self.config.timeout
        )
        self._async_http_client = async_http_client

    def _get_headers(self) -> Dict[str, str]:
        return dict(self._default_headers)

    def _get_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_http_client is None:
            self._async_http_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=self._default_headers,
                timeout=self.config.timeout
            )
        return self._async_http_client

    def _prepare_payload(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [m.to_wire_dict() for m in messages],
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        return payload

    def _parse_response(self, data: Dict[str, Any], latency_ms: float) -> LLMResponse:
        try:
            choices = data.get("choices", [])
            if not choices:
                raise LLMInvalidResponseError(
                    "OpenAI Gateway response contained no choices",
                    provider="openai_gateway",
                    raw_body=json.dumps(data)
                )

            choice = choices[0]
            msg = choice.get("message", {})
            raw_content = msg.get("content")
            raw_tool_calls = msg.get("tool_calls", [])

            parsed_tool_calls: List[ToolCall] = []
            final_content = raw_content

            # 1. Parse native tool calls
            if raw_tool_calls:
                for tc_data in raw_tool_calls:
                    parsed_tool_calls.append(ToolCall.model_validate(tc_data))
            elif self.enable_text_fallback and raw_content:
                # 2. Check for text fallback tool calling
                fb_tool_calls, cleaned = extract_text_fallback_tool_calls(raw_content)
                if fb_tool_calls:
                    parsed_tool_calls.extend(fb_tool_calls)
                    final_content = cleaned

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            return LLMResponse(
                content=final_content,
                tool_calls=parsed_tool_calls,
                finish_reason=choice.get("finish_reason", "stop"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                raw_response=data
            )
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMInvalidResponseError(
                f"Failed to parse OpenAI Gateway response: {e}",
                provider="openai_gateway",
                raw_body=json.dumps(data) if isinstance(data, dict) else str(data)
            ) from e

    def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        **kwargs
    ) -> LLMResponse:
        payload = self._prepare_payload(messages, tools, temperature, max_tokens, tool_choice)
        url = self._get_url()
        headers = self._get_headers()
        start_time = time.perf_counter()

        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._http_client.post(url, json=payload, headers=headers)

                if resp.status_code in (401, 403):
                    raise LLMAuthenticationError(
                        f"Authentication failed: {resp.text}",
                        provider="openai_gateway",
                        status_code=resp.status_code,
                        raw_body=resp.text
                    )

                if resp.status_code == 429:
                    if attempt >= self.config.max_retries:
                        raise LLMRateLimitError(
                            f"Rate limit exceeded after {attempt} retries: {resp.text}",
                            provider="openai_gateway",
                            status_code=429,
                            raw_body=resp.text
                        )
                    wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                    time.sleep(wait_sec)
                    continue

                if resp.status_code == 400 and ("context_length" in resp.text.lower() or "maximum context" in resp.text.lower()):
                    raise LLMContextLengthExceededError(
                        f"Context length exceeded: {resp.text}",
                        provider="openai_gateway",
                        status_code=400,
                        raw_body=resp.text
                    )

                if resp.status_code >= 500:
                    if attempt >= self.config.max_retries:
                        raise LLMProviderError(
                            f"Server error {resp.status_code}: {resp.text}",
                            provider="openai_gateway",
                            status_code=resp.status_code,
                            raw_body=resp.text
                        )
                    wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                    time.sleep(wait_sec)
                    continue

                resp.raise_for_status()
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                try:
                    resp_json = resp.json()
                except Exception as e:
                    raise LLMInvalidResponseError(
                        f"Malformed JSON in response: {resp.text}",
                        provider="openai_gateway",
                        raw_body=resp.text
                    ) from e
                return self._parse_response(resp_json, latency_ms)

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt >= self.config.max_retries:
                    raise LLMTimeoutError(
                        f"Gateway connection/timeout error: {e}",
                        provider="openai_gateway"
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                time.sleep(wait_sec)
            except httpx.HTTPStatusError as e:
                if attempt >= self.config.max_retries:
                    raise LLMProviderError(
                        f"HTTP error {e.response.status_code}: {e.response.text}",
                        provider="openai_gateway",
                        status_code=e.response.status_code,
                        raw_body=e.response.text
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                time.sleep(wait_sec)

        raise LLMProviderError("OpenAI Gateway call exceeded maximum retries", provider="openai_gateway")

    async def agenerate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        **kwargs
    ) -> LLMResponse:
        payload = self._prepare_payload(messages, tools, temperature, max_tokens, tool_choice)
        url = self._get_url()
        headers = self._get_headers()
        client = self._get_async_client()
        start_time = time.perf_counter()

        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await client.post(url, json=payload, headers=headers)

                if resp.status_code in (401, 403):
                    raise LLMAuthenticationError(
                        f"Authentication failed: {resp.text}",
                        provider="openai_gateway",
                        status_code=resp.status_code,
                        raw_body=resp.text
                    )

                if resp.status_code == 429:
                    if attempt >= self.config.max_retries:
                        raise LLMRateLimitError(
                            f"Rate limit exceeded after {attempt} retries: {resp.text}",
                            provider="openai_gateway",
                            status_code=429,
                            raw_body=resp.text
                        )
                    wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                    await asyncio.sleep(wait_sec)
                    continue

                if resp.status_code == 400 and ("context_length" in resp.text.lower() or "maximum context" in resp.text.lower()):
                    raise LLMContextLengthExceededError(
                        f"Context length exceeded: {resp.text}",
                        provider="openai_gateway",
                        status_code=400,
                        raw_body=resp.text
                    )

                if resp.status_code >= 500:
                    if attempt >= self.config.max_retries:
                        raise LLMProviderError(
                            f"Server error {resp.status_code}: {resp.text}",
                            provider="openai_gateway",
                            status_code=resp.status_code,
                            raw_body=resp.text
                        )
                    wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                    await asyncio.sleep(wait_sec)
                    continue

                resp.raise_for_status()
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                try:
                    resp_json = resp.json()
                except Exception as e:
                    raise LLMInvalidResponseError(
                        f"Malformed JSON in response: {resp.text}",
                        provider="openai_gateway",
                        raw_body=resp.text
                    ) from e
                return self._parse_response(resp_json, latency_ms)

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt >= self.config.max_retries:
                    raise LLMTimeoutError(
                        f"Gateway connection/timeout error: {e}",
                        provider="openai_gateway"
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                await asyncio.sleep(wait_sec)
            except httpx.HTTPStatusError as e:
                if attempt >= self.config.max_retries:
                    raise LLMProviderError(
                        f"HTTP error {e.response.status_code}: {e.response.text}",
                        provider="openai_gateway",
                        status_code=e.response.status_code,
                        raw_body=e.response.text
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                await asyncio.sleep(wait_sec)

        raise LLMProviderError("OpenAI Gateway call exceeded maximum retries", provider="openai_gateway")
