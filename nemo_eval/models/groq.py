"""
nemo_eval.models.groq
---------------------
Ultra-fast Groq API provider client with DeepSeek-R1 <think> token isolation,
automated exponential backoff, and rate limit handling.
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


def extract_think_reasoning(raw_text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts and isolates <think>...</think> reasoning Chain-of-Thought tokens from LLM output.

    Returns:
        Tuple[Optional[str], Optional[str]]: (reasoning_content, cleaned_content)
    """
    if raw_text is None:
        return None, None

    text = raw_text.strip()
    if not text:
        return None, None

    # Check for complete <think>...</think> tag
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return (reasoning if reasoning else None), (cleaned if cleaned else None)

    # Check for unclosed <think> tag (e.g. generation truncated during thinking)
    unclosed_pattern = r"<think>(.*)$"
    unclosed_match = re.search(unclosed_pattern, text, flags=re.DOTALL)
    if unclosed_match:
        reasoning = unclosed_match.group(1).strip()
        cleaned = re.sub(unclosed_pattern, "", text, flags=re.DOTALL).strip()
        return (reasoning if reasoning else None), (cleaned if cleaned else None)

    return None, text


class GroqLLMClient(BaseLLMClient):
    """
    Groq LPU API Provider client supporting Llama 3.3 and DeepSeek-R1 reasoning models.
    """
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

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
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        **kwargs
    ):
        resolved_key = api_key or os.getenv("GROQ_API_KEY", "mock_groq_key")
        resolved_url = (base_url or os.getenv("GROQ_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")

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

        self._default_headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "NeMoLongHorizonEval/1.0",
        }
        if self.config.extra_headers:
            self._default_headers.update(self.config.extra_headers)

        self._http_client = http_client or httpx.Client(
            base_url=self.config.base_url,
            headers=self._default_headers,
            timeout=self.config.timeout
        )
        self._async_http_client = async_http_client

    def _get_headers(self) -> Dict[str, str]:
        headers = dict(self._default_headers)
        return headers

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
                    "Groq response contained no choices",
                    provider="groq",
                    raw_body=json.dumps(data)
                )

            choice = choices[0]
            msg = choice.get("message", {})
            raw_content = msg.get("content")
            raw_tool_calls = msg.get("tool_calls", [])

            # Extract think reasoning tokens if present
            reasoning_content, content = extract_think_reasoning(raw_content)

            # Parse tool calls
            parsed_tool_calls: List[ToolCall] = []
            if raw_tool_calls:
                for tc_data in raw_tool_calls:
                    parsed_tool_calls.append(ToolCall.model_validate(tc_data))

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            return LLMResponse(
                content=content,
                reasoning_content=reasoning_content,
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
                f"Failed to parse Groq response: {e}",
                provider="groq",
                raw_body=json.dumps(data) if isinstance(data, dict) else str(data)
            ) from e

    def _parse_retry_after(self, response: httpx.Response, attempt: int) -> float:
        retry_after_header = response.headers.get("retry-after")
        if retry_after_header:
            try:
                return max(0.01, float(retry_after_header))
            except ValueError:
                pass
        return min(60.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))

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
                
                # Check authentication
                if resp.status_code in (401, 403):
                    raise LLMAuthenticationError(
                        f"Groq API authentication failed: {resp.text}",
                        provider="groq",
                        status_code=resp.status_code,
                        raw_body=resp.text
                    )

                # Check rate limit
                if resp.status_code == 429:
                    wait_sec = self._parse_retry_after(resp, attempt)
                    if attempt >= self.config.max_retries:
                        raise LLMRateLimitError(
                            f"Groq API rate limit exceeded after {attempt} retries: {resp.text}",
                            provider="groq",
                            status_code=429,
                            retry_after=wait_sec,
                            raw_body=resp.text
                        )
                    time.sleep(wait_sec)
                    continue

                # Check context length exceeded
                if resp.status_code == 400 and ("context_length" in resp.text.lower() or "maximum context" in resp.text.lower()):
                    raise LLMContextLengthExceededError(
                        f"Groq context length exceeded: {resp.text}",
                        provider="groq",
                        status_code=400,
                        raw_body=resp.text
                    )

                # Check general server errors
                if resp.status_code >= 500:
                    if attempt >= self.config.max_retries:
                        raise LLMProviderError(
                            f"Groq API server error {resp.status_code}: {resp.text}",
                            provider="groq",
                            status_code=resp.status_code,
                            raw_body=resp.text
                        )
                    wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                    time.sleep(wait_sec)
                    continue

                resp.raise_for_status()
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return self._parse_response(resp.json(), latency_ms)

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt >= self.config.max_retries:
                    raise LLMTimeoutError(
                        f"Groq API connection/timeout error: {e}",
                        provider="groq"
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                time.sleep(wait_sec)
            except httpx.HTTPStatusError as e:
                if attempt >= self.config.max_retries:
                    raise LLMProviderError(
                        f"Groq HTTP error {e.response.status_code}: {e.response.text}",
                        provider="groq",
                        status_code=e.response.status_code,
                        raw_body=e.response.text
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                time.sleep(wait_sec)

        raise LLMProviderError("Groq API call exceeded maximum retries", provider="groq")

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
                        f"Groq API authentication failed: {resp.text}",
                        provider="groq",
                        status_code=resp.status_code,
                        raw_body=resp.text
                    )

                if resp.status_code == 429:
                    wait_sec = self._parse_retry_after(resp, attempt)
                    if attempt >= self.config.max_retries:
                        raise LLMRateLimitError(
                            f"Groq API rate limit exceeded after {attempt} retries: {resp.text}",
                            provider="groq",
                            status_code=429,
                            retry_after=wait_sec,
                            raw_body=resp.text
                        )
                    await asyncio.sleep(wait_sec)
                    continue

                if resp.status_code == 400 and ("context_length" in resp.text.lower() or "maximum context" in resp.text.lower()):
                    raise LLMContextLengthExceededError(
                        f"Groq context length exceeded: {resp.text}",
                        provider="groq",
                        status_code=400,
                        raw_body=resp.text
                    )

                if resp.status_code >= 500:
                    if attempt >= self.config.max_retries:
                        raise LLMProviderError(
                            f"Groq API server error {resp.status_code}: {resp.text}",
                            provider="groq",
                            status_code=resp.status_code,
                            raw_body=resp.text
                        )
                    wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                    await asyncio.sleep(wait_sec)
                    continue

                resp.raise_for_status()
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return self._parse_response(resp.json(), latency_ms)

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt >= self.config.max_retries:
                    raise LLMTimeoutError(
                        f"Groq API connection/timeout error: {e}",
                        provider="groq"
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                await asyncio.sleep(wait_sec)
            except httpx.HTTPStatusError as e:
                if attempt >= self.config.max_retries:
                    raise LLMProviderError(
                        f"Groq HTTP error {e.response.status_code}: {e.response.text}",
                        provider="groq",
                        status_code=e.response.status_code,
                        raw_body=e.response.text
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                await asyncio.sleep(wait_sec)

        raise LLMProviderError("Groq API call exceeded maximum retries", provider="groq")
