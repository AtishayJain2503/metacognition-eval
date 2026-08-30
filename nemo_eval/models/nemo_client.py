"""
nemo_eval.models.nemo_client
----------------------------
Native NVIDIA NeMo / NVIDIA Inference Microservice (NIM) client with special token parsing
(<|begin_of_thought|>, <|tool_call|>) and Guardrail metadata ingestion.
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


def extract_nemo_special_tokens(raw_text: Optional[str]) -> Tuple[Optional[str], List[ToolCall], Optional[str]]:
    """
    Parses NVIDIA Nemotron and NIM special tokens:
    - `<|begin_of_thought|> ... <|end_of_thought|>` -> reasoning Chain-of-Thought
    - `<|tool_call|> ... <|/tool_call|>` or `<tool_call> ... </tool_call>` -> ToolCall list

    Returns:
        Tuple[Optional[str], List[ToolCall], Optional[str]]: (reasoning_content, tool_calls, cleaned_content)
    """
    if not raw_text:
        return None, [], raw_text

    text = raw_text.strip()
    reasoning_content: Optional[str] = None
    tool_calls: List[ToolCall] = []

    # 1. Parse Thought Tokens: <|begin_of_thought|> ... <|end_of_thought|>
    thought_pattern = r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>"
    thought_match = re.search(thought_pattern, text, flags=re.DOTALL)
    if thought_match:
        thought_str = thought_match.group(1).strip()
        reasoning_content = thought_str if thought_str else None
        text = re.sub(thought_pattern, "", text, flags=re.DOTALL).strip()
    else:
        # Check unclosed thought token
        unclosed_thought = r"<\|begin_of_thought\|>(.*)$"
        unclosed_match = re.search(unclosed_thought, text, flags=re.DOTALL)
        if unclosed_match:
            thought_str = unclosed_match.group(1).strip()
            reasoning_content = thought_str if thought_str else None
            text = re.sub(unclosed_thought, "", text, flags=re.DOTALL).strip()

    # 2. Parse Tool Call Tokens: <|tool_call|> ... <|/tool_call|> or <tool_call> ... </tool_call>
    tool_patterns = [
        r"<\|tool_call\|>(.*?)<\|/tool_call\|>",
        r"<tool_call>(.*?)</tool_call>",
    ]
    for pattern in tool_patterns:
        matches = list(re.finditer(pattern, text, flags=re.DOTALL))
        for m in matches:
            raw_call = m.group(1).strip()
            try:
                parsed = json.loads(raw_call)
                if isinstance(parsed, dict):
                    t_name = parsed.get("name") or parsed.get("tool") or parsed.get("function")
                    t_args = parsed.get("arguments") or parsed.get("parameters") or {}
                    if t_name:
                        tool_calls.append(ToolCall(
                            name=str(t_name),
                            arguments=t_args if isinstance(t_args, dict) else {"raw": t_args}
                        ))
                elif isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            t_name = item.get("name") or item.get("tool")
                            t_args = item.get("arguments") or item.get("parameters") or {}
                            if t_name:
                                tool_calls.append(ToolCall(
                                    name=str(t_name),
                                    arguments=t_args if isinstance(t_args, dict) else {"raw": t_args}
                                ))
                text = text.replace(m.group(0), "")
            except Exception:
                pass

    text = text.strip()
    return reasoning_content, tool_calls, (text if text else None)


class NeMoClient(BaseLLMClient):
    """
    NVIDIA NeMo / NIM microservice client supporting Nemotron models,
    guardrail telemetry, and special token parsing.
    """
    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        model_name: str = "nvidia/llama-3.1-nemotron-70b-instruct",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        timeout: float = 60.0,
        max_retries: int = 5,
        extra_headers: Optional[Dict[str, str]] = None,
        guardrails_enabled: bool = True,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        **kwargs
    ):
        resolved_key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY", "mock_nemo_key")
        resolved_url = (base_url or os.getenv("NVIDIA_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")

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

        self.guardrails_enabled = guardrails_enabled
        self._default_headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "NeMoLongHorizonEval/1.0",
        }
        if self.guardrails_enabled:
            self._default_headers["x-nvidia-guardrails"] = "enabled"
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

    def _parse_response(self, data: Dict[str, Any], headers: httpx.Headers, latency_ms: float) -> LLMResponse:
        try:
            choices = data.get("choices", [])
            if not choices:
                raise LLMInvalidResponseError(
                    "NeMo/NIM response contained no choices",
                    provider="nemo",
                    raw_body=json.dumps(data)
                )

            choice = choices[0]
            msg = choice.get("message", {})
            raw_content = msg.get("content")
            raw_tool_calls = msg.get("tool_calls", [])

            # 1. Parse special tokens
            reasoning_content, token_tool_calls, cleaned_content = extract_nemo_special_tokens(raw_content)

            # 2. Parse native tool calls
            parsed_tool_calls: List[ToolCall] = []
            if raw_tool_calls:
                for tc_data in raw_tool_calls:
                    parsed_tool_calls.append(ToolCall.model_validate(tc_data))
            if token_tool_calls:
                parsed_tool_calls.extend(token_tool_calls)

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            # 3. Ingest guardrail & NIM telemetry metadata
            raw_resp = dict(data)
            guardrail_info: Dict[str, Any] = {}
            if "guardrails" in data:
                guardrail_info = data["guardrails"]
            elif "moderation" in data:
                guardrail_info = data["moderation"]

            req_id = headers.get("nvcf-reqid") or headers.get("x-request-id")
            if req_id:
                guardrail_info["nvcf_reqid"] = req_id

            raw_resp["guardrails"] = guardrail_info

            return LLMResponse(
                content=cleaned_content,
                reasoning_content=reasoning_content,
                tool_calls=parsed_tool_calls,
                finish_reason=choice.get("finish_reason", "stop"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                raw_response=raw_resp
            )
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMInvalidResponseError(
                f"Failed to parse NeMo/NIM response: {e}",
                provider="nemo",
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
                        f"NeMo API authentication failed: {resp.text}",
                        provider="nemo",
                        status_code=resp.status_code,
                        raw_body=resp.text
                    )

                if resp.status_code == 429:
                    if attempt >= self.config.max_retries:
                        raise LLMRateLimitError(
                            f"NeMo API rate limit exceeded after {attempt} retries: {resp.text}",
                            provider="nemo",
                            status_code=429,
                            raw_body=resp.text
                        )
                    wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                    time.sleep(wait_sec)
                    continue

                if resp.status_code == 400 and ("context_length" in resp.text.lower() or "maximum context" in resp.text.lower()):
                    raise LLMContextLengthExceededError(
                        f"NeMo context length exceeded: {resp.text}",
                        provider="nemo",
                        status_code=400,
                        raw_body=resp.text
                    )

                if resp.status_code >= 500:
                    if attempt >= self.config.max_retries:
                        raise LLMProviderError(
                            f"NeMo server error {resp.status_code}: {resp.text}",
                            provider="nemo",
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
                        f"Malformed JSON in NeMo response: {resp.text}",
                        provider="nemo",
                        raw_body=resp.text
                    ) from e
                return self._parse_response(resp_json, resp.headers, latency_ms)

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt >= self.config.max_retries:
                    raise LLMTimeoutError(
                        f"NeMo connection/timeout error: {e}",
                        provider="nemo"
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                time.sleep(wait_sec)
            except httpx.HTTPStatusError as e:
                if attempt >= self.config.max_retries:
                    raise LLMProviderError(
                        f"NeMo HTTP error {e.response.status_code}: {e.response.text}",
                        provider="nemo",
                        status_code=e.response.status_code,
                        raw_body=e.response.text
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                time.sleep(wait_sec)

        raise LLMProviderError("NeMo API call exceeded maximum retries", provider="nemo")

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
                        f"NeMo API authentication failed: {resp.text}",
                        provider="nemo",
                        status_code=resp.status_code,
                        raw_body=resp.text
                    )

                if resp.status_code == 429:
                    if attempt >= self.config.max_retries:
                        raise LLMRateLimitError(
                            f"NeMo API rate limit exceeded after {attempt} retries: {resp.text}",
                            provider="nemo",
                            status_code=429,
                            raw_body=resp.text
                        )
                    wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                    await asyncio.sleep(wait_sec)
                    continue

                if resp.status_code == 400 and ("context_length" in resp.text.lower() or "maximum context" in resp.text.lower()):
                    raise LLMContextLengthExceededError(
                        f"NeMo context length exceeded: {resp.text}",
                        provider="nemo",
                        status_code=400,
                        raw_body=resp.text
                    )

                if resp.status_code >= 500:
                    if attempt >= self.config.max_retries:
                        raise LLMProviderError(
                            f"NeMo server error {resp.status_code}: {resp.text}",
                            provider="nemo",
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
                        f"Malformed JSON in NeMo response: {resp.text}",
                        provider="nemo",
                        raw_body=resp.text
                    ) from e
                return self._parse_response(resp_json, resp.headers, latency_ms)

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt >= self.config.max_retries:
                    raise LLMTimeoutError(
                        f"NeMo connection/timeout error: {e}",
                        provider="nemo"
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                await asyncio.sleep(wait_sec)
            except httpx.HTTPStatusError as e:
                if attempt >= self.config.max_retries:
                    raise LLMProviderError(
                        f"NeMo HTTP error {e.response.status_code}: {e.response.text}",
                        provider="nemo",
                        status_code=e.response.status_code,
                        raw_body=e.response.text
                    ) from e
                wait_sec = min(30.0, (2 ** attempt) * 0.5 + random.uniform(0.05, 0.2))
                await asyncio.sleep(wait_sec)

        raise LLMProviderError("NeMo API call exceeded maximum retries", provider="nemo")
