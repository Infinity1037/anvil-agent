from __future__ import annotations

import json
import time
from collections.abc import Callable
from threading import Event
from typing import Any

import httpx

from anvil.config import Config
from anvil.llm.parse import parse_assistant_choice, parse_tool_call, parse_tool_calls, parse_usage
from anvil.llm.types import LLMResponse, Message, Usage
from anvil.tools.base import ToolSpec

DeltaCallback = Callable[[str, str], None]


class LLMError(RuntimeError):
    pass


class LLMCancelled(LLMError):
    pass


class LLMStreamInterrupted(LLMError):
    def __init__(self, message: str, *, partial: bool) -> None:
        super().__init__(message)
        self.partial = partial


def _wait_before_retry(attempt: int, cancel: Event | None) -> None:
    delay = min(2**attempt, 16)
    if cancel is None:
        time.sleep(delay)
        return
    if cancel.wait(delay):
        raise LLMCancelled("cancelled")


class DeepSeekClient:
    """OpenAI-compatible Chat Completions client tuned for DeepSeek V4.

    When tools are enabled, DeepSeek requires every subsequent assistant
    message to include the previous `reasoning_content`. Dropping that field
    yields HTTP 400.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._http = httpx.Client(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(config.request_timeout, connect=30.0),
        )
        self._stream = None

    def close(self) -> None:
        self.abort()
        self._http.close()

    def abort(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.close()
        except Exception:
            pass

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        on_delta: DeltaCallback | None = None,
        stream: bool = True,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        cancel: Event | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        body = self._payload(
            messages,
            tools,
            stream=stream,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        last_error: Exception | None = None
        for attempt in range(5):
            if cancel is not None and cancel.is_set():
                raise LLMCancelled("cancelled")
            try:
                if stream:
                    return self._complete_stream(body, on_delta, cancel=cancel)
                return self._complete_once(body)
            except LLMCancelled:
                raise
            except LLMStreamInterrupted as exc:
                if exc.partial or attempt >= 4:
                    raise
                last_error = exc
                _wait_before_retry(attempt, cancel)
                continue
            except LLMError:
                raise
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in {429, 500, 502, 503, 504} and attempt < 4:
                    last_error = exc
                    _wait_before_retry(attempt, cancel)
                    continue
                detail = _response_detail(exc.response)
                raise LLMError(f"LLM HTTP {status}: {detail}") from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if cancel is not None and cancel.is_set():
                    raise LLMCancelled("cancelled") from exc
                if attempt < 4:
                    last_error = exc
                    _wait_before_retry(attempt, cancel)
                    continue
                raise LLMError(f"LLM request failed: {exc}") from exc
        raise LLMError(f"LLM request failed after retries: {last_error}")

    def _payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        stream: bool,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        use_thinking = self.config.thinking if thinking is None else thinking
        effort = self.config.reasoning_effort if reasoning_effort is None else reasoning_effort
        # With tools, DeepSeek requires every prior reasoning_content back.
        include_reasoning = bool(tools) or use_thinking
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                m.to_openai(include_reasoning=include_reasoning) for m in messages
            ],
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
            "thinking": {"type": "enabled" if use_thinking else "disabled"},
        }
        if use_thinking:
            payload["reasoning_effort"] = effort
        if tools:
            payload["tools"] = [tool.openai_schema() for tool in tools]
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _complete_once(self, body: dict[str, Any]) -> LLMResponse:
        response = self._http.post("/chat/completions", json=body)
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=response.request, response=response
            )
        data = response.json()
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM payload: {data!r}") from exc
        if not isinstance(choice, dict) or "message" not in choice:
            raise LLMError(f"Unexpected LLM payload: {data!r}")
        return parse_assistant_choice(choice, data.get("usage"))

    def _complete_stream(
        self,
        body: dict[str, Any],
        on_delta: DeltaCallback | None,
        cancel: Event | None = None,
    ) -> LLMResponse:
        with self._http.stream("POST", "/chat/completions", json=body) as response:
            self._stream = response
            try:
                return self._read_stream(response, on_delta, cancel)
            finally:
                if self._stream is response:
                    self._stream = None

    def _read_stream(
        self,
        response: httpx.Response,
        on_delta: DeltaCallback | None,
        cancel: Event | None,
    ) -> LLMResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        call_slots: dict[int, dict[str, str]] = {}
        usage = Usage()
        finish_reason: str | None = None
        saw_terminal = False
        saw_payload = False
        if response.status_code >= 400:
            response.read()
            raise httpx.HTTPStatusError(
                "error", request=response.request, response=response
            )
        try:
            for line in response.iter_lines():
                if cancel is not None and cancel.is_set():
                    raise LLMCancelled("cancelled")
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_terminal = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = parse_usage(chunk["usage"])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                if finish_reason:
                    saw_terminal = True
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content")
                if reasoning is None:
                    reasoning = delta.get("reasoning")
                if reasoning:
                    saw_payload = True
                    reasoning_parts.append(reasoning)
                    if on_delta:
                        on_delta("reasoning", reasoning)
                text = delta.get("content")
                if text:
                    saw_payload = True
                    content_parts.append(text)
                    if on_delta:
                        on_delta("content", text)
                raw_calls = delta.get("tool_calls") or []
                if raw_calls:
                    saw_payload = True
                for raw_call in raw_calls:
                    index = int(raw_call.get("index") or 0)
                    slot = call_slots.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    if raw_call.get("id"):
                        slot["id"] = raw_call["id"]
                    function = raw_call.get("function") or {}
                    if function.get("name"):
                        slot["name"] += function["name"]
                    if function.get("arguments"):
                        slot["arguments"] += function["arguments"]
        except (httpx.TimeoutException, httpx.TransportError, httpx.StreamError) as exc:
            if cancel is not None and cancel.is_set():
                raise LLMCancelled("cancelled") from exc
            raise LLMStreamInterrupted(
                "LLM stream interrupted before completion.", partial=saw_payload
            ) from exc
        if not saw_terminal:
            raise LLMStreamInterrupted(
                "LLM stream ended before completion.", partial=saw_payload
            )

        tool_calls = [
            parse_tool_call(
                call_slots[i]["id"] or f"call_{i}",
                call_slots[i]["name"],
                call_slots[i]["arguments"],
            )
            for i in sorted(call_slots)
            if call_slots[i]["name"]
        ]
        return LLMResponse(
            content="".join(content_parts) or None,
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )


# Kept for tests that imported the private helper.
_tool_call_from_parts = parse_tool_call
_parse_tool_calls = parse_tool_calls
_parse_usage = parse_usage


def _response_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except Exception:
        return response.text[:800]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(data)[:800]
