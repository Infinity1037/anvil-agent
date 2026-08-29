from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx

from anvil.config import Config
from anvil.llm.types import LLMResponse, Message, ToolCall, Usage
from anvil.tools.base import ToolSpec

DeltaCallback = Callable[[str, str], None]


class LLMError(RuntimeError):
    pass


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

    def close(self) -> None:
        self._http.close()

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        on_delta: DeltaCallback | None = None,
        stream: bool = True,
    ) -> LLMResponse:
        body = self._payload(messages, tools, stream=stream)
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                if stream:
                    return self._complete_stream(body, on_delta)
                return self._complete_once(body)
            except LLMError:
                raise
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in {429, 500, 502, 503, 504} and attempt < 4:
                    time.sleep(min(2**attempt, 16))
                    last_error = exc
                    continue
                detail = _response_detail(exc.response)
                raise LLMError(f"LLM HTTP {status}: {detail}") from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < 4:
                    time.sleep(min(2**attempt, 16))
                    last_error = exc
                    continue
                raise LLMError(f"LLM request failed: {exc}") from exc
        raise LLMError(f"LLM request failed after retries: {last_error}")

    def _payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        include_reasoning = self.config.thinking
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                m.to_openai(include_reasoning=include_reasoning) for m in messages
            ],
            "max_tokens": self.config.max_tokens,
            "stream": stream,
            "thinking": {"type": "enabled" if self.config.thinking else "disabled"},
        }
        if self.config.thinking:
            payload["reasoning_effort"] = self.config.reasoning_effort
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
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM payload: {data!r}") from exc
        return LLMResponse(
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content"),
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
            usage=_parse_usage(data.get("usage")),
            finish_reason=choice.get("finish_reason"),
        )

    def _complete_stream(
        self,
        body: dict[str, Any],
        on_delta: DeltaCallback | None,
    ) -> LLMResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        call_slots: dict[int, dict[str, str]] = {}
        usage = Usage()
        finish_reason: str | None = None

        with self._http.stream("POST", "/chat/completions", json=body) as response:
            if response.status_code >= 400:
                response.read()
                raise httpx.HTTPStatusError(
                    "error", request=response.request, response=response
                )
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = _parse_usage(chunk["usage"])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    reasoning_parts.append(reasoning)
                    if on_delta:
                        on_delta("reasoning", reasoning)
                text = delta.get("content")
                if text:
                    content_parts.append(text)
                    if on_delta:
                        on_delta("content", text)
                for raw_call in delta.get("tool_calls") or []:
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

        tool_calls = [
            _tool_call_from_parts(
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


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    if not raw:
        return []
    parsed: list[ToolCall] = []
    for item in raw:
        function = item.get("function") or {}
        parsed.append(
            _tool_call_from_parts(
                item.get("id") or f"call_{len(parsed)}",
                function.get("name") or "",
                function.get("arguments") or "",
            )
        )
    return parsed


def _tool_call_from_parts(call_id: str, name: str, arguments_raw: str) -> ToolCall:
    raw = arguments_raw if arguments_raw is not None else "{}"
    try:
        loaded = json.loads(raw) if str(raw).strip() else {}
    except json.JSONDecodeError:
        return ToolCall(
            id=call_id,
            name=name,
            arguments={},
            arguments_raw=arguments_raw,
            parse_error=True,
        )
    if not isinstance(loaded, dict):
        return ToolCall(
            id=call_id,
            name=name,
            arguments={},
            arguments_raw=arguments_raw,
            parse_error=True,
        )
    return ToolCall(id=call_id, name=name, arguments=loaded, arguments_raw=raw)


def _parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    prompt = int(raw.get("prompt_tokens") or 0)
    completion = int(raw.get("completion_tokens") or 0)
    total = int(raw.get("total_tokens") or (prompt + completion))
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _response_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except Exception:
        return response.text[:800]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(data)[:800]
