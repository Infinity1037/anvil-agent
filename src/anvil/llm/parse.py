from __future__ import annotations

import json
from typing import Any

from anvil.llm.types import LLMResponse, ToolCall, Usage


def parse_tool_call(call_id: str, name: str, arguments_raw: Any) -> ToolCall:
    """Turn one provider tool-call payload into a ToolCall.

    Invalid JSON is not an exception: the registry reports invalid_json.
    """
    if isinstance(arguments_raw, dict):
        dumped = json.dumps(arguments_raw, ensure_ascii=False)
        return ToolCall(
            id=call_id,
            name=name,
            arguments=arguments_raw,
            arguments_raw=dumped,
            parse_error=False,
        )
    raw = "" if arguments_raw is None else str(arguments_raw)
    try:
        loaded = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return ToolCall(
            id=call_id,
            name=name,
            arguments={},
            arguments_raw=raw,
            parse_error=True,
        )
    if not isinstance(loaded, dict):
        return ToolCall(
            id=call_id,
            name=name,
            arguments={},
            arguments_raw=raw,
            parse_error=True,
        )
    return ToolCall(id=call_id, name=name, arguments=loaded, arguments_raw=raw)


def parse_tool_calls(raw: Any) -> list[ToolCall]:
    """Extract tool calls from a chat-completions message.

    Entries that are not objects, or that have an empty function name, are
    skipped. Missing ids are filled in. finish_reason is not consulted.
    """
    if not raw or not isinstance(raw, list):
        return []
    parsed: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        name = str(function.get("name") or "")
        if not name:
            continue
        call_id = str(item.get("id") or f"call_{len(parsed)}")
        arguments = function.get("arguments")
        if arguments is None:
            arguments = "{}"
        parsed.append(parse_tool_call(call_id, name, arguments))
    return parsed


def parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    prompt = _as_int(raw.get("prompt_tokens"))
    completion = _as_int(raw.get("completion_tokens"))
    total = _as_int(raw.get("total_tokens"))
    if total == 0:
        total = prompt + completion
    details = raw.get("completion_tokens_details")
    reasoning = 0
    if isinstance(details, dict):
        reasoning = _as_int(details.get("reasoning_tokens"))
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reasoning_tokens=reasoning,
    )


def parse_assistant_choice(choice: Any, usage_raw: Any = None) -> LLMResponse:
    """Parse one non-stream chat-completions choice. Missing fields become empty."""
    if not isinstance(choice, dict):
        return LLMResponse(content=None, reasoning_content=None, tool_calls=[], usage=parse_usage(usage_raw))
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        message = {}
    return LLMResponse(
        content=message.get("content"),
        reasoning_content=message.get("reasoning_content"),
        tool_calls=parse_tool_calls(message.get("tool_calls")),
        usage=parse_usage(usage_raw),
        finish_reason=choice.get("finish_reason"),
    )


def complete_without_tools(finish_reason: str | None) -> str:
    """Stop reason when the model returned no executable tool calls.

    Provider finish_reason is ignored except for truncation.
    """
    reason = (finish_reason or "").strip().lower()
    if reason in {"length", "max_tokens"}:
        return "length"
    return "completed"


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
