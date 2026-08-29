from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    arguments_raw: str = ""
    parse_error: bool = False


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass
class Message:
    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_openai(self, *, include_reasoning: bool) -> dict[str, Any]:
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id or "",
                "content": self.content or "",
            }
        if self.role == "assistant":
            payload: dict[str, Any] = {
                "role": "assistant",
                "content": self.content,
            }
            if include_reasoning:
                # DeepSeek returns 400 if tools are present and this field is dropped.
                payload["reasoning_content"] = self.reasoning_content or ""
            if self.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments_raw
                            or _dump_args(call.arguments),
                        },
                    }
                    for call in self.tool_calls
                ]
            return payload
        return {"role": self.role, "content": self.content or ""}


@dataclass
class LLMResponse:
    content: str | None
    reasoning_content: str | None
    tool_calls: list[ToolCall]
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None

    def as_message(self) -> Message:
        return Message(
            role="assistant",
            content=self.content,
            reasoning_content=self.reasoning_content,
            tool_calls=self.tool_calls or None,
        )


def _dump_args(arguments: dict[str, Any]) -> str:
    import json

    return json.dumps(arguments, ensure_ascii=False)
