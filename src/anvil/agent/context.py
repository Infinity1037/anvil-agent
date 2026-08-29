from __future__ import annotations

from anvil.llm.types import Message

TOOL_RESULT_LIMIT = 8_000
PLACEHOLDER = "(old tool result truncated to save context)"


def estimate_tokens(messages: list[Message]) -> int:
    size = 0
    for message in messages:
        size += len(message.content or "")
        size += len(message.reasoning_content or "")
        if message.tool_calls:
            for call in message.tool_calls:
                size += len(call.name) + len(call.arguments_raw)
    return max(1, size // 3)


def clip_text(text: str, limit: int = TOOL_RESULT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head, tail = 5_000, 2_000
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n... [{omitted} characters truncated] ...\n\n{text[-tail:]}"


class ContextManager:
    """Cheap-first context control: clip on ingest, then shrink old tool results."""

    def __init__(self, budget: int) -> None:
        self.budget = budget

    def ingest_tool_result(self, text: str) -> str:
        return clip_text(text)

    def prepare(self, messages: list[Message]) -> list[Message]:
        if estimate_tokens(messages) <= int(self.budget * 0.75):
            return messages
        compacted = list(messages)
        self._shrink_old_tool_results(compacted, keep_tail=6)
        if estimate_tokens(compacted) <= int(self.budget * 0.9):
            return compacted
        self._shrink_old_tool_results(compacted, keep_tail=2)
        if estimate_tokens(compacted) <= int(self.budget * 0.9):
            return compacted
        compacted = self._drop_middle_turns(compacted)
        self._shrink_old_tool_results(compacted, keep_tail=1)
        return compacted

    def _shrink_old_tool_results(self, messages: list[Message], keep_tail: int) -> None:
        tool_indexes = [i for i, m in enumerate(messages) if m.role == "tool"]
        for index in tool_indexes[:-keep_tail] if keep_tail else tool_indexes:
            message = messages[index]
            if message.content and message.content != PLACEHOLDER and len(message.content) > 200:
                messages[index] = Message(
                    role="tool",
                    content=PLACEHOLDER,
                    tool_call_id=message.tool_call_id,
                )

    def _drop_middle_turns(self, messages: list[Message]) -> list[Message]:
        if len(messages) <= 8:
            return messages
        head = messages[:2]
        tail = messages[-6:]
        removed = len(messages) - len(head) - len(tail)
        marker = Message(
            role="user",
            content=f"[context compacted: {removed} older messages omitted]",
        )
        return head + [marker] + tail
