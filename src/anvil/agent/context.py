from __future__ import annotations

from pathlib import Path

from anvil.llm.types import Message, Usage
from anvil.tools.result import ToolResult

TOOL_RESULT_LIMIT = 8_000
PLACEHOLDER = "(old tool result truncated to save context)"


def message_chars(messages: list[Message]) -> int:
    size = 0
    for message in messages:
        size += len(message.content or "")
        size += len(message.reasoning_content or "")
        if message.tool_calls:
            for call in message.tool_calls:
                size += len(call.name) + len(call.arguments_raw)
    return size


def estimate_tokens(messages: list[Message]) -> int:
    """Uncalibrated fallback: characters / 3. Prefer ContextManager.estimate."""
    return max(1, message_chars(messages) // 3)


def clip_text(text: str, limit: int = TOOL_RESULT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head, tail = 5_000, 2_000
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n... [{omitted} characters truncated] ...\n\n{text[-tail:]}"


def has_tool_calls(message: Message) -> bool:
    return bool(message.tool_calls)


def unpaired_tools(messages: list[Message]) -> list[int]:
    """Indexes of tool messages that do not belong to the open assistant tool_calls."""
    pending: set[str] = set()
    orphans: list[int] = []
    for index, message in enumerate(messages):
        if message.role == "assistant" and message.tool_calls:
            pending = {call.id for call in message.tool_calls}
            continue
        if message.role == "tool":
            call_id = message.tool_call_id or ""
            if call_id not in pending:
                orphans.append(index)
            else:
                pending.discard(call_id)
            continue
        pending = set()
    return orphans


def unpaired_assistant_calls(messages: list[Message]) -> list[tuple[str, str]]:
    """(call_id, tool_name) from assistant tool_calls that have no tool result yet."""
    pending: dict[str, str] = {}
    leftover: list[tuple[str, str]] = []

    def flush() -> None:
        leftover.extend(pending.items())
        pending.clear()

    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            flush()
            for call in message.tool_calls:
                if call.id:
                    pending[call.id] = call.name
            continue
        if message.role == "tool":
            pending.pop(message.tool_call_id or "", None)
            continue
        flush()
    flush()
    return leftover


def pair_messages(messages: list[Message]) -> list[Message]:
    """Drop orphan tool rows and synthesize results for unpaired assistant calls.

    The returned list is a view. Callers must not treat it as the session log.
    """
    orphan = set(unpaired_tools(messages))
    view = [message for index, message in enumerate(messages) if index not in orphan]
    missing = unpaired_assistant_calls(view)
    if not missing:
        return view
    extra = [
        Message(
            role="tool",
            content=ToolResult.fail(
                "cancelled",
                "tool result missing from transcript",
                hint="Retry the call if it is still needed.",
            ).to_message_content(),
            tool_call_id=call_id,
        )
        for call_id, _name in missing
    ]
    return view + extra


class ContextManager:
    """Cheap-first context control. prepare() never mutates the input list."""

    def __init__(self, budget: int, workspace: Path | None = None) -> None:
        self.budget = budget
        self.workspace = workspace
        self._token_ratio: float | None = None

    def note_prompt_usage(self, view: list[Message], usage: Usage) -> None:
        chars = message_chars(view)
        if usage.prompt_tokens > 0 and chars > 0:
            self._token_ratio = usage.prompt_tokens / chars

    def estimate(self, messages: list[Message]) -> int:
        chars = message_chars(messages)
        if self._token_ratio is not None:
            return max(1, int(chars * self._token_ratio))
        return max(1, chars // 3)

    def ingest_tool_result(self, text: str, *, call_id: str = "") -> str:
        if self.workspace and len(text) > TOOL_RESULT_LIMIT:
            name = (call_id or "tool") + ".txt"
            safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)
            path = self.workspace / ".anvil" / "tool-output" / safe
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                preview = clip_text(text, limit=4_000)
                return (
                    f"[output truncated: {len(text)} characters; showing head and tail. "
                    "Re-run with a tighter command, glob, or offset/limit if you need more.]\n\n"
                    + preview
                )
            except OSError:
                return clip_text(text)
        return clip_text(text)

    def prepare(self, messages: list[Message]) -> list[Message]:
        view = pair_messages(list(messages))
        if self.estimate(view) <= int(self.budget * 0.75):
            return view
        self._shrink_old_tool_results(view, keep_tail=6)
        if self.estimate(view) <= int(self.budget * 0.9):
            return view
        self._shrink_old_tool_results(view, keep_tail=2)
        if self.estimate(view) <= int(self.budget * 0.9):
            return view
        view = self._drop_middle_turns(view)
        self._shrink_old_tool_results(view, keep_tail=1)
        return pair_messages(view)

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
        keep_head, keep_tail = 2, 6
        head_end = keep_head
        tail_start = max(keep_head, len(messages) - keep_tail)
        if head_end > 0 and has_tool_calls(messages[head_end - 1]):
            while head_end < len(messages) and messages[head_end].role == "tool":
                head_end += 1
        if (
            tail_start > 0
            and tail_start < len(messages)
            and messages[tail_start].role == "tool"
            and has_tool_calls(messages[tail_start - 1])
        ):
            tail_start -= 1
        if head_end >= tail_start:
            return messages
        removed = tail_start - head_end
        marker = Message(
            role="user",
            content=f"[context compacted: {removed} older messages omitted]",
        )
        return messages[:head_end] + [marker] + messages[tail_start:]
