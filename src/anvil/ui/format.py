"""Shared display strings. No TUI imports — keeps cli/ui/tui load order acyclic."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anvil.agent.context import ContextSnapshot
    from anvil.agent.loop import CompactResult
    from anvil.llm.types import Usage

STOP_LABELS = {
    "completed": "Done",
    "cancelled": "Interrupted",
    "max_turns": "Stopped: too many steps",
    "tool_errors": "Stopped: repeated tool errors",
    "no_progress": "Stopped: repeating the same tool call",
    "length": "Stopped: model output truncated",
    "llm_error": "Model request failed",
    "context_overflow": "Stopped: request exceeds the context budget",
}


def short_tokens(value: int) -> str:
    number = max(0, int(value))
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}".rstrip("0").rstrip(".") + "m"
    if number >= 1_000:
        return f"{number / 1_000:.1f}".rstrip("0").rstrip(".") + "k"
    return str(number)


def context_badge(snapshot: "ContextSnapshot", *, detailed: bool = True) -> str:
    percent = max(0, round(snapshot.usage_ratio * 100))
    label = f"ctx ≈{percent}%"
    if detailed:
        label += (
            f" ({short_tokens(snapshot.estimated_tokens)}/"
            f"{short_tokens(snapshot.budget)})"
        )
    return label


def context_report(snapshot: "ContextSnapshot", usage: "Usage") -> str:
    covered_messages = max(0, snapshot.covered_count - 1)
    checkpoint = (
        f"checkpoint covers {covered_messages} historical messages"
        if snapshot.covered_count
        else "checkpoint not active"
    )
    estimate = "calibrated by recent API usage" if snapshot.calibrated else "character estimate"
    return (
        f"context {context_badge(snapshot).removeprefix('ctx ')}\n"
        f"remaining ≈{short_tokens(snapshot.remaining_tokens)} tokens\n"
        f"messages {snapshot.history_messages} full / {snapshot.view_messages} active\n"
        f"{checkpoint}\n"
        f"estimate {estimate}\n"
        f"API tokens this process "
        f"{short_tokens(usage.prompt_tokens)} input / "
        f"{short_tokens(usage.completion_tokens)} output"
    )


def compact_result_text(result: "CompactResult") -> str:
    if result.status == "compacted":
        covered_messages = max(0, result.covered_count - 1)
        return (
            "上下文已压缩  "
            f"≈{short_tokens(result.before_tokens)} → {short_tokens(result.after_tokens)} tokens\n"
            f"checkpoint 覆盖 {covered_messages} 条历史消息；完整历史仍保留"
        )
    if result.status == "nothing_to_compact":
        return "当前没有足够的旧历史需要压缩；上下文保持不变"
    if result.status == "cancelled":
        return "已取消上下文压缩；原 checkpoint 保持不变"
    if result.status == "busy":
        return "当前任务仍在运行；请等待结束后再压缩上下文"
    detail = (result.error or "unknown error").strip()[:500]
    return f"上下文压缩失败：{detail}\n原 checkpoint 保持不变"


def tool_message_ok(content: str) -> bool:
    """True when a tool transcript line is a success. Matches ToolResult.to_message_content."""
    text = (content or "").lstrip()
    return not text.startswith("Error (") and not text.startswith("Error:")


def strip_internal(content: str) -> str:
    lines = [line for line in content.splitlines() if not line.startswith("[progress]")]
    text = "\n".join(lines).strip()
    if text.startswith("Error (") and "): " in text.split("\n", 1)[0]:
        first, _, rest = text.partition("\n")
        human = first.split("): ", 1)[1]
        return human if not rest else human + "\n" + rest
    if text.startswith("Error:"):
        return text[6:].strip()
    return text
