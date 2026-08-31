"""Shared display strings. No TUI imports — keeps cli/ui/tui load order acyclic."""

from __future__ import annotations

STOP_LABELS = {
    "completed": "Done",
    "cancelled": "Interrupted",
    "max_turns": "Stopped: too many steps",
    "tool_errors": "Stopped: repeated tool errors",
    "no_progress": "Stopped: repeating the same tool call",
    "length": "Stopped: model output truncated",
    "llm_error": "Model request failed",
}


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
