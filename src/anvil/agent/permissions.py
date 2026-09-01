from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff

from anvil.llm.types import ToolCall

PERMISSION_MODES = ("ask", "auto")
READ_ONLY_TOOLS = {
    "list_dir",
    "glob",
    "grep",
    "read_file",
    "load_skill",
    "todo",
}

SUMMARY_LINES = 10
PREVIEW_HARD_CAP = 4000


@dataclass(frozen=True)
class ApprovalDecision:
    """Outcome of one permission prompt. ``reason`` is only set on deny."""

    verdict: str
    reason: str = ""

    @classmethod
    def from_raw(cls, raw: object) -> ApprovalDecision:
        if isinstance(raw, cls):
            verdict = raw.verdict
            reason = raw.reason
        elif isinstance(raw, str):
            verdict = raw
            reason = ""
        else:
            return cls("deny")
        if verdict not in {"allow", "allow_session", "deny", "cancelled"}:
            return cls("deny")
        return cls(verdict, (reason or "").strip())


def parse_permission(token: str) -> str:
    raw = (token or "").strip().lower()
    if raw in {"yolo", "yes", "off"}:
        raw = "auto" if raw in {"yolo", "yes"} else "ask"
    if raw not in PERMISSION_MODES:
        raise ValueError("use /perm ask | auto")
    return raw


def needs_ask(mode: str, tool_name: str, session_allow: set[str] | None = None) -> bool:
    """True when a side-effecting tool must wait for the user in ask mode."""
    if (mode or "ask") == "auto":
        return False
    if tool_name in READ_ONLY_TOOLS:
        return False
    if session_allow and tool_name in session_allow:
        return False
    return True


def preview_call(call: ToolCall, *, max_lines: int | None = SUMMARY_LINES) -> str:
    """Plain-text preview from tool arguments. Default is the compact summary.

    Pass ``max_lines=None`` for the full body, still capped at PREVIEW_HARD_CAP.
    The handler has not run yet.
    """
    cap = PREVIEW_HARD_CAP if max_lines is None else max_lines
    args = call.arguments or {}
    if call.name == "run_shell":
        return _clip(str(args.get("command") or "").strip() or "(empty command)", cap)
    if call.name == "write_file":
        path = str(args.get("path") or "")
        body = str(args.get("content") or "")
        return f"{path}\n{_clip(body, cap)}".rstrip()
    if call.name == "edit_file":
        path = str(args.get("path") or "")
        old = str(args.get("old_string") or "")
        new = str(args.get("new_string") or "")
        diff = "\n".join(
            unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile=path,
                tofile=path,
                lineterm="",
                n=3,
            )
        )
        return f"{path}\n{_clip(diff or '(no changes)', cap)}".rstrip()
    parts = [f"{key}={args[key]!r}" for key in list(args)[:8]]
    return _clip(" ".join(parts) or call.name, cap)


def _clip(text: str, max_lines: int) -> str:
    lines = (text or "").splitlines() or [text or ""]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    hidden = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n… ({hidden} more lines)"
