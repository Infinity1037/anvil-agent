"""Approval summary and full-file preview. Pure rendering, no terminal I/O."""

from __future__ import annotations

from pathlib import Path

from rich.syntax import Syntax
from rich.text import Text

from anvil.agent.permissions import PREVIEW_HARD_CAP, SUMMARY_LINES, preview_call
from anvil.llm.types import ToolCall
from anvil.safety import DangerousCommandError, assert_safe_command
from anvil.tui.cards import (
    append_diff_row,
    cluster_rows,
    count_content_lines,
    diff_stats,
    span_diff,
)

_LEXERS = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".md": "markdown",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".css": "css",
    ".html": "html",
    ".htm": "html",
    ".sh": "bash",
    ".bash": "bash",
    ".ps1": "powershell",
    ".xml": "xml",
    ".sql": "sql",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
}


def lexer_for_path(path: str) -> str:
    return _LEXERS.get(Path(path or "").suffix.lower(), "text")


def highlight_lines(path: str, lines: list[str]) -> list[Text]:
    """Syntax-color ``lines``. Falls back to plain text if highlighting fails."""
    if not lines:
        return []
    code = "\n".join(lines)
    try:
        syntax = Syntax(code, lexer_for_path(path), theme="ansi_dark", background_color="default")
        rendered = syntax.highlight(code)
        parts = list(rendered.split("\n"))
        if len(parts) > len(lines) and not parts[-1].plain:
            parts = parts[: len(lines)]
        while len(parts) < len(lines):
            parts.append(Text())
        return parts[: len(lines)]
    except Exception:
        return [Text(line) for line in lines]


def approval_title(call: ToolCall) -> str:
    args = call.arguments or {}
    if call.name == "write_file":
        path = str(args.get("path") or "")
        n = count_content_lines(str(args.get("content") or ""))
        return f"允许 write_file?  {path}  {n} 行".rstrip()
    if call.name == "edit_file":
        path = str(args.get("path") or "")
        added, removed = diff_stats(
            span_diff(str(args.get("old_string") or ""), str(args.get("new_string") or ""))
        )
        return f"允许 edit_file?  {path}  +{added} -{removed}".rstrip()
    if call.name == "run_shell":
        return "允许 run_shell?"
    return f"允许 {call.name}?"


def approval_has_more(call: ToolCall) -> bool:
    args = call.arguments or {}
    if call.name == "write_file":
        return count_content_lines(str(args.get("content") or "")) > SUMMARY_LINES
    if call.name == "edit_file":
        clustered = cluster_rows(
            span_diff(str(args.get("old_string") or ""), str(args.get("new_string") or ""))
        )
        return len(clustered) > SUMMARY_LINES
    return False


def is_file_preview(call: ToolCall) -> bool:
    return call.name in {"write_file", "edit_file"}


def render_approval_summary(call: ToolCall) -> Text:
    return render_approval(call, limit=SUMMARY_LINES, expand_hint=True)


def render_approval_full(call: ToolCall) -> Text:
    return render_approval(call, limit=None, expand_hint=False)


def render_approval(call: ToolCall, *, limit: int | None, expand_hint: bool) -> Text:
    args = call.arguments or {}
    if call.name == "write_file":
        return render_write_preview(
            str(args.get("path") or ""),
            str(args.get("content") or ""),
            limit=limit,
            expand_hint=expand_hint,
        )
    if call.name == "edit_file":
        return render_edit_preview(
            str(args.get("path") or ""),
            str(args.get("old_string") or ""),
            str(args.get("new_string") or ""),
            limit=limit,
            expand_hint=expand_hint,
        )
    if call.name == "run_shell":
        return render_shell_preview(str(args.get("command") or ""), cwd=str(args.get("cwd") or ""))
    return Text(preview_call(call, max_lines=limit))


def render_write_preview(
    path: str,
    content: str,
    *,
    limit: int | None,
    expand_hint: bool = False,
) -> Text:
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    total = len(lines)
    cap = PREVIEW_HARD_CAP if limit is None else limit
    shown = lines[:cap]
    highlighted = highlight_lines(path, shown)
    body = Text()
    width = max(4, len(str(max(total, 1))))
    for index, line in enumerate(highlighted):
        if index:
            body.append("\n")
        body.append(f"{index + 1:>{width}}  ", style="dim")
        body.append_text(line)
    hidden = total - len(shown)
    if hidden > 0:
        body.append("\n")
        hint = f"     … {hidden} 行未显示"
        if expand_hint:
            hint += "  ctrl+e 看全文"
        body.append(hint, style="dim")
    elif not shown:
        body.append("(empty)", style="dim")
    return body


def render_edit_preview(
    path: str,
    old: str,
    new: str,
    *,
    limit: int | None,
    expand_hint: bool = False,
) -> Text:
    del path  # title carries the path; body is the hunks
    rows = span_diff(old, new)
    clustered = cluster_rows(rows)
    if not clustered:
        return Text("(no changes)", style="dim")
    cap = PREVIEW_HARD_CAP if limit is None else limit
    shown = clustered[:cap]
    body = Text()
    for index, row in enumerate(shown):
        if index:
            body.append("\n")
        _append_approval_diff_row(body, row)
    if len(clustered) > len(shown):
        shown_changes = sum(1 for row in shown if row.kind in {"add", "delete"})
        hidden = sum(1 for row in rows if row.kind in {"add", "delete"}) - shown_changes
        if hidden > 0:
            body.append("\n")
            hint = f"     … {hidden} 处改动未显示"
            if expand_hint:
                hint += "  ctrl+e 看全文"
            body.append(hint, style="dim")
    return body


def render_shell_preview(command: str, *, cwd: str = "") -> Text:
    body = Text()
    if cwd:
        body.append(f"cwd: {cwd}", style="dim")
        body.append("\n")
    body.append("$ ", style="bold")
    body.append((command or "").strip() or "(empty command)")
    if _command_looks_destructive(command):
        body.append("\n")
        body.append("风险: 匹配到危险命令模式", style="bold red")
    return body


def _append_approval_diff_row(body: Text, row) -> None:
    if row.kind == "gap":
        try:
            count = int(row.text)
        except ValueError:
            body.append(f"     … {row.text} …", style="dim")
            return
        body.append(f"     … {count} 行未改 …", style="dim")
        return
    append_diff_row(body, row)


def _command_looks_destructive(command: str) -> bool:
    try:
        assert_safe_command(command or "")
    except DangerousCommandError:
        return True
    return False
