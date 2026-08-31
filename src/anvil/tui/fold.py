"""Preview vs full text for in-place expand. No terminal I/O."""

from __future__ import annotations

from rich.cells import cell_len

THINKING_PREVIEW = 2
TOOL_PREVIEW = 3
SHELL_PREVIEW = 8
WRITE_PREVIEW = 10
GLANCE_SAMPLES = 3

TOOL_LABELS = {
    "thinking": "Thinking",
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "list_dir": "List",
    "glob": "Glob",
    "grep": "Grep",
    "run_shell": "Shell",
    "todo": "Todo",
}


def preview_limit(kind: str) -> int:
    if kind == "thinking":
        return THINKING_PREVIEW
    if kind == "run_shell":
        return SHELL_PREVIEW
    if kind in {"write_file", "edit_file"}:
        return WRITE_PREVIEW
    if kind == "todo":
        return 10
    return TOOL_PREVIEW


def wrap_to_width(text: str, width: int) -> list[str]:
    """Split on newlines, then wrap each logical line to a display width."""
    if not text:
        return []
    logical = text.splitlines()
    if width <= 0:
        return logical
    rows: list[str] = []
    for line in logical:
        rows.extend(_wrap_one(line, width) if line else [""])
    return rows


def visible_body(
    kind: str,
    full: str,
    *,
    expanded: bool,
    live: bool,
    width: int = 0,
) -> str:
    """Text actually shown on a transcript card.

    Truncation is by *visual* rows (after wrapping to ``width``), not raw
    newlines — DeepSeek thinking is often one long paragraph.
    """
    text = (full or "").rstrip()
    if kind == "thinking" and live:
        if not text:
            return "thinking…"
        lines = wrap_to_width(text, width)
        if expanded:
            body = text
        elif len(lines) <= THINKING_PREVIEW:
            body = "\n".join(lines) if lines else text
        else:
            body = "\n".join(lines[-THINKING_PREVIEW:])
        return "thinking…\n" + body
    if not text:
        return ""
    lines = wrap_to_width(text, width)
    limit = preview_limit(kind)
    if expanded:
        return text
    if len(lines) <= limit:
        return "\n".join(lines) if width else text
    rest = len(lines) - limit
    noun = "line" if rest == 1 else "lines"
    return "\n".join(lines[:limit]) + f"\n… ({rest} more {noun}, Ctrl+O to expand)"


def block_title(kind: str, summary: str = "") -> str:
    label = TOOL_LABELS.get(kind, kind)
    if summary:
        return f"{label}  {summary}"
    return label


def looks_like_diff(content: str) -> bool:
    return content.startswith("--- ") or "\n--- " in content


def _wrap_one(line: str, width: int) -> list[str]:
    if cell_len(line) <= width:
        return [line]
    rows: list[str] = []
    current = ""
    current_w = 0
    for token in _tokens(line):
        token_w = cell_len(token)
        if token_w > width:
            if current:
                rows.append(current.rstrip())
                current, current_w = "", 0
            rows.extend(_hard_wrap(token, width))
            continue
        if current and current_w + token_w > width:
            rows.append(current.rstrip())
            token = token.lstrip()
            token_w = cell_len(token)
            current, current_w = token, token_w
            continue
        current += token
        current_w += token_w
    if current:
        rows.append(current.rstrip())
    return rows or [""]


def _tokens(line: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []
    for char in line:
        if char.isspace():
            buf.append(char)
            tokens.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _hard_wrap(line: str, width: int) -> list[str]:
    rows: list[str] = []
    buf: list[str] = []
    used = 0
    for char in line:
        char_width = cell_len(char)
        if char_width <= 0:
            buf.append(char)
            continue
        if used + char_width > width and buf:
            rows.append("".join(buf))
            buf = [char]
            used = char_width
        else:
            buf.append(char)
            used += char_width
    rows.append("".join(buf) if buf else "")
    return rows or [""]
