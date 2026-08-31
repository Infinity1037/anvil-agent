"""Per-tool transcript cards. Pure rendering, no terminal I/O.

Write/Edit bodies come from tool arguments (and a parsed unified diff when
the result has one). Read/search stay as a one-line header unless expanded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from rich.text import Text

from anvil.tui.fold import (
    GLANCE_SAMPLES,
    TOOL_LABELS,
    WRITE_PREVIEW,
    visible_body,
)

_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_READ_LINES = re.compile(r"\((\d+) lines\)")


@dataclass(frozen=True)
class DiffRow:
    kind: str  # add, delete, context, gap
    line_no: int
    text: str


def key_argument(kind: str, arguments: dict | None) -> str:
    args = arguments or {}
    if kind in {"read_file", "write_file", "edit_file"}:
        return str(args.get("path") or args.get("file_path") or "")
    if kind == "list_dir":
        return str(args.get("path") or ".")
    if kind in {"glob", "grep"}:
        return str(args.get("pattern") or "")
    if kind == "run_shell":
        command = str(args.get("command") or "")
        return command if len(command) <= 80 else command[:77] + "..."
    if kind == "todo":
        from anvil.tools.todo import todo_summary

        return todo_summary(args.get("items") or [])
    return ""


def count_content_lines(content: str) -> int:
    if not content:
        return 0
    if content.endswith("\n"):
        return content.count("\n")
    return content.count("\n") + 1


def extract_unified_diff(content: str) -> str:
    text = content or ""
    if text.startswith("--- "):
        return text
    index = text.find("\n--- ")
    if index >= 0:
        return text[index + 1 :]
    return ""


def parse_unified_diff(content: str) -> list[DiffRow]:
    diff = extract_unified_diff(content)
    if not diff:
        return []
    rows: list[DiffRow] = []
    old_no = new_no = 0
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("@@"):
            match = _HUNK.match(line)
            if not match:
                continue
            in_hunk = True
            old_no = int(match.group(1))
            new_no = int(match.group(2))
            continue
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            rows.append(DiffRow("add", new_no, line[1:]))
            new_no += 1
        elif line.startswith("-"):
            rows.append(DiffRow("delete", old_no, line[1:]))
            old_no += 1
        elif line.startswith("\\"):
            continue
        else:
            code = line[1:] if line.startswith(" ") else line
            rows.append(DiffRow("context", new_no, code))
            old_no += 1
            new_no += 1
    return rows


def span_diff(old: str, new: str) -> list[DiffRow]:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    if not old_lines and not new_lines:
        return []
    rows: list[DiffRow] = []
    old_no = new_no = 1
    matcher = SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in new_lines[j1:j2]:
                rows.append(DiffRow("context", new_no, line))
                old_no += 1
                new_no += 1
            continue
        if tag in {"replace", "delete"}:
            for line in old_lines[i1:i2]:
                rows.append(DiffRow("delete", old_no, line))
                old_no += 1
        if tag in {"replace", "insert"}:
            for line in new_lines[j1:j2]:
                rows.append(DiffRow("add", new_no, line))
                new_no += 1
    return rows


def diff_stats(rows: list[DiffRow]) -> tuple[int, int]:
    added = sum(1 for row in rows if row.kind == "add")
    removed = sum(1 for row in rows if row.kind == "delete")
    return added, removed


def cluster_rows(rows: list[DiffRow], *, context: int = 3) -> list[DiffRow]:
    """Keep change hunks plus nearby context; elide the unchanged middle."""
    if not rows:
        return []
    changed = [index for index, row in enumerate(rows) if row.kind in {"add", "delete"}]
    if not changed:
        return []
    merge_gap = 2 * context
    clusters: list[tuple[int, int]] = []
    start = end = changed[0]
    for index in changed[1:]:
        if index - end <= merge_gap:
            end = index
        else:
            clusters.append((start, end))
            start = end = index
    clusters.append((start, end))
    out: list[DiffRow] = []
    prev_end = -1
    last = len(rows) - 1
    for cstart, cend in clusters:
        lo = max(0, cstart - context)
        hi = min(last, cend + context)
        if prev_end >= 0:
            gap = lo - prev_end - 1
            if gap > 0:
                out.append(DiffRow("gap", 0, str(gap)))
        out.extend(rows[lo : hi + 1])
        prev_end = hi
    return out


def format_diff_rows(rows: list[DiffRow], *, expanded: bool, limit: int = WRITE_PREVIEW) -> Text:
    if not rows:
        return Text()
    shown = rows if expanded or len(rows) <= limit else rows[:limit]
    body = Text()
    for index, row in enumerate(shown):
        if index:
            body.append("\n")
        append_diff_row(body, row)
    hidden = len(rows) - len(shown)
    if hidden > 0:
        noun = "line" if hidden == 1 else "lines"
        body.append("\n")
        body.append(f"     … ({hidden} more {noun}, Ctrl+O to expand)", style="dim")
    return body


def append_diff_row(body: Text, row: DiffRow) -> None:
    if row.kind == "gap":
        try:
            count = int(row.text)
        except ValueError:
            body.append(f"     … {row.text} …", style="dim")
            return
        noun = "unchanged line" if count == 1 else "unchanged lines"
        body.append(f"     … {count} {noun} …", style="dim")
        return
    body.append(f"{row.line_no:>4} ", style="dim")
    if row.kind == "add":
        body.append(f"+ {row.text}", style="green")
    elif row.kind == "delete":
        body.append(f"- {row.text}", style="red")
    else:
        body.append(f"  {row.text}", style="dim")


def numbered_content(content: str, *, expanded: bool, limit: int = WRITE_PREVIEW) -> Text:
    if content == "":
        return Text()
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        return Text()
    shown = lines if expanded or len(lines) <= limit else lines[:limit]
    body = Text()
    width = max(4, len(str(len(lines))))
    for index, line in enumerate(shown):
        if index:
            body.append("\n")
        body.append(f"{index + 1:>{width}}  ", style="dim")
        body.append(line)
    hidden = len(lines) - len(shown)
    if hidden > 0:
        noun = "line" if hidden == 1 else "lines"
        body.append("\n")
        body.append(
            f"     … ({hidden} more {noun}, {len(lines)} total, Ctrl+O to expand)",
            style="dim",
        )
    return body


def edit_rows(arguments: dict | None, result: str) -> list[DiffRow]:
    rows = parse_unified_diff(result)
    if rows:
        return rows
    args = arguments or {}
    old = str(args.get("old_string") or "")
    new = str(args.get("new_string") or "")
    if old or new:
        return span_diff(old, new)
    return []


def write_rows(arguments: dict | None, result: str) -> list[DiffRow]:
    return parse_unified_diff(result)


def tool_chip(
    kind: str,
    arguments: dict | None,
    result: str = "",
    *,
    live: bool = False,
    ok: bool | None = None,
) -> str:
    if live or ok is False:
        return ""
    args = arguments or {}
    if kind == "edit_file":
        added, removed = diff_stats(edit_rows(args, result))
        return _plus_minus(added, removed)
    if kind == "write_file":
        rows = write_rows(args, result)
        if rows:
            return _plus_minus(*diff_stats(rows))
        content = str(args.get("content") or "")
        if content:
            n = count_content_lines(content)
            return f"{n} {'line' if n == 1 else 'lines'}"
        match = re.search(r"\((\d+) lines\)", result)
        if match:
            n = int(match.group(1))
            return f"{n} {'line' if n == 1 else 'lines'}"
        return ""
    if kind == "read_file":
        match = _READ_LINES.search((result or "").splitlines()[0] if result else "")
        if match:
            n = int(match.group(1))
            return f"{n} {'line' if n == 1 else 'lines'}"
        n = len(_nonempty(result))
        return f"{n} {'line' if n == 1 else 'lines'}" if n else ""
    if kind == "grep":
        n = len(_nonempty(result))
        if n == 0:
            return "no matches"
        return f"{n} {'match' if n == 1 else 'matches'}"
    if kind == "glob":
        n = len(_nonempty(result))
        if n == 0:
            return "no files"
        return f"{n} {'file' if n == 1 else 'files'}"
    if kind == "list_dir":
        lines = _nonempty(result)
        if result.strip().startswith("(empty"):
            return "empty"
        n = len(lines)
        return f"{n} {'entry' if n == 1 else 'entries'}" if n else ""
    return ""


def tool_header(
    kind: str,
    arguments: dict | None,
    *,
    live: bool,
    ok: bool | None,
    result: str = "",
    summary: str = "",
) -> Text:
    if ok is False:
        color = "red"
    elif live:
        color = "cyan"
    else:
        color = "green"
    verb = "Using" if live else "Used"
    label = TOOL_LABELS.get(kind, kind)
    path = summary or key_argument(kind, arguments)
    header = Text()
    header.append("● ", style=f"bold {color}")
    header.append(f"{verb} {label}", style=f"bold {color}")
    if path:
        header.append(f"  {path}", style="dim")
    chip = tool_chip(kind, arguments, result, live=live, ok=ok)
    if chip:
        header.append("  · ", style="dim")
        _append_chip(header, chip, kind)
    return header


def render_body(
    kind: str,
    *,
    arguments: dict | None,
    result: str,
    live: bool,
    expanded: bool,
    ok: bool | None,
) -> Text:
    if ok is False:
        text = visible_body(kind, result, expanded=expanded, live=False)
        return Text(text, style="red") if text else Text()
    args = arguments or {}
    if kind == "write_file":
        rows = write_rows(args, result)
        if rows:
            return format_diff_rows(rows, expanded=expanded)
        content = str(args.get("content") or "")
        if content:
            return numbered_content(content, expanded=expanded)
        if live:
            return Text()
        return _dim_preview(kind, result, expanded)
    if kind == "edit_file":
        rows = edit_rows(args, result)
        if rows:
            return format_diff_rows(rows, expanded=expanded)
        if live:
            return Text()
        return _dim_preview(kind, result, expanded)
    if kind == "read_file":
        if not expanded:
            return Text()
        return _dim_preview(kind, result, True)
    if kind == "list_dir":
        if not expanded:
            return Text()
        return _dim_preview(kind, result, True)
    if kind in {"grep", "glob"}:
        if expanded:
            return _dim_preview(kind, result, True)
        return _glance(result)
    if kind == "todo":
        from anvil.tools.todo import coerce_todo_view

        return _todo_body(coerce_todo_view(result), expanded=expanded)
    if kind == "run_shell":
        text = visible_body(kind, result, expanded=expanded, live=live)
        if not text and live:
            return Text("running…", style="dim italic")
        return Text(text, style="dim")
    text = visible_body(kind, result, expanded=expanded, live=live)
    if not text and live:
        return Text("running…", style="dim italic")
    return Text(text, style="dim") if text else Text()


def render_card(
    kind: str,
    *,
    arguments: dict | None = None,
    result: str = "",
    live: bool = False,
    expanded: bool = False,
    ok: bool | None = None,
    summary: str = "",
) -> Text:
    card = Text()
    card.append_text(
        tool_header(
            kind,
            arguments,
            live=live,
            ok=ok,
            result=result,
            summary=summary,
        )
    )
    body = render_body(
        kind,
        arguments=arguments,
        result=result,
        live=live,
        expanded=expanded,
        ok=ok,
    )
    if body.plain:
        card.append("\n")
        card.append_text(body)
    elif live and kind not in {"write_file", "edit_file", "read_file", "list_dir", "grep", "glob"}:
        card.append("\n")
        card.append("running…", style="dim italic")
    return card


def body_is_truncated(
    kind: str,
    *,
    arguments: dict | None,
    result: str,
    ok: bool | None,
) -> bool:
    collapsed = render_body(
        kind, arguments=arguments, result=result, live=False, expanded=False, ok=ok
    ).plain
    full = render_body(
        kind, arguments=arguments, result=result, live=False, expanded=True, ok=ok
    ).plain
    return full != collapsed and ("Ctrl+O" in collapsed or len(full) > len(collapsed))


def _plus_minus(added: int, removed: int) -> str:
    parts: list[str] = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    return " ".join(parts)


def _append_chip(header: Text, chip: str, kind: str) -> None:
    if kind in {"edit_file", "write_file"} and (chip.startswith("+") or " -" in f" {chip}"):
        for index, part in enumerate(chip.split()):
            if index:
                header.append(" ")
            if part.startswith("+"):
                header.append(part, style="green")
            elif part.startswith("-"):
                header.append(part, style="red")
            else:
                header.append(part, style="dim")
        return
    header.append(chip, style="dim")


def _nonempty(text: str) -> list[str]:
    if not text:
        return []
    return [line for line in text.splitlines() if line.strip()]


def _glance(result: str) -> Text:
    lines = _nonempty(result)
    if not lines:
        return Text()
    samples = lines[:GLANCE_SAMPLES]
    rest = len(lines) - len(samples)
    shown = ", ".join(_path_from_grep(line) for line in samples)
    if rest > 0:
        shown += f", +{rest} more"
    return Text(shown, style="dim")


def _path_from_grep(line: str) -> str:
    first = line.find(":")
    if first <= 0:
        return line
    second = line.find(":", first + 1)
    if second <= 0:
        return line[:first]
    return line[:first]


def _dim_preview(kind: str, result: str, expanded: bool) -> Text:
    text = visible_body(kind, result, expanded=expanded, live=False)
    return Text(text, style="dim") if text else Text()


def _todo_body(content: str, *, expanded: bool) -> Text:
    shown = visible_body("todo", content, expanded=expanded, live=False)
    body = Text()
    for index, line in enumerate(shown.splitlines()):
        if index:
            body.append("\n")
        if line.lstrip().startswith("✓") or line.startswith("[x]"):
            body.append(line, style="dim")
        elif line.startswith("[>]"):
            body.append(line, style="bold cyan")
        else:
            body.append(line)
    return body
