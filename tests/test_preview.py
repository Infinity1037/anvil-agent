from anvil.llm.types import ToolCall
from anvil.tui.preview import (
    approval_has_more,
    approval_title,
    highlight_lines,
    lexer_for_path,
    render_approval_full,
    render_approval_summary,
    render_shell_preview,
)


def test_lexer_for_path_uses_suffix() -> None:
    assert lexer_for_path("src/app.py") == "python"
    assert lexer_for_path("notes.unknown") == "text"


def test_highlight_lines_keeps_source_text() -> None:
    lines = ["def add(a, b):", "    return a + b"]
    rendered = highlight_lines("math.py", lines)
    assert [part.plain for part in rendered] == lines


def test_write_summary_clips_and_full_keeps_tail() -> None:
    body = "\n".join(f"line_{i}" for i in range(40))
    call = ToolCall(id="1", name="write_file", arguments={"path": "w.py", "content": body})
    summary = render_approval_summary(call)
    assert "line_0" in summary.plain
    assert "line_9" in summary.plain
    assert "line_39" not in summary.plain
    assert "ctrl+e" in summary.plain
    assert approval_has_more(call)
    full = render_approval_full(call)
    assert "line_39" in full.plain
    assert "ctrl+e" not in full.plain
    assert "w.py" in approval_title(call)
    assert "40 行" in approval_title(call)


def test_edit_summary_uses_clustered_diff() -> None:
    old = "\n".join(f"line_{i}" for i in range(40))
    new_lines = old.split("\n")
    new_lines[1] = "LINE_1"
    new_lines[30] = "LINE_30"
    call = ToolCall(
        id="1",
        name="edit_file",
        arguments={"path": "a.py", "old_string": old, "new_string": "\n".join(new_lines)},
    )
    summary = render_approval_summary(call)
    assert "LINE_1" in summary.plain
    assert "行未改" in summary.plain
    assert "+2" in approval_title(call)
    assert "-2" in approval_title(call)


def test_shell_card_marks_destructive_pattern() -> None:
    text = render_shell_preview("rm -rf /", cwd="E:/proj")
    assert "rm -rf /" in text.plain
    assert "cwd: E:/proj" in text.plain
    assert "风险" in text.plain
    safe = render_shell_preview("python -m unittest")
    assert "python -m unittest" in safe.plain
    assert "风险" not in safe.plain
