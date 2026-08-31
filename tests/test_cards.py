import difflib

from anvil.tui.cards import (
    cluster_rows,
    count_content_lines,
    edit_rows,
    format_diff_rows,
    key_argument,
    numbered_content,
    parse_unified_diff,
    render_body,
    render_card,
    span_diff,
    tool_chip,
    tool_header,
)

_BEFORE = """class Ledger:
    def record(self, account: str, dollars: float) -> None:
        self._lines.append(Line(account, dollars))
    def void(self, index: int) -> None:
        self._lines[index].voided = True
"""
_AFTER = """class Ledger:
    def record(self, account: str, dollars: float) -> None:
        cents = round(dollars * 100)
        self._lines.append(Line(account, cents))
    def void(self, index: int) -> None:
        self._lines[index].voided = True
"""
def _udiff(path: str, before: str, after: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )


DIFF = "Edited ledger.py (1 replacement).\n" + _udiff("ledger.py", _BEFORE, _AFTER)


def test_key_argument_picks_path_and_command() -> None:
    assert key_argument("edit_file", {"path": "a/b.py"}) == "a/b.py"
    assert key_argument("run_shell", {"command": "python -m unittest -v"}) == "python -m unittest -v"
    assert key_argument("grep", {"pattern": "amount"}) == "amount"


def test_parse_unified_diff_skips_git_headers() -> None:
    rows = parse_unified_diff(DIFF)
    kinds = [row.kind for row in rows]
    assert "add" in kinds and "delete" in kinds
    assert all(not row.text.startswith("---") for row in rows)
    added = [row.text for row in rows if row.kind == "add"]
    assert "        cents = round(dollars * 100)" in added
    deleted = [row.text for row in rows if row.kind == "delete"]
    assert "        self._lines.append(Line(account, dollars))" in deleted
    first_add = next(row for row in rows if row.kind == "add")
    assert first_add.line_no == 3


def test_cluster_rows_elides_unchanged_middle() -> None:
    old = "\n".join(f"line_{i}" for i in range(40))
    new_lines = old.split("\n")
    new_lines[1] = "LINE_1"
    new_lines[30] = "LINE_30"
    clustered = cluster_rows(span_diff(old, "\n".join(new_lines)), context=3)
    kinds = [row.kind for row in clustered]
    assert "gap" in kinds
    assert any(row.text == "LINE_1" for row in clustered)
    assert any(row.text == "LINE_30" for row in clustered)
    gap = next(row for row in clustered if row.kind == "gap")
    assert int(gap.text) == 22
    assert clustered[0].text == "line_0"
    assert clustered[-1].text == "line_33"


def test_edit_header_uses_used_and_plus_minus_chip() -> None:
    header = tool_header(
        "edit_file",
        {"path": "ledger.py", "old_string": "a", "new_string": "b"},
        live=False,
        ok=True,
        result=DIFF,
    ).plain
    assert header.startswith("● Used Edit")
    assert "ledger.py" in header
    assert "+2" in header
    assert "-1" in header
    assert "--- a/" not in header


def test_using_header_while_live() -> None:
    header = tool_header(
        "write_file",
        {"path": "snake.py", "content": "print(1)\n"},
        live=True,
        ok=None,
    ).plain
    assert "Using Write" in header
    assert "snake.py" in header
    assert "·" not in header


def test_write_body_is_numbered_content_not_byte_count() -> None:
    content = "class Snake:\n    pass\n"
    body = render_body(
        "write_file",
        arguments={"path": "snake.py", "content": content},
        result="Wrote 20 bytes (2 lines) to snake.py.",
        live=False,
        expanded=False,
        ok=True,
    ).plain
    assert "Wrote 20 bytes" not in body
    assert "class Snake:" in body
    assert "1" in body.splitlines()[0]


def test_write_overwrite_prefers_diff_from_result() -> None:
    result = "Wrote 10 bytes (6 lines) to ledger.py.\n" + _udiff("ledger.py", _BEFORE, _AFTER)
    body = render_body(
        "write_file",
        arguments={"path": "ledger.py", "content": "x"},
        result=result,
        live=False,
        expanded=False,
        ok=True,
    ).plain
    assert "--- a/" not in body
    assert "+++" not in body
    assert "cents = round" in body
    assert any(ch.isdigit() for ch in body.splitlines()[0][:4])


def test_collapsed_write_caps_at_ten_lines() -> None:
    content = "\n".join(f"line_{i}" for i in range(1, 25))
    collapsed = numbered_content(content, expanded=False).plain
    expanded = numbered_content(content, expanded=True).plain
    assert "line_1" in collapsed
    assert "line_10" in collapsed
    assert "line_20" not in collapsed
    assert "Ctrl+O to expand" in collapsed
    assert "24 total" in collapsed
    assert "line_20" in expanded
    assert "Ctrl+O" not in expanded


def test_read_collapsed_has_no_body() -> None:
    result = "ledger.py (86 lines)\n   1|from dataclasses import dataclass\n   2|class Ledger:"
    body = render_body(
        "read_file",
        arguments={"path": "ledger.py"},
        result=result,
        live=False,
        expanded=False,
        ok=True,
    ).plain
    assert body == ""
    chip = tool_chip("read_file", {"path": "ledger.py"}, result)
    assert chip == "86 lines"
    expanded = render_body(
        "read_file",
        arguments={"path": "ledger.py"},
        result=result,
        live=False,
        expanded=True,
        ok=True,
    ).plain
    assert "from dataclasses" in expanded


def test_grep_glance_lists_a_few_paths() -> None:
    result = "a.py:1: x\nb.py:2: y\nc.py:3: z\nd.py:4: w"
    body = render_body(
        "grep",
        arguments={"pattern": "x"},
        result=result,
        live=False,
        expanded=False,
        ok=True,
    ).plain
    assert "a.py" in body
    assert "+1 more" in body
    assert tool_chip("grep", {"pattern": "x"}, result) == "4 matches"


def test_edit_from_old_new_when_result_has_no_diff() -> None:
    rows = edit_rows(
        {"path": "x.py", "old_string": "foo = 1", "new_string": "foo = 2"},
        "Edited x.py (1 replacement).",
    )
    assert any(row.kind == "delete" and "foo = 1" in row.text for row in rows)
    assert any(row.kind == "add" and "foo = 2" in row.text for row in rows)


def test_error_body_is_red_plain_text() -> None:
    body = render_body(
        "edit_file",
        arguments={"path": "x.py", "old_string": "a", "new_string": "b"},
        result="old_string was not found in x.py",
        live=False,
        expanded=False,
        ok=False,
    )
    assert "not found" in body.plain
    assert "+ " not in body.plain


def test_finished_card_plain_has_used_and_no_git_headers() -> None:
    card = render_card(
        "edit_file",
        arguments={"path": "ledger.py"},
        result=DIFF,
        live=False,
        expanded=False,
        ok=True,
    ).plain
    assert "Used Edit" in card
    assert "ledger.py" in card
    assert "cents = round" in card
    assert "--- a/ledger.py" not in card
    assert "@@ " not in card


def test_count_content_lines() -> None:
    assert count_content_lines("") == 0
    assert count_content_lines("a") == 1
    assert count_content_lines("a\nb") == 2
    assert count_content_lines("a\nb\n") == 2


def test_format_diff_rows_hint() -> None:
    rows = parse_unified_diff(DIFF)
    long_rows = rows * 8
    collapsed = format_diff_rows(long_rows, expanded=False, limit=10).plain
    assert "Ctrl+O to expand" in collapsed
    assert collapsed.count("\n") >= 10
