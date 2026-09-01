from pathlib import Path

from textual.events import Key

from anvil.tui.chrome import is_shift_enter
from anvil.tui.chrome import session_header
from anvil.tui.complete import (
    apply_mention,
    index_from_location,
    location_from_index,
    match_files,
    match_slash,
    mention_query,
    slash_query,
)


def test_skill_slash_completion_uses_discovered_metadata() -> None:
    items = match_slash(
        "/skill:rev",
        (("review-code", "Review source changes"), ("test-first", "Run tests")),
    )
    assert items is not None
    assert [(item.value, item.detail) for item in items] == [
        ("skill:review-code", "Review source changes")
    ]


def test_shift_enter_key_is_detected_without_windows_api() -> None:
    assert is_shift_enter(Key("shift+enter", None)) is True
    assert is_shift_enter(Key("enter", "\r")) is False


def test_slash_query_only_for_command_token() -> None:
    assert slash_query("/") == ""
    assert slash_query("/he") == "he"
    assert slash_query("/help extra") is None
    assert slash_query("please /help") is None
    assert slash_query("/help\n") is None


def test_match_slash_filters_and_includes_aliases() -> None:
    names = [item.value for item in match_slash("/") or []]
    assert "compact" in names
    assert "context" in names
    assert "help" in names
    assert "exit" in names
    assert "clear" in names
    assert "quit" not in names
    assert "new" not in names
    filtered = [item.label for item in match_slash("/ex") or []]
    assert "/expand" in filtered
    assert "/exit" in filtered
    aliases = [item.value for item in match_slash("/sessions") or []]
    assert aliases == ["resume"]
    assert [item.value for item in match_slash("/yolo") or []] == ["perm"]
    names = [item.value for item in match_slash("/") or []]
    assert "perm" in names
    assert "/effort" not in filtered
    assert match_slash("hello") is None


def test_slash_match_is_prefix_not_substring() -> None:
    labels = [item.label for item in match_slash("/e") or []]
    assert labels == ["/effort", "/expand", "/exit"]
    assert match_slash("/re")[0].label == "/resume"


def test_mention_query_uses_the_token_at_cursor() -> None:
    text = "see @led"
    assert mention_query(text) == (4, "led")
    assert mention_query("hello") is None
    assert mention_query("email@x.com") is None
    assert mention_query("a @b c", 4) == (2, "b")
    assert mention_query("a @b c", 6) is None


def test_apply_mention_replaces_the_at_token() -> None:
    assert apply_mention("see @led", 4, "led", "ledger.py") == "see @ledger.py "


def test_match_files_filters_by_prefix(tmp_path: Path) -> None:
    (tmp_path / "ledger.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("y\n", encoding="utf-8")
    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / "ledger_utils.py").write_text("z\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "skip.py").write_text("no\n", encoding="utf-8")
    hits = [item.value for item in match_files(tmp_path, "led")]
    assert "ledger.py" in hits
    assert "pkg/ledger_utils.py" in hits
    assert "readme.md" not in hits
    assert all(".venv" not in item for item in hits)


def test_session_header_includes_effort(tmp_path: Path) -> None:
    line = session_header(model="deepseek-v4-flash", effort="max", workspace=tmp_path / "broken_ledger")
    assert "deepseek-v4-flash" in line
    assert "max" in line
    assert "broken_ledger" in line


def test_cursor_index_roundtrip() -> None:
    text = "ab\ncd"
    assert index_from_location(text, (1, 1)) == 4
    assert location_from_index(text, 4) == (1, 1)
    assert location_from_index(text, 0) == (0, 0)
