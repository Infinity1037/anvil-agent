from pathlib import Path

from anvil.llm.types import ToolCall
from anvil.tools.base import ToolRegistry
from anvil.tools.fs import make_edit_file, make_list_dir, make_read_file, make_write_file


def _registry(workspace: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(make_list_dir(workspace))
    registry.register(make_read_file(workspace))
    registry.register(make_write_file(workspace))
    registry.register(make_edit_file(workspace))
    return registry


def _call(name: str, **arguments) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments, arguments_raw="")


def _text(result) -> str:
    return result.to_message_content()


def test_read_rejects_anvil_session_files(tmp_path: Path) -> None:
    hidden = tmp_path / ".anvil" / "sessions" / "x.jsonl"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("{}\n", encoding="utf-8")
    result = _registry(tmp_path).execute(_call("read_file", path=".anvil/sessions/x.jsonl"))
    assert result.ok is False
    assert result.error_code == "internal_path"


def test_read_numbers_lines(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    out = _text(_registry(tmp_path).execute(_call("read_file", path="note.txt")))
    assert "1|alpha" in out
    assert "2|beta" in out


def test_edit_requires_unique_match(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\ny = 1\nx = 1\n", encoding="utf-8")
    registry = _registry(tmp_path)
    duplicated = registry.execute(
        _call("edit_file", path="app.py", old_string="x = 1", new_string="x = 2")
    )
    assert duplicated.ok is False
    assert duplicated.error_code == "not_unique"
    assert _text(duplicated).startswith("Error")
    assert "2 locations" in duplicated.content or "matched 2" in duplicated.content

    missing = registry.execute(
        _call("edit_file", path="app.py", old_string="nope", new_string="x")
    )
    assert missing.ok is False
    assert missing.error_code == "no_match"
    assert "not found" in missing.content

    ok = _text(registry.execute(_call("edit_file", path="app.py", old_string="y = 1", new_string="y = 2")))
    assert "Edited" in ok
    assert "--- a/app.py" in ok
    assert "-y = 1" in ok
    assert "+y = 2" in ok
    assert "y = 2" in (tmp_path / "app.py").read_text(encoding="utf-8")


def test_write_and_list(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    written = _text(registry.execute(_call("write_file", path="pkg/hello.py", content="print(1)\n")))
    assert "hello.py" in written
    listed = _text(registry.execute(_call("list_dir", path="pkg")))
    assert "hello.py" in listed


def test_list_hides_secret_files_but_keeps_templates(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TOKEN=live\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("TOKEN=template\n", encoding="utf-8")

    listed = _registry(tmp_path).execute(_call("list_dir", path="."))

    assert listed.ok is True
    assert ".env.example" in listed.content
    assert "\nfile  .env " not in f"\n{listed.content}"


def test_write_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.execute(_call("write_file", path="a.txt", content="one\n"))
    blocked = registry.execute(_call("write_file", path="a.txt", content="two\n"))
    assert blocked.ok is False
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one\n"
    replaced = registry.execute(_call("write_file", path="a.txt", content="two\n", overwrite=True))
    assert replaced.ok is True
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "two\n"
    assert "--- a/a.txt" in replaced.content


def test_path_escape_is_an_error(tmp_path: Path) -> None:
    out = _registry(tmp_path).execute(_call("read_file", path="../secret.txt"))
    assert out.ok is False
    assert out.error_code == "path_escape"
    assert "escapes" in _text(out).lower()


def test_secret_file_has_error_code(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("KEY=1\n", encoding="utf-8")
    out = _registry(tmp_path).execute(_call("read_file", path=".env"))
    assert out.ok is False
    assert out.error_code == "secret_file"


def test_missing_arguments_have_error_code(tmp_path: Path) -> None:
    out = _registry(tmp_path).execute(_call("read_file"))
    assert out.ok is False
    assert out.error_code == "missing_arguments"
    assert "path" in out.content


def test_unknown_keys_are_stripped(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    out = _registry(tmp_path).execute(
        ToolCall(
            id="c1",
            name="list_dir",
            arguments={"path": ".", "surprise": True},
            arguments_raw="{}",
        )
    )
    assert out.ok, out.content
    assert "a.txt" in out.content


def test_empty_json_object_is_valid_for_optional_args(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    out = _registry(tmp_path).execute(
        ToolCall(id="c1", name="list_dir", arguments={}, arguments_raw="{}")
    )
    assert out.ok, out.content
    assert "a.txt" in out.content


def test_invalid_json_arguments_are_reported() -> None:
    from pathlib import Path

    registry = _registry(Path("."))
    out = registry.execute(
        ToolCall(
            id="c1",
            name="list_dir",
            arguments={},
            arguments_raw="{",
            parse_error=True,
        )
    )
    assert out.ok is False
    assert out.error_code == "invalid_json"
    assert "JSON" in _text(out)


def test_unknown_tool_has_error_code(tmp_path: Path) -> None:
    result = _registry(tmp_path).execute(_call("explode"))
    assert result.ok is False
    assert result.error_code == "unknown_tool"
