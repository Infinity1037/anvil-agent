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


def test_read_numbers_lines(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    out = _registry(tmp_path).execute(_call("read_file", path="note.txt"))
    assert "1|alpha" in out
    assert "2|beta" in out


def test_edit_requires_unique_match(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\ny = 1\nx = 1\n", encoding="utf-8")
    registry = _registry(tmp_path)
    duplicated = registry.execute(
        _call("edit_file", path="app.py", old_string="x = 1", new_string="x = 2")
    )
    assert duplicated.startswith("Error:")
    assert "2 locations" in duplicated or "matched 2" in duplicated

    missing = registry.execute(
        _call("edit_file", path="app.py", old_string="nope", new_string="x")
    )
    assert missing.startswith("Error:")
    assert "not found" in missing

    ok = registry.execute(
        _call("edit_file", path="app.py", old_string="y = 1", new_string="y = 2")
    )
    assert ok.startswith("Edited")
    assert "y = 2" in (tmp_path / "app.py").read_text(encoding="utf-8")


def test_write_and_list(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    written = registry.execute(_call("write_file", path="pkg/hello.py", content="print(1)\n"))
    assert "hello.py" in written
    listed = registry.execute(_call("list_dir", path="pkg"))
    assert "hello.py" in listed


def test_path_escape_is_an_error(tmp_path: Path) -> None:
    out = _registry(tmp_path).execute(_call("read_file", path="../secret.txt"))
    assert out.startswith("Error:")
    assert "escapes" in out.lower()


def test_empty_json_object_is_valid_for_optional_args(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    out = _registry(tmp_path).execute(
        ToolCall(id="c1", name="list_dir", arguments={}, arguments_raw="{}")
    )
    assert not out.startswith("Error:"), out
    assert "a.txt" in out


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
    assert out.startswith("Error:")
    assert "JSON" in out
