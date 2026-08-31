from pathlib import Path

from anvil.llm.types import ToolCall
from anvil.tools.base import ToolRegistry
from anvil.tools.fs import make_edit_file, make_read_file, make_write_file
from anvil.tools.observe import FileObserver


def _registry(workspace: Path, observer: FileObserver) -> ToolRegistry:
    registry = ToolRegistry(observer=observer)
    registry.register(make_read_file(workspace, observer))
    registry.register(make_write_file(workspace, observer))
    registry.register(make_edit_file(workspace, observer))
    return registry


def _call(name: str, **arguments) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments, arguments_raw="")


def test_edit_without_read_is_stale(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    observer = FileObserver()
    registry = _registry(tmp_path, observer)
    result = registry.execute(
        _call("edit_file", path="app.py", old_string="value = 1", new_string="value = 2")
    )
    assert result.ok is False
    assert result.error_code == "stale_read"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_read_then_edit_is_fresh(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    observer = FileObserver()
    registry = _registry(tmp_path, observer)
    assert registry.execute(_call("read_file", path="app.py")).ok
    result = registry.execute(
        _call("edit_file", path="app.py", old_string="value = 1", new_string="value = 2")
    )
    assert result.ok, result.content
    assert "Edited" in result.content
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "value = 2\n"


def test_external_change_after_read_is_stale(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    observer = FileObserver()
    registry = _registry(tmp_path, observer)
    registry.execute(_call("read_file", path="app.py"))
    target.write_text("value = 9\n", encoding="utf-8")
    result = registry.execute(
        _call("edit_file", path="app.py", old_string="value = 9", new_string="value = 2")
    )
    assert result.ok is False
    assert result.error_code == "stale_read"
    assert "changed" in result.content
    assert target.read_text(encoding="utf-8") == "value = 9\n"


def test_overwrite_without_read_is_stale(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    observer = FileObserver()
    registry = _registry(tmp_path, observer)
    result = registry.execute(_call("write_file", path="a.txt", content="two\n", overwrite=True))
    assert result.ok is False
    assert result.error_code == "stale_read"
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one\n"


def test_new_file_write_does_not_need_a_read(tmp_path: Path) -> None:
    observer = FileObserver()
    registry = _registry(tmp_path, observer)
    result = registry.execute(_call("write_file", path="new.txt", content="hi\n"))
    assert result.ok, result.content
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hi\n"


def test_observer_clear_forgets_reads(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    observer = FileObserver()
    registry = _registry(tmp_path, observer)
    registry.execute(_call("read_file", path="app.py"))
    observer.clear()
    result = registry.execute(
        _call("edit_file", path="app.py", old_string="value = 1", new_string="value = 2")
    )
    assert result.error_code == "stale_read"
