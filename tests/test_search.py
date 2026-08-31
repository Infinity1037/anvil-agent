from pathlib import Path

from anvil.llm.types import ToolCall
from anvil.tools.base import ToolRegistry
from anvil.tools.search import make_glob, make_grep


def test_glob_and_grep(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "util.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "src" / "skip.txt").write_text("add is mentioned here\n", encoding="utf-8")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "util.py").write_text("def add(a, b): pass\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(make_glob(tmp_path))
    registry.register(make_grep(tmp_path))

    found = registry.execute(
        ToolCall(id="1", name="glob", arguments={"pattern": "*.py"}, arguments_raw="")
    ).content
    assert "src/util.py" in found.replace("\\", "/")
    assert ".venv" not in found

    hits = registry.execute(
        ToolCall(
            id="2",
            name="grep",
            arguments={"pattern": r"def add", "glob": "*.py"},
            arguments_raw="",
        )
    ).content
    assert "src/util.py:1:" in hits.replace("\\", "/")
    assert ".venv" not in hits


def test_grep_invalid_regex_has_error_code(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(make_grep(tmp_path))
    out = registry.execute(
        ToolCall(id="1", name="grep", arguments={"pattern": "["}, arguments_raw="")
    )
    assert out.ok is False
    assert out.error_code == "invalid_regex"


def test_glob_empty_pattern_has_error_code(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(make_glob(tmp_path))
    out = registry.execute(
        ToolCall(id="1", name="glob", arguments={"pattern": "  "}, arguments_raw="")
    )
    assert out.ok is False
    assert out.error_code == "empty_query"
