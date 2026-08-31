from pathlib import Path

import pytest

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


def test_search_skips_secrets_but_keeps_env_templates(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TOKEN=live\n", encoding="utf-8")
    (tmp_path / ".env.production").write_text("TOKEN=prod\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("TOKEN=template\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(make_glob(tmp_path))
    registry.register(make_grep(tmp_path))

    found = registry.execute(
        ToolCall(id="1", name="glob", arguments={"pattern": "*"}, arguments_raw="")
    )
    hits = registry.execute(
        ToolCall(id="2", name="grep", arguments={"pattern": "TOKEN="}, arguments_raw="")
    )

    assert found.ok is True
    assert ".env.example" in found.content
    assert "\n.env\n" not in f"\n{found.content}\n"
    assert ".env.production" not in found.content
    assert hits.ok is True
    assert ".env.example:1:TOKEN=template" in hits.content
    assert "TOKEN=live" not in hits.content
    assert "TOKEN=prod" not in hits.content


def test_grep_rejects_a_direct_secret_path(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TOKEN=live\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(make_grep(tmp_path))

    out = registry.execute(
        ToolCall(
            id="1",
            name="grep",
            arguments={"pattern": "TOKEN", "path": ".env"},
            arguments_raw="",
        )
    )

    assert out.ok is False
    assert out.error_code == "secret_file"
    assert "TOKEN=live" not in out.content


def test_grep_does_not_follow_a_file_symlink_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE_SECRET\n", encoding="utf-8")
    link = workspace / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable on this system")
    registry = ToolRegistry()
    registry.register(make_grep(workspace))

    out = registry.execute(
        ToolCall(id="1", name="grep", arguments={"pattern": "OUTSIDE_SECRET"}, arguments_raw="")
    )

    assert out.ok is True
    assert "OUTSIDE_SECRET" not in out.content
