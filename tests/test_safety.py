from pathlib import Path

import pytest

from anvil.safety import (
    DangerousCommandError,
    InternalPathError,
    PathEscapeError,
    SecretFileError,
    assert_command_has_no_secret_reference,
    assert_not_internal,
    assert_not_secret,
    assert_safe_command,
    resolve_in_workspace,
)


def test_resolve_keeps_paths_inside_workspace(tmp_path: Path) -> None:
    inner = tmp_path / "src" / "app.py"
    inner.parent.mkdir()
    inner.write_text("x\n", encoding="utf-8")
    resolved = resolve_in_workspace(tmp_path, "src/app.py")
    assert resolved == inner.resolve()


def test_resolve_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        resolve_in_workspace(tmp_path, "../outside.txt")


def test_anvil_session_dir_is_blocked(tmp_path: Path) -> None:
    hidden = tmp_path / ".anvil" / "sessions" / "x.jsonl"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("{}\n", encoding="utf-8")
    with pytest.raises(InternalPathError):
        assert_not_internal(hidden, tmp_path)
    assert_not_internal(tmp_path / "snake.py", tmp_path)


def test_secret_files_are_blocked(tmp_path: Path) -> None:
    for name in (".env", ".env.staging", ".envrc", "private.pem", "signing.key"):
        with pytest.raises(SecretFileError):
            assert_not_secret(tmp_path / name)
    assert_not_secret(tmp_path / ".env.example")


def test_secret_references_in_shell_commands_are_blocked() -> None:
    with pytest.raises(SecretFileError):
        assert_command_has_no_secret_reference("Get-Content ./config/.env.local")
    with pytest.raises(SecretFileError):
        assert_command_has_no_secret_reference("cat keys/private.pem")
    assert_command_has_no_secret_reference("copy .env.example .env.template")


def test_dangerous_shell_commands_are_blocked() -> None:
    with pytest.raises(DangerousCommandError):
        assert_safe_command("rm -rf /")
    with pytest.raises(DangerousCommandError):
        assert_safe_command("shutdown /s")


def test_dangerous_command_is_a_tool_result(tmp_path: Path) -> None:
    from anvil.llm.types import ToolCall
    from anvil.tools.base import ToolRegistry
    from anvil.tools.shell import make_run_shell

    registry = ToolRegistry()
    registry.register(make_run_shell(tmp_path, timeout=5))
    out = registry.execute(
        ToolCall(id="1", name="run_shell", arguments={"command": "rm -rf /"}, arguments_raw="")
    )
    assert out.ok is False
    assert out.error_code == "dangerous_command"


def test_empty_shell_command_is_a_tool_result(tmp_path: Path) -> None:
    from anvil.llm.types import ToolCall
    from anvil.tools.base import ToolRegistry
    from anvil.tools.shell import make_run_shell

    registry = ToolRegistry()
    registry.register(make_run_shell(tmp_path, timeout=5))
    out = registry.execute(
        ToolCall(id="1", name="run_shell", arguments={"command": "  "}, arguments_raw="")
    )
    assert out.ok is False
    assert out.error_code == "empty_query"
