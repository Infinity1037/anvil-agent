from pathlib import Path

import pytest

from anvil.safety import (
    DangerousCommandError,
    PathEscapeError,
    SecretFileError,
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


def test_secret_files_are_blocked(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("KEY=1\n", encoding="utf-8")
    with pytest.raises(SecretFileError):
        assert_not_secret(env)


def test_dangerous_shell_commands_are_blocked() -> None:
    with pytest.raises(DangerousCommandError):
        assert_safe_command("rm -rf /")
    with pytest.raises(DangerousCommandError):
        assert_safe_command("shutdown /s")
