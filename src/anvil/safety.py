from __future__ import annotations

import re
from pathlib import Path

SECRET_NAMES = {".env", ".env.local", ".env.production", ".env.development"}
SECRET_SUFFIXES = {".pem", ".p12", ".pfx"}

_DANGEROUS = [
    re.compile(r"rm\s+-rf\s+[/~]", re.I),
    re.compile(r"rm\s+-rf\s+\*", re.I),
    re.compile(r"mkfs(\.\w+)?\b", re.I),
    re.compile(r"\bformat\s+[a-z]:", re.I),
    re.compile(r"\bshutdown\b", re.I),
    re.compile(r"\breboot\b", re.I),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;"),
    re.compile(r"Remove-Item\s+.*(-Recurse|-r).*[A-Za-z]:\\", re.I),
    re.compile(r"del\s+/s\s+/q\s+[A-Za-z]:\\", re.I),
]


class PathEscapeError(ValueError):
    pass


class SecretFileError(ValueError):
    pass


class InternalPathError(ValueError):
    pass


class DangerousCommandError(ValueError):
    pass


def resolve_in_workspace(workspace: Path, raw: str | None) -> Path:
    """Resolve a user-supplied path and reject anything outside the workspace."""
    ws = workspace.resolve()
    target = ws if not raw or raw in {".", "./"} else Path(raw)
    if not target.is_absolute():
        target = ws / target
    target = target.resolve()
    try:
        target.relative_to(ws)
    except ValueError as exc:
        raise PathEscapeError(f"Path escapes workspace: {raw!r}") from exc
    return target


def assert_not_secret(path: Path) -> None:
    name = path.name.lower()
    if name in SECRET_NAMES or path.suffix.lower() in SECRET_SUFFIXES:
        raise SecretFileError(f"Refusing to read or write secret file: {path.name}")


def assert_not_internal(path: Path, workspace: Path) -> None:
    try:
        relative = path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return
    if relative.parts and relative.parts[0] == ".anvil":
        raise InternalPathError(".anvil holds session logs, not project source.")


def assert_safe_command(command: str) -> None:
    for pattern in _DANGEROUS:
        if pattern.search(command):
            raise DangerousCommandError(
                f"Refusing to run dangerous command: {command!r}"
            )
