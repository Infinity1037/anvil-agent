from __future__ import annotations

import re
from pathlib import Path

ENV_TEMPLATE_NAMES = {".env.example", ".env.sample", ".env.template"}
SECRET_NAMES = {
    ".env",
    ".envrc",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
}
SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
_COMMAND_TOKEN_SPLIT = re.compile(r"[\s\"'`;|&()<>{}\[\]=,:]+")

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


def is_secret_path(path: Path) -> bool:
    name = path.name.lower()
    if name in ENV_TEMPLATE_NAMES:
        return False
    return (
        name in SECRET_NAMES
        or name.startswith(".env.")
        or path.suffix.lower() in SECRET_SUFFIXES
    )


def assert_not_secret(path: Path) -> None:
    if is_secret_path(path):
        raise SecretFileError(f"Refusing to read or write secret file: {path.name}")


def assert_command_has_no_secret_reference(command: str) -> None:
    """Reject obvious secret filenames without pretending to sandbox a shell."""
    normalized = command.replace("\\", "/")
    for token in _COMMAND_TOKEN_SPLIT.split(normalized):
        name = token.rsplit("/", 1)[-1].strip()
        if name and is_secret_path(Path(name)):
            raise SecretFileError(
                "Refusing to run a command that references a secret file."
            )


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
