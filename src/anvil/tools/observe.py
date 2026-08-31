from __future__ import annotations

import hashlib
from pathlib import Path

from anvil.tools.result import ToolResult


def digest_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class FileObserver:
    """Session-local digest of files the model has actually read.

    edit_file and overwrite write_file require a matching digest so a write
    cannot proceed against a file the model has not observed, or one that
    changed after the last read.
    """

    def __init__(self) -> None:
        self._digests: dict[str, str] = {}

    def clear(self) -> None:
        self._digests.clear()

    def remember(self, path: Path, content: str) -> None:
        self._digests[_key(path)] = digest_text(content)

    def check_against(self, path: Path, content: str) -> ToolResult | None:
        remembered = self._digests.get(_key(path))
        current = digest_text(content)
        name = path.name
        if remembered is None:
            return ToolResult.fail(
                "stale_read",
                f"{name} has not been read in this session.",
                hint="Call read_file on this path, then retry the edit.",
            )
        if remembered != current:
            return ToolResult.fail(
                "stale_read",
                f"{name} changed since it was last read.",
                hint="Call read_file again before editing.",
            )
        return None


def _key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)
