from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from anvil.safety import (
    SecretFileError,
    assert_not_internal,
    assert_not_secret,
    resolve_in_workspace,
)
from anvil.tools.base import ToolSpec
from anvil.tools.result import ToolResult


def _glob_match(pattern: str, path: Path, root: Path) -> bool:
    pat = pattern.replace("\\", "/")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.as_posix()
    simplified = pat.replace("**/", "").replace("**", "*")
    return any(
        fnmatch.fnmatch(candidate, pat) or fnmatch.fnmatch(candidate, simplified)
        for candidate in (relative, path.name, path.as_posix().replace("\\", "/"))
    )


SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".anvil",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
}


def _searchable_file(path: Path, workspace: Path) -> bool:
    try:
        resolved = path.resolve()
        resolved.relative_to(workspace.resolve())
        assert_not_secret(path)
        assert_not_secret(resolved)
    except (OSError, SecretFileError, ValueError):
        return False
    return True


def _iter_files(root: Path, glob_pattern: str | None, workspace: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in SKIP_DIRS and not (current / name).is_symlink()
        ]
        for name in filenames:
            path = current / name
            if not _searchable_file(path, workspace):
                continue
            if glob_pattern and not _glob_match(glob_pattern, path, root):
                continue
            yield path


def make_glob(workspace: Path) -> ToolSpec:
    def glob_files(pattern: str, path: str = ".") -> ToolResult:
        if not str(pattern).strip():
            return ToolResult.fail("empty_query", "pattern must not be empty.")
        root = resolve_in_workspace(workspace, path)
        assert_not_internal(root, workspace)
        if not root.exists():
            return ToolResult.fail("not_found", f"path not found: {path}")
        if root.is_file():
            assert_not_secret(root)
            root = root.parent
        matches: list[str] = []
        for file_path in _iter_files(root, None, workspace):
            rel = file_path.relative_to(workspace).as_posix()
            if not _glob_match(pattern, file_path, workspace) and not _glob_match(pattern, file_path, root):
                continue
            matches.append(rel)
            if len(matches) >= 200:
                break
        if not matches:
            return ToolResult.success(f"No files matched {pattern!r} under {path}")
        matches.sort()
        extra = "" if len(matches) < 200 else "\n... extra matches omitted"
        return ToolResult.success("\n".join(matches) + extra)

    return ToolSpec(
        name="glob",
        description="Find files by glob pattern (for example **/*.py). Prefer this over find/ls in the shell.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern. ** is supported via recursive walk + fnmatch.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search. Defaults to the workspace root.",
                },
            },
            "required": ["pattern"],
        },
        handler=glob_files,
        parallel_safe=True,
    )


def make_grep(workspace: Path) -> ToolSpec:
    def grep(
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        max_matches: int = 50,
    ) -> ToolResult:
        if not str(pattern):
            return ToolResult.fail("empty_query", "pattern must not be empty.")
        root = resolve_in_workspace(workspace, path)
        assert_not_internal(root, workspace)
        if not root.exists():
            return ToolResult.fail("not_found", f"path not found: {path}")
        if root.is_file():
            assert_not_secret(root)
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult.fail("invalid_regex", f"invalid regex: {exc}")
        try:
            cap = max(1, min(int(max_matches), 200))
        except (TypeError, ValueError):
            return ToolResult.fail("bad_arguments", "max_matches must be an integer.")
        hits: list[str] = []
        files = [root] if root.is_file() else _iter_files(root, glob, workspace)
        scanned = 0
        for file_path in files:
            scanned += 1
            try:
                if file_path.stat().st_size > 1_000_000:
                    continue
                text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            rel = file_path.relative_to(workspace).as_posix()
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{rel}:{line_no}:{line[:240]}")
                    if len(hits) >= cap:
                        joined = "\n".join(hits)
                        return ToolResult.success(
                            joined + f"\n... stopped at {cap} matches (scanned {scanned} files)"
                        )
        if not hits:
            return ToolResult.success(f"No matches for {pattern!r} under {path}")
        return ToolResult.success("\n".join(hits))

    return ToolSpec(
        name="grep",
        description=(
            "Search non-secret file contents with a Python regular expression. "
            "Prefer this over shell grep. Skips git/venv/node_modules."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {
                    "type": "string",
                    "description": "File or directory to search. Defaults to the workspace root.",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional filename glob such as *.py.",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum matches to return (default 50, max 200).",
                },
            },
            "required": ["pattern"],
        },
        handler=grep,
        parallel_safe=True,
    )
