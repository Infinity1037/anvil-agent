from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from anvil.safety import resolve_in_workspace
from anvil.tools.base import ToolSpec

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


def _iter_files(root: Path, glob_pattern: str | None):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        current = Path(dirpath)
        for name in filenames:
            path = current / name
            if glob_pattern and not _glob_match(glob_pattern, path, root):
                continue
            yield path


def make_glob(workspace: Path) -> ToolSpec:
    def glob_files(pattern: str, path: str = ".") -> str:
        root = resolve_in_workspace(workspace, path)
        if not root.exists():
            return f"Error: path not found: {path}"
        if root.is_file():
            root = root.parent
        matches: list[str] = []
        for file_path in _iter_files(root, None):
            rel = file_path.relative_to(workspace).as_posix()
            if not _glob_match(pattern, file_path, workspace) and not _glob_match(pattern, file_path, root):
                continue
            matches.append(rel)
            if len(matches) >= 200:
                break
        if not matches:
            return f"No files matched {pattern!r} under {path}"
        matches.sort()
        extra = "" if len(matches) < 200 else "\n... extra matches omitted"
        return "\n".join(matches) + extra

    return ToolSpec(
        name="glob",
        description="Find files by glob pattern (for example **/*.py) under a workspace path.",
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
    ) -> str:
        root = resolve_in_workspace(workspace, path)
        if not root.exists():
            return f"Error: path not found: {path}"
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex: {exc}"
        cap = max(1, min(int(max_matches), 200))
        hits: list[str] = []
        files = [root] if root.is_file() else _iter_files(root, glob)
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
                        return joined + f"\n... stopped at {cap} matches (scanned {scanned} files)"
        if not hits:
            return f"No matches for {pattern!r} under {path}"
        return "\n".join(hits)

    return ToolSpec(
        name="grep",
        description="Search file contents with a Python regular expression. Skips git/venv/node_modules.",
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
