from __future__ import annotations

from pathlib import Path

from anvil.safety import (
    PathEscapeError,
    SecretFileError,
    assert_not_secret,
    resolve_in_workspace,
)
from anvil.tools.base import ToolSpec

MAX_READ_BYTES = 200_000
DEFAULT_LINE_LIMIT = 400


def _numbered(lines: list[str], start_line: int) -> str:
    width = max(4, len(str(start_line + len(lines) - 1)))
    return "\n".join(
        f"{index:>{width}}|{line}"
        for index, line in enumerate(lines, start=start_line)
    )


def make_list_dir(workspace: Path) -> ToolSpec:
    def list_dir(path: str = ".") -> str:
        target = resolve_in_workspace(workspace, path)
        if not target.exists():
            return f"Error: path not found: {path}"
        if not target.is_dir():
            return f"Error: not a directory: {path}"
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not entries:
            return f"(empty directory) {path}"
        lines = []
        for item in entries[:500]:
            kind = "dir " if item.is_dir() else "file"
            size = "" if item.is_dir() else f" {item.stat().st_size}B"
            lines.append(f"{kind}  {item.name}{size}")
        if len(entries) > 500:
            lines.append(f"... {len(entries) - 500} more entries omitted")
        return "\n".join(lines)

    return ToolSpec(
        name="list_dir",
        description="List files and directories in a workspace path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory relative to the workspace. Defaults to the workspace root.",
                }
            },
            "required": [],
        },
        handler=list_dir,
        parallel_safe=True,
    )


def make_read_file(workspace: Path) -> ToolSpec:
    def read_file(path: str, offset: int | None = None, limit: int | None = None) -> str:
        target = resolve_in_workspace(workspace, path)
        assert_not_secret(target)
        if not target.exists():
            return f"Error: file not found: {path}"
        if target.is_dir():
            return f"Error: '{path}' is a directory. Use list_dir."
        size = target.stat().st_size
        if size > MAX_READ_BYTES and offset is None and limit is None:
            return (
                f"Error: file is {size} bytes. Use offset and limit to read a slice "
                f"(about {DEFAULT_LINE_LIMIT} lines per call)."
            )
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: '{path}' is not valid UTF-8 text."
        lines = text.splitlines()
        total = len(lines)
        start = 1 if offset is None else int(offset)
        if start < 1:
            return "Error: offset is 1-based and must be >= 1."
        if start > total + 1:
            return f"Error: offset {start} is past end of file ({total} lines)."
        count = DEFAULT_LINE_LIMIT if limit is None else int(limit)
        if count < 1:
            return "Error: limit must be >= 1."
        chunk = lines[start - 1 : start - 1 + count]
        rendered = _numbered(chunk, start)
        remaining = total - (start - 1 + len(chunk))
        header = f"{path} ({total} lines)"
        if remaining > 0:
            header += f" — showing {len(chunk)} lines, {remaining} more after this slice"
        return header + "\n" + rendered

    return ToolSpec(
        name="read_file",
        description=(
            "Read a UTF-8 text file with 1-based line numbers. "
            "Use offset and limit for large files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace."},
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return.",
                },
            },
            "required": ["path"],
        },
        handler=read_file,
        parallel_safe=True,
    )


def make_write_file(workspace: Path) -> ToolSpec:
    def write_file(path: str, content: str) -> str:
        target = resolve_in_workspace(workspace, path)
        assert_not_secret(target)
        if target.exists() and target.is_dir():
            return f"Error: '{path}' is a directory."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        lines = content.count("\n") + (0 if content.endswith("\n") or content == "" else 1)
        return f"Wrote {len(content.encode('utf-8'))} bytes ({lines} lines) to {path}"

    return ToolSpec(
        name="write_file",
        description=(
            "Create or overwrite a UTF-8 text file. Prefer edit_file for existing files "
            "when a small replacement is enough."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace."},
                "content": {"type": "string", "description": "Full file contents to write."},
            },
            "required": ["path", "content"],
        },
        handler=write_file,
        parallel_safe=False,
    )


def make_edit_file(workspace: Path) -> ToolSpec:
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        if old_string == new_string:
            return "Error: old_string and new_string are identical."
        if old_string == "":
            return "Error: old_string must not be empty. Use write_file to create a new file."
        target = resolve_in_workspace(workspace, path)
        assert_not_secret(target)
        if not target.exists():
            return f"Error: file not found: {path}. Use write_file to create it."
        if target.is_dir():
            return f"Error: '{path}' is a directory."
        text = target.read_text(encoding="utf-8")
        matches = text.count(old_string)
        if matches == 0:
            return (
                f"Error: old_string was not found in {path} "
                f"({len(text.splitlines())} lines). "
                "Read the file and copy the exact text, including whitespace."
            )
        if matches > 1:
            return (
                f"Error: old_string matched {matches} locations in {path}. "
                "Include more surrounding lines so the replacement is unique."
            )
        target.write_text(text.replace(old_string, new_string, 1), encoding="utf-8", newline="\n")
        return f"Edited {path} (1 replacement)."

    return ToolSpec(
        name="edit_file",
        description=(
            "Replace exactly one occurrence of old_string with new_string in an existing file. "
            "The match must be unique; add surrounding context if it is not."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace."},
                "old_string": {
                    "type": "string",
                    "description": "Exact text to find. Must occur exactly once.",
                },
                "new_string": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_string", "new_string"],
        },
        handler=edit_file,
        parallel_safe=False,
    )


def explain_fs_error(exc: Exception) -> str:
    if isinstance(exc, PathEscapeError):
        return f"Error: {exc}"
    if isinstance(exc, SecretFileError):
        return f"Error: {exc}"
    return f"Error: {type(exc).__name__}: {exc}"
