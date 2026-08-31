from __future__ import annotations

import difflib
from pathlib import Path

from anvil.safety import assert_not_internal, assert_not_secret, resolve_in_workspace
from anvil.tools.base import ToolSpec
from anvil.tools.observe import FileObserver
from anvil.tools.result import ToolResult

MAX_READ_BYTES = 200_000
DEFAULT_LINE_LIMIT = 400


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def unified_diff(path: str, before: str, after: str, context: int = 3) -> str:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
        n=context,
    )
    return "\n".join(diff)


def _numbered(lines: list[str], start_line: int) -> str:
    width = max(4, len(str(start_line + len(lines) - 1)))
    return "\n".join(
        f"{index:>{width}}|{line}"
        for index, line in enumerate(lines, start=start_line)
    )


def make_list_dir(workspace: Path) -> ToolSpec:
    def list_dir(path: str = ".") -> ToolResult:
        target = resolve_in_workspace(workspace, path)
        assert_not_internal(target, workspace)
        if not target.exists():
            return ToolResult.fail("not_found", f"path not found: {path}")
        if not target.is_dir():
            return ToolResult.fail("not_a_directory", f"not a directory: {path}")
        skip = {".git", ".anvil", "__pycache__", ".venv", "venv", ".pytest_cache", "node_modules"}
        entries = sorted(
            (item for item in target.iterdir() if item.name not in skip),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
        if not entries:
            return ToolResult.success(f"(empty directory) {path}")
        lines = []
        for item in entries[:500]:
            kind = "dir " if item.is_dir() else "file"
            size = "" if item.is_dir() else f" {item.stat().st_size}B"
            lines.append(f"{kind}  {item.name}{size}")
        if len(entries) > 500:
            lines.append(f"... {len(entries) - 500} more entries omitted")
        return ToolResult.success("\n".join(lines))

    return ToolSpec(
        name="list_dir",
        description="List files and directories in a workspace path. Prefer this over shell ls/dir.",
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


def make_read_file(workspace: Path, observer: FileObserver | None = None) -> ToolSpec:
    def read_file(path: str, offset: int | None = None, limit: int | None = None) -> ToolResult:
        target = resolve_in_workspace(workspace, path)
        assert_not_secret(target)
        assert_not_internal(target, workspace)
        if not target.exists():
            return ToolResult.fail("not_found", f"file not found: {path}")
        if target.is_dir():
            return ToolResult.fail(
                "not_a_file",
                f"'{path}' is a directory. Use list_dir.",
            )
        size = target.stat().st_size
        if size > MAX_READ_BYTES and offset is None and limit is None:
            return ToolResult.fail(
                "file_too_large",
                f"file is {size} bytes. Use offset and limit to read a slice "
                f"(about {DEFAULT_LINE_LIMIT} lines per call).",
            )
        try:
            text = _read_text(target)
        except UnicodeDecodeError:
            return ToolResult.fail("not_utf8", f"'{path}' is not valid UTF-8 text.")
        except OSError as exc:
            return ToolResult.fail("exception", str(exc))
        if observer is not None:
            observer.remember(target, text)
        lines = text.splitlines()
        total = len(lines)
        try:
            start = 1 if offset is None else int(offset)
            count = DEFAULT_LINE_LIMIT if limit is None else int(limit)
        except (TypeError, ValueError):
            return ToolResult.fail("bad_arguments", "offset and limit must be integers.")
        if start < 1:
            return ToolResult.fail("offset_invalid", "offset is 1-based and must be >= 1.")
        if start > total + 1:
            return ToolResult.fail(
                "offset_invalid",
                f"offset {start} is past end of file ({total} lines).",
            )
        if count < 1:
            return ToolResult.fail("offset_invalid", "limit must be >= 1.")
        chunk = lines[start - 1 : start - 1 + count]
        rendered = _numbered(chunk, start)
        remaining = total - (start - 1 + len(chunk))
        header = f"{path} ({total} lines)"
        if remaining > 0:
            header += f" — showing {len(chunk)} lines, {remaining} more after this slice"
        return ToolResult.success(header + "\n" + rendered)

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


def make_write_file(workspace: Path, observer: FileObserver | None = None) -> ToolSpec:
    def write_file(path: str, content: str, overwrite: bool = False) -> ToolResult:
        target = resolve_in_workspace(workspace, path)
        assert_not_secret(target)
        assert_not_internal(target, workspace)
        if target.exists() and target.is_dir():
            return ToolResult.fail("not_a_file", f"'{path}' is a directory.")
        existed = target.exists()
        if existed and not overwrite:
            return ToolResult.fail(
                "already_exists",
                f"{path} already exists. Use edit_file for a surgical change, "
                "or pass overwrite=true to replace the entire file.",
            )
        before = ""
        if existed:
            try:
                before = _read_text(target)
            except UnicodeDecodeError:
                return ToolResult.fail("not_utf8", f"'{path}' is not valid UTF-8 text.")
            except OSError as exc:
                return ToolResult.fail("write_failed", str(exc))
            if observer is not None:
                blocked = observer.check_against(target, before)
                if blocked is not None:
                    return blocked
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_text(target, content)
        except OSError as exc:
            return ToolResult.fail("write_failed", str(exc))
        if observer is not None:
            observer.remember(target, content)
        lines = content.count("\n") + (0 if content.endswith("\n") or content == "" else 1)
        header = (
            f"Wrote {len(content.encode('utf-8'))} bytes ({lines} lines) to {path}"
            + (" (overwrote existing file)." if existed else ".")
        )
        if existed:
            diff = unified_diff(path, before, content)
            if diff:
                return ToolResult.success(header + "\n" + diff)
        return ToolResult.success(header)

    return ToolSpec(
        name="write_file",
        description=(
            "Create a UTF-8 text file. Refuses to overwrite unless overwrite=true. "
            "Prefer edit_file for existing files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace."},
                "content": {"type": "string", "description": "Full file contents to write."},
                "overwrite": {
                    "type": "boolean",
                    "description": "Replace the file if it already exists. Default false.",
                },
            },
            "required": ["path", "content"],
        },
        handler=write_file,
        parallel_safe=False,
    )


def make_edit_file(workspace: Path, observer: FileObserver | None = None) -> ToolSpec:
    def edit_file(path: str, old_string: str, new_string: str) -> ToolResult:
        if old_string == new_string:
            return ToolResult.fail(
                "identical_edit",
                "old_string and new_string are identical.",
            )
        if old_string == "":
            return ToolResult.fail(
                "empty_old_string",
                "old_string must not be empty. Use write_file to create a new file.",
            )
        target = resolve_in_workspace(workspace, path)
        assert_not_secret(target)
        assert_not_internal(target, workspace)
        if not target.exists():
            return ToolResult.fail(
                "not_found",
                f"file not found: {path}. Use write_file to create it.",
            )
        if target.is_dir():
            return ToolResult.fail("not_a_file", f"'{path}' is a directory.")
        try:
            text = _read_text(target)
        except UnicodeDecodeError:
            return ToolResult.fail("not_utf8", f"'{path}' is not valid UTF-8 text.")
        except OSError as exc:
            return ToolResult.fail("exception", str(exc))
        if observer is not None:
            blocked = observer.check_against(target, text)
            if blocked is not None:
                return blocked
        matches = text.count(old_string)
        if matches == 0:
            return ToolResult.fail(
                "no_match",
                f"old_string was not found in {path} ({len(text.splitlines())} lines). "
                "Read the file and copy the exact text, including whitespace.",
                hint="Call read_file and copy the exact span to replace.",
            )
        if matches > 1:
            return ToolResult.fail(
                "not_unique",
                f"old_string matched {matches} locations in {path}. "
                "Include more surrounding lines so the replacement is unique.",
                hint="Add surrounding lines so the match occurs exactly once.",
            )
        updated = text.replace(old_string, new_string, 1)
        try:
            _write_text(target, updated)
        except OSError as exc:
            return ToolResult.fail("write_failed", str(exc))
        if observer is not None:
            observer.remember(target, updated)
        diff = unified_diff(path, text, updated)
        return ToolResult.success(f"Edited {path} (1 replacement).\n{diff}")

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
