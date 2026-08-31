from __future__ import annotations

from pathlib import Path

from anvil.tools.base import ToolRegistry, ToolSpec
from anvil.tools.fs import make_edit_file, make_list_dir, make_read_file, make_write_file
from anvil.tools.observe import FileObserver
from anvil.tools.result import ToolResult
from anvil.tools.search import make_glob, make_grep
from anvil.tools.shell import make_run_shell
from anvil.tools.todo import TodoStore, make_todo


def build_tools(
    workspace: Path,
    todo: TodoStore,
    shell_timeout: float,
    observer: FileObserver | None = None,
) -> ToolRegistry:
    observer = observer or FileObserver()
    registry = ToolRegistry(observer=observer)
    for factory in (
        make_list_dir(workspace),
        make_read_file(workspace, observer),
        make_write_file(workspace, observer),
        make_edit_file(workspace, observer),
        make_glob(workspace),
        make_grep(workspace),
        make_run_shell(workspace, timeout=shell_timeout, registry=registry),
        make_todo(todo),
    ):
        registry.register(factory)
    return registry


__all__ = [
    "FileObserver",
    "ToolRegistry",
    "ToolSpec",
    "ToolResult",
    "TodoStore",
    "build_tools",
]
