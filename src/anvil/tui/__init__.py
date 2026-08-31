"""Fullscreen TUI. Import submodules directly (``anvil.tui.app``) to avoid
pulling Textual when only card renderers are needed.
"""

from __future__ import annotations

__all__ = ["run_tui"]


def __getattr__(name: str):
    if name == "run_tui":
        from anvil.tui.app import run_tui

        return run_tui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
