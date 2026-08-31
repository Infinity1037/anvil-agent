from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anvil.ui.terminal import TerminalUI


def read_user_line(ui: TerminalUI) -> str:
    """Read one chat line. Ctrl+O expands the last turn's tool output."""
    if not sys.stdin.isatty():
        return ui.console.input("anvil> ").strip()
    try:
        return _prompt_toolkit_line(ui)
    except Exception:
        return ui.console.input("anvil> ").strip()


def _prompt_toolkit_line(ui: TerminalUI) -> str:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import run_in_terminal
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    bindings = KeyBindings()

    @bindings.add("c-o")
    def _expand(event) -> None:
        def _draw() -> None:
            ui.expand()

        run_in_terminal(_draw)

    @bindings.add(Keys.ControlC)
    def _ctrl_c(event) -> None:
        event.app.exit(exception=KeyboardInterrupt)

    session: PromptSession[str] = PromptSession(
        message="anvil> ",
        key_bindings=bindings,
    )
    return session.prompt().strip()
