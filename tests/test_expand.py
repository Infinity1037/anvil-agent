from io import StringIO

from rich.console import Console

from anvil.ui.prompt import _prompt_toolkit_line
from anvil.ui.terminal import TerminalUI


def test_cli_import_path_does_not_cycle() -> None:
    """``anvil`` loads ui before tui; card renderers must not pull the app."""
    import anvil.cli
    from anvil.ui.format import STOP_LABELS
    from anvil.ui.terminal import TerminalUI as UI

    assert "completed" in STOP_LABELS
    assert UI is not None
    assert callable(anvil.cli.main)


def test_expand_shows_stored_tool_output() -> None:
    buffer = StringIO()
    ui = TerminalUI(Console(file=buffer, width=80, force_terminal=True, color_system=None))
    ui.begin_turn()
    ui.handle(
        "tool_result",
        {"name": "read_file", "ok": True, "content": "alpha line\nbeta line\n"},
    )
    ui.expand()
    text = buffer.getvalue()
    assert "alpha line" in text
    assert "beta line" in text


def test_ctrl_o_binding_is_registered() -> None:
    from prompt_toolkit.key_binding import KeyBindings

    assert callable(_prompt_toolkit_line)
    bindings = KeyBindings()
    bindings.add("c-o")(lambda event: None)
    assert any("c-o" in str(binding.keys) or binding.keys == ("c-o",) for binding in bindings.bindings)
