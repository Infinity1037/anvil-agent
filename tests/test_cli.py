import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from rich.console import Console

from anvil.agent.context import ContextSnapshot
from anvil.agent.loop import CompactResult
from anvil.cli import _repl
from anvil.llm.types import Usage


def test_utf8_output_survives_a_gbk_parent_encoding() -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "gbk"
    code = (
        "from anvil.cli import _ensure_utf8_output; "
        "_ensure_utf8_output(); "
        "from rich.console import Console; "
        "Console().print('✓ 中文')"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert "✓ 中文" in result.stdout.decode("utf-8")


def test_plain_repl_supports_context_and_manual_compaction(monkeypatch, tmp_path: Path) -> None:
    output = StringIO()
    ui = SimpleNamespace(console=Console(file=output, force_terminal=False, width=120))
    commands = iter(["/context", "/compact keep test outcomes", "/exit"])
    monkeypatch.setattr("anvil.cli.read_user_line", lambda _ui: next(commands))

    class FakeAgent:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model="scripted", workspace=tmp_path)
            self.messages = [object()] * 12
            self.usage = Usage(prompt_tokens=12_000, completion_tokens=800)
            self.session = SimpleNamespace(
                cancel=Event(),
                effort_status=lambda: "effort max",
            )
            self.focus = ""

        def context_snapshot(self) -> ContextSnapshot:
            return ContextSnapshot(4_000, 20_000, 12, 5, 7, True)

        def compact(self, instruction="", cancel=None) -> CompactResult:
            self.focus = instruction
            return CompactResult("compacted", 4_000, 1_000, 7)

    agent = FakeAgent()
    assert _repl(agent, ui) == 0
    rendered = output.getvalue()
    assert "context ≈20%" in rendered
    assert "上下文已压缩" in rendered
    assert agent.focus == "keep test outcomes"
