from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from anvil.ui.format import STOP_LABELS, strip_internal
from anvil.tui.cards import key_argument, render_body, render_card
from anvil.tui.fold import TOOL_LABELS

THINKING_PREVIEW_LINES = 2


class TerminalUI:
    def __init__(self, console: Console | None = None, verbose: bool = False) -> None:
        self.console = console or Console()
        self.verbose = verbose
        self._printed_reasoning = False
        self._printed_content = False
        self._thinking = ""
        self._thinking_done = False
        self._outputs: list[dict] = []
        self._call_args: dict[str, dict] = {}

    def banner(self, workspace: str, model: str, thinking: bool) -> None:
        cwd = _short_path(workspace)
        self.console.print(f"[bold]Anvil[/bold]  [dim]{model}  {cwd}[/dim]")

    def begin_turn(self) -> None:
        self._outputs = []
        self._call_args = {}
        self._thinking = ""
        self._thinking_done = False
        self._printed_reasoning = False
        self._printed_content = False

    def expand(self) -> None:
        """Show the last turn's tool outputs in full."""
        if not self._outputs:
            self.console.print("[dim]Nothing to expand.[/dim]")
            return
        for item in self._outputs:
            self._print_full(item)

    def handle(self, kind: str, payload: dict) -> None:
        if kind == "turn":
            self._printed_reasoning = False
            self._printed_content = False
            if self.verbose:
                self.console.print(
                    Rule(f"step {payload['turn']}/{payload['max_turns']}", style="dim")
                )
        elif kind == "compact":
            if self.verbose:
                self.console.print("[dim]compacted context[/dim]")
        elif kind == "skill":
            self.console.print(f"[dim]activated project skill {payload.get('name')}[/dim]")
        elif kind == "delta":
            text = payload.get("text") or ""
            if payload.get("kind") == "reasoning":
                self._thinking += text
                if not self._printed_reasoning:
                    self.console.print("[dim italic]thinking…[/dim italic]")
                    self._printed_reasoning = True
                if self.verbose:
                    self.console.print(Text(text, style="dim italic"), end="")
            elif payload.get("kind") == "content":
                if self._printed_reasoning and not self._printed_content:
                    self._finalize_thinking()
                    self.console.print()
                self._printed_content = True
                self.console.print(text, end="")
        elif kind == "assistant":
            reasoning = payload.get("reasoning") or self._thinking
            if reasoning:
                self._thinking = reasoning
                if not self._printed_content:
                    self._finalize_thinking()
            if self._printed_reasoning or self._printed_content:
                self.console.print()
            if not self._printed_content and payload.get("content"):
                self.console.print(payload["content"])
                self._printed_content = True
            for call in payload.get("tool_calls") or []:
                name = call.get("name") or "tool"
                args = call.get("arguments") or {}
                cid = str(call.get("id") or "")
                if cid:
                    self._call_args[cid] = args
                self.console.print(
                    render_card(name, arguments=args, live=True, ok=None)
                )
            usage = payload.get("usage") or {}
            if self.verbose and usage:
                self.console.print(
                    f"[dim]tokens  in={usage.get('prompt_tokens', 0)}  "
                    f"out={usage.get('completion_tokens', 0)}[/dim]"
                )
        elif kind == "tool_result":
            self._render_tool_result(payload)
        elif kind == "error":
            self.console.print(f"[bold red]{payload.get('message')}[/bold red]")

    def _finalize_thinking(self) -> None:
        if self._thinking_done:
            return
        text = (self._thinking or "").strip()
        self._thinking_done = True
        if not text:
            return
        self._outputs.append({"name": "thinking", "content": text, "ok": True})
        if self.verbose and self._printed_reasoning:
            self.console.print()
            return
        lines = text.splitlines()
        if len(lines) <= THINKING_PREVIEW_LINES:
            self.console.print(Text(text, style="dim italic"))
            return
        preview = "\n".join(lines[:THINKING_PREVIEW_LINES])
        rest = len(lines) - THINKING_PREVIEW_LINES
        self.console.print(Text(preview, style="dim italic"))
        self.console.print(f"[dim]… ({rest} more lines, Ctrl+O to expand)[/dim]")

    def _render_tool_result(self, payload: dict) -> None:
        name = payload.get("name") or "tool"
        content = strip_internal(payload.get("content") or "")
        ok = bool(payload.get("ok"))
        cid = str(payload.get("id") or "")
        args = payload.get("arguments") or self._call_args.get(cid) or {}
        record = {"name": name, "content": content, "ok": ok, "arguments": args}
        self._outputs.append(record)
        if self.verbose:
            self._print_full(record)
            return
        already = cid in self._call_args
        if already:
            body = render_body(
                name,
                arguments=args,
                result=content,
                live=False,
                expanded=False,
                ok=ok,
            )
            if body.plain:
                self.console.print(body)
            return
        self.console.print(
            render_card(
                name,
                arguments=args,
                result=content,
                live=False,
                expanded=False,
                ok=ok,
            )
        )

    def _print_full(self, item: dict) -> None:
        name = item.get("name") or "tool"
        content = item.get("content") or ""
        ok = bool(item.get("ok"))
        args = item.get("arguments") or {}
        label = TOOL_LABELS.get(name, name)
        if name == "thinking":
            self.console.print(Panel(Text(content, style="dim italic"), title=label, border_style="dim"))
            return
        self.console.print(
            render_card(
                name,
                arguments=args,
                result=content,
                live=False,
                expanded=True,
                ok=ok,
            )
        )

    def result(self, stop_reason: str, turns: int, prompt_tokens: int, completion_tokens: int) -> None:
        if stop_reason == "completed" and not self.verbose:
            return
        label = STOP_LABELS.get(stop_reason, stop_reason)
        if self.verbose:
            self.console.print(
                f"[dim]{label} · {turns} steps · "
                f"tokens {prompt_tokens}+{completion_tokens}[/dim]"
            )
        elif stop_reason != "completed":
            self.console.print(f"[dim]{label}[/dim]")


def _short_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) <= 2:
        return path
    return str(Path(*parts[-2:]))


def _call_summary(name: str, arguments: dict) -> str:
    return key_argument(name, arguments)


def _strip_internal(content: str) -> str:
    return strip_internal(content)
