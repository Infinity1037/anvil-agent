from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text


class TerminalUI:
    def __init__(self, console: Console | None = None, verbose: bool = False) -> None:
        self.console = console or Console()
        self.verbose = verbose
        self._printed_reasoning = False
        self._printed_content = False

    def banner(self, workspace: str, model: str, thinking: bool) -> None:
        think = "thinking on" if thinking else "thinking off"
        self.console.print(
            Panel.fit(
                f"[bold]Anvil[/bold]  {model}  ·  {think}\n[dim]{workspace}[/dim]",
                border_style="cyan",
            )
        )

    def handle(self, kind: str, payload: dict) -> None:
        if kind == "turn":
            self._printed_reasoning = False
            self._printed_content = False
            self.console.print(
                Rule(f"turn {payload['turn']}/{payload['max_turns']}", style="cyan")
            )
        elif kind == "delta":
            text = payload.get("text") or ""
            if payload.get("kind") == "reasoning":
                if not self._printed_reasoning:
                    self.console.print("[dim italic]thinking[/dim italic]")
                    self._printed_reasoning = True
                self.console.print(Text(text, style="dim italic"), end="")
            elif payload.get("kind") == "content":
                if self._printed_reasoning and not self._printed_content:
                    self.console.print()
                self._printed_content = True
                self.console.print(text, end="")
        elif kind == "assistant":
            if self._printed_reasoning or self._printed_content:
                self.console.print()
            if not self._printed_reasoning and payload.get("reasoning"):
                self.console.print(
                    Panel(payload["reasoning"], title="thinking", border_style="dim")
                )
                self._printed_reasoning = True
            if not self._printed_content and payload.get("content"):
                self.console.print(payload["content"])
                self._printed_content = True
            for call in payload.get("tool_calls") or []:
                args = _short_args(call.get("arguments") or {})
                self.console.print(f"[bold cyan]● {call['name']}[/bold cyan] {args}")
            usage = payload.get("usage") or {}
            if self.verbose and usage:
                self.console.print(
                    f"[dim]tokens  in={usage.get('prompt_tokens', 0)}  "
                    f"out={usage.get('completion_tokens', 0)}[/dim]"
                )
        elif kind == "tool_result":
            preview = _preview(payload.get("content") or "", 500 if self.verbose else 280)
            style = "green" if payload.get("ok") else "red"
            self.console.print(f"[{style}]{preview}[/{style}]")
        elif kind == "error":
            self.console.print(f"[bold red]{payload.get('message')}[/bold red]")

    def result(self, stop_reason: str, turns: int, prompt_tokens: int, completion_tokens: int) -> None:
        self.console.print(
            f"[dim]{stop_reason} · {turns} turns · "
            f"tokens {prompt_tokens}+{completion_tokens}[/dim]"
        )

    def code(self, text: str, lexer: str = "text") -> None:
        self.console.print(Syntax(text, lexer, word_wrap=True))


def _short_args(arguments: dict) -> str:
    if not arguments:
        return ""
    try:
        dumped = json.dumps(arguments, ensure_ascii=False)
    except TypeError:
        dumped = str(arguments)
    if len(dumped) > 160:
        dumped = dumped[:157] + "..."
    return dumped


def _preview(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"
