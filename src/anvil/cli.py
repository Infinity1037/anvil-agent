from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from anvil import __version__
from anvil.agent.context import ContextManager
from anvil.agent.loop import Agent
from anvil.config import Config, ConfigError
from anvil.llm.openai_compat import DeepSeekClient
from anvil.tools import TodoStore, build_tools
from anvil.ui.terminal import TerminalUI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anvil",
        description="Anvil — a local coding agent powered by DeepSeek tool calling.",
    )
    parser.add_argument("prompt", nargs="*", help="Task for the agent. Omit to enter the REPL.")
    parser.add_argument(
        "-w",
        "--workspace",
        default=".",
        help="Workspace directory (default: current directory).",
    )
    parser.add_argument("--model", help="Model id (default: deepseek-v4-flash).")
    parser.add_argument("--max-turns", type=int, help="Maximum model/tool loops.")
    parser.add_argument("--no-thinking", action="store_true", help="Disable DeepSeek thinking mode.")
    parser.add_argument(
        "--effort",
        choices=["low", "high", "max"],
        help="DeepSeek reasoning_effort (thinking mode only).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show token usage per turn.")
    parser.add_argument("--version", action="version", version=f"anvil {__version__}")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        print(f"error: workspace is not a directory: {workspace}", file=sys.stderr)
        return 2

    console = Console()
    try:
        config = Config.from_env(
            workspace,
            model=args.model,
            thinking=False if args.no_thinking else None,
            reasoning_effort=args.effort,
            max_turns=args.max_turns,
        )
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    client = DeepSeekClient(config)
    todo = TodoStore()
    tools = build_tools(workspace, todo, config.shell_timeout)
    agent = Agent(config, client, tools, ContextManager(config.context_budget))
    ui = TerminalUI(console, verbose=args.verbose)
    ui.banner(str(workspace), config.model, config.thinking)

    prompt = " ".join(args.prompt).strip()
    try:
        if prompt:
            return _run_once(agent, ui, prompt)
        return _repl(agent, ui)
    finally:
        client.close()


def _run_once(agent: Agent, ui: TerminalUI, prompt: str) -> int:
    result = agent.run(prompt, on_event=ui.handle)
    ui.result(
        result.stop_reason,
        result.turns,
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
    )
    _write_transcript(agent)
    return 0 if result.stop_reason == "completed" else 1


def _repl(agent: Agent, ui: TerminalUI) -> int:
    ui.console.print("[dim]Type a task, or /help. Ctrl+C exits.[/dim]")
    while True:
        try:
            line = ui.console.input("[bold]anvil>[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            return 0
        if not line:
            continue
        if line in {"/quit", "/exit", "/q"}:
            return 0
        if line == "/help":
            ui.console.print("/reset  clear conversation\n/quit   exit\n/help   this list")
            continue
        if line == "/reset":
            agent.reset()
            ui.console.print("[dim]conversation cleared[/dim]")
            continue
        try:
            _run_once(agent, ui, line)
        except KeyboardInterrupt:
            ui.console.print("\n[dim]interrupted[/dim]")
    return 0


def _write_transcript(agent: Agent) -> None:
    folder = agent.config.workspace / ".anvil" / "transcripts"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = folder / f"{stamp}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for message in agent.messages:
                handle.write(json.dumps(message.to_openai(include_reasoning=True), ensure_ascii=False) + "\n")
    except OSError:
        return


if __name__ == "__main__":
    raise SystemExit(main())
