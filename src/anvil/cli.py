from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from anvil import __version__
from anvil.agent.context import ContextManager
from anvil.agent.loop import Agent, as_ui_handler
from anvil.config import Config, ConfigError
from anvil.llm.openai_compat import DeepSeekClient
from anvil.session import Session, find_session, latest_session
from anvil.tools import build_tools
from anvil.ui.format import compact_result_text, context_badge, context_report
from anvil.ui.prompt import read_user_line
from anvil.ui.terminal import TerminalUI


def _ensure_utf8_output() -> None:
    """Keep redirected and legacy Windows output from crashing on Unicode."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("_", "-")
        if encoding in {"utf-8", "utf8"}:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_output()
    parser = argparse.ArgumentParser(
        prog="anvil",
        description="Anvil — a local coding agent.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional first message. After it finishes, Anvil stays in the chat unless --once is set.",
    )
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
        help="DeepSeek reasoning_effort (default: max). Session /effort can change it later.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single task and exit (no interactive chat).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show token usage per turn.")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Line-oriented REPL instead of the full-screen TUI (Ctrl+O reprints below the prompt).",
    )
    parser.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Resume the most recent session in this workspace.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        help="Resume a workspace session. Pass an id, or omit to pick in the TUI.",
    )
    parser.add_argument(
        "--perm",
        choices=["ask", "auto"],
        help="ask: confirm edits and shell (TUI default). auto: do not prompt (--once/--plain default).",
    )
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
    session: Session | None = None
    open_resume_picker = False
    if args.continue_session:
        latest = latest_session(workspace)
        if latest is None:
            console.print("[dim]No previous session in this workspace; starting a new one.[/dim]")
        else:
            session = Session.load(config, latest.path)
    elif args.resume is not None:
        if args.resume:
            found = find_session(workspace, args.resume)
            if found is None:
                console.print(f"[red]No session matching {args.resume!r} in this workspace.[/red]")
                client.close()
                return 2
            session = Session.load(config, found.path)
        else:
            open_resume_picker = True
    if session is None:
        session = Session(config)
    tools = build_tools(workspace, session.todo, config.shell_timeout)
    agent = Agent(
        config,
        client,
        tools,
        ContextManager(config.context_budget, workspace=workspace),
        session=session,
    )

    prompt = " ".join(args.prompt).strip()
    use_tui = (
        not args.once
        and not args.plain
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )
    try:
        if args.once and not prompt:
            console.print("[red]--once requires a task on the command line.[/red]")
            return 2
        if args.perm:
            agent.session.set_permission(args.perm)
        elif not use_tui:
            agent.session.permission_mode = "auto"
        if use_tui:
            from anvil.tui.app import run_tui

            return run_tui(
                agent,
                initial=prompt,
                verbose=args.verbose,
                resume_picker=open_resume_picker,
            )
        ui = TerminalUI(console, verbose=args.verbose)
        ui.banner(str(workspace), config.model, config.thinking)
        if prompt:
            code = _run_once(agent, ui, prompt)
            if args.once:
                return code
        return _repl(agent, ui)
    finally:
        client.close()


def _run_once(agent: Agent, ui: TerminalUI, prompt: str) -> int:
    ui.begin_turn()
    agent.session.cancel.clear()
    try:
        result = agent.run(prompt, on_event=as_ui_handler(ui.handle), cancel=agent.session.cancel)
    except KeyboardInterrupt:
        agent.session.cancel.set()
        ui.console.print("\n[dim]Interrupted[/dim]")
        return 0
    ui.result(
        result.stop_reason,
        result.turns,
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
    )
    return 0 if result.stop_reason == "completed" else 1


def _repl(agent: Agent, ui: TerminalUI) -> int:
    ui.console.print("[dim]Enter send · Ctrl+O expand · Ctrl+C interrupt · /help[/dim]")
    while True:
        try:
            line = read_user_line(ui)
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            return 0
        if not line:
            continue
        if line in {"/quit", "/exit", "/q"}:
            return 0
        if line == "/help":
            ui.console.print(
                "Ctrl+O        reprint last turn's thinking/tool output below the prompt\n"
                "/expand       same as Ctrl+O (if the shortcut is swallowed)\n"
                "/status       model, session, tokens\n"
                "/context      current model-view budget and checkpoint\n"
                "/compact      compact old context; optional focus text\n"
                "/effort       thinking intensity: off | low | high | max\n"
                "/clear        start a new session in this workspace\n"
                "/resume       list or restore a previous session\n"
                "/exit         exit\n"
                "/help         this list\n"
                "(This is --plain mode. Default `anvil` expands in place.)"
            )
            continue
        if line == "/expand":
            ui.expand()
            continue
        if line in {"/new", "/clear"}:
            agent.new_session()
            ui.console.print("[dim]new session[/dim]")
            continue
        if line == "/resume" or line.startswith("/resume "):
            token = line[7:].strip()
            from anvil.session import find_session, list_sessions

            if not token:
                infos = list_sessions(agent.config.workspace)
                if not infos:
                    ui.console.print("[dim]no sessions in this workspace[/dim]")
                else:
                    for info in infos[:12]:
                        ui.console.print(f"[dim]{info.id}[/dim]  {info.title}")
                continue
            found = find_session(agent.config.workspace, token)
            if found is None:
                ui.console.print("[red]session not found[/red]")
                continue
            agent.attach_session(Session.load(agent.config, found.path))
            ui.console.print(f"[dim]resumed {found.id}[/dim]")
            continue
        if line == "/effort" or line.startswith("/effort "):
            arg = line[7:].strip()
            if not arg:
                ui.console.print(
                    f"[dim]{agent.session.effort_status()}  ·  /effort off|low|high|max[/dim]"
                )
                continue
            try:
                ui.console.print(f"[dim]{agent.session.set_effort(arg)}[/dim]")
            except ValueError as exc:
                ui.console.print(f"[red]{exc}[/red]")
            continue
        if line == "/status":
            snapshot = agent.context_snapshot()
            ui.console.print(
                f"model {agent.config.model}  {agent.session.effort_status()}\n"
                f"messages {len(agent.messages)}  "
                f"{context_badge(snapshot)}  "
                f"API tokens {agent.usage.prompt_tokens}+{agent.usage.completion_tokens}\n"
                f"workspace {agent.config.workspace}"
            )
            continue
        if line == "/context":
            ui.console.print(context_report(agent.context_snapshot(), agent.usage))
            continue
        if line == "/compact" or line.startswith("/compact "):
            instruction = line[len("/compact") :].strip()
            ui.console.print("[dim]compacting context…  Ctrl+C to cancel[/dim]")
            result = agent.compact(instruction, cancel=agent.session.cancel)
            style = "red" if result.status == "failed" else "dim"
            ui.console.print(compact_result_text(result), style=style)
            continue
        _run_once(agent, ui, line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
