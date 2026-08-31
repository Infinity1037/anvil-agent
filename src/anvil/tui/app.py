from __future__ import annotations

import time
from concurrent.futures import Future
from pathlib import Path
from threading import Lock

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Static

from anvil.agent.context import ContextSnapshot
from anvil.agent.loop import Agent, CompactResult
from anvil.events import AgentEvent
from anvil.config import effort_label
from anvil.agent.permissions import ApprovalDecision
from anvil.tui.chrome import (
    WELCOME_TEXT,
    ApprovalPanel,
    Composer,
    HelpScreen,
    LogView,
    PreviewScreen,
    ResumeScreen,
    SuggestList,
    footer_text,
    session_header,
)
from anvil.tui.complete import (
    SLASH_ALIASES,
    Suggestion,
    apply_mention,
    effort_suggestions,
    index_from_location,
    location_from_index,
    match_files,
    match_slash,
    mention_query,
    perm_suggestions,
)
from anvil.session import Session, find_session, list_sessions
from anvil.tui.widgets import AssistantBlock, FoldBlock, NoticeBlock, UserBlock
from anvil.ui.format import (
    STOP_LABELS,
    compact_result_text,
    context_badge,
    context_report,
    strip_internal,
    tool_message_ok,
)

# Paint coalescing for live tokens. First chunk is immediate; later deltas
# wait out the remainder of this window. Not a typewriter — the text is
# already in the queue, the terminal just does not redraw on every token.
STREAM_FLUSH_S = 0.05


class AnvilApp(App[None]):
    """Full-screen session: thinking/tool cards expand in place (Ctrl+O)."""

    TITLE = "Anvil"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen {
        layout: vertical;
        background: #0e0e0e;
    }
    #log {
        height: 1fr;
        width: 1fr;
        padding: 0 2 1 3;
        scrollbar-size: 1 1;
    }
    FoldBlock {
        margin-top: 1;
    }
    FoldBlock.thinking {
        color: #6b7280;
        margin-top: 1;
        margin-bottom: 0;
    }
    #suggest {
        display: none;
        max-height: 12;
        background: #161616;
        border: solid #3a3a3a;
        margin: 0 1;
        padding: 0 1;
        overflow-y: auto;
    }
    #composer {
        height: 5;
        background: #161616;
        border: tall #3a3a3a;
        margin: 0 1 0 1;
        padding: 0 1;
    }
    #status {
        height: 2;
        padding: 0 3 1 3;
        color: #9aa0a6;
        background: #0e0e0e;
    }
    """

    BINDINGS = [
        Binding("ctrl+o", "toggle_expand", "Expand", show=False, priority=True),
        Binding("ctrl+c", "interrupt", "Stop", show=False, priority=True),
        Binding("shift+tab", "cycle_perm", "Perm", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit", show=False, priority=True),
        Binding("f1", "show_help", "Help", show=False, priority=True),
        Binding("pageup", "page_up", show=False, priority=True),
        Binding("pagedown", "page_down", show=False, priority=True),
    ]

    def __init__(
        self,
        agent: Agent,
        *,
        initial: str = "",
        verbose: bool = False,
        resume_picker: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self._initial = initial
        self._verbose = verbose
        self._resume_picker = resume_picker
        self._expanded = False
        self._busy = False
        self._approving = False
        self._hold_suggest = False
        self._pin_bottom = True
        self._quit_armed = False
        self._quit_timer = None
        self._thinking: FoldBlock | None = None
        self._assistant: AssistantBlock | None = None
        self._tools: dict[str, FoldBlock] = {}
        self._approval_future: Future[ApprovalDecision] | None = None
        self._bridge = _EventBridge(self)

    def compose(self) -> ComposeResult:
        yield LogView(id="log")
        yield ApprovalPanel(id="approval")
        yield SuggestList(id="suggest")
        yield Composer(placeholder="输入任务…  Enter 发送，shift+enter 换行", id="composer")
        yield Static(id="status")

    def on_unmount(self) -> None:
        self._bridge.close()

    def on_mount(self) -> None:
        self._refresh_status()
        log = self.query_one("#log", VerticalScroll)
        has_history = len(self.agent.messages) > 1
        if not self._initial and not has_history:
            log.mount(NoticeBlock(WELCOME_TEXT))
        self.query_one("#composer", Composer).focus()
        self.agent.approver = self
        if self._initial:
            text = self._initial
            self._initial = ""
            self.call_after_refresh(self._submit, text)
        elif has_history:
            self.call_after_refresh(self._replay_session)
        elif self._resume_picker:
            self.call_after_refresh(self._open_resume_picker)

    def action_help_quit(self) -> None:
        """Textual binds Ctrl+C to a quit hint; we use it as interrupt/quit."""
        self.action_interrupt()

    def action_toggle_expand(self) -> None:
        self._expanded = not self._expanded
        for block in self.query(FoldBlock):
            block.set_expanded(self._expanded)
        self._refresh_status()
        if not self._busy:
            self.query_one("#composer", Composer).focus()

    def action_show_help(self) -> None:
        if isinstance(self.screen, HelpScreen):
            self.pop_screen()
            return
        self.push_screen(HelpScreen())

    def on_key(self, event: events.Key) -> None:
        if event.key != "escape":
            return
        if isinstance(self.screen, PreviewScreen):
            return
        if self._approving:
            return
        if self.suggestions_open() or isinstance(self.screen, (HelpScreen, ResumeScreen)):
            return
        if self._busy:
            self._disarm_quit()
            self._halt_run()
            event.prevent_default()
            event.stop()

    def _halt_run(self) -> None:
        self.agent.session.cancel.set()
        abort = getattr(getattr(self.agent, "llm", None), "abort", None)
        if callable(abort):
            abort()
        self._hide_approval("cancelled")

    def action_interrupt(self) -> None:
        if self._busy:
            self._disarm_quit()
            self._halt_run()
            return
        composer = self.query_one("#composer", Composer)
        if composer.text.strip():
            self._disarm_quit()
            composer.clear()
            self.action_suggest_hide()
            return
        if self._quit_armed:
            self.exit()
            return
        self._quit_armed = True
        self._refresh_status()
        if self._quit_timer is not None:
            self._quit_timer.stop()
        self._quit_timer = self.set_timer(1.5, self._disarm_quit)

    def _disarm_quit(self) -> None:
        self._quit_armed = False
        timer = self._quit_timer
        self._quit_timer = None
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        if self.is_running:
            self._refresh_status()

    def action_page_up(self) -> None:
        if isinstance(self.screen, PreviewScreen):
            self.screen.action_page_up()
            return
        self.unfollow_tail()
        self.query_one("#log", VerticalScroll).scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        if isinstance(self.screen, PreviewScreen):
            self.screen.action_page_down()
            return
        self.query_one("#log", VerticalScroll).scroll_page_down(animate=False)
        self.call_after_refresh(self.follow_if_at_bottom)

    def unfollow_tail(self) -> None:
        if self._pin_bottom:
            self._pin_bottom = False
            self._refresh_status()

    def follow_if_at_bottom(self) -> None:
        try:
            log = self.query_one("#log", VerticalScroll)
            at_bottom = float(log.scroll_y) >= float(log.max_scroll_y) - 1
        except Exception:
            at_bottom = True
        if at_bottom != self._pin_bottom:
            self._pin_bottom = at_bottom
            self._refresh_status()

    def suggestions_open(self) -> bool:
        try:
            listing = self.query_one("#suggest", SuggestList)
        except Exception:
            return False
        return bool(listing.display and listing.items)

    def action_suggest_hide(self) -> None:
        self._hold_suggest = False
        try:
            self.query_one("#suggest", SuggestList).set_items([])
        except Exception:
            pass
        self._refresh_status()

    def action_suggest_move(self, delta: int) -> None:
        listing = self.query_one("#suggest", SuggestList)
        if not listing.items:
            return
        current = listing.highlighted or 0
        listing.highlighted = (current + delta) % len(listing.items)

    def action_suggest_accept(self, submit_slash: bool = True) -> None:
        listing = self.query_one("#suggest", SuggestList)
        item = listing.current()
        if item is None:
            self.action_suggest_hide()
            return
        composer = self.query_one("#composer", Composer)
        if item.kind == "slash":
            if item.value == "effort":
                self._hold_suggest = True
                composer.clear()
                self._open_effort_picker()
                return
            if item.value == "resume":
                composer.clear()
                self.action_suggest_hide()
                self._open_resume_picker()
                return
            composer.text = f"/{item.value}"
            self.action_suggest_hide()
            if submit_slash:
                self.action_submit_composer()
            return
        if item.kind == "effort":
            composer.clear()
            self.action_suggest_hide()
            self._apply_effort(item.value)
            return
        if item.kind == "perm":
            composer.clear()
            self.action_suggest_hide()
            self._apply_perm(item.value)
            return
        if item.kind == "session":
            self.action_suggest_hide()
            composer.clear()
            self._resume_session(item.value)
            return
        cursor = index_from_location(composer.text, composer.cursor_location)
        found = mention_query(composer.text, cursor)
        if found is None:
            found = mention_query(composer.text, len(composer.text))
        if found is None:
            self.action_suggest_hide()
            return
        at_index, prefix = found
        updated = apply_mention(composer.text, at_index, prefix, item.value)
        composer.text = updated
        composer.cursor_location = location_from_index(updated, at_index + len("@" + item.value + " "))
        self.action_suggest_hide()

    def on_text_area_changed(self, event) -> None:
        if getattr(event, "text_area", None) is None:
            return
        if event.text_area.id != "composer" or self._busy:
            return
        if self._quit_armed:
            self._disarm_quit()
        self._refresh_suggestions()

    def _refresh_suggestions(self) -> None:
        composer = self.query_one("#composer", Composer)
        listing = self.query_one("#suggest", SuggestList)
        text = composer.text
        if self._hold_suggest and not (text or "").strip():
            return
        self._hold_suggest = False
        slash = match_slash(text)
        if slash is not None:
            listing.set_items(slash)
            self._refresh_status()
            return
        cursor = index_from_location(text, composer.cursor_location)
        found = mention_query(text, cursor)
        if found is not None:
            _, prefix = found
            listing.set_items(match_files(Path(self.agent.config.workspace), prefix))
            self._refresh_status()
            return
        listing.set_items([])
        self._refresh_status()

    def action_submit_composer(self) -> None:
        if self._busy:
            return
        if self.suggestions_open():
            self.action_suggest_accept(submit_slash=True)
            return
        composer = self.query_one("#composer", Composer)
        text = (composer.text or "").strip()
        composer.clear()
        self.action_suggest_hide()
        if not text:
            return
        if text.startswith("/") and "\n" not in text:
            parts = text[1:].split(None, 1)
            name = SLASH_ALIASES.get(parts[0].lower(), parts[0].lower())
            arg = parts[1].strip() if len(parts) > 1 else ""
            if self._run_slash(name, arg):
                return
            log = self.query_one("#log", VerticalScroll)
            log.mount(NoticeBlock(f"未知命令 {text}，输入 / 查看可用命令"))
            log.scroll_end(animate=False)
            return
        self._submit(text)

    def _run_slash(self, name: str, arg: str = "") -> bool:
        if name in {"quit", "exit"}:
            self.exit()
            return True
        if name == "expand":
            self.action_toggle_expand()
            return True
        if name in {"new", "clear"}:
            self._start_new_session()
            return True
        if name == "resume":
            if arg:
                return self._resume_session(arg)
            self._open_resume_picker()
            return True
        if name == "effort":
            if not arg:
                self._hold_suggest = True
                self.query_one("#composer", Composer).clear()
                self._open_effort_picker()
                return True
            self._apply_effort(arg)
            return True
        if name == "perm":
            if not arg:
                self._hold_suggest = True
                self.query_one("#composer", Composer).clear()
                self._open_perm_picker()
                return True
            self._apply_perm(arg)
            return True
        if name == "status":
            usage = self.agent.usage
            session = self.agent.session
            snapshot = self._context_snapshot()
            log = self.query_one("#log", VerticalScroll)
            log.mount(
                NoticeBlock(
                    f"model {self.agent.config.model}  "
                    f"{session.effort_status()}  {session.permission_status()}\n"
                    f"messages {len(self.agent.messages)}  "
                    f"session {session.id}\n"
                    f"{context_badge(snapshot)}  "
                    f"API tokens {usage.prompt_tokens}+{usage.completion_tokens}\n"
                    f"workspace {self.agent.config.workspace}"
                )
            )
            log.scroll_end(animate=False)
            return True
        if name == "context":
            self._show_context()
            return True
        if name == "compact":
            self._start_manual_compaction(arg)
            return True
        if name == "help":
            self.action_show_help()
            return True
        return False

    def _show_context(self) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(NoticeBlock(context_report(self._context_snapshot(), self.agent.usage)))
        log.scroll_end(animate=False)

    def _context_snapshot(self) -> ContextSnapshot:
        getter = getattr(self.agent, "context_snapshot", None)
        if callable(getter):
            return getter()
        messages = getattr(self.agent, "messages", [])
        budget = int(getattr(self.agent.config, "context_budget", 100_000))
        return ContextSnapshot(
            estimated_tokens=0,
            budget=budget,
            history_messages=len(messages),
            view_messages=len(messages),
        )

    def _start_manual_compaction(self, instruction: str) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(NoticeBlock("正在压缩上下文…  Esc/Ctrl+C 可取消"))
        self._pin_bottom = True
        self._disarm_quit()
        self._set_busy(True)
        log.scroll_end(animate=False)
        self._run_manual_compaction(instruction)

    @work(thread=True, exclusive=True)
    def _run_manual_compaction(self, instruction: str) -> None:
        self.agent.session.cancel.clear()
        try:
            result = self.agent.compact(
                instruction,
                on_event=self._bridge.push,
                cancel=self.agent.session.cancel,
            )
        except Exception as exc:
            result = CompactResult(
                "failed",
                0,
                0,
                error=f"{type(exc).__name__}: {exc}",
            )
        try:
            self.call_from_thread(self._finish_manual_compaction, result)
        except Exception:
            pass

    def _finish_manual_compaction(self, result: CompactResult) -> None:
        self._bridge.flush()
        if result.status != "compacted":
            log = self.query_one("#log", VerticalScroll)
            style = "bold red" if result.status == "failed" else "dim"
            log.mount(NoticeBlock(compact_result_text(result), style=style))
            log.scroll_end(animate=False)
        self._set_busy(False)

    def _start_new_session(self) -> None:
        self.agent.new_session()
        log = self.query_one("#log", VerticalScroll)
        log.remove_children()
        self._thinking = None
        self._assistant = None
        self._tools = {}
        self._expanded = False
        log.mount(NoticeBlock("已开始新会话。"))
        log.mount(NoticeBlock(WELCOME_TEXT))
        self.action_suggest_hide()
        self._refresh_status()

    def _perm_id(self) -> str:
        return str(getattr(self.agent.session, "permission_mode", None) or "ask")

    def action_cycle_perm(self) -> None:
        if self._approving or isinstance(self.screen, (HelpScreen, ResumeScreen, PreviewScreen)):
            return
        current = self._perm_id()
        nxt = "auto" if current == "ask" else "ask"
        self._apply_perm(nxt)

    def _open_perm_picker(self) -> None:
        self._hold_suggest = True
        current = self._perm_id()
        items = perm_suggestions(current)
        highlight = next((index for index, item in enumerate(items) if item.value == current), 0)
        self.query_one("#suggest", SuggestList).set_items(items, highlighted=highlight)
        self._refresh_status()
        self.query_one("#composer", Composer).focus()

    def _apply_perm(self, token: str) -> None:
        log = self.query_one("#log", VerticalScroll)
        session = self.agent.session
        before = session.permission_status()
        try:
            line = session.set_permission(token)
        except ValueError as exc:
            log.mount(NoticeBlock(str(exc)))
            log.scroll_end(animate=False)
            return
        self._refresh_status()
        if line != before:
            log.mount(NoticeBlock(f"权限 {self._perm_id()}"))
            log.scroll_end(animate=False)

    def decide(self, call, cancel=None) -> ApprovalDecision:
        """Approver used by the agent worker thread."""
        future: Future[ApprovalDecision] = Future()

        def show() -> None:
            if not self.is_running:
                future.set_result(ApprovalDecision("cancelled"))
                return
            self._open_approval(call, future)

        self.call_from_thread(show)
        while not future.done():
            if cancel is not None and cancel.is_set():
                self.call_from_thread(self._hide_approval, ApprovalDecision("cancelled"))
                break
            time.sleep(0.05)
        try:
            return ApprovalDecision.from_raw(future.result(timeout=2))
        except Exception:
            return ApprovalDecision("cancelled")

    def _open_approval(self, call, future: Future[ApprovalDecision]) -> None:
        self._approval_future = future
        self._approving = True
        panel = self.query_one("#approval", ApprovalPanel)
        composer = self.query_one("#composer", Composer)
        composer.display = False
        self.action_suggest_hide()
        panel.set_request(call, lambda value: self._finish_approval(future, value))
        self._refresh_status()

    def open_approval_preview(self, title: str, body) -> None:
        if isinstance(self.screen, PreviewScreen):
            return
        self.push_screen(PreviewScreen(title, body), self._after_preview)

    def _after_preview(self, _result=None) -> None:
        if not self._approving:
            return
        try:
            self.query_one("#approval", ApprovalPanel).focus_choices()
        except Exception:
            pass

    def _hide_approval(self, value: object = "cancelled") -> None:
        if isinstance(self.screen, PreviewScreen):
            try:
                self.pop_screen()
            except Exception:
                pass
        try:
            panel = self.query_one("#approval", ApprovalPanel)
            if panel.display:
                panel.hide_panel()
        except Exception:
            pass
        self._finish_approval(self._approval_future, value)

    def _finish_approval(self, future: Future[ApprovalDecision] | None, value: object) -> None:
        self._approving = False
        try:
            composer = self.query_one("#composer", Composer)
            composer.display = True
            if not self._busy:
                composer.focus()
        except Exception:
            pass
        if self.is_running:
            self._refresh_status()
        if future is not None and not future.done():
            future.set_result(ApprovalDecision.from_raw(value if value is not None else "deny"))
        self._approval_future = None

    def _effort_id(self) -> str:
        session = self.agent.session
        return effort_label(
            bool(getattr(session, "thinking", self.agent.config.thinking)),
            str(getattr(session, "reasoning_effort", None) or "max"),
        )

    def _open_effort_picker(self) -> None:
        self._hold_suggest = True
        current = self._effort_id()
        items = effort_suggestions(current)
        highlight = next((index for index, item in enumerate(items) if item.value == current), 0)
        self.query_one("#suggest", SuggestList).set_items(items, highlighted=highlight)
        self._refresh_status()
        self.query_one("#composer", Composer).focus()

    def _apply_effort(self, token: str) -> None:
        log = self.query_one("#log", VerticalScroll)
        before = self.agent.session.effort_status()
        try:
            line = self.agent.session.set_effort(token)
        except ValueError as exc:
            log.mount(NoticeBlock(str(exc)))
            log.scroll_end(animate=False)
            return
        self._refresh_status()
        if line != before:
            log.mount(NoticeBlock(f"思考强度 {self._effort_id()}"))
            log.scroll_end(animate=False)

    def _open_resume_picker(self) -> None:
        infos = list_sessions(self.agent.config.workspace)
        log = self.query_one("#log", VerticalScroll)
        if not infos:
            log.mount(NoticeBlock("本工作区还没有可恢复的会话。"))
            log.scroll_end(animate=False)
            return
        current = self.agent.session.id
        items = []
        for info in infos[:20]:
            when = (info.created or "").replace("T", " ")[:16]
            mark = "  ← current" if info.id == current else ""
            items.append(
                Suggestion(
                    "session",
                    info.id,
                    info.title or info.id,
                    f"{when}  {info.messages} msgs{mark}",
                )
            )
        self.push_screen(ResumeScreen(items), self._picked_resume)

    def _picked_resume(self, session_id: str | None) -> None:
        if not session_id or session_id == self.agent.session.id:
            return
        self._resume_session(session_id)

    def _resume_session(self, token: str) -> bool:
        found = find_session(self.agent.config.workspace, token)
        if found is None:
            log = self.query_one("#log", VerticalScroll)
            log.mount(NoticeBlock(f"找不到会话 {token}"))
            log.scroll_end(animate=False)
            return True
        self.agent.attach_session(Session.load(self.agent.config, found.path))
        self._replay_session()
        return True

    def _replay_session(self) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.remove_children()
        log.mount(NoticeBlock(f"已恢复 {self.agent.session.id}"))
        pending: dict[str, FoldBlock] = {}
        for message in self.agent.messages:
            if message.role == "system":
                continue
            if message.role == "user":
                text = (message.content or "").strip()
                if not text or text.startswith("["):
                    continue
                log.mount(UserBlock(text))
            elif message.role == "assistant":
                if message.reasoning_content:
                    thinking = FoldBlock("thinking")
                    thinking.set_full(message.reasoning_content)
                    thinking.finalize()
                    log.mount(thinking)
                if message.content:
                    log.mount(AssistantBlock(message.content))
                for call in message.tool_calls or []:
                    block = FoldBlock(call.name, arguments=call.arguments)
                    block.live = False
                    pending[call.id] = block
                    log.mount(block)
            elif message.role == "tool":
                block = pending.get(message.tool_call_id or "")
                if block is None:
                    continue
                content = message.content or ""
                block.set_result(content, tool_message_ok(content))
        self._thinking = None
        self._assistant = None
        self._tools = {}
        log.scroll_end(animate=False)
        self._refresh_status()

    def _submit(self, text: str) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(UserBlock(text))
        self._thinking = None
        self._assistant = None
        self._tools = {}
        self._pin_bottom = True
        self._disarm_quit()
        self._set_busy(True)
        log.scroll_end(animate=False)
        self._run_agent(text)

    @work(thread=True, exclusive=True)
    def _run_agent(self, text: str) -> None:
        self.agent.session.cancel.clear()
        try:
            self.agent.run(text, on_event=self._bridge.push, cancel=self.agent.session.cancel)
        except Exception as exc:
            try:
                self.call_from_thread(
                    self._apply_event,
                    AgentEvent("error", {"message": f"{type(exc).__name__}: {exc}"}),
                )
            except Exception:
                pass
        finally:
            try:
                self.call_from_thread(self._idle)
            except Exception:
                pass

    def _idle(self) -> None:
        self._bridge.flush()
        if self._thinking is not None:
            self._thinking.finalize()
        self._set_busy(False)

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        try:
            composer = self.query_one("#composer", Composer)
            composer.disabled = value
            if value:
                self.query_one("#log", VerticalScroll).focus()
            else:
                composer.focus()
        except Exception:
            pass
        self._refresh_status()

    def _refresh_status(self) -> None:
        try:
            widget = self.query_one("#status", Static)
            cfg = self.agent.config
            width = widget.size.width or self.size.width
            snapshot = self._context_snapshot()
            identity = session_header(
                model=cfg.model,
                effort=self._effort_id(),
                perm=self._perm_id(),
                workspace=cfg.workspace,
                context=context_badge(snapshot, detailed=width >= 90),
            )
            reason_open = False
            if self._approving:
                try:
                    reason_open = self.query_one("#approval", ApprovalPanel).reason_open
                except Exception:
                    reason_open = False
            widget.update(
                footer_text(
                    identity=identity,
                    expanded=self._expanded,
                    busy=self._busy,
                    suggesting=self.suggestions_open(),
                    quit_armed=self._quit_armed,
                    follow=self._pin_bottom,
                    approving=self._approving,
                    approval_reason=reason_open,
                    width=width,
                )
            )
        except Exception:
            pass

    def _follow_log(self) -> None:
        if not self._pin_bottom:
            return
        try:
            self.query_one("#log", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    def _apply_event(self, event: AgentEvent) -> None:
        if not self.is_running:
            return
        try:
            log = self.query_one("#log", VerticalScroll)
        except Exception:
            return
        kind = event.kind
        payload = event.payload
        if kind == "turn":
            self._thinking = None
            self._assistant = None
        elif kind == "delta":
            self._on_delta(log, payload)
        elif kind == "assistant":
            self._on_assistant(log, payload)
        elif kind == "tool_result":
            self._on_tool_result(log, payload)
        elif kind == "compact":
            if payload.get("semantic"):
                log.mount(
                    NoticeBlock(
                        compact_result_text(
                            CompactResult(
                                "compacted",
                                int(payload.get("before_tokens") or 0),
                                int(payload.get("after_tokens") or 0),
                                int(payload.get("covered_count") or 0),
                            )
                        )
                    )
                )
            else:
                log.mount(
                    NoticeBlock(
                        "已折叠较早的上下文（完整记录仍在本工作区会话文件中）"
                    )
                )
        elif kind == "error":
            log.mount(NoticeBlock(str(payload.get("message") or "error"), style="bold red"))
        elif kind == "ended":
            reason = payload.get("stop_reason") or ""
            if reason and reason != "completed":
                log.mount(NoticeBlock(STOP_LABELS.get(reason, reason)))
            elif self._verbose:
                log.mount(
                    NoticeBlock(
                        f"{STOP_LABELS.get(reason, reason)} · {payload.get('turns', 0)} steps"
                    )
                )
        self._follow_log()

    def _on_delta(self, log: VerticalScroll, payload: dict) -> None:
        chunk = payload.get("text") or ""
        if payload.get("kind") == "reasoning":
            if self._thinking is None or not self._thinking.live:
                block = FoldBlock("thinking")
                block.set_expanded(self._expanded)
                self._thinking = block
                log.mount(block)
            self._thinking.append_full(chunk)
            return
        if payload.get("kind") != "content" or not chunk:
            return
        if self._thinking is not None:
            self._thinking.finalize()
        if self._assistant is None:
            self._assistant = AssistantBlock()
            log.mount(self._assistant)
        self._assistant.append(chunk)

    def _on_assistant(self, log: VerticalScroll, payload: dict) -> None:
        reasoning = payload.get("reasoning") or ""
        if reasoning:
            if self._thinking is None:
                block = FoldBlock("thinking")
                block.set_expanded(self._expanded)
                self._thinking = block
                log.mount(block)
            self._thinking.set_full(reasoning)
            self._thinking.finalize()
        elif self._thinking is not None:
            self._thinking.finalize()
        content = payload.get("content") or ""
        if content:
            if self._assistant is None:
                self._assistant = AssistantBlock()
                log.mount(self._assistant)
            self._assistant.set_text(content)
        for call in payload.get("tool_calls") or []:
            cid = str(call.get("id") or "")
            name = call.get("name") or "tool"
            args = call.get("arguments") or {}
            if cid and cid in self._tools:
                continue
            block = FoldBlock(name, arguments=args)
            block.live = True
            block.set_expanded(self._expanded)
            block.refresh_body()
            if cid:
                self._tools[cid] = block
            log.mount(block)

    def _on_tool_result(self, log: VerticalScroll, payload: dict) -> None:
        cid = str(payload.get("id") or "")
        name = payload.get("name") or "tool"
        content = strip_internal(payload.get("content") or "")
        block = self._tools.get(cid) if cid else None
        if block is None:
            block = FoldBlock(name, arguments=payload.get("arguments") or {})
            block.set_expanded(self._expanded)
            if cid:
                self._tools[cid] = block
            log.mount(block)
        ok = payload.get("ok")
        if ok is None:
            ok = tool_message_ok(payload.get("content") or "")
        block.set_result(content, bool(ok))


class _EventBridge:
    """Batch worker-thread events onto the UI thread.

    Delta tokens share a ~50ms paint window so the transcript does not
    jump on every SSE chunk. Non-delta events (tools, errors, end of
    turn) flush immediately.
    """

    def __init__(self, app: AnvilApp) -> None:
        self.app = app
        self.lock = Lock()
        self.queue: list[AgentEvent] = []
        self.scheduled = False
        self.last_flush_at: float | None = None
        self._timer = None

    def push(self, event: AgentEvent) -> None:
        urgent = event.kind != "delta"
        with self.lock:
            self.queue.append(event)
            already = self.scheduled
            self.scheduled = True
            if already and not urgent:
                return
        try:
            self.app.call_from_thread(self._wake)
        except Exception:
            with self.lock:
                self.scheduled = False

    def _wake(self) -> None:
        if not self.app.is_running:
            self.close()
            return
        self._cancel_timer()
        with self.lock:
            batch = list(self.queue)
        if not batch:
            with self.lock:
                self.scheduled = False
            return
        urgent = any(event.kind != "delta" for event in batch)
        delay = stream_flush_delay(time.monotonic(), self.last_flush_at, urgent=urgent)
        if delay <= 0:
            self.flush()
            return
        self._timer = self.app.set_timer(delay, self.flush)

    def flush(self, _timer=None) -> None:
        self._cancel_timer()
        with self.lock:
            batch = self.queue
            self.queue = []
            self.scheduled = False
        if not batch:
            return
        self.last_flush_at = time.monotonic()
        if not self.app.is_running:
            return
        for event in coalesce_events(batch):
            self.app._apply_event(event)

    def close(self) -> None:
        self._cancel_timer()
        with self.lock:
            self.queue = []
            self.scheduled = False

    def _cancel_timer(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is None:
            return
        try:
            timer.stop()
        except Exception:
            pass


def stream_flush_delay(now: float, last_flush_at: float | None, *, urgent: bool) -> float:
    """Seconds to wait before the next paint. Zero means flush now."""
    if urgent or last_flush_at is None:
        return 0.0
    return max(0.0, STREAM_FLUSH_S - (now - last_flush_at))


def coalesce_events(events: list[AgentEvent]) -> list[AgentEvent]:
    merged: list[AgentEvent] = []
    for event in events:
        if (
            merged
            and event.kind == "delta"
            and merged[-1].kind == "delta"
            and event.payload.get("kind") == merged[-1].payload.get("kind")
        ):
            prev = merged[-1]
            merged[-1] = AgentEvent(
                kind="delta",
                payload={
                    "kind": event.payload.get("kind"),
                    "text": (prev.payload.get("text") or "") + (event.payload.get("text") or ""),
                },
            )
        else:
            merged.append(event)
    return merged


def run_tui(
    agent: Agent,
    *,
    initial: str = "",
    verbose: bool = False,
    resume_picker: bool = False,
) -> int:
    AnvilApp(
        agent, initial=initial, verbose=verbose, resume_picker=resume_picker
    ).run()
    return 0
