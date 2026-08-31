"""Session chrome: shortcut strip, help overlay, input box, suggestions."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from anvil.agent.permissions import ApprovalDecision
from anvil.llm.types import ToolCall
from anvil.tui.complete import Suggestion
from anvil.tui.preview import (
    approval_has_more,
    approval_title,
    is_file_preview,
    render_approval_full,
    render_approval_summary,
)

HELP_TEXT = """\
键位
  ctrl+o         在思考链 / 工具卡片上原位展开或收起
  ctrl+e         确认条里查看完整写入 / 改动
  enter          发送
  shift+enter    换行
  tab            拒绝时填写原因
  /              命令补全
  @              引用工作区文件
  esc            停止当前任务
  shift+tab      切换权限 ask / auto
  ctrl+c         停止当前任务；空闲时再按一次退出
  ctrl+q         退出
  pageup/dn      滚动上面的对话
  ?  或  F1      打开 / 关闭本说明

命令
  /help          本说明
  /status        模型、会话与 token
  /context       查看模型视图、预算与 checkpoint
  /compact [重点] 手动压缩旧上下文
  /effort        选择思考强度（off / low / high / max）
  /perm          权限 ask（改前确认）/ auto（不问）
  /clear         开始新会话（工作区不变）
  /resume        打开会话列表并恢复
  /expand        同 ctrl+o
  /exit          退出

Esc 或 q 关闭
"""

WELCOME_TEXT = "会在当前工作区读文件、改代码、跑命令。/clear 新会话  ·  /effort 思考强度  ·  /perm 权限  ·  /resume 恢复  ·  键位见下方。"


def session_header(
    *,
    model: str,
    effort: str,
    workspace: Path,
    perm: str = "ask",
    context: str = "",
) -> str:
    cwd = Path(workspace)
    short = str(Path(*cwd.parts[-2:])) if len(cwd.parts) > 2 else str(cwd)
    fields = ["Anvil", model, effort, perm]
    if context:
        fields.append(context)
    fields.append(short)
    return "  ".join(fields)


def status_plain(
    *,
    expanded: bool,
    busy: bool,
    suggesting: bool = False,
    quit_armed: bool = False,
    follow: bool = True,
    approving: bool = False,
    approval_reason: bool = False,
) -> str:
    if quit_armed:
        return "再按一次 ctrl+c 退出  ·  ctrl+q 直接退出"
    if approving and approval_reason:
        return "输入原因  enter 提交拒绝  esc 取消输入"
    if approving:
        return "等待确认  enter 允许  s 本会话  esc 拒绝  ctrl+e 看全文  tab 原因"
    if busy and not follow:
        return "已离开底部  pageup/dn 滚动  ·  ctrl+c 停止"
    if busy:
        return "工作中…  esc/ctrl+c 停止  ·  pageup 向上看  ·  ctrl+o 展开"
    if suggesting:
        return "↑↓ 选择  ·  tab 补全  ·  enter 确认  ·  esc 关闭"
    fold = "收起" if expanded else "展开"
    return f"ctrl+o {fold}  ·  enter 发送  ·  shift+enter 换行  ·  ? 帮助"


def footer_text(
    *,
    identity: str,
    expanded: bool,
    busy: bool,
    suggesting: bool = False,
    quit_armed: bool = False,
    follow: bool = True,
    approving: bool = False,
    approval_reason: bool = False,
    width: int = 0,
) -> Text:
    ident = Text(identity, style="bold")
    hints = status_text(
        expanded=expanded,
        busy=busy,
        suggesting=suggesting,
        quit_armed=quit_armed,
        follow=follow,
        approving=approving,
        approval_reason=approval_reason,
    )
    if width and width > 8:
        ident.truncate(width)
        hints = hints.copy()
        hints.truncate(width)
    ident.append("\n")
    ident.append_text(hints)
    return ident


def status_text(
    *,
    expanded: bool,
    busy: bool,
    suggesting: bool = False,
    quit_armed: bool = False,
    follow: bool = True,
    approving: bool = False,
    approval_reason: bool = False,
) -> Text:
    line = status_plain(
        expanded=expanded,
        busy=busy,
        suggesting=suggesting,
        quit_armed=quit_armed,
        follow=follow,
        approving=approving,
        approval_reason=approval_reason,
    )
    renderable = Text()
    parts = line.split("  ·  ")
    for index, part in enumerate(parts):
        if index:
            renderable.append("  ·  ", style="dim")
        if " " in part:
            key, _, label = part.partition(" ")
            renderable.append(key, style="bold")
            renderable.append(" " + label, style="dim")
        else:
            renderable.append(part, style="dim")
    return renderable


def is_shift_enter(event: events.Key) -> bool:
    """True when the user wants a newline.

    Headless / Unix typically send ``shift+enter``. Windows Terminal's
    console API reports the same CR as Enter, so we also look at the
    physical Shift key.
    """
    key = event.key or ""
    if key == "shift+enter" or "shift+enter" in (event.aliases or []):
        return True
    if key == "enter" and _win_shift_down():
        return True
    return False


def _win_shift_down() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.user32.GetKeyState(0x10) & 0x8000)
    except Exception:
        return False


class Composer(TextArea):
    """Multiline prompt: Enter sends, shift+enter inserts a newline."""

    BINDINGS = [
        Binding("ctrl+o", "app.toggle_expand", show=False, priority=True),
        Binding("f1", "app.show_help", show=False, priority=True),
        Binding("ctrl+c", "app.interrupt", show=False, priority=True),
        Binding("shift+tab", "app.cycle_perm", show=False, priority=True),
    ]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("show_line_numbers", False)
        kwargs.setdefault("compact", True)
        kwargs.setdefault("soft_wrap", True)
        kwargs.setdefault("highlight_cursor_line", False)
        kwargs.setdefault("tab_behavior", "indent")
        super().__init__(**kwargs)

    def on_key(self, event: events.Key) -> None:
        text = self.text or ""
        if event.character == "?" and not text.strip():
            self.app.action_show_help()
            event.prevent_default()
            event.stop()
            return
        if is_shift_enter(event):
            self.insert("\n")
            event.prevent_default()
            event.stop()
            return
        if self.app.suggestions_open():
            if event.key == "up":
                self.app.action_suggest_move(-1)
                event.prevent_default()
                event.stop()
                return
            if event.key == "down":
                self.app.action_suggest_move(1)
                event.prevent_default()
                event.stop()
                return
            if event.key == "escape":
                self.app.action_suggest_hide()
                event.prevent_default()
                event.stop()
                return
            if event.key == "tab":
                self.app.action_suggest_accept(submit_slash=False)
                event.prevent_default()
                event.stop()
                return
            if event.key == "enter":
                self.app.action_suggest_accept(submit_slash=True)
                event.prevent_default()
                event.stop()
                return
        if event.key == "tab":
            event.prevent_default()
            event.stop()
            return
        if event.key == "enter":
            self.app.action_submit_composer()
            event.prevent_default()
            event.stop()
            return


class LogView(VerticalScroll):
    """Transcript list. Wheel/keys away from the end unfollow the live tail."""

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:  # noqa: ARG002
        unfollow = getattr(self.app, "unfollow_tail", None)
        if callable(unfollow):
            unfollow()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:  # noqa: ARG002
        follow = getattr(self.app, "follow_if_at_bottom", None)
        if callable(follow):
            self.call_after_refresh(follow)

    def on_key(self, event: events.Key) -> None:
        if event.key in {"up", "pageup", "home"}:
            unfollow = getattr(self.app, "unfollow_tail", None)
            if callable(unfollow):
                unfollow()
        elif event.key in {"down", "pagedown", "end"}:
            follow = getattr(self.app, "follow_if_at_bottom", None)
            if callable(follow):
                self.call_after_refresh(follow)


class SuggestList(OptionList):
    can_focus = False

    def __init__(self, **kwargs) -> None:
        super().__init__(compact=True, markup=False, **kwargs)
        self.items: list[Suggestion] = []
        self.display = False

    def set_items(self, items: list[Suggestion], *, highlighted: int | None = None) -> None:
        self.items = items
        self.clear_options()
        if not items:
            self.display = False
            return
        for index, item in enumerate(items):
            prompt = item.label if not item.detail else f"{item.label}  {item.detail}"
            self.add_option(Option(prompt, id=f"s{index}"))
        if highlighted is None:
            highlighted = 0
        self.highlighted = max(0, min(highlighted, len(items) - 1))
        self.display = True
        # Border eats two rows; without this, 5 commands show as 3 and 3 files as 1.
        self.styles.height = min(len(items), 8) + 2
        self.refresh(layout=True)

    def current(self) -> Suggestion | None:
        if not self.items:
            return None
        index = self.highlighted if self.highlighted is not None else 0
        if index < 0 or index >= len(self.items):
            return None
        return self.items[index]


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "close", show=False),
        Binding("q", "close", show=False),
        Binding("f1", "close", show=False),
        Binding("question_mark", "close", show=False),
    ]

    CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-box {
        width: 72;
        max-width: 90%;
        height: auto;
        max-height: 90%;
        border: round #5a5a5a;
        background: #161616;
        padding: 1 2;
        color: #d0d0d0;
    }
    """

    def compose(self):
        yield Vertical(Static(HELP_TEXT, markup=False), id="help-box")

    def action_close(self) -> None:
        self.dismiss()


class ResumeScreen(ModalScreen[str | None]):
    """Pick a workspace session. Enter restores, Esc cancels."""

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("q", "cancel", show=False),
        Binding("enter", "select", show=False),
    ]

    CSS = """
    ResumeScreen {
        align: center middle;
    }
    #resume-box {
        width: 80;
        max-width: 94%;
        height: auto;
        max-height: 80%;
        border: round #5a5a5a;
        background: #161616;
        padding: 1 1;
        color: #d0d0d0;
    }
    #resume-hint {
        height: 1;
        color: #9aa0a6;
        padding: 0 1 1 1;
    }
    #resume-list {
        height: auto;
        max-height: 16;
        padding: 0 1;
    }
    """

    def __init__(self, items: list[Suggestion], **kwargs) -> None:
        super().__init__(**kwargs)
        self._items = items

    def compose(self):
        listing = OptionList(compact=True, markup=False, id="resume-list")
        yield Vertical(
            Static("恢复会话    ↑↓ 选择  Enter 打开  Esc 取消", id="resume-hint", markup=False),
            listing,
            id="resume-box",
        )

    def on_mount(self) -> None:
        listing = self.query_one("#resume-list", OptionList)
        for index, item in enumerate(self._items):
            prompt = item.label if not item.detail else f"{item.label}  {item.detail}"
            listing.add_option(Option(prompt, id=f"r{index}"))
        if self._items:
            listing.highlighted = 0
            listing.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        listing = self.query_one("#resume-list", OptionList)
        index = listing.highlighted if listing.highlighted is not None else 0
        if index < 0 or index >= len(self._items):
            self.dismiss(None)
            return
        self.dismiss(self._items[index].value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.action_select()


class ApprovalPanel(Vertical):
    """Ask before a side-effecting tool. Docks above the input; the log stays visible."""

    can_focus = True

    BINDINGS = [
        Binding("escape", "escape", show=False, priority=True),
        Binding("n", "deny", show=False, priority=True),
        Binding("y", "allow", show=False, priority=True),
        Binding("s", "allow_session", show=False, priority=True),
        Binding("enter", "select", show=False, priority=True),
        Binding("tab", "reason", show=False, priority=True),
        Binding("ctrl+e", "preview", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    ApprovalPanel {
        display: none;
        height: auto;
        max-height: 22;
        background: #161616;
        border: tall #3a3a3a;
        margin: 0 1;
        padding: 0 1 1 1;
    }
    #approval-hint {
        height: 1;
        color: #9aa0a6;
        padding: 0 1;
    }
    #approval-preview {
        height: auto;
        max-height: 12;
        color: #d0d0d0;
        padding: 0 1 1 1;
    }
    #approval-list {
        height: 3;
        min-height: 3;
        padding: 0 1;
    }
    #approval-reason {
        display: none;
        height: 3;
        background: #161616;
        border: tall #3a3a3a;
        margin: 0 1 0 1;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._on_choice = None
        self._call: ToolCall | None = None
        self._full: Text | None = None
        self._title = ""
        self._feedback = False
        self._choices = (
            ("allow", "允许这次"),
            ("allow_session", "本会话放行此工具"),
            ("deny", "拒绝"),
        )

    @property
    def reason_open(self) -> bool:
        return self._feedback

    def compose(self):
        yield Static("", id="approval-hint", markup=False)
        yield Static("", id="approval-preview", markup=False)
        yield OptionList(compact=True, markup=False, id="approval-list")
        yield Input(placeholder="拒绝原因（可选）", id="approval-reason")

    def set_request(self, call: ToolCall, on_choice) -> None:
        self._on_choice = on_choice
        self._call = call
        self._full = render_approval_full(call)
        self._title = approval_title(call)
        self._close_reason(keep_text=False)
        hint = self._title
        if is_file_preview(call) and approval_has_more(call):
            hint = f"{self._title}    ctrl+e 看全文"
        self.query_one("#approval-hint", Static).update(hint)
        self.query_one("#approval-preview", Static).update(render_approval_summary(call))
        listing = self.query_one("#approval-list", OptionList)
        listing.clear_options()
        for value, label in self._choices:
            listing.add_option(Option(label, id=value))
        listing.highlighted = 0
        self.display = True
        listing.focus()

    def hide_panel(self) -> None:
        self._on_choice = None
        self._call = None
        self._full = None
        self._close_reason(keep_text=False)
        self.display = False

    def focus_choices(self) -> None:
        if self._feedback:
            self.query_one("#approval-reason", Input).focus()
            return
        self.query_one("#approval-list", OptionList).focus()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if self._feedback and action in {"allow", "allow_session", "deny"}:
            return False
        return True

    def _emit(self, value: str, reason: str = "") -> None:
        callback = self._on_choice
        self._on_choice = None
        self._close_reason(keep_text=False)
        self.display = False
        if callback is not None:
            callback(ApprovalDecision.from_raw(ApprovalDecision(value, reason)))

    def action_allow(self) -> None:
        self._emit("allow")

    def action_allow_session(self) -> None:
        self._emit("allow_session")

    def action_deny(self) -> None:
        self._emit("deny")

    def action_escape(self) -> None:
        if self._feedback:
            self._close_reason()
            return
        self._emit("deny")

    def action_reason(self) -> None:
        if self._feedback:
            self._close_reason()
            return
        listing = self.query_one("#approval-list", OptionList)
        listing.highlighted = 2
        self._open_reason()

    def action_preview(self) -> None:
        opener = getattr(self.app, "open_approval_preview", None)
        if not callable(opener) or self._full is None:
            return
        opener(self._title, self._full)

    def action_select(self) -> None:
        if self._feedback:
            reason = self.query_one("#approval-reason", Input).value
            self._emit("deny", reason)
            return
        listing = self.query_one("#approval-list", OptionList)
        index = listing.highlighted if listing.highlighted is not None else 0
        if index < 0 or index >= len(self._choices):
            self._emit("deny")
            return
        self._emit(self._choices[index][0])

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.action_select()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.action_select()

    def _open_reason(self) -> None:
        self._feedback = True
        field = self.query_one("#approval-reason", Input)
        field.display = True
        field.focus()
        refresh = getattr(self.app, "_refresh_status", None)
        if callable(refresh):
            refresh()

    def _close_reason(self, *, keep_text: bool = True) -> None:
        self._feedback = False
        try:
            field = self.query_one("#approval-reason", Input)
        except Exception:
            return
        field.display = False
        if not keep_text:
            field.value = ""
        if self.display:
            self.query_one("#approval-list", OptionList).focus()
        refresh = getattr(self.app, "_refresh_status", None)
        if callable(refresh):
            refresh()


class PreviewScreen(ModalScreen[None]):
    """Full-screen file/diff viewer. Closing returns to the approval panel."""

    BINDINGS = [
        Binding("escape", "close", show=False, priority=True),
        Binding("q", "close", show=False, priority=True),
        Binding("ctrl+e", "close", show=False, priority=True),
        Binding("enter", "ignore", show=False, priority=True),
        Binding("y", "ignore", show=False, priority=True),
        Binding("s", "ignore", show=False, priority=True),
        Binding("n", "ignore", show=False, priority=True),
        Binding("up", "line_up", show=False, priority=True),
        Binding("down", "line_down", show=False, priority=True),
        Binding("k", "line_up", show=False, priority=True),
        Binding("j", "line_down", show=False, priority=True),
        Binding("pageup", "page_up", show=False, priority=True),
        Binding("pagedown", "page_down", show=False, priority=True),
        Binding("home", "top", show=False, priority=True),
        Binding("end", "bottom", show=False, priority=True),
        Binding("g", "top", show=False, priority=True),
        Binding("G", "bottom", show=False, priority=True),
        Binding("shift+g", "bottom", show=False, priority=True),
    ]

    CSS = """
    PreviewScreen {
        background: #0e0e0e;
        layout: vertical;
    }
    #preview-head {
        height: 1;
        color: #e5e7eb;
        text-style: bold;
        padding: 0 2;
        background: #161616;
    }
    #preview-scroll {
        height: 1fr;
        padding: 0 2 0 2;
        scrollbar-size: 1 1;
    }
    #preview-body {
        height: auto;
        color: #d0d0d0;
    }
    #preview-foot {
        height: 1;
        color: #9aa0a6;
        padding: 0 2;
        background: #161616;
    }
    """

    def __init__(self, title: str, body: Text, **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._body = body

    def compose(self):
        yield Static(f" Preview  {self._title}", id="preview-head", markup=False)
        yield VerticalScroll(
            Static(self._body, id="preview-body", markup=False),
            id="preview-scroll",
        )
        yield Static(
            "↑↓ 行  pgup/dn 页  g/G 顶/底  esc 返回确认",
            id="preview-foot",
            markup=False,
        )

    def on_mount(self) -> None:
        self.query_one("#preview-scroll", VerticalScroll).focus()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_ignore(self) -> None:
        return

    def action_line_up(self) -> None:
        self.query_one("#preview-scroll", VerticalScroll).scroll_up(animate=False)

    def action_line_down(self) -> None:
        self.query_one("#preview-scroll", VerticalScroll).scroll_down(animate=False)

    def action_page_up(self) -> None:
        self.query_one("#preview-scroll", VerticalScroll).scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        self.query_one("#preview-scroll", VerticalScroll).scroll_page_down(animate=False)

    def action_top(self) -> None:
        self.query_one("#preview-scroll", VerticalScroll).scroll_home(animate=False)

    def action_bottom(self) -> None:
        self.query_one("#preview-scroll", VerticalScroll).scroll_end(animate=False)
