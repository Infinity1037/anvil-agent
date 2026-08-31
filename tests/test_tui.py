from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from anvil.agent.context import ContextSnapshot
from anvil.agent.loop import CompactResult
from anvil.config import apply_effort_token, effort_label
from anvil.events import AgentEvent
from anvil.tui.app import AnvilApp
from anvil.llm.types import ToolCall
from anvil.tui.chrome import (
    ApprovalPanel,
    Composer,
    PreviewScreen,
    ResumeScreen,
    SuggestList,
    status_plain,
)
from anvil.tui.widgets import AssistantBlock, FoldBlock, NoticeBlock, UserBlock


def _composer(app: AnvilApp) -> Composer:
    return app.query_one("#composer", Composer)


def _set_composer(app: AnvilApp, text: str) -> None:
    composer = _composer(app)
    composer.text = text
    composer.cursor_location = (0, len(text.split("\n")[-1])) if "\n" not in text else composer.cursor_location


def _plain(widget) -> str:
    visual = getattr(widget, "visual", None)
    if visual is not None and hasattr(visual, "plain"):
        return str(visual.plain)
    content = getattr(widget, "content", None)
    if content is not None:
        return str(getattr(content, "plain", content))
    return str(widget)


class FakeSession:
    def __init__(self) -> None:
        self.cancel = Event()
        self.thinking = True
        self.reasoning_effort = "max"
        self.permission_mode = "ask"
        self.session_allow: set[str] = set()
        self.id = "test-session"

    def set_effort(self, token: str) -> str:
        self.thinking, self.reasoning_effort = apply_effort_token(
            self.thinking, self.reasoning_effort, token
        )
        return self.effort_status()

    def effort_status(self) -> str:
        return f"effort {effort_label(self.thinking, self.reasoning_effort)}"

    def set_permission(self, token: str) -> str:
        raw = (token or "").strip().lower()
        if raw in {"yolo", "yes"}:
            raw = "auto"
        if raw not in {"ask", "auto"}:
            raise ValueError("use /perm ask | auto")
        self.permission_mode = raw
        if raw == "ask":
            self.session_allow = set()
        return self.permission_status()

    def permission_status(self) -> str:
        return f"perm {self.permission_mode}"

    def allow_tool(self, name: str) -> None:
        self.session_allow.add(name)


class FakeAgent:
    def __init__(self, workspace: Path, script: list[tuple[str, dict]]) -> None:
        self.config = SimpleNamespace(
            workspace=workspace,
            model="scripted",
            thinking=True,
            reasoning_effort="max",
        )
        self.session = FakeSession()
        self.usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        self.messages: list[object] = []
        self._script = script
        self.compact_calls: list[str] = []
        self.compact_result = CompactResult("nothing_to_compact", 1200, 1200)

    def reset(self) -> None:
        self.messages.clear()

    def new_session(self) -> None:
        self.reset()

    def run(self, task: str, on_event=None, cancel=None) -> None:
        self.messages.append(task)
        for kind, payload in self._script:
            if cancel is not None and cancel.is_set():
                if on_event:
                    on_event(AgentEvent("ended", {"stop_reason": "cancelled", "turns": 1}))
                return
            if on_event:
                on_event(AgentEvent(kind, payload))

    def context_snapshot(self) -> ContextSnapshot:
        return ContextSnapshot(
            estimated_tokens=1200,
            budget=20_000,
            history_messages=len(self.messages),
            view_messages=len(self.messages),
            calibrated=True,
        )

    def compact(self, instruction="", on_event=None, cancel=None) -> CompactResult:
        self.compact_calls.append(instruction)
        if self.compact_result.status == "compacted" and on_event:
            on_event(
                AgentEvent(
                    "compact",
                    {
                        "before_tokens": self.compact_result.before_tokens,
                        "after_tokens": self.compact_result.after_tokens,
                        "covered_count": self.compact_result.covered_count,
                        "semantic": True,
                        "trigger": "manual",
                    },
                )
            )
        return self.compact_result


THINKING = (
    "The user just said hello. This is a simple greeting.\n"
    "I should respond politely and briefly.\n"
    "There is no task yet so I should not explore the workspace.\n"
    "Just greet back and ask what they need.\n"
)

WRITE_CONTENT = "\n".join(f"line_{i} = {i}" for i in range(1, 25))
WRITE_OUTPUT = f"Wrote 400 bytes (24 lines) to snake.py."


def _greeting_script() -> list[tuple[str, dict]]:
    return [
        ("turn", {"turn": 1, "max_turns": 8}),
        ("delta", {"kind": "reasoning", "text": THINKING}),
        ("delta", {"kind": "content", "text": "Hello! I am Anvil."}),
        (
            "assistant",
            {
                "content": "Hello! I am Anvil.",
                "reasoning": THINKING,
                "tool_calls": [],
            },
        ),
        ("ended", {"stop_reason": "completed", "turns": 1}),
    ]


def _write_script() -> list[tuple[str, dict]]:
    return [
        ("turn", {"turn": 1, "max_turns": 8}),
        ("delta", {"kind": "reasoning", "text": THINKING}),
        (
            "assistant",
            {
                "content": "",
                "reasoning": THINKING,
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "write_file",
                        "arguments": {"path": "snake.py", "content": WRITE_CONTENT},
                    }
                ],
            },
        ),
        (
            "tool_result",
            {"id": "c1", "name": "write_file", "ok": True, "content": WRITE_OUTPUT},
        ),
        ("ended", {"stop_reason": "completed", "turns": 1}),
    ]


async def _wait_idle(app: AnvilApp, pilot, timeout: float = 4.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not app._busy and list(app.query(FoldBlock)):
            await pilot.pause(0.05)
            return
        await pilot.pause(0.05)
    raise AssertionError("TUI stayed busy or never mounted a fold card")


def test_ctrl_o_toggles_once_from_the_composer(tmp_path: Path) -> None:
    asyncio.run(_ctrl_o_toggles_once_from_the_composer(tmp_path))


async def _ctrl_o_toggles_once_from_the_composer(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        _composer(app).focus()
        assert app._expanded is False
        await pilot.press("ctrl+o")
        await pilot.pause(0.05)
        assert app._expanded is True
        await pilot.press("ctrl+o")
        await pilot.pause(0.05)
        assert app._expanded is False


def test_question_mark_opens_help_when_input_is_empty(tmp_path: Path) -> None:
    asyncio.run(_question_mark_opens_help_when_input_is_empty(tmp_path))


async def _question_mark_opens_help_when_input_is_empty(tmp_path: Path) -> None:
    from anvil.tui.chrome import HelpScreen

    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        _composer(app).focus()
        await pilot.press("question_mark")
        await pilot.pause(0.1)
        assert isinstance(app.screen, HelpScreen)
        assert _composer(app).text == ""
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not isinstance(app.screen, HelpScreen)


def test_status_bar_is_visible_on_launch(tmp_path: Path) -> None:
    asyncio.run(_status_bar_is_visible_on_launch(tmp_path))


async def _status_bar_is_visible_on_launch(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause(0.05)
        plain = _plain(app.query_one("#status"))
        assert "ctrl+o" in plain.lower()
        assert "展开" in plain
        assert "scripted" in plain
        assert "ask" in plain
        assert "ctx" in plain
        assert list(app.query("#header")) == []


def test_context_command_shows_budget_details(tmp_path: Path) -> None:
    asyncio.run(_context_command_shows_budget_details(tmp_path))


async def _context_command_shows_budget_details(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(110, 26)) as pilot:
        _set_composer(app, "/context")
        await pilot.press("enter")
        await pilot.pause(0.05)
        notices = [_plain(child) for child in app.query(NoticeBlock)]
        assert any("context ≈6%" in text for text in notices)
        assert any("20k" in text and "checkpoint not active" in text for text in notices)


def test_compact_command_passes_focus_without_adding_chat_message(tmp_path: Path) -> None:
    asyncio.run(_compact_command_passes_focus_without_adding_chat_message(tmp_path))


async def _compact_command_passes_focus_without_adding_chat_message(tmp_path: Path) -> None:
    agent = FakeAgent(tmp_path, [])
    app = AnvilApp(agent)
    async with app.run_test(size=(110, 26)) as pilot:
        _set_composer(app, "/compact 保留测试结果")
        await pilot.press("enter")
        for _ in range(80):
            if not app._busy:
                break
            await pilot.pause(0.05)
        assert app._busy is False
        assert agent.compact_calls == ["保留测试结果"]
        assert agent.messages == []
        notices = [_plain(child) for child in app.query(NoticeBlock)]
        assert any("没有足够" in text for text in notices)


def test_ctrl_o_expands_thinking_on_the_same_card(tmp_path: Path) -> None:
    asyncio.run(_ctrl_o_expands_thinking_on_the_same_card(tmp_path))


async def _ctrl_o_expands_thinking_on_the_same_card(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, _greeting_script()))
    async with app.run_test(size=(100, 36)) as pilot:
        _set_composer(app, "hello")
        await pilot.press("enter")
        await _wait_idle(app, pilot)

        bar = _plain(app.query_one("#status"))
        assert "ctrl+o" in bar.lower()

        log_kids = list(app.query_one("#log").children)
        kinds = [type(child).__name__ for child in log_kids]
        assert "UserBlock" in kinds
        assert kinds.index("FoldBlock") < kinds.index("AssistantBlock")
        thinking = next(child for child in log_kids if isinstance(child, FoldBlock))
        assert thinking.kind == "thinking"
        assert thinking.expanded is False
        collapsed = thinking.shown()
        assert "simple greeting" in collapsed
        assert "Ctrl+O to expand" in collapsed
        assert "Just greet back" not in collapsed
        assert "Hello! I am Anvil." in app.query_one(AssistantBlock)._text
        reply = _plain(app.query_one(AssistantBlock))
        assert reply.startswith("● ")
        assert "Hello! I am Anvil." in reply

        before_id = id(thinking)
        before_count = len(log_kids)
        await pilot.press("ctrl+o")
        await pilot.pause(0.05)

        log_kids_after = list(app.query_one("#log").children)
        assert len(log_kids_after) == before_count
        thinking_after = next(child for child in log_kids_after if isinstance(child, FoldBlock))
        assert id(thinking_after) == before_id
        assert thinking_after.expanded is True
        shown = thinking_after.shown()
        assert "Just greet back" in shown
        assert "Ctrl+O to expand" not in shown
        assert "收起" in _plain(app.query_one("#status"))


def test_ctrl_o_expands_tool_output_on_the_same_card(tmp_path: Path) -> None:
    asyncio.run(_ctrl_o_expands_tool_output_on_the_same_card(tmp_path))


async def _ctrl_o_expands_tool_output_on_the_same_card(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, _write_script()))
    async with app.run_test(size=(100, 40)) as pilot:
        _set_composer(app, "fix ledger")
        await pilot.press("enter")
        await _wait_idle(app, pilot)

        blocks = list(app.query(FoldBlock))
        kinds = [block.kind for block in blocks]
        assert kinds == ["thinking", "write_file"]
        write = blocks[1]
        assert write.expanded is False
        plain = _plain(write)
        assert "Used Write" in plain
        assert "snake.py" in plain
        assert "24 lines" in plain
        assert "Wrote 400 bytes" not in write.shown()
        assert "line_1 =" in write.shown()
        assert "line_10 =" in write.shown()
        assert "line_20 =" not in write.shown()
        assert "Ctrl+O to expand" in write.shown()

        await pilot.press("ctrl+o")
        await pilot.pause(0.05)

        still = list(app.query(FoldBlock))
        assert [id(block) for block in still] == [id(block) for block in blocks]
        assert still[1].expanded is True
        assert "line_20 =" in still[1].shown()
        assert "Ctrl+O to expand" not in still[1].shown()
        assert list(app.query(UserBlock))


def test_wide_layout_folds_short_greeting_thinking(tmp_path: Path) -> None:
    asyncio.run(_wide_layout_folds_short_greeting_thinking(tmp_path))


async def _wide_layout_folds_short_greeting_thinking(tmp_path: Path) -> None:
    thinking = ("Need a plan for this workspace change. " * 24).strip() + " No need to call tools yet."
    script = [
        ("turn", {"turn": 1, "max_turns": 8}),
        ("delta", {"kind": "reasoning", "text": thinking}),
        ("delta", {"kind": "content", "text": "你好！我是 **Anvil**。"}),
        (
            "assistant",
            {"content": "你好！我是 **Anvil**。", "reasoning": thinking, "tool_calls": []},
        ),
        ("ended", {"stop_reason": "completed", "turns": 1}),
    ]
    app = AnvilApp(FakeAgent(tmp_path, script))
    async with app.run_test(size=(160, 32)) as pilot:
        _set_composer(app, "你好")
        await pilot.press("enter")
        await _wait_idle(app, pilot)
        block = next(child for child in app.query(FoldBlock) if child.kind == "thinking")
        assert block.has_hidden_lines()
        assert "Ctrl+O to expand" in block.shown()
        await pilot.press("ctrl+o")
        await pilot.pause(0.05)
        assert block.expanded is True
        assert "tools yet" in block.shown()
        assert "收起" in _plain(app.query_one("#status"))


def test_agent_error_is_shown_in_the_log(tmp_path: Path) -> None:
    asyncio.run(_agent_error_is_shown_in_the_log(tmp_path))


async def _agent_error_is_shown_in_the_log(tmp_path: Path) -> None:
    class Boom(FakeAgent):
        def run(self, task: str, on_event=None, cancel=None) -> None:
            raise RuntimeError("simulated failure")

    app = AnvilApp(Boom(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_composer(app, "hello")
        await pilot.press("enter")
        import time

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if not app._busy:
                break
            await pilot.pause(0.05)
        await pilot.pause(0.05)
        texts = [_plain(child) for child in app.query_one("#log").children]
        joined = "\n".join(texts)
        assert "simulated failure" in joined
        assert app._busy is False


def test_real_agent_events_render_thinking_above_reply(tmp_path: Path) -> None:
    asyncio.run(_real_agent_events_render_thinking_above_reply(tmp_path))


async def _real_agent_events_render_thinking_above_reply(tmp_path: Path) -> None:
    from anvil.agent.context import ContextManager
    from anvil.agent.loop import Agent
    from anvil.config import Config
    from anvil.llm.types import LLMResponse, Usage
    from anvil.session import Session
    from anvil.tools import TodoStore, build_tools

    class StreamingLLM:
        def __init__(self) -> None:
            self.script = [
                LLMResponse(
                    content="Hello! I am Anvil.",
                    reasoning_content=THINKING,
                    tool_calls=[],
                    usage=Usage(prompt_tokens=4, completion_tokens=6, total_tokens=10),
                )
            ]

        def complete(self, messages, tools, *, on_delta=None, stream=True, **_kwargs):
            response = self.script.pop(0)
            if on_delta:
                if response.reasoning_content:
                    on_delta("reasoning", response.reasoning_content)
                if response.content:
                    on_delta("content", response.content)
            return response

    config = Config(
        api_key="test",
        base_url="http://localhost",
        model="scripted",
        thinking=True,
        reasoning_effort="low",
        max_turns=4,
        max_tokens=256,
        context_budget=20_000,
        request_timeout=5,
        shell_timeout=5,
        workspace=tmp_path,
    )
    agent = Agent(
        config,
        StreamingLLM(),
        build_tools(tmp_path, TodoStore(), 5),
        ContextManager(20_000),
        session=Session(config),
    )
    app = AnvilApp(agent)
    async with app.run_test(size=(100, 32)) as pilot:
        _set_composer(app, "hello")
        await pilot.press("enter")
        await _wait_idle(app, pilot)
        kinds = [type(child).__name__ for child in app.query_one("#log").children]
        assert kinds.index("FoldBlock") < kinds.index("AssistantBlock")
        thinking = next(child for child in app.query(FoldBlock) if child.kind == "thinking")
        assert "Ctrl+O to expand" in thinking.shown()
        assert "Hello! I am Anvil." in app.query_one(AssistantBlock)._text


def test_idle_ctrl_c_does_not_quit_immediately(tmp_path: Path) -> None:
    asyncio.run(_idle_ctrl_c_does_not_quit_immediately(tmp_path))


async def _idle_ctrl_c_does_not_quit_immediately(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause(0.05)
        assert app.is_running
        assert app._quit_armed is True
        status = _plain(app.query_one("#status"))
        assert "ctrl+c" in status.lower() or "退出" in status


def test_header_shows_effort(tmp_path: Path) -> None:
    asyncio.run(_header_shows_effort(tmp_path))


async def _header_shows_effort(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause(0.05)
        footer = _plain(app.query_one("#status"))
        assert "scripted" in footer
        assert "max" in footer
        assert "ask" in footer


def test_effort_command_opens_picker_instead_of_dumping_status(tmp_path: Path) -> None:
    asyncio.run(_effort_command_opens_picker(tmp_path))


async def _effort_command_opens_picker(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_composer(app, "/effort")
        app._refresh_suggestions()
        await pilot.press("enter")
        await pilot.pause(0.05)
        listing = app.query_one("#suggest", SuggestList)
        assert listing.display is True
        values = [item.value for item in listing.items]
        assert values == ["off", "low", "high", "max"]
        notices = [_plain(child) for child in app.query(NoticeBlock)]
        assert not any(text.strip() == "effort max" for text in notices)
        listing.highlighted = 2
        await pilot.press("enter")
        await pilot.pause(0.05)
        assert app.agent.session.reasoning_effort == "high"
        assert app.agent.session.thinking is True
        footer = _plain(app.query_one("#status"))
        assert "high" in footer
        notices = [_plain(child) for child in app.query(NoticeBlock)]
        assert any("high" in text for text in notices)


def test_perm_command_opens_picker(tmp_path: Path) -> None:
    asyncio.run(_perm_command_opens_picker(tmp_path))


async def _perm_command_opens_picker(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_composer(app, "/perm")
        app._refresh_suggestions()
        await pilot.press("enter")
        await pilot.pause(0.05)
        listing = app.query_one("#suggest", SuggestList)
        assert listing.display is True
        values = [item.value for item in listing.items]
        assert values == ["ask", "auto"]
        listing.highlighted = 1
        await pilot.press("enter")
        await pilot.pause(0.05)
        assert app.agent.session.permission_mode == "auto"
        footer = _plain(app.query_one("#status"))
        assert "auto" in footer


def test_approval_dock_enter_allows_and_keeps_log(tmp_path: Path) -> None:
    asyncio.run(_approval_dock_enter_allows_and_keeps_log(tmp_path))


async def _approval_dock_enter_allows_and_keeps_log(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    future: Future = Future()
    body = "\n".join(f"line_{i}" for i in range(40))
    call = ToolCall(id="1", name="write_file", arguments={"path": "w.py", "content": body})

    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause(0.05)
        app._open_approval(call, future)
        await pilot.pause(0.05)
        panel = app.query_one("#approval", ApprovalPanel)
        assert panel.display is True
        assert app.query_one("#composer", Composer).display is False
        assert "等待确认" in _plain(app.query_one("#status"))
        preview = _plain(panel.query_one("#approval-preview"))
        assert "line_0" in preview
        assert "line_39" not in preview
        assert "ctrl+e" in preview.lower()
        listing = panel.query_one("#approval-list")
        assert listing.option_count == 3
        assert listing.get_option_at_index(2).prompt == "拒绝"
        await pilot.press("enter")
        await pilot.pause(0.05)
        decision = future.result(timeout=1)
        assert getattr(decision, "verdict", decision) == "allow"
        assert panel.display is False
        assert app.query_one("#composer", Composer).display is True


def test_approval_ctrl_e_opens_full_preview_without_deciding(tmp_path: Path) -> None:
    asyncio.run(_approval_ctrl_e_opens_full_preview_without_deciding(tmp_path))


async def _approval_ctrl_e_opens_full_preview_without_deciding(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    future: Future = Future()
    body = "\n".join(f"line_{i}" for i in range(40))
    call = ToolCall(id="1", name="write_file", arguments={"path": "w.py", "content": body})

    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause(0.05)
        app._open_approval(call, future)
        await pilot.pause(0.05)
        await pilot.press("ctrl+e")
        await pilot.pause(0.05)
        assert isinstance(app.screen, PreviewScreen)
        assert "line_39" in _plain(app.screen.query_one("#preview-body"))
        assert not future.done()
        await pilot.press("escape")
        await pilot.pause(0.05)
        assert not isinstance(app.screen, PreviewScreen)
        panel = app.query_one("#approval", ApprovalPanel)
        assert panel.display is True
        assert not future.done()
        await pilot.press("escape")
        await pilot.pause(0.05)
        decision = future.result(timeout=1)
        assert getattr(decision, "verdict", decision) == "deny"
        assert getattr(decision, "reason", "") == ""


def test_approval_tab_sends_deny_reason(tmp_path: Path) -> None:
    asyncio.run(_approval_tab_sends_deny_reason(tmp_path))


async def _approval_tab_sends_deny_reason(tmp_path: Path) -> None:
    from textual.widgets import Input

    app = AnvilApp(FakeAgent(tmp_path, []))
    future: Future = Future()
    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"path": "snake.py", "content": "print(1)\n"},
    )

    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause(0.05)
        app._open_approval(call, future)
        await pilot.pause(0.05)
        await pilot.press("tab")
        await pilot.pause(0.05)
        field = app.query_one("#approval-reason", Input)
        assert field.display is True
        field.value = "这是账本，不要写贪吃蛇"
        await pilot.press("enter")
        await pilot.pause(0.05)
        decision = future.result(timeout=1)
        assert getattr(decision, "verdict", decision) == "deny"
        assert "这是账本" in getattr(decision, "reason", "")


def test_follow_log_respects_pin(tmp_path: Path) -> None:
    asyncio.run(_follow_log_respects_pin(tmp_path))


async def _follow_log_respects_pin(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause(0.05)
        app._pin_bottom = False
        log = app.query_one("#log")
        before = log.scroll_y
        app._follow_log()
        assert log.scroll_y == before
        app.unfollow_tail()
        assert app._pin_bottom is False
        app._pin_bottom = True
        app._follow_log()
        app.unfollow_tail()
        assert app._pin_bottom is False
        app._apply_event(AgentEvent("delta", {"kind": "reasoning", "text": "still thinking "}))
        assert app._pin_bottom is False


def test_slash_menu_filters_commands(tmp_path: Path) -> None:
    asyncio.run(_slash_menu_filters_commands(tmp_path))


async def _slash_menu_filters_commands(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_composer(app, "/")
        app._refresh_suggestions()
        listing = app.query_one("#suggest", SuggestList)
        assert listing.display is True
        labels = [item.label for item in listing.items]
        assert "/help" in labels
        assert "/clear" in labels
        assert "/resume" in labels
        assert "/exit" in labels
        assert "/new" not in labels
        assert "/quit" not in labels
        assert "/reset" not in labels
        await pilot.pause(0.05)
        assert listing.size.height >= 5
        _set_composer(app, "/ex")
        app._refresh_suggestions()
        labels = [item.label for item in listing.items]
        assert "/expand" in labels
        assert "/help" not in labels
        await pilot.press("enter")
        await pilot.pause(0.05)
        assert app._expanded is True
        assert _composer(app).text == ""


def test_resume_command_opens_session_picker(tmp_path: Path) -> None:
    asyncio.run(_resume_command_opens_session_picker(tmp_path))


async def _resume_command_opens_session_picker(tmp_path: Path) -> None:
    from anvil.config import Config
    from anvil.llm.types import Message
    from anvil.session import Session

    cfg = Config(
        api_key="test",
        base_url="http://localhost",
        model="scripted",
        thinking=True,
        reasoning_effort="max",
        max_turns=4,
        max_tokens=256,
        context_budget=20_000,
        request_timeout=5,
        shell_timeout=5,
        workspace=tmp_path,
    )
    session = Session(cfg)
    session.append(Message(role="user", content="fix the ledger"))
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_composer(app, "/re")
        app._refresh_suggestions()
        labels = [item.label for item in app.query_one("#suggest", SuggestList).items]
        assert labels == ["/resume"]
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert isinstance(app.screen, ResumeScreen)
        assert any("fix the ledger" in item.label for item in app.screen._items)
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not isinstance(app.screen, ResumeScreen)


def test_at_menu_lists_workspace_files(tmp_path: Path) -> None:
    asyncio.run(_at_menu_lists_workspace_files(tmp_path))


async def _at_menu_lists_workspace_files(tmp_path: Path) -> None:
    (tmp_path / "ledger.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("y\n", encoding="utf-8")
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_composer(app, "@")
        app._refresh_suggestions()
        listing = app.query_one("#suggest", SuggestList)
        values = [item.value for item in listing.items]
        assert "ledger.py" in values
        assert "notes.md" in values
        await pilot.pause(0.05)
        assert listing.size.height >= len(values)
        _set_composer(app, "@led")
        app._refresh_suggestions()
        listing = app.query_one("#suggest", SuggestList)
        values = [item.value for item in listing.items]
        assert "ledger.py" in values
        assert "notes.md" not in values
        await pilot.press("tab")
        await pilot.pause(0.05)
        assert "@ledger.py" in _composer(app).text


def test_shift_enter_inserts_newline(tmp_path: Path) -> None:
    asyncio.run(_shift_enter_inserts_newline(tmp_path))


async def _shift_enter_inserts_newline(tmp_path: Path) -> None:
    app = AnvilApp(FakeAgent(tmp_path, []))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_composer(app, "hello")
        _composer(app).cursor_location = (0, 5)
        await pilot.press("shift+enter")
        await pilot.pause(0.05)
        assert "\n" in _composer(app).text
