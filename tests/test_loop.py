import threading
import time
from pathlib import Path
from threading import Event

from anvil.agent.context import ContextManager
from anvil.agent.loop import Agent, _fingerprint
from anvil.config import Config
from anvil.llm.types import LLMResponse, ToolCall, Usage
from anvil.session import Session
from anvil.tools import TodoStore, build_tools


class ScriptedLLM:
    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)

    def complete(self, messages, tools, *, on_delta=None, stream=True, **_kwargs):
        if not self.script:
            raise AssertionError("unexpected extra model call")
        return self.script.pop(0)


def _config(workspace: Path) -> Config:
    return Config(
        api_key="test",
        base_url="http://localhost",
        model="scripted",
        thinking=False,
        reasoning_effort="low",
        max_turns=8,
        max_tokens=256,
        context_budget=20_000,
        request_timeout=5,
        shell_timeout=5,
        workspace=workspace,
    )


def test_agent_edits_file_and_stops(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="read_file",
                        arguments={"path": "app.py"},
                        arguments_raw='{"path":"app.py"}',
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="2",
                        name="edit_file",
                        arguments={
                            "path": "app.py",
                            "old_string": "value = 1",
                            "new_string": "value = 2",
                        },
                        arguments_raw="{}",
                    )
                ],
            ),
            LLMResponse(content="Updated app.py.", reasoning_content="", tool_calls=[]),
        ]
    )
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("Set value to 2")
    assert result.stop_reason == "completed"
    assert result.turns == 3
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "value = 2\n"
    assert result.usage.prompt_tokens == 10


def test_cancel_during_model_call_stops(tmp_path: Path) -> None:
    started = Event()

    class SlowLLM:
        def complete(self, messages, tools, *, on_delta=None, stream=True, cancel=None, **_kwargs):
            started.set()
            for _ in range(200):
                if cancel is not None and cancel.is_set():
                    class LLMCancelled(Exception):
                        pass

                    raise LLMCancelled("cancelled")
                time.sleep(0.01)
            raise AssertionError("model call was not cancelled")

    agent = Agent(
        _config(tmp_path),
        SlowLLM(),
        build_tools(tmp_path, TodoStore(), 5),
        ContextManager(20_000),
    )
    holder: dict[str, object] = {}

    def run() -> None:
        holder["result"] = agent.run("hello")

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(2)
    agent.session.cancel.set()
    worker.join(3)
    assert not worker.is_alive()
    result = holder["result"]
    assert result.stop_reason == "cancelled"


def test_unknown_tool_does_not_crash(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[ToolCall(id="1", name="explode", arguments={}, arguments_raw="{}")],
            ),
            LLMResponse(content="I could not explode.", reasoning_content="", tool_calls=[]),
        ]
    )
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("do it")
    assert result.stop_reason == "completed"
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert "unknown tool" in (tool_messages[0].content or "")
    assert "unknown_tool" in (tool_messages[0].content or "")


def test_repeated_identical_calls_get_a_progress_nudge(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    read = LLMResponse(
        content="",
        reasoning_content="",
        tool_calls=[
            ToolCall(
                id="r",
                name="read_file",
                arguments={"path": "app.py"},
                arguments_raw='{"path":"app.py"}',
            )
        ],
    )
    llm = ScriptedLLM(
        [read, read, read, LLMResponse(content="I should try something else.", reasoning_content="", tool_calls=[])]
    )
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("read it")
    assert result.stop_reason == "completed"
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert any("[progress]" in (m.content or "") for m in tool_messages)


def test_cancel_stops_before_next_tool(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    cancel = Event()

    class CancellingLLM(ScriptedLLM):
        def complete(self, messages, tools, *, on_delta=None, stream=True, **kwargs):
            response = super().complete(
                messages, tools, on_delta=on_delta, stream=stream, **kwargs
            )
            cancel.set()
            return response

    llm = CancellingLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="read_file",
                        arguments={"path": "app.py"},
                        arguments_raw='{"path":"app.py"}',
                    )
                ],
            )
        ]
    )
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("read it", cancel=cancel)
    assert result.stop_reason == "cancelled"
    assert any(m.role == "user" for m in agent.messages)


def test_fifth_identical_call_stops_with_no_progress(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    read = LLMResponse(
        content="",
        reasoning_content="",
        tool_calls=[
            ToolCall(
                id="r",
                name="read_file",
                arguments={"path": "app.py"},
                arguments_raw='{"path":"app.py"}',
            )
        ],
    )
    llm = ScriptedLLM([read, read, read, read, read])
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("read it")
    assert result.stop_reason == "no_progress"
    assert result.turns == 5
    assert llm.script == []
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert any("repeated 5 times" in (m.content or "") for m in tool_messages)


def test_new_user_message_resets_repeat_counts(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    read = LLMResponse(
        content="",
        reasoning_content="",
        tool_calls=[
            ToolCall(
                id="r",
                name="read_file",
                arguments={"path": "app.py"},
                arguments_raw='{"path":"app.py"}',
            )
        ],
    )
    done = LLMResponse(content="ok", reasoning_content="", tool_calls=[])
    llm = ScriptedLLM([read, read, done, read, read, read, done])
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    first = agent.run("first")
    assert first.stop_reason == "completed"
    second = agent.run("second")
    assert second.stop_reason == "completed"
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert sum(1 for m in tool_messages if "[progress]" in (m.content or "")) == 1


def test_empty_content_without_tools_completes(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [LLMResponse(content="", reasoning_content="", tool_calls=[], finish_reason="stop")]
    )
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("hi")
    assert result.stop_reason == "completed"
    assert result.turns == 1


def test_finish_reason_length_stops_even_without_tools(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="partial",
                reasoning_content="",
                tool_calls=[],
                finish_reason="length",
            )
        ]
    )
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("go")
    assert result.stop_reason == "length"
    assert result.final_text == "partial"


def test_finish_reason_tool_calls_without_calls_completes(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="done",
                reasoning_content="",
                tool_calls=[],
                finish_reason="tool_calls",
            )
        ]
    )
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("go")
    assert result.stop_reason == "completed"


def test_finish_reason_stop_still_runs_tools(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="read_file",
                        arguments={"path": "app.py"},
                        arguments_raw='{"path":"app.py"}',
                    )
                ],
                finish_reason="stop",
            ),
            LLMResponse(content="read it", reasoning_content="", tool_calls=[], finish_reason="stop"),
        ]
    )
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("read")
    assert result.stop_reason == "completed"
    assert any(m.role == "tool" for m in agent.messages)


def test_edit_without_prior_read_reports_stale_read(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="edit_file",
                        arguments={
                            "path": "app.py",
                            "old_string": "value = 1",
                            "new_string": "value = 2",
                        },
                        arguments_raw="{}",
                    )
                ],
            ),
            LLMResponse(content="need to read first", reasoning_content="", tool_calls=[]),
        ]
    )
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("edit")
    assert result.stop_reason == "completed"
    tool = next(m for m in agent.messages if m.role == "tool")
    assert "stale_read" in (tool.content or "")
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_consecutive_tool_errors_stop(tmp_path: Path) -> None:
    script = [
        LLMResponse(
            content="",
            reasoning_content="",
            tool_calls=[
                ToolCall(
                    id=str(index),
                    name=f"explode_{index}",
                    arguments={},
                    arguments_raw="{}",
                )
            ],
        )
        for index in range(5)
    ]
    llm = ScriptedLLM(script)
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("explode")
    assert result.stop_reason == "tool_errors"
    assert result.turns == 5


def test_max_turns_stops(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    object.__setattr__(cfg, "max_turns", 2)
    read = LLMResponse(
        content="",
        reasoning_content="",
        tool_calls=[
            ToolCall(id="1", name="list_dir", arguments={}, arguments_raw="{}")
        ],
    )
    llm = ScriptedLLM([read, read, read])
    session = Session(cfg)
    agent = Agent(cfg, llm, build_tools(tmp_path, session.todo, 5), ContextManager(20_000), session=session)
    result = agent.run("loop")
    assert result.stop_reason == "max_turns"
    assert result.turns == 2


def test_python_dash_c_commands_share_a_fingerprint() -> None:
    first = ToolCall(id="1", name="run_shell", arguments={"command": 'python -c "print(1)"'})
    second = ToolCall(id="2", name="run_shell", arguments={"command": "python  -c  'print(999)'"})
    third = ToolCall(id="3", name="run_shell", arguments={"command": "python -m unittest"})
    assert _fingerprint(first) == _fingerprint(second)
    assert _fingerprint(first) != _fingerprint(third)


def test_five_different_python_c_calls_stop_with_no_progress(tmp_path: Path) -> None:
    calls = [
        LLMResponse(
            content="",
            reasoning_content="",
            tool_calls=[
                ToolCall(
                    id=str(index),
                    name="run_shell",
                    arguments={"command": f'python -c "print({index})"'},
                )
            ],
        )
        for index in range(5)
    ]
    llm = ScriptedLLM(calls)
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    result = agent.run("probe")
    assert result.stop_reason == "no_progress"
    assert result.turns == 5
