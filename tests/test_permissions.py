from pathlib import Path

from anvil.agent.context import ContextManager
from anvil.agent.loop import Agent
from anvil.agent.permissions import (
    ApprovalDecision,
    needs_ask,
    parse_permission,
    preview_call,
)
from anvil.config import Config
from anvil.llm.types import LLMResponse, ToolCall
from anvil.session import Session
from anvil.tools import TodoStore, build_tools


class ScriptedLLM:
    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)

    def complete(self, messages, tools, *, on_delta=None, stream=True, **_kwargs):
        if not self.script:
            raise AssertionError("unexpected extra model call")
        return self.script.pop(0)


class ScriptedApprover:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.seen: list[str] = []

    def decide(self, call, cancel=None) -> str:
        self.seen.append(call.name)
        if not self.answers:
            raise AssertionError("unexpected extra approval")
        return self.answers.pop(0)


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


def test_needs_ask_skips_reads_and_auto_mode() -> None:
    assert needs_ask("ask", "read_file") is False
    assert needs_ask("ask", "load_skill") is False
    assert needs_ask("ask", "edit_file") is True
    assert needs_ask("auto", "edit_file") is False
    assert needs_ask("ask", "edit_file", {"edit_file"}) is False


def test_parse_permission_aliases() -> None:
    assert parse_permission("ASK") == "ask"
    assert parse_permission("yolo") == "auto"


def test_preview_shows_edit_and_shell() -> None:
    edit = ToolCall(
        id="1",
        name="edit_file",
        arguments={"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"},
    )
    text = preview_call(edit)
    assert "a.py" in text
    assert "x = 1" in text
    assert "x = 2" in text
    shell = ToolCall(id="2", name="run_shell", arguments={"command": "pytest -q"})
    assert preview_call(shell) == "pytest -q"
    body = "\n".join(f"line_{i}" for i in range(40))
    write = ToolCall(id="3", name="write_file", arguments={"path": "w.py", "content": body})
    summary = preview_call(write)
    assert "line_0" in summary
    assert "line_39" not in summary
    assert "more lines" in summary
    full = preview_call(write, max_lines=None)
    assert "line_0" in full
    assert "line_39" in full
    assert "more lines" not in full


def test_denied_edit_does_not_change_file(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
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
                    )
                ],
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
                    )
                ],
            ),
            LLMResponse(content="stopped", reasoning_content="", tool_calls=[]),
        ]
    )
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    agent.session.permission_mode = "ask"
    agent.approver = ScriptedApprover(["deny"])
    result = agent.run("change it")
    assert result.stop_reason == "completed"
    assert target.read_text(encoding="utf-8") == "value = 1\n"
    denied = [m for m in agent.messages if m.role == "tool" and "rejected" in (m.content or "")]
    assert denied
    assert "permission_denied" in (denied[0].content or "")


def test_denied_reason_reaches_the_model(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "app.py"})],
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
                    )
                ],
            ),
            LLMResponse(content="stopped", reasoning_content="", tool_calls=[]),
        ]
    )
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    agent.session.permission_mode = "ask"
    agent.approver = ScriptedApprover(
        [ApprovalDecision("deny", "这是账本，不要改这个文件")]
    )
    result = agent.run("change it")
    assert result.stop_reason == "completed"
    assert target.read_text(encoding="utf-8") == "value = 1\n"
    denied = [m for m in agent.messages if m.role == "tool" and "permission_denied" in (m.content or "")]
    assert denied
    assert "这是账本，不要改这个文件" in (denied[0].content or "")
    assert "Do not retry" in (denied[0].content or "")


def test_allow_session_skips_second_prompt(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    edit = LLMResponse(
        content="",
        reasoning_content="",
        tool_calls=[
            ToolCall(
                id="e",
                name="edit_file",
                arguments={"path": "app.py", "old_string": "value = 1", "new_string": "value = 2"},
            )
        ],
    )
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[ToolCall(id="r", name="read_file", arguments={"path": "app.py"})],
            ),
            edit,
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="e2",
                        name="edit_file",
                        arguments={"path": "app.py", "old_string": "value = 2", "new_string": "value = 3"},
                    )
                ],
            ),
            LLMResponse(content="done", reasoning_content="", tool_calls=[]),
        ]
    )
    approver = ScriptedApprover(["allow_session"])
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    agent.session.permission_mode = "ask"
    agent.approver = approver
    result = agent.run("edit twice")
    assert result.stop_reason == "completed"
    assert target.read_text(encoding="utf-8") == "value = 3\n"
    assert approver.seen == ["edit_file"]
    assert approver.answers == []


def test_auto_mode_does_not_ask(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="write_file",
                        arguments={"path": "n.py", "content": "ok\n"},
                    )
                ],
            ),
            LLMResponse(content="ok", reasoning_content="", tool_calls=[]),
        ]
    )
    approver = ScriptedApprover(["deny"])
    agent = Agent(_config(tmp_path), llm, build_tools(tmp_path, TodoStore(), 5), ContextManager(20_000))
    agent.session.permission_mode = "auto"
    agent.approver = approver
    result = agent.run("write")
    assert result.stop_reason == "completed"
    assert (tmp_path / "n.py").read_text(encoding="utf-8") == "ok\n"
    assert approver.seen == []


def test_permission_mode_persists_across_load(tmp_path: Path) -> None:
    session = Session(_config(tmp_path))
    assert session.set_permission("auto") == "perm auto"
    loaded = Session.load(_config(tmp_path), session._log_path)
    assert loaded.permission_mode == "auto"
    loaded.start_new()
    assert loaded.permission_mode == "ask"
