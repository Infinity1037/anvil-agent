from pathlib import Path

from anvil.agent.context import ContextManager
from anvil.agent.loop import Agent
from anvil.config import Config
from anvil.llm.types import LLMResponse, ToolCall, Usage
from anvil.tools import TodoStore, build_tools


class ScriptedLLM:
    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)

    def complete(self, messages, tools, *, on_delta=None, stream=True):
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
    assert tool_messages[0].content.startswith("Error: unknown tool")
