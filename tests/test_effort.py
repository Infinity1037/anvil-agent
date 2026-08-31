from pathlib import Path

import pytest

from anvil.config import Config, ConfigError, effort_label, parse_effort
from anvil.llm.openai_compat import DeepSeekClient
from anvil.llm.types import Message, ToolCall
from anvil.session import Session
from anvil.tools.base import ToolSpec
from anvil.tui.complete import effort_suggestions, match_slash


def _config(workspace: Path, **overrides) -> Config:
    values = dict(
        api_key="test",
        base_url="http://localhost",
        model="scripted",
        thinking=True,
        reasoning_effort="max",
        max_turns=8,
        max_tokens=256000,
        context_budget=20_000,
        request_timeout=5,
        shell_timeout=5,
        workspace=workspace,
    )
    values.update(overrides)
    return Config(**values)


def test_parse_effort_accepts_three_levels() -> None:
    assert parse_effort("max") == "max"
    assert parse_effort(" HIGH ") == "high"
    with pytest.raises(ConfigError):
        parse_effort("off")
    with pytest.raises(ConfigError):
        parse_effort("medium")


def test_effort_label_hides_level_when_thinking_is_off() -> None:
    assert effort_label(True, "max") == "max"
    assert effort_label(False, "max") == "off"


def test_session_effort_persists_across_load(tmp_path: Path) -> None:
    session = Session(_config(tmp_path))
    assert session.set_effort("") == "effort max"
    assert session.set_effort("low") == "effort low"
    assert session.thinking is True
    assert session.set_effort("off") == "effort off"
    loaded = Session.load(_config(tmp_path, reasoning_effort="max"), session._log_path)
    assert loaded.thinking is False
    assert loaded.reasoning_effort == "low"
    assert loaded.effort_status() == "effort off"


def test_new_session_restores_config_effort(tmp_path: Path) -> None:
    session = Session(_config(tmp_path))
    session.set_effort("low")
    session.start_new()
    assert session.thinking is True
    assert session.reasoning_effort == "max"


def test_payload_sends_override_and_keeps_reasoning_when_tools_present(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    client = DeepSeekClient(cfg)
    try:
        low = client._payload(
            [Message(role="user", content="hi")],
            [],
            stream=False,
            thinking=True,
            reasoning_effort="low",
        )
        assert low["thinking"] == {"type": "enabled"}
        assert low["reasoning_effort"] == "low"
        assert low["max_tokens"] == 256000

        off = client._payload(
            [Message(role="user", content="hi")],
            [],
            stream=False,
            thinking=False,
        )
        assert off["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in off

        tool = ToolSpec(
            name="list_dir",
            description="list",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "",
        )
        history = [
            Message(
                role="assistant",
                content="",
                reasoning_content="look around",
                tool_calls=[ToolCall(id="c1", name="list_dir", arguments={})],
            )
        ]
        with_tools = client._payload(history, [tool], stream=False, thinking=False)
        assert with_tools["messages"][0]["reasoning_content"] == "look around"
    finally:
        client.close()


def test_todo_tool_returns_a_checklist() -> None:
    from anvil.tools.todo import TodoStore, coerce_todo_view

    store = TodoStore()
    result = store.replace(
        [
            {"id": "1", "content": "fix cents", "status": "in_progress"},
            {"id": "2", "content": "run tests", "status": "pending"},
            {"id": "3", "content": "read files", "status": "completed"},
        ]
    )
    assert result.ok
    assert "[>]  fix cents" in result.content
    assert "[ ]  run tests" in result.content
    assert "✓" in result.content
    assert "read files" in result.content
    assert "{" not in result.content
    dumped = '[\n  {"id": "1", "content": "fix cents", "status": "in_progress"}\n]'
    assert "[>]  fix cents" in coerce_todo_view(dumped)


def test_slash_menu_lists_effort_and_levels() -> None:
    names = [item.value for item in match_slash("/") or []]
    assert "effort" in names
    labels = [item.label for item in match_slash("/effort ") or []]
    assert labels == ["/effort off", "/effort low", "/effort high", "/effort max"]
    only_max = [item.value for item in match_slash("/effort m") or []]
    assert only_max == ["effort max"]


def test_effort_picker_marks_the_current_level() -> None:
    items = effort_suggestions("max")
    assert [item.kind for item in items] == ["effort"] * 4
    assert [item.value for item in items] == ["off", "low", "high", "max"]
    current = next(item for item in items if item.value == "max")
    assert "当前" in current.detail
    assert "当前" not in next(item for item in items if item.value == "low").detail
