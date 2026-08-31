from threading import Event

import pytest

from anvil.config import Config
from anvil.llm.openai_compat import DeepSeekClient, LLMCancelled
from anvil.llm.types import Message


class _FakeStream:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    def iter_lines(self):
        yield from self._lines

    def read(self) -> bytes:
        return b""


def _client(tmp_path) -> DeepSeekClient:
    return DeepSeekClient(
        Config(
            api_key="test",
            base_url="http://localhost",
            model="scripted",
            thinking=True,
            reasoning_effort="max",
            max_turns=8,
            max_tokens=256,
            context_budget=20_000,
            request_timeout=5,
            shell_timeout=5,
            workspace=tmp_path,
        )
    )


def test_read_stream_parses_content_and_reasoning(tmp_path) -> None:
    client = _client(tmp_path)
    try:
        stream = _FakeStream(
            [
                "",
                ": keep-alive",
                'data: {"choices":[{"delta":{"reasoning_content":"think "}}]}',
                'data: {"choices":[{"delta":{"reasoning_content":"first"}}]}',
                'data: {"choices":[{"delta":{"content":"你好"}}]}',
                'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
        response = client._read_stream(stream, on_delta=None, cancel=None)
        assert response.reasoning_content == "think first"
        assert response.content == "你好"
        assert response.tool_calls == []
        assert response.finish_reason == "stop"
    finally:
        client.close()


def test_read_stream_parses_tool_calls(tmp_path) -> None:
    client = _client(tmp_path)
    try:
        stream = _FakeStream(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"list_dir","arguments":""}}]}}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}',
                'data: {"choices":[{"delta":{}, "finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        )
        response = client._read_stream(stream, on_delta=None, cancel=None)
        assert response.content is None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "list_dir"
        assert response.tool_calls[0].id == "c1"
    finally:
        client.close()


def test_read_stream_raises_when_cancelled(tmp_path) -> None:
    client = _client(tmp_path)
    cancel = Event()
    cancel.set()
    try:
        with pytest.raises(LLMCancelled):
            client._read_stream(_FakeStream(["data: {}"]), on_delta=None, cancel=cancel)
    finally:
        client.close()


def test_assistant_without_tools_never_sends_null_content() -> None:
    payload = Message(role="assistant", content=None, reasoning_content="hi").to_openai(
        include_reasoning=True
    )
    assert payload["content"] == ""
    assert "tool_calls" not in payload
    assert payload["reasoning_content"] == "hi"
