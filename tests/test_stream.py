import time
from threading import Event, Timer

import httpx
import pytest

import anvil.llm.openai_compat as openai_compat
from anvil.config import Config
from anvil.llm.openai_compat import (
    DeepSeekClient,
    LLMCancelled,
    LLMStreamInterrupted,
)
from anvil.llm.types import Message


class _FakeStream:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    def iter_lines(self):
        yield from self._lines

    def read(self) -> bytes:
        return b""


class _InterruptedStream(_FakeStream):
    def iter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'
        raise httpx.ReadError("connection dropped")


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


def test_partial_stream_interruption_is_not_retryable(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path)
    attempts = 0

    def interrupted(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        return client._read_stream(_InterruptedStream([]), on_delta=None, cancel=None)

    monkeypatch.setattr(client, "_complete_stream", interrupted)
    try:
        with pytest.raises(LLMStreamInterrupted) as caught:
            client.complete([], [], stream=True)
        assert caught.value.partial is True
        assert attempts == 1
    finally:
        client.close()


def test_stream_without_a_terminal_event_is_rejected(tmp_path) -> None:
    client = _client(tmp_path)
    try:
        with pytest.raises(LLMStreamInterrupted) as caught:
            client._read_stream(
                _FakeStream(['data: {"choices":[{"delta":{"content":"partial"}}]}']),
                on_delta=None,
                cancel=None,
            )
        assert caught.value.partial is True
    finally:
        client.close()


def test_empty_interrupted_stream_is_retried(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path)
    attempts = 0

    def complete_stream(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return client._read_stream(_FakeStream([]), on_delta=None, cancel=None)
        return client._read_stream(
            _FakeStream(
                [
                    'data: {"choices":[{"delta":{"content":"recovered"},"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            ),
            on_delta=None,
            cancel=None,
        )

    monkeypatch.setattr(client, "_complete_stream", complete_stream)
    monkeypatch.setattr(openai_compat, "_wait_before_retry", lambda *_args: None)
    try:
        response = client.complete([], [], stream=True)
        assert response.content == "recovered"
        assert attempts == 2
    finally:
        client.close()


def test_cancel_interrupts_retry_backoff(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path)
    cancel = Event()
    attempts = 0

    def unavailable(_body):
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(client, "_complete_once", unavailable)
    timer = Timer(0.05, cancel.set)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(LLMCancelled):
            client.complete([], [], stream=False, cancel=cancel)
        assert time.monotonic() - started < 0.5
        assert attempts == 1
    finally:
        timer.cancel()
        client.close()


def test_assistant_without_tools_never_sends_null_content() -> None:
    payload = Message(role="assistant", content=None, reasoning_content="hi").to_openai(
        include_reasoning=True
    )
    assert payload["content"] == ""
    assert "tool_calls" not in payload
    assert payload["reasoning_content"] == "hi"
