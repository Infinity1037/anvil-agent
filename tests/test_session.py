import json
from pathlib import Path

from anvil.agent.context import ContextManager
from anvil.config import Config
from anvil.llm.types import Message, ToolCall
from anvil.session import Session, find_session, latest_session, list_sessions


def _config(workspace: Path) -> Config:
    return Config(
        api_key="test",
        base_url="http://localhost",
        model="scripted",
        thinking=True,
        reasoning_effort="low",
        max_turns=8,
        max_tokens=256,
        context_budget=20_000,
        request_timeout=5,
        shell_timeout=5,
        workspace=workspace,
    )


def test_session_persists_and_reloads(tmp_path: Path) -> None:
    session = Session(_config(tmp_path))
    session.append(Message(role="user", content="hello there"))
    session.append(
        Message(
            role="assistant",
            content="hi",
            reasoning_content="greet",
            tool_calls=[ToolCall(id="c1", name="list_dir", arguments={"path": "."}, arguments_raw="{}")],
        )
    )
    session.append(Message(role="tool", content="file ledger.py", tool_call_id="c1"))
    path = session._log_path
    loaded = Session.load(_config(tmp_path), path)
    assert loaded.id == session.id
    roles = [item.role for item in loaded.messages]
    assert roles[0] == "system"
    assert "user" in roles
    assert loaded.messages[-1].tool_call_id == "c1"
    assistant = next(item for item in loaded.messages if item.role == "assistant")
    assert assistant.reasoning_content == "greet"
    assert assistant.tool_calls and assistant.tool_calls[0].name == "list_dir"


def test_list_sessions_is_workspace_local(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    first = Session(_config(tmp_path))
    first.append(Message(role="user", content="fix the ledger"))
    Session(_config(other)).append(Message(role="user", content="secret other project"))
    infos = list_sessions(tmp_path)
    assert len(infos) == 1
    assert infos[0].title.startswith("fix the ledger")
    assert latest_session(tmp_path).id == first.id
    assert find_session(tmp_path, first.id[:8]) is not None
    assert find_session(other, first.id) is None


def test_find_session_rejects_an_external_jsonl_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    other = tmp_path / "other"
    workspace.mkdir()
    other.mkdir()
    external = Session(_config(other))
    external.append(Message(role="user", content="outside conversation"))

    assert find_session(workspace, str(external._log_path)) is None


def test_start_new_writes_a_second_file(tmp_path: Path) -> None:
    session = Session(_config(tmp_path))
    session.append(Message(role="user", content="first"))
    old = session.id
    session.start_new()
    session.append(Message(role="user", content="second"))
    assert session.id != old
    infos = list_sessions(tmp_path)
    assert len(infos) == 2
    titles = {item.title for item in infos}
    assert "first" in titles
    assert "second" in titles


def test_jsonl_skips_bad_lines(tmp_path: Path) -> None:
    session = Session(_config(tmp_path))
    session.append(Message(role="user", content="keep me"))
    path = session._log_path
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text
        + "{not json\n"
        + "null\n"
        + '{"type": "noise", "role": ""}\n'
        + '{"type": "message", "role": "assistant", "content": "still here"}\n',
        encoding="utf-8",
    )
    loaded = Session.load(_config(tmp_path), path)
    roles = [item.role for item in loaded.messages]
    assert "user" in roles
    assert "assistant" in roles
    assert any(item.content == "still here" for item in loaded.messages)
    assert any(item.content == "keep me" for item in loaded.messages)


def test_huge_tool_result_is_spilled_to_workspace(tmp_path: Path) -> None:
    manager = ContextManager(budget=20_000, workspace=tmp_path)
    blob = "X" * 12_000
    stored = manager.ingest_tool_result(blob, call_id="call9")
    assert "truncated" in stored
    assert ".anvil" not in stored
    saved = tmp_path / ".anvil" / "tool-output" / "call9.txt"
    assert saved.is_file()
    assert saved.read_text(encoding="utf-8") == blob
    assert blob not in stored or len(stored) < len(blob)


def test_load_refreshes_stale_system_prompt_and_keeps_date(tmp_path: Path) -> None:
    session = Session(_config(tmp_path))
    session.append(Message(role="user", content="hello"))
    path = session._log_path
    lines = path.read_text(encoding="utf-8").splitlines()
    rewritten = []
    for line in lines:
        payload = json.loads(line)
        if payload.get("role") == "system":
            payload["content"] = (
                "You are Anvil, a stale prompt.\n"
                "Date: 2026-01-15 (captured at session start; it does not update)"
            )
        rewritten.append(json.dumps(payload, ensure_ascii=False))
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    loaded = Session.load(_config(tmp_path), path)
    system = loaded.messages[0].content or ""
    assert system.startswith("You are Anvil")
    assert "# Language" in system
    assert "Date: 2026-01-15" in system
    assert "stale prompt" not in system
    assert "hello" in (loaded.messages[1].content or "")
    disk = path.read_text(encoding="utf-8")
    assert "# Language" in disk
    assert "Date: 2026-01-15" in disk
