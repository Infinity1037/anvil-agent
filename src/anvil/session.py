from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from anvil.agent.prompts import build_system_prompt, prompt_date
from anvil.config import Config
from anvil.llm.types import Message, Usage
from anvil.tools.todo import TodoStore

SESSIONS_DIR = ".anvil/sessions"


def _new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def sessions_dir(workspace: Path) -> Path:
    return workspace / ".anvil" / "sessions"


@dataclass(frozen=True)
class SessionInfo:
    id: str
    path: Path
    created: str
    title: str
    mtime: float
    messages: int


class Session:
    """Owns conversation history. The model sees a compacted view; this log stays complete.

    Transcripts live in the workspace: `.anvil/sessions/<id>.jsonl`.
    """

    def __init__(self, config: Config, todo: TodoStore | None = None) -> None:
        self.config = config
        self.todo = todo or TodoStore()
        self.usage = Usage()
        self.cancel = threading.Event()
        self.id = _new_id()
        self._log_path = sessions_dir(config.workspace) / f"{self.id}.jsonl"
        self.thinking = config.thinking
        self.reasoning_effort = config.reasoning_effort
        self.permission_mode = "ask"
        self.session_allow: set[str] = set()
        self._created = datetime.now(timezone.utc).isoformat()
        self.messages: list[Message] = [
            Message(role="system", content=build_system_prompt(config.workspace))
        ]
        self._write_meta()
        self._write(self.messages[0])

    @classmethod
    def load(cls, config: Config, path: Path, todo: TodoStore | None = None) -> Session:
        session = cls.__new__(cls)
        session.config = config
        session.todo = todo or TodoStore()
        session.usage = Usage()
        session.cancel = threading.Event()
        session._log_path = path
        session.id = path.stem
        session.thinking = config.thinking
        session.reasoning_effort = config.reasoning_effort
        session.permission_mode = "ask"
        session.session_allow = set()
        session._created = _header_created(path)
        session.messages = []
        meta_id, messages, settings = read_session_file(path)
        if meta_id:
            session.id = meta_id
        if "thinking" in settings:
            session.thinking = bool(settings["thinking"])
        if settings.get("reasoning_effort") in {"low", "high", "max"}:
            session.reasoning_effort = str(settings["reasoning_effort"])
        if settings.get("permission_mode") in {"ask", "auto"}:
            session.permission_mode = str(settings["permission_mode"])
        if not messages:
            session.messages = [Message(role="system", content=build_system_prompt(config.workspace))]
        else:
            session.messages = messages
            if session.messages[0].role != "system":
                session.messages.insert(
                    0, Message(role="system", content=build_system_prompt(config.workspace))
                )
        session._refresh_system()
        return session

    def start_new(self) -> None:
        """Begin a fresh transcript in this workspace."""
        self.cancel.clear()
        self.usage = Usage()
        self.todo.items = []
        self.thinking = self.config.thinking
        self.reasoning_effort = self.config.reasoning_effort
        self.permission_mode = "ask"
        self.session_allow = set()
        self.id = _new_id()
        self._created = datetime.now(timezone.utc).isoformat()
        self._log_path = sessions_dir(self.config.workspace) / f"{self.id}.jsonl"
        self.messages = [Message(role="system", content=build_system_prompt(self.config.workspace))]
        self._write_meta()
        self._write(self.messages[0])

    def append(self, message: Message) -> None:
        self.messages.append(message)
        self._write(message)

    def set_effort(self, token: str) -> str:
        from anvil.config import apply_effort_token

        raw = token.strip()
        if not raw:
            return self.effort_status()
        self.thinking, self.reasoning_effort = apply_effort_token(
            self.thinking, self.reasoning_effort, raw
        )
        self._write_settings()
        return self.effort_status()

    def set_permission(self, token: str) -> str:
        from anvil.agent.permissions import parse_permission

        raw = token.strip()
        if not raw:
            return self.permission_status()
        self.permission_mode = parse_permission(raw)
        if self.permission_mode == "ask":
            self.session_allow = set()
        self._write_settings()
        return self.permission_status()

    def allow_tool(self, name: str) -> None:
        if name:
            self.session_allow.add(name)

    def permission_status(self) -> str:
        return f"perm {self.permission_mode}"

    def effort_status(self) -> str:
        if not self.thinking:
            return "effort off"
        return f"effort {self.reasoning_effort}"

    def _refresh_system(self) -> None:
        """Replace a stale system prompt with the current assembly. Keep the original date."""
        old = self.messages[0] if self.messages and self.messages[0].role == "system" else None
        captured = prompt_date(old.content or "") if old else None
        fresh = Message(
            role="system",
            content=build_system_prompt(self.config.workspace, now=captured),
        )
        if old is None:
            self.messages.insert(0, fresh)
            self._rewrite()
            return
        if old.content == fresh.content:
            return
        self.messages[0] = fresh
        self._rewrite()

    def _rewrite(self) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._log_path.with_name(self._log_path.name + ".tmp")
            created = getattr(self, "_created", "") or datetime.now(timezone.utc).isoformat()
            rows = [
                {
                    "type": "session",
                    "id": self.id,
                    "created": created,
                    "workspace": str(self.config.workspace),
                },
                {
                    "type": "settings",
                    "thinking": self.thinking,
                    "reasoning_effort": self.reasoning_effort,
                    "permission_mode": self.permission_mode,
                },
            ]
            with tmp.open("w", encoding="utf-8") as handle:
                for payload in rows:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                for message in self.messages:
                    handle.write(json.dumps(message.to_record(), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self._log_path)
        except OSError:
            return

    def _write_meta(self) -> None:
        self._append_json(
            {
                "type": "session",
                "id": self.id,
                "created": getattr(self, "_created", "") or datetime.now(timezone.utc).isoformat(),
                "workspace": str(self.config.workspace),
            }
        )

    def _write(self, message: Message) -> None:
        self._append_json(message.to_record())

    def _write_settings(self) -> None:
        self._append_json(
            {
                "type": "settings",
                "thinking": self.thinking,
                "reasoning_effort": self.reasoning_effort,
                "permission_mode": self.permission_mode,
            }
        )

    def _append_json(self, payload: dict) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            return


def _header_created(path: Path) -> str:
    try:
        first = path.read_text(encoding="utf-8").splitlines()[:1]
        if first:
            header = json.loads(first[0])
            if isinstance(header, dict) and header.get("type") == "session":
                return str(header.get("created") or "") or datetime.now(timezone.utc).isoformat()
    except (OSError, json.JSONDecodeError):
        pass
    return datetime.now(timezone.utc).isoformat()


def read_session_file(path: Path) -> tuple[str | None, list[Message], dict]:
    meta_id: str | None = None
    messages: list[Message] = []
    settings: dict = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, [], {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "session":
            meta_id = str(payload.get("id") or meta_id or "")
            continue
        if payload.get("type") == "settings":
            if "thinking" in payload:
                settings["thinking"] = bool(payload.get("thinking"))
            effort = payload.get("reasoning_effort")
            if effort in {"low", "high", "max"}:
                settings["reasoning_effort"] = effort
            mode = payload.get("permission_mode")
            if mode in {"ask", "auto"}:
                settings["permission_mode"] = mode
            continue
        message = Message.from_record(payload)
        if message is not None:
            messages.append(message)
    return meta_id or None, messages, settings


def list_sessions(workspace: Path) -> list[SessionInfo]:
    folder = sessions_dir(workspace)
    if not folder.is_dir():
        return []
    infos: list[SessionInfo] = []
    for path in folder.glob("*.jsonl"):
        info = _session_info(path)
        if info is not None:
            infos.append(info)
    infos.sort(key=lambda item: item.mtime, reverse=True)
    return infos


def latest_session(workspace: Path) -> SessionInfo | None:
    infos = list_sessions(workspace)
    return infos[0] if infos else None


def find_session(workspace: Path, token: str) -> SessionInfo | None:
    token = token.strip()
    if not token:
        return None
    infos = list_sessions(workspace)
    for info in infos:
        if info.id == token:
            return info
    matches = [info for info in infos if info.id.startswith(token)]
    if len(matches) == 1:
        return matches[0]
    return None


def _session_info(path: Path) -> SessionInfo | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    meta_id, messages, _settings = read_session_file(path)
    title = ""
    created = ""
    try:
        first = path.read_text(encoding="utf-8").splitlines()[:1]
        if first:
            header = json.loads(first[0])
            if isinstance(header, dict) and header.get("type") == "session":
                created = str(header.get("created") or "")
    except (OSError, json.JSONDecodeError):
        created = ""
    for message in messages:
        if message.role == "user" and (message.content or "").strip():
            text = (message.content or "").strip().replace("\n", " ")
            if text.startswith("["):
                continue
            title = text[:60]
            break
    return SessionInfo(
        id=meta_id or path.stem,
        path=path,
        created=created or datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        title=title or path.stem,
        mtime=stat.st_mtime,
        messages=len(messages),
    )
