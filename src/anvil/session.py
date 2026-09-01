from __future__ import annotations

import hashlib
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
from anvil.skills import SkillStore
from anvil.tools.todo import TodoStore

SESSIONS_DIR = ".anvil/sessions"
MAX_ACTIVE_SKILLS = 4


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


@dataclass(frozen=True)
class CompactionCheckpoint:
    summary: str
    covered_count: int
    source_hash: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    def to_record(self) -> dict:
        return {
            "type": "compaction",
            "summary": self.summary,
            "covered_count": self.covered_count,
            "source_hash": self.source_hash,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
        }


@dataclass(frozen=True)
class SkillActivation:
    name: str
    content: str
    source_index: int
    source_hash: str

    def to_record(self) -> dict:
        return {
            "type": "skill_activation",
            "name": self.name,
            "content": self.content,
            "source_index": self.source_index,
            "source_hash": self.source_hash,
        }


class Session:
    """Owns conversation history. The model sees a compacted view; this log stays complete.

    Transcripts live in the workspace: `.anvil/sessions/<id>.jsonl`.
    """

    def __init__(
        self,
        config: Config,
        todo: TodoStore | None = None,
        skills: SkillStore | None = None,
    ) -> None:
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
        self.compaction: CompactionCheckpoint | None = None
        self.skills = skills or SkillStore(config.workspace)
        self.active_skills: dict[str, SkillActivation] = {}
        self._created = datetime.now(timezone.utc).isoformat()
        self.messages: list[Message] = [
            Message(
                role="system",
                content=build_system_prompt(config.workspace, skills=self.skills),
            )
        ]
        self._write_meta()
        self._write(self.messages[0])

    @classmethod
    def load(
        cls,
        config: Config,
        path: Path,
        todo: TodoStore | None = None,
        skills: SkillStore | None = None,
    ) -> Session:
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
        session.compaction = None
        session.skills = skills or SkillStore(config.workspace)
        session.active_skills = {}
        session._created = _header_created(path)
        session.messages = []
        meta_id, messages, settings, compaction, activations = _read_session_data(path)
        if meta_id:
            session.id = meta_id
        if "thinking" in settings:
            session.thinking = bool(settings["thinking"])
        if settings.get("reasoning_effort") in {"low", "high", "max"}:
            session.reasoning_effort = str(settings["reasoning_effort"])
        if settings.get("permission_mode") in {"ask", "auto"}:
            session.permission_mode = str(settings["permission_mode"])
        if not messages:
            session.messages = [
                Message(
                    role="system",
                    content=build_system_prompt(config.workspace, skills=session.skills),
                )
            ]
        else:
            session.messages = messages
            if session.messages[0].role != "system":
                session.messages.insert(
                    0,
                    Message(
                        role="system",
                        content=build_system_prompt(config.workspace, skills=session.skills),
                    ),
                )
        session.compaction = _valid_compaction(compaction, session.messages)
        session.active_skills = _valid_skill_activations(
            activations,
            session.messages,
            session._active_skill_char_budget(),
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
        self.compaction = None
        self.active_skills = {}
        self.skills.refresh()
        self.id = _new_id()
        self._created = datetime.now(timezone.utc).isoformat()
        self._log_path = sessions_dir(self.config.workspace) / f"{self.id}.jsonl"
        self.messages = [
            Message(
                role="system",
                content=build_system_prompt(self.config.workspace, skills=self.skills),
            )
        ]
        self._write_meta()
        self._write(self.messages[0])

    def append(self, message: Message) -> None:
        self.messages.append(message)
        self._write(message)

    def append_skill_source(self, message: Message, name: str, content: str) -> bool:
        """Persist a source message and its activation before publishing either in memory."""
        if not self.can_activate_skill(name, content):
            return False
        source_index = len(self.messages)
        item = SkillActivation(
            name=name,
            content=content,
            source_index=source_index,
            source_hash=_message_hash(message),
        )
        if not self._append_json_rows([message.to_record(), item.to_record()]):
            return False
        self.messages.append(message)
        self.active_skills[name] = item
        return True

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

    def save_compaction(
        self,
        summary: str,
        covered_count: int,
        *,
        usage: Usage | None = None,
        model: str = "",
    ) -> CompactionCheckpoint | None:
        if not summary.strip() or covered_count <= 1 or covered_count > len(self.messages):
            return None
        item = CompactionCheckpoint(
            summary=summary.strip(),
            covered_count=covered_count,
            source_hash=_messages_hash(self.messages, covered_count),
            prompt_tokens=(usage.prompt_tokens if usage else 0),
            completion_tokens=(usage.completion_tokens if usage else 0),
            model=model,
        )
        if not self._append_json(item.to_record()):
            return None
        self.compaction = item
        return item

    def can_activate_skill(self, name: str, content: str) -> bool:
        if not name or not content:
            return False
        latest = dict(self.active_skills)
        latest[name] = SkillActivation(name, content, 1, "pending")
        return (
            len(latest) <= MAX_ACTIVE_SKILLS
            and sum(len(item.content) for item in latest.values())
            <= self._active_skill_char_budget()
        )

    def _active_skill_char_budget(self) -> int:
        return max(2_000, min(60_000, int(self.config.context_budget * 0.6)))

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
            content=build_system_prompt(
                self.config.workspace,
                now=captured,
                skills=self.skills,
            ),
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
            if self.compaction is not None:
                rows.append(self.compaction.to_record())
            rows.extend(item.to_record() for item in self.active_skills.values())
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

    def _append_json(self, payload: dict) -> bool:
        return self._append_json_rows([payload])

    def _append_json_rows(self, payloads: list[dict]) -> bool:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                for payload in payloads:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            return False
        return True


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
    meta_id, messages, settings, _compaction, _activations = _read_session_data(path)
    return meta_id, messages, settings


def _read_session_data(
    path: Path,
) -> tuple[
    str | None,
    list[Message],
    dict,
    list[CompactionCheckpoint],
    list[SkillActivation],
]:
    meta_id: str | None = None
    messages: list[Message] = []
    settings: dict = {}
    compactions: list[CompactionCheckpoint] = []
    activations: list[SkillActivation] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, [], {}, [], []
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
        if payload.get("type") == "compaction":
            item = _checkpoint_from_record(payload)
            if item is not None:
                compactions.append(item)
            continue
        if payload.get("type") == "skill_activation":
            item = _skill_activation_from_record(payload)
            if item is not None:
                activations.append(item)
            continue
        message = Message.from_record(payload)
        if message is not None:
            messages.append(message)
    return meta_id or None, messages, settings, compactions, activations


def _checkpoint_from_record(payload: dict) -> CompactionCheckpoint | None:
    summary = payload.get("summary")
    source_hash = payload.get("source_hash")
    try:
        covered_count = int(payload.get("covered_count") or 0)
        prompt_tokens = int(payload.get("prompt_tokens") or 0)
        completion_tokens = int(payload.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return None
    if not isinstance(summary, str) or not summary.strip():
        return None
    if not isinstance(source_hash, str) or not source_hash:
        return None
    return CompactionCheckpoint(
        summary=summary.strip(),
        covered_count=covered_count,
        source_hash=source_hash,
        prompt_tokens=max(0, prompt_tokens),
        completion_tokens=max(0, completion_tokens),
        model=str(payload.get("model") or ""),
    )


def _valid_compaction(
    candidates: list[CompactionCheckpoint], messages: list[Message]
) -> CompactionCheckpoint | None:
    for item in reversed(candidates):
        if item.covered_count <= 1 or item.covered_count > len(messages):
            continue
        if item.source_hash == _messages_hash(messages, item.covered_count):
            return item
    return None


def _skill_activation_from_record(payload: dict) -> SkillActivation | None:
    name = payload.get("name")
    content = payload.get("content")
    source_hash = payload.get("source_hash")
    try:
        source_index = int(payload.get("source_index") or 0)
    except (TypeError, ValueError):
        return None
    if not all(isinstance(value, str) and value for value in (name, content, source_hash)):
        return None
    return SkillActivation(name, content, source_index, source_hash)


def _valid_skill_activations(
    candidates: list[SkillActivation],
    messages: list[Message],
    char_budget: int,
) -> dict[str, SkillActivation]:
    valid: dict[str, SkillActivation] = {}
    for item in candidates:
        if item.source_index <= 0 or item.source_index >= len(messages):
            continue
        source = messages[item.source_index]
        if item.source_hash != _message_hash(source):
            continue
        if item.content != (source.content or ""):
            continue
        next_items = dict(valid)
        next_items[item.name] = item
        if len(next_items) > MAX_ACTIVE_SKILLS:
            continue
        if sum(len(value.content) for value in next_items.values()) > char_budget:
            continue
        valid = next_items
    return valid


def _messages_hash(messages: list[Message], covered_count: int) -> str:
    rows = [message.to_record() for message in messages[1:covered_count]]
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _message_hash(message: Message) -> str:
    encoded = json.dumps(
        message.to_record(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
