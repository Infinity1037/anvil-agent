"""Slash-command and @file matching. No terminal I/O."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("help", "键位与命令"),
    ("status", "模型、会话与 token"),
    ("context", "查看当前上下文占用"),
    ("compact", "压缩旧上下文，可附保留重点"),
    ("effort", "思考强度 off/low/high/max"),
    ("perm", "权限 ask/auto"),
    ("clear", "开始新会话"),
    ("resume", "打开会话列表并恢复"),
    ("expand", "展开或收起输出"),
    ("exit", "退出"),
)

EFFORT_CHOICES: tuple[tuple[str, str], ...] = (
    ("off", "关闭思考"),
    ("low", "快、浅想"),
    ("high", "日常平衡"),
    ("max", "最强思考"),
)

SLASH_ALIASES = {
    "q": "exit",
    "quit": "exit",
    "new": "clear",
    "sessions": "resume",
    "yolo": "perm",
}

PERM_CHOICES: tuple[tuple[str, str], ...] = (
    ("ask", "改文件和命令前确认"),
    ("auto", "不询问直接执行"),
)

SKIP_DIRS = {
    ".git",
    ".anvil",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    "node_modules",
}


@dataclass(frozen=True)
class Suggestion:
    kind: str
    value: str
    label: str
    detail: str = ""


def slash_query(text: str) -> str | None:
    """Return the command prefix when the whole input is a slash token."""
    if "\n" in text:
        return None
    if not text.startswith("/"):
        return None
    if " " in text:
        return None
    return text[1:]


def match_slash(text: str) -> list[Suggestion] | None:
    effort_args = match_effort_args(text)
    if effort_args is not None:
        return effort_args
    perm_args = match_perm_args(text)
    if perm_args is not None:
        return perm_args
    query = slash_query(text)
    if query is None:
        return None
    needle = query.lower()
    items: list[Suggestion] = []
    seen: set[str] = set()
    details = {name: detail for name, detail in SLASH_COMMANDS}
    for name, detail in SLASH_COMMANDS:
        if needle and not name.startswith(needle):
            continue
        seen.add(name)
        items.append(Suggestion("slash", name, f"/{name}", detail))
    if needle:
        for alias, target in SLASH_ALIASES.items():
            if target in seen or not alias.startswith(needle):
                continue
            seen.add(target)
            items.append(
                Suggestion("slash", target, f"/{target}", details.get(target, ""))
            )
    return items


def effort_suggestions(current: str) -> list[Suggestion]:
    items: list[Suggestion] = []
    for value, detail in EFFORT_CHOICES:
        extra = f"{detail}  当前" if value == current else detail
        items.append(Suggestion("effort", value, value, extra))
    return items


def match_effort_args(text: str) -> list[Suggestion] | None:
    if "\n" in text or not text.startswith("/"):
        return None
    raw = text[1:]
    if not raw.lower().startswith("effort"):
        return None
    rest = raw[len("effort") :]
    if rest != "" and not rest.startswith(" "):
        return None
    if rest == "":
        return None
    needle = rest.strip().lower()
    items: list[Suggestion] = []
    for value, detail in EFFORT_CHOICES:
        if needle and not value.startswith(needle):
            continue
        items.append(Suggestion("slash", f"effort {value}", f"/effort {value}", detail))
    return items


def perm_suggestions(current: str) -> list[Suggestion]:
    items: list[Suggestion] = []
    for value, detail in PERM_CHOICES:
        extra = f"{detail}  当前" if value == current else detail
        items.append(Suggestion("perm", value, value, extra))
    return items


def match_perm_args(text: str) -> list[Suggestion] | None:
    if "\n" in text or not text.startswith("/"):
        return None
    raw = text[1:]
    if not raw.lower().startswith("perm"):
        return None
    rest = raw[len("perm") :]
    if rest != "" and not rest.startswith(" "):
        return None
    if rest == "":
        return None
    needle = rest.strip().lower()
    items: list[Suggestion] = []
    for value, detail in PERM_CHOICES:
        if needle and not value.startswith(needle):
            continue
        items.append(Suggestion("slash", f"perm {value}", f"/perm {value}", detail))
    return items


def mention_query(text: str, cursor: int | None = None) -> tuple[int, str] | None:
    """Return (at_index, prefix) if the cursor is in an @token."""
    if cursor is None:
        cursor = len(text)
    cursor = max(0, min(cursor, len(text)))
    before = text[:cursor]
    at = before.rfind("@")
    if at < 0:
        return None
    if at > 0 and not before[at - 1].isspace():
        return None
    token = before[at + 1 :]
    if any(ch.isspace() for ch in token):
        return None
    return at, token


def apply_mention(text: str, at_index: int, prefix: str, path: str) -> str:
    insert = "@" + path.replace("\\", "/") + " "
    start = at_index
    end = at_index + 1 + len(prefix)
    return text[:start] + insert + text[end:]


def match_files(root: Path, prefix: str, *, limit: int = 12, scan_cap: int = 400) -> list[Suggestion]:
    prefix_l = prefix.replace("\\", "/").lower()
    hits: list[Suggestion] = []
    scanned = 0
    root = root.resolve()
    if not root.is_dir():
        return []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")
        )
        rel_dir = Path(dirpath).resolve().relative_to(root).as_posix()
        for name in sorted(filenames):
            scanned += 1
            if scanned > scan_cap:
                return hits
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            rel = rel.replace("\\", "/")
            if prefix_l and prefix_l not in rel.lower() and not name.lower().startswith(prefix_l):
                continue
            hits.append(Suggestion("file", rel, rel, "file"))
            if len(hits) >= limit:
                return hits
    return hits


def index_from_location(text: str, location: tuple[int, int]) -> int:
    row, column = location
    lines = text.split("\n")
    if row < 0:
        return 0
    if row >= len(lines):
        return len(text)
    return sum(len(line) + 1 for line in lines[:row]) + min(column, len(lines[row]))


def location_from_index(text: str, index: int) -> tuple[int, int]:
    index = max(0, min(index, len(text)))
    row = 0
    remaining = index
    for line in text.split("\n"):
        if remaining <= len(line):
            return row, remaining
        remaining -= len(line) + 1
        row += 1
    return row, 0
