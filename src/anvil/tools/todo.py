from __future__ import annotations

import json
from typing import Any

from anvil.tools.base import ToolSpec
from anvil.tools.result import ToolResult

ALLOWED_STATUS = {"pending", "in_progress", "completed"}

STATUS_MARK = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": " ✓",
}


def render_todo_list(items: list[dict[str, str]]) -> str:
    if not items:
        return "(empty todo list)"
    lines: list[str] = []
    for item in items:
        mark = STATUS_MARK.get(item.get("status") or "", "[ ]")
        lines.append(f"{mark}  {item.get('content') or ''}")
    return "\n".join(lines)


def coerce_todo_view(content: str) -> str:
    text = (content or "").strip()
    if not text.startswith("["):
        return content
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return content
    if not isinstance(data, list):
        return content
    cleaned: list[dict[str, str]] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        cleaned.append(
            {
                "id": str(raw.get("id") or ""),
                "content": str(raw.get("content") or raw.get("title") or ""),
                "status": str(raw.get("status") or "pending"),
            }
        )
    return render_todo_list(cleaned)


def todo_summary(items: list) -> str:
    if not isinstance(items, list) or not items:
        return ""
    done = sum(1 for item in items if isinstance(item, dict) and item.get("status") == "completed")
    return f"{done}/{len(items)}"


class TodoStore:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def replace(self, items: list[dict[str, Any]]) -> ToolResult:
        cleaned: list[dict[str, str]] = []
        in_progress = 0
        for raw in items:
            if not isinstance(raw, dict):
                return ToolResult.fail("todo_invalid", "each todo item must be an object.")
            item_id = str(raw.get("id") or "").strip()
            content = str(raw.get("content") or "").strip()
            status = str(raw.get("status") or "").strip()
            if not item_id or not content:
                return ToolResult.fail("todo_invalid", "each todo needs non-empty id and content.")
            if status not in ALLOWED_STATUS:
                return ToolResult.fail(
                    "todo_invalid",
                    f"status must be one of {sorted(ALLOWED_STATUS)}.",
                )
            if status == "in_progress":
                in_progress += 1
            cleaned.append({"id": item_id, "content": content, "status": status})
        if in_progress > 1:
            return ToolResult.fail(
                "todo_invalid",
                "keep at most one todo in_progress at a time.",
            )
        self.items = cleaned
        return ToolResult.success(render_todo_list(self.items))


def make_todo(store: TodoStore) -> ToolSpec:
    def todo(items: list[dict[str, Any]]) -> ToolResult:
        return store.replace(items)

    return ToolSpec(
        name="todo",
        description=(
            "Replace the current task list. Use this for multi-step work. "
            "Keep at most one item in_progress."
        ),
        parameters={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "The full todo list after this update.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["id", "content", "status"],
                    },
                }
            },
            "required": ["items"],
        },
        handler=todo,
        parallel_safe=False,
    )
