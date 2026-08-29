from __future__ import annotations

import json
from typing import Any

from anvil.tools.base import ToolSpec

ALLOWED_STATUS = {"pending", "in_progress", "completed"}


class TodoStore:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def replace(self, items: list[dict[str, Any]]) -> str:
        cleaned: list[dict[str, str]] = []
        in_progress = 0
        for raw in items:
            if not isinstance(raw, dict):
                return "Error: each todo item must be an object."
            item_id = str(raw.get("id") or "").strip()
            content = str(raw.get("content") or "").strip()
            status = str(raw.get("status") or "").strip()
            if not item_id or not content:
                return "Error: each todo needs non-empty id and content."
            if status not in ALLOWED_STATUS:
                return f"Error: status must be one of {sorted(ALLOWED_STATUS)}."
            if status == "in_progress":
                in_progress += 1
            cleaned.append({"id": item_id, "content": content, "status": status})
        if in_progress > 1:
            return "Error: keep at most one todo in_progress at a time."
        self.items = cleaned
        return json.dumps(self.items, ensure_ascii=False, indent=2)


def make_todo(store: TodoStore) -> ToolSpec:
    def todo(items: list[dict[str, Any]]) -> str:
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
