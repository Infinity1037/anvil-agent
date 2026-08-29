from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from anvil.llm.types import ToolCall

Handler = Callable[..., str]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    parallel_safe: bool = False

    @property
    def required(self) -> list[str]:
        required = self.parameters.get("required") or []
        return list(required)

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def execute(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "(none)"
            return f"Error: unknown tool '{call.name}'. Available: {available}"
        if call.arguments_raw and not call.arguments:
            return (
                "Error: tool arguments were not valid JSON. "
                f"Received: {call.arguments_raw[:500]}"
            )
        missing = [name for name in tool.required if name not in call.arguments]
        if missing:
            return f"Error: missing required arguments: {', '.join(missing)}"
        try:
            result = tool.handler(**call.arguments)
        except TypeError as exc:
            return f"Error: bad arguments for {call.name}: {exc}"
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"
        if not isinstance(result, str):
            return json.dumps(result, ensure_ascii=False)
        return result

    def execute_batch(self, calls: list[ToolCall]) -> list[str]:
        if not calls:
            return []
        if len(calls) == 1:
            return [self.execute(calls[0])]
        if all(self._tools.get(call.name) and self._tools[call.name].parallel_safe for call in calls):
            results: list[str | None] = [None] * len(calls)
            with ThreadPoolExecutor(max_workers=min(8, len(calls))) as pool:
                futures = {
                    pool.submit(self.execute, call): index
                    for index, call in enumerate(calls)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            return [item if item is not None else "Error: missing tool result" for item in results]
        return [self.execute(call) for call in calls]
