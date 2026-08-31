from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Event
from typing import Any, Callable

from anvil.llm.types import ToolCall
from anvil.safety import (
    DangerousCommandError,
    InternalPathError,
    PathEscapeError,
    SecretFileError,
)
from anvil.tools.observe import FileObserver
from anvil.tools.result import ToolResult

Handler = Callable[..., str | ToolResult]


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

    @property
    def property_names(self) -> set[str]:
        properties = self.parameters.get("properties") or {}
        return set(properties) if isinstance(properties, dict) else set()

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
    def __init__(self, observer: FileObserver | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.observer = observer or FileObserver()
        self.cancel: Event | None = None

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            return self._execute(call)
        except Exception as exc:
            return ToolResult.fail("exception", f"{type(exc).__name__}: {exc}")

    def _execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "(none)"
            return ToolResult.fail(
                "unknown_tool",
                f"unknown tool '{call.name}'. Available: {available}",
                hint="Call one of the listed tools.",
            )
        if call.parse_error:
            return ToolResult.fail(
                "invalid_json",
                f"tool arguments were not valid JSON. Received: {call.arguments_raw[:500]}",
                hint="Resend the tool call with a JSON object.",
            )
        arguments = _known_arguments(tool, call.arguments)
        missing = [name for name in tool.required if name not in arguments]
        if missing:
            return ToolResult.fail(
                "missing_arguments",
                f"missing required arguments: {', '.join(missing)}",
            )
        try:
            result = tool.handler(**arguments)
        except PathEscapeError as exc:
            return ToolResult.fail(
                "path_escape",
                str(exc),
                hint="Use a path inside the workspace.",
            )
        except SecretFileError as exc:
            return ToolResult.fail(
                "secret_file",
                str(exc),
                hint="Do not read or write secret files.",
            )
        except InternalPathError as exc:
            return ToolResult.fail(
                "internal_path",
                str(exc),
                hint="Stay in project source files.",
            )
        except DangerousCommandError as exc:
            return ToolResult.fail("dangerous_command", str(exc))
        except TypeError as exc:
            return ToolResult.fail("bad_arguments", f"bad arguments for {call.name}: {exc}")
        except Exception as exc:
            return ToolResult.fail("exception", f"{type(exc).__name__}: {exc}")
        if isinstance(result, ToolResult):
            return result
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        return ToolResult.success(result)

    def execute_batch(self, calls: list[ToolCall], cancel: Event | None = None) -> list[ToolResult]:
        if not calls:
            return []
        cancelled = ToolResult.fail("cancelled", "cancelled before execution")
        if cancel and cancel.is_set():
            return [cancelled for _ in calls]
        previous = self.cancel
        self.cancel = cancel
        try:
            return self._run_batch(calls, cancel, cancelled)
        finally:
            self.cancel = previous

    def _run_batch(
        self,
        calls: list[ToolCall],
        cancel: Event | None,
        cancelled: ToolResult,
    ) -> list[ToolResult]:
        def run_one(call: ToolCall) -> ToolResult:
            if cancel and cancel.is_set():
                return cancelled
            return self.execute(call)

        if len(calls) == 1:
            return [run_one(calls[0])]
        if all(self._tools.get(call.name) and self._tools[call.name].parallel_safe for call in calls):
            results: list[ToolResult | None] = [None] * len(calls)
            with ThreadPoolExecutor(max_workers=min(8, len(calls))) as pool:
                futures = {pool.submit(run_one, call): index for index, call in enumerate(calls)}
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            return [item if item is not None else cancelled for item in results]
        return [run_one(call) for call in calls]


def _known_arguments(tool: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    allowed = tool.property_names
    if not allowed:
        return dict(arguments)
    return {key: value for key, value in arguments.items() if key in allowed}
