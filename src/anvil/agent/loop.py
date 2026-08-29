from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from anvil.agent.context import ContextManager
from anvil.agent.prompts import build_system_prompt
from anvil.config import Config
from anvil.llm.types import Message, Usage
from anvil.tools.base import ToolRegistry

EventCallback = Callable[[str, dict], None]


@dataclass
class RunResult:
    final_text: str
    turns: int
    stop_reason: str
    usage: Usage = field(default_factory=Usage)


class Agent:
    def __init__(
        self,
        config: Config,
        llm,
        tools: ToolRegistry,
        context: ContextManager,
    ) -> None:
        self.config = config
        self.llm = llm
        self.tools = tools
        self.context = context
        self.messages: list[Message] = [
            Message(role="system", content=build_system_prompt(config.workspace))
        ]
        self.usage = Usage()

    def reset(self) -> None:
        self.messages = [Message(role="system", content=build_system_prompt(self.config.workspace))]
        self.usage = Usage()

    def run(self, task: str, on_event: EventCallback | None = None) -> RunResult:
        self.messages.append(Message(role="user", content=task))
        consecutive_errors = 0

        for turn in range(1, self.config.max_turns + 1):
            _emit(on_event, "turn", {"turn": turn, "max_turns": self.config.max_turns})
            view = self.context.prepare(self.messages)
            try:
                response = self.llm.complete(
                    view,
                    self.tools.specs(),
                    on_delta=lambda kind, text: _emit(on_event, "delta", {"kind": kind, "text": text}),
                )
            except Exception as exc:
                _emit(on_event, "error", {"message": str(exc)})
                return RunResult(
                    final_text=f"Stopped: model request failed: {exc}",
                    turns=turn,
                    stop_reason="llm_error",
                    usage=self.usage,
                )

            self.usage.add(response.usage)
            self.messages.append(response.as_message())
            _emit(
                on_event,
                "assistant",
                {
                    "content": response.content or "",
                    "reasoning": response.reasoning_content or "",
                    "tool_calls": [
                        {"name": call.name, "arguments": call.arguments, "id": call.id}
                        for call in response.tool_calls
                    ],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                },
            )

            if not response.tool_calls:
                return RunResult(
                    final_text=(response.content or "").strip(),
                    turns=turn,
                    stop_reason="completed",
                    usage=self.usage,
                )

            results = self.tools.execute_batch(response.tool_calls)
            had_error = False
            for call, raw in zip(response.tool_calls, results):
                clipped = self.context.ingest_tool_result(raw)
                self.messages.append(
                    Message(role="tool", content=clipped, tool_call_id=call.id)
                )
                is_error = clipped.startswith("Error:")
                had_error = had_error or is_error
                _emit(
                    on_event,
                    "tool_result",
                    {
                        "id": call.id,
                        "name": call.name,
                        "ok": not is_error,
                        "content": clipped,
                    },
                )
            consecutive_errors = consecutive_errors + 1 if had_error else 0
            if consecutive_errors >= 5:
                text = "Stopped: too many consecutive tool errors."
                return RunResult(final_text=text, turns=turn, stop_reason="tool_errors", usage=self.usage)

        text = f"Stopped: reached max_turns={self.config.max_turns}."
        return RunResult(
            final_text=text,
            turns=self.config.max_turns,
            stop_reason="max_turns",
            usage=self.usage,
        )


def _emit(on_event: EventCallback | None, kind: str, payload: dict) -> None:
    if on_event:
        on_event(kind, payload)
