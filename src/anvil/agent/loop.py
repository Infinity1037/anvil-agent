from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Event

from anvil.agent.context import ContextManager
from anvil.agent.permissions import ApprovalDecision, needs_ask
from anvil.config import Config
from anvil.events import AgentEvent, EventCallback, emit
from anvil.llm.parse import complete_without_tools
from anvil.llm.types import Message, ToolCall, Usage
from anvil.session import Session
from anvil.tools.base import ToolRegistry
from anvil.tools.result import ToolResult

REPEAT_WARN = 3
REPEAT_STOP = 5


@dataclass
class RunResult:
    final_text: str
    turns: int
    stop_reason: str
    usage: Usage = field(default_factory=Usage)


class Agent:
    """Inner tool loop for one user turn. History lives on Session."""

    def __init__(
        self,
        config: Config,
        llm,
        tools: ToolRegistry,
        context: ContextManager,
        session: Session | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.tools = tools
        self.context = context
        self.session = session or Session(config)
        self.approver = None
        self._repeat_counts: dict[str, int] = {}

    @property
    def messages(self) -> list[Message]:
        return self.session.messages

    @property
    def usage(self) -> Usage:
        return self.session.usage

    def new_session(self) -> None:
        self.session.start_new()
        self.reset_turn_state(clear_observer=True)

    def attach_session(self, session: Session) -> None:
        self.session = session
        self.reset_turn_state(clear_observer=True)

    def reset_turn_state(self, *, clear_observer: bool = False) -> None:
        self._repeat_counts = {}
        if clear_observer:
            self.tools.observer.clear()

    def run(
        self,
        task: str,
        on_event: EventCallback | None = None,
        cancel: Event | None = None,
    ) -> RunResult:
        flag = cancel or self.session.cancel
        flag.clear()
        self._repeat_counts = {}
        self.session.append(Message(role="user", content=task))
        consecutive_errors = 0

        for turn in range(1, self.config.max_turns + 1):
            if _cancelled(flag):
                return self._stop("cancelled", turn, "Stopped: cancelled.", on_event)
            emit(on_event, "turn", {"turn": turn, "max_turns": self.config.max_turns})
            log = self.session.messages
            before_tokens = self.context.estimate(log)
            view = self.context.prepare(log)
            after_tokens = self.context.estimate(view)
            if after_tokens < before_tokens:
                emit(
                    on_event,
                    "compact",
                    {"before_tokens": before_tokens, "after_tokens": after_tokens},
                )
            try:
                response = self.llm.complete(
                    view,
                    self.tools.specs(),
                    on_delta=lambda kind, text: emit(on_event, "delta", {"kind": kind, "text": text}),
                    thinking=self.session.thinking,
                    reasoning_effort=self.session.reasoning_effort,
                    cancel=flag,
                )
            except KeyboardInterrupt:
                flag.set()
                return self._stop("cancelled", turn, "Stopped: cancelled.", on_event)
            except Exception as exc:
                if type(exc).__name__ == "LLMCancelled":
                    return self._stop("cancelled", turn, "Stopped: cancelled.", on_event)
                emit(on_event, "error", {"message": str(exc)})
                return self._stop("llm_error", turn, f"Stopped: model request failed: {exc}", on_event)

            self.session.usage.add(response.usage)
            self.context.note_prompt_usage(view, response.usage)
            self.session.append(response.as_message())
            emit(
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

            if _cancelled(flag):
                return self._stop("cancelled", turn, "Stopped: cancelled.", on_event)

            if not response.tool_calls:
                reason = complete_without_tools(response.finish_reason)
                text = (response.content or "").strip()
                if reason == "length" and not text:
                    text = "Stopped: model output truncated."
                return self._stop(reason, turn, text, on_event)

            results = self._execute_calls(response.tool_calls, flag)
            had_error = False
            stop_progress = False
            for call, result in zip(response.tool_calls, results):
                if _cancelled(flag) or result.error_code == "cancelled":
                    self._append_tool(call, result, on_event)
                    return self._stop("cancelled", turn, "Stopped: cancelled.", on_event)
                text = self.context.ingest_tool_result(
                    result.to_message_content(), call_id=call.id
                )
                text, halt = self._note_repeated_call(call, text)
                self.session.append(Message(role="tool", content=text, tool_call_id=call.id))
                emit(
                    on_event,
                    "tool_result",
                    {
                        "id": call.id,
                        "name": call.name,
                        "ok": result.ok,
                        "content": text,
                        "error_code": result.error_code,
                    },
                )
                had_error = had_error or (
                    not result.ok and result.error_code not in {"permission_denied", "cancelled"}
                )
                stop_progress = stop_progress or halt
            if stop_progress:
                return self._stop(
                    "no_progress",
                    turn,
                    "Stopped: repeating the same tool call.",
                    on_event,
                )
            consecutive_errors = consecutive_errors + 1 if had_error else 0
            if consecutive_errors >= 5:
                return self._stop(
                    "tool_errors",
                    turn,
                    "Stopped: too many consecutive tool errors.",
                    on_event,
                )

        return self._stop(
            "max_turns",
            self.config.max_turns,
            f"Stopped: reached max_turns={self.config.max_turns}.",
            on_event,
        )

    def _execute_calls(self, calls: list[ToolCall], flag: Event | None) -> list[ToolResult]:
        mode = getattr(self.session, "permission_mode", "ask")
        allowed = getattr(self.session, "session_allow", set())
        if not any(needs_ask(mode, call.name, allowed) for call in calls):
            return self.tools.execute_batch(calls, cancel=flag)
        previous = self.tools.cancel
        self.tools.cancel = flag
        try:
            return self._execute_serial(calls, flag)
        finally:
            self.tools.cancel = previous

    def _execute_serial(self, calls: list[ToolCall], flag: Event | None) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in calls:
            if _cancelled(flag):
                results.append(ToolResult.fail("cancelled", "cancelled before execution"))
                continue
            if needs_ask(self.session.permission_mode, call.name, self.session.session_allow):
                decision = self._ask(call, flag)
                verdict = decision.verdict
                if verdict == "allow_session":
                    self.session.allow_tool(call.name)
                    verdict = "allow"
                if verdict != "allow":
                    cancelled = verdict == "cancelled" or _cancelled(flag)
                    results.append(_denied_result(cancelled=cancelled, reason=decision.reason))
                    continue
            results.append(self.tools.execute(call))
        return results

    def _ask(self, call: ToolCall, flag: Event | None) -> ApprovalDecision:
        if self.approver is None:
            return ApprovalDecision("allow")
        try:
            raw = self.approver.decide(call, cancel=flag)
        except Exception:
            return ApprovalDecision("deny")
        decision = ApprovalDecision.from_raw(raw)
        if decision.verdict == "cancelled" or _cancelled(flag):
            return ApprovalDecision("cancelled")
        return decision

    def _append_tool(self, call: ToolCall, result: ToolResult, on_event: EventCallback | None) -> None:
        text = result.to_message_content()
        self.session.append(Message(role="tool", content=text, tool_call_id=call.id))
        emit(
            on_event,
            "tool_result",
            {
                "id": call.id,
                "name": call.name,
                "ok": result.ok,
                "content": text,
                "error_code": result.error_code,
            },
        )

    def _stop(
        self,
        reason: str,
        turns: int,
        text: str,
        on_event: EventCallback | None,
    ) -> RunResult:
        emit(on_event, "ended", {"stop_reason": reason, "turns": turns})
        return RunResult(final_text=text, turns=turns, stop_reason=reason, usage=self.session.usage)

    def _note_repeated_call(self, call: ToolCall, result: str) -> tuple[str, bool]:
        fingerprint = _fingerprint(call)
        count = self._repeat_counts.get(fingerprint, 0) + 1
        self._repeat_counts[fingerprint] = count
        if count == REPEAT_WARN:
            result += (
                "\n\n[progress] This exact tool call was repeated 3 times. "
                "Inspect the previous result and change approach."
            )
        if count >= REPEAT_STOP:
            result += (
                "\n\n[progress] This exact tool call was repeated 5 times. Stopping."
            )
            return result, True
        return result, False


def _denied_result(*, cancelled: bool, reason: str = "") -> ToolResult:
    if cancelled:
        return ToolResult.fail("cancelled", "cancelled before execution")
    if reason:
        return ToolResult.fail(
            "permission_denied",
            f"User rejected this tool call: {reason}",
            hint="Do not retry the same call. Follow the user's reason.",
        )
    return ToolResult.fail(
        "permission_denied",
        "User rejected this tool call.",
        hint="Change approach or ask the user to approve.",
    )


def _cancelled(flag: Event | None) -> bool:
    return bool(flag and flag.is_set())


def _fingerprint(call: ToolCall) -> str:
    if call.parse_error:
        return f"{call.name}:invalid:{_collapse_ws(call.arguments_raw)}"
    if call.name == "run_shell":
        return f"run_shell:{_shell_shape(str(call.arguments.get('command') or ''))}"
    dumped = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
    return f"{call.name}:{_collapse_ws(dumped)}"


def _collapse_ws(text: str) -> str:
    return " ".join((text or "").split())


def _shell_shape(command: str) -> str:
    tokens = command.split()
    if not tokens:
        return ""
    prog = tokens[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if prog in {"python", "python3", "py"} and "-c" in tokens[:4]:
        return f"{prog} -c"
    return _collapse_ws(command)


def as_ui_handler(handle) -> EventCallback:
    def _adapt(event: AgentEvent) -> None:
        handle(event.kind, event.payload)

    return _adapt
