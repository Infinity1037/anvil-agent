from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Event, Lock

from anvil.agent.context import ContextManager, ContextSnapshot, PinnedContext, normalize_summary
from anvil.agent.permissions import ApprovalDecision, needs_ask
from anvil.config import Config
from anvil.events import AgentEvent, EventCallback, emit
from anvil.llm.parse import complete_without_tools
from anvil.llm.types import Message, ToolCall, Usage
from anvil.session import Session
from anvil.tools.base import ToolRegistry
from anvil.tools.result import ToolResult
from anvil.tools.skills import make_load_skill

REPEAT_WARN = 3
REPEAT_STOP = 5
COMPACTION_OUTPUT_TOKENS = 4_096


@dataclass
class RunResult:
    final_text: str
    turns: int
    stop_reason: str
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class CompactResult:
    status: str
    before_tokens: int
    after_tokens: int
    covered_count: int = 0
    error: str = ""


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
        self.tools.register(make_load_skill(self.session.skills))
        self.approver = None
        self._repeat_counts: dict[str, int] = {}
        self._compaction_attempts = 0
        self._operation_lock = Lock()
        self._restore_context_checkpoint()

    @property
    def messages(self) -> list[Message]:
        return self.session.messages

    @property
    def usage(self) -> Usage:
        return self.session.usage

    def context_snapshot(self) -> ContextSnapshot:
        return self.context.snapshot(self.session.messages)

    def skill_infos(self):
        return self.session.skills.skills

    def run_skill(
        self,
        name: str,
        task: str = "",
        on_event: EventCallback | None = None,
        cancel: Event | None = None,
    ) -> RunResult:
        """Activate one project Skill at an idle boundary, then run its task."""
        with self._operation_lock:
            flag = cancel or self.session.cancel
            flag.clear()
            loaded = self.session.skills.load(name, task)
            if not loaded.ok:
                text = loaded.to_message_content()
                emit(on_event, "error", {"message": text})
                return RunResult(text, 0, "skill_error")
            token = (name or "").strip().lower()
            source = Message(role="user", content=loaded.content)
            if not self.session.append_skill_source(source, token, loaded.content):
                text = "Stopped: the Skill activation could not be persisted or exceeds its budget."
                emit(on_event, "error", {"message": text})
                return RunResult(text, 0, "skill_error")
            self._restore_skill_pins()
            emit(on_event, "skill", {"name": token, "trigger": "manual"})
            request = task.strip() or "Apply the activated project skill now."
            return self._run(request, on_event=on_event, cancel=flag)

    def compact(
        self,
        instruction: str = "",
        on_event: EventCallback | None = None,
        cancel: Event | None = None,
    ) -> CompactResult:
        """Manually compact at an idle boundary without adding a chat message."""
        if not self._operation_lock.acquire(blocking=False):
            current = self.context_snapshot().estimated_tokens
            return CompactResult(
                "busy",
                current,
                current,
                error="wait for the active turn to finish",
            )
        flag = cancel or self.session.cancel
        try:
            flag.clear()
            result = self._compact_once(
                self.session.messages,
                flag,
                instruction=instruction,
            )
            if result.status == "compacted":
                emit(
                    on_event,
                    "compact",
                    {
                        "before_tokens": result.before_tokens,
                        "after_tokens": result.after_tokens,
                        "covered_count": result.covered_count,
                        "semantic": True,
                        "trigger": "manual",
                    },
                )
            return result
        finally:
            self._operation_lock.release()

    def new_session(self) -> None:
        self.session.start_new()
        self.context.reset()
        self.tools.register(make_load_skill(self.session.skills))
        self.reset_turn_state(clear_observer=True)

    def attach_session(self, session: Session) -> None:
        self.session = session
        self.context.reset()
        self.tools.register(make_load_skill(self.session.skills))
        self._restore_context_checkpoint()
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
        with self._operation_lock:
            return self._run(task, on_event=on_event, cancel=cancel)

    def _run(
        self,
        task: str,
        on_event: EventCallback | None = None,
        cancel: Event | None = None,
    ) -> RunResult:
        flag = cancel or self.session.cancel
        flag.clear()
        self._repeat_counts = {}
        self._compaction_attempts = 0
        self.session.append(Message(role="user", content=task))
        consecutive_errors = 0

        for turn in range(1, self.config.max_turns + 1):
            if _cancelled(flag):
                return self._stop("cancelled", turn, "Stopped: cancelled.", on_event)
            emit(on_event, "turn", {"turn": turn, "max_turns": self.config.max_turns})
            log = self.session.messages
            view, context_status = self._prepare_model_view(log, on_event, flag)
            if context_status == "cancelled":
                return self._stop("cancelled", turn, "Stopped: cancelled.", on_event)
            if view is None:
                return self._stop(
                    "context_overflow",
                    turn,
                    "Stopped: the current request is too large for the configured context budget.",
                    on_event,
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
                is_skill = call.name == "load_skill" and result.ok
                raw = result.to_message_content()
                text = raw if is_skill else self.context.ingest_tool_result(raw, call_id=call.id)
                if result.state_changed:
                    self._repeat_counts.clear()
                text, halt = self._note_repeated_call(call, text)
                message = Message(role="tool", content=text, tool_call_id=call.id)
                if is_skill:
                    name = str(call.arguments.get("name") or "").strip().lower()
                    if self.session.append_skill_source(message, name, text):
                        self._restore_skill_pins()
                        emit(on_event, "skill", {"name": name, "trigger": "model"})
                    else:
                        result = ToolResult.fail(
                            "skill_activation_failed",
                            "Skill activation could not be persisted or exceeds its budget.",
                        )
                        text = result.to_message_content()
                        message = Message(role="tool", content=text, tool_call_id=call.id)
                        self.session.append(message)
                else:
                    self.session.append(message)
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

    def _prepare_model_view(
        self,
        log: list[Message],
        on_event: EventCallback | None,
        flag: Event | None,
    ) -> tuple[list[Message] | None, str]:
        before_tokens = self.context.snapshot(log).estimated_tokens
        had_checkpoint = self.context.checkpoint is not None
        view = self.context.prepare(log, preserve_history=True)
        semantic = False

        if self.context.should_compact(view) and self._compaction_attempts == 0:
            self._compaction_attempts += 1
            compacted = self._compact_once(log, flag)
            if compacted.status == "cancelled":
                return None, "cancelled"
            if compacted.status == "compacted":
                semantic = True
                view = self.context.prepare(log, preserve_history=True)

        if not self.context.fits(view):
            view = self.context.prepare(log)
        if not self.context.fits(view):
            return None, "context_overflow"

        after_tokens = self.context.estimate(view)
        if semantic or (not had_checkpoint and after_tokens < before_tokens):
            emit(
                on_event,
                "compact",
                {
                    "before_tokens": before_tokens,
                    "after_tokens": after_tokens,
                    "covered_count": (
                        self.context.checkpoint[1] if self.context.checkpoint else 0
                    ),
                    "semantic": semantic,
                    "trigger": "auto" if semantic else "cheap",
                },
            )
        return view, "ok"

    def _compact_once(
        self,
        log: list[Message],
        flag: Event | None,
        *,
        instruction: str = "",
    ) -> CompactResult:
        before = self.context.snapshot(log).estimated_tokens
        request = self.context.compaction_request(log, instruction=instruction)
        if request is None:
            return CompactResult("nothing_to_compact", before, before)
        try:
            response = self.llm.complete(
                request.messages,
                [],
                stream=False,
                thinking=False,
                cancel=flag,
                max_tokens=COMPACTION_OUTPUT_TOKENS,
            )
        except KeyboardInterrupt:
            if flag is not None:
                flag.set()
            return CompactResult("cancelled", before, before)
        except Exception as exc:
            if type(exc).__name__ == "LLMCancelled" or _cancelled(flag):
                return CompactResult("cancelled", before, before)
            return CompactResult("failed", before, before, error=str(exc))

        self.session.usage.add(response.usage)
        if _cancelled(flag):
            return CompactResult("cancelled", before, before)
        if response.tool_calls:
            return CompactResult(
                "failed", before, before, error="summary returned tool calls"
            )
        if complete_without_tools(response.finish_reason) == "length":
            return CompactResult("failed", before, before, error="summary was truncated")
        summary = normalize_summary(response.content or "")
        previous_covered = self.context.checkpoint[1] if self.context.checkpoint else 1
        if not summary or request.covered_count <= previous_covered:
            return CompactResult("failed", before, before, error="summary was empty or stale")

        checkpoint = self.session.save_compaction(
            summary,
            request.covered_count,
            usage=response.usage,
            model=self.config.model,
        )
        if checkpoint is None:
            return CompactResult("failed", before, before, error="checkpoint could not be saved")
        self.context.restore_checkpoint(checkpoint.summary, checkpoint.covered_count)
        after = self.context.snapshot(log).estimated_tokens
        return CompactResult(
            "compacted",
            before,
            after,
            covered_count=checkpoint.covered_count,
        )

    def _restore_context_checkpoint(self) -> None:
        item = getattr(self.session, "compaction", None)
        if item is not None:
            self.context.restore_checkpoint(item.summary, item.covered_count)
        self._restore_skill_pins()

    def _restore_skill_pins(self) -> None:
        pins = [
            PinnedContext(item.name, item.content, item.source_index)
            for item in self.session.active_skills.values()
        ]
        self.context.set_skill_pins(pins)

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
