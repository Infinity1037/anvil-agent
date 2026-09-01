from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one local tool call. Handlers never throw to the model loop."""

    ok: bool
    content: str
    error_code: str | None = None
    hint: str | None = None
    state_changed: bool = False

    @classmethod
    def success(cls, content: str, *, state_changed: bool = False) -> ToolResult:
        return cls(ok=True, content=content, state_changed=state_changed)

    @classmethod
    def fail(cls, error_code: str, content: str, hint: str | None = None) -> ToolResult:
        return cls(ok=False, content=content, error_code=error_code, hint=hint)

    def to_message_content(self) -> str:
        if self.ok:
            return self.content
        code = self.error_code or "tool_error"
        lines = [f"Error ({code}): {self.content}"]
        if self.hint:
            lines.append(f"Hint: {self.hint}")
        return "\n".join(lines)
