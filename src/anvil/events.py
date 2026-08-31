from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


EventCallback = Callable[[AgentEvent], None]


def emit(on_event: EventCallback | None, kind: str, payload: dict[str, Any] | None = None) -> None:
    if on_event:
        on_event(AgentEvent(kind=kind, payload=payload or {}))
