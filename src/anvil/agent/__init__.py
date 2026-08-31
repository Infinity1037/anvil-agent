"""Agent runtime with lazy exports so session helpers can import prompts safely."""

from __future__ import annotations

__all__ = ["Agent", "CompactResult", "RunResult"]


def __getattr__(name: str):
    if name in {"Agent", "CompactResult", "RunResult"}:
        from anvil.agent.loop import Agent, CompactResult, RunResult

        return {"Agent": Agent, "CompactResult": CompactResult, "RunResult": RunResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
