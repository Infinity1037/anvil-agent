from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

EFFORT_LEVELS = ("low", "high", "max")


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs without overwriting existing environment variables."""
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _truthy(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw not in {None, ""} else default
    except ValueError as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Config:
    api_key: str
    base_url: str
    model: str
    thinking: bool
    reasoning_effort: str
    max_turns: int
    max_tokens: int
    context_budget: int
    request_timeout: float
    shell_timeout: float
    workspace: Path
    context_window: int = 200_000

    @classmethod
    def from_env(
        cls,
        workspace: Path,
        *,
        model: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        max_turns: int | None = None,
    ) -> Config:
        workspace = workspace.resolve()
        load_dotenv(Path.cwd() / ".env")
        load_dotenv(workspace / ".env")

        api_key = (
            os.environ.get("ANVIL_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        if not api_key:
            raise ConfigError(
                "Missing API key. Set DEEPSEEK_API_KEY (or ANVIL_API_KEY) "
                "in the environment or a gitignored .env file."
            )

        effort = parse_effort(reasoning_effort or os.environ.get("ANVIL_REASONING_EFFORT") or "max")

        max_tokens_value = _positive_int_env("ANVIL_MAX_TOKENS", 32_000)
        context_window_value = _positive_int_env("ANVIL_CONTEXT_WINDOW", 200_000)
        context_budget_value = _positive_int_env("ANVIL_CONTEXT_BUDGET", 160_000)
        if context_budget_value + max_tokens_value > context_window_value:
            raise ConfigError(
                "ANVIL_CONTEXT_BUDGET + ANVIL_MAX_TOKENS must not exceed "
                "ANVIL_CONTEXT_WINDOW"
            )

        return cls(
            api_key=api_key,
            base_url=(os.environ.get("ANVIL_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
            model=model or os.environ.get("ANVIL_MODEL") or "deepseek-v4-flash",
            thinking=_truthy(os.environ.get("ANVIL_THINKING"), True) if thinking is None else thinking,
            reasoning_effort=effort,
            max_turns=max_turns or int(os.environ.get("ANVIL_MAX_TURNS") or "40"),
            max_tokens=max_tokens_value,
            context_budget=context_budget_value,
            request_timeout=float(os.environ.get("ANVIL_REQUEST_TIMEOUT") or "300"),
            shell_timeout=float(os.environ.get("ANVIL_SHELL_TIMEOUT") or "60"),
            workspace=workspace,
            context_window=context_window_value,
        )


def parse_effort(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value not in EFFORT_LEVELS:
        raise ConfigError("reasoning effort must be low, high, or max")
    return value


def effort_label(thinking: bool, effort: str) -> str:
    return "off" if not thinking else effort


def apply_effort_token(thinking: bool, effort: str, token: str) -> tuple[bool, str]:
    raw = (token or "").strip().lower()
    if not raw:
        return thinking, effort
    if raw in {"off", "none", "disable", "disabled"}:
        return False, effort
    if raw in EFFORT_LEVELS:
        return True, raw
    raise ValueError("use /effort off | low | high | max")


class ConfigError(RuntimeError):
    pass
