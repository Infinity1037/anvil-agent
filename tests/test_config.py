from pathlib import Path

import pytest

from anvil.config import Config, ConfigError


_CONTEXT_ENV = (
    "ANVIL_MAX_TOKENS",
    "ANVIL_CONTEXT_WINDOW",
    "ANVIL_CONTEXT_BUDGET",
)


def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    for name in _CONTEXT_ENV:
        monkeypatch.delenv(name, raising=False)


def test_deepseek_v4_context_defaults_are_distinct(monkeypatch, tmp_path: Path) -> None:
    _isolated_env(monkeypatch, tmp_path)

    config = Config.from_env(tmp_path)

    assert config.context_window == 200_000
    assert config.context_budget == 160_000
    assert config.max_tokens == 32_000
    assert config.context_budget + config.max_tokens < config.context_window


def test_context_budget_and_output_must_fit_context_window(
    monkeypatch, tmp_path: Path
) -> None:
    _isolated_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ANVIL_CONTEXT_WINDOW", "100000")
    monkeypatch.setenv("ANVIL_CONTEXT_BUDGET", "80000")
    monkeypatch.setenv("ANVIL_MAX_TOKENS", "30000")

    with pytest.raises(ConfigError, match="must not exceed"):
        Config.from_env(tmp_path)


def test_context_values_reject_non_integer_environment(
    monkeypatch, tmp_path: Path
) -> None:
    _isolated_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ANVIL_CONTEXT_WINDOW", "one-million")

    with pytest.raises(ConfigError, match="must be a positive integer"):
        Config.from_env(tmp_path)
