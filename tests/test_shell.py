import base64
import os
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from anvil.llm.types import ToolCall
from anvil.tools.base import ToolRegistry
from anvil.tools.shell import make_run_shell


def _call(command: str) -> ToolCall:
    return ToolCall(
        id="shell-test",
        name="run_shell",
        arguments={"command": command},
        arguments_raw="",
    )


def _python_command(code: str) -> str:
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode('{encoded}'))"
    if os.name == "nt":
        return f'& "{sys.executable}" -c "{runner}"'
    return shlex.join([sys.executable, "-c", runner])


def _registry(workspace: Path, timeout: float = 5) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(make_run_shell(workspace, timeout=timeout, registry=registry))
    return registry


def test_shell_captures_successful_output(tmp_path: Path) -> None:
    result = _registry(tmp_path).execute(
        _call(
            _python_command(
                "import sys; print('shell-ok'); print('shell-warning', file=sys.stderr)"
            )
        )
    )

    assert result.ok is True
    assert result.error_code is None
    assert "exit_code: 0" in result.content
    assert "shell-ok" in result.content
    assert "stderr:\nshell-warning" in result.content


def test_shell_nonzero_exit_is_a_command_failure(tmp_path: Path) -> None:
    result = _registry(tmp_path).execute(_call("exit 7"))

    assert result.ok is False
    assert result.error_code == "command_failed"
    assert "exit_code: 7" in result.content


def test_shell_timeout_returns_promptly(tmp_path: Path) -> None:
    started = time.monotonic()
    result = _registry(tmp_path, timeout=0.2).execute(
        _call(_python_command("import time; time.sleep(5)"))
    )

    assert result.ok is False
    assert result.error_code == "command_timeout"
    assert time.monotonic() - started < 4


def test_shell_cancel_kills_the_child_process_tree(tmp_path: Path) -> None:
    marker = tmp_path / "late-marker.txt"
    code = (
        "import time; from pathlib import Path; "
        f"time.sleep(1); Path({str(marker)!r}).write_text('late', encoding='utf-8')"
    )
    registry = _registry(tmp_path, timeout=10)
    cancel = Event()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(registry.execute_batch, [_call(_python_command(code))], cancel)
        time.sleep(0.2)
        cancel.set()
        result = future.result(timeout=5)[0]

    assert result.ok is False
    assert result.error_code == "cancelled"
    time.sleep(1.2)
    assert not marker.exists()


def test_shell_keeps_large_output_for_context_spill(tmp_path: Path) -> None:
    result = _registry(tmp_path).execute(
        _call(_python_command("print('X' * 25000)"))
    )

    assert result.ok is True
    assert "X" * 25000 in result.content
    assert "...[truncated]..." not in result.content


def test_shell_does_not_inherit_secret_environment_variables(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("ANVIL_TEST_VISIBLE", "ordinary-setting")
    code = (
        "import os; "
        "print('secret=' + str('DEEPSEEK_API_KEY' in os.environ)); "
        "print('visible=' + os.environ.get('ANVIL_TEST_VISIBLE', ''))"
    )

    result = _registry(tmp_path).execute(_call(_python_command(code)))

    assert result.ok is True
    assert "secret=False" in result.content
    assert "visible=ordinary-setting" in result.content
    assert "must-not-reach-child" not in result.content


def test_shell_rejects_secret_file_references(tmp_path: Path) -> None:
    command = "Get-Content .env.production" if os.name == "nt" else "cat .env.production"
    result = _registry(tmp_path).execute(_call(command))

    assert result.ok is False
    assert result.error_code == "secret_file"
    assert ".env.production" not in result.content


def test_shell_allows_template_file_references(tmp_path: Path) -> None:
    result = _registry(tmp_path).execute(_call("echo .env.example"))

    assert result.ok is True
    assert ".env.example" in result.content
