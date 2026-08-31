from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from anvil.safety import InternalPathError, assert_safe_command
from anvil.tools.base import ToolSpec
from anvil.tools.result import ToolResult

if TYPE_CHECKING:
    from anvil.tools.base import ToolRegistry


def detect_shell() -> tuple[str, list[str]]:
    if os.name == "nt":
        if shutil.which("pwsh"):
            return "pwsh", ["pwsh", "-NoProfile", "-NonInteractive", "-Command"]
        return "powershell", ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
    if shutil.which("bash"):
        return "bash", ["bash", "-lc"]
    return "sh", ["sh", "-c"]


def _mentions_anvil(command: str) -> bool:
    normalized = command.replace("\\", "/")
    return "/.anvil/" in f"/{normalized}" or normalized.strip().startswith(".anvil")


def _stop_process(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.communicate(timeout=2)
    except Exception:
        pass


def make_run_shell(
    workspace: Path,
    timeout: float,
    registry: ToolRegistry | None = None,
) -> ToolSpec:
    shell_name, prefix = detect_shell()

    def run_shell(command: str) -> ToolResult:
        if not command or not command.strip():
            return ToolResult.fail("empty_query", "command must not be empty.")
        assert_safe_command(command)
        if _mentions_anvil(command):
            raise InternalPathError(".anvil holds session logs, not project source.")
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            proc = subprocess.Popen(
                prefix + [command],
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except OSError as exc:
            return ToolResult.fail(
                "exception",
                f"failed to start shell ({shell_name}): {exc}",
            )
        deadline = time.monotonic() + timeout
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiter = pool.submit(proc.communicate)
            while not waiter.done():
                cancel = None if registry is None else registry.cancel
                if cancel is not None and cancel.is_set():
                    _stop_process(proc)
                    try:
                        waiter.result(timeout=2)
                    except Exception:
                        pass
                    return ToolResult.fail("cancelled", f"cancelled: {command}")
                if time.monotonic() > deadline:
                    _stop_process(proc)
                    try:
                        waiter.result(timeout=2)
                    except Exception:
                        pass
                    return ToolResult.fail(
                        "command_timeout",
                        f"command timed out after {timeout:.0f}s.\n$ {command}",
                        hint="Avoid interactive prompts; chain commands because cwd resets each call.",
                    )
                time.sleep(0.05)
            stdout, stderr = waiter.result()
        body = "".join(
            part
            for part in (
                f"$ {command}\n",
                f"exit_code: {proc.returncode}\n",
                f"stdout:\n{stdout}" if stdout else "stdout: (empty)\n",
                f"stderr:\n{stderr}" if stderr else "",
            )
        ).rstrip()
        if len(body) > 20_000:
            body = body[:12_000] + "\n...[truncated]...\n" + body[-6_000:]
        return ToolResult.success(body)

    return ToolSpec(
        name="run_shell",
        description=(
            f"Run a command in the workspace using {shell_name}. "
            "Each call starts in the workspace root; cd does not persist. "
            "Use this for tests, git, and language tooling."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute.",
                }
            },
            "required": ["command"],
        },
        handler=run_shell,
        parallel_safe=False,
    )
