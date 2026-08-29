from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from anvil.safety import DangerousCommandError, assert_safe_command
from anvil.tools.base import ToolSpec


def detect_shell() -> tuple[str, list[str]]:
    if os.name == "nt":
        if shutil.which("pwsh"):
            return "pwsh", ["pwsh", "-NoProfile", "-NonInteractive", "-Command"]
        return "powershell", ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
    if shutil.which("bash"):
        return "bash", ["bash", "-lc"]
    return "sh", ["sh", "-c"]


def make_run_shell(workspace: Path, timeout: float) -> ToolSpec:
    shell_name, prefix = detect_shell()

    def run_shell(command: str) -> str:
        if not command or not command.strip():
            return "Error: command must not be empty."
        try:
            assert_safe_command(command)
        except DangerousCommandError as exc:
            return f"Error: {exc}"
        try:
            completed = subprocess.run(
                prefix + [command],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:
            partial = ((exc.stdout or "") + (exc.stderr or "")).strip()
            tail = partial[-2000:] if partial else "(no output captured)"
            return (
                f"Error: command timed out after {timeout:.0f}s.\n"
                f"Partial output:\n{tail}\n"
                "Tip: avoid interactive prompts; chain commands with && because cwd resets each call."
            )
        except OSError as exc:
            return f"Error: failed to start shell ({shell_name}): {exc}"

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        body = "".join(
            part
            for part in (
                f"exit_code: {completed.returncode}\n",
                f"stdout:\n{stdout}" if stdout else "stdout: (empty)\n",
                f"stderr:\n{stderr}" if stderr else "",
            )
        ).rstrip()
        if len(body) > 20_000:
            body = body[:12_000] + "\n...[truncated]...\n" + body[-6_000:]
        return body

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
