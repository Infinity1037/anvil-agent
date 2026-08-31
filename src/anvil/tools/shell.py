from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from anvil.safety import (
    InternalPathError,
    assert_command_has_no_secret_reference,
    assert_safe_command,
)
from anvil.tools.base import ToolSpec
from anvil.tools.result import ToolResult

if TYPE_CHECKING:
    from anvil.tools.base import ToolRegistry


_SENSITIVE_ENV_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|"
    r"CREDENTIALS?|AUTH(?:ORIZATION)?|COOKIE)(?:_|$)",
    re.IGNORECASE,
)


def _sanitized_shell_env() -> dict[str, str]:
    """Pass through ordinary process settings without exposing credentials."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not _SENSITIVE_ENV_NAME.search(key)
    }
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


class _WindowsKillJob:
    """A Windows job whose close operation kills the assigned process tree."""

    def __init__(self, proc: subprocess.Popen) -> None:
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimits),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise OSError(error, "SetInformationJobObject failed")
        if not kernel32.AssignProcessToJobObject(handle, int(proc._handle)):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise OSError(error, "AssignProcessToJobObject failed")
        self._kernel32 = kernel32
        self._handle = handle

    def close(self) -> None:
        handle = self._handle
        if handle:
            self._handle = None
            self._kernel32.CloseHandle(handle)


def _attach_kill_job(proc: subprocess.Popen) -> _WindowsKillJob | None:
    if os.name != "nt":
        return None
    try:
        return _WindowsKillJob(proc)
    except OSError:
        return None


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


def _stop_process(proc: subprocess.Popen, kill_job: _WindowsKillJob | None = None) -> None:
    if proc.poll() is not None:
        return
    if kill_job is not None:
        kill_job.close()
    elif os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        if proc.poll() is None:
            proc.kill()
    except OSError:
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
        assert_command_has_no_secret_reference(command)
        assert_safe_command(command)
        if _mentions_anvil(command):
            raise InternalPathError(".anvil holds session logs, not project source.")
        env = _sanitized_shell_env()
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
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    if os.name == "nt"
                    else 0
                ),
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            return ToolResult.fail(
                "exception",
                f"failed to start shell ({shell_name}): {exc}",
            )
        kill_job = _attach_kill_job(proc)
        deadline = time.monotonic() + timeout
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                waiter = pool.submit(proc.communicate)
                while not waiter.done():
                    cancel = None if registry is None else registry.cancel
                    if cancel is not None and cancel.is_set():
                        _stop_process(proc, kill_job)
                        try:
                            waiter.result(timeout=2)
                        except Exception:
                            pass
                        return ToolResult.fail("cancelled", f"cancelled: {command}")
                    if time.monotonic() > deadline:
                        _stop_process(proc, kill_job)
                        try:
                            waiter.result(timeout=2)
                        except Exception:
                            pass
                        return ToolResult.fail(
                            "command_timeout",
                            f"command timed out after {timeout:g}s.\n$ {command}",
                            hint="Avoid interactive prompts; chain commands because cwd resets each call.",
                        )
                    time.sleep(0.05)
                stdout, stderr = waiter.result()
        finally:
            if kill_job is not None:
                kill_job.close()
        body = "".join(
            part
            for part in (
                f"$ {command}\n",
                f"exit_code: {proc.returncode}\n",
                f"stdout:\n{stdout}" if stdout else "stdout: (empty)\n",
                f"stderr:\n{stderr}" if stderr else "",
            )
        ).rstrip()
        if proc.returncode:
            return ToolResult.fail(
                "command_failed",
                body,
                hint="Inspect stdout/stderr, fix the cause, and run a relevant command again.",
            )
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
