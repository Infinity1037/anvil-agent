import os
import subprocess
import sys


def test_utf8_output_survives_a_gbk_parent_encoding() -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "gbk"
    code = (
        "from anvil.cli import _ensure_utf8_output; "
        "_ensure_utf8_output(); "
        "from rich.console import Console; "
        "Console().print('✓ 中文')"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert "✓ 中文" in result.stdout.decode("utf-8")
