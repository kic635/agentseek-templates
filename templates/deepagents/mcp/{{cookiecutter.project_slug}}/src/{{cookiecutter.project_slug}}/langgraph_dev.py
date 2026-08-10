"""Compatibility launcher for the AgentSeek API development runtime."""

from __future__ import annotations

import os
import shutil
import subprocess


def main() -> int:
    """Run the AgentSeek API development server with the configured bind host."""
    executable = shutil.which("agentseek-api")
    if executable is None:
        raise SystemExit("agentseek-api is unavailable; run `uv sync` first")
    host = os.environ.get("LANGGRAPH_HOST") or "127.0.0.1"
    command = [
        executable,
        "dev",
        "--host",
        host,
        "--port",
        "{{ cookiecutter.langgraph_port }}",
    ]
    return subprocess.call(command)  # noqa: S603 - resolved project dependency with a fixed argv shape


if __name__ == "__main__":
    raise SystemExit(main())
