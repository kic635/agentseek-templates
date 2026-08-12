"""Verify an AgentSeek API release imports at the declared dependency floor."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def _non_empty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must not be empty")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentseek-api-version", required=True, type=_non_empty)
    parser.add_argument("--pydantic-settings-version", required=True, type=_non_empty)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="agentseek-api-floor-") as directory:
        root = Path(directory)
        subprocess.run(["uv", "venv", str(root / ".venv")], check=True)
        python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                f"agentseek-api=={args.agentseek_api_version}",
                f"pydantic-settings=={args.pydantic_settings_version}",
            ],
            check=True,
        )
        subprocess.run(
            [str(python), "-c", "import agentseek_api; from agentseek_api.main import app"],
            check=True,
        )
    print("AgentSeek API CLI imports at the declared pydantic-settings floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
