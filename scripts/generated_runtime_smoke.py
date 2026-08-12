"""Run the offline lifecycle smoke against one freshly rendered template."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from cookiecutter.main import cookiecutter

ROOT = Path(__file__).resolve().parents[1]
FAKE_PROVIDER = ROOT / "scripts" / "fake_openai_server.py"


def _request(url: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("content-type", "application/json")
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - loopback smoke target
        return json.load(response)


def _wait_for_api(
    base_url: str,
    process: subprocess.Popen[bytes],
    provider: subprocess.Popen[bytes],
    timeout: float = 60,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"agentseek dev exited with status {process.returncode}\n{_read_log(process)}")
        if provider.poll() is not None:
            raise RuntimeError(f"fake provider exited with status {provider.returncode}\n{_read_log(provider)}")
        try:
            _request(f"{base_url}/health")
            return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError(f"API did not become ready at {base_url}/health; process={process.poll()}\n{_read_log(process)}")


def _read_log(process: subprocess.Popen[bytes]) -> str:
    log_path = getattr(process, "_agentseek_log_path", None)
    if not log_path:
        return "(process log unavailable)"
    path = Path(log_path)
    return path.read_text(encoding="utf-8", errors="replace")[-12000:] if path.exists() else "(log file unavailable)"


def _run_checked(command: list[str], *, cwd: Path, env: dict[str, str], template: str) -> None:
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{template}: command failed with status {error.returncode}: {command}") from error


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        # The child can exit between poll() and killpg(). Its process group
        # disappearing means cleanup has already completed.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    template_root = ROOT / "templates" / args.template
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="template-smoke-", dir=args.output_root))
    generated = Path(cookiecutter(template=str(template_root), output_dir=str(run_root), no_input=True))
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.update(
        {
            "AGENTSEEK_MODEL_PROVIDER": "openai",
            "AGENTSEEK_MODEL": "gpt-4o-mini",
            "AGENTSEEK_API_KEY": "smoke-test-key",
            "AGENTSEEK_API_BASE": "http://127.0.0.1:2025/v1",
            "OPENAI_API_KEY": "smoke-test-key",
            "OPENAI_API_BASE": "http://127.0.0.1:2025/v1",
            "OPENAI_MODEL": "gpt-4o-mini",
            "SENTINEL": "from-shell",
        }
    )
    env_file = generated / ".env"
    env_file.write_text(
        (generated / ".env.example").read_text(encoding="utf-8")
        + "\nSENTINEL=from-dotenv\n"
        + "OPENAI_API_KEY=smoke-test-key\nOPENAI_API_BASE=http://127.0.0.1:2025/v1\n",
        encoding="utf-8",
    )

    try:
        _run_checked(["uv", "sync"], cwd=generated, env=env, template=args.template)
    except RuntimeError as error:
        raise RuntimeError(f"{args.template}: generated dependency installation failed: {error}") from error
    frontend = generated / "frontend" / "package.json"
    if frontend.is_file():
        _run_checked(["npm", "install", "--prefix", "frontend"], cwd=generated, env=env, template=args.template)

    log_dir = Path(tempfile.mkdtemp(prefix="template-smoke-logs-"))
    provider_log = log_dir / "fake-provider.log"
    lifecycle_log = log_dir / "agentseek-dev.log"
    provider_stream = provider_log.open("w", encoding="utf-8")
    provider = subprocess.Popen(
        [sys.executable, str(FAKE_PROVIDER)], cwd=generated, env=env, stdout=provider_stream, stderr=subprocess.STDOUT
    )
    provider._agentseek_log_path = str(provider_log)  # type: ignore[attr-defined]
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    lifecycle_stream = lifecycle_log.open("w", encoding="utf-8")
    lifecycle = subprocess.Popen(
        ["uv", "run", "agentseek", "dev", "--skip-check"],
        cwd=generated,
        env=env,
        stdout=lifecycle_stream,
        stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt",
        creationflags=creationflags,
    )
    lifecycle._agentseek_log_path = str(lifecycle_log)  # type: ignore[attr-defined]
    try:
        if provider.poll() is not None:
            raise RuntimeError(
                f"{args.template}: fake provider exited with status {provider.returncode}\n{_read_log(provider)}"
            )
        base_url = "http://127.0.0.1:2024"
        _wait_for_api(base_url, lifecycle, provider)
        assistants = _request(f"{base_url}/assistants/search", method="POST", payload={"limit": 1})
        assistant_id = (
            assistants[0]["assistant_id"] if isinstance(assistants, list) else assistants["data"][0]["assistant_id"]
        )
        thread = _request(f"{base_url}/threads", method="POST", payload={"metadata": {"suite": "template-matrix"}})
        run = _request(
            f"{base_url}/threads/{thread['thread_id']}/runs",
            method="POST",
            payload={"assistant_id": assistant_id, "input": {"message": "hello smoke"}},
        )
        result = _request(f"{base_url}/threads/{thread['thread_id']}/runs/{run['run_id']}/wait")
        if result.get("status") != "success":
            raise RuntimeError(f"run did not succeed: {result}")
        print(json.dumps({"template": args.template, "status": result["status"]}))
    finally:
        _terminate(lifecycle)
        _terminate(provider)
        lifecycle_stream.close()
        provider_stream.close()
        print(f"smoke logs: {log_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
