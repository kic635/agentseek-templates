from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generated_runtime_smoke.py"
SPEC = importlib.util.spec_from_file_location("generated_runtime_smoke", SCRIPT)
assert SPEC and SPEC.loader
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class _Process:
    pid = 1234

    def __init__(self, *, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, *, timeout: float) -> int:
        del timeout
        self.wait_calls += 1
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9


def test_terminate_ignores_process_group_race(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _Process()

    def missing_process_group(pid: int, sig: int) -> None:
        assert pid == process.pid
        assert sig == smoke.signal.SIGTERM
        raise ProcessLookupError

    monkeypatch.setattr(smoke.os, "name", "posix")
    monkeypatch.setattr(smoke.os, "killpg", missing_process_group)

    smoke._terminate(process)

    assert process.wait_calls == 1


def test_terminate_does_not_signal_process_that_already_exited(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _Process(returncode=0)
    killpg = pytest.MonkeyPatch()
    killpg.setattr(smoke.os, "killpg", lambda *_args: pytest.fail("must not signal exited process"))
    try:
        smoke._terminate(process)
    finally:
        killpg.undo()

    assert process.wait_calls == 0
