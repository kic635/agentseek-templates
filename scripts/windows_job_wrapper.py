"""Keep a Windows runtime command in an atomically assigned Job Object."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

JOB_CLEANUP_TIMEOUT_SECONDS = 10.0
EMPTY_TREE_MARKER_SCHEMA_VERSION = 1
NONCE_PATTERN = re.compile(r"[0-9a-f]{64}")


class _OwnedChild(Protocol):
    def wait(self) -> int: ...

    def close_remaining_tree(self, *, timeout: float) -> None: ...

    def ensure_closed(self, *, timeout: float) -> None: ...

    def terminate_and_reap(self, *, timeout: float) -> None: ...

    def close(self) -> None: ...


class _SupervisorType(Protocol):
    @staticmethod
    def start(
        command: list[str],
        *,
        env: dict[str, str],
        cwd: str,
    ) -> _OwnedChild: ...


def _write_empty_tree_marker(marker: Path, *, nonce: str, owner_pid: int) -> None:
    if not marker.is_absolute() or marker.exists() or marker.is_symlink():
        raise RuntimeError("Windows Job Object marker path is invalid")
    if not marker.parent.is_dir():
        raise RuntimeError("Windows Job Object marker parent is unavailable")
    payload = (
        json.dumps(
            {
                "nonce": nonce,
                "owner_pid": owner_pid,
                "schema_version": EMPTY_TREE_MARKER_SCHEMA_VERSION,
                "status": "empty",
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = -1
    try:
        descriptor = os.open(marker, flags, 0o600)
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Windows Job Object marker write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
    except BaseException:
        if descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        marker.unlink(missing_ok=True)
        raise


def run_owned_command(
    command: Sequence[str],
    *,
    marker: Path,
    nonce: str,
    environment: Mapping[str, str],
    cwd: str,
    supervisor_type: _SupervisorType,
    owner_pid: int,
) -> int:
    if not command or not all(isinstance(item, str) and item for item in command):
        raise RuntimeError("Windows Job Object wrapper received an invalid command")
    if NONCE_PATTERN.fullmatch(nonce) is None:
        raise RuntimeError("Windows Job Object wrapper received an invalid nonce")
    if type(owner_pid) is not int or owner_pid <= 0:
        raise RuntimeError("Windows Job Object wrapper received an invalid owner PID")

    child = supervisor_type.start(list(command), env=dict(environment), cwd=cwd)
    failure: BaseException | None = None
    exit_code: int | None = None
    try:
        exit_code = child.wait()
        child.close_remaining_tree(timeout=JOB_CLEANUP_TIMEOUT_SECONDS)
    except BaseException as exc:
        failure = exc
        with contextlib.suppress(BaseException):
            child.terminate_and_reap(timeout=JOB_CLEANUP_TIMEOUT_SECONDS)
    try:
        child.ensure_closed(timeout=JOB_CLEANUP_TIMEOUT_SECONDS)
    except BaseException as exc:
        if failure is None:
            failure = exc
    try:
        child.close()
    except BaseException as exc:
        if failure is None:
            failure = exc

    if failure is not None:
        raise failure
    if type(exit_code) is not int:
        raise RuntimeError("Windows Job Object child returned an invalid exit code")
    _write_empty_tree_marker(marker, nonce=nonce, owner_pid=owner_pid)
    return exit_code


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("marker", type=Path)
    parser.add_argument("nonce")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.command:
        parser.error("a child command is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    if os.name != "nt":
        raise RuntimeError("Windows Job Object wrapper requires Windows")
    args = _parse_args(argv)
    from agentseek_api.process_supervisor import ForegroundChildSupervisor

    return run_owned_command(
        args.command,
        marker=args.marker,
        nonce=args.nonce,
        environment=os.environ,
        cwd=os.getcwd(),
        supervisor_type=ForegroundChildSupervisor,
        owner_pid=os.getpid(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
