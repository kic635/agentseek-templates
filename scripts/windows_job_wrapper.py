"""Keep a Windows runtime command in an atomically assigned Job Object."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, Protocol

JOB_CLEANUP_TIMEOUT_SECONDS = 10.0
EMPTY_TREE_MARKER_SCHEMA_VERSION = 1
NONCE_PATTERN = re.compile(r"[0-9a-f]{64}")
NONCE_PAYLOAD_BYTES = 65
CLEANUP_REQUEST_PAYLOAD = b"cleanup\n"
CLEANUP_REQUEST_POLL_SECONDS = 0.05


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


def _read_nonce(stream: BinaryIO) -> str:
    try:
        payload = stream.read(NONCE_PAYLOAD_BYTES + 1)
    except BaseException as exc:
        raise RuntimeError("Windows Job Object nonce input is unavailable") from exc
    if not isinstance(payload, bytes) or len(payload) != NONCE_PAYLOAD_BYTES or payload[-1:] != b"\n":
        raise RuntimeError("Windows Job Object nonce input is invalid")
    try:
        nonce = payload[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Windows Job Object nonce input is invalid") from exc
    if NONCE_PATTERN.fullmatch(nonce) is None:
        raise RuntimeError("Windows Job Object nonce input is invalid")
    return nonce


def _cleanup_request_path(marker: Path) -> Path:
    return marker.with_name(f"{marker.name}.cleanup-request")


def _consume_cleanup_request(marker: Path) -> bool:
    request = _cleanup_request_path(marker)
    try:
        payload = request.read_bytes()
    except OSError:
        return False
    if payload != CLEANUP_REQUEST_PAYLOAD:
        return False
    try:
        request.unlink()
    except OSError:
        return False
    return True


def _wait_for_child_or_cleanup(child: _OwnedChild, marker: Path) -> tuple[int, bool]:
    completed = threading.Event()
    results: list[int] = []
    failures: list[BaseException] = []

    def wait_for_child() -> None:
        try:
            results.append(child.wait())
        except BaseException as exc:
            failures.append(exc)
        finally:
            completed.set()

    waiter = threading.Thread(target=wait_for_child, name="agentseek-runtime-wait", daemon=True)
    waiter.start()
    cleanup_requested = False
    while not completed.wait(CLEANUP_REQUEST_POLL_SECONDS):
        if _consume_cleanup_request(marker):
            cleanup_requested = True
            break

    if cleanup_requested:
        child.terminate_and_reap(timeout=JOB_CLEANUP_TIMEOUT_SECONDS)
        if not completed.wait(JOB_CLEANUP_TIMEOUT_SECONDS):
            raise RuntimeError("Windows Job Object child wait did not finish")
    waiter.join(timeout=0)
    if failures:
        raise failures[0]
    if len(results) != 1 or type(results[0]) is not int:
        raise RuntimeError("Windows Job Object child returned an invalid exit code")
    return (0 if cleanup_requested else results[0]), cleanup_requested


def run_owned_command(
    command: Sequence[str],
    *,
    marker: Path,
    nonce_stream: BinaryIO,
    environment: Mapping[str, str],
    cwd: str,
    supervisor_type: _SupervisorType,
    owner_pid: int,
) -> int:
    if not command or not all(isinstance(item, str) and item for item in command):
        raise RuntimeError("Windows Job Object wrapper received an invalid command")
    if type(owner_pid) is not int or owner_pid <= 0:
        raise RuntimeError("Windows Job Object wrapper received an invalid owner PID")
    nonce = _read_nonce(nonce_stream)

    child = supervisor_type.start(list(command), env=dict(environment), cwd=cwd)
    failure: BaseException | None = None
    exit_code: int | None = None
    try:
        exit_code, cleanup_requested = _wait_for_child_or_cleanup(child, marker)
        if not cleanup_requested:
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
        nonce_stream=sys.stdin.buffer,
        environment=os.environ,
        cwd=os.getcwd(),
        supervisor_type=ForegroundChildSupervisor,
        owner_pid=os.getpid(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
