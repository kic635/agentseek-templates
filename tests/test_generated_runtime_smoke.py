from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generated_runtime_smoke.py"
SPEC = importlib.util.spec_from_file_location("generated_runtime_smoke", SCRIPT)
assert SPEC and SPEC.loader
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)

FAKE_PROVIDER_SCRIPT = SCRIPT.with_name("fake_openai_server.py")
FAKE_PROVIDER_SPEC = importlib.util.spec_from_file_location("fake_openai_server", FAKE_PROVIDER_SCRIPT)
assert FAKE_PROVIDER_SPEC and FAKE_PROVIDER_SPEC.loader
fake_provider = importlib.util.module_from_spec(FAKE_PROVIDER_SPEC)
FAKE_PROVIDER_SPEC.loader.exec_module(fake_provider)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ("127.0.0.1", 2025)),
        (["--host", "127.0.0.1", "--port", "43125"], ("127.0.0.1", 43125)),
    ],
)
def test_fake_provider_address_defaults_and_explicit_override(
    argv: list[str],
    expected: tuple[str, int],
) -> None:
    args = fake_provider._parse_args(argv)

    assert (args.host, args.port) == expected


def _assert_no_workflow_secrets(value: object) -> None:
    if isinstance(value, str):
        assert "secrets." not in value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_workflow_secrets(key)
            _assert_no_workflow_secrets(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_no_workflow_secrets(item)


class _Process:
    pid = 1234

    def __init__(self, *, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.kill_calls = 0
        self.poll_calls = 0
        self.wait_calls = 0
        self.wait_timeouts: list[float] = []

    def poll(self) -> int | None:
        self.poll_calls += 1
        return self.returncode

    def wait(self, *, timeout: float) -> int:
        self.wait_calls += 1
        self.wait_timeouts.append(timeout)
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.kill_calls += 1
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


def test_terminate_signals_owned_group_when_leader_already_exited(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _Process(returncode=0)
    signals: list[int] = []

    def kill_group(pid: int, sig: int) -> None:
        assert pid == process.pid
        signals.append(sig)
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(smoke.os, "name", "posix")
    monkeypatch.setattr(smoke.os, "killpg", kill_group)

    smoke._terminate(process)

    assert signals == [smoke.signal.SIGTERM, 0]
    assert process.wait_calls == 0


def test_posix_group_check_treats_permission_denied_as_no_owned_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process(returncode=0)

    def deny_group_check(_pid: int, sig: int) -> None:
        assert sig == 0
        raise PermissionError

    monkeypatch.setattr(smoke.os, "killpg", deny_group_check)

    assert smoke._wait_for_posix_group_exit(process, 1) is True
    assert process.poll_calls == 1


def test_posix_group_check_retries_permission_denied_until_leader_is_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process()

    def poll_until_reaped() -> int | None:
        process.poll_calls += 1
        if process.poll_calls == 1:
            return None
        process.returncode = 0
        return 0

    def deny_group_check(_pid: int, _sig: int) -> None:
        raise PermissionError

    process.poll = poll_until_reaped  # type: ignore[method-assign]
    monkeypatch.setattr(smoke.os, "killpg", deny_group_check)

    assert smoke._wait_for_posix_group_exit(process, 1) is True
    assert process.poll_calls == 2


def test_posix_group_check_reaps_zombie_leader_before_testing_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process()
    reaped = False

    def reap_leader() -> int:
        nonlocal reaped
        process.poll_calls += 1
        process.returncode = 0
        reaped = True
        return 0

    def check_group(_pid: int, sig: int) -> None:
        assert sig == 0
        if reaped:
            raise ProcessLookupError

    process.poll = reap_leader  # type: ignore[method-assign]
    monkeypatch.setattr(smoke.os, "killpg", check_group)

    assert smoke._wait_for_posix_group_exit(process, 1) is True
    assert process.poll_calls == 1


def test_terminate_allows_launcher_to_clean_nested_process_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _Process()
    wait_timeouts: list[float] = []
    monkeypatch.setattr(smoke.os, "name", "posix")
    monkeypatch.setattr(smoke.os, "killpg", lambda *_args: None)
    monkeypatch.setattr(
        smoke,
        "_wait_for_posix_group_exit",
        lambda _pid, timeout: wait_timeouts.append(timeout) or True,
        raising=False,
    )

    smoke._terminate(process)

    assert wait_timeouts[0] >= 30


def test_terminate_force_kills_entire_posix_group_after_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _Process()
    signals: list[tuple[int, int]] = []
    wait_timeouts: list[float] = []
    monkeypatch.setattr(smoke.os, "name", "posix")
    monkeypatch.setattr(smoke.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(
        smoke,
        "_wait_for_posix_group_exit",
        lambda _pid, timeout: wait_timeouts.append(timeout) or len(wait_timeouts) == 2,
        raising=False,
    )

    smoke._terminate(process)

    assert signals == [
        (process.pid, smoke.signal.SIGTERM),
        (process.pid, smoke.signal.SIGKILL),
    ]
    assert wait_timeouts[0] >= 30
    assert wait_timeouts[1] <= 10
    assert process.kill_calls == 0


def test_windows_cleanup_uses_absolute_system_taskkill_with_minimal_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    system_root = tmp_path / "Windows"
    taskkill = system_root / "System32" / "taskkill.exe"
    taskkill.parent.mkdir(parents=True)
    taskkill.touch()
    process = _Process()
    recorded: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> smoke.subprocess.CompletedProcess[bytes]:
        recorded.update(command=command, **kwargs)
        return smoke.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(smoke.subprocess, "run", run)
    monkeypatch.setattr(
        smoke.runtime_proof,
        "resolve_executable",
        lambda _name: pytest.fail("cleanup must not search ambient PATH"),
    )

    smoke._terminate_windows_process(
        process,
        {
            "SystemRoot": str(system_root),
            "PATH": str(tmp_path / "hostile-bin"),
            "SMOKE_SECRET": "must-not-propagate",
        },
    )

    assert recorded["command"] == [str(taskkill.resolve()), "/PID", str(process.pid), "/T", "/F"]
    assert recorded["env"] == {"SystemRoot": str(system_root)}
    assert recorded["timeout"] == smoke.WINDOWS_TASKKILL_TIMEOUT_SECONDS
    assert recorded["check"] is False


def test_windows_cleanup_reaps_leader_when_taskkill_validation_fails(tmp_path: Path) -> None:
    system_root = tmp_path / "Windows"
    system_root.mkdir()
    process = _Process()

    with pytest.raises(RuntimeError, match=smoke.WINDOWS_TREE_CLEANUP_ERROR):
        smoke._terminate_windows_process(process, {"SystemRoot": str(system_root)})

    assert process.kill_calls == 1
    assert process.wait_calls == 1


def test_windows_cleanup_fails_closed_when_leader_already_exited(tmp_path: Path) -> None:
    system_root = tmp_path / "Windows"
    taskkill = system_root / "System32" / "taskkill.exe"
    taskkill.parent.mkdir(parents=True)
    taskkill.touch()
    process = _Process(returncode=1)

    with pytest.raises(RuntimeError, match=smoke.WINDOWS_TREE_CLEANUP_ERROR):
        smoke._terminate_windows_process(process, {"SystemRoot": str(system_root)})


@pytest.mark.parametrize("taskkill_result", [1, subprocess.TimeoutExpired("taskkill", 1)])
def test_windows_cleanup_fails_closed_when_taskkill_cannot_prove_tree_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    taskkill_result: int | subprocess.TimeoutExpired,
) -> None:
    system_root = tmp_path / "Windows"
    taskkill = system_root / "System32" / "taskkill.exe"
    taskkill.parent.mkdir(parents=True)
    taskkill.touch()
    process = _Process()

    def run(command: list[str], **_kwargs: object) -> smoke.subprocess.CompletedProcess[bytes]:
        if isinstance(taskkill_result, BaseException):
            raise taskkill_result
        return smoke.subprocess.CompletedProcess(command, taskkill_result)

    monkeypatch.setattr(smoke.subprocess, "run", run)

    with pytest.raises(RuntimeError, match=smoke.WINDOWS_TREE_CLEANUP_ERROR):
        smoke._terminate_windows_process(process, {"SystemRoot": str(system_root)})

    assert process.kill_calls == 1
    assert process.wait_calls == 1


def test_process_cleanup_attempts_provider_after_lifecycle_error(monkeypatch: pytest.MonkeyPatch) -> None:
    lifecycle = _Process()
    provider = _Process()
    attempted: list[_Process] = []

    def terminate(process: _Process, **_kwargs: object) -> None:
        attempted.append(process)
        if process is lifecycle:
            raise RuntimeError("lifecycle cleanup failed")

    monkeypatch.setattr(smoke, "_terminate", terminate)

    with pytest.raises(RuntimeError, match="lifecycle cleanup failed"):
        smoke._terminate_processes(lifecycle, provider)

    assert attempted == [lifecycle, provider]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group ownership regression")
def test_run_checked_timeout_reaps_owned_tree_and_reports_value_free_tail(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    child_pid_path = run_root / "child.pid"
    secret = "synthetic-provider-secret"
    script = """
import os
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(sys.argv[1]).write_text(str(child.pid), encoding="utf-8")
print(os.environ["SMOKE_SECRET"] * 20000, flush=True)
time.sleep(60)
"""
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="timed out") as exc_info:
        smoke._run_checked(
            [sys.executable, "-c", script, str(child_pid_path)],
            cwd=run_root,
            env={**os.environ, "SMOKE_SECRET": secret},
            label="bounded command regression",
            run_root=run_root,
            timeout=0.25,
        )

    assert time.monotonic() - started < 10
    assert secret not in str(exc_info.value)
    assert "output tail:" in str(exc_info.value)
    assert len(str(exc_info.value)) <= smoke.COMMAND_OUTPUT_TAIL_BYTES + 256
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, smoke.signal.SIGKILL)
        pytest.fail("timed-out command descendant remained alive")

    logs = list((run_root / "command-logs").glob("*.log"))
    assert len(logs) == 1
    assert logs[0].stat().st_mode & 0o777 == 0o600
    assert logs[0].stat().st_size <= smoke.COMMAND_OUTPUT_TAIL_BYTES


def test_run_checked_caps_both_logs_and_redacts_short_sensitive_values(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    secret = "abc"
    script = """
import os
import sys

print("o" * 20000)
print("e" * 20000, file=sys.stderr)
print(f"prefix{os.environ['SHORT_SECRET']}suffix", file=sys.stderr)
raise SystemExit(3)
"""

    with pytest.raises(RuntimeError, match="failed with status 3") as exc_info:
        smoke._run_checked(
            [sys.executable, "-c", script],
            cwd=run_root,
            env={**os.environ, "SHORT_SECRET": secret},
            label="bounded redaction regression",
            run_root=run_root,
            timeout=5,
        )

    assert secret not in str(exc_info.value)
    logs = list((run_root / "command-logs").glob("*"))
    assert len(logs) == 2
    assert all(path.stat().st_size <= smoke.COMMAND_OUTPUT_TAIL_BYTES for path in logs)
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in logs)


@pytest.mark.skipif(os.name == "nt", reason="POSIX inherited-pipe regression")
def test_run_checked_fails_closed_without_blocking_on_detached_inherited_pipe(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    child_pid_path = run_root / "detached.pid"
    script = """
import subprocess
import sys
from pathlib import Path

child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(7)"],
    start_new_session=True,
)
Path(sys.argv[1]).write_text(str(child.pid), encoding="utf-8")
"""
    started = time.monotonic()

    try:
        with pytest.raises(RuntimeError, match="command cleanup could not be verified"):
            smoke._run_checked(
                [sys.executable, "-c", script, str(child_pid_path)],
                cwd=run_root,
                env=os.environ,
                label="detached inherited pipe regression",
                run_root=run_root,
                timeout=2,
            )
        assert time.monotonic() - started < 6.5
    finally:
        if child_pid_path.is_file():
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, smoke.signal.SIGKILL)


@pytest.mark.parametrize(
    ("template", "required_name"),
    [
        ("deepagents/mcp", "AGENTSEEK_MODEL_API_KEY"),
        ("deepagents/research", "TAVILY_API_KEY"),
        ("langchain/cli-remote", "BUB_API_KEY"),
        ("langchain/agentic-rag-hybrid", "SILICONFLOW_API_KEY"),
    ],
)
def test_smoke_profiles_supply_synthetic_readiness_values(template: str, required_name: str) -> None:
    value = smoke.PROFILES[template].environment[required_name]
    assert value.startswith("smoke-")


def test_default_catalog_create_has_no_source_override(tmp_path: Path) -> None:
    command = smoke.build_create_command(
        Path("/external/launcher/bin/agentseek"),
        template="langchain/markdown-messages",
        output_root=tmp_path,
        catalog_mode="default",
    )
    assert command == [
        "/external/launcher/bin/agentseek",
        "create",
        "langchain/markdown-messages",
        "--no-input",
        "--output-dir",
        str(tmp_path),
    ]


def test_default_catalog_port_context_uses_packaged_locked_template(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    cache_root = run_root / "cookiecutter-cache"
    prepared = cache_root / "catalog" / "templates" / "langchain" / "cli-remote"
    prepared.mkdir(parents=True)
    (prepared / "cookiecutter.json").write_text('{"langgraph_port": "2024"}\n', encoding="utf-8")
    recorded: dict[str, object] = {}

    def run_checked(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        label: str,
        run_root: Path,
        timeout: float,
    ) -> smoke.subprocess.CompletedProcess[str]:
        recorded.update(command=command, cwd=cwd, env=env, label=label, run_root=run_root, timeout=timeout)
        return smoke.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"template": str(prepared)}),
            stderr="",
        )

    monkeypatch.setattr(smoke, "_run_checked", run_checked)
    source_checkout = tmp_path / "missing-source-checkout"
    launcher_python = Path("/external/launcher/bin/python")
    launcher_env = {"HOME": str(run_root / "home")}

    result = smoke._port_template_root(
        "default",
        template="langchain/cli-remote",
        source_template_root=source_checkout,
        launcher_python=launcher_python,
        launcher_environment=launcher_env,
        run_root=run_root,
    )

    assert result == prepared.resolve()
    assert recorded == {
        "command": [
            str(launcher_python),
            "-c",
            smoke.DEFAULT_CATALOG_TEMPLATE_SCRIPT,
            "langchain/cli-remote",
            str(cache_root),
        ],
        "cwd": run_root,
        "env": launcher_env,
        "label": "prepare packaged default catalog template",
        "run_root": run_root,
        "timeout": smoke.CATALOG_PREPARE_TIMEOUT_SECONDS,
    }


def test_contract_probe_asserts_shell_value() -> None:
    assert smoke.assert_contract_probe({"output": {"sentinel": "from-shell"}}) == "from-shell"
    with pytest.raises(RuntimeError, match="from-shell"):
        smoke.assert_contract_probe({"output": {"sentinel": "from-dotenv"}})


def test_contract_probe_manifest_preserves_graphs_and_extracts_sentinel(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    package = generated / "src" / "probe_project"
    package.mkdir(parents=True)
    (generated / "pyproject.toml").write_text(
        '[project]\nname = "probe-project"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    original_graphs = {
        "agent": "./src/probe_project/agent.py:graph",
        "another": "./src/probe_project/another.py:graph",
    }
    (generated / "langgraph.json").write_text(
        json.dumps({"graphs": original_graphs}),
        encoding="utf-8",
    )

    smoke._install_contract_probe(generated, "agent")

    manifest = json.loads((generated / "langgraph.json").read_text(encoding="utf-8"))
    assert {name: manifest["graphs"][name] for name in original_graphs} == original_graphs
    module = "./src/probe_project/_release_contract_probe.py"
    assert manifest["graphs"]["release_contract_probe"] == {
        "graph": f"{module}:graph",
        "prepare_input": f"{module}:prepare_input",
        "extract_output": f"{module}:extract_output",
    }
    probe_source = (package / "_release_contract_probe.py").read_text(encoding="utf-8")
    assert "def prepare_input(" in probe_source
    assert "def extract_output(" in probe_source


def test_profiles_name_a_real_graph_and_minimal_input() -> None:
    assert smoke.PROFILES["deepagents/mcp"].graph_id == "mcp"
    assert smoke.PROFILES["langchain/cli-remote"].graph_id == "agent"
    assert smoke.PROFILES["langchain/rubric"].graph_id == "rubric-demo"
    assert smoke.PROFILES["langchain/rubric"].run_input == {"request": {}}


def test_frontend_install_uses_resolved_windows_executable(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    (generated / "frontend").mkdir(parents=True)
    (generated / "frontend" / "package.json").write_text("{}\n", encoding="utf-8")
    npm = Path(r"C:\hostedtoolcache\node\npm.CMD")
    assert smoke.build_frontend_install_command(generated, npm) == [
        str(npm),
        "install",
        "--prefix",
        "frontend",
    ]


def test_frontend_install_is_absent_for_cli_remote(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    assert smoke.build_frontend_install_command(generated, Path("/usr/bin/npm")) is None


def test_windows_runtime_uses_sqlite_without_embedded_seekdb(tmp_path: Path) -> None:
    environment = smoke.build_runtime_environment("win32", "AMD64", tmp_path, "sqlite")
    assert environment["SEEKDB_EMBED"] == "false"
    assert environment["METADATA_DB_BACKEND"] == "sqlite"
    assert environment["METADATA_DB_URL"].startswith("sqlite+aiosqlite:///")
    assert "SEEKDB_EMBED_DIR" not in environment


def test_linux_runtime_uses_external_embedded_directory() -> None:
    with tempfile.TemporaryDirectory(prefix="as-runtime-") as raw_run_root:
        run_root = Path(raw_run_root)
        environment = smoke.build_runtime_environment("linux", "x86_64", run_root, "embedded")
        assert environment["SEEKDB_EMBED"] == "true"
        embed_dir = Path(environment["SEEKDB_EMBED_DIR"])
        assert embed_dir == (run_root / "sdb").resolve()
        assert len(os.fsencode(embed_dir / "run" / "sql.sock")) <= 107
        assert "METADATA_DB_URL" not in environment


@pytest.mark.parametrize(
    ("platform_name", "machine_name", "socket_limit"),
    [("darwin", "arm64", 103), ("linux", "x86_64", 107)],
)
def test_embedded_runtime_rejects_overlong_socket_path_before_mkdir(
    tmp_path: Path,
    platform_name: str,
    machine_name: str,
    socket_limit: int,
) -> None:
    run_root = tmp_path / ("long-root-" + "x" * 120)
    socket_path = run_root / "sdb" / "run" / "sql.sock"
    assert len(os.fsencode(socket_path)) > socket_limit

    with pytest.raises(RuntimeError, match="^embedded seekdb socket path exceeds platform limit$"):
        smoke.build_runtime_environment(platform_name, machine_name, run_root, "embedded")

    assert not (run_root / "sdb").exists()


def test_auto_runtime_rejects_unsupported_intel_macos(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported runtime platform"):
        smoke.build_runtime_environment("darwin", "x86_64", tmp_path, "auto")


def test_candidate_wheel_rejects_relative_paths() -> None:
    with pytest.raises(RuntimeError, match="absolute"):
        smoke.validate_candidate_wheel(
            Path("dist/agentseek-0.1.3-py3-none-any.whl"),
            version="0.1.3",
            forbidden_roots=[smoke.ROOT],
        )


def test_profiles_cover_exact_retained_runtime_matrix() -> None:
    assert set(smoke.PROFILES) == {
        "deepagents/content-builder",
        "deepagents/mcp",
        "deepagents/research",
        "langchain/agentic-rag",
        "langchain/agentic-rag-hybrid",
        "langchain/cli-remote",
        "langchain/markdown-messages",
        "langchain/rubric",
    }


def test_runtime_workflow_covers_every_retained_migration() -> None:
    workflow = yaml.safe_load((smoke.ROOT / ".github" / "workflows" / "main.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["migrated-local-runtime-matrix"]
    expected_matrix = [
        {
            "template": "deepagents/content-builder",
            "os": "ubuntu-latest",
            "id": "content-builder-linux",
            "database": "embedded",
        },
        {
            "template": "deepagents/mcp",
            "os": "ubuntu-latest",
            "id": "mcp-linux",
            "database": "embedded",
        },
        {
            "template": "deepagents/mcp",
            "os": "windows-latest",
            "id": "mcp-windows",
            "database": "sqlite",
        },
        {
            "template": "deepagents/research",
            "os": "ubuntu-latest",
            "id": "research-linux",
            "database": "embedded",
        },
        {
            "template": "langchain/agentic-rag",
            "os": "ubuntu-latest",
            "id": "agentic-rag-linux",
            "database": "embedded",
        },
        {
            "template": "langchain/agentic-rag-hybrid",
            "os": "ubuntu-latest",
            "id": "agentic-rag-hybrid-linux",
            "database": "embedded",
        },
        {
            "template": "langchain/cli-remote",
            "os": "ubuntu-latest",
            "id": "cli-remote-linux",
            "database": "embedded",
        },
        {
            "template": "langchain/markdown-messages",
            "os": "ubuntu-latest",
            "id": "markdown-messages-linux",
            "database": "embedded",
        },
        {
            "template": "langchain/rubric",
            "os": "ubuntu-latest",
            "id": "rubric-linux",
            "database": "embedded",
        },
    ]
    expected_harness_command = (
        "uv run python scripts/generated_runtime_smoke.py"
        ' --template "${{ matrix.template }}"'
        " --catalog-mode source"
        " --agentseek-version 0.1.2"
        " --agentseek-api-version 0.2.2"
        ' --database-mode "${{ matrix.database }}"'
        ' --output-root "${{ env.GENERATED_RUNTIME_ROOT }}/${{ matrix.id }}"'
        ' --proof-output "${{ runner.temp }}/runtime-proof/${{ matrix.id }}.json"'
    )
    expected_upload = {
        "name": "Upload published-runtime proof",
        "uses": "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        "with": {
            "name": "runtime-proof-${{ matrix.id }}",
            "path": "${{ runner.temp }}/runtime-proof/${{ matrix.id }}.json",
            "if-no-files-found": "error",
        },
    }

    assert set(job) == {"name", "runs-on", "timeout-minutes", "strategy", "steps"}
    assert job["name"] == "${{ matrix.template }} (${{ matrix.os }})"
    assert job["runs-on"] == "${{ matrix.os }}"
    assert job["timeout-minutes"] == 30
    assert job["strategy"] == {
        "fail-fast": False,
        "matrix": {"include": expected_matrix},
    }
    assert job["steps"] == [
        {
            "uses": "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
            "with": {"persist-credentials": False},
        },
        {
            "uses": "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
            "with": {"python-version": "3.12"},
        },
        {
            "uses": "actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020",
            "with": {"node-version": "22"},
        },
        {
            "uses": "astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e",
            "with": {"version": "0.9.28", "enable-cache": True},
        },
        {"run": "uv sync --frozen --dev"},
        {
            "name": "Render, install, and exercise the generated lifecycle",
            "env": {"GENERATED_RUNTIME_ROOT": "${{ runner.temp }}/generated-runtime"},
            "run": expected_harness_command,
        },
        expected_upload,
    ]
    assert "if" not in job["steps"][-1]
    _assert_no_workflow_secrets(job)


def test_candidate_wheel_requires_matching_distribution_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "agentseek-0.1.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "agentseek-0.1.3.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: agentseek\nVersion: 0.1.2\n",
        )

    with pytest.raises(RuntimeError, match="metadata"):
        smoke.validate_candidate_wheel(wheel, version="0.1.3", forbidden_roots=[smoke.ROOT])


def test_candidate_wheel_stages_validated_bytes_before_source_mutation(tmp_path: Path) -> None:
    source = tmp_path / "external" / "agentseek-0.1.3-py3-none-any.whl"
    source.parent.mkdir()
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "agentseek-0.1.3.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: agentseek\nVersion: 0.1.3\n",
        )
        archive.writestr("agentseek/__init__.py", '__version__ = "0.1.3"\n')
    validated_bytes = source.read_bytes()
    expected_digest = hashlib.sha256(validated_bytes).hexdigest()
    artifact_dir = tmp_path / "run" / "artifacts"

    artifact = smoke.validate_candidate_wheel(
        source,
        version="0.1.3",
        forbidden_roots=[smoke.ROOT],
        destination=artifact_dir,
    )
    source.write_bytes(b"mutated after staging")

    assert artifact.path == artifact_dir / source.name
    assert artifact.path.read_bytes() == validated_bytes
    assert artifact.sha256 == expected_digest
    assert artifact.url == source.resolve().as_uri()
    assert not list(artifact_dir.glob(".*.tmp"))


def test_catalog_proof_records_observed_lock_without_render_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    template = "langchain/cli-remote"
    lock = {
        "catalog_repository": "https://github.com/agentseek-ai/agentseek-templates.git",
        "catalog_commit": "a" * 40,
        "catalog_release": "v0.1.0",
        "core_repository": "https://github.com/ob-labs/agentseek.git",
        "core_commit": "b" * 40,
        "core_release": "core-snapshot-v0.1.0",
        "templates": {template: "CLI Remote"},
        "template_digests": {template: "c" * 64},
    }
    completed = smoke.subprocess.CompletedProcess(
        [],
        0,
        stdout=json.dumps({"sha256": "d" * 64, "lock": lock}),
        stderr="",
    )
    monkeypatch.setattr(smoke, "_run_checked", lambda *_args, **_kwargs: completed)
    launcher_root = tmp_path / "run" / "launcher-venv"
    launcher_root.mkdir(parents=True)

    record = smoke._read_catalog_proof(
        launcher_root / "bin" / "python",
        {},
        launcher_root,
        template,
        "default",
    )

    assert record["sha256"] == "d" * 64
    assert record["template_digest"] == "c" * 64
    assert "used_for_render" not in record


def test_generated_lock_validates_uv_flat_index_wheel(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    wheel = artifact_dir / "agentseek_api-0.2.2-py3-none-any.whl"
    wheel.write_bytes(b"verified wheel bytes")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    artifact = smoke.WheelArtifact(
        name="agentseek-api",
        version="0.2.2",
        filename=wheel.name,
        path=wheel,
        sha256=digest,
        url="https://files.pythonhosted.org/agentseek_api-0.2.2-py3-none-any.whl",
    )
    generated = tmp_path / "rendered" / "project"
    generated.mkdir(parents=True)
    (generated / "uv.lock").write_text(
        """version = 1

[[package]]
name = "agentseek-api"
version = "0.2.2"
source = { registry = "../../artifacts" }
wheels = [
    { path = "agentseek_api-0.2.2-py3-none-any.whl" },
]
""",
        encoding="utf-8",
    )

    assert smoke._validate_api_lock(generated, artifact) == {
        "version": "0.2.2",
        "wheel_filename": wheel.name,
        "wheel_sha256": digest,
    }


def test_release_harness_requires_python_312() -> None:
    smoke.require_release_python(3, 12)
    with pytest.raises(RuntimeError, match="Python 3.12"):
        smoke.require_release_python(3, 13)
