from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import zipfile
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
        self.kill_calls = 0
        self.wait_calls = 0
        self.wait_timeouts: list[float] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, *, timeout: float) -> int:
        self.wait_calls += 1
        self.wait_timeouts.append(timeout)
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


class _StubbornProcess(_Process):
    def wait(self, *, timeout: float) -> int:
        self.wait_calls += 1
        self.wait_timeouts.append(timeout)
        if self.wait_calls == 1:
            raise smoke.subprocess.TimeoutExpired("agentseek dev", timeout)
        self.returncode = -9
        return -9


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


def test_terminate_allows_launcher_to_clean_nested_process_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _Process()
    monkeypatch.setattr(smoke.os, "name", "posix")
    monkeypatch.setattr(smoke.os, "killpg", lambda *_args: None)

    smoke._terminate(process)

    assert process.wait_timeouts[0] >= 30


def test_terminate_force_kills_entire_posix_group_after_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _StubbornProcess()
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(smoke.os, "name", "posix")
    monkeypatch.setattr(smoke.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    smoke._terminate(process)

    assert signals == [
        (process.pid, smoke.signal.SIGTERM),
        (process.pid, smoke.signal.SIGKILL),
    ]
    assert process.kill_calls == 0


def test_process_cleanup_attempts_provider_after_lifecycle_error(monkeypatch: pytest.MonkeyPatch) -> None:
    lifecycle = _Process()
    provider = _Process()
    attempted: list[_Process] = []

    def terminate(process: _Process) -> None:
        attempted.append(process)
        if process is lifecycle:
            raise RuntimeError("lifecycle cleanup failed")

    monkeypatch.setattr(smoke, "_terminate", terminate)

    with pytest.raises(RuntimeError, match="lifecycle cleanup failed"):
        smoke._terminate_processes(lifecycle, provider)

    assert attempted == [lifecycle, provider]


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
    ) -> smoke.subprocess.CompletedProcess[str]:
        recorded.update(command=command, cwd=cwd, env=env, label=label)
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


def test_candidate_wheel_requires_matching_distribution_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "agentseek-0.1.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "agentseek-0.1.3.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: agentseek\nVersion: 0.1.2\n",
        )

    with pytest.raises(RuntimeError, match="metadata"):
        smoke.validate_candidate_wheel(wheel, version="0.1.3", forbidden_roots=[smoke.ROOT])


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
