from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import threading
import urllib.error
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runtime_proof.py"
SPEC = importlib.util.spec_from_file_location("runtime_proof", SCRIPT)
assert SPEC and SPEC.loader
proof = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = proof
SPEC.loader.exec_module(proof)


class _Response:
    def __init__(self, payload: bytes, final_url: str) -> None:
        self._payload = payload
        self._final_url = final_url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self) -> bytes:
        return self._payload

    def geturl(self) -> str:
        return self._final_url


def _launcher_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    repository = tmp_path / "repository"
    cache = tmp_path / "external" / "cache"
    home = tmp_path / "external" / "home"
    temporary = tmp_path / "external" / "tmp"
    tools = tmp_path / "external" / "tools"
    repository.mkdir()
    tools.mkdir(parents=True)
    return repository, cache, home, temporary, tools


def _metadata(name: str, version: str, wheel: bytes, *, url: str | None = None) -> dict[str, object]:
    filename = f"{name}-{version}-py3-none-any.whl"
    return {
        "urls": [
            {
                "digests": {"sha256": hashlib.sha256(wheel).hexdigest()},
                "filename": filename,
                "packagetype": "bdist_wheel",
                "url": url or f"https://files.pythonhosted.org/packages/{filename}",
            }
        ]
    }


def test_build_launcher_environment_sanitizes_hostile_ambient_state(tmp_path: Path) -> None:
    repository, cache, home, temporary, tools = _launcher_paths(tmp_path)
    certificate = tmp_path / "external" / "ca.pem"
    certificate.write_text("test certificate", encoding="utf-8")
    source = {
        "PATH": str(repository / "bin"),
        "PYTHONPATH": str(repository),
        "VIRTUAL_ENV": str(repository / ".venv"),
        "PIP_INDEX_URL": "https://user:secret@example.test/simple",
        "UV_INDEX_URL": "https://example.test/simple",
        "UV_CONFIG_FILE": str(repository / "uv.toml"),
        "PIP_CONFIG_FILE": str(repository / "pip.conf"),
        "OPENAI_API_KEY": "openai-secret",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "LANGSMITH_API_KEY": "langsmith-secret",
        "LANGSMITH_TRACING": "true",
        "LANGCHAIN_TRACING_V2": "true",
        "AGENTSEEK_API_KEY": "agentseek-secret",
        "HTTP_PROXY": "http://proxy.example.test:8080",
        "SYSTEMROOT": r"C:\Windows",
        "SSL_CERT_FILE": str(certificate),
        "UNRELATED_AMBIENT_VALUE": "must-not-pass",
    }

    environment = proof.build_launcher_environment(
        source,
        cache_dir=cache,
        home_dir=home,
        temp_dir=temporary,
        path_entries=[tools],
        forbidden_roots=[repository],
    )

    assert environment["PATH"] == str(tools.resolve())
    assert environment["UV_CACHE_DIR"] == str(cache.resolve())
    assert environment["HOME"] == str(home.resolve())
    assert environment["USERPROFILE"] == str(home.resolve())
    assert environment["TMPDIR"] == str(temporary.resolve())
    assert environment["TMP"] == str(temporary.resolve())
    assert environment["TEMP"] == str(temporary.resolve())
    assert environment["UV_NO_CONFIG"] == "1"
    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert environment["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert environment["LANGSMITH_TRACING"] == "false"
    assert environment["LANGCHAIN_TRACING_V2"] == "false"
    assert environment["NO_PROXY"] == "127.0.0.1,localhost"
    assert environment["no_proxy"] == "127.0.0.1,localhost"
    assert environment["HTTP_PROXY"] == source["HTTP_PROXY"]
    assert environment["SYSTEMROOT"] == source["SYSTEMROOT"]
    assert environment["SSL_CERT_FILE"] == str(certificate)
    assert (
        not {
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "PIP_INDEX_URL",
            "UV_INDEX_URL",
            "UV_CONFIG_FILE",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "LANGSMITH_API_KEY",
            "AGENTSEEK_API_KEY",
            "UNRELATED_AMBIENT_VALUE",
        }
        & environment.keys()
    )
    for directory in (cache, home, temporary):
        assert directory.is_dir()
        if os.name != "nt":
            assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_build_launcher_environment_rejects_control_path_inside_repository(tmp_path: Path) -> None:
    repository, cache, home, temporary, tools = _launcher_paths(tmp_path)
    certificate = repository / "ca.pem"
    certificate.write_text("test certificate", encoding="utf-8")

    with pytest.raises(RuntimeError, match="participating repository"):
        proof.build_launcher_environment(
            {"SSL_CERT_FILE": str(certificate)},
            cache_dir=cache,
            home_dir=home,
            temp_dir=temporary,
            path_entries=[tools],
            forbidden_roots=[repository],
        )


@pytest.mark.parametrize("directory_name", ["cache", "home", "temporary"])
def test_build_launcher_environment_rejects_managed_directory_inside_repository(
    tmp_path: Path, directory_name: str
) -> None:
    repository, cache, home, temporary, tools = _launcher_paths(tmp_path)
    paths = {"cache": cache, "home": home, "temporary": temporary}
    paths[directory_name] = repository / directory_name

    with pytest.raises(RuntimeError, match="participating repository"):
        proof.build_launcher_environment(
            {},
            cache_dir=paths["cache"],
            home_dir=paths["home"],
            temp_dir=paths["temporary"],
            path_entries=[tools],
            forbidden_roots=[repository],
        )


def test_build_launcher_environment_rejects_relative_or_duplicate_path_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, cache, home, temporary, tools = _launcher_paths(tmp_path)
    monkeypatch.chdir(tools.parent)

    with pytest.raises(RuntimeError, match="absolute"):
        proof.build_launcher_environment(
            {},
            cache_dir=cache,
            home_dir=home,
            temp_dir=temporary,
            path_entries=[Path("tools")],
            forbidden_roots=[repository],
        )

    with pytest.raises(RuntimeError, match="duplicate"):
        proof.build_launcher_environment(
            {},
            cache_dir=cache,
            home_dir=home,
            temp_dir=temporary,
            path_entries=[tools, tools],
            forbidden_roots=[repository],
        )


def test_build_launcher_environment_rejects_invalid_control_values(tmp_path: Path) -> None:
    repository, cache, home, temporary, tools = _launcher_paths(tmp_path)

    with pytest.raises(RuntimeError, match="HTTP_PROXY"):
        proof.build_launcher_environment(
            {"HTTP_PROXY": "http://proxy.example.test\nINJECTED=value"},
            cache_dir=cache,
            home_dir=home,
            temp_dir=temporary,
            path_entries=[tools],
            forbidden_roots=[repository],
        )


def test_build_launcher_environment_requires_windows_bootstrap_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, cache, home, temporary, tools = _launcher_paths(tmp_path)
    monkeypatch.setattr(proof, "_is_windows", lambda: True)

    with pytest.raises(RuntimeError, match="system root"):
        proof.build_launcher_environment(
            {},
            cache_dir=cache,
            home_dir=home,
            temp_dir=temporary,
            path_entries=[tools],
            forbidden_roots=[repository],
        )


def test_select_universal_wheel_requires_exactly_one_candidate() -> None:
    candidate = {
        "filename": "agentseek-0.1.2-py3-none-any.whl",
        "packagetype": "bdist_wheel",
        "url": "https://files.pythonhosted.org/agentseek.whl",
    }
    assert proof.select_universal_wheel({"urls": [candidate]}) is candidate

    with pytest.raises(RuntimeError, match="exactly one"):
        proof.select_universal_wheel({"urls": []})

    with pytest.raises(RuntimeError, match="exactly one"):
        proof.select_universal_wheel({"urls": [candidate, dict(candidate)]})


def test_select_universal_wheel_ignores_non_universal_artifacts() -> None:
    universal = {
        "filename": "agentseek-0.1.2-py3-none-any.whl",
        "packagetype": "bdist_wheel",
    }
    payload = {
        "urls": [
            {"filename": "agentseek-0.1.2.tar.gz", "packagetype": "sdist"},
            {"filename": "agentseek-0.1.2-cp312-cp312-macosx.whl", "packagetype": "bdist_wheel"},
            universal,
        ]
    }
    assert proof.select_universal_wheel(payload) is universal


def test_verify_sha256_returns_digest_and_rejects_mismatch() -> None:
    payload = b"published wheel bytes"
    digest = hashlib.sha256(payload).hexdigest()

    assert proof.verify_sha256(payload, digest.upper()) == digest
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        proof.verify_sha256(payload, "0" * 64)


def test_validate_import_path_requires_environment_ownership(tmp_path: Path) -> None:
    environment = tmp_path / "runtime"
    module_path = environment / "lib" / "python3.12" / "site-packages" / "agentseek" / "__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()

    assert proof.validate_import_path(module_path, environment) == module_path.resolve()
    with pytest.raises(RuntimeError, match="outside expected environment"):
        proof.validate_import_path(tmp_path / "checkout" / "agentseek" / "__init__.py", environment)


def test_resolve_executable_accepts_windows_command_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proof.shutil, "which", lambda name: r"C:\Tools\npm.CMD" if name == "npm" else None)

    assert str(proof.resolve_executable("npm")) == r"C:\Tools\npm.CMD"


def test_resolve_executable_rejects_missing_or_relative_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proof.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="not found"):
        proof.resolve_executable("missing")

    monkeypatch.setattr(proof.shutil, "which", lambda _name: "bin/tool")
    with pytest.raises(RuntimeError, match="absolute"):
        proof.resolve_executable("tool")


def test_download_published_wheel_records_verified_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = b"published universal wheel"
    metadata_url = "https://pypi.org/pypi/agentseek/0.1.2/json"
    artifact_url = "https://files.pythonhosted.org/packages/agentseek-0.1.2-py3-none-any.whl"
    metadata = _metadata("agentseek", "0.1.2", wheel, url=artifact_url)
    requested_urls: list[str] = []

    def open_url(request: Any, *, timeout: float) -> _Response:
        assert timeout > 0
        requested_urls.append(request.full_url)
        if request.full_url == metadata_url:
            return _Response(json.dumps(metadata).encode(), metadata_url)
        assert request.full_url == artifact_url
        return _Response(wheel, artifact_url)

    monkeypatch.setattr(proof.urllib.request, "urlopen", open_url)
    destination = tmp_path / "artifacts"

    artifact = proof.download_published_wheel("agentseek", "0.1.2", destination)

    expected_path = destination / "agentseek-0.1.2-py3-none-any.whl"
    assert artifact == proof.WheelArtifact(
        name="agentseek",
        version="0.1.2",
        filename=expected_path.name,
        path=expected_path,
        sha256=hashlib.sha256(wheel).hexdigest(),
        url=artifact_url,
    )
    assert requested_urls == [metadata_url, artifact_url]
    assert expected_path.read_bytes() == wheel
    with pytest.raises(FrozenInstanceError):
        artifact.sha256 = "changed"  # type: ignore[misc]


def test_download_published_wheel_does_not_write_unverified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = b"tampered wheel"
    metadata = _metadata("agentseek", "0.1.2", b"expected wheel")
    responses = iter(
        [
            _Response(json.dumps(metadata).encode(), "https://pypi.org/pypi/agentseek/0.1.2/json"),
            _Response(wheel, str(metadata["urls"][0]["url"])),  # type: ignore[index]
        ]
    )
    monkeypatch.setattr(proof.urllib.request, "urlopen", lambda *_args, **_kwargs: next(responses))
    destination = tmp_path / "artifacts"

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        proof.download_published_wheel("agentseek", "0.1.2", destination)

    assert not (destination / "agentseek-0.1.2-py3-none-any.whl").exists()


def test_download_published_wheel_rejects_insecure_redirect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = b"published universal wheel"
    metadata = _metadata("agentseek", "0.1.2", wheel)
    responses = iter(
        [
            _Response(json.dumps(metadata).encode(), "https://pypi.org/pypi/agentseek/0.1.2/json"),
            _Response(wheel, "http://mirror.example.test/agentseek.whl"),
        ]
    )
    monkeypatch.setattr(proof.urllib.request, "urlopen", lambda *_args, **_kwargs: next(responses))
    destination = tmp_path / "artifacts"

    with pytest.raises(RuntimeError, match="HTTPS"):
        proof.download_published_wheel("agentseek", "0.1.2", destination)

    assert not destination.exists()


def test_download_published_wheel_bounds_transport_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def unavailable(*_args: object, **_kwargs: object) -> _Response:
        nonlocal attempts
        attempts += 1
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(proof.urllib.request, "urlopen", unavailable)
    monkeypatch.setattr(proof.time, "sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="download failed"):
        proof.download_published_wheel("agentseek", "0.1.2", tmp_path / "artifacts")

    assert attempts == proof.DOWNLOAD_ATTEMPTS


def test_download_published_wheel_does_not_publish_a_partial_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = b"published universal wheel"
    metadata = _metadata("agentseek", "0.1.2", wheel)

    def download(url: str) -> bytes:
        if url == "https://pypi.org/pypi/agentseek/0.1.2/json":
            return json.dumps(metadata).encode()
        return wheel

    real_write = os.write
    write_calls = 0

    def fail_after_partial_write(file_descriptor: int, payload: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return real_write(file_descriptor, payload[:4])
        raise OSError("synthetic write failure")

    monkeypatch.setattr(proof, "_download_bytes", download)
    monkeypatch.setattr(proof.os, "write", fail_after_partial_write)
    destination = tmp_path / "artifacts"

    with pytest.raises(OSError, match="synthetic write failure"):
        proof.download_published_wheel("agentseek", "0.1.2", destination)

    assert not (destination / "agentseek-0.1.2-py3-none-any.whl").exists()
    assert list(destination.glob(".agentseek-0.1.2-py3-none-any.whl.*.tmp")) == []


def test_collect_import_record_proves_real_environment_import() -> None:
    import pytest as pytest_package

    environment_root = Path(sys.prefix)
    record = proof.collect_import_record(Path(sys.executable), "pytest", "pytest", environment_root)

    assert record.distribution == "pytest"
    assert record.version == pytest_package.__version__
    assert Path(record.module_path).resolve() == Path(pytest_package.__file__).resolve()
    assert Path(record.python).resolve() == Path(sys.executable).resolve()
    with pytest.raises(FrozenInstanceError):
        record.version = "changed"  # type: ignore[misc]


def test_collect_import_record_uses_minimal_environment_and_safe_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "runtime"
    python = environment / "bin" / "python"
    module_path = environment / "site-packages" / "agentseek" / "__init__.py"
    python.parent.mkdir(parents=True)
    python.touch()
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "checkout"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-provider-secret")
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:secret@example.test/simple")
    monkeypatch.setattr(proof, "_is_windows", lambda: False)
    captured: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured.update(kwargs)
        payload = {
            "distribution": "agentseek",
            "module_path": str(module_path),
            "python": str(python),
            "version": "0.1.2",
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(proof.subprocess, "run", run)

    record = proof.collect_import_record(python, "agentseek", "agentseek", environment)

    assert record.version == "0.1.2"
    assert captured["command"] == [str(python), "-c", proof.IMPORT_SCRIPT, "agentseek", "agentseek"]
    assert captured["cwd"] == environment.resolve()
    assert captured["timeout"] == 30.0
    assert captured["env"] == {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONUTF8": "1",
    }


def test_collect_import_record_passes_only_windows_system_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment = tmp_path / "runtime"
    python = environment / "Scripts" / "python.exe"
    module_path = environment / "Lib" / "site-packages" / "agentseek" / "__init__.py"
    python.parent.mkdir(parents=True)
    python.touch()
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    monkeypatch.setattr(proof, "_is_windows", lambda: True)
    for name in ("SYSTEMROOT", "SystemRoot", "WINDIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-provider-secret")
    captured: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        payload = {
            "distribution": "agentseek",
            "module_path": str(module_path),
            "python": str(python),
            "version": "0.1.2",
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(proof.subprocess, "run", run)

    proof.collect_import_record(python, "agentseek", "agentseek", environment)

    assert captured["env"] == {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONUTF8": "1",
        "SystemRoot": r"C:\Windows",
    }


def test_collect_import_record_rejects_import_outside_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "runtime"
    environment.mkdir()
    python = environment / "bin" / "python"
    outside = tmp_path / "checkout" / "agentseek" / "__init__.py"
    payload = {
        "distribution": "agentseek",
        "module_path": str(outside),
        "python": str(python),
        "version": "0.1.2",
    }
    completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(proof.subprocess, "run", lambda *_args, **_kwargs: completed)

    with pytest.raises(RuntimeError) as exc_info:
        proof.collect_import_record(python, "agentseek", "agentseek", environment)

    assert str(exc_info.value) == "import probe module is outside expected environment"
    assert str(outside) not in str(exc_info.value)


def test_collect_import_record_requires_the_requested_environment_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "runtime"
    requested_python = environment / "bin" / "python"
    different_python = environment / "bin" / "python-other"
    module_path = environment / "site-packages" / "agentseek" / "__init__.py"
    requested_python.parent.mkdir(parents=True)
    requested_python.touch()
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    payload = {
        "distribution": "agentseek",
        "module_path": str(module_path),
        "python": str(different_python),
        "version": "0.1.2",
    }
    completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(proof.subprocess, "run", lambda *_args, **_kwargs: completed)

    with pytest.raises(RuntimeError) as exc_info:
        proof.collect_import_record(requested_python, "agentseek", "agentseek", environment)

    assert str(exc_info.value) == "import probe returned an unexpected interpreter"
    assert str(different_python) not in str(exc_info.value)


def test_collect_import_record_reports_subprocess_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment = tmp_path / "runtime"
    environment.mkdir()
    python = environment / "bin" / "python"
    completed = subprocess.CompletedProcess([], 1, stdout="", stderr="synthetic-provider-secret")
    monkeypatch.setattr(proof.subprocess, "run", lambda *_args, **_kwargs: completed)

    with pytest.raises(RuntimeError) as exc_info:
        proof.collect_import_record(python, "missing", "missing", environment)

    assert str(exc_info.value) == "import probe failed"
    assert "synthetic-provider-secret" not in str(exc_info.value)


def test_collect_import_record_reports_value_free_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment = tmp_path / "runtime"
    environment.mkdir()
    python = environment / "bin" / "python"

    def time_out(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            cmd=["synthetic-provider-secret"],
            timeout=30.0,
            output="synthetic-provider-secret",
            stderr="synthetic-provider-secret",
        )

    monkeypatch.setattr(proof.subprocess, "run", time_out)

    with pytest.raises(RuntimeError) as exc_info:
        proof.collect_import_record(python, "agentseek", "agentseek", environment)

    assert str(exc_info.value) == "import probe timed out"
    assert "synthetic-provider-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_write_proof_rejects_output_inside_repository_before_creating_parent(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output = repository / "evidence" / "runtime-proof.json"

    with pytest.raises(RuntimeError, match="outside participating repositories"):
        proof.write_proof(output, {"status": "passed"}, [repository])

    assert not output.parent.exists()


def test_write_proof_serializes_stable_json_with_one_trailing_newline(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "evidence" / "runtime-proof.json"

    proof.write_proof(output, {"z": 2, "a": 1}, [repository])

    assert output.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "z": 2\n}\n'


def test_write_proof_preserves_previous_file_and_cleans_temporary_on_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "evidence" / "runtime-proof.json"
    output.parent.mkdir()
    previous = '{\n  "status": "previous"\n}\n'
    output.write_text(previous, encoding="utf-8")

    def fail_publish(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic publish failure")

    monkeypatch.setattr(proof.os, "replace", fail_publish)

    with pytest.raises(OSError, match="synthetic publish failure"):
        proof.write_proof(output, {"status": "replacement"}, [repository])

    assert output.read_text(encoding="utf-8") == previous
    assert list(output.parent.glob(".runtime-proof.json.*.tmp")) == []


def test_write_proof_keeps_previous_bytes_visible_until_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "evidence" / "runtime-proof.json"
    output.parent.mkdir()
    previous = '{\n  "status": "previous"\n}\n'
    replacement = '{\n  "status": "replacement"\n}\n'
    output.write_text(previous, encoding="utf-8")
    before_publish = threading.Event()
    allow_publish = threading.Event()
    writer_errors: list[BaseException] = []
    real_replace = os.replace

    def paused_replace(source: Path, destination: Path) -> None:
        assert source.parent == output.parent
        assert destination == output
        before_publish.set()
        if not allow_publish.wait(timeout=5):
            raise RuntimeError("test did not release atomic publish")
        real_replace(source, destination)

    def write_replacement() -> None:
        try:
            proof.write_proof(output, {"status": "replacement"}, [repository])
        except BaseException as exc:  # pragma: no cover - asserted by the parent thread
            writer_errors.append(exc)

    monkeypatch.setattr(proof.os, "replace", paused_replace)
    writer = threading.Thread(target=write_replacement)
    writer.start()
    try:
        assert before_publish.wait(timeout=5)
        assert output.read_text(encoding="utf-8") == previous
    finally:
        allow_publish.set()
        writer.join(timeout=5)

    assert not writer.is_alive()
    assert writer_errors == []
    assert output.read_text(encoding="utf-8") == replacement
    assert list(output.parent.glob(".runtime-proof.json.*.tmp")) == []


def test_write_proof_resolves_external_output_without_creating_repository_intermediates(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    intermediate = repository / "must-not-be-created"
    resolved_output = tmp_path / "evidence" / "runtime-proof.json"
    unresolved_output = intermediate / ".." / ".." / "evidence" / "runtime-proof.json"

    proof.write_proof(unresolved_output, {"status": "passed"}, [repository])

    assert resolved_output.read_text(encoding="utf-8") == '{\n  "status": "passed"\n}\n'
    assert list(repository.iterdir()) == []
