from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath


@dataclass(frozen=True)
class WheelArtifact:
    name: str
    version: str
    filename: str
    path: Path
    sha256: str
    url: str


@dataclass(frozen=True)
class ImportRecord:
    distribution: str
    version: str
    module_path: str
    python: str


CONTROL_PASSTHROUGH_NAMES = (
    "SYSTEMROOT",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
CONTROL_PATH_NAMES = frozenset(
    {
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    }
)
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 30.0
DOWNLOAD_RETRY_DELAY_SECONDS = 0.25
IMPORT_PROBE_TIMEOUT_SECONDS = 30.0
IMPORT_PROBE_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONUTF8": "1",
}
WINDOWS_IMPORT_BOOTSTRAP_NAMES = ("SYSTEMROOT", "SystemRoot", "WINDIR")


def _is_windows() -> bool:
    return os.name == "nt"


def build_launcher_environment(
    source: Mapping[str, str],
    *,
    cache_dir: Path,
    home_dir: Path,
    temp_dir: Path,
    path_entries: Sequence[Path],
    forbidden_roots: Sequence[Path],
) -> dict[str, str]:
    roots = tuple(path.resolve() for path in forbidden_roots)

    def outside_repositories(path: Path) -> Path:
        resolved = path.resolve()
        if any(resolved == root or resolved.is_relative_to(root) for root in roots):
            raise RuntimeError(f"control path is inside a participating repository: {resolved}")
        return resolved

    directories = tuple(outside_repositories(path) for path in (cache_dir, home_dir, temp_dir))
    for directory in directories:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not _is_windows():
            directory.chmod(0o700)

    resolved_entries: list[Path] = []
    for path in path_entries:
        if not path.is_absolute():
            raise RuntimeError(f"launcher PATH entry is not absolute: {path}")
        resolved = outside_repositories(path)
        if not resolved.is_dir() or resolved in resolved_entries:
            raise RuntimeError(f"launcher PATH entry is missing or duplicated: {resolved}")
        resolved_entries.append(resolved)

    result: dict[str, str] = {}
    for name in CONTROL_PASSTHROUGH_NAMES:
        value = source.get(name)
        if not value:
            continue
        if "\x00" in value or "\n" in value or "\r" in value:
            raise RuntimeError(f"launcher control variable {name} contains an unsupported value")
        if name in CONTROL_PATH_NAMES:
            path = Path(value)
            if not path.is_absolute():
                raise RuntimeError(f"launcher control variable {name} is not an absolute path")
            outside_repositories(path)
        result[name] = value

    if _is_windows() and not any(result.get(name) for name in ("SYSTEMROOT", "SystemRoot", "WINDIR")):
        raise RuntimeError("Windows launcher environment has no system root")

    result.update(
        {
            "PATH": os.pathsep.join(map(str, resolved_entries)),
            "HOME": str(directories[1]),
            "USERPROFILE": str(directories[1]),
            "TMPDIR": str(directories[2]),
            "TMP": str(directories[2]),
            "TEMP": str(directories[2]),
            "UV_CACHE_DIR": str(directories[0]),
            "UV_NO_CONFIG": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return result


def select_universal_wheel(payload: Mapping[str, object]) -> Mapping[str, object]:
    urls = payload.get("urls")
    if not isinstance(urls, list):
        raise RuntimeError("PyPI response has no artifact list")
    matches = [
        item
        for item in urls
        if isinstance(item, dict)
        and item.get("packagetype") == "bdist_wheel"
        and str(item.get("filename", "")).endswith("-py3-none-any.whl")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one universal wheel, found {len(matches)}")
    return matches[0]


def validate_import_path(module_path: Path, environment_root: Path) -> Path:
    resolved = module_path.resolve()
    try:
        resolved.relative_to(environment_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"import path {resolved} is outside expected environment {environment_root}") from exc
    return resolved


def resolve_executable(name: str) -> Path:
    raw = shutil.which(name)
    if raw is None:
        raise RuntimeError(f"required executable was not found: {name}")
    candidate = Path(raw)
    if not candidate.is_absolute() and not PureWindowsPath(raw).is_absolute():
        raise RuntimeError(f"resolved executable path is not absolute: {raw}")
    return candidate


def verify_sha256(payload: bytes, expected: str) -> str:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected.lower():
        raise RuntimeError(f"artifact SHA-256 mismatch: expected {expected.lower()}, got {actual}")
    return actual


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    file_descriptor = -1
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(file_descriptor, remaining)
            if written <= 0:
                raise OSError("atomic write made no progress")
            remaining = remaining[written:]
        os.fsync(file_descriptor)
        os.close(file_descriptor)
        file_descriptor = -1
        os.replace(temporary_path, path)
    finally:
        if file_descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(file_descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _download_bytes(url: str) -> bytes:
    if urllib.parse.urlparse(url).scheme.lower() != "https":
        raise RuntimeError(f"artifact download URL must use HTTPS: {url}")

    request = urllib.request.Request(url, headers={"User-Agent": "agentseek-runtime-proof/1"})
    last_error: BaseException | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                final_url = response.geturl()
                if urllib.parse.urlparse(final_url).scheme.lower() != "https":
                    raise RuntimeError(f"artifact redirect target must use HTTPS: {final_url}")
                return response.read()
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < DOWNLOAD_ATTEMPTS:
                time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS * (attempt + 1))

    raise RuntimeError(f"download failed after {DOWNLOAD_ATTEMPTS} attempts: {url}") from last_error


def _safe_wheel_filename(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError("PyPI universal wheel has no filename")
    if Path(value).name != value or PureWindowsPath(value).name != value:
        raise RuntimeError(f"PyPI universal wheel has an unsafe filename: {value}")
    return value


def download_published_wheel(name: str, version: str, destination: Path) -> WheelArtifact:
    metadata_url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        metadata = json.loads(_download_bytes(metadata_url))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"PyPI returned invalid metadata for {name}=={version}") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"PyPI returned invalid metadata for {name}=={version}")

    selected = select_universal_wheel(metadata)
    filename = _safe_wheel_filename(selected.get("filename"))
    url = selected.get("url")
    digests = selected.get("digests")
    if not isinstance(url, str) or not url:
        raise RuntimeError("PyPI universal wheel has no download URL")
    if not isinstance(digests, dict) or not isinstance(digests.get("sha256"), str):
        raise RuntimeError("PyPI universal wheel has no SHA-256 digest")

    payload = _download_bytes(url)
    digest = verify_sha256(payload, digests["sha256"])
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    artifact_path = destination / filename
    _atomic_write_bytes(artifact_path, payload)
    return WheelArtifact(
        name=name,
        version=version,
        filename=filename,
        path=artifact_path,
        sha256=digest,
        url=url,
    )


IMPORT_SCRIPT = """
import importlib
import importlib.metadata
import json
import sys

distribution, module_name = sys.argv[1:]
module = importlib.import_module(module_name)
print(json.dumps({
    "distribution": distribution,
    "version": importlib.metadata.version(distribution),
    "module_path": str(module.__file__),
    "python": sys.executable,
}))
"""


def _build_import_probe_environment(source: Mapping[str, str]) -> dict[str, str]:
    environment = dict(IMPORT_PROBE_ENVIRONMENT)
    if not _is_windows():
        return environment

    for name in WINDOWS_IMPORT_BOOTSTRAP_NAMES:
        value = source.get(name)
        if not value:
            continue
        if "\x00" in value or "\n" in value or "\r" in value:
            raise RuntimeError("import probe Windows system root is invalid")
        environment[name] = value
        return environment
    raise RuntimeError("import probe has no Windows system root")


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def collect_import_record(
    python: Path,
    distribution: str,
    module: str,
    environment_root: Path,
) -> ImportRecord:
    resolved_environment = environment_root.resolve()
    requested_python = _absolute_without_resolving_symlinks(python)
    if not resolved_environment.is_dir() or not _is_within(requested_python, resolved_environment):
        raise RuntimeError("import probe interpreter is outside expected environment")

    try:
        completed = subprocess.run(
            [str(requested_python), "-c", IMPORT_SCRIPT, distribution, module],
            check=False,
            capture_output=True,
            cwd=resolved_environment,
            encoding="utf-8",
            env=_build_import_probe_environment(os.environ),
            errors="replace",
            text=True,
            timeout=IMPORT_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("import probe timed out") from None
    except OSError:
        raise RuntimeError("import probe execution failed") from None
    if completed.returncode != 0:
        raise RuntimeError("import probe failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("import probe returned invalid JSON") from None
    if not isinstance(payload, dict):
        raise RuntimeError("import probe returned invalid record")

    returned_distribution = payload.get("distribution")
    version = payload.get("version")
    module_path = payload.get("module_path")
    returned_python = payload.get("python")
    if returned_distribution != distribution:
        raise RuntimeError("import probe returned the wrong distribution")
    if not all(isinstance(value, str) and value for value in (version, module_path, returned_python)):
        raise RuntimeError("import probe returned an incomplete record")

    returned_python_path = _absolute_without_resolving_symlinks(Path(returned_python))
    if not _is_within(returned_python_path, resolved_environment) or os.path.normcase(
        str(returned_python_path)
    ) != os.path.normcase(str(requested_python)):
        raise RuntimeError("import probe returned an unexpected interpreter")
    try:
        resolved_module = validate_import_path(Path(module_path), resolved_environment)
    except (OSError, RuntimeError):
        raise RuntimeError("import probe module is outside expected environment") from None
    return ImportRecord(
        distribution=distribution,
        version=version,
        module_path=str(resolved_module),
        python=str(returned_python_path),
    )


def write_proof(path: Path, payload: Mapping[str, object], forbidden_roots: Sequence[Path]) -> None:
    resolved_path = path.resolve()
    roots = tuple(root.resolve() for root in forbidden_roots)
    if any(resolved_path == root or resolved_path.is_relative_to(root) for root in roots):
        raise RuntimeError(f"proof output must be outside participating repositories: {resolved_path}")

    resolved_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    serialized = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(resolved_path, serialized)
