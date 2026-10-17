"""Metadata evidence for local inference inputs; weight bytes are not hashed."""

import hashlib
import sys
from pathlib import Path

from . import __version__
from .task_contracts import TaskError, encoded


def _file_identity(path):
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def _model_fingerprint(model):
    entries = []
    if not model.is_dir():
        raise TaskError("model_changed", "The accepted model directory is unavailable.")
    for path in sorted(model.rglob("*")):
        if path.is_symlink() and not path.exists():
            raise TaskError("model_changed", "The model contains a broken link.")
        if path.is_file():
            identity = _file_identity(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.suffix == ".json" and identity[2] <= 1024 * 1024 else ""
            entries.append([str(path.relative_to(model)), *identity, digest])
    return hashlib.sha256(encoded(entries)).hexdigest()


def capture_identity(model, manifest, executable):
    try:
        requested_model = str(Path(model).absolute())
        requested_executable = str(Path(executable).absolute())
        model = Path(model).resolve(strict=True)
        executable = Path(executable).resolve(strict=True)
        return {"schema": "task-identity/1", "requested_model_path": requested_model,
                "model_path": str(model), "model_fingerprint": _model_fingerprint(model),
                "backend": manifest["id"], "backend_version": manifest["version"],
                "backend_manifest": manifest, "backend_manifest_hash": hashlib.sha256(encoded(manifest)).hexdigest(),
                "requested_executable": requested_executable, "executable": str(executable), "executable_identity": _file_identity(executable),
                "engine_version": __version__, "python_version": sys.version.split()[0]}
    except (OSError, ValueError, KeyError) as error:
        raise TaskError("identity_unavailable", "Cannot inspect local model/backend identity: {0}".format(error)) from error


def check_identity(expected):
    from .backends import BackendError, load_manifests
    try:
        if expected["schema"] != "task-identity/1":
            raise TaskError("identity_changed", "Unsupported task identity algorithm.")
        manifest = load_manifests()[expected["backend"]]
        current = capture_identity(Path(expected["requested_model_path"]), manifest, Path(expected["requested_executable"]))
        for key in ("model_path", "model_fingerprint", "executable", "executable_identity", "backend_manifest_hash"):
            if current[key] != expected[key]:
                raise TaskError("identity_changed", "Accepted {0} changed.".format(key), "Start a new task with the current local inputs.")
    except (KeyError, TypeError) as error:
        raise TaskError("identity_changed", "Task identity evidence is invalid.") from error
    except (OSError, ValueError, BackendError) as error:
        raise TaskError("identity_unavailable", "Cannot read the current backend identity.",
                        "Repair the backend manifest and retry the identity check.") from error
