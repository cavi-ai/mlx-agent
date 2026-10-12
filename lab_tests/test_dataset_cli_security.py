from __future__ import annotations

import json
import os
import signal
import socket
import stat
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from video_lab.cli import app
from video_lab.dataset_cli import (
    DatasetCliError,
    DatasetExit,
    _cancellation,
    _open_request,
)

runner = CliRunner()


def _valid_request() -> dict[str, object]:
    binding = {
        "artifact_id": "artifact_1",
        "sha256": "b" * 64,
        "byte_size": 2,
        "media_type": "application/json",
    }
    return {
        "schema_version": "dataset-compilation-request.v1",
        "operation_key": "trial_one",
        "compiler_id": "wan_curated_v1",
        "source": {**binding, "role": "source_manifest"},
        "recipe": {**binding, "artifact_id": "artifact_2", "role": "transform_recipe"},
        "output_roles": ["batch"],
        "reproduces_receipt_id": None,
    }


def test_dataset_request_rejects_symlink_without_disclosing_path(tmp_path: Path) -> None:
    secret = tmp_path / "secret-name.json"
    secret.write_text("{}")
    (tmp_path / "request.json").symlink_to(secret)
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "dataset",
            "validate",
            "--request-root",
            str(tmp_path),
            "--request-file",
            "request.json",
            "--json",
        ],
    )
    assert result.exit_code == 70
    assert "secret-name" not in result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "integrity_failure"


def test_dataset_request_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    (tmp_path / "request.json").write_text('{"schema_version":"x","schema_version":"x"}')
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "dataset",
            "validate",
            "--request-root",
            str(tmp_path),
            "--request-file",
            "request.json",
            "--json",
        ],
    )
    assert result.exit_code == 65
    assert json.loads(result.stdout)["error"]["code"] == "rejected"


def test_validate_does_not_import_ml_modules(tmp_path: Path, monkeypatch: object) -> None:
    imported: list[str] = []
    original = __import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] in {"torch", "diffusers", "transformers", "peft"}:
            imported.append(name)
            raise AssertionError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    (tmp_path / "request.json").write_text("{}")
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "dataset",
            "validate",
            "--request-root",
            str(tmp_path),
            "--request-file",
            "request.json",
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert imported == []


def test_dataset_request_rejects_fifo_nonblocking(tmp_path: Path) -> None:
    fifo = tmp_path / "request.json"
    os.mkfifo(fifo)
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "dataset",
            "validate",
            "--request-root",
            str(tmp_path),
            "--request-file",
            "request.json",
            "--json",
        ],
    )
    assert result.exit_code == 70


@pytest.mark.parametrize(
    ("code", "exit_code"),
    [
        ("rejected", DatasetExit.REJECTED),
        ("needs_review", DatasetExit.NEEDS_REVIEW),
        ("conflict", DatasetExit.CONFLICT),
        ("in_progress", DatasetExit.IN_PROGRESS),
        ("cancelled", DatasetExit.CANCELLED),
        ("integrity_failure", DatasetExit.INTEGRITY),
        ("usage_error", DatasetExit.USAGE),
        ("internal_failure", DatasetExit.INTERNAL),
    ],
)
def test_dataset_exit_code_matrix_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str, exit_code: DatasetExit
) -> None:
    (tmp_path / "request.json").write_text(
        json.dumps(_valid_request(), sort_keys=True, separators=(",", ":"))
    )

    def fail(**_: object) -> object:
        raise DatasetCliError(code, "safe diagnostic", exit_code)

    monkeypatch.setattr("video_lab.dataset_cli.build_dataset_application", fail)
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "dataset",
            "compile",
            "--request-root",
            str(tmp_path),
            "--request-file",
            "request.json",
            "--json",
        ],
    )
    assert result.exit_code == int(exit_code)
    assert json.loads(result.stdout)["error"] == {"code": code, "message": "safe diagnostic"}
    assert result.stderr == ""


def test_agent_subprocess_rejection_is_byte_stable_and_offline(tmp_path: Path) -> None:
    (tmp_path / "request.json").write_text('{"schema_version":"wrong"}')
    command = [
        str(Path.cwd() / ".venv/bin/video-lab"),
        "--database",
        str(tmp_path / "db"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "dataset",
        "validate",
        "--request-root",
        str(tmp_path),
        "--request-file",
        "request.json",
        "--json",
    ]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    first = subprocess.run(command, capture_output=True, check=False, env=environment)
    second = subprocess.run(command, capture_output=True, check=False, env=environment)
    assert (first.returncode, first.stdout, first.stderr) == (
        second.returncode,
        second.stdout,
        second.stderr,
    )
    assert first.returncode == DatasetExit.REJECTED


def test_partial_component_roots_are_usage_error(tmp_path: Path) -> None:
    (tmp_path / "request.json").write_text(
        json.dumps(_valid_request(), sort_keys=True, separators=(",", ":"))
    )
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "dataset",
            "compile",
            "--request-root",
            str(tmp_path),
            "--request-file",
            "request.json",
            "--tokenizer-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == DatasetExit.USAGE
    assert json.loads(result.stdout)["error"]["code"] == "usage_error"


def test_missing_component_roots_fail_before_wan_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "request.json").write_text(
        json.dumps(_valid_request(), sort_keys=True, separators=(",", ":"))
    )
    imported: list[str] = []
    original = __import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name == "video_lab.compilers.wan_curated":
            imported.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    missing = tmp_path / "missing-local-component"
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "dataset",
            "compile",
            "--request-root",
            str(tmp_path),
            "--request-file",
            "request.json",
            "--tokenizer-root",
            str(missing),
            "--text-encoder-root",
            str(missing),
            "--vae-root",
            str(missing),
            "--scheduler-root",
            str(missing),
            "--json",
        ],
    )
    assert result.exit_code == DatasetExit.REJECTED
    assert imported == []


def test_signal_requests_bounded_cancellation() -> None:
    with _cancellation() as cancelled:
        assert cancelled() is False
        signal.raise_signal(signal.SIGTERM)
        assert cancelled() is True


def test_compile_signal_returns_stable_cancelled_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "request.json").write_text(
        json.dumps(_valid_request(), sort_keys=True, separators=(",", ":"))
    )

    class Compilation:
        def __init__(self, cancelled: object) -> None:
            self.cancelled = cancelled

        def compile(self, request: object) -> object:
            signal.raise_signal(signal.SIGTERM)
            assert self.cancelled()
            from video_lab.dataset_compiler import DatasetCompilationError

            raise DatasetCompilationError("compilation cancelled at batch boundary")

    class Application:
        def __init__(self, cancelled: object) -> None:
            self.compilation = Compilation(cancelled)

    monkeypatch.setattr(
        "video_lab.dataset_cli.build_dataset_application",
        lambda **values: Application(values["cancelled"]),
    )
    result = runner.invoke(
        app,
        [
            "--database",
            str(tmp_path / "db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "dataset",
            "compile",
            "--request-root",
            str(tmp_path),
            "--request-file",
            "request.json",
            "--json",
        ],
    )
    assert result.exit_code == DatasetExit.CANCELLED
    assert json.loads(result.stdout)["error"]["code"] == "cancelled"
    assert result.stderr == ""


def test_request_path_depth_boundary_and_overflow_do_not_leak_fds(tmp_path: Path) -> None:
    current = tmp_path
    parts: list[str] = []
    for index in range(31):
        part = f"d{index}"
        parts.append(part)
        current /= part
        current.mkdir()
    (current / "request.json").write_text("{}")
    relative = "/".join((*parts, "request.json"))
    before = len(tuple(Path("/dev/fd").iterdir()))
    assert _open_request(tmp_path, relative) == b"{}"
    assert len(tuple(Path("/dev/fd").iterdir())) == before
    with pytest.raises(DatasetCliError) as failure:
        _open_request(tmp_path, "x/" + relative)
    assert failure.value.exit_code == DatasetExit.INTEGRITY
    assert len(tuple(Path("/dev/fd").iterdir())) == before


def test_all_descriptor_close_failures_are_attempted_and_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "request.json").write_text("{}")
    original = os.close
    attempts = 0

    def close_then_fail(descriptor: int) -> None:
        nonlocal attempts
        attempts += 1
        original(descriptor)
        if attempts <= 2:
            raise OSError("/private/secret/close")

    monkeypatch.setattr(os, "close", close_then_fail)
    with pytest.raises(DatasetCliError) as failure:
        _open_request(tmp_path, "request.json")
    assert failure.value.safe_message == "request descriptor cleanup failed"
    assert attempts >= 3
    assert "/private" not in failure.value.safe_message


def test_request_rejects_socket_device_and_ancestor_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    socket_path = Path("request.sock")
    channel = socket.socket(socket.AF_UNIX)
    socket_created = False
    try:
        try:
            channel.bind(str(socket_path))
        except PermissionError:
            pass
        else:
            socket_created = True
            with pytest.raises(DatasetCliError):
                _open_request(tmp_path, socket_path.name)
    finally:
        channel.close()
        if socket_created:
            socket_path.unlink(missing_ok=True)
    with pytest.raises(DatasetCliError):
        _open_request(Path("/dev"), "null")
    real = tmp_path / "real"
    real.mkdir()
    (real / "request.json").write_text("{}")
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    with pytest.raises(DatasetCliError):
        _open_request(tmp_path, "linked/request.json")


def test_request_replacement_after_open_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}")
    original = os.read
    swapped = False

    def read_then_swap(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        data = original(descriptor, count)
        if data == b"" and not swapped:
            request.rename(tmp_path / "old.json")
            request.write_text("{}")
            swapped = True
        return data

    monkeypatch.setattr(os, "read", read_then_swap)
    with pytest.raises(DatasetCliError):
        _open_request(tmp_path, "request.json")


def test_request_rejects_ancestor_swap_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ancestor = tmp_path / "nested"
    ancestor.mkdir()
    (ancestor / "request.json").write_text("{}")
    original = os.read
    swapped = False

    def read_then_swap(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        data = original(descriptor, count)
        if data == b"" and not swapped:
            detached = tmp_path / "detached"
            ancestor.rename(detached)
            ancestor.symlink_to(detached, target_is_directory=True)
            swapped = True
        return data

    monkeypatch.setattr(os, "read", read_then_swap)
    with pytest.raises(DatasetCliError):
        _open_request(tmp_path, "nested/request.json")


def test_request_rejects_socket_mode_without_opening_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}")
    target_inode = request.stat().st_ino
    original = os.fstat

    def socket_status(descriptor: int) -> os.stat_result:
        info = original(descriptor)
        if info.st_ino == target_inode:
            values = list(info)
            values[0] = stat.S_IFSOCK | 0o600
            return os.stat_result(values)
        return info

    monkeypatch.setattr(os, "fstat", socket_status)
    with pytest.raises(DatasetCliError):
        _open_request(tmp_path, "request.json")


def test_dataset_help_documents_exit_codes_and_command_specific_flags() -> None:
    root = runner.invoke(app, ["dataset", "--help"])
    validate = runner.invoke(app, ["dataset", "validate", "--help"])
    compile_help = runner.invoke(app, ["dataset", "compile", "--help"])
    assert root.exit_code == validate.exit_code == compile_help.exit_code == 0
    assert "64 usage" in root.stdout
    assert "70" in root.stdout and "integrity/security" in root.stdout
    assert "--source-root" in validate.stdout
    assert "--tokenizer-root" not in validate.stdout
    assert "--tokenizer-root" in compile_help.stdout
    assert "--source-root" not in compile_help.stdout


@pytest.mark.parametrize("command", ["compile", "reproduce"])
def test_compile_and_reproduce_reject_offline_before_backend_import(
    command: str, tmp_path: Path
) -> None:
    (tmp_path / "request.json").write_text('{"schema_version":"wrong"}')
    arguments = [
        "--database",
        str(tmp_path / "db"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "dataset",
        command,
        "--request-root",
        str(tmp_path),
        "--request-file",
        "request.json",
        "--tokenizer-root",
        str(tmp_path),
        "--text-encoder-root",
        str(tmp_path),
        "--vae-root",
        str(tmp_path),
        "--scheduler-root",
        str(tmp_path),
        "--json",
    ]
    result = runner.invoke(app, arguments, env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    assert result.exit_code == DatasetExit.REJECTED
    assert result.stderr == ""
