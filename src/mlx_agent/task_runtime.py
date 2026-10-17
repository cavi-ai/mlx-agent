"""Durable local inference supervised independently of the calling client."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

from .backends import BackendError, load_manifests
from .compute import ComputeBusy, ComputeLease, group_members, process_alive, process_identity
from .decide import DecideError, plan_decide, run_decide
from .music import MusicError, plan_music, run_music
from .task_contracts import MAX_REQUEST_BYTES, TERMINAL, TaskError, encoded, validate_request
from .task_identity import capture_identity, check_identity
from .task_store import TaskStore
from .wiring import redact_secrets

POLL_SECONDS = .1
STARTUP_SECONDS = 15
STOP_SECONDS = 5


def _error(error):
    return {"code": getattr(error, "code", "task_execution_failed"),
            "message": redact_secrets(str(error))[-2000:],
            "remediation": getattr(error, "remediation", "Inspect this task's local log and retry after resolving the failure.")}


def _write_json(path, value):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def _plan(request, directory):
    parameters = request["parameters"]
    if request["kind"] == "classification":
        path = directory / "decision.json"
        _write_json(path, parameters)
        return plan_decide(request["model_path"], str(path))
    return plan_music(request["model_path"], parameters["caption"], parameters["lyrics"], str(directory / "output.wav"),
                      duration=parameters["duration"], steps=parameters["steps"], seed=parameters["seed"])


def _spawn(role, identifier, store, launcher=subprocess.Popen):
    path = store.artifact_path(identifier, role + ".log")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    try:
        with os.fdopen(descriptor, "wb") as output:
            return launcher([sys.executable, "-m", "mlx_agent.task_worker", "--role", role, "--id", identifier,
                             "--state-dir", str(store.root)], stdin=subprocess.DEVNULL, stdout=output,
                            stderr=subprocess.STDOUT, start_new_session=True, env=environment, umask=0o077)
    except OSError as error:
        raise TaskError("task_launch_failed", "Cannot launch the task worker: {0}".format(error)) from error


def start_task(request_path, state_dir=None, launcher=subprocess.Popen):
    store = TaskStore(state_dir)
    try:
        path = Path(request_path)
        if path.is_symlink() or not path.is_file():
            raise TaskError("invalid_request", "The task request must be a readable regular file.")
        with path.open("rb") as stream:
            raw = stream.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise TaskError("request_too_large", "Task requests are limited to 1 MiB.")
        request = validate_request(json.loads(raw))
        with tempfile.TemporaryDirectory(prefix="mlx-task-plan-") as directory:
            plan = _plan(request, Path(directory))
            identity = capture_identity(Path(request["model_path"]), load_manifests()[plan["backend"]], Path(plan["argv"][0]))
    except (OSError, ValueError, DecideError, MusicError, BackendError) as error:
        raise TaskError(getattr(error, "code", "invalid_request"), str(error), getattr(error, "remediation", "Check the local model, request and installed backend.")) from error
    record = store.create(request, identity)
    try:
        _spawn("supervisor", record["id"], store, launcher)
    except TaskError as error:
        return store.transition(record["id"], ("starting",), {"state": "failed", "error": _error(error)})
    return store.get(record["id"])


def _unresolved(record):
    execution = record["execution"]
    if not execution:
        return False
    try:
        return bool(group_members(execution["pid"]))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return True


def task_status(identifier, state_dir=None):
    store = TaskStore(state_dir)
    record = store.get(identifier)
    if record["state"] in TERMINAL:
        return record
    owner = record["supervisor"]
    if owner:
        observed = process_identity(owner["pid"])
        alive = process_alive(owner["pid"])
        if alive and observed == owner["identity"]:
            return record
        if alive and observed is None:
            return dict(record, error={"code": "task_identity_unavailable", "message": "The live supervisor identity could not be inspected.", "remediation": "Retry status; no process was signaled."})
    else:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(record["created_at"])).total_seconds()
        if age < STARTUP_SECONDS:
            return record
    try:
        return store.transition(identifier, ("starting", "running"), {"state": "failed", "execution_unresolved": _unresolved(record),
            "error": {"code": "task_interrupted", "message": "The task supervisor stopped before recording a result.",
                      "remediation": "Inspect the task log and any unresolved execution before starting new work."}})
    except TaskError as error:
        if error.code == "task_state_changed":
            return store.get(identifier)
        raise


def cancel_task(identifier, state_dir=None):
    return TaskStore(state_dir).request_cancel(identifier)


def task_result(identifier, state_dir=None):
    store = TaskStore(state_dir)
    record = task_status(identifier, state_dir)
    if record["state"] != "succeeded":
        raise TaskError("task_not_succeeded", "This task has no successful result.", "Inspect its status and error before retrying.")
    path = store.artifact_path(identifier, "result.json")
    reference = record["result"]
    try:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(65536), b""):
                size += len(chunk)
                digest.update(chunk)
        if str(path) != reference.get("path") or size != reference.get("bytes") or digest.hexdigest() != reference.get("sha256"):
            raise TaskError("result_changed", "The saved task result changed.")
        inline = size <= 8192
        result = json.loads(path.read_bytes()) if inline else None
        if record["kind"] == "music_generation":
            # Music metadata is small; the audio remains a separately owned artifact.
            result = result if result is not None else json.loads(path.read_bytes())
            output = _artifact_reference(store.artifact_path(identifier, "output.wav"))
            if output != result.get("output_artifact"):
                raise TaskError("result_changed", "The saved music output changed.")
        return {"task": record, "artifact": reference, "inline": inline, "result": result if inline else None}
    except (OSError, ValueError) as error:
        raise TaskError("result_unavailable", "The saved task result is unavailable.") from error


def _artifact_reference(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}


def _known_members(process, identity, previous):
    members = group_members(process.pid)
    if process_identity(process.pid) == identity:
        return {pid: observed for pid in members if (observed := process_identity(pid))}
    return {pid: value for pid, value in previous.items() if pid in members and process_identity(pid) == value}


def _stop(process, identity, known, grace=STOP_SECONDS):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        members = group_members(process.pid)
        if not members:
            process.poll()
            return True
        known = _known_members(process, identity, known)
        if not known:
            return False
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return True
        except OSError:
            if not group_members(process.pid):
                return True
            raise
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            process.poll()
            if not group_members(process.pid):
                return True
            time.sleep(POLL_SECONDS)
    return not group_members(process.pid)


def _finish(store, identifier, state, **changes):
    try:
        return store.transition(identifier, ("starting", "running"), dict(changes, state=state))
    except TaskError as error:
        if error.code == "task_cancel_requested" and state == "succeeded":
            return store.transition(identifier, ("starting", "running"), {"state": "cancelled", "error": None})
        if error.code != "task_state_changed":
            raise
        return store.get(identifier)


def supervise(identifier, state_dir, lease_factory=ComputeLease):
    store = TaskStore(state_dir)
    process, identity, known = None, None, {}
    claimed = False
    try:
        own_identity = process_identity(os.getpid())
        if not own_identity:
            raise TaskError("task_identity_unavailable", "Cannot establish the supervisor identity.")
        record = store.get(identifier)
        if record["state"] != "starting" or record["supervisor"] is not None:
            return 1
        store.transition(identifier, ("starting",), {"supervisor": {"pid": os.getpid(), "identity": own_identity}})
        claimed = True
        with lease_factory("task:" + identifier) as lease:
            record = store.get(identifier)
            if record["cancel_requested"]:
                _finish(store, identifier, "cancelled")
                return 0
            check_identity(record["identity"])
            store.read_request(identifier)
            process = _spawn("execute", identifier, store)
            identity = process_identity(process.pid)
            if not identity:
                raise TaskError("task_identity_unavailable", "Cannot establish the execution worker identity.")
            known = {process.pid: identity}
            lease.bind_process_group(process.pid)
            store.transition(identifier, ("starting",), {"state": "running", "execution": {"pid": process.pid, "identity": identity}})
            while process.poll() is None:
                known = _known_members(process, identity, known)
                record = store.get(identifier)
                if record["cancel_requested"]:
                    stopped = _stop(process, identity, known)
                    _finish(store, identifier, "cancelled" if stopped else "failed", execution_unresolved=not stopped,
                            error=None if stopped else _error(TaskError("task_stop_unconfirmed", "The task execution group could not be stopped safely.")))
                    return 0 if stopped else 1
                time.sleep(POLL_SECONDS)
            if group_members(process.pid) and not _stop(process, identity, known):
                raise TaskError("task_stop_unconfirmed", "The execution worker exited but its group remains active.")
            record = store.get(identifier)
            if record["cancel_requested"]:
                _finish(store, identifier, "cancelled")
                return 0
            outcome_path = store.artifact_path(identifier, "execution-result.json")
            try:
                outcome = json.loads(outcome_path.read_bytes())
            except (OSError, ValueError) as error:
                raise TaskError("task_execution_failed", "The execution worker did not produce a readable result.") from error
            if process.returncode != 0 or outcome.get("error"):
                error = outcome.get("error") or _error(TaskError("task_execution_failed", "The execution worker failed."))
                _finish(store, identifier, "failed", error=error)
                return 1
            check_identity(record["identity"])
            store.read_request(identifier)
            result = store.write_result(identifier, outcome["result"])
            # Re-read cancellation at the last publication boundary.
            if store.get(identifier)["cancel_requested"]:
                _finish(store, identifier, "cancelled")
            else:
                _finish(store, identifier, "succeeded", result=result, error=None)
            return 0
    except (TaskError, ComputeBusy, OSError, ValueError, subprocess.SubprocessError) as error:
        if not claimed:
            print(str(error), file=sys.stderr)
            return 1
        unresolved = False
        if process is not None:
            try:
                unresolved = not _stop(process, identity, known)
            except (OSError, ValueError, subprocess.SubprocessError):
                unresolved = True
        if isinstance(error, ComputeBusy):
            error = TaskError("compute_busy", str(error), "Wait for the current host model operation to stop, then start a new task.")
        try:
            if not unresolved and store.get(identifier)["cancel_requested"]:
                _finish(store, identifier, "cancelled", error=None)
            else:
                _finish(store, identifier, "failed", error=_error(error), execution_unresolved=unresolved)
        except TaskError as storage_error:
            print(str(storage_error), file=sys.stderr)
        return 1


def execute(identifier, state_dir):
    store = TaskStore(state_dir)
    try:
        # Never run until the supervisor has durably bound the group to its lease.
        deadline = time.monotonic() + STARTUP_SECONDS
        while True:
            record = store.get(identifier)
            if record["state"] in TERMINAL or record["cancel_requested"]:
                return 1
            if record["execution"] and record["execution"]["pid"] == os.getpid() and record["state"] == "running":
                break
            if time.monotonic() >= deadline:
                return 1
            time.sleep(POLL_SECONDS)
        check_identity(record["identity"])
        request = store.read_request(identifier)
        directory = store.artifact_path(identifier, "execution-result.json").parent
        plan = _plan(request, directory)
        result = run_decide(plan) if request["kind"] == "classification" else run_music(plan)
        if request["kind"] == "music_generation":
            path = store.artifact_path(identifier, "output.wav")
            with wave.open(str(path), "rb") as audio:
                if audio.getnframes() <= 0 or audio.getnchannels() <= 0 or not audio.readframes(1):
                    raise TaskError("invalid_audio_output", "Music generation did not produce usable WAV audio.")
            result["output_artifact"] = _artifact_reference(path)
        _write_json(store.artifact_path(identifier, "execution-result.json"), {"result": result})
        return 0
    except (TaskError, DecideError, MusicError, BackendError, OSError, ValueError, wave.Error) as error:
        try:
            _write_json(store.artifact_path(identifier, "execution-result.json"), {"error": _error(error)})
        except (TaskError, OSError):
            pass
        return 1
