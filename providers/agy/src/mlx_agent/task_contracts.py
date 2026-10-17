"""Versioned contracts for durable local inference tasks."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

from .decide import DecideError, validate_request as validate_decision
from .music import MusicError, validate_parameters as validate_music

REQUEST_SCHEMA = "task-request/1"
RUN_SCHEMA = "task-run/1"
MAX_REQUEST_BYTES = 1024 * 1024
KINDS = ("classification", "music_generation")
STATES = ("starting", "running", "succeeded", "failed", "cancelled")
TERMINAL = ("succeeded", "failed", "cancelled")


class TaskError(RuntimeError):
    def __init__(self, code, message, remediation="Inspect the task inputs and retry."):
        super().__init__(message)
        self.code, self.remediation = code, remediation

    def as_dict(self):
        return {"code": self.code, "message": str(self), "remediation": self.remediation}


def encoded(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as error:
        raise TaskError("invalid_task", "Task content must be finite UTF-8 JSON.") from error


def task_id(value):
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError()
    except ValueError as error:
        raise TaskError("invalid_task_id", "Pass a canonical task UUID.") from error
    return value


def validate_request(value):
    if not isinstance(value, dict) or set(value) != {"schema", "kind", "model_path", "parameters"}:
        raise TaskError("invalid_request", "A task request needs schema, kind, model_path and parameters.")
    if value["schema"] != REQUEST_SCHEMA or value["kind"] not in KINDS:
        raise TaskError("unsupported_task", "Unsupported task schema or kind.", "Run mlx-agent task capabilities --json.")
    path = value["model_path"]
    if not isinstance(path, str) or not Path(path).is_absolute() or any(ord(c) < 32 for c in path):
        raise TaskError("invalid_request", "model_path must be an absolute local path.")
    parameters = value["parameters"]
    if not isinstance(parameters, dict):
        raise TaskError("invalid_request", "parameters must be an object.")
    try:
        if value["kind"] == "classification":
            if set(parameters) != {"state", "questions"}:
                raise TaskError("invalid_request", "Classification parameters need state and questions.")
            validate_decision(parameters)
        else:
            if "caption" not in parameters or set(parameters) - {"caption", "lyrics", "duration", "steps", "seed"}:
                raise TaskError("invalid_request", "Music parameters need a caption and supported generation settings.")
            parameters = validate_music(parameters["caption"], parameters.get("lyrics", "[instrumental]"),
                                        parameters.get("duration", 15.0), parameters.get("steps", 30), parameters.get("seed", 42))
    except (DecideError, MusicError) as error:
        raise TaskError(error.code, str(error), error.remediation) from error
    result = dict(value, parameters=parameters)
    raw = encoded(result)
    if len(raw) > MAX_REQUEST_BYTES:
        raise TaskError("request_too_large", "Task requests are limited to 1 MiB.")
    return json.loads(raw)


def validate_record(value):
    required = {"schema", "id", "kind", "state", "model_path", "created_at", "started_at", "completed_at",
                "cancel_requested", "identity", "request", "result", "error", "supervisor", "execution", "execution_unresolved"}
    if not isinstance(value, dict) or set(value) != required or value.get("schema") != RUN_SCHEMA:
        raise TaskError("invalid_task_record", "Unknown or incomplete task record.", "Preserve the store and use a compatible engine.")
    task_id(value["id"])
    if value["kind"] not in KINDS or value["state"] not in STATES:
        raise TaskError("invalid_task_record", "Unknown task kind or state.")
    if not isinstance(value["model_path"], str) or not Path(value["model_path"]).is_absolute():
        raise TaskError("invalid_task_record", "The task model path is invalid.")
    if any(type(value[key]) is not bool for key in ("cancel_requested", "execution_unresolved")):
        raise TaskError("invalid_task_record", "Task lifecycle flags must be boolean.")
    for key in ("created_at", "started_at", "completed_at"):
        if value[key] is None and key != "created_at":
            continue
        try:
            stamp = datetime.fromisoformat(value[key])
            if stamp.tzinfo is None:
                raise ValueError()
        except (ValueError, TypeError) as error:
            raise TaskError("invalid_task_record", "Task timestamps require a timezone.") from error
    if (value["state"] in TERMINAL) != (value["completed_at"] is not None):
        raise TaskError("invalid_task_record", "Task completion timestamp does not match its state.")
    for key in ("identity", "request"):
        if not isinstance(value[key], dict):
            raise TaskError("invalid_task_record", "Task provenance is missing.")
    identity = value["identity"]
    identity_fields = {"schema", "requested_model_path", "model_path", "model_fingerprint", "backend",
                       "backend_version", "backend_manifest", "backend_manifest_hash", "requested_executable",
                       "executable", "executable_identity", "engine_version", "python_version"}
    if (set(identity) != identity_fields or identity.get("schema") != "task-identity/1"
            or not _absolute_path(identity.get("requested_model_path"))
            or Path(identity["requested_model_path"]).absolute() != Path(value["model_path"]).absolute()
            or not _absolute_path(identity.get("model_path"))
            or not _sha256(identity.get("model_fingerprint"))
            or not _nonempty_string(identity.get("backend"))
            or not _nonempty_string(identity.get("backend_version"))
            or not isinstance(identity.get("backend_manifest"), dict)
            or identity["backend_manifest"].get("id") != identity.get("backend")
            or identity["backend_manifest"].get("version") != identity.get("backend_version")
            or not _sha256(identity.get("backend_manifest_hash"))
            or hashlib.sha256(encoded(identity["backend_manifest"])).hexdigest() != identity["backend_manifest_hash"]
            or not _absolute_path(identity.get("requested_executable"))
            or not _absolute_path(identity.get("executable"))
            or not _file_identity(identity.get("executable_identity"))
            or not _nonempty_string(identity.get("engine_version"))
            or not _nonempty_string(identity.get("python_version"))):
        raise TaskError("invalid_task_record", "Task model and runtime identity evidence is invalid.")
    _artifact_reference(value["request"])
    if value["result"] is not None:
        _artifact_reference(value["result"])
    if value["error"] is not None:
        error = value["error"]
        if (not isinstance(error, dict) or set(error) != {"code", "message", "remediation"}
                or any(not _nonempty_string(error.get(key)) for key in ("code", "message", "remediation"))):
            raise TaskError("invalid_task_record", "Task error evidence is invalid.")
    for key in ("supervisor", "execution"):
        process = value[key]
        if process is not None and (not isinstance(process, dict) or set(process) != {"pid", "identity"}
                or type(process.get("pid")) is not int or process["pid"] <= 0
                or not _nonempty_string(process.get("identity"))):
            raise TaskError("invalid_task_record", "Task process identity evidence is invalid.")
    if value["execution"] is not None and value["supervisor"] is None:
        raise TaskError("invalid_task_record", "Task execution evidence has no supervisor.")
    if value["state"] == "succeeded" and value["result"] is None:
        raise TaskError("invalid_task_record", "Successful tasks require a durable result.")
    if value["state"] == "failed" and value["error"] is None:
        raise TaskError("invalid_task_record", "Failed tasks require an actionable error.")
    if value["state"] == "running" and (value["started_at"] is None or value["supervisor"] is None or value["execution"] is None):
        raise TaskError("invalid_task_record", "Running tasks require start and process evidence.")
    if value["state"] == "succeeded" and (value["started_at"] is None or value["supervisor"] is None
            or value["execution"] is None or value["error"] is not None or value["execution_unresolved"]):
        raise TaskError("invalid_task_record", "Successful task lifecycle evidence is inconsistent.")
    if value["state"] != "succeeded" and value["result"] is not None:
        raise TaskError("invalid_task_record", "Only successful tasks may contain result evidence.")
    if value["state"] == "starting" and value["started_at"] is not None:
        raise TaskError("invalid_task_record", "Starting tasks cannot have a start timestamp.")
    if value["state"] == "cancelled" and (not value["cancel_requested"] or value["result"] is not None
            or value["error"] is not None or value["execution_unresolved"]):
        raise TaskError("invalid_task_record", "Cancelled task lifecycle evidence is inconsistent.")
    if value["state"] not in ("succeeded", "failed") and value["error"] is not None:
        raise TaskError("invalid_task_record", "Only failed tasks may contain error evidence.")
    if value["execution_unresolved"] and value["state"] != "failed":
        raise TaskError("invalid_task_record", "Only failed tasks may retain unresolved execution evidence.")
    encoded(value)
    return value


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _absolute_path(value):
    return isinstance(value, str) and bool(value) and Path(value).is_absolute() and not any(ord(char) < 32 for char in value)


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _file_identity(value):
    return isinstance(value, list) and len(value) == 4 and all(type(part) is int and part >= 0 for part in value)


def _artifact_reference(value):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}
            or not _absolute_path(value.get("path")) or not _sha256(value.get("sha256"))
            or type(value.get("bytes")) is not int or value["bytes"] < 0):
        raise TaskError("invalid_task_record", "Task artifact evidence is invalid.")


def capabilities():
    return {"schema": "task-capabilities/1", "request_schema": REQUEST_SCHEMA, "run_schema": RUN_SCHEMA,
            "tasks": [{"kind": kind, "result_schema": "decide/1" if kind == "classification" else "music/1",
                       "cancellation": True, "requires_local_model": True} for kind in KINDS]}
