"""Versioned contracts for durable local inference tasks."""

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
    for key in ("result", "error", "supervisor", "execution"):
        if value[key] is not None and not isinstance(value[key], dict):
            raise TaskError("invalid_task_record", "Task execution evidence is invalid.")
    if value["state"] == "succeeded" and value["result"] is None:
        raise TaskError("invalid_task_record", "Successful tasks require a durable result.")
    if value["state"] == "failed" and value["error"] is None:
        raise TaskError("invalid_task_record", "Failed tasks require an actionable error.")
    encoded(value)
    return value


def capabilities():
    return {"schema": "task-capabilities/1", "request_schema": REQUEST_SCHEMA, "run_schema": RUN_SCHEMA,
            "tasks": [{"kind": kind, "result_schema": "decide/1" if kind == "classification" else "music/1",
                       "cancellation": True, "requires_local_model": True} for kind in KINDS]}
