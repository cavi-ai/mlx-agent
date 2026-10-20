"""Strict, versioned coding-capture contracts, separate from inference tasks."""

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Dict
from urllib.parse import urlsplit

from .task_contracts import MAX_REQUEST_BYTES, STATES, TERMINAL, TaskError, encoded, task_id

KIND = "coding_capture"
REQUEST_SCHEMA = "coding-capture-request/1"
RUN_SCHEMA = "coding-capture-run/1"
RESULT_SCHEMA = "coding-capture-result/1"
DEFAULT_LIMITS = {"artifact_bytes": 33554432, "total_bytes": 134217728}
METADATA_RESERVE = 65536
MAX_RECORD_BYTES = 8192
PROCESS_ROLES = frozenset(("supervisor", "client", "private_server", "relay"))


def _require(condition, message, code="invalid_capture"):
    if not condition:
        raise TaskError(code, message)


def _fields(value, required, optional=()):
    _require(isinstance(value, dict) and set(required) <= set(value)
             and not set(value) - set(required) - set(optional), "Unknown or incomplete capture fields.")


def _text(value, maximum=4096):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum and "\0" not in value


def _path(value):
    return (_text(value) and Path(value).is_absolute() and ".." not in Path(value).parts
            and not any(ord(char) < 32 for char in value))


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _integer(value, minimum=0, maximum=2 ** 63 - 1):
    return type(value) is int and minimum <= value <= maximum


def _timestamp(value):
    try:
        return isinstance(value, str) and datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def _copy(value):
    return json.loads(encoded(value))


def validate_limits(value):
    _fields(value, ("artifact_bytes", "total_bytes"))
    _require(_integer(value["artifact_bytes"], 1, DEFAULT_LIMITS["artifact_bytes"])
             and _integer(value["total_bytes"], 2 * METADATA_RESERVE, DEFAULT_LIMITS["total_bytes"])
             and value["total_bytes"] % 512 == 0
             and value["artifact_bytes"] <= value["total_bytes"] - METADATA_RESERVE,
             "Capture limits require whole bytes within the supported artifact and owned-data limits.")
    return _copy(value)


def validate_reference(value, expected_name=None):
    """References expose relative artifact names, never a mounted filesystem path."""
    _fields(value, ("name", "sha256", "bytes"))
    name = value["name"]
    _require(_text(name, 128) and not name.startswith(".") and "/" not in name and "\\" not in name
             and not any(ord(char) < 32 for char in name)
             and (expected_name is None or name == expected_name)
             and _digest(value["sha256"]) and _integer(value["bytes"], 0, DEFAULT_LIMITS["artifact_bytes"]),
             "Capture artifact identity is invalid.")
    return value


def _receipt(value):
    _fields(value, ("path", "sha256", "bytes"))
    _require(_path(value["path"]) and _digest(value["sha256"])
             and _integer(value["bytes"], 1, MAX_REQUEST_BYTES), "Serve receipt identity is invalid.")


def _workbench(value):
    _fields(value, ("model_signature", "environment_fingerprint", "workload_id"))
    _require(_text(value["model_signature"], 512) and _text(value["environment_fingerprint"], 512),
             "Original Workbench model and environment identities are required.")
    task_id(value["workload_id"])


def validate_process(value):
    _fields(value, ("pid", "identity", "started_at"))
    _require(_integer(value["pid"], 1) and _text(value["identity"], 2048)
             and isinstance(value["started_at"], list) and len(value["started_at"]) == 2
             and _integer(value["started_at"][0], 1) and _integer(value["started_at"][1], 0, 999999),
             "Capture processes require PID, command and precise kernel start identity.")


def validate_context(value):
    _fields(value, ("serve", "client", "project"), ("workbench_context",))
    serve, client, project = value["serve"], value["client"], value["project"]
    _fields(serve, ("receipt", "base_url", "model_path", "model_fingerprint", "runtime", "process"))
    _receipt(serve["receipt"])
    validate_process(serve["process"])
    try:
        url = urlsplit(serve["base_url"])
        local_url = (url.scheme == "http" and url.hostname == "127.0.0.1" and url.port is not None
                     and 1 <= url.port <= 65535 and not url.username and not url.password
                     and not url.query and not url.fragment and url.path == "/v1")
    except (ValueError, TypeError, AttributeError):
        local_url = False
    _require(local_url and _path(serve["model_path"]) and _digest(serve["model_fingerprint"])
             and _text(serve["runtime"], 128), "Capture requires a pinned local compatible endpoint and model.")
    _fields(client, ("path", "file_identity", "version", "configuration_sha256", "policy_sha256"))
    _require(_path(client["path"]) and _text(client["version"], 128)
             and _digest(client["configuration_sha256"]) and _digest(client["policy_sha256"])
             and isinstance(client["file_identity"], list) and len(client["file_identity"]) == 4
             and all(_integer(part) for part in client["file_identity"]), "Capture client identity is invalid.")
    _fields(project, ("path", "file_identity", "context_sha256"))
    _require(_path(project["path"]) and _digest(project["context_sha256"])
             and isinstance(project["file_identity"], list) and len(project["file_identity"]) == 2
             and all(_integer(part) for part in project["file_identity"]), "Capture project identity is invalid.")
    if "workbench_context" in value:
        _workbench(value["workbench_context"])
    return _copy(value)


def validate_capture_request(value: Dict) -> Dict:
    _fields(value, ("schema", "kind", "model_path", "serve_receipt", "client_path", "project_path", "prompt", "limits"),
            ("workbench_context",))
    _require(value["schema"] == REQUEST_SCHEMA and value["kind"] == KIND,
             "Unsupported coding capture request.", "unsupported_capture")
    _require(all(_path(value[key]) for key in ("model_path", "client_path", "project_path"))
             and _text(value["prompt"], MAX_REQUEST_BYTES), "Capture requires absolute local paths and a coding request.")
    _receipt(value["serve_receipt"])
    validate_limits(value["limits"])
    if "workbench_context" in value:
        _workbench(value["workbench_context"])
    _require(len(encoded(value)) <= MAX_REQUEST_BYTES, "Capture requests are limited to 1 MiB.", "request_too_large")
    return _copy(value)


def validate_capture_record(value: Dict) -> Dict:
    _fields(value, ("schema", "id", "kind", "state", "created_at", "started_at", "completed_at", "cancel_requested",
                    "request", "preview_hash", "context", "processes", "client_outcome", "attribution", "result",
                    "error", "execution_unresolved", "artifacts_removed_at"))
    _require(value["schema"] == RUN_SCHEMA and value["kind"] == KIND and value["state"] in STATES,
             "Unsupported capture record.")
    task_id(value["id"])
    validate_reference(value["request"], "request.json")
    _require(_digest(value["preview_hash"]), "Capture preview identity is invalid.")
    validate_context(value["context"])
    _require(all(type(value[key]) is bool for key in ("cancel_requested", "execution_unresolved")),
             "Capture lifecycle flags must be boolean.")
    _require(_timestamp(value["created_at"]) and all(value[key] is None or _timestamp(value[key])
             for key in ("started_at", "completed_at", "artifacts_removed_at")), "Capture timestamps need a timezone.")
    state = value["state"]
    _require((state in TERMINAL) == (value["completed_at"] is not None), "Capture completion state is inconsistent.")
    _require(value["client_outcome"] in ("pending", "succeeded", "failed", "cancelled")
             and value["attribution"] in ("pending", "verified", "failed"), "Unknown capture outcome.")
    processes = value["processes"]
    _require(isinstance(processes, dict) and not set(processes) - PROCESS_ROLES,
             "Only capture-owned process roles may be recorded.")
    for process in processes.values():
        validate_process(process)
    _require(not set(processes) - {"supervisor"} or "supervisor" in processes,
             "Owned capture execution requires its supervisor.")
    if value["result"] is not None:
        validate_reference(value["result"], "result.json")
    if value["error"] is not None:
        _fields(value["error"], ("code", "message", "remediation"))
        _require(all(_text(part) for part in value["error"].values()), "Capture errors require actionable details.")
    _require(state != "starting" or value["started_at"] is None, "Starting capture cannot have a start timestamp.")
    _require(state != "running" or (value["started_at"] is not None and {"supervisor", "client"} <= set(processes)),
             "Running capture requires client and supervisor identities.")
    if state == "succeeded":
        _require(value["client_outcome"] == "succeeded" and value["attribution"] == "verified"
                 and value["result"] is not None and value["error"] is None and not value["execution_unresolved"]
                 and value["started_at"] is not None and {"supervisor", "client"} <= set(processes),
                 "Successful capture requires durable attributed execution evidence.")
    _require(state == "succeeded" or value["result"] is None, "Only successful capture publishes a result reference.")
    _require((state == "failed") == (value["error"] is not None), "Failed capture requires an actionable error.")
    _require(not value["execution_unresolved"] or state == "failed", "Unresolved execution must remain failed.")
    _require(state != "cancelled" or (value["cancel_requested"] and value["client_outcome"] == "cancelled"
             and not value["execution_unresolved"]), "Capture cancellation evidence is inconsistent.")
    _require(value["artifacts_removed_at"] is None or (state in TERMINAL and not value["execution_unresolved"]),
             "Active or unresolved capture artifacts cannot be removed.")
    _require(len(encoded(value)) <= MAX_RECORD_BYTES, "Capture task metadata exceeds 8,192 bytes.", "capture_metadata_limit")
    return value


def validate_capture_result(value: Dict) -> Dict:
    _fields(value, ("schema", "task_id", "request_sha256", "preview_hash", "context", "session", "requests", "artifacts",
                    "client_exit_code", "elapsed_seconds", "completed_at", "attribution"))
    _require(value["schema"] == RESULT_SCHEMA, "Unsupported capture result.")
    task_id(value["task_id"])
    _require(_digest(value["request_sha256"]) and _digest(value["preview_hash"]), "Capture result hashes are invalid.")
    validate_context(value["context"])
    session = value["session"]
    _fields(session, ("id", "parent_id", "children"))
    _require(_text(session["id"], 128) and (session["parent_id"] is None or _text(session["parent_id"], 128))
             and session["parent_id"] != session["id"] and isinstance(session["children"], list), "Invalid captured session.")
    sessions = {session["id"]}
    for child in session["children"]:
        _fields(child, ("id", "parent_id"))
        _require(_text(child["id"], 128) and child["id"] not in sessions
                 and _text(child["parent_id"], 128) and child["parent_id"] in sessions,
                 "Capture child sessions require an observed parent relationship.")
        sessions.add(child["id"])
    _require(isinstance(value["artifacts"], list) and isinstance(value["requests"], list), "Invalid capture artifact set.")
    artifacts = {}
    for artifact in value["artifacts"]:
        validate_reference(artifact)
        _require(artifact["name"] not in artifacts, "Capture artifact names must be unique.")
        artifacts[artifact["name"]] = artifact
    for request in value["requests"]:
        _fields(request, ("artifact", "session_id", "parent_session_id", "project_id", "request_model", "response_model"))
        validate_reference(request["artifact"])
        _require(artifacts.get(request["artifact"]["name"]) == request["artifact"], "Capture transport reference is missing.")
        _require(_text(request["session_id"], 128)
                 and (request["parent_session_id"] is None or _text(request["parent_session_id"], 128))
                 and (request["project_id"] is None or _text(request["project_id"], 256))
                 and _text(request["request_model"], 256)
                 and (request["response_model"] is None or _text(request["response_model"], 256)), "Invalid observed request identity.")
        if value["attribution"] == "verified":
            _require(request["session_id"] in sessions and request["response_model"] is not None,
                     "Verified transport must join an exported session and response model.")
    elapsed = value["elapsed_seconds"]
    _require(_integer(value["client_exit_code"], -255, 255) and type(elapsed) in (int, float)
             and math.isfinite(elapsed) and elapsed >= 0 and _timestamp(value["completed_at"])
             and value["attribution"] in ("pending", "verified", "failed"), "Capture execution outcome is invalid.")
    _require(value["attribution"] != "verified" or bool(value["requests"]), "Verified capture requires observed requests.")
    _require(len(encoded(value)) <= DEFAULT_LIMITS["artifact_bytes"], "Capture result exceeds its artifact limit.")
    return _copy(value)


def capture_capabilities() -> Dict:
    return {"schema": "coding-capture-capabilities/1", "kind": KIND, "request_schema": REQUEST_SCHEMA,
            "run_schema": RUN_SCHEMA, "result_schema": RESULT_SCHEMA, "limits": dict(DEFAULT_LIMITS),
            "metadata_reserve_bytes": METADATA_RESERVE, "cancellation": True, "explicit_removal": True}
