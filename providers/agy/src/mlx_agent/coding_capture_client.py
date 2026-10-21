"""Observed OpenCode v2 session events and explicit-session transfer evidence."""

import json
import math
from typing import Callable, Dict

from .coding_capture_contracts import DEFAULT_LIMITS
from .task_contracts import TaskError, encoded

EVENT_TYPES = frozenset(("step_start", "step_finish", "text", "reasoning", "tool_use", "error"))
MESSAGE_TYPES = frozenset(("agent-switched", "model-switched", "location-switched", "user", "synthetic", "system",
                           "skill", "shell", "assistant", "compaction", "idle"))


def _require(condition, message):
    if not condition:
        raise TaskError("invalid_capture_session", message, "Keep the retained client output; do not substitute another saved session.")


def is_identifier(value, prefix):
    return isinstance(value, str) and value.startswith(prefix) and len(prefix) < len(value) <= 256 and not any(ord(c) < 33 for c in value)


def _milliseconds(value):
    # The supported client emits epoch milliseconds, not seconds or ISO strings.
    return type(value) in (int, float) and math.isfinite(value) and 946684800000 <= value <= 253402300799999


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate keys cannot bind session evidence.")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs)
        encoded(value)  # Also reject nested nonfinite numbers and invalid UTF-8.
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise TaskError("invalid_capture_session", "Client output is not complete finite UTF-8 JSON.") from error


class OpenCodeEvents:
    """Incremental NDJSON reader; the observed root identity cannot change."""

    def __init__(self, on_event: Callable, artifact_bytes: int):
        if type(artifact_bytes) is not int or not 1 <= artifact_bytes <= DEFAULT_LIMITS["artifact_bytes"]:
            raise TaskError("invalid_capture_limits", "Client output requires an accepted artifact byte limit.")
        self.on_event = on_event
        self.artifact_bytes = artifact_bytes
        self.session_id = None
        self.has_error = False
        self._buffer = bytearray()
        self._bytes = 0
        self._finished = False

    def feed(self, chunk: bytes):
        _require(isinstance(chunk, bytes) and not self._finished, "Client event stream has already ended or is not bytes.")
        self._bytes += len(chunk)
        if self._bytes > self.artifact_bytes:
            raise TaskError("capture_artifact_limit", "Client event output exceeds its accepted artifact limit.")
        self._buffer.extend(chunk)
        while True:
            end = self._buffer.find(b"\n")
            if end < 0:
                return
            line = bytes(self._buffer[:end])
            del self._buffer[:end + 1]
            value = _json(line)
            _require(isinstance(value, dict) and value.get("type") in EVENT_TYPES
                     and is_identifier(value.get("sessionID"), "ses") and _milliseconds(value.get("timestamp")),
                     "Client events require a supported type, observed session ID and epoch-millisecond timestamp.")
            if self.session_id is not None:
                _require(value["sessionID"] == self.session_id, "Client events changed the observed root session.")
            self.session_id = value["sessionID"]
            self.has_error = self.has_error or value["type"] == "error"
            self.on_event(value)

    def finish(self):
        _require(not self._buffer, "Client event output ended with a partial JSON line.")
        _require(self.session_id is not None, "No client session was observed.")
        self._finished = True


def validate_session_export(raw: bytes, session_id: str, model: str) -> Dict:
    """Validate identity-bearing v2 transfer fields and retain the original body."""
    _require(isinstance(raw, bytes) and len(raw) <= DEFAULT_LIMITS["artifact_bytes"], "Session export exceeds its byte limit.")
    _require(is_identifier(session_id, "ses") and isinstance(model, str) and "/" in model,
             "An observed session and exact provider/model selection are required.")
    provider, selected_model = model.split("/", 1)
    _require(bool(provider) and bool(selected_model), "The selected model is incomplete.")
    value = _json(raw)
    _require(isinstance(value, dict) and set(value) == {"info", "messages"}, "Unsupported session-transfer shape.")
    info, messages = value["info"], value["messages"]
    _require(isinstance(info, dict) and info.get("id") == session_id and is_identifier(info.get("projectID"), ""),
             "Explicit session export does not match the observed session and project.")
    if "parentID" in info:
        _require(is_identifier(info["parentID"], "ses") and info["parentID"] != session_id, "Invalid exported parent relationship.")
    times = info.get("time")
    _require(isinstance(times, dict) and all(_milliseconds(times.get(key)) for key in ("created", "updated")),
             "Exported session time is not epoch milliseconds.")

    def check_model(reference):
        _require(isinstance(reference, dict) and reference.get("providerID") == provider
                 and reference.get("id") == selected_model, "Exported session model differs from the selected local model.")

    if "model" in info:
        check_model(info["model"])
    _require(isinstance(messages, list) and bool(messages), "Export has no retained messages.")
    identities, assistant = set(), False
    for message in messages:
        _require(isinstance(message, dict) and is_identifier(message.get("id"), "msg_"), "Export contains an invalid message identity.")
        _require(message["id"] not in identities, "Export contains duplicate message identities.")
        identities.add(message["id"])
        _require(message.get("type") in MESSAGE_TYPES, "Export contains an unsupported message type.")
        if message["type"] == "idle":
            _require(message.get("outcome") in ("succeeded", "failed", "interrupted"),
                     "Export contains an unsupported idle outcome.")
        _require("sessionID" not in message or message["sessionID"] == session_id, "Export contains another session's message.")
        if "model" in message:
            check_model(message["model"])
        if message["type"] == "assistant":
            check_model(message.get("model"))
            _require(isinstance(message.get("content"), list), "Exported assistant content is incomplete.")
            assistant = True
    _require(assistant, "Export has no model-attributable assistant message.")
    return value
