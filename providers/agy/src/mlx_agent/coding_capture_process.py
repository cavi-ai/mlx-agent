"""Bounded execution of CLI commands attached to one owned client server."""

import os
import selectors
import subprocess
import time
from urllib.parse import urlsplit

from .capture_platform import capture_process_identity, stop_owned
from .coding_capture_client import OpenCodeEvents, is_identifier, validate_session_export
from .coding_capture_contracts import DEFAULT_LIMITS
from .task_contracts import MAX_REQUEST_BYTES, TaskError


def _execute(profile, arguments, payload, role, stdout_name, stderr_name, on_stdout=None):
    """Retain a bounded prefix on failure; it can never become successful evidence."""
    limit = profile["artifact_bytes"]
    if type(limit) is not int or not 1 <= limit <= DEFAULT_LIMITS["artifact_bytes"]:
        raise TaskError("invalid_capture_limits", "Client output requires an accepted byte limit.")
    try:
        url = urlsplit(profile["server_url"])
        valid = (url.scheme == "http" and url.hostname == "127.0.0.1" and url.port is not None
                 and 0 < url.port < 65536 and url.path in ("", "/")
                 and not url.username and not url.password and not url.query and not url.fragment)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise TaskError("capture_client_unavailable", "Select the authenticated owned loopback client server.")
    profile["check_context"]()
    if profile["cancelled"]():
        raise TaskError("capture_cancelled", "Capture was cancelled before the client command started.")
    if profile["environment"].get("PWD") != profile["project_path"]:
        raise TaskError("capture_context_changed", "Client PWD differs from the accepted project.")

    started = time.monotonic()
    process, identity, failure, finished, unresolved = None, None, None, None, False
    output, diagnostic = bytearray(), bytearray()
    selector = selectors.DefaultSelector()
    try:
        environment = dict(profile["environment"])
        environment.pop("MLX_AGENT_CAPTURE_EXTENSION_FD", None)
        process = subprocess.Popen([profile["client_path"], *arguments], cwd=profile["project_path"],
                                   env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        identity = capture_process_identity(process.pid)
        # The supervisor must persist ownership before any prompt bytes leave.
        profile["on_process"](role, identity)
        for stream, data in ((process.stdout, output), (process.stderr, diagnostic)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, data)
        if payload:
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE, None)
        else:
            process.stdin.close()
        offset, drain_deadline = 0, None
        while selector.get_map() or process.poll() is None:
            if profile["cancelled"]():
                raise TaskError("capture_cancelled", "Capture was cancelled during the client command.")
            if process.poll() is not None:
                drain_deadline = drain_deadline or time.monotonic() + 5
                if time.monotonic() >= drain_deadline:
                    raise TaskError("capture_execution_unresolved", "A client descendant retained an output stream.")
            for key, event in selector.select(0.2):
                if event & selectors.EVENT_WRITE:
                    try:
                        offset += os.write(key.fd, payload[offset:offset + 16384])
                    except BrokenPipeError:
                        offset = len(payload)
                    if offset == len(payload):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                chunk = os.read(key.fd, 16384)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                buffer = key.data
                remaining = limit - len(buffer)
                buffer.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    raise TaskError("capture_artifact_limit", "Client command output exceeded its accepted artifact limit.")
                if buffer is output and on_stdout is not None:
                    on_stdout(chunk)
        process.wait(timeout=5)
        finished = time.monotonic()
        profile["check_context"]()
    except TaskError as error:
        failure = error
        unresolved = error.code == "capture_execution_unresolved"
    except (OSError, subprocess.SubprocessError) as error:
        failure = TaskError("capture_client_failed", "The owned client command could not finish.",
                            "Inspect its retained diagnostics and start a new capture.")
    finally:
        try:
            if process is not None:
                try:
                    stop_owned(process, identity)
                except (TaskError, OSError, subprocess.SubprocessError):
                    failure = TaskError("capture_ownership_lost", "Owned command termination remains unresolved.",
                                        "Retain the capture and inspect its recorded process identities.")
                    unresolved = True
        finally:
            selector.close()
            if process is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()
    elapsed = (finished or time.monotonic()) - started
    artifacts = [profile["storage"].write_artifact(stdout_name, bytes(output)),
                 profile["storage"].write_artifact(stderr_name, bytes(diagnostic))]
    return {"exit_code": process.returncode if process is not None else None,
            "elapsed_seconds": elapsed, "process": identity, "artifacts": artifacts,
            "stdout": bytes(output), "execution_unresolved": unresolved,
            "error": failure.as_dict() if failure else None}


def run_client(profile, prompt, on_event):
    try:
        payload = prompt.encode("utf-8")
    except (AttributeError, UnicodeError) as error:
        raise TaskError("invalid_capture", "A coding request must be UTF-8 text.") from error
    if not payload.strip() or len(payload) > MAX_REQUEST_BYTES or b"\0" in payload:
        raise TaskError("invalid_capture", "A coding request must contain at most 1 MiB of text.")
    events = OpenCodeEvents(on_event, profile["artifact_bytes"])
    result = _execute(profile, ["run", "--server", profile["server_url"], "--format", "json", "--model", profile["model"]],
                      payload, "client", "client.ndjson", "client-stderr.log", events.feed)
    result.pop("stdout")
    if result["error"] is None and result["exit_code"] != 0:
        result["error"] = TaskError("capture_client_failed", "The owned client returned a nonzero exit code.",
                                   "Inspect the retained client diagnostics.").as_dict()
    if result["error"] is None:
        try:
            events.finish()
            if events.has_error:
                raise TaskError("capture_client_failed", "The client reported a session error.")
        except TaskError as error:
            result["error"] = error.as_dict()
    result.update(session_id=events.session_id, has_error=events.has_error)
    return result


def export_session(profile, session_id, storage):
    if not is_identifier(session_id, "ses"):
        raise TaskError("invalid_capture_session", "Export requires an observed explicit session identity.")
    result = _execute(dict(profile, storage=storage),
                      ["session", "export", session_id, "--server", profile["server_url"]],
                      b"", "exporter", "session.json", "export-stderr.log")
    raw = result.pop("stdout")
    result["session"] = None
    if result["error"] is None and result["exit_code"] != 0:
        result["error"] = TaskError("capture_export_failed", "The explicit session export returned a nonzero exit code.",
                                   "Keep the bounded profile and diagnostics for recovery.").as_dict()
    if result["error"] is None:
        try:
            session = validate_session_export(raw, session_id, profile["model"])
            if session["info"].get("location", {}).get("directory") != profile["project_path"]:
                raise TaskError("invalid_capture_session", "Export does not describe the accepted project directory.")
            result["session"] = session
        except TaskError as error:
            result["error"] = error.as_dict()
    return result
