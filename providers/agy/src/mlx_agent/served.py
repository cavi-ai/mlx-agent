"""Run a convert verb on a typed endpoint that already serves the same model, so the model loads once."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .inference_runner import InferenceError

TERMINAL = ("succeeded", "failed", "cancelled")
POLL_SECONDS = 0.5
# convert verb -> (task kind, CLI argument names that are task parameters)
VERBS = {
    "speak": ("speak", ("text", "voice", "speed", "lang_code")),
    "music": ("music_generation", ("caption", "lyrics", "duration", "steps", "seed", "bpm", "time_signature",
                                   "reference")),
    "generate": ("image_generation", ("prompt", "width", "height", "steps", "seed")),
    "video": ("video_generation", ("prompt", "width", "height", "frames", "steps", "seed", "fps", "denoiser_cache",
                                   "cache_memory_mib")),
}
REMEDIATION = {
    "model_busy": "The served model is running another job; wait for it to finish and retry.",
}


def request(verb, arguments):
    """The task kind and its parameters for a convert verb, taken from the CLI arguments."""
    if verb == "describe":
        media = "video" if arguments.video else "image"
        names = ("prompt", "max_tokens", "temperature", "max_pixels") + (("fps",) if media == "video" else ())
        parameters = {name: getattr(arguments, name) for name in names}
        parameters[media + "_path"] = str(Path(getattr(arguments, media)).expanduser().resolve())
        kind = media + "_understanding"
    else:
        kind, names = VERBS[verb]
        parameters = {name: getattr(arguments, name) for name in names}
    return kind, {name: value for name, value in parameters.items() if value is not None}


def endpoint(model, kind, discover=None):
    """The live typed endpoint bound to this model directory that advertises this kind, or None."""
    if discover is None:
        from .service_registry import discover_services as discover
    wanted = Path(model).expanduser().resolve()
    try:
        services = discover()["services"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    for service in services:
        try:
            if (service.get("runtime") == "tasks" and service.get("alive") and service.get("argv_match")
                    and kind in (service.get("task_kinds") or ()) and Path(service["path"]).resolve() == wanted):
                return service
        except (OSError, TypeError, KeyError):
            continue
    return None


def _key(service):
    from .serve import receipts_root
    from .serving_access import read_access_key
    from .workflow_sets import _private_json
    try:
        config = _private_json(receipts_root(service["receipts_dir"]) / "{0}.jit-config.json".format(service["port"]))
        if config.get("access_key_file") is None:
            if config.get("api_key_sha256") is None:
                return None
            raise ValueError("no key file")
        return read_access_key(config["access_key_file"], config.get("api_key_sha256"))[0]
    except (OSError, ValueError, KeyError) as error:
        raise InferenceError(
            "endpoint_key_unavailable",
            "The model is served on port {0}, but its access key is unreadable: {1}".format(service["port"], error),
            "Run as the endpoint's owner, or stop that endpoint before running the model locally.",
        ) from None


def _open(port, key, method, path, body=None, timeout=30):
    from .inference_auth import loopback_opener
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = None if body is None else json.dumps(body).encode()
    call = urllib.request.Request("http://127.0.0.1:{0}{1}".format(port, path), data=data, headers=headers,
                                  method=method)
    try:
        return loopback_opener().open(call, timeout=timeout)
    except urllib.error.HTTPError as error:
        try:
            detail = json.load(error)["error"]
            code, message = str(detail["code"]), str(detail["message"])
        except (ValueError, KeyError, TypeError):
            code, message = "endpoint_failed", "The model endpoint answered HTTP {0}.".format(error.code)
        raise InferenceError(code, message, REMEDIATION.get(code, "Inspect the endpoint with serve discover.")) from None
    except OSError as error:
        raise InferenceError("endpoint_unavailable", "The model endpoint on port {0} did not answer: {1}".format(port, error),
                             "Inspect the endpoint with serve discover.") from None


def _json(port, key, method, path, body=None):
    with _open(port, key, method, path, body) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise InferenceError("endpoint_failed", "The model endpoint returned an invalid answer.",
                             "Inspect the endpoint with serve discover.")
    return value


def _download(port, key, identifier, out, reference):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    digest, size = hashlib.sha256(), 0
    with _open(port, key, "GET", "/v1/tasks/{0}/artifacts/output".format(identifier), timeout=120) as response, \
            os.fdopen(os.open(out, flags, 0o644), "wb") as target:
        for chunk in iter(lambda: response.read(65536), b""):
            digest.update(chunk)
            size += len(chunk)
            target.write(chunk)
    if size != reference.get("bytes") or digest.hexdigest() != reference.get("sha256"):
        Path(out).unlink()
        raise InferenceError("artifact_changed", "The served output does not match its recorded digest.",
                             "Run the job again.")


def run(service, kind, parameters, out, timeout, clock=time.monotonic, sleep=time.sleep):
    """Submit one job to the endpoint, wait for it, and write its output artifact to out."""
    port = service["port"]
    key = _key(service)
    deadline = clock() + timeout
    record = _json(port, key, "POST", "/v1/tasks", {"kind": kind, "parameters": parameters})
    identifier = record["id"]
    while record["state"] not in TERMINAL:
        if clock() >= deadline:
            _json(port, key, "POST", "/v1/tasks/{0}/cancel".format(identifier), {})
            raise InferenceError("served_timeout", "The served job exceeded {0} s and was cancelled.".format(timeout),
                                 "Raise --timeout or lower the job size.")
        sleep(POLL_SECONDS)
        record = _json(port, key, "GET", "/v1/tasks/{0}".format(identifier))
    if record["state"] != "succeeded":
        error = record.get("error") if isinstance(record.get("error"), dict) else {}
        raise InferenceError(str(error.get("code", "task_" + record["state"])),
                             str(error.get("message", "The served job ended {0}.".format(record["state"]))),
                             "Inspect the job with the endpoint's task status.")
    outcome = _json(port, key, "GET", "/v1/tasks/{0}/result".format(identifier))
    result = outcome.get("result")
    if result is None:
        result = _json(port, key, "GET", "/v1/tasks/{0}/artifacts/result".format(identifier))
    if out is not None:
        _download(port, key, identifier, out, result["output_artifact"])
    value = {name: item for name, item in result.items() if name != "output_artifact"}
    if out is not None:
        value["path"] = str(out)
    value["served"] = {"port": port, "task_id": identifier}
    return value
