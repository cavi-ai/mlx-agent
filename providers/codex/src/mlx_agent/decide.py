"""Answer typed questions about a state with a converted classification model through its backend (read-only)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .backends import (
    backend_python,
    is_installed,
    load_manifests,
    load_registries,
    lookup,
)
from .inference_runner import InferenceError, RunnerContract, invoke

DECIDE_SCHEMA = "decide/1"
DECIDE_RUNNER = Path(__file__).resolve().with_name("decide_runner.py")
DEFAULT_TIMEOUT_SECONDS = 300
MAX_REQUEST_BYTES = 1024 * 1024
QUESTION_TYPES = ("choice", "score", "noul")


class DecideError(InferenceError):
    """Classified decision failure safe to surface in a result envelope."""


def _invalid(message):
    return DecideError(
        "invalid_request", message,
        'Pass a JSON object {"state": ..., "questions": {"id": {"type": "choice|score|noul", "instructions": ...}}}.',
    )


def read_request(request_path):
    path = Path(request_path).expanduser()
    if not path.is_file():
        raise DecideError("request_not_found", "No request file at {0}.".format(path), "Pass a JSON request file.")
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise _invalid("The request is larger than {0} bytes.".format(MAX_REQUEST_BYTES))
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _invalid("The request is not readable JSON: {0}".format(error)) from error
    validate_request(request)
    return path


def validate_request(request):
    """Validate decision content without reading or running a model."""
    if not isinstance(request, dict) or "state" not in request:
        raise _invalid("The request needs a state.")
    if not isinstance(request["state"], (str, dict, list)):
        raise _invalid("The state is text or a JSON object.")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise _invalid("The request needs at least one question.")
    for name, question in questions.items():
        if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
            raise _invalid("Question {0!r} needs a type of {1}.".format(name, ", ".join(QUESTION_TYPES)))
        if "instructions" not in question:
            raise _invalid("Question {0!r} needs instructions.".format(name))
    return request


def plan_decide(model_path, request_path, manifests=None, registries=None, root=None):
    """Pick the classification backend for a converted model directory; pure apart from reading inputs."""
    model = Path(model_path).expanduser()
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DecideError(
            "model_unreadable", "No readable config.json in {0}: {1}".format(model, error),
            "Pass the converted model directory.",
        ) from error
    model_type = config.get("model_type") if isinstance(config, dict) else None
    request = read_request(request_path)
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    hits = sorted(
        (hit for hit in lookup(model_type, manifests, registries) if hit["category"] == "classification"),
        key=lambda hit: hit["backend"],
    )
    if not hits:
        raise DecideError(
            "not_classification_model", "No classification backend implements model_type {0!r}.".format(model_type),
            "Decisions apply to converted classification models such as Laya.",
        )
    manifest = manifests[hits[0]["backend"]]
    if not is_installed(manifest, root):
        raise DecideError(
            "backend_not_installed", "The {0} backend is not installed.".format(manifest["id"]),
            "Install it first: mlx-agent backend install {0}.".format(manifest["id"]),
        )
    argv = [
        str(backend_python(manifest, root)), str(DECIDE_RUNNER),
        "--module", hits[0]["module"], "--model", str(model), "--request", str(request),
    ]
    return {"backend": manifest["id"], "model": str(model), "request": str(request), "argv": argv}


CONTRACT = RunnerContract(
    verb="decide", error=DecideError, accept=lambda value, plan: isinstance(value.get("answers"), dict),
    action="answer the request",
    remediation="Check the request's questions and that the directory is a complete converted model.",
    activity="The decision", timeout_remediation="Use a shorter state or a larger --timeout.",
)


def run_decide(plan, manifests=None, root=None, timeout=DEFAULT_TIMEOUT_SECONDS, runner=subprocess.run):
    result = invoke(plan, CONTRACT, manifests, root, timeout, runner)
    return {
        "schema": DECIDE_SCHEMA, "backend": plan["backend"], "model": plan["model"], "request": plan["request"],
        "answers": result["answers"], "usage": result.get("usage"),
        "seconds": result.get("seconds"), "load_seconds": result.get("load_seconds"),
    }
