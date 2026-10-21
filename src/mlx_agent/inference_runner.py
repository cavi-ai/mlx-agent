"""Run one inference runner inside its backend's environment and read the result it prints."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Mapping, Tuple

from .backends import BackendError, backend_environment, load_manifests, sync_ports
from .wiring import redact_secrets

MAX_FAILURE_LINES = 6
MAX_FAILURE_CHARS = 800


class InferenceError(RuntimeError):
    """Classified inference failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


@dataclass(frozen=True)
class RunnerContract:
    """How one verb's runner reports a result, a refusal and a failure.

    Failures raise `error` with `<verb>_timeout` or `<verb>_failed`; `action` and refusal messages are
    formatted with the plan, and every runner detail is redacted before it reaches a message.
    """

    verb: str
    error: type
    accept: Callable[[dict, dict], bool]
    action: str
    remediation: str
    activity: str
    timeout_remediation: str
    refusals: Mapping[str, Callable[[dict, str], Tuple[str, str]]] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)


def redacted(text):
    """Runner text with credentials removed, cut to the last MAX_FAILURE_CHARS characters."""
    return redact_secrets(text)[-MAX_FAILURE_CHARS:]


def invoke(plan, contract, manifests=None, root=None, timeout=None, runner=subprocess.run):
    """Run plan["argv"] for plan["backend"]; return the last printed object the contract accepts.

    A printed refusal wins over the exit status; an accepted object counts only when the runner exits 0.
    """
    manifests = load_manifests() if manifests is None else manifests
    try:
        sync_ports(manifests[plan["backend"]], root)
    except BackendError as error:
        raise contract.error(error.code, str(error), error.remediation) from error
    environment = backend_environment()
    environment.update(contract.environment)
    try:
        completed = runner(plan["argv"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=timeout, env=environment, check=False)
    except subprocess.TimeoutExpired as error:
        raise contract.error(
            "{0}_timeout".format(contract.verb), "{0} exceeded {1} s.".format(contract.activity, timeout),
            contract.timeout_remediation,
        ) from error
    for line in reversed((completed.stdout or "").splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        refusal = contract.refusals.get(value.get("error"))
        if refusal is not None:
            message, remediation = refusal(plan, redacted(str(value.get("detail") or "no detail")))
            raise contract.error(value["error"], message, remediation)
        if contract.accept(value, plan):
            if completed.returncode == 0:
                return value
            break
    tail = "\n".join((completed.stderr or "").strip().splitlines()[-MAX_FAILURE_LINES:]) or "no output"
    raise contract.error(
        "{0}_failed".format(contract.verb),
        "The {0} backend could not {1}: {2}".format(plan["backend"], contract.action.format(**plan), redacted(tail)),
        contract.remediation,
    )
