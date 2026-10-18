"""Bounded CLI responses for the engine-owned inference task service."""

from pathlib import Path

from .contracts import ErrorDetail, ResultEnvelope
from .task_contracts import TaskError, capabilities, encoded
from .task_runtime import cancel_task, start_task, task_result, task_status, verify_task_model
from .task_store import TaskStore
from .wiring import redact_secrets

MAX_RESPONSE_BYTES = 12288


def add_arguments(parser):
    commands = parser.add_subparsers(dest="task_command", required=True)
    descriptions = {
        "capabilities": "describe supported local inference tasks",
        "start": "start a durable task from a local JSON request",
        "status": "inspect a task without starting inference",
        "list": "list recent local tasks",
        "result": "retrieve a successful result or its owned artifact reference",
        "cancel": "request cancellation of one task",
        "verify-model": "check the task's recorded model and runtime identity without running it",
    }
    for command, description in descriptions.items():
        action = commands.add_parser(command, help=description)
        action.add_argument("--state-dir", type=Path, help="absolute task state directory; does not change host compute ownership")
        action.add_argument("--json", action="store_true", help="emit the versioned JSON result envelope")
        if command == "start":
            action.add_argument("--request", type=Path, required=True, help="task-request/1 or task-request/2 JSON file; accepted inputs are retained privately")
        elif command in ("status", "result", "cancel", "verify-model"):
            action.add_argument("--id", required=True, help="task UUID returned by start")
        elif command == "list":
            action.add_argument("--limit", type=int, default=20, help="maximum recent records, 1..100 (default: 20)")


def summary(record):
    value = dict(record)
    value["identity"] = {key: record["identity"][key] for key in
                         ("backend", "backend_version", "engine_version", "python_version", "model_fingerprint")
                         if key in record["identity"]}
    for key in ("supervisor", "execution"):
        if value[key] is not None:
            value[key] = {"pid": value[key].get("pid")}
    if value["error"]:
        value["error"] = {"code": str(value["error"].get("code", "task_failed"))[:100],
                          "message": redact_secrets(str(value["error"].get("message", "Task failed.")))[:1000],
                          "remediation": redact_secrets(str(value["error"].get("remediation", "Inspect task status.")))[:500]}
    return value


def _bounded(envelope):
    payload = envelope.to_dict()
    while len(encoded(payload)) > MAX_RESPONSE_BYTES and payload["data"].get("tasks"):
        payload["data"]["tasks"].pop()
        payload["data"]["has_more"] = True
    if len(encoded(payload)) > MAX_RESPONSE_BYTES and payload["data"].get("inline"):
        payload["data"]["inline"] = False
        payload["data"]["result"] = None
    if len(encoded(payload)) > MAX_RESPONSE_BYTES:
        raise TaskError("task_response_too_large", "Task metadata exceeds the response size limit.", "Inspect the private task store and preserve its records.")
    return payload


def run(arguments):
    action = arguments.task_command
    operation = "task." + action
    try:
        store = TaskStore(arguments.state_dir)
        if action == "capabilities":
            data = capabilities()
        elif action == "verify-model":
            data = verify_task_model(arguments.id, store.root)
        elif action == "list":
            data = store.list_page(arguments.limit)
            data["tasks"] = [summary(record) for record in data["tasks"]]
        elif action == "result":
            data = task_result(arguments.id, store.root)
            data["task"] = summary(data["task"])
        else:
            record = (start_task(arguments.request, store.root) if action == "start" else
                      cancel_task(arguments.id, store.root) if action == "cancel" else task_status(arguments.id, store.root))
            data = {"task": summary(record)}
        envelope = ResultEnvelope.ok(operation, data)
        if action == "start" and data["task"]["state"] == "failed":
            error = data["task"]["error"]
            envelope = ResultEnvelope(operation, "error", data=data, error=ErrorDetail(**error))
        payload = _bounded(envelope)
    except TaskError as error:
        payload = ResultEnvelope.fail(operation, error.code, redact_secrets(str(error))[:2000], error.remediation[:1000]).to_dict()
    print(encoded(payload if arguments.json else payload.get("data") if payload["status"] == "ok" else payload).decode())
    return 0 if payload["status"] == "ok" else 2
