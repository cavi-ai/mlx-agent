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
        "storage-status": "inspect the opt-in submission storage high-water mark",
        "storage-budget": "set the submission high-water mark in bytes, or zero to disable it",
        "expire-media": "preview or confirm expiration of one settled task's owned media",
        "media-provenance": "inspect preserved media references, including expired artifacts",
    }
    for command, description in descriptions.items():
        action = commands.add_parser(command, help=description)
        action.add_argument("--state-dir", type=Path, help="absolute task state directory; does not change host compute ownership")
        action.add_argument("--json", action="store_true", help="emit the versioned JSON result envelope")
        if command == "start":
            action.add_argument("--request", type=Path, required=True, help="task-request/1 or task-request/2 JSON file; accepted inputs are retained privately")
        elif command in ("status", "result", "cancel", "verify-model", "expire-media", "media-provenance"):
            action.add_argument("--id", required=True, help="task UUID returned by start")
            if command == "expire-media":
                action.add_argument("--confirm", help="confirmation value returned by the preview")
        elif command == "storage-budget":
            action.add_argument("--bytes", type=int, required=True, help="submission high-water mark; zero disables it")
        elif command == "list":
            action.add_argument("--limit", type=int, default=20, help="maximum recent records, 1..100 (default: 20)")
        elif command == "capabilities":
            action.add_argument("--model-path", type=Path, help="inspect optional model-specific settings without loading weights")


def summary(record):
    value = dict(record)
    value["identity"] = {key: record["identity"][key] for key in
                         ("backend", "backend_version", "engine_version", "python_version", "model_fingerprint")
                         if key in record["identity"]}
    for key in ("supervisor", "execution"):
        if value[key] is not None:
            value[key] = {"pid": value[key].get("pid")}
    if value["error"]:
        original = value["error"]
        value["error"] = {"code": str(original.get("code", "task_failed"))[:100],
                          "message": redact_secrets(str(original.get("message", "Task failed.")))[:1000],
                          "remediation": redact_secrets(str(original.get("remediation", "Inspect task status.")))[:500]}
        if original.get('code') == 'model_already_served' and 'data' in original:
            from .model_serving import public_error_data
            value['error']['data'] = public_error_data(original['data'])
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
        store = TaskStore(arguments.state_dir) if action != "capabilities" else None
        if action == "capabilities":
            data = capabilities(model_path=getattr(arguments, "model_path", None))
        elif action in ("storage-status", "storage-budget", "expire-media", "media-provenance"):
            from . import task_storage
            data = (task_storage.status(store) if action == "storage-status" else
                    task_storage.set_budget(store, arguments.bytes) if action == "storage-budget" else
                    task_storage.provenance(store, arguments.id) if action == "media-provenance" else
                    task_storage.expire(store, arguments.id, arguments.confirm) if arguments.confirm else
                    task_storage.preview(store, arguments.id))
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
            if action == "status":
                from .task_storage import expiry_status
                data["media_expiration"] = expiry_status(store, record["id"])
        envelope = ResultEnvelope.ok(operation, data)
        if action == "start" and data["task"]["state"] == "failed":
            error = data["task"]["error"]
            envelope = ResultEnvelope(operation, "error", data=data,
                                      error=ErrorDetail(error['code'], error['message'], error['remediation']))
        payload = _bounded(envelope)
    except TaskError as error:
        payload = ResultEnvelope.fail(operation, error.code, redact_secrets(str(error))[:2000], error.remediation[:1000]).to_dict()
    print(encoded(payload if arguments.json else payload.get("data") if payload["status"] == "ok" else payload).decode())
    return 0 if payload["status"] == "ok" else 2
