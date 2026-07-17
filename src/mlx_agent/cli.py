"""Command-line entry points for the dependency-free MLX agent core."""

import argparse
import json
import os
import sys
import urllib.parse
from pathlib import Path

from .adoption import AdoptionRequest, AdoptionWorkflow
from .contracts import ErrorDetail, ResultEnvelope
from .discovery import DiscoveryRequest, DiscoveryService
from .host import HostInventory
from .huggingface import HuggingFaceClient
from .models import ROLES, render_md, wire
from .transactions import COOPERATIVE_CONCURRENCY_NOTE, ConcurrentTransactionError, Receipt, Transaction, _assert_safe_target, _read_regular, rollback
from .verification import Verifier
from .wiring import ConfigAdapter


FIXTURE_WARNING = {
    "code": "synthetic_fixture",
    "message": "Fixture-backed discovery; this is not live Hugging Face evidence.",
}


def _fixture_http_get(payload):

    def get(url, timeout=10.0):
        del timeout
        if "/tree/main" in url:
            repo = urllib.parse.unquote(url.split("/api/models/", 1)[1].split("/tree/main", 1)[0])
            return payload["trees"].get(repo, [])
        if "/api/models/" in url:
            repo = urllib.parse.unquote(url.split("/api/models/", 1)[1])
            return payload["details"].get(repo, {})
        if url.startswith("http://127.0.0.1") or url.startswith("http://localhost"):
            raise OSError("fixture does not emulate local runtime endpoints")
        return payload["models"]

    return get


def _discovery_service_from_environment(state_dir=None):
    fixture = os.environ.get("MLX_AGENT_FIXTURE")
    if not fixture:
        return DiscoveryService(state_dir=state_dir), None, None
    try:
        payload = json.loads(Path(fixture).read_text())
        _validate_fixture(payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return None, None, ResultEnvelope.fail(
            "discover", "invalid_fixture", "MLX_AGENT_FIXTURE is invalid: {0}".format(error),
            "Use a valid test fixture or unset MLX_AGENT_FIXTURE to run live discovery.",
        )
    service = DiscoveryService(
        host=HostInventory(**payload["host"]),
        huggingface=HuggingFaceClient(http_get=_fixture_http_get(payload)),
        state_dir=state_dir,
        cache_enabled=False,
    )
    return service, FIXTURE_WARNING, None


def _validate_fixture(payload):
    if not isinstance(payload, dict):
        raise ValueError("fixture root must be an object")
    if not isinstance(payload.get("models"), list):
        raise ValueError("fixture.models must be a list")
    for index, model in enumerate(payload["models"]):
        if not isinstance(model, dict):
            raise ValueError("fixture.models[{0}] must be an object".format(index))
        identifiers = [model[key] for key in ("id", "modelId") if key in model]
        if not identifiers:
            raise ValueError("fixture.models[{0}] must contain id or modelId".format(index))
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ValueError("fixture.models[{0}].id and modelId must be non-empty strings".format(index))
        for counter in ("downloads", "likes"):
            if counter in model and (not isinstance(model[counter], int) or isinstance(model[counter], bool)):
                raise ValueError("fixture.models[{0}].{1} must be an integer".format(index, counter))
    if not isinstance(payload.get("details"), dict):
        raise ValueError("fixture.details must be an object")
    if not isinstance(payload.get("trees"), dict):
        raise ValueError("fixture.trees must be an object")
    host = payload.get("host")
    if not isinstance(host, dict):
        raise ValueError("fixture.host must be an object")
    if set(host) != {"ram_gb", "chip", "ollama", "lmstudio"}:
        raise ValueError("fixture.host must contain only ram_gb, chip, ollama, and lmstudio")
    if host["ram_gb"] is not None and (not isinstance(host["ram_gb"], int) or isinstance(host["ram_gb"], bool)):
        raise ValueError("fixture.host.ram_gb must be an integer or null")
    if host["chip"] is not None and not isinstance(host["chip"], str):
        raise ValueError("fixture.host.chip must be a string or null")
    if not isinstance(host["ollama"], bool) or not isinstance(host["lmstudio"], bool):
        raise ValueError("fixture.host runtime flags must be booleans")


def _add_discovery_arguments(parser):
    parser.add_argument("--role", choices=[role for role, _keywords, _label in ROLES])
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--memory-gb", type=float, help="maximum host memory budget in GB (keeps 20%% runtime headroom)")
    parser.add_argument("--quantization", help="normalized quantization such as 4bit or q8")
    parser.add_argument("--license", dest="licenses", action="append", help="allow only this license (repeatable)")
    parser.add_argument("--publisher", dest="publishers", action="append", help="allow only this publisher (repeatable)")
    parser.add_argument("--runtime", choices=["ollama", "lmstudio", "mlx_lm", "mlx-vlm", "litellm"], help="require a runtime compatible with the model role")
    parser.add_argument("--exclude-gated", dest="include_gated", action="store_false", default=True, help="exclude gated repositories")
    parser.add_argument("--include-gated", dest="include_gated", action="store_true", help="include gated repositories (the legacy default)")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--refresh", action="store_true", help="bypass a fresh cache and fetch live evidence")
    cache_group.add_argument("--offline", action="store_true", help="use only a matching local cache entry")
    parser.add_argument("--state-dir", help="directory for versioned discovery cache entries")
    parser.add_argument("--new", action="store_true", help="sort by most-recently-updated")
    parser.add_argument("--fast", action="store_true", help="skip per-model enrichment (name heuristics only)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--wire", metavar="REPO", help="emit setup + config for a model, instead of discovering")
    parser.add_argument("--target", choices=["ollama", "lmstudio", "mlx_lm", "mlx-vlm", "litellm"], default="mlx_lm")
    parser.add_argument("--port", type=int, default=8080)


def _run_discovery(arguments, legacy):
    if arguments.wire:
        print(wire(arguments.wire, arguments.target, arguments.port))
        return 0
    service, fixture_warning, fixture_error = _discovery_service_from_environment(arguments.state_dir)
    result = fixture_error or service.discover(DiscoveryRequest(
        role=arguments.role,
        memory_gb=arguments.memory_gb,
        quantization=arguments.quantization,
        licenses=arguments.licenses,
        include_gated=arguments.include_gated,
        publishers=arguments.publishers,
        runtime=arguments.runtime,
        refresh=arguments.refresh,
        offline=arguments.offline,
        limit=arguments.limit,
        new=arguments.new,
        fast=arguments.fast,
    ))
    if fixture_warning:
        result = ResultEnvelope.ok("discover", result.data, warnings=[fixture_warning])
        print("warning: synthetic fixture-backed discovery; not live Hugging Face evidence.", file=sys.stderr)
    value = result.to_dict()
    if legacy:
        report = value["data"] if result.status == "ok" else {"host": service.host.to_dict() if service else HostInventory().to_dict(), "error": value["error"]["message"], "roles": {}}
        print(json.dumps(report, indent=2) if arguments.json else render_md(report))
        return 0 if result.status == "ok" else 2
    if arguments.json:
        print(json.dumps(value, indent=2))
    elif result.status == "ok":
        print(render_md(value["data"]))
    else:
        error = value["error"]
        print("discover failed [{0}]: {1}\nremediation: {2}".format(error["code"], error["message"], error["remediation"]))
    return 0 if result.status == "ok" else 2


def _adoption_state_path(arguments):
    return arguments.state or os.environ.get("MLX_AGENT_ADOPTION_STATE")


def _emit_adoption_result(result, as_json):
    value = result.to_dict()
    if as_json:
        print(json.dumps(value, indent=2))
    elif result.status == "ok":
        state = result.data["state"]
        print("Adoption {0}: {1}".format(state["status"], state["phase"]))
        if state["recommendations"]:
            for item in state["recommendations"]:
                print("{0}: {1} [{2}]".format(item["role"], item["repo"], item["evidence_strength"]))
    else:
        error = value["error"]
        print("{0} failed [{1}]: {2}\nremediation: {3}".format(
            result.operation, error["code"], error["message"], error["remediation"]
        ))
    return 0 if result.status == "ok" else 2


def _run_adoption(arguments):
    operation = "adopt-{0}".format(arguments.adopt_command)
    state_path = _adoption_state_path(arguments)
    if not state_path:
        return _emit_adoption_result(ResultEnvelope.fail(
            operation,
            "state_path_required",
            "No adoption state path was supplied.",
            "Pass --state PATH or set MLX_AGENT_ADOPTION_STATE.",
        ), arguments.json)

    if arguments.adopt_command == "status":
        workflow = AdoptionWorkflow(
            discovery_service=DiscoveryService(), verifier=Verifier(), state_path=state_path
        )
        try:
            state = workflow.status(state_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            return _emit_adoption_result(ResultEnvelope.fail(
                operation,
                "adoption_state_invalid",
                "Adoption state could not be read: {0}".format(error),
                "Check --state PATH and restore a schema-version 1.0 adoption handoff.",
            ), arguments.json)
        return _emit_adoption_result(
            ResultEnvelope.ok(operation, {"state": state.to_dict()}), arguments.json
        )

    service, fixture_warning, fixture_error = _discovery_service_from_environment()
    if fixture_error:
        error = fixture_error.to_dict()["error"]
        return _emit_adoption_result(ResultEnvelope.fail(
            operation,
            error["code"],
            error["message"],
            error["remediation"],
            retryable=error["retryable"],
        ), arguments.json)
    workflow = AdoptionWorkflow(
        discovery_service=service,
        verifier=Verifier(metadata_client=getattr(service, "_huggingface", None)),
        state_path=state_path,
    )
    try:
        if arguments.adopt_command == "start":
            state = workflow.start(AdoptionRequest(
                roles=tuple(arguments.roles or ("general",)),
                state_path=state_path,
                shortlist_limit=arguments.shortlist_limit,
                allow_network=arguments.allow_network and not arguments.offline,
                offline=arguments.offline,
                refresh=arguments.refresh,
                fast=arguments.fast,
            ))
        else:
            state = workflow.resume(state_path)
        while state.phase != "complete":
            state = workflow.advance(state)
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        return _emit_adoption_result(ResultEnvelope.fail(
            operation,
            "adoption_failed",
            "Adoption workflow could not continue: {0}".format(error),
            "Inspect the saved state with 'adopt status', resolve the reported issue, and resume.",
        ), arguments.json)
    warnings = [fixture_warning] if fixture_warning else []
    return _emit_adoption_result(
        ResultEnvelope.ok(operation, {"state": state.to_dict()}, warnings=warnings),
        arguments.json,
    )


def _add_adoption_arguments(parser):
    actions = parser.add_subparsers(dest="adopt_command", required=True)
    start = actions.add_parser("start", help="start and durably run a model adoption workflow")
    start.add_argument("--state", help="adoption handoff path")
    start.add_argument("--role", dest="roles", action="append", choices=[role for role, _keywords, _label in ROLES])
    start.add_argument("--shortlist-limit", type=int, default=4)
    start.add_argument("--offline", action="store_true", help="use cached discovery and no metadata network requests")
    start.add_argument("--refresh", action="store_true", help="refresh model discovery")
    start.add_argument("--fast", action="store_true", help="use heuristic-only discovery enrichment")
    start.add_argument("--no-network", dest="allow_network", action="store_false", default=True, help="do not inspect missing-model metadata")
    start.add_argument("--json", action="store_true")
    for name in ("resume", "status"):
        action = actions.add_parser(name, help="{0} an adoption handoff".format(name))
        action.add_argument("--state", help="adoption handoff path")
        action.add_argument("--json", action="store_true")


def _emit_wire_result(result, as_json, human=None):
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
    elif human is not None:
        print(human)
    elif result.status == "ok":
        print(json.dumps(result.data, indent=2))
    else:
        error = result.to_dict()["error"]
        print("{0} failed [{1}]: {2}\nremediation: {3}".format(
            result.operation, error["code"], error["message"], error["remediation"]
        ))
    return 0 if result.status == "ok" else 2


def _receipt_data(receipt):
    value = receipt.to_dict()
    value["receipt_path"] = receipt.receipt_path
    return value


def _wire_failure(operation, code, message, remediation, data=None):
    return ResultEnvelope(
        operation=operation, status="error", data=data or {},
        warnings=[{"code": "cooperative_concurrency", "message": COOPERATIVE_CONCURRENCY_NOTE}],
        error=ErrorDetail(code, message, remediation),
    )


def _wire_ok(operation, data):
    return ResultEnvelope.ok(operation, data, warnings=[{
        "code": "cooperative_concurrency", "message": COOPERATIVE_CONCURRENCY_NOTE,
    }])


def _wire_render(arguments):
    path = _assert_safe_target(arguments.path)
    existing = _read_regular(path).decode("utf-8")
    adapter = ConfigAdapter.detect(path, runtime=arguments.target)
    content = adapter.render(arguments.model, arguments.target, existing)
    adapter.validate(content)
    return path, adapter, content


def _run_wire(arguments):
    operation = "wire-{0}".format(arguments.wire_command)
    try:
        if arguments.wire_command == "status":
            location = _assert_safe_target(arguments.receipt)
            receipt = Receipt.from_dict(json.loads(_read_regular(location).decode("utf-8")), str(location))
            return _emit_wire_result(_wire_ok(operation, {"receipt": _receipt_data(receipt)}), arguments.json)
        if arguments.wire_command == "rollback":
            if not arguments.confirm:
                return _emit_wire_result(_wire_failure(
                    operation, "confirmation_required", "Rollback was not started without --confirm.",
                    "Review the receipt with 'wire status RECEIPT', then run 'wire rollback RECEIPT --confirm'.",
                ), arguments.json)
            receipt = rollback(arguments.receipt)
            result = _wire_ok(operation, {"receipt": _receipt_data(receipt)}) if receipt.status == "rolled_back" else _wire_failure(
                operation, receipt.status, "Rollback did not complete; receipt status is {0}.".format(receipt.status),
                "Inspect the receipt validations and restore the verified backup manually.", {"receipt": _receipt_data(receipt)},
            )
            return _emit_wire_result(result, arguments.json)

        path, adapter, content = _wire_render(arguments)
        if arguments.wire_command == "render":
            return _emit_wire_result(_wire_ok(operation, {
                "path": str(path), "runtime": arguments.target, "config": content,
                "validation": {"parse": True},
            }), arguments.json, human=content)
        transaction = Transaction(receipts_dir=arguments.receipts_dir)
        preview = transaction.preview([{
            "path": str(path), "content": content, "runtime": arguments.target,
            "adapter": adapter, "endpoint": arguments.endpoint,
        }])
        if not arguments.confirm:
            result = _wire_ok(operation, {"preview": preview, "requires_confirmation": True})
            if arguments.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(preview["diff"])
                print("Confirmation required: rerun with --confirm to apply this transaction.")
            return 2
        if not arguments.preview_hash:
            return _emit_wire_result(_wire_failure(
                operation, "preview_hash_required", "--confirm requires the preview hash from a prior preview.",
                "Run wire apply without --confirm, inspect the preview, then pass its --preview-hash value.",
                {"preview": preview},
            ), arguments.json)
        if arguments.preview_hash != preview["preview_hash"]:
            return _emit_wire_result(_wire_failure(
                operation, "preview_stale", "The supplied preview hash does not match the current preview.",
                "Generate and inspect a new preview before confirming this mutation.",
                {"preview": preview},
            ), arguments.json)
        if not arguments.json:
            print(preview["diff"])
        receipt = transaction.apply(arguments.preview_hash)
        data = {"preview": preview, "receipt": _receipt_data(receipt)}
        result = _wire_ok(operation, data) if receipt.status == "applied" else _wire_failure(
            operation, receipt.status, "Wire did not apply; receipt status is {0}.".format(receipt.status),
            "Inspect the receipt validation results and use the recorded recovery path before retrying.", data,
        )
        return _emit_wire_result(result, arguments.json)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        code = "cooperative_lock_busy" if isinstance(error, ConcurrentTransactionError) else ("preview_stale" if str(error).startswith("preview is stale") else "wire_failed")
        return _emit_wire_result(_wire_failure(
            operation, code, "Wire could not complete: {0}".format(error),
            "Correct the target configuration or receipt, then render a new preview.",
        ), arguments.json)


def _add_wire_arguments(parser):
    actions = parser.add_subparsers(dest="wire_command", required=True)
    for name in ("render", "apply"):
        action = actions.add_parser(name, help="{0} a deterministic runtime configuration ({1})".format(name, "advisory lock protects cooperative writers" if name == "apply" else "safe render"))
        action.add_argument("model", help="Hugging Face model repository")
        action.add_argument("--target", choices=["ollama", "lmstudio", "mlx_lm", "mlx-vlm", "litellm"], default="mlx_lm")
        action.add_argument("--path", required=True, help="target configuration file")
        action.add_argument("--json", action="store_true")
        if name == "apply":
            action.add_argument("--confirm", action="store_true", help="explicitly authorize this reviewed mutation")
            action.add_argument("--preview-hash", help="hash returned by the separately reviewed wire apply preview")
            action.add_argument("--receipts-dir", help="directory for non-secret transaction receipts")
            action.add_argument("--endpoint", help="optional local runtime health endpoint")
    status = actions.add_parser("status", help="inspect a Wire receipt")
    status.add_argument("receipt")
    status.add_argument("--json", action="store_true")
    restore = actions.add_parser("rollback", help="restore a Wire receipt's exact backup")
    restore.add_argument("receipt")
    restore.add_argument("--confirm", action="store_true", help="explicitly authorize this rollback")
    restore.add_argument("--json", action="store_true")


def legacy_scout_main(argv=None):
    parser = argparse.ArgumentParser(description="Discover MLX models on HuggingFace for this host.")
    _add_discovery_arguments(parser)
    return _run_discovery(parser.parse_args(argv), legacy=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="MLX agent command-line core.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    discover = subcommands.add_parser("discover", help="discover MLX models for this host")
    _add_discovery_arguments(discover)
    adopt = subcommands.add_parser("adopt", help="run or inspect resumable model adoption")
    _add_adoption_arguments(adopt)
    wire_command = subcommands.add_parser("wire", help="render, apply, inspect, or roll back runtime wiring")
    _add_wire_arguments(wire_command)
    arguments = parser.parse_args(argv)
    if arguments.command == "adopt":
        return _run_adoption(arguments)
    if arguments.command == "wire":
        return _run_wire(arguments)
    return _run_discovery(arguments, legacy=False)
