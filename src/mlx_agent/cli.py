"""Command-line entry points for the dependency-free MLX agent core."""

import argparse
import json
import os
import sys
import urllib.parse
from pathlib import Path

from .adoption import AdoptionRequest, AdoptionWorkflow
from .contracts import ResultEnvelope
from .discovery import DiscoveryRequest, DiscoveryService
from .host import HostInventory
from .huggingface import HuggingFaceClient
from .models import ROLES, render_md, wire
from .verification import Verifier


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
        return _emit_adoption_result(fixture_error, arguments.json)
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
    arguments = parser.parse_args(argv)
    if arguments.command == "adopt":
        return _run_adoption(arguments)
    return _run_discovery(arguments, legacy=False)
