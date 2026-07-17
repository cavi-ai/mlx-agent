"""Command-line entry points for the dependency-free MLX agent core."""

import argparse
import json
import os
import sys
import urllib.parse
from pathlib import Path

from .contracts import ResultEnvelope
from .discovery import DiscoveryRequest, DiscoveryService
from .host import HostInventory
from .huggingface import HuggingFaceClient
from .models import ROLES, render_md, wire


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


def _discovery_service_from_environment():
    fixture = os.environ.get("MLX_AGENT_FIXTURE")
    if not fixture:
        return DiscoveryService(), None, None
    try:
        payload = json.loads(Path(fixture).read_text())
        _validate_fixture(payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return None, None, ResultEnvelope.fail(
            "discover", "invalid_fixture", "MLX_AGENT_FIXTURE is invalid: {0}".format(error),
            "Use a valid test fixture or unset MLX_AGENT_FIXTURE to run live discovery.",
        )
    service = DiscoveryService(host=HostInventory(**payload["host"]), huggingface=HuggingFaceClient(http_get=_fixture_http_get(payload)))
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
    service, fixture_warning, fixture_error = _discovery_service_from_environment()
    result = fixture_error or service.discover(DiscoveryRequest(limit=arguments.limit, role=arguments.role, new=arguments.new, fast=arguments.fast))
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


def legacy_scout_main(argv=None):
    parser = argparse.ArgumentParser(description="Discover MLX models on HuggingFace for this host.")
    _add_discovery_arguments(parser)
    return _run_discovery(parser.parse_args(argv), legacy=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="MLX agent command-line core.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    discover = subcommands.add_parser("discover", help="discover MLX models for this host")
    _add_discovery_arguments(discover)
    arguments = parser.parse_args(argv)
    return _run_discovery(arguments, legacy=False)
