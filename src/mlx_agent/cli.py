"""Command-line entry points for the dependency-free MLX agent core."""

import argparse
import json
import os
import urllib.parse
from pathlib import Path

from .discovery import DiscoveryRequest, DiscoveryService
from .host import HostInventory
from .huggingface import HuggingFaceClient
from .models import ROLES, render_md, wire


def _fixture_http_get(path):
    payload = json.loads(Path(path).read_text())

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
        return DiscoveryService()
    payload = json.loads(Path(fixture).read_text())
    return DiscoveryService(host=HostInventory(**payload["host"]), huggingface=HuggingFaceClient(http_get=_fixture_http_get(fixture)))


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
    service = _discovery_service_from_environment()
    result = service.discover(DiscoveryRequest(limit=arguments.limit, role=arguments.role, new=arguments.new, fast=arguments.fast))
    value = result.to_dict()
    if legacy:
        report = value["data"] if result.status == "ok" else {"host": service.host.to_dict(), "error": value["error"]["message"], "roles": {}}
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
