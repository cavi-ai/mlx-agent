"""Command-line entry points for the dependency-free MLX agent core."""

import argparse
import json
import os
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from .adoption import ADOPTION_SCHEMA_VERSION, AdoptionRequest, AdoptionWorkflow
from .bench import (
    BENCH_PROBE_ID,
    GEN_TOKENS_DEFAULT,
    RUNS_DEFAULT,
    TIMEOUT_DEFAULT,
    BenchError,
    measure_runtime,
)
from .contracts import ErrorDetail, ResultEnvelope
from .convert import (
    Q_BITS_CHOICES,
    ConvertError,
    plan_convert,
    start_convert,
    status_convert,
)
from .discovery import DiscoveryRequest, DiscoveryService
from .fleet import (
    FleetConfigAdapter,
    FleetError,
    assignments_from_adoption,
    parse_assignments,
    parse_runtime_map,
)
from .fuse import (
    FuseError,
    plan_fuse,
    start_fuse,
    status_fuse,
)
from .host import HostInventory
from .huggingface import HuggingFaceClient
from .installer import Installer, InstallerConflictError
from .interview import build_intent, run_interview
from .lora import (
    BATCH_DEFAULT,
    ITERS_DEFAULT,
    LAYERS_DEFAULT,
    LR_DEFAULT,
    LoraError,
    plan_lora,
    start_lora,
    status_lora,
)
from .modality import ALL_FACET_IDS, FOUNDATION_IDS, resolve_facets, resolve_modalities
from .models import DISCOVERY_ROLES, render_md, wire
from .project_blueprint import (
    build_brief,
    generate_design_pack,
    render_design_pack,
    write_design_pack,
)
from .providers import ProviderRegistry
from .research import generate_pack, render_pack, write_pack
from .serve import (
    MAX_TOKENS_DEFAULT,
    ServeError,
    default_model_present,
    load_recipes,
    plan_start,
    receipts_root,
    start_serve,
    status_serve,
    stop_serve,
    wired_port_claim,
)
from .transactions import (
    COOPERATIVE_CONCURRENCY_NOTE,
    ConcurrentTransactionError,
    Receipt,
    Transaction,
    _assert_safe_target,
    _read_regular,
    preview_rollback,
    rollback,
)
from .verification import (
    LMStudioRuntimeClient,
    MLXVLMRuntimeClient,
    OllamaRuntimeClient,
    OpenAICompatibleRuntimeClient,
    Verifier,
)
from .watch import (
    WatchError,
    build_snapshot,
    collect_owned,
    diff_snapshots,
    read_baseline,
    snapshot_candidates,
    write_snapshot,
)
from .wiring import ConfigAdapter


FIXTURE_WARNING = {
    "code": "synthetic_fixture",
    "message": "Fixture-backed discovery; this is not live Hugging Face evidence.",
}


def _fixture_http_get(payload):

    def get(url, timeout=10.0):
        del timeout
        if "/api/datasets" in url:
            return payload.get("datasets", [])
        if "/tree/main" in url:
            repo = urllib.parse.unquote(url.split("/api/models/", 1)[1].split("/tree/main", 1)[0])
            return payload["trees"].get(repo, [])
        if "/api/models/" in url:
            repo = urllib.parse.unquote(url.split("/api/models/", 1)[1])
            return payload["details"].get(repo, {})
        if url.startswith("http://127.0.0.1") or url.startswith("http://localhost"):
            raise OSError("fixture does not emulate local runtime endpoints")
        query = urllib.parse.urlsplit(url).query
        params = urllib.parse.parse_qs(query)
        filters = params.get("filter", [])
        if "peft" in filters:
            return payload.get("adapters", [])
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
        huggingface=HuggingFaceClient(
            http_get=_fixture_http_get(payload),
            card_get=lambda url, timeout=8: None,
        ),
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
    parser.add_argument("--role", choices=DISCOVERY_ROLES)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--memory-gb", type=float, help="maximum host memory budget in GB (keeps 20%% runtime headroom)")
    parser.add_argument("--quantization", help="normalized quantization such as 4bit or q8")
    parser.add_argument("--license", dest="licenses", action="append", help="allow only this license (repeatable)")
    parser.add_argument("--publisher", dest="publishers", action="append", help="allow only this publisher (repeatable)")
    parser.add_argument("--runtime", choices=["ollama", "lmstudio", "mlx_lm", "mlx-vlm", "litellm"], help="require a runtime compatible with the model role")
    parser.add_argument("--context", dest="context_tokens", type=int, default=None, help="tighten fit checks to weights plus KV cache at this context length")
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
    if arguments.context_tokens is not None and not 1024 <= arguments.context_tokens <= 1048576:
        result = ResultEnvelope.fail(
            "discover",
            "invalid_arguments",
            "--context must be between 1024 and 1048576 tokens.",
            "Pass a bounded context length, or omit --context for weights-only fit.",
        )
        if arguments.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            error = result.to_dict()["error"]
            print("discover failed [{0}]: {1}\nremediation: {2}".format(error["code"], error["message"], error["remediation"]))
        return 2
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
        context_tokens=arguments.context_tokens,
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


def _run_inspect_host(arguments):
    host, warnings = HostInventory.inspect(HuggingFaceClient()._http_get)
    result = ResultEnvelope.ok("inspect-host", host.to_dict(), warnings=warnings)
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("Host inventory: {0}".format(json.dumps(host.to_dict(), sort_keys=True)))
        for warning in warnings:
            print("warning [{0}/{1}]: {2}".format(warning["code"], warning["probe"], warning["message"]))
    return 0


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
        requested_roles = state.get("request", {}).get("roles", [])
        recommended_roles = {
            item.get("role") for item in state.get("recommendations", [])
        }
        if (
            state.get("status") == "complete"
            and "tool-use" in requested_roles
            and "tool-use" not in recommended_roles
        ):
            print("No verified tool-use model was found.")
            print(
                "No model was downloaded. Install a shortlisted candidate in a "
                "supported local runtime and start adoption again."
            )
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
                "Check --state PATH and restore a schema-version {0} adoption handoff.".format(
                    ADOPTION_SCHEMA_VERSION
                ),
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
    start.add_argument("--role", dest="roles", action="append", choices=DISCOVERY_ROLES)
    start.add_argument("--shortlist-limit", type=int, default=4)
    start.add_argument("--offline", action="store_true", help="use cached discovery and no metadata network requests")
    start.add_argument("--refresh", action="store_true", help="refresh model discovery")
    start.add_argument("--fast", action="store_true", help="use heuristic-only discovery enrichment")
    start.add_argument("--no-network", dest="allow_network", action="store_false", default=True, help="do not inspect missing-model metadata")
    start.add_argument("--measure", action="store_true", help="measure verified shortlist candidates (bench) before ranking")
    start.add_argument("--json", action="store_true")
    for name in ("resume", "status"):
        action = actions.add_parser(name, help="{0} an adoption handoff".format(name))
        action.add_argument("--state", help="adoption handoff path")
        action.add_argument("--json", action="store_true")


def _add_research_arguments(parser):
    parser.add_argument("--domain", help="one-line domain description (required unless --interview)")
    parser.add_argument("--role", dest="roles", action="append", choices=DISCOVERY_ROLES, help="model role to research (repeatable)")
    parser.add_argument("--keyword", dest="keywords", action="append", help="domain keyword to prioritize (repeatable)")
    parser.add_argument(
        "--modality",
        dest="modalities",
        action="append",
        choices=list(FOUNDATION_IDS),
        help="foundational modality to seed (repeatable): audio, video, document-vision",
    )
    parser.add_argument(
        "--facet",
        dest="facets",
        action="append",
        choices=list(ALL_FACET_IDS),
        help="modality facet to prioritize (repeatable)",
    )
    parser.add_argument("--license", dest="licenses", action="append", help="allow only this license (repeatable)")
    parser.add_argument("--memory-gb", type=float, help="host memory budget in GB")
    parser.add_argument("--notes", default="", help="free-form constraints")
    parser.add_argument("--project", default=str(Path.cwd()), help="project root; the pack is written under <project>/mlx-research")
    parser.add_argument("--limit", type=int, default=6, help="candidates fetched per role during discovery; the pack is also capped at this many total")
    parser.add_argument("--interview", action="store_true", help="ask questions interactively on stdin")
    parser.add_argument("--no-write", dest="write", action="store_false", default=True, help="render the pack without writing a file")
    parser.add_argument("--json", action="store_true")


def _stdin_reader(question):
    prompt = question["prompt"]
    if question.get("kind") == "multi":
        prompt += " [{0}]".format(", ".join(question.get("choices", ())))
    return input("{0}\n> ".format(prompt))


def _emit_research(result, arguments, markdown=None):
    value = result.to_dict()
    if arguments.json:
        print(json.dumps(value, indent=2))
    elif result.status == "ok":
        if "path" in result.data:
            print("Research pack written to {0}".format(result.data["path"]))
        else:
            print(markdown if markdown is not None else "Research pack generated.")
    else:
        error = value["error"]
        print("research failed [{0}]: {1}\nremediation: {2}".format(
            error["code"], error["message"], error["remediation"]
        ))
    return 0 if result.status == "ok" else 2


def _run_research(arguments):
    operation = "research"
    try:
        cli_modalities = tuple(arguments.modalities or ())
        cli_facets = tuple(arguments.facets or ())
        if arguments.interview:
            intent = run_interview(
                _stdin_reader,
                preset_modalities=cli_modalities,
                preset_facets=cli_facets,
            )
        elif not arguments.domain:
            return _emit_research(ResultEnvelope.fail(
                operation, "domain_required",
                "No domain description was supplied.",
                "Pass --domain \"...\" or use --interview.",
            ), arguments)
        else:
            detect_text = " ".join(
                [arguments.domain] + list(arguments.keywords or [])
            )
            modalities = resolve_modalities(cli=cli_modalities, text=detect_text)
            if not modalities:
                return _emit_research(ResultEnvelope.fail(
                    operation, "modality_required",
                    "No foundational modality was supplied or detected from the domain text.",
                    "Pass --modality audio|video|document-vision (repeatable), "
                    "or use --interview to choose interactively.",
                ), arguments)
            facets = resolve_facets(modalities, cli=cli_facets, text=detect_text)
            intent = build_intent({
                "domain": arguments.domain,
                "roles": arguments.roles or [],
                "keywords": ",".join(arguments.keywords or []),
                "license": ",".join(arguments.licenses or []),
                "memory_gb": arguments.memory_gb,
                "notes": arguments.notes,
                "modalities": list(modalities),
                "facets": list(facets),
            })
    except (ValueError, EOFError) as error:
        return _emit_research(ResultEnvelope.fail(
            operation, "invalid_intent", str(error),
            "Correct the interview answers or flags and retry.",
        ), arguments)

    service, fixture_warning, fixture_error = _discovery_service_from_environment()
    if fixture_error:
        error = fixture_error.to_dict()["error"]
        return _emit_research(ResultEnvelope.fail(
            operation, error["code"], error["message"], error["remediation"],
        ), arguments)
    hf_client = getattr(service, "_huggingface", None) or HuggingFaceClient()
    moment = datetime.now(timezone.utc)
    try:
        pack = generate_pack(intent, service, hf_client, limit=arguments.limit, now=moment)
    except (OSError, TypeError, ValueError) as error:
        return _emit_research(ResultEnvelope.fail(
            operation, "research_failed",
            "Research could not complete: {0}".format(error),
            "Check network access to huggingface.co, or unset MLX_AGENT_FIXTURE.",
        ), arguments)
    markdown = render_pack(pack)
    data = {"pack": pack.to_dict()}
    if arguments.write:
        try:
            data["path"] = str(write_pack(
                markdown, intent, root=arguments.project, now=moment, pack=pack,
            ))
        except (OSError, ValueError) as error:
            return _emit_research(ResultEnvelope.fail(
                operation, "write_failed",
                "Research pack could not be written: {0}".format(error),
                "Choose a writable --project directory without a symlinked mlx-research folder.",
            ), arguments)
    warnings = [fixture_warning] if fixture_warning else []
    return _emit_research(
        ResultEnvelope.ok(operation, data, warnings=warnings), arguments, markdown
    )


def _add_blueprint_arguments(parser):
    parser.add_argument("--goal", help="one-line project goal (required)")
    parser.add_argument(
        "--modality",
        dest="modalities",
        action="append",
        choices=list(FOUNDATION_IDS),
        help="foundational modality to include (repeatable)",
    )
    parser.add_argument("--memory-gb", type=float, help="host memory budget in GB")
    parser.add_argument("--notes", default="", help="free-form constraints")
    parser.add_argument(
        "--project",
        default=str(Path.cwd()),
        help="project root; the pack is written under <project>/mlx-blueprints",
    )
    parser.add_argument(
        "--no-write",
        dest="write",
        action="store_false",
        default=True,
        help="render the pack without writing a file",
    )
    parser.add_argument("--json", action="store_true")


def _emit_blueprint(result, arguments, markdown=None):
    value = result.to_dict()
    if arguments.json:
        print(json.dumps(value, indent=2))
    elif result.status == "ok":
        if "path" in result.data:
            print("Project design pack written to {0}".format(result.data["path"]))
        else:
            print(markdown if markdown is not None else "Project design pack generated.")
    else:
        error = value["error"]
        print("blueprint failed [{0}]: {1}\nremediation: {2}".format(
            error["code"], error["message"], error["remediation"]
        ))
    return 0 if result.status == "ok" else 2


def _run_blueprint(arguments):
    operation = "blueprint"
    try:
        if not arguments.goal:
            return _emit_blueprint(ResultEnvelope.fail(
                operation, "goal_required",
                "No project goal was supplied.",
                "Pass --goal \"...\".",
            ), arguments)
        brief = build_brief({
            "goal": arguments.goal,
            "modalities": arguments.modalities or [],
            "memory_gb": arguments.memory_gb,
            "notes": arguments.notes,
        })
    except (ValueError, TypeError) as error:
        return _emit_blueprint(ResultEnvelope.fail(
            operation, "invalid_brief", str(error),
            "Correct the flags and retry.",
        ), arguments)

    moment = datetime.now(timezone.utc)
    pack = generate_design_pack(brief, now=moment)
    markdown = render_design_pack(pack)
    data = {"pack": pack.to_dict()}
    if arguments.write:
        try:
            data["path"] = str(write_design_pack(
                markdown, brief, root=arguments.project, now=moment, pack=pack,
            ))
        except (OSError, ValueError) as error:
            return _emit_blueprint(ResultEnvelope.fail(
                operation, "write_failed",
                "Design pack could not be written: {0}".format(error),
                "Choose a writable --project directory without a symlinked mlx-blueprints folder.",
            ), arguments)
    return _emit_blueprint(ResultEnvelope.ok(operation, data), arguments, markdown)


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
                preview = preview_rollback(arguments.receipt)
                result = _wire_ok(
                    operation, {"preview": preview, "requires_confirmation": True}
                )
                if arguments.json:
                    print(json.dumps(result.to_dict(), indent=2))
                else:
                    print(json.dumps(preview, indent=2))
                    print(
                        "Confirmation required: rerun with --confirm --preview-hash PREVIEW_HASH."
                    )
                return 2
            if not arguments.preview_hash:
                return _emit_wire_result(_wire_failure(
                    operation,
                    "preview_hash_required",
                    "--confirm requires the hash from a reviewed rollback preview.",
                    "Run wire rollback without --confirm, inspect the current-state preview, then pass --preview-hash.",
                ), arguments.json)
            receipt = rollback(arguments.receipt, preview_hash=arguments.preview_hash)
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


_BENCH_RUNTIMES = ("ollama", "lmstudio", "mlx_lm", "mlx-vlm", "litellm")


def _bench_runtime_client(name):
    if name == "ollama":
        return OllamaRuntimeClient()
    if name == "lmstudio":
        return LMStudioRuntimeClient()
    if name == "mlx-vlm":
        return MLXVLMRuntimeClient()
    if name == "mlx_lm":
        return OpenAICompatibleRuntimeClient("mlx_lm", "http://127.0.0.1:8080")
    if name == "litellm":
        return OpenAICompatibleRuntimeClient("litellm", "http://127.0.0.1:4000")
    raise ValueError("unsupported bench runtime: {0}".format(name))


def _add_bench_arguments(parser):
    actions = parser.add_subparsers(dest="bench_command", required=True)
    run = actions.add_parser(
        "run",
        help="measure a model already served by a local runtime (never downloads)",
    )
    run.add_argument("--repo", required=True, help="model identifier exactly as the runtime serves it")
    run.add_argument("--runtime", required=True, choices=_BENCH_RUNTIMES)
    run.add_argument("--role", default="general", choices=DISCOVERY_ROLES)
    run.add_argument("--runs", type=int, default=RUNS_DEFAULT, help="timed runs after one warm-up (1-10)")
    run.add_argument("--gen-tokens", type=int, default=GEN_TOKENS_DEFAULT, help="tokens generated per run (16-2048)")
    run.add_argument("--timeout", type=float, default=TIMEOUT_DEFAULT, help="total measurement deadline in seconds")
    run.add_argument("--export", default=None, help="append an anonymized result line to this JSONL file")
    run.add_argument("--json", action="store_true")
    aggregate = actions.add_parser(
        "aggregate",
        help="deduplicate and median-aggregate bench export files",
    )
    aggregate.add_argument("--exports", required=True, help="directory of .jsonl export files")
    aggregate.add_argument("--out", default=None, help="aggregate output path (default: print only)")
    aggregate.add_argument("--json", action="store_true")


def _run_bench(arguments):
    operation = "bench-{0}".format(arguments.bench_command)
    try:
        if arguments.bench_command == "aggregate":
            from .bench import aggregate_exports

            exports_dir = Path(arguments.exports)
            if not exports_dir.is_dir():
                raise BenchError(
                    "invalid_arguments",
                    "The exports directory does not exist: {0}".format(arguments.exports),
                    "Point --exports at a directory of bench export .jsonl files.",
                )
            files = sorted(exports_dir.glob("*.jsonl"))
            aggregate = aggregate_exports(files)
            if arguments.out:
                destination = Path(arguments.out)
                content = (json.dumps(aggregate, indent=2, sort_keys=True) + "\n").encode("utf-8")
                destination.write_bytes(content)
            data = {
                "entries": aggregate["entries"],
                "sources": len(files),
                "written": str(arguments.out) if arguments.out else None,
            }
            result = ResultEnvelope.ok(operation, data)
            if arguments.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print("Aggregate: {0} entries from {1} export file(s){2}".format(
                    len(aggregate["entries"]), len(files),
                    " -> {0}".format(arguments.out) if arguments.out else "",
                ))
            return 0
        chip = None
        try:
            chip = HostInventory.inspect().chip
        except Exception:
            chip = None
        measurement = measure_runtime(
            arguments.repo,
            _bench_runtime_client(arguments.runtime),
            runs=arguments.runs,
            gen_tokens=arguments.gen_tokens,
            timeout=arguments.timeout,
            chip=chip,
        )
        evidence = measurement.to_evidence(role=arguments.role).to_dict()
        data = {"measurement": measurement.to_dict(), "evidence": evidence}
        if arguments.export:
            from . import __version__
            from .bench import append_export, export_record

            destination = append_export(
                arguments.export, export_record(measurement, __version__)
            )
            data["exported"] = str(destination)
        result = ResultEnvelope.ok(operation, data)
        if arguments.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("Bench: {0} on {1} ({2} runs)".format(
                measurement.repo, measurement.runtime, measurement.runs
            ))
            ttft = "{0} ms".format(measurement.ttft_ms) if measurement.ttft_ms is not None else "n/a"
            prefill = (
                "{0} tok/s".format(measurement.prefill_toks)
                if measurement.prefill_toks is not None
                else "n/a"
            )
            print("  decode:  {0} tok/s (spread {1}%)".format(measurement.decode_toks, measurement.spread_pct))
            print("  ttft:    {0}".format(ttft))
            print("  prefill: {0}".format(prefill))
        return 0
    except BenchError as error:
        result = ResultEnvelope.fail(operation, error.code, str(error), error.remediation)
    except (TypeError, ValueError) as error:
        result = ResultEnvelope.fail(
            operation,
            "invalid_arguments",
            str(error),
            "Correct the bench arguments and retry.",
        )
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        payload = result.to_dict()["error"]
        print("bench failed [{0}]: {1}\nremediation: {2}".format(
            payload["code"], payload["message"], payload["remediation"]
        ))
    return 2


def _add_serve_arguments(parser):
    actions = parser.add_subparsers(dest="serve_command", required=True)
    start = actions.add_parser(
        "start",
        help="preview, then confirmation-gated launch of a local MLX server",
    )
    start.add_argument("--repo", required=True, help="publisher/model present in the local Hugging Face cache")
    start.add_argument("--runtime", required=True, choices=["mlx_lm", "mlx-vlm"])
    start.add_argument("--port", type=int, default=None, help="loopback port (defaults per runtime recipe)")
    start.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    start.add_argument("--adapter-path", default=None, help="LoRA adapter directory (mlx_lm only)")
    start.add_argument("--launchd", action="store_true", help="install a launchd agent plist instead of spawning now")
    start.add_argument("--launchd-dir", default=None, help="launchd target directory (defaults to ~/Library/LaunchAgents)")
    start.add_argument("--confirm", action="store_true", help="authorize this reviewed server launch")
    start.add_argument("--preview-hash", help="hash returned by the separately reviewed serve start preview")
    start.add_argument("--receipts-dir", default=None, help="receipt root (defaults to the current directory)")
    start.add_argument("--hf-cache", default=None, help="Hugging Face cache root for the local-model gate")
    start.add_argument("--json", action="store_true")
    stop = actions.add_parser("stop", help="stop exactly the server a serve receipt owns")
    stop.add_argument("--port", type=int, required=True)
    stop.add_argument("--receipts-dir", default=None)
    stop.add_argument("--json", action="store_true")
    status = actions.add_parser("status", help="cross-check serve receipts against live processes")
    status.add_argument("--receipts-dir", default=None)
    status.add_argument("--json", action="store_true")


def _run_serve(arguments):
    operation = "serve-{0}".format(arguments.serve_command)
    try:
        if arguments.serve_command == "status":
            entries = status_serve(arguments.receipts_dir)
            return _emit_serve_result(
                ResultEnvelope.ok(operation, {"servers": entries}), arguments.json,
                human=_serve_status_human(entries),
            )
        if arguments.serve_command == "stop":
            outcome = stop_serve(arguments.port, arguments.receipts_dir)
            return _emit_serve_result(
                ResultEnvelope.ok(operation, outcome), arguments.json,
                human="serve {0}: port {1} (pid {2})".format(
                    outcome["status"], outcome["port"], outcome["pid"]
                ),
            )
        recipes = load_recipes()
        plan = plan_start(
            arguments.repo,
            arguments.runtime,
            recipes,
            port=arguments.port,
            max_tokens=arguments.max_tokens,
            adapter_path=arguments.adapter_path,
        )
        if arguments.launchd:
            return _run_serve_launchd(arguments, plan, operation)
        if not arguments.confirm:
            result = ResultEnvelope.ok(
                operation, {"plan": plan, "requires_confirmation": True}
            )
            if arguments.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print("Serve plan: {0} on 127.0.0.1:{1}".format(plan["repo"], plan["port"]))
                print("  argv: {0}".format(" ".join(plan["argv"])))
                print("  readiness: {0}".format(plan["readiness"]))
                print("  preview_hash: {0}".format(plan["preview_hash"]))
                print("Confirmation required: rerun with --confirm --preview-hash PREVIEW_HASH.")
            return 2
        wired_claims = wired_port_claim([
            arguments.receipts_dir or str(Path.cwd())
        ])
        outcome = start_serve(
            plan,
            receipts_dir=arguments.receipts_dir,
            confirm=True,
            preview_hash=arguments.preview_hash,
            model_present=default_model_present(arguments.hf_cache),
            wired_claims=wired_claims,
        )
        receipt = outcome["receipt"]
        return _emit_serve_result(
            ResultEnvelope.ok(operation, outcome), arguments.json,
            human="serve started: {0} on 127.0.0.1:{1} (pid {2})\n  log: {3}".format(
                receipt["repo"], receipt["port"], receipt["pid"], receipt["log_path"]
            ),
        )
    except ServeError as error:
        result = ResultEnvelope.fail(operation, error.code, str(error), error.remediation)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        result = ResultEnvelope.fail(
            operation,
            "serve_failed",
            str(error),
            "Correct the serve arguments or receipt state, then retry.",
        )
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        payload = result.to_dict()["error"]
        print("serve failed [{0}]: {1}\nremediation: {2}".format(
            payload["code"], payload["message"], payload["remediation"]
        ))
    return 2


def _run_serve_launchd(arguments, plan, operation):
    """Preview-confirm installation of a launchd agent for a serve plan."""
    from .serve import LaunchdPlistAdapter, launchd_label, render_launchd_plist

    launchd_dir = Path(arguments.launchd_dir) if arguments.launchd_dir else (
        Path.home() / "Library" / "LaunchAgents"
    )
    target = launchd_dir / "{0}.plist".format(launchd_label(plan["port"]))
    log_path = str(receipts_root(arguments.receipts_dir) / "{0}.log".format(plan["port"]))
    content = render_launchd_plist(plan, log_path=log_path)
    adapter = LaunchdPlistAdapter(target)
    adapter.validate(content)
    if target.exists():
        result = ResultEnvelope.fail(
            operation,
            "output_exists",
            "A launchd plist already exists at {0}.".format(target),
            "Unload and remove it yourself; serve never overwrites launchd agents.",
        )
        return _emit_serve_result(result, arguments.json)
    if not launchd_dir.is_dir():
        result = ResultEnvelope.fail(
            operation,
            "invalid_arguments",
            "The launchd directory does not exist: {0}".format(launchd_dir),
            "Create it yourself, or pass --launchd-dir.",
        )
        return _emit_serve_result(result, arguments.json)
    if not default_model_present(arguments.hf_cache)(plan["repo"]):
        result = ResultEnvelope.fail(
            operation,
            "model_not_local",
            "The model is not present in the local Hugging Face cache.",
            "Download it with the runtime's own pull command first; serve never downloads.",
        )
        return _emit_serve_result(result, arguments.json)
    transaction = Transaction(receipts_dir=arguments.receipts_dir)
    preview = transaction.preview([{
        "path": str(target), "content": content, "runtime": "launchd",
        "adapter": adapter, "endpoint": None,
    }])
    if not arguments.confirm:
        result = ResultEnvelope.ok(
            operation, {"preview": preview, "requires_confirmation": True}
        )
        if arguments.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(preview["diff"])
            print("Confirmation required: rerun with --confirm --preview-hash PREVIEW_HASH.")
        return 2
    if not arguments.preview_hash:
        result = ResultEnvelope.fail(
            operation,
            "preview_hash_required",
            "--confirm requires the preview hash from a prior preview.",
            "Run serve start --launchd without --confirm, inspect the preview, then pass --preview-hash.",
        )
        return _emit_serve_result(result, arguments.json)
    if arguments.preview_hash != preview["preview_hash"]:
        result = ResultEnvelope.fail(
            operation,
            "preview_stale",
            "The supplied preview hash does not match the current preview.",
            "Generate and inspect a new preview before confirming this mutation.",
        )
        return _emit_serve_result(result, arguments.json)
    receipt = transaction.apply(arguments.preview_hash)
    data = {"receipt": _receipt_data(receipt), "plist": str(target)}
    result = ResultEnvelope.ok(operation, data)
    return _emit_serve_result(
        result, arguments.json,
        human="launchd agent installed: {0}\n  load it with: launchctl bootstrap gui/$(id -u) {1}".format(
            target, target
        ),
    )


def _emit_serve_result(result, as_json, human=None):
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
    elif human:
        print(human)
    elif result.status == "ok":
        print(json.dumps(result.to_dict()["data"], indent=2))
    else:
        payload = result.to_dict()["error"]
        print("{0} failed [{1}]: {2}\nremediation: {3}".format(
            result.operation, payload["code"], payload["message"], payload["remediation"]
        ))
    return 0 if result.status == "ok" else 2


def _serve_status_human(entries):
    if not entries:
        return "No serve receipts."
    lines = []
    for entry in entries:
        state = "alive" if entry["alive"] else "dead"
        if entry["alive"] and not entry["argv_match"]:
            state = "alive (argv mismatch)"
        lines.append("  127.0.0.1:{0} {1} ({2}) pid {3} - {4}".format(
            entry["port"], entry["repo"], entry["runtime"], entry["pid"], state
        ))
    return "Serve receipts:\n" + "\n".join(lines)


def _add_fleet_arguments(parser):
    actions = parser.add_subparsers(dest="fleet_command", required=True)
    for name in ("render", "apply"):
        action = actions.add_parser(
            name,
            help="{0} a one-shot per-role router configuration".format(name),
        )
        action.add_argument("--path", required=True, help="target router configuration file")
        action.add_argument("--assign", action="append", default=None, metavar="ROLE=REPO",
                            help="per-role model assignment (repeatable)")
        action.add_argument("--from-adoption", default=None, metavar="STATE",
                            help="read role assignments from a completed adopt handoff")
        action.add_argument("--runtime-map", action="append", default=None, metavar="ROLE=RUNTIME",
                            help="per-role runtime override: mlx_lm or mlx-vlm (repeatable)")
        action.add_argument("--allow-missing", action="store_true",
                            help="warn instead of failing when a model is not in a local inventory")
        action.add_argument("--json", action="store_true")
        if name == "apply":
            action.add_argument("--confirm", action="store_true", help="explicitly authorize this reviewed mutation")
            action.add_argument("--preview-hash", help="hash returned by the separately reviewed fleet apply preview")
            action.add_argument("--receipts-dir", help="directory for non-secret transaction receipts")
            action.add_argument("--endpoint", help="optional local runtime health endpoint")


def _fleet_inputs(arguments):
    if arguments.assign and arguments.from_adoption:
        raise FleetError(
            "invalid_arguments",
            "--assign and --from-adoption are mutually exclusive.",
            "Pick explicit assignments or an adoption handoff, not both.",
        )
    if arguments.from_adoption:
        assignments = assignments_from_adoption(arguments.from_adoption)
    else:
        assignments = parse_assignments(arguments.assign)
    runtime_map = parse_runtime_map(arguments.runtime_map)
    return assignments, runtime_map


def _fleet_missing_models(assignments):
    from .model_doctor import (
        collect_runtime_inventory,
        default_hf_cache,
        inspect_hf_cache,
    )

    known = {item["id"].casefold() for item in inspect_hf_cache(default_hf_cache())}
    runtime_clients = [
        OllamaRuntimeClient(),
        LMStudioRuntimeClient(),
        OpenAICompatibleRuntimeClient("mlx_lm", "http://127.0.0.1:8080"),
        MLXVLMRuntimeClient(),
        OpenAICompatibleRuntimeClient("litellm", "http://127.0.0.1:4000"),
    ]
    runtime_inventory, _errors = collect_runtime_inventory(runtime_clients)
    known.update(item["id"].casefold() for item in runtime_inventory)
    return sorted(
        repo for repo in set(assignments.values()) if repo.casefold() not in known
    )


def _run_fleet(arguments):
    operation = "fleet-{0}".format(arguments.fleet_command)
    try:
        assignments, runtime_map = _fleet_inputs(arguments)
        path = _assert_safe_target(arguments.path)
        existing = _read_regular(path).decode("utf-8")
        adapter = FleetConfigAdapter(path)
        content = adapter.render(assignments, runtime_map, existing=existing)
        missing = _fleet_missing_models(assignments)
        warnings = []
        if missing and not arguments.allow_missing:
            return _emit_wire_result(_wire_failure(
                operation,
                "model_not_local",
                "Models are not present in a local inventory: {0}".format(", ".join(missing)),
                "Download them first, or pass --allow-missing to record a reviewed warning.",
            ), arguments.json)
        if missing:
            warnings.append({
                "code": "model_not_local",
                "message": "Not in a local inventory: {0}".format(", ".join(missing)),
            })
        if arguments.fleet_command == "render":
            result = _wire_ok(operation, {
                "path": str(path), "assignments": assignments, "config": content,
                "validation": {"parse": True},
            })
            result.warnings.extend(warnings)
            return _emit_wire_result(result, arguments.json, human=content)
        transaction = Transaction(receipts_dir=arguments.receipts_dir)
        preview = transaction.preview([{
            "path": str(path), "content": content, "runtime": "litellm",
            "adapter": adapter, "endpoint": arguments.endpoint,
        }])
        if not arguments.confirm:
            result = _wire_ok(operation, {"preview": preview, "requires_confirmation": True})
            result.warnings.extend(warnings)
            if arguments.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(preview["diff"])
                print("Confirmation required: rerun with --confirm to apply this transaction.")
            return 2
        if not arguments.preview_hash:
            return _emit_wire_result(_wire_failure(
                operation, "preview_hash_required", "--confirm requires the preview hash from a prior preview.",
                "Run fleet apply without --confirm, inspect the preview, then pass its --preview-hash value.",
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
            operation, receipt.status, "Fleet did not apply; receipt status is {0}.".format(receipt.status),
            "Inspect the receipt validation results and use the recorded recovery path before retrying.", data,
        )
        result.warnings.extend(warnings)
        return _emit_wire_result(result, arguments.json)
    except FleetError as error:
        return _emit_wire_result(_wire_failure(
            operation, error.code, str(error), error.remediation,
        ), arguments.json)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        code = "cooperative_lock_busy" if isinstance(error, ConcurrentTransactionError) else ("preview_stale" if str(error).startswith("preview is stale") else "fleet_failed")
        return _emit_wire_result(_wire_failure(
            operation, code, "Fleet could not complete: {0}".format(error),
            "Correct the assignments or target configuration, then render a new preview.",
        ), arguments.json)


def _watch_runtime_clients():
    return [
        OllamaRuntimeClient(),
        LMStudioRuntimeClient(),
        OpenAICompatibleRuntimeClient("mlx_lm", "http://127.0.0.1:8080"),
        MLXVLMRuntimeClient(),
        OpenAICompatibleRuntimeClient("litellm", "http://127.0.0.1:4000"),
    ]


def _add_watch_arguments(parser):
    actions = parser.add_subparsers(dest="watch_command", required=True)
    for name in ("snapshot", "diff"):
        action = actions.add_parser(
            name,
            help="{0} the owned-model Hugging Face watch state".format(name),
        )
        action.add_argument("--state-dir", default="./.mlx-agent-watch",
                            help="watch state root (default ./.mlx-agent-watch)")
        action.add_argument("--limit", type=int, default=6,
                            help="discovery results per role (default 6)")
        action.add_argument("--json", action="store_true")


def _run_watch(arguments):
    operation = "watch-{0}".format(arguments.watch_command)
    try:
        service, fixture_warning, fixture_error = _discovery_service_from_environment()
        if fixture_error is not None:
            return _emit_wire_result(fixture_error, arguments.json)
        result = service.discover(DiscoveryRequest(limit=arguments.limit))
        if result.status != "ok":
            error = result.to_dict()["error"]
            return _emit_wire_result(ResultEnvelope.fail(
                operation, error["code"], error["message"], error["remediation"]
            ), arguments.json)
        owned = collect_owned(_watch_runtime_clients())
        candidates = snapshot_candidates(result.data)
        if arguments.watch_command == "snapshot":
            snapshot = build_snapshot(owned, candidates)
            destination = write_snapshot(arguments.state_dir, snapshot)
            data = {
                "state": str(destination),
                "owned": len(snapshot["owned"]),
                "candidates": len(snapshot["candidates"]),
                "rotated_previous": snapshot["previous"] is not None,
            }
            result = ResultEnvelope.ok(operation, data)
            if arguments.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print("Watch snapshot: {0} owned, {1} tracked repos -> {2}".format(
                    data["owned"], data["candidates"], data["state"]
                ))
            return 0
        baseline = read_baseline(arguments.state_dir)
        current = build_snapshot(owned, candidates)
        findings = diff_snapshots(baseline, current)
        data = {
            "baseline_created_at": baseline["created_at"],
            "findings": findings,
        }
        warnings = [
            {"code": finding["code"], "message": "{0}: {1}".format(finding["repo"], finding["detail"])}
            for finding in findings
        ]
        result = ResultEnvelope.ok(operation, data, warnings=warnings)
        if arguments.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("Watch diff against {0}: {1} finding(s)".format(
                baseline["created_at"], len(findings)
            ))
            for finding in findings:
                print("  [{0}] {1} - {2}".format(finding["code"], finding["repo"], finding["detail"]))
        return 0
    except WatchError as error:
        result = ResultEnvelope.fail(operation, error.code, str(error), error.remediation)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        result = ResultEnvelope.fail(
            operation,
            "watch_failed",
            str(error),
            "Correct the state directory and retry; watch only writes its own state file.",
        )
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        payload = result.to_dict()["error"]
        print("watch failed [{0}]: {1}\nremediation: {2}".format(
            payload["code"], payload["message"], payload["remediation"]
        ))
    return 2


def _add_convert_arguments(parser):
    actions = parser.add_subparsers(dest="convert_command", required=True)
    start = actions.add_parser(
        "start",
        help="preview, then confirmation-gated quantization of a cached model",
    )
    start.add_argument("--repo", required=True, help="publisher/model present in the local Hugging Face cache")
    start.add_argument("--q-bits", type=int, default=4, choices=Q_BITS_CHOICES)
    start.add_argument("--out", default=None, help="output directory (default <model>-MLX-<bits>bit)")
    start.add_argument("--confirm", action="store_true", help="authorize this reviewed conversion")
    start.add_argument("--preview-hash", help="hash returned by the separately reviewed convert preview")
    start.add_argument("--receipts-dir", default=None)
    start.add_argument("--hf-cache", default=None)
    start.add_argument("--json", action="store_true")
    status = actions.add_parser("status", help="cross-check convert receipts against live processes")
    status.add_argument("--receipts-dir", default=None)
    status.add_argument("--json", action="store_true")


def _run_convert(arguments):
    operation = "convert-{0}".format(arguments.convert_command)
    try:
        if arguments.convert_command == "status":
            entries = status_convert(arguments.receipts_dir)
            return _emit_serve_result(
                ResultEnvelope.ok(operation, {"jobs": entries}), arguments.json,
                human=_convert_status_human(entries),
            )
        plan = plan_convert(arguments.repo, q_bits=arguments.q_bits, out=arguments.out)
        if not arguments.confirm:
            result = ResultEnvelope.ok(
                operation, {"plan": plan, "requires_confirmation": True}
            )
            if arguments.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print("Convert plan: {0} -> {1} ({2}bit)".format(
                    plan["repo"], plan["out"], plan["q_bits"]
                ))
                print("  argv: {0}".format(" ".join(plan["argv"])))
                print("  preview_hash: {0}".format(plan["preview_hash"]))
                print("Confirmation required: rerun with --confirm --preview-hash PREVIEW_HASH.")
            return 2
        outcome = start_convert(
            plan,
            receipts_dir=arguments.receipts_dir,
            confirm=True,
            preview_hash=arguments.preview_hash,
            model_present=default_model_present(arguments.hf_cache),
        )
        receipt = outcome["receipt"]
        return _emit_serve_result(
            ResultEnvelope.ok(operation, outcome), arguments.json,
            human="convert started: {0} -> {1} (pid {2})\n  log: {3}".format(
                receipt["repo"], receipt["out"], receipt["pid"], receipt["log_path"]
            ),
        )
    except ConvertError as error:
        result = ResultEnvelope.fail(operation, error.code, str(error), error.remediation)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        result = ResultEnvelope.fail(
            operation,
            "convert_failed",
            str(error),
            "Correct the convert arguments or receipt state, then retry.",
        )
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        payload = result.to_dict()["error"]
        print("convert failed [{0}]: {1}\nremediation: {2}".format(
            payload["code"], payload["message"], payload["remediation"]
        ))
    return 2


def _convert_status_human(entries):
    if not entries:
        return "No convert receipts."
    lines = []
    for entry in entries:
        lines.append("  {0} ({1}bit) -> {2}: {3}".format(
            entry["repo"], entry["q_bits"], entry["out"], entry["state"]
        ))
    return "Convert jobs:\n" + "\n".join(lines)


def _add_lora_arguments(parser):
    actions = parser.add_subparsers(dest="lora_command", required=True)
    start = actions.add_parser(
        "start",
        help="preview, then confirmation-gated LoRA training on a cached model",
    )
    start.add_argument("--repo", required=True, help="publisher/model base present in the local Hugging Face cache")
    start.add_argument("--data", required=True, help="dataset directory with train.jsonl")
    start.add_argument("--iters", type=int, default=ITERS_DEFAULT)
    start.add_argument("--batch-size", type=int, default=BATCH_DEFAULT)
    start.add_argument("--learning-rate", type=float, default=LR_DEFAULT)
    start.add_argument("--num-layers", type=int, default=LAYERS_DEFAULT, help="-1 trains all layers")
    start.add_argument("--out", default=None, help="adapter output directory (default <model>-lora)")
    start.add_argument("--confirm", action="store_true", help="authorize this reviewed training run")
    start.add_argument("--preview-hash", help="hash returned by the separately reviewed lora preview")
    start.add_argument("--receipts-dir", default=None)
    start.add_argument("--hf-cache", default=None)
    start.add_argument("--json", action="store_true")
    status = actions.add_parser("status", help="cross-check lora receipts against live processes")
    status.add_argument("--receipts-dir", default=None)
    status.add_argument("--json", action="store_true")


def _run_lora(arguments):
    operation = "lora-{0}".format(arguments.lora_command)
    try:
        if arguments.lora_command == "status":
            entries = status_lora(arguments.receipts_dir)
            return _emit_serve_result(
                ResultEnvelope.ok(operation, {"jobs": entries}), arguments.json,
                human=_lora_status_human(entries),
            )
        plan = plan_lora(
            arguments.repo,
            arguments.data,
            iters=arguments.iters,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            num_layers=arguments.num_layers,
            out=arguments.out,
        )
        if not arguments.confirm:
            result = ResultEnvelope.ok(
                operation, {"plan": plan, "requires_confirmation": True}
            )
            if arguments.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print("LoRA plan: {0} + {1} train lines -> {2}".format(
                    plan["repo"], plan["dataset"]["train_lines"], plan["out"]
                ))
                print("  argv: {0}".format(" ".join(plan["argv"])))
                print("  preview_hash: {0}".format(plan["preview_hash"]))
                print("Confirmation required: rerun with --confirm --preview-hash PREVIEW_HASH.")
            return 2
        outcome = start_lora(
            plan,
            receipts_dir=arguments.receipts_dir,
            confirm=True,
            preview_hash=arguments.preview_hash,
            model_present=default_model_present(arguments.hf_cache),
        )
        receipt = outcome["receipt"]
        return _emit_serve_result(
            ResultEnvelope.ok(operation, outcome), arguments.json,
            human="lora started: {0} -> {1} (pid {2})\n  log: {3}".format(
                receipt["repo"], receipt["out"], receipt["pid"], receipt["log_path"]
            ),
        )
    except LoraError as error:
        result = ResultEnvelope.fail(operation, error.code, str(error), error.remediation)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        result = ResultEnvelope.fail(
            operation,
            "lora_failed",
            str(error),
            "Correct the lora arguments or receipt state, then retry.",
        )
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        payload = result.to_dict()["error"]
        print("lora failed [{0}]: {1}\nremediation: {2}".format(
            payload["code"], payload["message"], payload["remediation"]
        ))
    return 2


def _lora_status_human(entries):
    if not entries:
        return "No lora receipts."
    lines = []
    for entry in entries:
        lines.append("  {0} -> {1}: {2}".format(
            entry["repo"], entry["out"], entry["state"]
        ))
    return "LoRA jobs:\n" + "\n".join(lines)


def _add_fuse_arguments(parser):
    actions = parser.add_subparsers(dest="fuse_command", required=True)
    start = actions.add_parser(
        "start",
        help="preview, then confirmation-gated LoRA fuse into the base model",
    )
    start.add_argument("--repo", required=True, help="publisher/model base present in the local Hugging Face cache")
    start.add_argument("--adapter", required=True, help="completed lora adapter directory")
    start.add_argument("--out", default=None, help="fused output directory (default <model>-fused)")
    start.add_argument("--confirm", action="store_true", help="authorize this reviewed fuse")
    start.add_argument("--preview-hash", help="hash returned by the separately reviewed fuse preview")
    start.add_argument("--receipts-dir", default=None)
    start.add_argument("--hf-cache", default=None)
    start.add_argument("--json", action="store_true")
    status = actions.add_parser("status", help="cross-check fuse receipts against live processes")
    status.add_argument("--receipts-dir", default=None)
    status.add_argument("--json", action="store_true")


def _run_fuse(arguments):
    operation = "fuse-{0}".format(arguments.fuse_command)
    try:
        if arguments.fuse_command == "status":
            entries = status_fuse(arguments.receipts_dir)
            return _emit_serve_result(
                ResultEnvelope.ok(operation, {"jobs": entries}), arguments.json,
                human=_fuse_status_human(entries),
            )
        plan = plan_fuse(arguments.repo, arguments.adapter, out=arguments.out)
        if not arguments.confirm:
            result = ResultEnvelope.ok(
                operation, {"plan": plan, "requires_confirmation": True}
            )
            if arguments.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print("Fuse plan: {0} + {1} -> {2}".format(
                    plan["repo"], plan["adapter"], plan["out"]
                ))
                print("  argv: {0}".format(" ".join(plan["argv"])))
                print("  preview_hash: {0}".format(plan["preview_hash"]))
                print("Confirmation required: rerun with --confirm --preview-hash PREVIEW_HASH.")
            return 2
        outcome = start_fuse(
            plan,
            receipts_dir=arguments.receipts_dir,
            confirm=True,
            preview_hash=arguments.preview_hash,
            model_present=default_model_present(arguments.hf_cache),
        )
        receipt = outcome["receipt"]
        return _emit_serve_result(
            ResultEnvelope.ok(operation, outcome), arguments.json,
            human="fuse started: {0} -> {1} (pid {2})\n  log: {3}".format(
                receipt["repo"], receipt["out"], receipt["pid"], receipt["log_path"]
            ),
        )
    except FuseError as error:
        result = ResultEnvelope.fail(operation, error.code, str(error), error.remediation)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        result = ResultEnvelope.fail(
            operation,
            "fuse_failed",
            str(error),
            "Correct the fuse arguments or receipt state, then retry.",
        )
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        payload = result.to_dict()["error"]
        print("fuse failed [{0}]: {1}\nremediation: {2}".format(
            payload["code"], payload["message"], payload["remediation"]
        ))
    return 2


def _fuse_status_human(entries):
    if not entries:
        return "No fuse receipts."
    lines = []
    for entry in entries:
        lines.append("  {0} -> {1}: {2}".format(
            entry["repo"], entry["out"], entry["state"]
        ))
    return "Fuse jobs:\n" + "\n".join(lines)


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
    restore.add_argument("--preview-hash", help="hash returned by the separately reviewed rollback preview")
    restore.add_argument("--json", action="store_true")


def _installer_registry():
    root = Path(__file__).resolve().parents[2]
    home = os.environ.get("MLX_AGENT_HOME")
    config_root = os.environ.get("MLX_AGENT_CONFIG_ROOT") or os.environ.get("XDG_STATE_HOME")
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    return ProviderRegistry(
        root / "plugin.json", home=home, config_root=config_root,
        xdg_config_home=xdg_config_home,
    )


def _add_installer_arguments(parser, include_providers=True):
    if include_providers:
        parser.add_argument("providers", nargs="*", help="explicit provider IDs; omit to inspect detected choices")
    parser.add_argument("--scope", choices=["user", "project"], default="user")
    parser.add_argument("--project", default=str(Path.cwd()), help="project root for project-scoped installation")
    parser.add_argument("--dry-run", action="store_true", help="show the reviewed plan without mutating files")
    parser.add_argument("--confirm", action="store_true", help="authorize a separately reviewed installer preview")
    parser.add_argument("--preview-hash", help="preview hash returned by the installer dry run")
    parser.add_argument("--json", action="store_true")


def _add_doctor_arguments(parser):
    _add_installer_arguments(parser)
    parser.add_argument(
        "--wired-root",
        action="append",
        default=None,
        help="root scanned for Wire-managed configs (repeatable; defaults to the project root)",
    )
    parser.add_argument(
        "--hf-cache",
        default=None,
        help="Hugging Face cache root (defaults to ~/.cache/huggingface/hub)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="preview deletion of incomplete HF cache snapshots (requires --confirm --preview-hash to execute)",
    )


def _run_model_doctor(arguments):
    from .model_doctor import (
        PruneError,
        default_hf_cache,
        execute_prune,
        plan_prune,
        run_model_doctor,
    )

    operation = "doctor-models"
    wired_roots = arguments.wired_root or [arguments.project]
    hf_cache = arguments.hf_cache or default_hf_cache()
    if arguments.prune:
        try:
            plan = plan_prune(hf_cache)
            outcome = execute_prune(
                plan,
                confirm=arguments.confirm,
                preview_hash=arguments.preview_hash,
            )
        except PruneError as error:
            result = ResultEnvelope.fail(operation, error.code, str(error), error.remediation)
            if arguments.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                payload = result.to_dict()["error"]
                print("doctor failed [{0}]: {1}\nremediation: {2}".format(
                    payload["code"], payload["message"], payload["remediation"]
                ))
            return 2
        if outcome["status"] == "preview":
            result = ResultEnvelope.ok(
                operation, {"plan": plan, "requires_confirmation": True}
            )
            if arguments.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                if not plan["candidates"]:
                    print("Nothing to prune: no incomplete cache snapshots.")
                for candidate in plan["candidates"]:
                    print("  prune: {0} ({1:.1f} GB) - {2}".format(
                        candidate["repo"], candidate["bytes"] / 1e9, candidate["path"]
                    ))
                if plan["candidates"]:
                    print("IRREVERSIBLE: these cache directories will be deleted permanently.")
                    print("  preview_hash: {0}".format(plan["preview_hash"]))
                    print("Confirmation required: rerun with --prune --confirm --preview-hash PREVIEW_HASH.")
            return 0 if not plan["candidates"] else 2
        result = ResultEnvelope.ok(operation, outcome)
        if arguments.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            removed = [item for item in outcome["removed"] if item.get("removed")]
            print("Pruned {0} incomplete snapshot(s).".format(len(removed)))
            for item in removed:
                print("  removed: {0}".format(item["repo"]))
        return 0
    runtime_clients = [
        OllamaRuntimeClient(),
        LMStudioRuntimeClient(),
        OpenAICompatibleRuntimeClient("mlx_lm", "http://127.0.0.1:8080"),
        MLXVLMRuntimeClient(),
        OpenAICompatibleRuntimeClient("litellm", "http://127.0.0.1:4000"),
    ]
    try:
        report = run_model_doctor(wired_roots, hf_cache, runtime_clients)
    except (OSError, TypeError, ValueError) as error:
        result = ResultEnvelope.fail(
            operation,
            "doctor_failed",
            str(error),
            "Correct the scanned roots and retry; doctor never mutates anything.",
        )
        if arguments.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            payload = result.to_dict()["error"]
            print("doctor failed [{0}]: {1}\nremediation: {2}".format(
                payload["code"], payload["message"], payload["remediation"]
            ))
        return 2
    warnings = [
        {"code": finding["code"], "message": "{0}: {1}".format(
            finding.get("path") or finding.get("model") or finding["code"],
            finding["remediation"],
        )}
        for finding in report["findings"]
    ]
    result = ResultEnvelope.ok(operation, report, warnings=warnings)
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        summary = report["summary"]
        print("Doctor: {0} model(s), {1} wired config(s), {2} finding(s)".format(
            summary["models"], summary["wired_configs"], summary["findings"]
        ))
        print("  hf-cache disk: {0:.1f} GB".format(summary["hf_cache_bytes"] / 1e9))
        for finding in report["findings"]:
            location = finding.get("path") or finding.get("model") or ""
            print("  [{0}] {1} - {2}".format(finding["code"], location, finding["remediation"]))
        for endpoint in report["endpoints"]:
            print("  endpoint {0}: {1}".format(endpoint["endpoint"], endpoint["status"]))
        for runtime_error in report["runtime_errors"]:
            print("  runtime unreachable: {0}".format(runtime_error))
    return 0


def _installer_result(operation, data, as_json, status="ok", error=None):
    result = ResultEnvelope.ok(operation, data) if status == "ok" else ResultEnvelope.fail(
        operation, error[0], error[1], error[2]
    )
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
    elif status == "ok":
        if data.get("selection_required"):
            available = [item["id"] for item in data["providers"] if item["available"]]
            print("No provider was selected. Detected providers: {0}".format(", ".join(available) if available else "none"))
        elif "preview" in data:
            compatibility = data.get("plan", {}).get("compatibility", [])
            if compatibility:
                print("Provider compatibility:")
                for item in compatibility:
                    details = "{0}: {1}".format(item["id"], item["state"])
                    if item.get("version"):
                        details += " (version {0})".format(item["version"])
                    if item.get("error"):
                        details += " - {0}".format(item["error"])
                    print(details)
            print(data["preview"]["diff"])
            print("Confirmation required: rerun with --confirm --preview-hash PREVIEW_HASH.")
        else:
            print(json.dumps(data, indent=2))
    else:
        print("{0} failed [{1}]: {2}\nremediation: {3}".format(operation, error[0], error[1], error[2]))
    return 0 if status == "ok" else 2


def _run_installer(arguments):
    operation = arguments.command
    if operation == "doctor" and "models" in (getattr(arguments, "providers", None) or []):
        return _run_model_doctor(arguments)
    try:
        installer = Installer(_installer_registry(), project_root=arguments.project)
        detections = [item.to_dict() for item in installer.detected()]
        selected = tuple(getattr(arguments, "providers", ()) or ())
        if operation == "providers" or not selected:
            return _installer_result(operation, {
                "providers": detections,
                "selection_required": operation != "providers",
                "mutated": False,
            }, arguments.json)
        plan = installer.plan(operation, selected, arguments.scope, arguments.project)
        if arguments.dry_run or (operation != "doctor" and not arguments.confirm):
            return _installer_result(operation, {"plan": plan.to_dict(), "preview": plan.preview, "mutated": False}, arguments.json)
        if operation != "doctor" and not arguments.preview_hash:
            return _installer_result(operation, {"plan": plan.to_dict(), "preview": plan.preview}, arguments.json, "error", (
                "preview_hash_required", "--confirm requires the hash from an inspected preview.",
                "Run the command with --dry-run, inspect its preview, then pass --preview-hash.",
            ))
        if operation != "doctor" and arguments.preview_hash != plan.preview["preview_hash"]:
            return _installer_result(operation, {"plan": plan.to_dict(), "preview": plan.preview}, arguments.json, "error", (
                "preview_stale", "The supplied preview hash does not match the current plan.",
                "Generate and inspect a new preview before confirming this mutation.",
            ))
        result = installer.execute(plan, arguments.preview_hash if operation != "doctor" else False)
        data = result if isinstance(result, dict) else result.to_dict()
        return _installer_result(operation, {"result": data, "mutated": operation != "doctor"}, arguments.json)
    except InstallerConflictError as error:
        lock_code = "legacy_lock_busy" if str(error).startswith("legacy_lock_busy") else (
            "legacy_lock_recreated" if str(error).startswith("legacy_lock_recreated") else "artifact_conflict"
        )
        return _installer_result(operation, {}, arguments.json, "error", (
            lock_code, str(error),
            "Stop older mlx-agent processes and remove only recreated legacy locks." if lock_code != "artifact_conflict" else "Preserve the modified file or restore it to the receipt hash before retrying.",
        ))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _installer_result(operation, {}, arguments.json, "error", (
            "installer_failed", str(error), "Inspect the provider selection and preview, then correct the reported path or manifest issue.",
        ))


def legacy_scout_main(argv=None):
    parser = argparse.ArgumentParser(description="Discover MLX models on HuggingFace for this host.")
    _add_discovery_arguments(parser)
    return _run_discovery(parser.parse_args(argv), legacy=True)


def build_parser():
    parser = argparse.ArgumentParser(description="MLX agent command-line core.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    discover = subcommands.add_parser("discover", help="discover MLX models for this host")
    _add_discovery_arguments(discover)
    inspect_host = subcommands.add_parser("inspect-host", help="inspect local Apple Silicon and runtime inventory without model discovery")
    inspect_host.add_argument("--json", action="store_true")
    adopt = subcommands.add_parser("adopt", help="run or inspect resumable model adoption")
    _add_adoption_arguments(adopt)
    research = subcommands.add_parser("research", help="build a read-only domain research pack (markdown)")
    _add_research_arguments(research)
    blueprint = subcommands.add_parser(
        "blueprint",
        help="emit a read-only MLX project design pack (quant/train guidance; no scaffolding)",
    )
    _add_blueprint_arguments(blueprint)
    wire_command = subcommands.add_parser("wire", help="render, apply, inspect, or roll back runtime wiring")
    _add_wire_arguments(wire_command)
    bench_command = subcommands.add_parser("bench", help="measure a locally served model without downloading it")
    _add_bench_arguments(bench_command)
    serve_command = subcommands.add_parser("serve", help="preview and launch a local MLX server (confirmation-gated)")
    _add_serve_arguments(serve_command)
    fleet_command = subcommands.add_parser("fleet", help="render or apply a one-shot per-role router configuration")
    _add_fleet_arguments(fleet_command)
    watch_command = subcommands.add_parser("watch", help="snapshot and diff owned models against Hugging Face")
    _add_watch_arguments(watch_command)
    convert_command = subcommands.add_parser("convert", help="preview and quantize a cached model (confirmation-gated)")
    _add_convert_arguments(convert_command)
    lora_command = subcommands.add_parser("lora", help="preview and run LoRA training on a cached model (confirmation-gated)")
    _add_lora_arguments(lora_command)
    fuse_command = subcommands.add_parser("fuse", help="preview and fuse a LoRA adapter into its base (confirmation-gated)")
    _add_fuse_arguments(fuse_command)
    providers_command = subcommands.add_parser("providers", help="list detected supported provider CLIs")
    _add_installer_arguments(providers_command, include_providers=False)
    for name in ("install", "update", "uninstall", "doctor"):
        installer_command = subcommands.add_parser(name, help="{0} declared provider artifacts safely".format(name))
        if name == "doctor":
            _add_doctor_arguments(installer_command)
        else:
            _add_installer_arguments(installer_command)
    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "adopt":
        return _run_adoption(arguments)
    if arguments.command == "wire":
        return _run_wire(arguments)
    if arguments.command == "bench":
        return _run_bench(arguments)
    if arguments.command == "serve":
        return _run_serve(arguments)
    if arguments.command == "fleet":
        return _run_fleet(arguments)
    if arguments.command == "watch":
        return _run_watch(arguments)
    if arguments.command == "convert":
        return _run_convert(arguments)
    if arguments.command == "lora":
        return _run_lora(arguments)
    if arguments.command == "fuse":
        return _run_fuse(arguments)
    if arguments.command in {"providers", "install", "update", "uninstall", "doctor"}:
        return _run_installer(arguments)
    if arguments.command == "inspect-host":
        return _run_inspect_host(arguments)
    if arguments.command == "research":
        return _run_research(arguments)
    if arguments.command == "blueprint":
        return _run_blueprint(arguments)
    return _run_discovery(arguments, legacy=False)
