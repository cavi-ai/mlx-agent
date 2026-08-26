"""Read-only model inventory, wiring drift, and endpoint health diagnostics.

Doctor never deletes, moves, repairs, downloads, or starts anything. It reads
the Hugging Face cache, queries already-running loopback runtimes, parses
Wire-managed configuration, and reports classified findings with remediation
text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .verification import _safe_error, installed_model_ids
from .wiring import (
    _OLLAMA,
    _STATE_CHANGING_PATH_PARTS,
    ConfigAdapter,
    validate_health_endpoint,
)


MAX_INVENTORY_ENTRIES = 500
MAX_FINDINGS = 100
MAX_ENDPOINTS = 50
MAX_WALK_FILES = 2000
MAX_CONFIG_BYTES = 1024 * 1024
MAX_RECEIPTS = 50
_WIRE_MARKER = "MLX_AGENT_WIRE"
_SKIP_DIRECTORIES = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "Library",
}
_CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".modelfile"}
_HEALTH_TIMEOUT = 2.0


def default_hf_cache():
    return Path.home() / ".cache" / "huggingface" / "hub"


def inspect_hf_cache(cache_dir):
    """Summarize model repos in a Hugging Face cache without reading weights."""
    root = Path(cache_dir)
    inventory = []
    if not root.is_dir():
        return inventory
    for entry in sorted(root.iterdir()):
        if len(inventory) >= MAX_INVENTORY_ENTRIES:
            break
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        repo = entry.name[len("models--"):].replace("--", "/")
        refs_main = _read_text_quiet(entry / "refs" / "main")
        snapshots = entry / "snapshots"
        revisions = []
        total_bytes = 0
        missing_blobs = 0
        seen_blobs = set()
        if snapshots.is_dir():
            for revision in sorted(snapshots.iterdir()):
                if not revision.is_dir():
                    continue
                revisions.append(revision.name)
                for dirpath, dirnames, filenames in os.walk(revision):
                    dirnames[:] = [name for name in dirnames if not name.startswith(".")]
                    for filename in filenames:
                        location = Path(dirpath) / filename
                        try:
                            resolved = location.resolve(strict=True)
                        except OSError:
                            missing_blobs += 1
                            continue
                        if resolved in seen_blobs:
                            continue
                        seen_blobs.add(resolved)
                        try:
                            total_bytes += resolved.stat().st_size
                        except OSError:
                            missing_blobs += 1
        complete = bool(refs_main and revisions and missing_blobs == 0)
        inventory.append({
            "id": repo,
            "source": "hf-cache",
            "bytes": total_bytes,
            "revisions": revisions[:8],
            "complete": complete,
        })
    return inventory


def _read_text_quiet(path):
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def collect_runtime_inventory(runtime_clients):
    """Collect served model IDs from already-running loopback runtimes."""
    inventory = []
    errors = []
    for runtime in runtime_clients:
        name = str(getattr(runtime, "name", runtime.__class__.__name__))
        try:
            response = runtime.list_models()
            for model_id in sorted(installed_model_ids(response)):
                if len(inventory) >= MAX_INVENTORY_ENTRIES:
                    break
                inventory.append({
                    "id": model_id,
                    "source": name,
                    "bytes": _runtime_model_size(response, model_id),
                    "complete": True,
                })
        except Exception as error:
            errors.append("{0}: {1}".format(name, _safe_error(error)))
    return inventory, errors


def _runtime_model_size(response, model_id):
    if not isinstance(response, dict):
        return None
    for record in response.get("models", []) or []:
        if not isinstance(record, dict):
            continue
        if str(record.get("name") or record.get("id") or "") != model_id:
            continue
        size = record.get("size")
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            return size
    return None


def scan_wired_configs(roots):
    """Find Wire-managed configs under roots and extract their references."""
    wired = []
    for root in roots:
        for location in _walk_config_files(root):
            if len(wired) >= MAX_ENDPOINTS:
                return wired
            content = _read_bounded(location)
            if content is None or _WIRE_MARKER not in content:
                continue
            try:
                adapter = ConfigAdapter.detect(location)
                reference = adapter_references(adapter, content)
            except (TypeError, ValueError):
                continue
            if reference is None:
                continue
            wired.append({
                "path": str(location),
                "runtime": reference.get("runtime") or adapter.runtime,
                "model": reference.get("model"),
                "endpoint": reference.get("endpoint"),
            })
    return wired


def _walk_config_files(root):
    root = Path(root)
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    yielded = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in _SKIP_DIRECTORIES and not name.startswith(".mlx-agent-receipts")
        ]
        for filename in sorted(filenames):
            if yielded >= MAX_WALK_FILES:
                return
            location = Path(dirpath) / filename
            if location.suffix.lower() not in _CONFIG_SUFFIXES and filename.lower() != "modelfile":
                continue
            yielded += 1
            yield location


def _read_bounded(location):
    try:
        if location.stat().st_size > MAX_CONFIG_BYTES:
            return None
        return location.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def adapter_references(adapter, content):
    """Extract the model and endpoint a Wire-managed config references."""
    if adapter.runtime == "ollama":
        match = _OLLAMA.fullmatch(content)
        if match is None:
            return None
        return {"model": match.group(1), "endpoint": None}
    if adapter.runtime == "litellm":
        model = re.search(r"(?m)^      model: openai/([^\s]+)$", content)
        base = re.search(r"(?m)^      api_base: (http[^\s]+)$", content)
        if model is None:
            return None
        return {
            "model": model.group(1),
            "endpoint": base.group(1) if base else None,
        }
    try:
        document = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    marker = document.get("mlx_agent_wire")
    if not isinstance(marker, dict):
        return None
    provider = marker.get("provider") if isinstance(marker.get("provider"), dict) else {}
    runtime = marker.get("runtime")
    return {
        "model": marker.get("model"),
        "endpoint": provider.get("base_url"),
        "runtime": runtime if isinstance(runtime, str) else None,
    }


def scan_receipt_after_hashes(roots):
    """Read after_hashes from Wire receipts without enforcing receipt validity."""
    records = []
    for root in roots:
        root = Path(root)
        candidates = []
        if (root / ".mlx-agent-receipts").is_dir():
            candidates.append(root / ".mlx-agent-receipts")
        else:
            for dirpath, dirnames, _filenames in os.walk(root):
                if ".mlx-agent-receipts" in dirnames:
                    candidates.append(Path(dirpath) / ".mlx-agent-receipts")
                    dirnames.remove(".mlx-agent-receipts")
        for receipts_dir in candidates:
            for receipt in sorted(receipts_dir.glob("*/receipt.json")):
                if len(records) >= MAX_RECEIPTS:
                    return records
                try:
                    value = json.loads(receipt.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                after_hashes = value.get("after_hashes")
                if not isinstance(after_hashes, dict):
                    continue
                records.append({
                    "receipt": str(receipt),
                    "transaction_id": value.get("transaction_id"),
                    "status": value.get("status"),
                    "after_hashes": {
                        str(key): str(item) for key, item in after_hashes.items()
                    },
                })
    return records


def check_endpoint(endpoint, timeout=_HEALTH_TIMEOUT):
    """Classify a wired loopback endpoint as ok, reachable, or down."""
    if not endpoint:
        return None
    try:
        url = validate_health_endpoint(endpoint)
        parsed_path = urllib.parse.urlparse(url).path.rstrip("/")
        if parsed_path.endswith("/v1"):
            url = url.rstrip("/") + "/models"
        request = urllib.request.Request(url, method="GET")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    except (OSError, ValueError, urllib.error.URLError):
        return "down"
    return "ok" if 200 <= status < 400 else "reachable"


def run_model_doctor(wired_roots, hf_cache, runtime_clients):
    """Run every read-only check and return the doctor report."""
    wired_roots = [Path(root) for root in wired_roots]
    inventory = inspect_hf_cache(hf_cache)
    runtime_inventory, runtime_errors = collect_runtime_inventory(runtime_clients)
    inventory.extend(runtime_inventory)
    known_ids = {item["id"].casefold() for item in inventory}

    wired = scan_wired_configs(wired_roots)
    receipts = scan_receipt_after_hashes(wired_roots)

    findings = []
    for config in wired:
        model = config.get("model")
        if model and model.casefold() not in known_ids:
            findings.append({
                "code": "drift_missing_model",
                "path": config["path"],
                "model": model,
                "remediation": (
                    "Re-download the model into a local inventory, or re-run "
                    "wire render for a model that is present."
                ),
            })

    for receipt in receipts:
        for target, expected in receipt["after_hashes"].items():
            location = Path(target)
            if not location.is_file():
                findings.append({
                    "code": "drift_missing_file",
                    "path": target,
                    "receipt_id": receipt["transaction_id"],
                    "remediation": (
                        "The wired file is gone; restore it from the receipt "
                        "backup or roll the receipt back."
                    ),
                })
                continue
            try:
                digest = hashlib.sha256(location.read_bytes()).hexdigest()
            except OSError:
                continue
            if digest != expected:
                findings.append({
                    "code": "drift_hash_mismatch",
                    "path": target,
                    "receipt_id": receipt["transaction_id"],
                    "remediation": (
                        "The wired file changed after wiring; inspect it, then "
                        "re-apply or roll back the receipt."
                    ),
                })

    endpoints = []
    claims = {}
    for config in wired:
        endpoint = config.get("endpoint")
        if not endpoint:
            continue
        origin = _endpoint_origin_key(endpoint)
        if origin is None:
            continue
        previous = claims.get(origin)
        if previous is not None and previous["runtime"] != config["runtime"]:
            findings.append({
                "code": "drift_endpoint_conflict",
                "path": config["path"],
                "model": config.get("model"),
                "remediation": (
                    "Two wired configs for different runtimes claim {0}; "
                    "move one runtime to its own port.".format(origin)
                ),
            })
        else:
            claims.setdefault(origin, config)
        if len(endpoints) < MAX_ENDPOINTS:
            endpoints.append({
                "endpoint": origin,
                "path": config["path"],
                "status": check_endpoint(endpoint),
            })

    for item in inventory:
        if item["source"] == "hf-cache" and not item.get("complete", True):
            findings.append({
                "code": "hf_cache_incomplete",
                "path": None,
                "model": item["id"],
                "remediation": (
                    "The cached snapshot is incomplete; re-download the model "
                    "with the runtime's own pull command."
                ),
            })

    findings = findings[:MAX_FINDINGS]
    total_bytes = sum(
        item["bytes"] or 0 for item in inventory if item["source"] == "hf-cache"
    )
    return {
        "inventory": inventory,
        "summary": {
            "models": len(inventory),
            "hf_cache_bytes": total_bytes,
            "wired_configs": len(wired),
            "findings": len(findings),
        },
        "wired": wired,
        "endpoints": endpoints,
        "findings": findings,
        "runtime_errors": runtime_errors,
    }


def _endpoint_origin_key(endpoint):
    try:
        url = validate_health_endpoint(endpoint)
    except ValueError:
        return None
    parsed = urllib.parse.urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    decoded = parsed.path or "/"
    parts = {part for part in decoded.lower().split("/") if part}
    if parts & _STATE_CHANGING_PATH_PARTS:
        return None
    return "{0}://{1}:{2}".format(parsed.scheme.lower(), parsed.hostname.lower(), port)


PRUNE_PREVIEW_VERSION = "1.0"
MAX_PRUNE_CANDIDATES = 100


class PruneError(RuntimeError):
    """Classified prune failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def plan_prune(cache_dir):
    """List incomplete HF cache snapshots eligible for deletion; read-only."""
    root = Path(cache_dir)
    candidates = []
    for item in inspect_hf_cache(root):
        if item.get("complete", True):
            continue
        location = root / "models--{0}".format(item["id"].replace("/", "--"))
        candidates.append({
            "repo": item["id"],
            "path": str(location),
            "bytes": item.get("bytes") or 0,
        })
        if len(candidates) >= MAX_PRUNE_CANDIDATES:
            break
    canonical = json.dumps(candidates, sort_keys=True, separators=(",", ":"))
    return {
        "cache": str(root),
        "candidates": candidates,
        "total_bytes": sum(item["bytes"] for item in candidates),
        "preview_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def execute_prune(plan, confirm=False, preview_hash=None, remover=None):
    """Delete exactly the reviewed candidates; irreversible by design."""
    if not confirm:
        return {"status": "preview", "plan": plan, "requires_confirmation": True}
    if not preview_hash:
        raise PruneError(
            "preview_hash_required",
            "--confirm requires the hash from a reviewed prune preview.",
            "Run doctor models --prune without --confirm, inspect the plan, then pass --preview-hash.",
        )
    if preview_hash != plan["preview_hash"]:
        raise PruneError(
            "preview_stale",
            "The supplied preview hash does not match the current prune plan.",
            "Re-run doctor models --prune and review the fresh candidate list.",
        )
    if remover is None:
        import shutil

        def remover(path):
            shutil.rmtree(path)

    cache = Path(plan["cache"]).resolve()
    removed = []
    for candidate in plan["candidates"]:
        location = Path(candidate["path"])
        resolved = location.resolve()
        if resolved == cache or cache not in resolved.parents:
            raise PruneError(
                "unsafe_target",
                "Refusing to remove a path outside the cache: {0}".format(candidate["path"]),
                "Only cache-owned model directories are ever pruned.",
            )
        if not location.is_dir():
            removed.append({"repo": candidate["repo"], "removed": False, "reason": "already gone"})
            continue
        remover(str(location))
        removed.append({
            "repo": candidate["repo"],
            "removed": not location.exists(),
            "bytes": candidate["bytes"],
        })
    return {"status": "pruned", "removed": removed}
