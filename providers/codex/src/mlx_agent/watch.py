"""Stateful Hugging Face diff digest focused on models you own."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .model_doctor import (
    collect_runtime_inventory,
    default_hf_cache,
    inspect_hf_cache,
    scan_wired_configs,
)
from .intake_source import IntakeSourceError, parse_hf_source
from .models import base_name
from .transactions import _atomic_in_directory


WATCH_SCHEMA_VERSION = "1.0"
STATE_FILENAME = "watch-state.json"
MAX_OWNED = 500
MAX_CANDIDATES = 2000
MAX_FINDINGS = 100


class WatchError(RuntimeError):
    """Classified watch failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def state_file(state_dir):
    return Path(state_dir) / "watch" / STATE_FILENAME


def validate_owned_repos(repos):
    """Explicit ownership accepts canonical repo IDs, never URLs or paths."""
    validated = {}
    for repo in repos or []:
        try:
            source = parse_hf_source(repo)
            if source["repo"] != repo or source.get("file") or source.get("subfolder"):
                raise IntakeSourceError("Expected an org/name repository ID.")
        except (IntakeSourceError, TypeError) as error:
            raise WatchError("invalid_owned_repo", "Invalid owned repository: {0!r}".format(repo),
                             "Pass --owned-repo as a Hugging Face org/name ID.") from error
        validated.setdefault(repo.casefold(), repo)
    if len(validated) > MAX_OWNED:
        raise WatchError("too_many_owned_repos", "Too many explicit owned repositories.",
                         "Watch accepts at most {0} owned repositories.".format(MAX_OWNED))
    return list(validated.values())


def _scope_path(path):
    try:
        return str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError) as error:
        raise WatchError("invalid_scope", "An ownership path could not be resolved: {0}".format(path),
                         "Choose an accessible client path or Hugging Face cache.") from error


def collect_owned(runtime_clients, hf_cache=None, wired_roots=None, *, include_hf_cache=True,
                  owned_repos=None, evidence=None):
    """Collect only selected inventories, preserving which sources were checked."""
    explicit = validate_owned_repos(owned_repos)
    roots = list(wired_roots) if wired_roots is not None else [Path.cwd()]
    cache = Path(hf_cache) if hf_cache is not None else (default_hf_cache() if include_hf_cache else None)
    runtime_clients = list(runtime_clients)
    runtime_sources = {str(getattr(client, "name", client.__class__.__name__)) for client in runtime_clients}
    scope = {"version": 1, "wired_roots": sorted({_scope_path(root) for root in roots}),
             "hf_cache": _scope_path(cache) if include_hf_cache else None,
             "owned_repos": sorted(repo.casefold() for repo in explicit),
             "runtime_sources": sorted(runtime_sources)}
    checked = {"managed"}
    # The shared scanner deliberately skips unreadable, invalid and oversized
    # configs. It proves presence, but cannot prove a wired model is absent.
    unchecked = {"wired"} if roots else set()
    owned = []
    seen = set()
    truncated = False

    def add(model_id, source):
        nonlocal truncated
        if not isinstance(model_id, str) or not model_id:
            return
        key = model_id.casefold()
        if key in seen:
            return
        if len(owned) >= MAX_OWNED:
            truncated = True
            return
        seen.add(key)
        owned.append({"id": model_id, "source": source})

    for repo in explicit:
        add(repo, "managed")
    if include_hf_cache:
        try:
            cache_inventory = inspect_hf_cache(cache)
            if cache.is_dir() and len(cache_inventory) < MAX_OWNED:
                checked.add("hf-cache")
            else:
                unchecked.add("hf-cache")
            for item in cache_inventory:
                add(item["id"], "hf-cache")
        except OSError:
            unchecked.add("hf-cache")
    runtime_inventory, errors = collect_runtime_inventory(runtime_clients)
    failed = {name for name in runtime_sources if any(error.startswith(name + ":") for error in errors)}
    if len(runtime_inventory) >= MAX_OWNED:
        failed = runtime_sources
    checked.update(runtime_sources - failed)
    unchecked.update(failed)
    for item in runtime_inventory:
        add(item["id"], item["source"])
    for config in scan_wired_configs([Path(root) for root in scope["wired_roots"]]):
        add(config.get("model"), "wired")
    if truncated:
        unchecked.update(checked)
        checked.clear()
    if evidence is not None:
        evidence.update(scope=scope, ownership_checks={"checked_sources": sorted(checked),
                                                     "unchecked_sources": sorted(unchecked)})
    return owned


def snapshot_candidates(discovery_data):
    """Flatten a discovery result into a bounded repo -> signal mapping."""
    candidates = {}
    roles = discovery_data.get("roles") if isinstance(discovery_data, dict) else None
    if not isinstance(roles, dict):
        return candidates
    for records in roles.values():
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            repo = record.get("repo")
            if not isinstance(repo, str) or not repo or repo in candidates:
                continue
            facts = record.get("facts") if isinstance(record.get("facts"), dict) else {}
            candidates[repo] = {
                "base": record.get("base") or base_name(repo),
                "weight_bytes": facts.get("weight_bytes"),
                "gated": facts.get("gated"),
                "license": record.get("license"),
            }
            if len(candidates) >= MAX_CANDIDATES:
                return candidates
    return candidates


def build_snapshot(owned, candidates, now=None, scope=None, ownership_checks=None):
    snapshot = {
        "schema_version": WATCH_SCHEMA_VERSION,
        "created_at": now or _utc_now(),
        "owned": list(owned)[:MAX_OWNED],
        "candidates": dict(candidates),
        "previous": None,
    }
    if scope is not None:
        snapshot["scope"] = scope
    if ownership_checks is not None:
        snapshot["ownership_checks"] = ownership_checks
    return snapshot


def write_snapshot(state_dir, snapshot):
    """Persist the snapshot, rotating any existing one into `previous`."""
    destination = state_file(state_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = None
    if destination.is_file():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            existing = None
    if isinstance(existing, dict) and existing.get("schema_version") == WATCH_SCHEMA_VERSION:
        snapshot["previous"] = {
            key: existing.get(key)
            for key in ("schema_version", "created_at", "owned", "candidates", "scope", "ownership_checks")
            if key in existing
        }
        snapshot["previous"]["previous"] = None
    content = (json.dumps(snapshot, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    _atomic_in_directory(destination.parent, destination.name, content, 0o600)
    return destination


def read_baseline(state_dir):
    destination = state_file(state_dir)
    if not destination.is_file():
        raise WatchError(
            "missing_baseline",
            "No watch snapshot exists yet.",
            "Run mlx-agent watch snapshot first to record a baseline.",
        )
    try:
        value = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise WatchError(
            "invalid_state",
            "The watch state file could not be read: {0}".format(error),
            "Remove the corrupted file and run watch snapshot again.",
        )
    if not isinstance(value, dict) or value.get("schema_version") != WATCH_SCHEMA_VERSION:
        raise WatchError(
            "invalid_state",
            "The watch state file has an unsupported schema.",
            "Run watch snapshot to record a fresh baseline.",
        )
    return value


def _can_infer_missing(baseline, current, source):
    scope = current.get("scope")
    required = {"version", "wired_roots", "hf_cache", "owned_repos", "runtime_sources"}
    if (not isinstance(scope, dict) or not required.issubset(scope)
            or type(scope.get("version")) is not int or scope["version"] != 1
            or baseline.get("scope") != scope):
        return False
    for snapshot in (baseline, current):
        checks = snapshot.get("ownership_checks")
        if not isinstance(checks, dict):
            return False
        checked, unchecked = checks.get("checked_sources"), checks.get("unchecked_sources")
        if (not isinstance(checked, list) or not isinstance(unchecked, list)
                or source not in checked or source in unchecked):
            return False
    return True


def diff_snapshots(baseline, current):
    """Classify changes relevant to owned models between two snapshots."""
    findings = []
    owned_ids = {item["id"].casefold() for item in current.get("owned", [])}
    owned_bases = {base_name(item["id"]) for item in current.get("owned", [])}
    baseline_candidates = baseline.get("candidates") or {}

    for repo, info in sorted((current.get("candidates") or {}).items()):
        if len(findings) >= MAX_FINDINGS:
            break
        base = base_name(info.get("base") or repo)
        if repo.casefold() not in owned_ids and base_name(repo) not in owned_bases and base not in owned_bases:
            continue
        previous = baseline_candidates.get(repo)
        if previous is None:
            findings.append({
                "code": "new_quant_of_owned",
                "repo": repo,
                "base": base,
                "detail": "new tracked repository matching an owned model",
            })
            continue
        old_bytes = previous.get("weight_bytes")
        new_bytes = info.get("weight_bytes")
        if old_bytes is not None and new_bytes is not None and old_bytes != new_bytes:
            findings.append({
                "code": "updated_tracked_repo",
                "repo": repo,
                "base": base,
                "detail": "weight bytes changed from {0} to {1}".format(old_bytes, new_bytes),
            })
        # "unknown" (or a missing value) means the Hub did not report access; that is not a change.
        if previous.get("gated") != info.get("gated") and not {previous.get("gated"), info.get("gated")} & {None, "unknown"}:
            findings.append({
                "code": "gated_changed",
                "repo": repo,
                "base": base,
                "detail": "gated changed from {0} to {1}".format(
                    previous.get("gated"), info.get("gated")
                ),
            })

    for item in baseline.get("owned", []):
        if len(findings) >= MAX_FINDINGS:
            break
        if item["id"].casefold() not in owned_ids and _can_infer_missing(baseline, current, item.get("source")):
            findings.append({
                "code": "owned_missing",
                "repo": item["id"],
                "base": base_name(item["id"]),
                "detail": "previously owned via {0}; no longer present".format(
                    item.get("source", "unknown")
                ),
            })
    return findings
