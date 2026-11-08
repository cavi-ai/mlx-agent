"""Confirmation-gated downloads of Hugging Face snapshots for intake."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .backends import (
    backend_environment, downloads_root, load_manifests, port_downloads, port_requirements, recipe_for,
    snapshot_files, spawn_with_env,
)
from .convert import _default_module_present, _write_receipt, receipts_root
from .intake_source import parse_hf_source, validate_file, validate_revision
from .serve import _pid_alive
from .source_evidence import completed_fetch, immutable_revision, safe_path
from . import fetch_storage

FETCH_RECEIPT_KIND = "fetch"
FETCH_RUNNER = Path(__file__).resolve().with_name("fetch_runner.py")
FETCH_MODULES = ("huggingface_hub",)
FETCH_IGNORE_PATTERNS = (
    "*.h5", "*.msgpack", "*.onnx", "*.onnx_data", "*.tflite", "*.ot", "*.mlmodel", "*.gguf", "*.mp4",
    "onnx/*", "coreml/*",
)
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class FetchError(RuntimeError):
    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def plan_fetch(text, revision=None, file=None, hf_cache=None, local_dir=None, python=None, model_type=None,
               manifests=None, downloads_dir=None, additional_files=None, storage_root=None):
    """A download plan. A snapshot of a ported model type (``model_type`` from intake) takes only the
    port's files; any other snapshot skips formats MLX never reads. A subfolder source takes only that folder."""
    source = parse_hf_source(text)
    if revision:
        source["revision"] = validate_revision(revision)
    if file is not None:
        source["file"] = validate_file(file, allow_support=True)
    local = None
    if local_dir is not None:
        local = Path(str(local_dir)).expanduser()
        if not local.is_absolute():
            raise FetchError("invalid_arguments", "--local-dir must be an absolute path.",
                             "Pass the full destination directory.")
    extra = sorted(additional_files or [])
    if extra and (not source["file"] or source.get("subfolder") or not immutable_revision(source["revision"]) or local is None or hf_cache is not None):
        raise FetchError("invalid_arguments", "Additional files require a primary file, immutable revision and new local destination without a shared cache.",
                         "Select the exact files at one commit and pass --local-dir.")
    if len(extra) > 64 or len(set(extra)) != len(extra) or source["file"] in extra:
        raise FetchError("invalid_arguments", "The additional file set is duplicated or exceeds its limit.", "Select each companion or shard once (at most 64).")
    for name in extra:
        validate_file(name, allow_support=True)
    fresh = local is not None and hf_cache is None and immutable_revision(source["revision"])
    if fresh and source["file"] and source.get("subfolder"):
        raise FetchError("invalid_arguments", "A managed file selection must name its repository-relative path without a separate subfolder.", "Use org/name with --file folder/name instead.")
    python = str(python or sys.executable)
    argv = [python, str(FETCH_RUNNER), "--repo", source["repo"], "--revision", source["revision"]]
    ignore_patterns, allow_patterns = [], []
    prefix = source["subfolder"] + "/" if source.get("subfolder") else ""
    ported = recipe = None
    requires, downloads = {}, []
    support_dir = None
    if not source["file"] and model_type:
        manifests = load_manifests() if manifests is None else manifests
        ported = snapshot_files(model_type, manifests)
        recipe = None if prefix else recipe_for(source["repo"], manifests)
        recipe = recipe if recipe and recipe["port"] == model_type else None
    if source["file"]:
        argv += ["--file", source["file"]]
        for name in extra:
            argv += ["--additional-file", name]
    elif recipe:
        # The recipe's model is its pinned base snapshot plus the LoRA this repository ships.
        allow_patterns = [recipe["lora"]]
        ignore_patterns = list(FETCH_IGNORE_PATTERNS)
        argv += ["--allow", recipe["lora"]]
        if not fresh:
            argv += ["--base-repo", recipe["base"], "--base-revision", recipe["base_revision"]]
            for pattern in ignore_patterns:
                argv += ["--base-ignore", pattern]
    else:
        if ported:
            allow_patterns = [prefix + name for name in ported]
        elif prefix:
            allow_patterns = [prefix + "*"]
        if not ported:
            ignore_patterns = list(FETCH_IGNORE_PATTERNS)
        for name in allow_patterns:
            argv += ["--allow", name]
        for pattern in ignore_patterns:
            argv += ["--ignore", pattern]
        requires = port_requirements(model_type, manifests) if ported and not prefix else {}
        if not fresh:
            for repo, names in sorted(requires.items()):
                argv += [item for name in names for item in ("--require", repo + "=" + name)]
        downloads = port_downloads(model_type, manifests) if ported and not prefix else []
        for entry in downloads:
            argv += ["--download", "{0}={1}={2}".format(entry["sha256"], entry["size"], entry["url"])]
        if downloads:
            if fresh:
                support_dir = safe_path(downloads_dir) if downloads_dir is not None else safe_path(local) / "support"
                if support_dir == safe_path(local) or not support_dir.is_relative_to(safe_path(local)):
                    raise FetchError("invalid_arguments", "Managed support files must stay beneath the reviewed local destination.", "Choose a support folder inside --local-dir.")
            else:
                support_dir = Path(downloads_dir) if downloads_dir is not None else downloads_root()
            argv += ["--downloads-dir", str(support_dir)]
    if hf_cache:
        argv += ["--cache-dir", str(hf_cache)]
    if local is not None:
        argv += ["--local-dir", str(local)]
    if fresh:
        argv += ["--fresh-destination", "--capture-identities"]
        if source.get("subfolder"):
            argv += ["--source-subfolder", source["subfolder"]]
    slug = _UNSAFE.sub("-", "{0}--{1}".format(
        source["repo"].replace("/", "--"), source["file"] or (source.get("subfolder") or "snapshot").replace("/", "--")
    )).strip("-.")[:160]
    plan = {
        "repo": source["repo"], "revision": source["revision"], "file": source["file"],
        "additional_files": extra, "fresh_destination": fresh,
        "downloads_dir": str(support_dir) if support_dir is not None else None,
        "subfolder": source.get("subfolder"),
        "base": {"repo": recipe["base"], "revision": recipe["base_revision"]} if recipe else None,
        "cache_dir": str(hf_cache) if hf_cache else None,
        "local_dir": str(local) if local is not None else None,
        "ignore_patterns": ignore_patterns,
        "allow_patterns": allow_patterns,
        "slug": slug, "argv": argv, "network": ["huggingface.co"],
    }
    dependencies = {repo: {"repo": repo, "revision": None, "files": list(names)} for repo, names in sorted(requires.items())}
    if recipe:
        dependencies[recipe["base"]] = {"repo": recipe["base"], "revision": recipe["base_revision"], "files": []}
    plan["required_dependencies"] = list(dependencies.values())
    if requires:
        plan["requires"] = {repo: list(names) for repo, names in sorted(requires.items())}
    if downloads:
        plan["downloads"] = [dict(entry) for entry in downloads]
        plan["network"] = plan["network"] + sorted({entry["url"].split("/")[2] for entry in downloads})
    if storage_root is not None:
        if not fresh:
            raise FetchError("invalid_arguments", "Storage admission requires an immutable managed destination.",
                             "Resolve the commit and pass --local-dir without --hf-cache.")
        try:
            storage = fetch_storage.inspect(plan, storage_root)
        except fetch_storage.StorageError as error:
            raise FetchError(error.code, str(error), "Inspect the registered source receipt; preserve existing files.") from error
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise FetchError("storage_invalid", str(error), "Repair the storage index without replacing model files.") from error
        if storage.get("local_dir"):
            previous = plan["local_dir"]
            plan["local_dir"] = storage["local_dir"]
            argv[argv.index("--local-dir") + 1] = storage["local_dir"]
            if support_dir is not None:
                plan["downloads_dir"] = str(Path(storage["local_dir"]) / Path(support_dir).relative_to(previous))
                argv[argv.index("--downloads-dir") + 1] = plan["downloads_dir"]
        plan["storage"] = storage
        if storage["mode"] == "reuse":
            plan["argv"] = []
            plan["network"] = []
        if storage["mode"] == "download":
            from .convert import cached_snapshot
            cached = cached_snapshot(source["repo"], revision=source["revision"])
            if cached is not None:
                raise FetchError("source_already_cached", "This revision already has a Hugging Face cache snapshot at {0}.".format(cached),
                                 "Scan or register that cache and use its existing source; inspect incomplete files before retrying.")
    # Approval binds the source, recipe and filesystem/network action. Admission
    # state is independently rechecked under the lock and must not invalidate a
    # saved source approval merely because its own first launch registered it.
    approval = {key: value for key, value in plan.items() if key != "storage"}
    if plan.get("storage", {}).get("mode") == "reuse":
        approval["reuse"] = {key: plan["storage"][key] for key in ("receipt", "fetch_preview_hash")}
    canonical = json.dumps(approval, sort_keys=True, separators=(",", ":"))
    plan["preview_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def start_fetch(plan, receipts_dir=None, confirm=False, preview_hash=None, spawn=None,
                pid_alive=_pid_alive, module_present=None, now=_utc_now, env=None):
    if not confirm:
        return {"status": "preview", "plan": plan, "requires_confirmation": True}
    if not preview_hash:
        raise FetchError("preview_hash_required", "--confirm requires the hash from a reviewed fetch preview.",
                         "Run intake fetch without --confirm, review it, then pass --preview-hash.")
    if preview_hash != plan["preview_hash"]:
        raise FetchError("preview_stale", "The supplied preview hash does not match this fetch plan.",
                         "Re-run intake fetch without --confirm and review the fresh plan.")
    if plan.get("storage"):
        try:
            with fetch_storage.admission(plan) as index:
                if plan["storage"]["mode"] == "reuse":
                    return fetch_storage.reuse_outcome(plan)
                return _launch_fetch(plan, receipts_dir, spawn, pid_alive, module_present, now, env, index)
        except fetch_storage.StorageError as error:
            raise FetchError(error.code, str(error), "Review a fresh preview before retrying; preserve the registered source.") from error
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise FetchError("storage_invalid", str(error), "Inspect the storage index and source receipt before retrying.") from error
    return _launch_fetch(plan, receipts_dir, spawn, pid_alive, module_present, now, env)


def _launch_fetch(plan, receipts_dir, spawn, pid_alive, module_present, now, env, index=None):
    if index is not None:
        from .convert import cached_snapshot
        if cached_snapshot(plan["repo"], revision=plan["revision"]) is not None:
            raise FetchError("source_already_cached", "This revision now exists in the Hugging Face cache.", "Use the existing cache source instead of creating another copy.")
    missing = (module_present or _default_module_present)(FETCH_MODULES)
    if missing:
        raise FetchError("runtime_not_installed",
                         "Downloads need {0} in this interpreter.".format(", ".join(missing)),
                         "Run mlx-agent from the workbench .venv; fetch never installs runtimes.")
    if plan.get("fresh_destination"):
        destination = safe_path(plan["local_dir"])
        if destination.exists():
            raise FetchError("destination_exists", "The managed download destination already exists.", "Choose a fresh destination; existing files are never replaced.")
    root = receipts_root(receipts_dir, kind=FETCH_RECEIPT_KIND)
    slug = plan["slug"] + ("-" + uuid.uuid4().hex if plan.get("fresh_destination") else "")
    receipt_path = root / "{0}.json".format(slug)
    existing = _read_json(receipt_path)
    if existing and not existing.get("completed_at") and isinstance(existing.get("pid"), int) \
            and pid_alive(existing["pid"]) and not Path(existing.get("marker", "")).exists():
        raise FetchError("job_in_progress", "A download of {0} is still running.".format(plan["repo"]),
                         "Wait for it to finish (intake status).")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "{0}.done.json".format(slug)
    if marker.exists():
        marker.unlink()
    log_path = root / "{0}.log".format(slug)
    started_at = now()
    proof_token = uuid.uuid4().hex if plan.get("fresh_destination") else None
    argv = list(plan["argv"]) + ["--marker", str(marker)]
    if proof_token:
        argv += ["--proof-token", proof_token, "--preview-hash", plan["preview_hash"], "--started-at", started_at]
    receipt = {
        "schema_version": "1.0", "kind": FETCH_RECEIPT_KIND, "repo": plan["repo"],
        "revision": plan["revision"], "file": plan["file"], "subfolder": plan.get("subfolder"),
        "additional_files": plan.get("additional_files", []), "fresh_destination": plan.get("fresh_destination", False), "proof_token": proof_token,
        "downloads_dir": plan.get("downloads_dir"), "downloads": plan.get("downloads", []),
        "required_dependencies": plan.get("required_dependencies", []),
        "local_dir": plan["local_dir"], "cache_dir": plan["cache_dir"], "slug": slug, "argv": argv, "pid": None,
        "log_path": str(log_path), "marker": str(marker), "started_at": started_at,
        "preview_hash": plan["preview_hash"], "completed_at": None, "exit_status": None,
    }
    for key in ("allow_patterns", "ignore_patterns", "base", "requires"):
        if key in plan:
            receipt[key] = plan[key]
    if index is not None:
        # Publish intent before launching. A crash cannot lose the only pointer
        # to a potentially running worker and admit a duplicate source.
        _write_receipt(root, receipt, receipt_path.name)
        fetch_storage.register(index, plan, receipt_path)
    receipt["pid"] = (spawn or spawn_with_env)(argv, str(log_path), backend_environment(env))
    _write_receipt(root, receipt, receipt_path.name)
    return {"status": "started", "receipt": receipt}


def status_fetch(receipts_dir=None, pid_alive=_pid_alive):
    root = receipts_root(receipts_dir, kind=FETCH_RECEIPT_KIND)
    entries = []
    if not root.is_dir():
        return entries
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".done.json"):
            continue
        receipt = _read_json(path)
        if not receipt or receipt.get("kind") != FETCH_RECEIPT_KIND or not isinstance(receipt.get("pid"), int):
            continue
        entry = {
            "receipt": str(path), "repo": receipt.get("repo"), "revision": receipt.get("revision"),
            "file": receipt.get("file"), "subfolder": receipt.get("subfolder"), "local_dir": receipt.get("local_dir"),
            "additional_files": receipt.get("additional_files", []), "preview_hash": receipt.get("preview_hash"), "files": [],
            "downloads_dir": receipt.get("downloads_dir"), "downloads": receipt.get("downloads", []),
            "required_dependencies": receipt.get("required_dependencies", []),
            "state": "failed",
            "processes_quiescent": not pid_alive(receipt["pid"]),
            "path": None, "log_path": receipt.get("log_path"), "started_at": receipt.get("started_at"),
            "completed_at": None,
        }
        marker = _read_json(receipt.get("marker", ""))
        if marker:
            bound = not receipt.get("fresh_destination") or all(marker.get(key) == receipt.get(key) for key in (
                "repo", "revision", "file", "subfolder", "additional_files", "local_dir", "preview_hash", "proof_token", "started_at", "downloads_dir", "downloads"))
            if receipt.get("fresh_destination") and bound and marker.get("exit_status") == "done":
                try:
                    completed_fetch(path)
                except (OSError, ValueError, KeyError, TypeError):
                    bound = False
            entry["state"] = "done" if marker.get("exit_status") == "done" and bound else "failed"
            entry["files"] = marker.get("files", []) if bound else []
            entry["path"] = marker.get("path") if bound else None
            entry["completed_at"] = marker.get("finished_at")
        elif pid_alive(receipt["pid"]):
            entry["state"] = "running"
        entries.append(entry)
    return entries
