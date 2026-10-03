"""Confirmation-gated downloads of Hugging Face snapshots for intake."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .backends import backend_environment, load_manifests, snapshot_files, spawn_with_env
from .convert import _default_module_present, _write_receipt, receipts_root
from .intake_source import parse_hf_source, validate_file, validate_revision
from .serve import _pid_alive

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
               manifests=None):
    """A download plan. A snapshot of a ported model type (``model_type`` from intake) takes only the
    port's files; any other snapshot skips formats MLX never reads. A subfolder source takes only that folder."""
    source = parse_hf_source(text)
    if revision:
        source["revision"] = validate_revision(revision)
    if file is not None:
        source["file"] = validate_file(file)
    local = None
    if local_dir is not None:
        local = Path(str(local_dir)).expanduser()
        if not local.is_absolute():
            raise FetchError("invalid_arguments", "--local-dir must be an absolute path.",
                             "Pass the full destination directory.")
    python = str(python or sys.executable)
    argv = [python, str(FETCH_RUNNER), "--repo", source["repo"], "--revision", source["revision"]]
    ignore_patterns, allow_patterns = [], []
    prefix = source["subfolder"] + "/" if source.get("subfolder") else ""
    ported = None
    if not source["file"] and model_type:
        ported = snapshot_files(model_type, load_manifests() if manifests is None else manifests)
    if source["file"]:
        argv += ["--file", source["file"]]
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
    if hf_cache:
        argv += ["--cache-dir", str(hf_cache)]
    if local is not None:
        argv += ["--local-dir", str(local)]
    slug = _UNSAFE.sub("-", "{0}--{1}".format(
        source["repo"].replace("/", "--"), source["file"] or (source.get("subfolder") or "snapshot").replace("/", "--")
    )).strip("-.")[:160]
    plan = {
        "repo": source["repo"], "revision": source["revision"], "file": source["file"],
        "subfolder": source.get("subfolder"),
        "cache_dir": str(hf_cache) if hf_cache else None,
        "local_dir": str(local) if local is not None else None,
        "ignore_patterns": ignore_patterns,
        "allow_patterns": allow_patterns,
        "slug": slug, "argv": argv, "network": ["huggingface.co"],
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
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
    missing = (module_present or _default_module_present)(FETCH_MODULES)
    if missing:
        raise FetchError("runtime_not_installed",
                         "Downloads need {0} in this interpreter.".format(", ".join(missing)),
                         "Run mlx-agent from the workbench .venv; fetch never installs runtimes.")
    root = receipts_root(receipts_dir, kind=FETCH_RECEIPT_KIND)
    receipt_path = root / "{0}.json".format(plan["slug"])
    existing = _read_json(receipt_path)
    if existing and not existing.get("completed_at") and isinstance(existing.get("pid"), int) \
            and pid_alive(existing["pid"]) and not Path(existing.get("marker", "")).exists():
        raise FetchError("job_in_progress", "A download of {0} is still running.".format(plan["repo"]),
                         "Wait for it to finish (intake status).")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "{0}.done.json".format(plan["slug"])
    if marker.exists():
        marker.unlink()
    log_path = root / "{0}.log".format(plan["slug"])
    argv = list(plan["argv"]) + ["--marker", str(marker)]
    pid = (spawn or spawn_with_env)(argv, str(log_path), backend_environment(env))
    receipt = {
        "schema_version": "1.0", "kind": FETCH_RECEIPT_KIND, "repo": plan["repo"],
        "revision": plan["revision"], "file": plan["file"], "subfolder": plan.get("subfolder"),
        "local_dir": plan["local_dir"], "cache_dir": plan["cache_dir"], "slug": plan["slug"], "argv": argv, "pid": pid,
        "log_path": str(log_path), "marker": str(marker), "started_at": now(),
        "preview_hash": plan["preview_hash"], "completed_at": None, "exit_status": None,
    }
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
            "state": "failed",
            "path": None, "log_path": receipt.get("log_path"), "started_at": receipt.get("started_at"),
            "completed_at": None,
        }
        marker = _read_json(receipt.get("marker", ""))
        if marker:
            entry["state"] = "done" if marker.get("exit_status") == "done" else "failed"
            entry["path"] = marker.get("path")
            entry["completed_at"] = marker.get("finished_at")
        elif pid_alive(receipt["pid"]):
            entry["state"] = "running"
        entries.append(entry)
    return entries
