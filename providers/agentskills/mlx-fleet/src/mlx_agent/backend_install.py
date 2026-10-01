"""Confirmation-gated install and removal of optional converter backends."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .backends import (
    BACKENDS_DIR,
    BackendError,
    backend_environment,
    backend_target,
    backends_root,
    load_manifests,
    load_registries,
    spawn_with_env,
)
from .convert import _write_receipt, receipts_root
from .serve import _pid_alive

BACKEND_RECEIPT_KIND = "backend"
BACKEND_RECEIPT_SCHEMA_VERSION = "1.0"
BACKEND_RUNNER = Path(__file__).resolve().with_name("backend_runner.py")
REQUIRED_PYTHON = (3, 12)
INSTALL_HOSTS = ("pypi.org", "files.pythonhosted.org")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _finalize(plan):
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["preview_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def default_trash_dir():
    if sys.platform == "darwin":
        return Path.home() / ".Trash"
    return Path.home() / ".local" / "share" / "Trash" / "files"


def _optional_manifest(backend_id, manifests):
    manifest = manifests.get(backend_id)
    if manifest is None:
        raise BackendError(
            "unknown_backend", "No declared backend named {0}.".format(backend_id),
            "Run backend list to see the declared backends.",
        )
    if manifest["builtin"]:
        raise BackendError(
            "builtin_backend", "{0} ships with the main runtime.".format(backend_id),
            "Install it with the workbench runtime (make install); backend install manages optional backends only.",
        )
    return manifest


def _refuse_symlinks(target):
    for path in (target, target.parent):
        if path.is_symlink():
            raise BackendError(
                "symlink_refused", "{0} is a symbolic link.".format(path),
                "Point XDG_DATA_HOME at a real directory.",
            )


def _check_hash(plan, confirm, preview_hash, verb):
    if not preview_hash:
        raise BackendError(
            "preview_hash_required",
            "--confirm requires the hash from a reviewed backend {0} preview.".format(verb),
            "Run backend {0} without --confirm, review it, then pass --preview-hash.".format(verb),
        )
    if preview_hash != plan["preview_hash"]:
        raise BackendError(
            "preview_stale", "The supplied preview hash does not match this {0} plan.".format(verb),
            "Re-run backend {0} without --confirm and review the fresh plan.".format(verb),
        )


def _receipt_file(receipts_dir, backend_id):
    return receipts_root(receipts_dir, kind=BACKEND_RECEIPT_KIND) / "{0}.json".format(backend_id)


def _read_receipt(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("kind") != BACKEND_RECEIPT_KIND:
        return None
    if not isinstance(value.get("pid"), int) or isinstance(value.get("pid"), bool):
        return None
    return value


def _running(receipt, pid_alive):
    return bool(receipt) and not receipt.get("completed_at") and pid_alive(receipt["pid"])


def plan_install(backend_id, manifests=None, root=None, python=None, version_info=None, trash_dir=None):
    manifests = load_manifests() if manifests is None else manifests
    manifest = _optional_manifest(backend_id, manifests)
    version_info = sys.version_info if version_info is None else version_info
    if tuple(version_info[:2]) != REQUIRED_PYTHON:
        raise BackendError(
            "python_mismatch",
            "Backend locks target Python 3.12; this agent runs {0}.{1}.".format(version_info[0], version_info[1]),
            "Run mlx-agent with the workbench .venv interpreter (Python 3.12).",
        )
    lock = BACKENDS_DIR / manifest["lock"]
    if not lock.is_file():
        raise BackendError("lock_missing", "The lock file {0} is missing.".format(lock.name), "Reinstall mlx-agent; the lock ships with it.")
    content = lock.read_bytes()
    target = backend_target(manifest, root)
    python = str(python or sys.executable)
    trash = Path(trash_dir) if trash_dir is not None else default_trash_dir()
    plan = _finalize({
        "id": manifest["id"], "version": manifest["version"], "package": manifest["package"],
        "target": str(target), "lock": str(lock),
        "lock_sha256": hashlib.sha256(content).hexdigest(),
        "requirements": sum(1 for line in content.decode("utf-8").splitlines() if line[:1].isalnum()),
        "python": python, "network": list(INSTALL_HOSTS),
        "argv": [
            python, str(BACKEND_RUNNER), "--id", manifest["id"], "--version", manifest["version"],
            "--package", manifest["package"], "--lock", str(lock), "--target", str(target),
            "--python", python, "--trash", str(trash),
        ],
    })
    plan["argv"] = plan["argv"] + ["--preview-hash", plan["preview_hash"]]
    return plan


def start_install(plan, receipts_dir=None, confirm=False, preview_hash=None, spawn=None,
                  pid_alive=_pid_alive, now=_utc_now, env=None):
    if not confirm:
        return {"status": "preview", "plan": plan, "requires_confirmation": True}
    _check_hash(plan, confirm, preview_hash, "install")
    target = Path(plan["target"])
    _refuse_symlinks(target)
    if target.exists():
        raise BackendError(
            "target_exists", "{0} already exists.".format(target),
            "Remove it first: mlx-agent backend remove {0}.".format(plan["id"]),
        )
    receipt_path = _receipt_file(receipts_dir, plan["id"])
    if _running(_read_receipt(receipt_path), pid_alive):
        raise BackendError(
            "job_in_progress", "An install of {0} is still running.".format(plan["id"]),
            "Wait for it to finish (backend list).",
        )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = receipt_path.with_suffix(".log")
    pid = (spawn or spawn_with_env)(plan["argv"], str(log_path), backend_environment(env))
    receipt = {
        "schema_version": BACKEND_RECEIPT_SCHEMA_VERSION, "kind": BACKEND_RECEIPT_KIND,
        "id": plan["id"], "version": plan["version"], "target": plan["target"],
        "argv": list(plan["argv"]), "pid": pid, "log_path": str(log_path),
        "started_at": now(), "preview_hash": plan["preview_hash"],
        "completed_at": None, "exit_status": None, "removed_at": None,
    }
    _write_receipt(receipt_path.parent, receipt, receipt_path.name)
    return {"status": "started", "receipt": receipt}


def _state(manifest, info, receipt, pid_alive):
    if info["installed"]:
        return "installed"
    if manifest["builtin"] or receipt is None or receipt.get("removed_at"):
        return "absent"
    if _running(receipt, pid_alive):
        return "installing"
    if receipt.get("exit_status") == "installed":
        return "absent"
    return "failed"


def list_backends(manifests=None, root=None, receipts_dir=None, pid_alive=_pid_alive,
                  find_spec=None, now=_utc_now):
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root, find_spec or importlib.util.find_spec)
    entries = []
    for backend_id in sorted(manifests):
        manifest = manifests[backend_id]
        info = registries[backend_id]
        receipt_path = _receipt_file(receipts_dir, backend_id)
        receipt = None if manifest["builtin"] else _read_receipt(receipt_path)
        state = _state(manifest, info, receipt, pid_alive)
        if receipt and not receipt.get("completed_at") and state in ("installed", "failed"):
            receipt["completed_at"] = now()
            receipt["exit_status"] = state
            _write_receipt(receipt_path.parent, receipt, receipt_path.name)
        entries.append({
            "id": backend_id, "version": manifest["version"], "builtin": manifest["builtin"],
            "state": state, "categories": sorted(manifest["categories"]),
            "model_types": sum(len(entry.get("model_types", ())) for entry in info["registry"].values()),
            "registry_source": info["source"],
            "target": None if manifest["builtin"] else str(backend_target(manifest, root)),
            "log_path": receipt.get("log_path") if receipt else None,
            "started_at": receipt.get("started_at") if receipt else None,
            "completed_at": receipt.get("completed_at") if receipt else None,
        })
    return {
        "schema": "backends/1",
        "root": str(Path(root) if root is not None else backends_root()),
        "backends": entries,
    }


def plan_remove(backend_id, manifests=None, root=None, trash_dir=None):
    manifests = load_manifests() if manifests is None else manifests
    manifest = _optional_manifest(backend_id, manifests)
    trash = Path(trash_dir) if trash_dir is not None else default_trash_dir()
    return _finalize({
        "id": backend_id, "target": str(backend_target(manifest, root)),
        "action": "move_to_trash", "trash": str(trash),
    })


def remove_backend(plan, receipts_dir=None, confirm=False, preview_hash=None,
                   pid_alive=_pid_alive, now=_utc_now, move=shutil.move):
    if not confirm:
        return {"status": "preview", "plan": plan, "requires_confirmation": True}
    _check_hash(plan, confirm, preview_hash, "remove")
    target = Path(plan["target"])
    _refuse_symlinks(target)
    if not target.exists():
        raise BackendError(
            "not_installed", "{0} is not installed at {1}.".format(plan["id"], target),
            "Nothing to remove; run backend list.",
        )
    receipt_path = _receipt_file(receipts_dir, plan["id"])
    receipt = _read_receipt(receipt_path)
    if _running(receipt, pid_alive):
        raise BackendError(
            "job_in_progress", "An install of {0} is still running.".format(plan["id"]),
            "Wait for it to finish (backend list).",
        )
    trash = Path(plan["trash"])
    trash.mkdir(parents=True, exist_ok=True)
    destination = trash / "{0}-backend-{1}".format(
        plan["id"], datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    move(str(target), str(destination))
    if receipt:
        receipt["removed_at"] = now()
        _write_receipt(receipt_path.parent, receipt, receipt_path.name)
    return {"status": "removed", "id": plan["id"], "moved_to": str(destination)}
