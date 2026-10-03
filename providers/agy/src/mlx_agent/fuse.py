"""Confirmation-gated LoRA fuse jobs on the convert machinery."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .convert import (
    _read_receipt,
    _write_receipt,
    receipts_root,
)
from .serve import _argv_matches, _pid_alive, _pid_command
from .wiring import _MODEL


FUSE_RECEIPT_SCHEMA_VERSION = "1.0"
FUSE_RECEIPT_KIND = "fuse"
EXECUTABLE = "mlx_lm.fuse"


class FuseError(RuntimeError):
    """Classified fuse failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_adapter(adapter_path):
    """Confirm the adapter directory looks like an mlx_lm LoRA output."""
    root = Path(adapter_path)
    if not root.is_dir():
        raise FuseError(
            "adapter_invalid",
            "The adapter path is not a directory: {0}".format(adapter_path),
            "Point --adapter at a completed lora output directory.",
        )
    config = root / "adapter_config.json"
    if not config.is_file():
        raise FuseError(
            "adapter_invalid",
            "The adapter directory lacks adapter_config.json.",
            "Point --adapter at a completed lora output directory.",
        )
    try:
        value = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise FuseError(
            "adapter_invalid",
            "adapter_config.json could not be parsed: {0}".format(error),
            "Repair or regenerate the adapter before fusing.",
        )
    if not isinstance(value, dict):
        raise FuseError(
            "adapter_invalid",
            "adapter_config.json must contain a JSON object.",
            "Repair or regenerate the adapter before fusing.",
        )
    return {"adapter_config": True}


def plan_fuse(repo, adapter, out=None):
    """Validate inputs and render the exact fuse plan; side-effect free."""
    if not isinstance(repo, str) or not _MODEL.fullmatch(repo):
        raise FuseError(
            "invalid_repo",
            "fuse requires a safe publisher/model base identifier.",
            "Pass --repo as publisher/model exactly as it appears in the Hugging Face cache.",
        )
    validate_adapter(adapter)
    if out is None:
        out = "{0}-fused".format(repo.split("/", 1)[1])
    plan = {
        "repo": repo,
        "adapter": str(adapter),
        "out": str(out),
        "argv": [
            EXECUTABLE,
            "--model", repo,
            "--adapter-path", str(adapter),
            "--save-path", str(out),
        ],
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["preview_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def start_fuse(plan, receipts_dir=None, confirm=False, preview_hash=None,
               which=None, model_present=None, spawn=None, now=_utc_now,
               pid_alive=None):
    """Execute a reviewed fuse plan; the only mutating entry point."""
    from .serve import _default_spawn, _default_which

    which = which or _default_which
    spawn = spawn or _default_spawn
    pid_alive = pid_alive or _pid_alive
    root = receipts_root(receipts_dir, FUSE_RECEIPT_KIND)

    if not confirm:
        return {"status": "preview", "plan": plan, "requires_confirmation": True}
    if not preview_hash:
        raise FuseError(
            "preview_hash_required",
            "--confirm requires the hash from a reviewed fuse preview.",
            "Run fuse start without --confirm, inspect the plan, then pass --preview-hash.",
        )
    if preview_hash != plan["preview_hash"]:
        raise FuseError(
            "preview_stale",
            "The supplied preview hash does not match this fuse plan.",
            "Re-run fuse start without --confirm and review the fresh plan.",
        )
    if which(EXECUTABLE) is None:
        raise FuseError(
            "runtime_not_installed",
            "The {0} executable is not installed.".format(EXECUTABLE),
            "Install mlx-lm yourself (pip install mlx-lm); fuse never installs runtimes.",
        )
    if model_present is not None and not model_present(plan["repo"]):
        raise FuseError(
            "model_not_local",
            "The base model is not present in the local Hugging Face cache.",
            "Download it with the runtime's own pull command first; fuse never downloads.",
        )
    if Path(plan["out"]).exists():
        raise FuseError(
            "output_exists",
            "The fused output path already exists: {0}".format(plan["out"]),
            "Pick a fresh --out path; fuse never overwrites.",
        )
    for receipt_path in sorted(root.glob("*.json")) if root.is_dir() else []:
        existing = _read_receipt(receipt_path, FUSE_RECEIPT_KIND)
        if existing is None or existing.get("completed_at"):
            continue
        if pid_alive(existing["pid"]):
            raise FuseError(
                "job_in_progress",
                "A fuse job is still running (pid {0}).".format(existing["pid"]),
                "Wait for it to finish (fuse status) before starting another.",
            )

    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "{0}-fuse.log".format(plan["repo"].split("/", 1)[1])
    pid = spawn(plan["argv"], str(log_path))
    receipt = {
        "schema_version": FUSE_RECEIPT_SCHEMA_VERSION,
        "kind": FUSE_RECEIPT_KIND,
        "repo": plan["repo"],
        "adapter": plan["adapter"],
        "out": plan["out"],
        "argv": list(plan["argv"]),
        "pid": pid,
        "log_path": str(log_path),
        "started_at": now(),
        "preview_hash": plan["preview_hash"],
        "completed_at": None,
        "exit_status": None,
    }
    _write_receipt(root, receipt, "{0}-fuse.json".format(plan["repo"].split("/", 1)[1]))
    return {"status": "started", "receipt": receipt}


def status_fuse(receipts_dir=None, pid_alive=_pid_alive, pid_command=_pid_command,
                now=_utc_now):
    """Cross-check fuse receipts against live processes; marks exits once."""
    root = receipts_root(receipts_dir, FUSE_RECEIPT_KIND)
    entries = []
    if not root.is_dir():
        return entries
    for path in sorted(root.glob("*.json")):
        receipt = _read_receipt(path, FUSE_RECEIPT_KIND)
        if receipt is None:
            continue
        entry = {
            "receipt": str(path),
            "repo": receipt["repo"],
            "adapter": receipt.get("adapter"),
            "out": receipt["out"],
            "pid": receipt["pid"],
            "log_path": receipt.get("log_path"),
            "started_at": receipt.get("started_at"),
            "completed_at": receipt.get("completed_at"),
        }
        if receipt.get("completed_at"):
            entry["state"] = receipt.get("exit_status") or "done"
            entries.append(entry)
            continue
        if pid_alive(receipt["pid"]):
            command = pid_command(receipt["pid"])
            entry["state"] = "running" if _argv_matches(
                {"argv": receipt["argv"], "port": None, "repo": receipt["repo"]},
                command,
                require_port=False,
            ) else "unknown"
            entries.append(entry)
            continue
        exit_status = "done" if Path(receipt["out"]).exists() else "failed"
        receipt["completed_at"] = now()
        receipt["exit_status"] = exit_status
        _write_receipt(root, receipt, path.name)
        entry["state"] = exit_status
        entry["completed_at"] = receipt["completed_at"]
        entries.append(entry)
    return entries
