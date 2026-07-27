"""Confirmation-gated local MLX quantization jobs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .serve import (
    _argv_matches,
    _pid_alive,
    _pid_command,
    _terminate_pid,
)
from .transactions import _atomic_in_directory
from .wiring import _MODEL


CONVERT_RECEIPT_SCHEMA_VERSION = "1.0"
CONVERT_RECEIPT_KIND = "convert"
Q_BITS_CHOICES = (4, 8)
EXECUTABLE = "mlx_lm.convert"
MAX_LOG_TAIL_BYTES = 64 * 1024


class ConvertError(RuntimeError):
    """Classified convert failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def receipts_root(root=None, kind="convert"):
    base = Path(root) if root is not None else Path.cwd()
    return base / ".mlx-agent-receipts" / kind


def plan_convert(repo, q_bits=4, out=None):
    """Render the exact conversion plan; pure and side-effect free."""
    if not isinstance(repo, str) or not _MODEL.fullmatch(repo):
        raise ConvertError(
            "invalid_repo",
            "convert requires a safe publisher/model identifier.",
            "Pass --repo as publisher/model exactly as it appears in the Hugging Face cache.",
        )
    if not isinstance(q_bits, int) or isinstance(q_bits, bool):
        raise ConvertError(
            "invalid_arguments", "q_bits must be an integer.",
            "Pass --q-bits 4 or --q-bits 8.",
        )
    if q_bits not in Q_BITS_CHOICES:
        raise ConvertError(
            "invalid_arguments",
            "q_bits must be one of {0}.".format(list(Q_BITS_CHOICES)),
            "Only bounded 4bit and 8bit recipes are supported in v1.",
        )
    if out is None:
        name = repo.split("/", 1)[1]
        out = "{0}-MLX-{1}bit".format(name, q_bits)
    plan = {
        "repo": repo,
        "q_bits": q_bits,
        "out": str(out),
        "argv": [
            EXECUTABLE,
            "--hf-path", repo,
            "--mlx-path", str(out),
            "--q-bits", str(q_bits),
        ],
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["preview_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def _receipt_path(root, plan):
    return root / "{0}-{1}bit.json".format(
        plan["repo"].split("/", 1)[1], plan["q_bits"]
    )


def _read_receipt(path, kind=CONVERT_RECEIPT_KIND):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("kind") != kind:
        return None
    if not isinstance(value.get("pid"), int) or isinstance(value.get("pid"), bool):
        return None
    if not isinstance(value.get("argv"), list) or not isinstance(value.get("repo"), str):
        return None
    return value


def _write_receipt(root, receipt, filename):
    content = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    _atomic_in_directory(root, filename, content, 0o600)


def start_convert(plan, receipts_dir=None, confirm=False, preview_hash=None,
                  which=None, model_present=None, spawn=None, now=_utc_now,
                  pid_alive=None):
    """Execute a reviewed conversion plan; the only mutating entry point."""
    from .serve import _default_spawn, _default_which

    which = which or _default_which
    spawn = spawn or _default_spawn
    pid_alive = pid_alive or _pid_alive
    root = receipts_root(receipts_dir)

    if not confirm:
        return {"status": "preview", "plan": plan, "requires_confirmation": True}
    if not preview_hash:
        raise ConvertError(
            "preview_hash_required",
            "--confirm requires the hash from a reviewed convert preview.",
            "Run convert start without --confirm, inspect the plan, then pass --preview-hash.",
        )
    if preview_hash != plan["preview_hash"]:
        raise ConvertError(
            "preview_stale",
            "The supplied preview hash does not match this convert plan.",
            "Re-run convert start without --confirm and review the fresh plan.",
        )
    if which(EXECUTABLE) is None:
        raise ConvertError(
            "runtime_not_installed",
            "The {0} executable is not installed.".format(EXECUTABLE),
            "Install mlx-lm yourself (pip install mlx-lm); convert never installs runtimes.",
        )
    if model_present is not None and not model_present(plan["repo"]):
        raise ConvertError(
            "model_not_local",
            "The source model is not present in the local Hugging Face cache.",
            "Download it with the runtime's own pull command first; convert never downloads.",
        )
    if Path(plan["out"]).exists():
        raise ConvertError(
            "output_exists",
            "The output path already exists: {0}".format(plan["out"]),
            "Pick a fresh --out path; convert never overwrites.",
        )
    for receipt_path in sorted(root.glob("*.json")) if root.is_dir() else []:
        existing = _read_receipt(receipt_path)
        if existing is None or existing.get("completed_at"):
            continue
        if pid_alive(existing["pid"]):
            raise ConvertError(
                "job_in_progress",
                "A convert job is still running (pid {0}).".format(existing["pid"]),
                "Wait for it to finish (convert status) before starting another.",
            )

    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "{0}-{1}bit.log".format(plan["repo"].split("/", 1)[1], plan["q_bits"])
    pid = spawn(plan["argv"], str(log_path))
    receipt = {
        "schema_version": CONVERT_RECEIPT_SCHEMA_VERSION,
        "kind": CONVERT_RECEIPT_KIND,
        "repo": plan["repo"],
        "q_bits": plan["q_bits"],
        "out": plan["out"],
        "argv": list(plan["argv"]),
        "pid": pid,
        "log_path": str(log_path),
        "started_at": now(),
        "preview_hash": plan["preview_hash"],
        "completed_at": None,
        "exit_status": None,
    }
    _write_receipt(
        root, receipt,
        "{0}-{1}bit.json".format(receipt["repo"].split("/", 1)[1], receipt["q_bits"]),
    )
    return {"status": "started", "receipt": receipt}


def status_convert(receipts_dir=None, pid_alive=_pid_alive, pid_command=_pid_command,
                   now=_utc_now):
    """Cross-check convert receipts against live processes; marks exits once."""
    root = receipts_root(receipts_dir)
    entries = []
    if not root.is_dir():
        return entries
    for path in sorted(root.glob("*.json")):
        receipt = _read_receipt(path)
        if receipt is None:
            continue
        entry = {
            "receipt": str(path),
            "repo": receipt["repo"],
            "q_bits": receipt["q_bits"],
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
        _write_receipt(
            root, receipt,
            "{0}-{1}bit.json".format(receipt["repo"].split("/", 1)[1], receipt["q_bits"]),
        )
        entry["state"] = exit_status
        entry["completed_at"] = receipt["completed_at"]
        entries.append(entry)
    return entries
