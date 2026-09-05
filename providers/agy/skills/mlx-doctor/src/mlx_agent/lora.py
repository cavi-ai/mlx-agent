"""Confirmation-gated LoRA training jobs on the convert machinery."""

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


LORA_RECEIPT_SCHEMA_VERSION = "1.0"
LORA_RECEIPT_KIND = "lora"
EXECUTABLE = "mlx_lm.lora"
ITERS_MIN, ITERS_MAX, ITERS_DEFAULT = 1, 100000, 1000
BATCH_MIN, BATCH_MAX, BATCH_DEFAULT = 1, 64, 4
LR_MIN, LR_MAX, LR_DEFAULT = 1e-6, 1e-2, 1e-5
LAYERS_MIN, LAYERS_MAX, LAYERS_DEFAULT = -1, 128, 16
MAX_DATASET_LINES = 10000
DATASET_SAMPLE_LINES = 50


class LoraError(RuntimeError):
    """Classified lora failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_dataset(data_dir):
    """Validate an mlx_lm training dataset directory; bounded and read-only."""
    root = Path(data_dir)
    train = root / "train.jsonl"
    if not root.is_dir() or not train.is_file():
        raise LoraError(
            "dataset_invalid",
            "The dataset directory must contain train.jsonl: {0}".format(data_dir),
            "Provide a directory with train.jsonl (text or messages per line).",
        )
    summary = {"train_lines": _validate_jsonl(train), "valid_lines": 0}
    valid = root / "valid.jsonl"
    if valid.is_file():
        summary["valid_lines"] = _validate_jsonl(valid)
    return summary


def _validate_jsonl(path):
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                count += 1
                if count > MAX_DATASET_LINES:
                    raise LoraError(
                        "dataset_invalid",
                        "{0} exceeds {1} lines.".format(path.name, MAX_DATASET_LINES),
                        "Split the dataset; lora validates bounded datasets only.",
                    )
                if count > DATASET_SAMPLE_LINES:
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    raise LoraError(
                        "dataset_invalid",
                        "{0} line {1} is not valid JSON.".format(path.name, count),
                        "Fix the dataset line; every line must be one JSON object.",
                    )
                _validate_record(record, path.name, count)
    except OSError as error:
        raise LoraError(
            "dataset_invalid",
            "The dataset could not be read: {0}".format(error),
            "Pass a readable dataset directory.",
        )
    if count == 0:
        raise LoraError(
            "dataset_invalid",
            "{0} is empty.".format(path.name),
            "Provide at least one training example.",
        )
    return count


def _validate_record(record, name, line):
    if not isinstance(record, dict):
        raise LoraError(
            "dataset_invalid",
            "{0} line {1} must be a JSON object.".format(name, line),
            "Use {\"text\": ...} or {\"messages\": [...]} per line.",
        )
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
            ):
                raise LoraError(
                    "dataset_invalid",
                    "{0} line {1} has a malformed message.".format(name, line),
                    "Each message needs string role and content fields.",
                )
        return
    raise LoraError(
        "dataset_invalid",
        "{0} line {1} needs a non-empty text field or a messages array.".format(name, line),
        "Use {\"text\": ...} or {\"messages\": [{\"role\": ..., \"content\": ...}]}.",
    )


def _validate_hyperparameters(iters, batch_size, learning_rate, num_layers):
    if not isinstance(iters, int) or isinstance(iters, bool) or not ITERS_MIN <= iters <= ITERS_MAX:
        raise LoraError(
            "invalid_arguments",
            "iters must be between {0} and {1}.".format(ITERS_MIN, ITERS_MAX),
            "Pass a bounded --iters value.",
        )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not BATCH_MIN <= batch_size <= BATCH_MAX:
        raise LoraError(
            "invalid_arguments",
            "batch-size must be between {0} and {1}.".format(BATCH_MIN, BATCH_MAX),
            "Pass a bounded --batch-size value.",
        )
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not LR_MIN <= float(learning_rate) <= LR_MAX
    ):
        raise LoraError(
            "invalid_arguments",
            "learning-rate must be between {0} and {1}.".format(LR_MIN, LR_MAX),
            "Pass a bounded --learning-rate value.",
        )
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or not LAYERS_MIN <= num_layers <= LAYERS_MAX:
        raise LoraError(
            "invalid_arguments",
            "num-layers must be between {0} and {1}.".format(LAYERS_MIN, LAYERS_MAX),
            "Pass a bounded --num-layers value (-1 trains all layers).",
        )


def plan_lora(repo, data, iters=ITERS_DEFAULT, batch_size=BATCH_DEFAULT,
              learning_rate=LR_DEFAULT, num_layers=LAYERS_DEFAULT, out=None):
    """Validate inputs and render the exact training plan; side-effect free."""
    if not isinstance(repo, str) or not _MODEL.fullmatch(repo):
        raise LoraError(
            "invalid_repo",
            "lora requires a safe publisher/model base identifier.",
            "Pass --repo as publisher/model exactly as it appears in the Hugging Face cache.",
        )
    _validate_hyperparameters(iters, batch_size, learning_rate, num_layers)
    dataset = validate_dataset(data)
    if out is None:
        out = "{0}-lora".format(repo.split("/", 1)[1])
    plan = {
        "repo": repo,
        "data": str(data),
        "iters": iters,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "num_layers": num_layers,
        "out": str(out),
        "dataset": dataset,
        "argv": [
            EXECUTABLE,
            "--model", repo,
            "--train",
            "--data", str(data),
            "--iters", str(iters),
            "--batch-size", str(batch_size),
            "--learning-rate", repr(float(learning_rate)),
            "--num-layers", str(num_layers),
            "--adapter-path", str(out),
        ],
    }
    canonical = json.dumps(
        {key: plan[key] for key in sorted(plan) if key != "preview_hash"},
        sort_keys=True, separators=(",", ":"),
    )
    plan["preview_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def _receipt_filename(plan):
    return "{0}-lora.json".format(plan["repo"].split("/", 1)[1])


def start_lora(plan, receipts_dir=None, confirm=False, preview_hash=None,
               which=None, model_present=None, spawn=None, now=_utc_now,
               pid_alive=None):
    """Execute a reviewed training plan; the only mutating entry point."""
    from .serve import _default_spawn, _default_which

    which = which or _default_which
    spawn = spawn or _default_spawn
    pid_alive = pid_alive or _pid_alive
    root = receipts_root(receipts_dir, LORA_RECEIPT_KIND)

    if not confirm:
        return {"status": "preview", "plan": plan, "requires_confirmation": True}
    if not preview_hash:
        raise LoraError(
            "preview_hash_required",
            "--confirm requires the hash from a reviewed lora preview.",
            "Run lora start without --confirm, inspect the plan, then pass --preview-hash.",
        )
    if preview_hash != plan["preview_hash"]:
        raise LoraError(
            "preview_stale",
            "The supplied preview hash does not match this lora plan.",
            "Re-run lora start without --confirm and review the fresh plan.",
        )
    if which(EXECUTABLE) is None:
        raise LoraError(
            "runtime_not_installed",
            "The {0} executable is not installed.".format(EXECUTABLE),
            "Install mlx-lm yourself (pip install mlx-lm); lora never installs runtimes.",
        )
    if model_present is not None and not model_present(plan["repo"]):
        raise LoraError(
            "model_not_local",
            "The base model is not present in the local Hugging Face cache.",
            "Download it with the runtime's own pull command first; lora never downloads.",
        )
    if Path(plan["out"]).exists():
        raise LoraError(
            "output_exists",
            "The adapter output path already exists: {0}".format(plan["out"]),
            "Pick a fresh --out path; lora never overwrites.",
        )
    for receipt_path in sorted(root.glob("*.json")) if root.is_dir() else []:
        existing = _read_receipt(receipt_path, LORA_RECEIPT_KIND)
        if existing is None or existing.get("completed_at"):
            continue
        if pid_alive(existing["pid"]):
            raise LoraError(
                "job_in_progress",
                "A lora job is still running (pid {0}).".format(existing["pid"]),
                "Wait for it to finish (lora status) before starting another.",
            )

    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "{0}-lora.log".format(plan["repo"].split("/", 1)[1])
    pid = spawn(plan["argv"], str(log_path))
    receipt = {
        "schema_version": LORA_RECEIPT_SCHEMA_VERSION,
        "kind": LORA_RECEIPT_KIND,
        "repo": plan["repo"],
        "data": plan["data"],
        "iters": plan["iters"],
        "out": plan["out"],
        "argv": list(plan["argv"]),
        "pid": pid,
        "log_path": str(log_path),
        "started_at": now(),
        "preview_hash": plan["preview_hash"],
        "completed_at": None,
        "exit_status": None,
    }
    _write_receipt(root, receipt, _receipt_filename(plan))
    return {"status": "started", "receipt": receipt}


def status_lora(receipts_dir=None, pid_alive=_pid_alive, pid_command=_pid_command,
                now=_utc_now):
    """Cross-check lora receipts against live processes; marks exits once."""
    root = receipts_root(receipts_dir, LORA_RECEIPT_KIND)
    entries = []
    if not root.is_dir():
        return entries
    for path in sorted(root.glob("*.json")):
        receipt = _read_receipt(path, LORA_RECEIPT_KIND)
        if receipt is None:
            continue
        entry = {
            "receipt": str(path),
            "repo": receipt["repo"],
            "data": receipt.get("data"),
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
