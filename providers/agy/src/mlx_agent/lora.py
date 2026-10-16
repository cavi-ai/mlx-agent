"""Confirmation-gated LoRA training jobs on the convert machinery."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import struct
import sys
import uuid
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
MAX_DATASET_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 1024 * 1024


class LoraError(RuntimeError):
    """Classified lora failure safe to surface in a result envelope."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def inspect_lora(repo=None, data=None, batch_size=BATCH_DEFAULT, hf_cache=None):
    """Read independent prerequisites without a plan, receipt, download or model load."""
    from .convert import cached_snapshot
    from .serve import _default_which

    _validate_hyperparameters(ITERS_DEFAULT, batch_size, LR_DEFAULT, LAYERS_DEFAULT)
    checks = []

    def check(identity, inspect):
        item = {"id": identity}
        try:
            item.update(inspect())
            item.setdefault("status", "ready")
        except LoraError as error:
            item.update(status="blocked", code=error.code, message=str(error), remediation=error.remediation)
        except (OSError, UnicodeError, ValueError) as error:
            item.update(status="blocked", code="input_unreadable", message=str(error),
                        remediation="Choose readable local inputs, then check setup again.")
        checks.append(item)

    def runtime():
        executable = _default_which(EXECUTABLE)
        if executable is None:
            raise LoraError("runtime_not_installed", "The mlx_lm.lora executable is not installed.",
                            "Install the training runtime with make install, then check setup again.")
        return {"message": "Training executable found.", "path": str(executable)}

    def base():
        if not repo:
            raise LoraError("repo_required", "Choose a base model.", "Select a cached language-model base or enter its repo id.")
        if not isinstance(repo, str) or not _MODEL.fullmatch(repo):
            raise LoraError("invalid_repo", "The base needs a publisher/model repo id.", "Enter the exact cached repo id.")
        snapshot = cached_snapshot(repo, hf_cache)
        if snapshot is None:
            raise LoraError("model_not_local", "The base model is not in the local Hugging Face cache.",
                            "Cache the base model before training; checking setup never downloads it.")
        identity = _training_base_identity(snapshot)
        return {"message": "Cached architecture is supported by MLX-LM.", "path": str(snapshot), "identity": identity}

    def split(filename):
        if not data:
            raise LoraError("dataset_required", "Choose a dataset directory.", "Choose a directory with train.jsonl and valid.jsonl.")
        path = Path(data) / filename
        if not path.exists():
            code = "validation_required" if filename == "valid.jsonl" else "dataset_invalid"
            raise LoraError(code, filename + " is missing.", "Provide separate training and validation examples.")
        digest = _dataset_digest(path)
        count = _validate_jsonl(path)
        if _dataset_digest(path) != digest:
            raise LoraError("dataset_changed", filename + " changed during inspection.", "Finish editing the dataset, then check setup again.")
        result = {"examples": count, "sha256": digest, "path": str(path)}
        if count < batch_size:
            result.update(status="blocked", code="dataset_too_small",
                          message="{0} has {1} examples; batch size is {2}.".format(filename, count, batch_size),
                          remediation="Reduce batch size or add examples to this split, then check setup again.")
        else:
            result["message"] = "{0} examples; enough for batch size {1}.".format(count, batch_size)
        return result

    check("runtime", runtime)
    check("base", base)
    for name, filename in (("train", "train.jsonl"), ("valid", "valid.jsonl")):
        check(name, lambda filename=filename: split(filename))
    return {"schema_version": 1, "ready": all(item["status"] == "ready" for item in checks),
            "repo": repo, "data": data, "batch_size": batch_size, "checks": checks}


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
    summary["sha256"] = {}
    for name in ("train.jsonl", "valid.jsonl"):
        path = root / name
        if path.exists():
            summary["sha256"][name] = _dataset_digest(path)
    return summary


def _dataset_digest(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_DATASET_BYTES:
        raise LoraError("dataset_invalid", "Dataset files must be bounded regular files.", "Use a local dataset without links.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_jsonl(path):
    count = 0
    _dataset_digest(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in iter(lambda: handle.readline(MAX_RECORD_BYTES + 1), ""):
                count += 1
                if count > MAX_DATASET_LINES:
                    raise LoraError(
                        "dataset_invalid",
                        "{0} exceeds {1} lines.".format(path.name, MAX_DATASET_LINES),
                        "Split the dataset; lora validates bounded datasets only.",
                    )
                if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
                    raise LoraError("dataset_invalid", "{0} line {1} exceeds the record size limit.".format(path.name, count), "Split oversized examples.")
                stripped = line.strip()
                if not stripped:
                    raise LoraError("dataset_invalid", "{0} line {1} is an empty record.".format(path.name, count), "Remove blank lines.")
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    raise LoraError(
                        "dataset_invalid",
                        "{0} line {1} is not valid JSON.".format(path.name, count),
                        "Fix the dataset line; every line must be one JSON object.",
                    )
                _validate_record(record, path.name, count)
    except (OSError, UnicodeError) as error:
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


def _training_base_identity(snapshot):
    from .backends import BackendError, load_manifests, load_registries, lookup
    from .jit_serve import model_fingerprint
    from .taxonomy import classify, read_config_type

    model_type, keys = read_config_type(snapshot)
    if not model_type:
        raise LoraError("model_config_unavailable", "Training needs a readable local config.json with model_type.",
                        "Choose a complete cached base model, then preview again.")
    if classify("", model_type=model_type, config_keys=keys, local=True)["type"] == "speculative_draft":
        raise LoraError("unsupported_training_model", "A speculative drafter is not a standalone training base.",
                        "Choose the complete target language model instead.")
    try:
        manifests = {"mlx-lm": load_manifests()["mlx-lm"]}
        registries = load_registries(manifests)
        hits = lookup(model_type, manifests, registries)
    except (BackendError, OSError, ValueError, KeyError) as error:
        raise LoraError("training_backend_unavailable", "The MLX-LM model registry could not be read.",
                        "Restore the declared MLX-LM backend before previewing training.") from error
    if not hits:
        raise LoraError("unsupported_training_model", "MLX-LM does not declare the {0} model architecture.".format(model_type),
                        "Choose a cached language-model base recognized by MLX-LM; other backends are not used for text LoRA.")
    return {"path": str(snapshot), "revision": snapshot.name, "fingerprint": model_fingerprint(snapshot),
            "model_type": model_type, "loader": hits[0]["module"], "registry_source": registries["mlx-lm"]["source"]}


def plan_lora(repo, data, iters=ITERS_DEFAULT, batch_size=BATCH_DEFAULT,
              learning_rate=LR_DEFAULT, num_layers=LAYERS_DEFAULT, out=None, hf_cache=None):
    """Validate inputs and render the exact training plan; side-effect free."""
    if not isinstance(repo, str) or not _MODEL.fullmatch(repo):
        raise LoraError(
            "invalid_repo",
            "lora requires a safe publisher/model base identifier.",
            "Pass --repo as publisher/model exactly as it appears in the Hugging Face cache.",
        )
    _validate_hyperparameters(iters, batch_size, learning_rate, num_layers)
    dataset = validate_dataset(data)
    if not dataset["valid_lines"]:
        raise LoraError(
            "validation_required", "Training requires a nonempty valid.jsonl validation set.",
            "Provide separate training and validation examples; the MLX-LM trainer requires both splits.",
        )
    for filename, key in (("train.jsonl", "train_lines"), ("valid.jsonl", "valid_lines")):
        if dataset[key] < batch_size:
            raise LoraError(
                "dataset_too_small",
                "{0} has {1} examples, fewer than batch-size {2}.".format(filename, dataset[key], batch_size),
                "Reduce --batch-size or add examples to this split, then preview again.",
            )
    from .convert import cached_snapshot
    snapshot = cached_snapshot(repo, hf_cache)
    base = _training_base_identity(snapshot) if snapshot else None
    if out is None:
        out = "{0}-lora".format(repo.split("/", 1)[1])
    out = str(Path(out).expanduser().absolute())
    plan = {
        "repo": repo,
        "base_identity": base,
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
    native_spawn = spawn is None
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
    run_id = uuid.uuid4().hex
    name = "{0}-{1}-lora".format(plan["repo"].split("/", 1)[1], run_id)
    log_path = root / (name + ".log")
    result_path = root / (name + ".result")
    if validate_dataset(plan["data"]) != plan["dataset"]:
        raise LoraError("preview_stale", "Dataset changed after preview.", "Review a fresh training plan.")
    if plan.get("base_identity"):
        from .jit_serve import model_fingerprint
        if model_fingerprint(plan["base_identity"]["path"]) != plan["base_identity"]["fingerprint"]:
            raise LoraError("preview_stale", "Base model changed after preview.", "Review a fresh training plan.")
    dataset_root = root / (name + ".dataset")
    dataset_root.mkdir(mode=0o700)
    for filename, digest in plan["dataset"]["sha256"].items():
        source = Path(plan["data"]) / filename
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise LoraError("preview_stale", "Dataset changed during snapshot.", "Review a fresh training plan.")
        target = dataset_root / filename
        target.write_bytes(payload)
        target.chmod(0o600)
    argv = list(plan["argv"])
    argv[argv.index("--data") + 1] = str(dataset_root)
    if plan.get("base_identity"):
        argv[argv.index("--model") + 1] = plan["base_identity"]["path"]
    training_argv = list(argv)
    if native_spawn:
        argv = [sys.executable, str(Path(__file__).with_name("training_runner.py")), "--result", str(result_path), "--", *argv]
    pid = spawn(argv, str(log_path))
    receipt = {
        "schema_version": LORA_RECEIPT_SCHEMA_VERSION,
        "kind": LORA_RECEIPT_KIND,
        "run_id": run_id,
        "dataset": plan["dataset"],
        "dataset_snapshot": str(dataset_root),
        "base_identity": plan.get("base_identity"),
        "result_path": str(result_path),
        "repo": plan["repo"],
        "data": plan["data"],
        "iters": plan["iters"],
        "out": plan["out"],
        "argv": training_argv,
        "pid": pid,
        "log_path": str(log_path),
        "started_at": now(),
        "preview_hash": plan["preview_hash"],
        "completed_at": None,
        "exit_status": None,
    }
    _write_receipt(root, receipt, name + ".json")
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
            entry["adapter_identity"] = receipt.get("adapter_identity")
            entry["state"] = receipt.get("exit_status") or "done"
            entries.append(entry)
            continue
        if pid_alive(receipt["pid"]):
            command = pid_command(receipt["pid"])
            entry["state"] = "running" if _argv_matches(
                {"argv": receipt["argv"], "port": None, "repo": None, "path": receipt.get("base_identity", {}).get("path") if receipt.get("base_identity") else receipt["repo"]},
                command,
                require_port=False,
            ) else "unknown"
            entries.append(entry)
            continue
        result = None
        if receipt.get("result_path"):
            try:
                candidate = Path(receipt["result_path"])
                if not candidate.is_symlink() and candidate.stat().st_size <= 4096:
                    result = json.loads(candidate.read_text())
            except (OSError, ValueError):
                pass
        if not isinstance(result, dict) or type(result.get("exit_code")) is not int:
            entry["state"] = "unknown"
            entries.append(entry)
            continue
        adapter = adapter_identity(receipt["out"]) if result["exit_code"] == 0 else None
        exit_status = "done" if result["exit_code"] == 0 and adapter else "failed"
        receipt["adapter_identity"] = adapter
        receipt["process_exit_code"] = result["exit_code"]
        entry["adapter_identity"] = adapter
        receipt["completed_at"] = now()
        receipt["exit_status"] = exit_status
        _write_receipt(root, receipt, path.name)
        entry["state"] = exit_status
        entry["completed_at"] = receipt["completed_at"]
        entries.append(entry)
    return entries


def adapter_identity(output):
    """Verify adapter tensor structure and return immutable content identities."""
    root = Path(output)
    tensor = root / "adapters.safetensors"
    config = root / "adapter_config.json"
    try:
        if root.is_symlink() or tensor.is_symlink() or config.is_symlink():
            return None
        if not tensor.is_file() or not config.is_file() or config.stat().st_size > 1024 * 1024:
            return None
        json.loads(config.read_text())
        size = tensor.stat().st_size
        with tensor.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                return None
            length = struct.unpack("<Q", prefix)[0]
            if length > 16 * 1024 * 1024 or length + 8 >= size:
                return None
            header = json.loads(stream.read(length))
        entries = [value for key, value in header.items() if key != "__metadata__"]
        if not entries:
            return None
        for value in entries:
            offsets = value["data_offsets"]
            if len(offsets) != 2 or any(type(x) is not int for x in offsets):
                return None
            if not 0 <= offsets[0] < offsets[1] <= size - 8 - length:
                return None
        identities = {}
        for path in (tensor, config):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            identities[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        return identities
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def stop_lora(pid, receipts_dir=None, pid_alive=_pid_alive, pid_command=_pid_command, terminate=os.killpg):
    root = receipts_root(receipts_dir, LORA_RECEIPT_KIND)
    for path in sorted(root.glob("*.json")) if root.is_dir() else []:
        receipt = _read_receipt(path, LORA_RECEIPT_KIND)
        if receipt is None or receipt["pid"] != pid or receipt.get("completed_at"):
            continue
        if not pid_alive(pid):
            raise LoraError("job_not_running", "Training is no longer running.", "Refresh training status.")
        if not _argv_matches({"argv": receipt["argv"], "port": None, "repo": None, "path": receipt.get("base_identity", {}).get("path") if receipt.get("base_identity") else receipt["repo"]},
                             pid_command(pid), require_port=False) or os.getpgid(pid) != pid:
            raise LoraError("process_mismatch", "The training process identity changed.", "Inspect the retained receipt before stopping it.")
        terminate(pid, signal.SIGTERM)
        receipt["completed_at"] = _utc_now()
        receipt["exit_status"] = "cancelled"
        _write_receipt(root, receipt, path.name)
        return {"status": "cancelled", "receipt": str(path)}
    raise LoraError("receipt_not_found", "No active training receipt matches this PID.", "Refresh training status.")
