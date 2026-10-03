"""Confirmation-gated local MLX quantization jobs."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .backends import (
    BackendError,
    backend_environment,
    backend_python,
    load_manifests,
    port_bits,
    port_converter,
    recipe_for,
    read_install_marker,
    spawn_with_env,
    sync_ports,
)
from .gguf import GGUFError, describe_gguf
from .intake_source import validate_subfolder
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
GGUF_RUNNER = Path(__file__).resolve().with_name("gguf_runner.py")
GGUF_REQUIRED_MODULES = ("torch", "transformers", "gguf")
MAX_LOG_TAIL_BYTES = 64 * 1024
_UNSAFE_SLUG = re.compile(r"[^A-Za-z0-9._-]+")
_MODEL_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


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


def cached_snapshot(repo, hf_cache=None, revision=None):
    """``<cache>/models--org--name/snapshots/<revision or refs/main>`` when that snapshot is cached, else None."""
    from .model_doctor import default_hf_cache

    entry = (Path(hf_cache) if hf_cache else default_hf_cache()) / "models--{0}".format(repo.replace("/", "--"))
    if revision is None:
        try:
            revision = (entry / "refs" / "main").read_text(encoding="utf-8").strip()
        except OSError:
            return None
    if not revision or "/" in revision or revision in (".", ".."):
        return None
    snapshot = entry / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


def cached_subfolder(repo, subfolder, hf_cache=None):
    """``<cached snapshot>/<subfolder>`` when that folder is cached, else None."""
    snapshot = cached_snapshot(repo, hf_cache)
    folder = snapshot / subfolder if snapshot is not None else None
    return folder if folder is not None and folder.is_dir() else None


def plan_convert(repo, q_bits=4, out=None, backend=None, manifests=None, backends_root_dir=None, model_type=None,
                 subfolder=None, hf_cache=None):
    """Render the exact conversion plan; side-effect free (a subfolder is looked up in the cache).

    ``model_type`` (from intake) selects a port's own converter when the port
    ships one; otherwise the backend's converter runs. A ``subfolder``
    checkpoint converts from its cached folder, so every converter reads it
    as a local directory.
    """
    if not isinstance(repo, str) or not _MODEL.fullmatch(repo):
        raise ConvertError(
            "invalid_repo",
            "convert requires a safe publisher/model identifier.",
            "Pass --repo as publisher/model exactly as it appears in the Hugging Face cache.",
        )
    if model_type is not None and (not isinstance(model_type, str) or not _MODEL_TYPE.fullmatch(model_type)):
        raise ConvertError(
            "invalid_arguments", "model_type must be a config model_type such as qwen2 or laya.",
            "Pass --model-type exactly as intake resolve reports it.",
        )
    _validate_q_bits(q_bits)
    if subfolder is not None:
        try:
            validate_subfolder(subfolder)
        except ValueError as error:
            raise ConvertError("invalid_arguments", str(error), "Pass --subfolder as the folder path inside the repository.") from error
    name = repo.split("/", 1)[1] + ("-" + subfolder.replace("/", "-") if subfolder else "")
    if out is None:
        out = "{0}-MLX-{1}bit".format(name, q_bits)
    hf_path = repo
    if subfolder:
        folder = cached_subfolder(repo, subfolder, hf_cache)
        if folder is None:
            raise ConvertError(
                "subfolder_not_cached", "{0}/{1} is not in the Hugging Face cache.".format(repo, subfolder),
                "Download it first: mlx-agent intake fetch {0}/{1}.".format(repo, subfolder),
            )
        hf_path = str(folder)
    flags = ["--hf-path", hf_path, "--mlx-path", str(out), "--quantize", "--q-bits", str(q_bits)]
    plan = {
        "repo": repo,
        "source": dict({"kind": "hf-cache", "repo": repo}, **({"subfolder": subfolder} if subfolder else {})),
        "slug": "{0}-{1}bit".format(name, q_bits),
        "q_bits": q_bits,
        "out": str(out),
    }
    if backend in (None, "mlx-lm"):
        plan["argv"] = [EXECUTABLE] + flags
        return _finalize_plan(plan)
    try:
        manifests = load_manifests() if manifests is None else manifests
    except BackendError as error:
        raise ConvertError(error.code, str(error), error.remediation) from error
    manifest = manifests.get(backend)
    if manifest is None or manifest["builtin"]:
        raise ConvertError(
            "unknown_backend",
            "No optional backend named {0}.".format(backend),
            "Run mlx-agent backend list to see the declared backends.",
        )
    widths = port_bits(model_type, backend, manifests)
    if widths and q_bits not in widths:
        raise ConvertError(
            "invalid_arguments",
            "The {0} port converts at {1} bits only.".format(model_type, " or ".join(str(width) for width in widths)),
            "Pass --q-bits {0}.".format(widths[-1]),
        )
    python = backend_python(manifest, backends_root_dir)
    plan["backend"] = backend
    plan["backends_root"] = str(python.parent.parent.parent)
    converter = port_converter(model_type, backend, manifests)
    recipe = recipe_for(repo, manifests) if converter else None
    if recipe and recipe["backend"] == backend and recipe["port"] == model_type:
        flags = _recipe_flags(recipe, flags, hf_cache)
        plan["recipe"] = {key: recipe[key] for key in ("base", "base_revision", "lora", "lora_scale")}
    plan["argv"] = [str(python), "-m", converter or manifest["convert"]] + flags
    ports = sorted(name for names in (manifest.get("ports") or {}).values() for name in names)
    if ports:
        plan["ports"] = ports
    if converter:
        plan["port_converter"] = converter
    return _finalize_plan(plan)


def _recipe_flags(recipe, flags, hf_cache):
    """Converter flags for a recipe: the cached base snapshot and the cached LoRA, both local."""
    base = cached_snapshot(recipe["base"], hf_cache, revision=recipe["base_revision"])
    snapshot = cached_snapshot(recipe["repo"], hf_cache)
    lora = snapshot / recipe["lora"] if snapshot is not None else None
    if base is None or lora is None or not lora.is_file():
        raise ConvertError(
            "recipe_not_cached",
            "{0} needs {1}@{2} and its {3} in the Hugging Face cache.".format(
                recipe["repo"], recipe["base"], recipe["base_revision"][:7], recipe["lora"]),
            "Download it first: mlx-agent intake fetch {0} --model-type {1}.".format(recipe["repo"], recipe["port"]),
        )
    flags = list(flags)
    flags[flags.index("--hf-path") + 1] = str(base)
    return flags + ["--lora", str(lora), "--lora-scale", repr(float(recipe["lora_scale"]))]


def _finalize_plan(plan):
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["preview_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def _validate_q_bits(q_bits):
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


def _slugify(value, q_bits):
    slug = _UNSAFE_SLUG.sub("-", str(value)).strip("-.") or "model"
    return "{0}-{1}bit".format(slug[:96], q_bits)


def plan_gguf_convert(gguf_path, q_bits=4, out=None, runner=None, describe=describe_gguf):
    """Render the exact GGUF to MLX conversion plan; pure and side-effect free."""
    _validate_q_bits(q_bits)
    location = Path(str(gguf_path)).expanduser()
    if location.suffix.lower() != ".gguf":
        raise ConvertError(
            "invalid_source",
            "convert --gguf requires a path ending in .gguf.",
            "Pass the GGUF weights file itself, not its directory.",
        )
    if not location.is_file():
        raise ConvertError(
            "source_not_found",
            "No GGUF file at {0}.".format(location),
            "Run convert scan to list the GGUF files under your configured roots.",
        )
    try:
        entry = describe(location)
    except GGUFError as error:
        raise ConvertError(error.code, str(error), error.remediation) from error
    if not entry.get("readable"):
        detail = entry.get("error") or {}
        raise ConvertError(
            detail.get("code", "gguf_header_unreadable"),
            detail.get("message", "The GGUF header could not be parsed."),
            "Re-download the file; a truncated GGUF cannot be converted.",
        )
    shard = entry.get("shard")
    if shard is not None and shard["index"] != 1:
        raise ConvertError(
            "shard_not_first",
            "{0} is shard {1} of {2}.".format(location.name, shard["index"], shard["total"]),
            "Point convert at the -00001-of-{0:05d} shard; the loader reads the rest itself.".format(
                shard["total"]
            ),
        )
    if out is None:
        out = "{0}-MLX-{1}bit".format(location.stem, q_bits)
    resolved_runner = Path(runner) if runner is not None else GGUF_RUNNER
    plan = {
        "repo": str(location),
        "source": {
            "kind": "gguf",
            "path": str(location),
            "name": location.name,
            "bytes": entry["bytes"],
            "signature": entry.get("signature"),
            "architecture": entry.get("architecture"),
            "quantization": entry.get("quantization"),
            "model_key": entry.get("model_key"),
        },
        "slug": _slugify(location.stem, q_bits),
        "q_bits": q_bits,
        "out": str(out),
        "argv": [
            sys.executable,
            str(resolved_runner),
            "--gguf", str(location),
            "--out", str(out),
            "--q-bits", str(q_bits),
        ],
    }
    if entry.get("signature"):
        plan["argv"].extend(["--signature", entry["signature"]])
    return _finalize_plan(plan)


def _slug(value):
    slug = value.get("slug")
    if isinstance(slug, str) and slug:
        return slug
    return "{0}-{1}bit".format(value["repo"].split("/", 1)[1], value["q_bits"])


def _receipt_path(root, plan):
    return root / "{0}.json".format(_slug(plan))


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


def _source_kind(value):
    source = value.get("source")
    if isinstance(source, dict) and isinstance(source.get("kind"), str):
        return source["kind"]
    return "hf-cache"


def _default_module_present(names):
    import importlib.util

    return [name for name in names if importlib.util.find_spec(name) is None]


def start_convert(plan, receipts_dir=None, confirm=False, preview_hash=None,
                  which=None, model_present=None, spawn=None, now=_utc_now,
                  pid_alive=None, module_present=None):
    """Execute a reviewed conversion plan; the only mutating entry point."""
    from .serve import _default_spawn, _default_which

    which = which or _default_which
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
    backend = plan.get("backend")
    if backend:
        manifest = load_manifests().get(backend)
        root_dir = plan.get("backends_root")
        if (
            manifest is None
            or read_install_marker(manifest, root_dir) is None
            or not Path(plan["argv"][0]).exists()
        ):
            raise ConvertError(
                "backend_not_installed",
                "The {0} backend is not installed.".format(backend),
                "Install it first: mlx-agent backend install {0}; convert never installs runtimes.".format(backend),
            )
    elif which(EXECUTABLE) is None:
        raise ConvertError(
            "runtime_not_installed",
            "The {0} executable is not installed.".format(EXECUTABLE),
            "Install mlx-lm yourself (pip install mlx-lm); convert never installs runtimes.",
        )
    if _source_kind(plan) == "gguf":
        missing = (module_present or _default_module_present)(GGUF_REQUIRED_MODULES)
        if missing:
            raise ConvertError(
                "runtime_not_installed",
                "GGUF conversion needs these modules in this interpreter: {0}.".format(
                    ", ".join(missing)
                ),
                "Install them yourself (for example: uv pip install torch transformers gguf); convert never installs runtimes.",
            )
        if not Path(plan["source"]["path"]).is_file():
            raise ConvertError(
                "source_not_found",
                "The source GGUF is no longer at {0}.".format(plan["source"]["path"]),
                "Re-run convert scan and preview a fresh plan.",
            )
    elif model_present is not None and not model_present(plan["repo"]):
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

    if backend:
        try:
            sync_ports(manifest, plan.get("backends_root"))
        except BackendError as error:
            raise ConvertError(error.code, str(error), error.remediation) from error
    root.mkdir(parents=True, exist_ok=True)
    slug = _slug(plan)
    log_path = root / "{0}.log".format(slug)
    if spawn is None and backend:
        environment = backend_environment()

        def spawn(argv, log_path):
            return spawn_with_env(argv, log_path, environment)

    spawn = spawn or _default_spawn
    pid = spawn(plan["argv"], str(log_path))
    receipt = {
        "schema_version": CONVERT_RECEIPT_SCHEMA_VERSION,
        "kind": CONVERT_RECEIPT_KIND,
        "repo": plan["repo"],
        "source": dict(plan.get("source") or {"kind": "hf-cache", "repo": plan["repo"]}),
        "slug": slug,
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
    if backend:
        receipt["backend"] = backend
    _write_receipt(root, receipt, "{0}.json".format(slug))
    return {"status": "started", "receipt": receipt}


def load_receipts(receipts_dir=None):
    """Read every convert receipt; used to pair GGUF sources with their outputs."""
    root = receipts_root(receipts_dir)
    receipts = []
    if not root.is_dir():
        return receipts
    for path in sorted(root.glob("*.json")):
        receipt = _read_receipt(path)
        if receipt is not None:
            receipts.append(receipt)
    return receipts


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
            "source": _source_kind(receipt),
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
        _write_receipt(root, receipt, "{0}.json".format(_slug(receipt)))
        entry["state"] = exit_status
        entry["completed_at"] = receipt["completed_at"]
        entries.append(entry)
    return entries
