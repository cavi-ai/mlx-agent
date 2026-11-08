"""Confirmation-gated local MLX quantization jobs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .backends import (
    QUANTIZED_BITS,
    UNQUANTIZED_BITS,
    BackendError,
    backend_environment,
    backend_python,
    conversion_bits,
    download_name,
    download_path,
    load_manifests,
    port_bits,
    port_converter,
    port_downloads,
    port_requirements,
    recipe_for,
    read_install_marker,
    spawn_with_env,
    sync_ports,
)
from .gguf import GGUF_PORT_BITS, GGUFError, describe_gguf, find_projector, file_signature, read_gguf_header, PORT_PROJECTORS
from .intake_source import validate_subfolder
from .serve import (
    _argv_matches,
    _pid_alive,
    _pid_command,
    _terminate_pid,
)
from .transactions import _atomic_in_directory
from .wiring import _MODEL
from .source_evidence import completed_fetch, fingerprint, safe_path, read_json


CONVERT_RECEIPT_SCHEMA_VERSION = "1.0"
CONVERT_RECEIPT_KIND = "convert"
Q_BITS_CHOICES = QUANTIZED_BITS
# The music requantizer reloads an exported MLX checkpoint, checked at these widths only.
MUSIC_REQUANTIZE_BITS = (4, 8)
EXECUTABLE = "mlx_lm.convert"
GGUF_RUNNER = Path(__file__).resolve().with_name("gguf_runner.py")
MUSIC_REQUANTIZE_RUNNER = Path(__file__).resolve().with_name("music_requantize_runner.py")
GGUF_REQUIRED_MODULES = ("torch", "transformers", "gguf")
GGUF_PORT_REQUIRED_MODULES = ("gguf", "mlx")
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
    """``<cache>/models--org--name/snapshots/<revision or refs/main>`` when that snapshot is cached, else None.

    A revision-pinned download writes no ``refs/main``; its only snapshot is then the cached one.
    """
    from .model_doctor import default_hf_cache

    entry = (Path(hf_cache) if hf_cache else default_hf_cache()) / "models--{0}".format(repo.replace("/", "--"))
    if revision is None:
        try:
            revision = (entry / "refs" / "main").read_text(encoding="utf-8").strip()
        except OSError:
            try:
                snapshots = [path for path in (entry / "snapshots").iterdir() if path.is_dir()]
            except OSError:
                return None
            return snapshots[0] if len(snapshots) == 1 else None
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
                 subfolder=None, hf_cache=None, source_path=None, downloads_dir=None, fetch_receipt=None,
                 dependency_fetch_receipts=None, require_completion=False):
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
    _validate_q_bits(q_bits, (UNQUANTIZED_BITS,))
    if subfolder is not None:
        try:
            validate_subfolder(subfolder)
        except ValueError as error:
            raise ConvertError("invalid_arguments", str(error), "Pass --subfolder as the folder path inside the repository.") from error
    primary, dependencies = _source_records(fetch_receipt, dependency_fetch_receipts, source_path)
    if primary:
        if primary["origin"]["repo"] != repo or primary["origin"].get("file") is not None or primary["origin"].get("subfolder") != subfolder:
            raise ConvertError("source_mismatch", "The completed fetch does not match this repo checkpoint.", "Select the exact downloaded checkpoint.")
        source_path = primary["path"]
    name = repo.split("/", 1)[1] + ("-" + subfolder.replace("/", "-") if subfolder else "")
    if out is None:
        out = "{0}-MLX-{1}bit".format(name, q_bits)
    local_source = Path(source_path).expanduser().absolute() if source_path is not None else cached_snapshot(repo, hf_cache)
    if source_path is not None and not local_source.is_dir():
        raise ConvertError("source_not_found", "The downloaded source directory is missing.",
                           "Download or select the source again before previewing.")
    hf_path = str(local_source) if local_source is not None else repo
    if subfolder and source_path is None:
        folder = cached_subfolder(repo, subfolder, hf_cache)
        if folder is None:
            raise ConvertError(
                "subfolder_not_cached", "{0}/{1} is not in the Hugging Face cache.".format(repo, subfolder),
                "Download it first: mlx-agent intake fetch {0}/{1}.".format(repo, subfolder),
            )
        hf_path = str(folder)
    flags = ["--hf-path", hf_path, "--mlx-path", str(out)]
    if q_bits != UNQUANTIZED_BITS:
        flags += ["--quantize", "--q-bits", str(q_bits)]
    plan = {
        "repo": repo,
        "source": dict({"kind": "hf-cache", "repo": repo}, **({"subfolder": subfolder} if subfolder else {})),
        "slug": "{0}-{1}bit".format(name, q_bits),
        "q_bits": q_bits,
        "out": str(out),
    }
    if hf_path != repo:
        source = Path(hf_path)
        # Bind the reviewed plan to the local source, including config contents
        # and weight identities. Snapshot links may point to cache-owned blobs.
        plan["source"]["path"] = str(source)
        plan["source"]["fingerprint"] = local_source_fingerprint(source)
    if q_bits == UNQUANTIZED_BITS and backend in (None, "mlx-lm"):
        raise ConvertError("invalid_arguments", "This conversion has no unquantized recipe.", _bits_hint(Q_BITS_CHOICES))
    if model_type is not None:
        plan["model_type"] = model_type
    if backend in (None, "mlx-lm"):
        plan["argv"] = [EXECUTABLE] + flags
        return _finish_source_plan(plan, primary, dependencies, require_completion)
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
    widths = conversion_bits(model_type, backend, manifests)
    if q_bits == UNQUANTIZED_BITS and q_bits not in widths:
        raise ConvertError("invalid_arguments", "This conversion has no unquantized recipe.", _bits_hint(widths))
    if q_bits not in widths:
        owner = "{0} port".format(model_type) if port_bits(model_type, backend, manifests) else "{0} converter".format(backend)
        raise ConvertError(
            "invalid_arguments",
            "The {0} converts at {1} bits only.".format(owner, _joined(widths)),
            _bits_hint(widths),
        )
    if backend == "mlx-audio" and _is_quantized_music_source(hf_path) and q_bits not in MUSIC_REQUANTIZE_BITS:
        raise ConvertError(
            "invalid_arguments",
            "Exported music checkpoints requantize at {0} bits only.".format(_joined(MUSIC_REQUANTIZE_BITS)),
            _bits_hint(MUSIC_REQUANTIZE_BITS),
        )
    python = backend_python(manifest, backends_root_dir)
    plan["backend"] = backend
    plan["backends_root"] = str(python.parent.parent.parent)
    converter = port_converter(model_type, backend, manifests)
    recipe = recipe_for(repo, manifests) if converter else None
    if recipe and recipe["backend"] == backend and recipe["port"] == model_type:
        flags = _recipe_flags(recipe, flags, hf_cache, local_source=local_source if source_path is not None else None, dependencies=dependencies)
        plan["recipe"] = {key: recipe[key] for key in ("base", "base_revision", "lora", "lora_scale")}
    requires = _required_sources(repo, model_type, manifests, hf_cache, dependencies=dependencies if source_path is not None else None) if converter else {}
    for name, path in requires.items():
        flags += ["--requires", "{0}={1}".format(name, path)]
    if requires:
        plan["requires"] = requires
    if primary and primary.get("support"):
        recorded_support = primary["support"]["downloads_dir"]
        if downloads_dir is not None and safe_path(downloads_dir) != safe_path(recorded_support):
            raise ConvertError("source_mismatch", "The support directory differs from the completed managed fetch.", "Use the support directory recorded by that fetch.")
        downloads_dir = recorded_support
    if source_path is not None and converter and port_downloads(model_type, manifests) and downloads_dir is None:
        raise ConvertError("requirement_not_cached", "Managed conversion needs an explicit support directory.", "Download support into the managed source and pass its fetch receipt or --downloads-dir.")
    downloads = _required_downloads(repo, model_type, manifests, downloads_dir) if converter else {}
    for name, path in downloads.items():
        flags += ["--download", "{0}={1}".format(name, path)]
    if downloads:
        plan["downloads"] = downloads
        plan["downloads_dir"] = str(downloads_dir) if downloads_dir is not None else None
    if backend == "mlx-audio" and _is_quantized_music_source(hf_path):
        plan["argv"] = [str(python), str(MUSIC_REQUANTIZE_RUNNER)] + flags
    else:
        plan["argv"] = [str(python), "-m", converter or manifest["convert"]] + flags
    ports = sorted(name for names in (manifest.get("ports") or {}).values() for name in names)
    if ports:
        plan["ports"] = ports
    if converter:
        plan["port_converter"] = converter
    return _finish_source_plan(plan, primary, dependencies, require_completion)


def _is_quantized_music_source(path):
    """Exported MLX music weights need the music loader, not the raw converter."""
    try:
        config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(config, dict) and config.get("model_type") == "minimax_music3"
            and isinstance(config.get("quantization"), dict) and bool(config["quantization"]))


def _required_sources(repo, model_type, manifests, hf_cache, dependencies=None):
    """Cached snapshots of the other repositories a port's converter reads, each holding the files it needs."""
    found = {}
    for name, files in sorted(port_requirements(model_type, manifests).items()):
        if dependencies is not None:
            proof = next((item for item in dependencies if item["origin"]["repo"] == name), None)
            snapshot = Path(proof["path"]) if proof else None
        else:
            snapshot = cached_snapshot(name, hf_cache)
        missing = [item for item in files if snapshot is None or not (snapshot / item).is_file()]
        if missing:
            raise ConvertError(
                "requirement_not_cached",
                "{0} also needs {1} from {2} in the Hugging Face cache.".format(model_type, ", ".join(missing), name),
                "Download it first: mlx-agent intake fetch {0} --model-type {1}.".format(repo, model_type),
            )
        found[name] = str(snapshot)
    return found


def _required_downloads(repo, model_type, manifests, downloads_dir):
    """The files a port's converter reads from outside Hugging Face, each where ``intake fetch`` stored it."""
    found = {}
    for entry in port_downloads(model_type, manifests):
        path = download_path(entry, downloads_dir)
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ConvertError(
                "requirement_not_cached",
                "{0} also needs {1} from {2}.".format(model_type, download_name(entry), entry["url"].split("/")[2]),
                "Download it first: mlx-agent intake fetch {0} --model-type {1}.".format(repo, model_type),
            )
        found[download_name(entry)] = str(path)
    return found


def _recipe_flags(recipe, flags, hf_cache, local_source=None, dependencies=()):
    """Converter flags for a recipe: the cached base snapshot and the cached LoRA, both local."""
    if local_source is not None:
        proof = next((item for item in dependencies if item["origin"]["repo"] == recipe["base"]
                      and item["origin"]["revision"].lower() == recipe["base_revision"].lower()
                      and item["origin"].get("subfolder") is None), None)
        base = Path(proof["path"]) if proof else None
        snapshot = Path(local_source)
    else:
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


def local_source_fingerprint(path):
    source = Path(path)
    entries = []
    for item in sorted(source.rglob("*")):
        if item.is_file():
            info = item.stat()
            content = hashlib.sha256(item.read_bytes()).hexdigest() if item.name.endswith(".json") and info.st_size <= 1024 * 1024 else ""
            entries.append([str(item.relative_to(source)), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, content])
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def _source_records(fetch_receipt, dependency_fetch_receipts, selected_path=None):
    try:
        primary = completed_fetch(fetch_receipt, selected_path) if fetch_receipt else None
        dependencies = [completed_fetch(value, bundle_root=True) for value in dependency_fetch_receipts or []]
        repos = [item["origin"]["repo"].casefold() for item in dependencies]
        if len(dependencies) > 32 or len(set(repos)) != len(repos):
            raise ValueError("Dependency fetch receipts must name distinct repositories (at most 32).")
        return primary, dependencies
    except (OSError, ValueError) as error:
        raise ConvertError("source_evidence_invalid", str(error), "Download the exact immutable source and dependencies into fresh managed folders.") from error


def _finish_source_plan(plan, primary, dependencies, require_completion):
    if primary:
        plan["source"].update(origin=primary["origin"], fetch_receipt=primary["fetch_receipt"], fingerprint=primary["fingerprint"])
        if primary.get("support"):
            plan["source"]["support"] = primary["support"]
    if dependencies:
        for dependency in dependencies:
            if not any(value == dependency["path"] or value.endswith("=" + dependency["path"]) for value in plan["argv"]):
                raise ConvertError("unused_dependency", "A dependency receipt is not used by this conversion plan.", "Pass only this converter's reviewed dependencies.")
        plan["source"]["dependencies"] = dependencies
    if plan.get("projector"):
        plan["source"]["projector"] = dict(plan["projector"])
    if plan.get("recipe"):
        plan["source"]["recipe"] = dict(plan["recipe"])
    if require_completion:
        plan["require_completion"] = True
    return _finalize_plan(plan)


def _finalize_plan(plan):
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["preview_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def _joined(widths):
    widths = [str(width) for width in widths]
    return widths[0] if len(widths) == 1 else "{0} or {1}".format(", ".join(widths[:-1]), widths[-1])


def _bits_hint(widths):
    return "Pass --q-bits {0}.".format(_joined(widths))


def _validate_q_bits(q_bits, extra=()):
    if not isinstance(q_bits, int) or isinstance(q_bits, bool):
        raise ConvertError("invalid_arguments", "q_bits must be an integer.", _bits_hint(Q_BITS_CHOICES))
    if q_bits not in Q_BITS_CHOICES + tuple(extra):
        raise ConvertError(
            "invalid_arguments",
            "q_bits must be one of {0}.".format(list(Q_BITS_CHOICES + tuple(extra))),
            _bits_hint(Q_BITS_CHOICES),
        )


def _slugify(value, q_bits):
    slug = _UNSAFE_SLUG.sub("-", str(value)).strip("-.") or "model"
    return "{0}-{1}bit".format(slug[:96], q_bits)


def plan_gguf_convert(gguf_path, q_bits=4, out=None, runner=None, describe=describe_gguf,
                      projector=None, require_projector=False, fetch_receipt=None,
                      dependency_fetch_receipts=None, require_completion=False):
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
    draft = entry.get("draft")
    if draft is not None and draft.get("port") is None:
        raise ConvertError(
            "unsupported_draft",
            "{0} is a {1} speculative-decoding drafter for {2}; it runs only beside that model and no port converts it.".format(
                location.name, entry.get("architecture"), draft.get("target") or "its target model"),
            "Convert the target model itself; DeepSeek-V4 DSpark is the only drafter family mlx-agent converts.",
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
    if draft is not None:
        plan["draft"] = dict(draft)
    port = draft["port"] if draft is not None else entry.get("port")
    if port is not None and q_bits not in GGUF_PORT_BITS:
        raise ConvertError(
            "invalid_arguments",
            "The {0} port converts GGUF files at {1} bits only.".format(port, _joined(GGUF_PORT_BITS)),
            _bits_hint(GGUF_PORT_BITS),
        )
    if port is not None:
        plan["port_converter"] = port
        plan["argv"].extend(["--port", port])
        selected_projector = _explicit_projector(location, port, projector) if projector else find_projector(location, port)
        if selected_projector is not None:
            selected_projector["signature"] = file_signature(selected_projector["path"])
            selected_projector["fingerprint"] = fingerprint(Path(selected_projector["path"]).resolve())
            plan["projector"] = selected_projector
            plan["argv"].extend(["--projector", selected_projector["path"]])
    elif projector:
        raise ConvertError("invalid_projector", "This GGUF converter does not support a projector.", "Select a supported vision model.")
    if require_projector:
        if "projector" not in plan:
            raise ConvertError("projector_required", "The reviewed vision projector is missing or ambiguous.", "Download and select the exact matching projector before updating.")
        plan["require_projector"] = True
    primary, dependencies = _source_records(fetch_receipt, dependency_fetch_receipts, str(location))
    if primary and plan.get("projector"):
        proof = read_json(primary["fetch_receipt"])
        root = safe_path(proof["local_dir"])
        target = safe_path(plan["projector"]["path"])
        relative = str(target.relative_to(root)) if target.is_relative_to(root) else None
        if relative not in primary["origin"].get("additional_files", []):
            raise ConvertError("source_mismatch", "The projector is outside the reviewed download file set.", "Include it with --additional-file at the same immutable revision.")
        plan["projector"]["origin"] = dict(primary["origin"], file=relative, additional_files=[])
        plan["projector"]["fetch_receipt"] = primary["fetch_receipt"]
    return _finish_source_plan(plan, primary, dependencies, require_completion)


def _explicit_projector(model, port, projector):
    target = safe_path(projector)
    try:
        source = read_gguf_header(model)["metadata"]
        candidate = read_gguf_header(target)["metadata"]
        width = source.get("{0}.embedding_length".format(source.get("general.architecture")))
        if (target == Path(model).resolve() or candidate.get("general.architecture") != "clip"
                or candidate.get("clip.projector_type") != PORT_PROJECTORS.get(port)
                or candidate.get("clip.vision.projection_dim") != width):
            raise ValueError("The selected projector does not match this model's vision type and embedding width.")
    except (GGUFError, OSError, ValueError) as error:
        raise ConvertError("invalid_projector", str(error), "Select the matching projector from the reviewed download.") from error
    return {"path": str(target), "name": target.name, "bytes": target.stat().st_size, "signature": file_signature(target), "fingerprint": fingerprint(target.resolve())}


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


def _write_receipt(root, receipt, filename, expected=None):
    from .conversion_monitor import receipt_lock
    path = Path(root) / filename
    with receipt_lock(path):
        if expected is not None and _read_receipt(path) != expected:
            return False
        content = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        _atomic_in_directory(root, filename, content, 0o600)
    return True


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
                  pid_alive=None, module_present=None, wait_for_compute_lease=False, attempt_id=None):
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
    if attempt_id is not None and (not plan.get("require_completion") or not isinstance(attempt_id, str)
                                   or re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None):
        raise ConvertError("invalid_arguments", "A conversion attempt requires --require-completion and a 32-character hexadecimal token.", "Use the durable queue to create a new attempt.")
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
        required = GGUF_PORT_REQUIRED_MODULES if plan.get("port_converter") else GGUF_REQUIRED_MODULES
        missing = (module_present or _default_module_present)(required)
        if missing:
            raise ConvertError(
                "runtime_not_installed",
                "GGUF conversion needs these modules in this interpreter: {0}.".format(
                    ", ".join(missing)
                ),
                "Install them yourself (for example: uv pip install {0}); convert never installs runtimes.".format(" ".join(required)),
            )
        for label, path in (("source GGUF", plan["source"]["path"]), ("vision projector", (plan.get("projector") or {}).get("path"))):
            if path is not None and not Path(path).is_file():
                raise ConvertError(
                    "source_not_found",
                    "The {0} is no longer at {1}.".format(label, path),
                    "Re-run convert scan and preview a fresh plan.",
                )
    elif plan.get("source", {}).get("path") is not None:
        if not Path(plan["source"]["path"]).is_dir():
            raise ConvertError("source_not_found", "The reviewed local source is no longer present.", "Preview the conversion again.")
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
        if attempt_id is not None and existing and existing.get("attempt_id") == attempt_id:
            raise ConvertError("attempt_exists", "This conversion attempt already has a receipt.", "Inspect its existing receipt before retrying with a new attempt.")
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
    launch_id = uuid.uuid4().hex if plan.get("require_completion") else None
    slug = _slug(plan) + ("-" + launch_id if launch_id else "")
    receipt_path = root / (slug + ".json")
    if receipt_path.is_symlink() or (launch_id and receipt_path.exists()):
        raise ConvertError("receipt_exists", "This conversion launch already has a receipt.", "Inspect conversion status before retrying.")
    log_path = root / "{0}.log".format(slug)
    if spawn is None:
        environment = backend_environment()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"

        def spawn(argv, log_path):
            return spawn_with_env(argv, log_path, environment)

    spawn = spawn or _default_spawn
    started_at = now()
    monitor_argv = None
    if launch_id:
        request_path = root / (".monitor-" + launch_id + ".json")
        request = {"launch_id": launch_id, "started_at": started_at,
                   "attempt_id": attempt_id,
                   "wait_for_compute_lease": wait_for_compute_lease,
                   "receipt_path": str(receipt_path.absolute()), "plan": plan}
        _atomic_in_directory(root, request_path.name, (json.dumps(request, sort_keys=True) + "\n").encode(), 0o600)
        monitor_argv = [sys.executable, str(Path(__file__).with_name("conversion_monitor.py")), "--request", str(request_path.absolute())]
    pid = spawn(monitor_argv or plan["argv"], str(log_path))
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
        "started_at": started_at,
        "preview_hash": plan["preview_hash"],
        "completed_at": None,
        "exit_status": None,
    }
    if backend:
        receipt["backend"] = backend
    if plan.get("model_type"):
        receipt["model_type"] = plan["model_type"]
    if plan.get("downloads_dir"):
        receipt["downloads_dir"] = plan["downloads_dir"]
    if launch_id:
        receipt.update(require_completion=True, launch_id=launch_id, monitor_argv=monitor_argv,
                       wait_for_compute_lease=wait_for_compute_lease, compute_lease_registered=False,
                       completion_observed=False, exit_code=None)
    if attempt_id is not None:
        receipt["attempt_id"] = attempt_id
    _write_receipt(root, receipt, "{0}.json".format(slug))
    result = {"status": "started", "receipt": receipt}
    if launch_id:
        result["receipt_path"] = str(receipt_path.absolute())
    return result


def register_monitor_lease(path, expected):
    from .conversion_monitor import receipt_lock
    with receipt_lock(path):
        current = read_json(path)
        keys = ("launch_id", "attempt_id", "pid", "source", "argv", "preview_hash", "out", "started_at")
        if any(current.get(key) != expected.get(key) for key in keys) or not current.get("wait_for_compute_lease"):
            raise ConvertError("receipt_changed", "The conversion receipt changed before compute registration.", "Inspect conversion status before retrying.")
        current["compute_lease_registered"] = True
        _atomic_in_directory(Path(path).parent, Path(path).name, (json.dumps(current, sort_keys=True) + "\n").encode(), 0o600)
        expected.update(current)


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


def _processes_quiescent(receipt):
    """Prove the recorded isolated group has no live members without signalling."""
    from .compute import group_members
    group = receipt.get("execution_group")
    if (not receipt.get("require_completion") or not isinstance(group, dict)
            or type(group.get("pgid")) is not int or group["pgid"] != receipt["pid"]
            or not isinstance(group.get("leader_identity"), str) or not group["leader_identity"]):
        return False
    try:
        return not group_members(group["pgid"])
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _check_completed_output(receipt):
    if receipt.get("completion_observed") is not True or receipt.get("exit_code") != 0:
        raise ValueError("The conversion has no observed successful worker completion.")
    output = safe_path(receipt["out"])
    evidence = read_json(output / "mlx-converter.json")
    expected = {"schema_version": "1.1", "tool": "mlx-agent.convert", "source": receipt["source"],
                "q_bits": receipt["q_bits"], "backend": receipt.get("backend", "mlx-lm"),
                "model_type": receipt.get("model_type"), "preview_hash": receipt.get("preview_hash"),
                "launch_id": receipt["launch_id"], "attempt_id": receipt.get("attempt_id"),
                "out": str(output)}
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError("The output provenance does not match its completed conversion receipt.")
    if evidence.get("output_fingerprint") != fingerprint(output, exclude=("mlx-converter.json",)):
        raise ValueError("The converted output changed after observed completion.")


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
            "source": _source_kind(receipt), "source_details": dict(receipt.get("source") or {}),
            "preview_hash": receipt.get("preview_hash"), "backend": receipt.get("backend", "mlx-lm"),
            "model_type": receipt.get("model_type"), "downloads_dir": receipt.get("downloads_dir"), "completion_observed": receipt.get("completion_observed", False),
            "exit_code": receipt.get("exit_code"), "exit_status": receipt.get("exit_status"),
            "attempt_id": receipt.get("attempt_id"),
            "processes_quiescent": _processes_quiescent(receipt),
            "q_bits": receipt["q_bits"],
            "out": receipt["out"],
            "pid": receipt["pid"],
            "log_path": receipt.get("log_path"),
            "started_at": receipt.get("started_at"),
            "completed_at": receipt.get("completed_at"),
        }
        if receipt.get("completed_at"):
            entry["state"] = receipt.get("exit_status") or "done"
            if receipt.get("require_completion") and entry["state"] == "done":
                try:
                    _check_completed_output(receipt)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    entry["state"] = "failed"
                    entry["error"] = str(error)
            entries.append(entry)
            continue
        if pid_alive(receipt["pid"]):
            command = pid_command(receipt["pid"])
            entry["state"] = "running" if _argv_matches(
                {"argv": receipt.get("monitor_argv") or receipt["argv"], "port": None,
                 "repo": receipt["monitor_argv"][-1] if receipt.get("monitor_argv") else receipt["repo"]},
                command,
                require_port=False,
            ) else "unknown"
            entries.append(entry)
            continue
        if receipt.get("require_completion"):
            entry["state"] = "failed"
            entry["error"] = "The conversion monitor exited without observed worker completion."
            entries.append(entry)
            continue
        exit_status = "done" if Path(receipt["out"]).exists() else "failed"
        original = dict(receipt)
        receipt["completed_at"] = now()
        receipt["exit_status"] = exit_status
        if not _write_receipt(root, receipt, "{0}.json".format(_slug(receipt)), expected=original):
            entry["state"] = "unknown"
            entries.append(entry)
            continue
        entry["state"] = exit_status
        entry["completed_at"] = receipt["completed_at"]
        entries.append(entry)
    return entries
