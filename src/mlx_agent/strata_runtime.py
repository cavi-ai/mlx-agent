"""Pinned, optional Strata-MLX serving for explicitly prepared local GGUF profiles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import runpy
import shlex
import sys
import tempfile
from pathlib import Path

ENGINE_REVISION = "ac0523f51c9935805904de889747d2b4ce011516"
HTTP_REVISION = "a0116b2782a5b82622358c04a06496485d12b2d6"
MODEL_MARKER = "strata-model.json"
INSTALL_MARKER = "installed.json"
_FIRST = re.compile(r"Qwen3\.8-Flash-Next-GSQ-RCO-(Q2_0|IQ3_S)-00001-of-00002\.gguf")
# GGUF block layouts implemented by the pinned engine. No tensor data is loaded here.
_BLOCKS = {0: (1, 4), 1: (1, 2), 30: (1, 2), 2: (32, 18), 6: (32, 22), 8: (32, 34),
           11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
           20: (32, 18), 23: (256, 136), 42: (64, 18), 16: (256, 66), 17: (256, 74),
           18: (256, 98), 21: (256, 110), 22: (256, 82), 29: (256, 56)}


class StrataError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.remediation = "Install the optional Strata runtime and prepare a supported local GGUF profile; existing MLX models are unchanged."


def runtime_root():
    return Path(os.environ.get("MLX_AGENT_STRATA_ROOT", Path.home() / ".cache/mlx-agent/strata")).expanduser().absolute()


def engine_launcher(root):
    root = Path(root).absolute()
    program = "import sys; sys.path.insert(0, " + repr(str(root / "engine")) + "); from strata_mlx.serve import main; main()"
    return "#!/bin/sh\nexec " + shlex.quote(str(root / "engine/.venv/bin/python")) + " -c " + shlex.quote(program) + ' "$@"\n'


def _json(path):
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Unsafe or oversized metadata file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Metadata must be an object")
    return value


def source_digest(root):
    root = Path(root)
    digest = hashlib.sha256()
    paths = []
    for folder in ("engine/strata_mlx", "http/serve", "http/tools"):
        directory = root / folder
        if directory.is_symlink():
            raise ValueError("Runtime source directories must not be links")
        paths.extend(directory.rglob("*.py"))
    paths.append(root / "engine/uv.lock")
    paths.append(root / "strata-mlx")
    for path in sorted(paths):
        if not path.exists():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Runtime source must remain inside its installation")
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def installed_runtime(root=None):
    root = Path(root) if root is not None else runtime_root()
    try:
        marker = _json(root / INSTALL_MARKER)
        python = root / "engine/.venv/bin/python"
        engine = root / "strata-mlx"
        required = (root / "engine/strata_mlx/serve.py", root / "http/serve/server.py", root / "http/tools/strata_tokenizer.py")
        if (marker.get("engine_revision") != ENGINE_REVISION or marker.get("http_revision") != HTTP_REVISION
                or not python.is_file() or not engine.is_file() or engine.is_symlink() or not os.access(engine, os.X_OK)
                or engine.read_text() != engine_launcher(root) or not all(p.is_file() for p in required)
                or marker.get("source_sha256") != source_digest(root)):
            raise ValueError("Missing, modified or incompatible Strata installation")
        return {"root": str(root.absolute()), "python": str(python.absolute()), "engine": str(engine.absolute()), "fingerprint": marker["source_sha256"]}
    except (OSError, ValueError, TypeError) as error:
        raise StrataError("strata_not_installed", str(error)) from error


def _local(root, name, directory=False):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("Profile paths must be relative and remain inside the model directory")
    path = root / name
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Profile paths must not contain symbolic links")
    if not path.resolve().is_relative_to(root.resolve()) or not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError("Missing profile asset: " + name)
    return path


def file_identity(path):
    stat = path.stat()
    with path.open("rb") as handle:
        header = hashlib.sha256(handle.read(8 << 20)).hexdigest()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "device": stat.st_dev, "inode": stat.st_ino, "header_sha256": header}


def read_model(path):
    root = Path(path).expanduser().absolute()
    try:
        if root.is_symlink():
            raise ValueError("The model profile directory must not be a symbolic link")
        config = _json(root / "config.json")
        if config.get("serving_runtime") != "strata" or config.get("model_type") != "qwen4_exp":
            raise StrataError("strata_model_required", "Strata requires a prepared Qwen3.8-Flash-Next GGUF profile; MLX VQ weights are not supported.")
        profile = _json(root / MODEL_MARKER)
        if set(profile) != {"schema_version", "gguf", "mtp"} or type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
            raise ValueError("Unsupported Strata model profile")
        first = _local(root, profile["gguf"])
        if not _FIRST.fullmatch(first.name):
            raise ValueError("Only the tested Q2_0 and IQ3_S Qwen3.8-Flash-Next GGUF pairs are supported")
        second = _local(root, str(first.relative_to(root)).replace("00001-of-00002", "00002-of-00002"))
        from mlx_agent.gguf import read_gguf_header
        for index, weight in enumerate((first, second)):
            header = read_gguf_header(weight)
            architecture = header["metadata"].get("general.architecture")
            if architecture != "qwen4exp" and not (index == 1 and architecture is None and header["tensor_count"] == 1):
                raise ValueError("GGUF architecture is not Qwen3.8-Flash-Next")
        assets = [root / "config.json", root / MODEL_MARKER, first, second]
        tokenizer = _local(root, "tokenizer", directory=True)
        for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja"):
            assets.append(_local(root, "tokenizer/" + name))
        mtp = None
        mtp_bytes = 0
        if profile["mtp"] is not None:
            mtp = _local(root, profile["mtp"], directory=True)
            manifest_path = _local(root, str(mtp.relative_to(root) / "mtp-manifest.json"))
            if manifest_path.stat().st_size > 1024 * 1024:
                raise ValueError("Oversized MTP manifest")
            tensors = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(tensors, list) or len(tensors) != 31:
                raise ValueError("A complete, verified MTP head is required")
            assets.append(mtp / "mtp-manifest.json")
            for tensor in tensors:
                if not isinstance(tensor, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(tensor.get("sha256", ""))):
                    raise ValueError("MTP tensor verification metadata is missing")
                location = _local(root, str(mtp.relative_to(root) / tensor["file"]))
                if location.stat().st_size != tensor["bytes"]:
                    raise ValueError("MTP tensor size changed")
                mtp_bytes += location.stat().st_size
                assets.append(location)
        identities = {str(asset.relative_to(root)): file_identity(asset) for asset in assets}
        fingerprint = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
        # The second shard is the on-disk n-gram lookup table, not resident
        # weights. Bound dense expansion by 20%, retain the full MTP file
        # sizes, and reserve the pinned engine's 5 GB working allowance plus
        # its 30 KB/token float16 cache allowance at the configured capacity.
        memory_estimate = math.ceil(first.stat().st_size * 1.2 + mtp_bytes + 5e9 + 131072 * 30000)
        return {"root": str(root), "gguf": str(first), "tokenizer": str(tokenizer), "mtp": str(mtp) if mtp else None,
                "quantization": "GSQ-RCO " + _FIRST.fullmatch(first.name).group(1),
                "mtp_bytes": mtp_bytes,
                "fingerprint": fingerprint, "weight_bytes": first.stat().st_size + second.stat().st_size,
                "memory_estimate_bytes": memory_estimate}
    except StrataError:
        raise
    except FileNotFoundError as error:
        raise StrataError("strata_model_incomplete", str(error)) from error
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise StrataError("strata_model_invalid", str(error)) from error


def memory_estimate(model, expert_memory_gb=None, context_tokens=131072):
    """Conservative resident estimate for the reviewed paging budget and context."""
    if type(context_tokens) is not int or not 512 <= context_tokens <= 131072:
        raise StrataError("invalid_arguments", "Strata context must be an integer from 512 to 131072 tokens.")
    if expert_memory_gb is None:
        return model["memory_estimate_bytes"] - 131072 * 30000 + context_tokens * 30000
    if (type(expert_memory_gb) not in (int, float) or not math.isfinite(expert_memory_gb)
            or not 0.5 <= expert_memory_gb <= 64):
        raise StrataError("invalid_arguments", "Expert memory must be a finite number from 0.5 to 64 GB.")
    from mlx_agent.gguf import GGUFError, read_gguf_header
    try:
        header = read_gguf_header(model["gguf"], include_tensors=True)
        experts = dense = floats = 0
        file_bytes = Path(model["gguf"]).stat().st_size
        for tensor in header["tensors"]:
            block, size = _BLOCKS[tensor["type"]]
            count = math.prod(tensor["shape"])
            byte_count = count // block * size
            if count % block or header["tensor_data_offset"] + tensor["offset"] + byte_count > file_bytes:
                raise ValueError("Tensor data exceeds its GGUF file")
            if tensor["name"].endswith("_exps.weight"):
                experts += byte_count
            else:
                dense += byte_count
                if tensor["type"] == 14 and tensor["name"] != "output.weight":
                    floats += count * 2
        if not experts or not dense:
            raise ValueError("Expert and dense tensor sizes are required")
        mtp_bytes = model.get("mtp_bytes", 0)
        # Count dense expansion, Q6_K prompt copies, full draft files, runtime
        # working memory and context cache. The n-gram companion remains on disk.
        return math.ceil(min(experts, expert_memory_gb * 1e9) + dense * 2 + floats
                         + mtp_bytes + 5e9 + context_tokens * 30000)
    except (GGUFError, OSError, ValueError, KeyError) as error:
        raise StrataError("strata_memory_unknown", "Cannot estimate paged Strata memory: " + str(error)) from error


def frontend_config(model, runtime, speculation=False, expert_memory_gb=None, context_tokens=131072):
    memory_estimate(model, expert_memory_gb, context_tokens)
    args = ["--native", model["gguf"], "--spec", "4" if speculation else "1",
            "--suffix-draft", "auto" if speculation else "0", "--max-context", str(context_tokens)]
    if expert_memory_gb is not None:
        args += ["--expert-memory", str(expert_memory_gb)]
    if model["mtp"]:
        args += ["--mtp", model["mtp"]]
    return {"exe": runtime["engine"], "args": args, "cwd": runtime["root"], "tokenizer": model["tokenizer"],
            "model_name": model["root"], "host": "127.0.0.1", "open_browser": False, "prevent_sleep": False,
            "sampling": {"temperature": 0.0}, "engine_silence_s": 900}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--speculation", choices=["on", "off"], default="off")
    parser.add_argument("--expert-memory-gb", type=float)
    parser.add_argument("--context-tokens", type=int, default=131072)
    parser.add_argument("--expected-model-fingerprint", required=True)
    parser.add_argument("--expected-runtime-fingerprint", required=True)
    args = parser.parse_args()
    runtime = installed_runtime(args.runtime_root)
    model = read_model(args.model)
    if model["fingerprint"] != args.expected_model_fingerprint or runtime["fingerprint"] != args.expected_runtime_fingerprint:
        raise StrataError("preview_stale", "The model assets or Strata runtime changed after the serving preview.")
    os.environ.update(MLX_AGENT_STRATA_ROOT=runtime["root"], HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", MLX_ENABLE_TF32="0")
    # The only config reaching the frontend is constructed here; model files cannot supply code or commands.
    config = frontend_config(model, runtime, args.speculation == "on", args.expert_memory_gb, args.context_tokens)
    source = Path(runtime["root"]) / "http"
    sys.path[:0] = [str(source), str(source / "tools")]
    with tempfile.TemporaryDirectory(prefix="mlx-agent-strata-") as directory:
        location = Path(directory) / "server.json"
        location.write_text(json.dumps(config), encoding="utf-8")
        location.chmod(0o600)
        location.with_name("server.shared-settings.json").write_text(json.dumps({"max_tokens": args.max_tokens}), encoding="utf-8")
        sys.argv = ["strata-server", "--engine", "strata", "--config", str(location), "--host", "127.0.0.1", "--port", str(args.port)]
        engine_log = Path(directory) / "engine.log"
        config["log"] = str(engine_log)
        location.write_text(json.dumps(config), encoding="utf-8")
        try:
            runpy.run_path(str(source / "serve/server.py"), run_name="__main__")
        finally:
            if engine_log.is_file():
                with engine_log.open("rb") as handle:
                    handle.seek(max(0, engine_log.stat().st_size - 65536))
                    print(handle.read().decode("utf-8", errors="replace"), file=sys.stderr)
    return 0


if __name__ == "__main__":
    # The Strata interpreter gets this package alone, never the installing environment's site-packages.
    _package = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "mlx_agent", _package / "__init__.py", submodule_search_locations=[str(_package)])
    sys.modules["mlx_agent"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["mlx_agent"])
    raise SystemExit(main())
