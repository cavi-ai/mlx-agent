#!/usr/bin/env python3
"""Standalone GGUF to MLX conversion worker.

Spawned by ``mlx_agent.convert`` with an absolute path, so it must import
nothing from :mod:`mlx_agent`. Two bounded steps, in order:

1. dequantize the GGUF back to Hugging Face weights with ``transformers``;
2. quantize those weights into MLX with the ``mlx_lm.convert`` executable.

The intermediate fp16 checkpoint is large and temporary; it is removed unless
``--keep-intermediate`` is passed. ``--port`` instead hands the GGUF to a
bundled port that maps its tensors straight into MLX shards, for architectures
``transformers`` cannot load (llama.cpp's DSpark drafters, Qwen3.5). The worker writes a
provenance marker into the output so a later scan can tell which GGUF produced
it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _paths_without_script_dir(entries, script_dir, cwd):
    """Drop the runner directory so ``import gguf`` hits the installed package.

    Script launch puts this package directory on ``sys.path[0]``, which shadows
    the installed ``gguf`` package with sibling ``gguf.py`` (no ``GGUFReader``).
    """
    script_dir = Path(script_dir).resolve()
    cwd = Path(cwd).resolve()

    def _is_within_script_dir(candidate):
        try:
            candidate.resolve().relative_to(script_dir)
            return True
        except (ValueError, OSError):
            return False

    cleaned = []
    for entry in entries:
        if entry in ("", "."):
            if cwd != script_dir:
                cleaned.append(entry)
            continue
        try:
            entry_path = Path(entry)
            if not entry_path.is_absolute():
                entry_path = cwd / entry_path
            if _is_within_script_dir(entry_path):
                continue
        except OSError:
            cleaned.append(entry)
            continue
        cleaned.append(entry)
    return cleaned


sys.path[:] = _paths_without_script_dir(sys.path, Path(__file__).resolve().parent, Path.cwd())


PROVENANCE_NAME = "mlx-converter.json"
SIGNATURE_CHUNK_BYTES = 1024 * 1024
EXECUTABLE = "mlx_lm.convert"
REQUIRED_MODULES = ("torch", "transformers", "gguf")
PORT_REQUIRED_MODULES = ("gguf", "mlx")
PORTS_DIR = Path(__file__).resolve().parent / "resources" / "ports"
# --port name to its converter under resources/ports/<backend>/<name>/convert.py.
GGUF_PORTS = {"deepseek_v4_dspark": "mlx-lm", "qwen35": "mlx-vlm"}
SHARD_BYTES = 5 << 30


def _log(message):
    print("[mlx-converter] {0}".format(message), flush=True)


def _signature(path, chunk_bytes=SIGNATURE_CHUNK_BYTES):
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_bytes))
        if size > chunk_bytes * 2:
            handle.seek(-chunk_bytes, os.SEEK_END)
            digest.update(handle.read(chunk_bytes))
    return digest.hexdigest()


def _missing_modules(names=REQUIRED_MODULES):
    missing = []
    for name in names:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    return missing


def dequantize(gguf_path, work_dir):
    """Materialize Hugging Face weights from a GGUF file."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = str(gguf_path.parent)
    filename = gguf_path.name
    _log("dequantizing {0}".format(filename))
    model = AutoModelForCausalLM.from_pretrained(directory, gguf_file=filename)
    model.save_pretrained(str(work_dir))
    del model
    tokenizer = AutoTokenizer.from_pretrained(directory, gguf_file=filename)
    tokenizer.save_pretrained(str(work_dir))
    _log("dequantized to {0}".format(work_dir))


def quantize(work_dir, out_dir, q_bits):
    """Hand the dequantized checkpoint to mlx_lm.convert."""
    executable = shutil.which(EXECUTABLE)
    if executable is None:
        raise RuntimeError(
            "{0} is not on PATH; install mlx-lm into this interpreter".format(EXECUTABLE)
        )
    argv = [
        executable,
        "--hf-path", str(work_dir),
        "--mlx-path", str(out_dir),
        "-q",
        "--q-bits", str(q_bits),
    ]
    _log("running {0}".format(" ".join(argv)))
    completed = subprocess.run(argv, stdin=subprocess.DEVNULL, check=False)
    if completed.returncode != 0:
        raise RuntimeError("{0} exited {1}".format(EXECUTABLE, completed.returncode))


def load_port(name):
    """The bundled port's ``convert`` module, loaded by path (this runner imports nothing from mlx_agent)."""
    location = PORTS_DIR / GGUF_PORTS[name] / name / "convert.py"
    spec = importlib.util.spec_from_file_location("mlx_agent_gguf_port_{0}".format(name), location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShardWriter:
    """Writes ``model-0000k-of-0000n.safetensors`` shards of at most ``limit`` bytes, then the index."""

    def __init__(self, directory, limit=SHARD_BYTES):
        self.directory = Path(directory)
        self.limit = limit
        self.pending = {}
        self.pending_bytes = 0
        self.shards = []
        self.weight_map = {}
        self.total = 0

    def add(self, name, array):
        if self.pending and self.pending_bytes + array.nbytes > self.limit:
            self._flush()
        self.pending[name] = array
        self.pending_bytes += array.nbytes
        self.total += array.nbytes

    def _flush(self):
        import mlx.core as mx

        path = self.directory / "model-{0:05d}.partial.safetensors".format(len(self.shards) + 1)
        mx.save_safetensors(str(path), self.pending, metadata={"format": "mlx"})
        self.shards.append((path, sorted(self.pending)))
        self.pending, self.pending_bytes = {}, 0

    def finish(self):
        if self.pending:
            self._flush()
        count = len(self.shards)
        for index, (path, names) in enumerate(self.shards, start=1):
            final = "model-{0:05d}-of-{1:05d}.safetensors".format(index, count)
            path.rename(self.directory / final)
            self.weight_map.update({name: final for name in names})
        index = {"metadata": {"total_size": self.total}, "weight_map": dict(sorted(self.weight_map.items()))}
        (self.directory / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return count


def convert_with_port(name, gguf_path, out_dir, q_bits, projector=None):
    """Run a bundled port into a fresh ``out_dir``; returns the port's summary."""
    out_dir.mkdir(parents=True)
    _log("converting {0} with the {1} port".format(gguf_path.name, name))
    extra = {} if projector is None else {"projector": projector}
    return load_port(name).convert(gguf_path, out_dir, q_bits, ShardWriter(out_dir), log=_log, **extra)


def write_provenance(out_dir, gguf_path, q_bits, signature, port=None, projector=None):
    marker = {
        "schema_version": "1.0",
        "tool": "mlx-agent.gguf",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "q_bits": q_bits,
        "source": {
            "kind": "gguf",
            "path": str(gguf_path),
            "name": gguf_path.name,
            "bytes": gguf_path.stat().st_size,
            "signature": signature,
        },
    }
    if port is not None:
        marker["port"] = port
    if projector is not None:
        marker["projector"] = {"path": str(projector), "name": projector.name, "bytes": projector.stat().st_size}
    target = Path(out_dir) / PROVENANCE_NAME
    target.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--q-bits", type=int, required=True, choices=(4, 8))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--signature", default=None)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--port", choices=sorted(GGUF_PORTS), default=None)
    parser.add_argument("--projector", default=None, help="the model's vision projector GGUF, read by --port")
    arguments = parser.parse_args(argv)

    gguf_path = Path(arguments.gguf).expanduser()
    out_dir = Path(arguments.out).expanduser()
    projector = Path(arguments.projector).expanduser() if arguments.projector else None
    if not gguf_path.is_file():
        _log("source GGUF not found: {0}".format(gguf_path))
        return 2
    if projector is not None and (arguments.port is None or not projector.is_file()):
        _log("--projector needs --port and an existing file: {0}".format(projector))
        return 2
    if out_dir.exists():
        _log("output already exists: {0}".format(out_dir))
        return 2
    missing = _missing_modules(PORT_REQUIRED_MODULES if arguments.port else REQUIRED_MODULES)
    if missing:
        _log("missing required modules: {0}".format(", ".join(missing)))
        return 2
    if arguments.port:
        try:
            convert_with_port(arguments.port, gguf_path, out_dir, arguments.q_bits, projector=projector)
            write_provenance(
                out_dir, gguf_path, arguments.q_bits,
                arguments.signature or _signature(gguf_path), port=arguments.port, projector=projector,
            )
        except (OSError, RuntimeError, ValueError, ImportError, KeyError) as error:
            _log("conversion failed: {0}".format(error))
            if out_dir.is_dir():
                shutil.rmtree(out_dir, ignore_errors=True)
                _log("removed partial output {0}".format(out_dir))
            return 1
        _log("converted {0} -> {1}".format(gguf_path.name, out_dir))
        return 0

    work_dir = Path(arguments.work_dir) if arguments.work_dir else out_dir.with_name(
        out_dir.name + ".hf-intermediate"
    )
    if work_dir.exists():
        _log("intermediate directory already exists: {0}".format(work_dir))
        return 2
    work_dir.mkdir(parents=True)
    try:
        dequantize(gguf_path, work_dir)
        quantize(work_dir, out_dir, arguments.q_bits)
        write_provenance(
            out_dir, gguf_path, arguments.q_bits,
            arguments.signature or _signature(gguf_path),
        )
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        _log("conversion failed: {0}".format(error))
        return 1
    finally:
        if not arguments.keep_intermediate and work_dir.is_dir():
            shutil.rmtree(work_dir, ignore_errors=True)
            _log("removed intermediate {0}".format(work_dir))
    _log("converted {0} -> {1}".format(gguf_path.name, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
