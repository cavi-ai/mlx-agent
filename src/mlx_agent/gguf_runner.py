#!/usr/bin/env python3
"""Standalone GGUF to MLX conversion worker.

Spawned by ``mlx_agent.convert`` with an absolute path, so it must import
nothing from :mod:`mlx_agent`. Two bounded steps, in order:

1. dequantize the GGUF back to Hugging Face weights with ``transformers``;
2. quantize those weights into MLX with the ``mlx_lm.convert`` executable.

The intermediate fp16 checkpoint is large and temporary; it is removed unless
``--keep-intermediate`` is passed. ``--port`` instead hands the GGUF to a
bundled port that writes the MLX weights itself, for architectures
``transformers`` cannot load (llama.cpp's DSpark drafters). The worker writes a
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
GGUF_PORTS = {"deepseek_v4_dspark": "mlx-lm"}


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


def convert_with_port(name, gguf_path, out_dir, q_bits):
    """Run a bundled port into a fresh ``out_dir``; returns the port's summary."""
    out_dir.mkdir(parents=True)
    _log("converting {0} with the {1} port".format(gguf_path.name, name))
    return load_port(name).convert(gguf_path, out_dir, q_bits, log=_log)


def write_provenance(out_dir, gguf_path, q_bits, signature, port=None):
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
    arguments = parser.parse_args(argv)

    gguf_path = Path(arguments.gguf).expanduser()
    out_dir = Path(arguments.out).expanduser()
    if not gguf_path.is_file():
        _log("source GGUF not found: {0}".format(gguf_path))
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
            convert_with_port(arguments.port, gguf_path, out_dir, arguments.q_bits)
            write_provenance(
                out_dir, gguf_path, arguments.q_bits,
                arguments.signature or _signature(gguf_path), port=arguments.port,
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
