"""Opt-in, source-bound M5 prompt processing for the MLX-LM server."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import sys
from pathlib import Path

VERSIONS = {"mlx": "0.31.1", "mlx-metal": "0.31.1", "mlx-lm": "0.29.1"}
SOURCE_FILES = ("prefill_runtime.py", "_m5_prefill.py", "resources/m5_prefill.metal")


def identity():
    """Read source/dependency identities without importing MLX or touching the GPU."""
    versions = {}
    for name in VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    facts = {"name": "m5-q4-prefill-v1", "versions": versions, "sources": sources}
    facts["fingerprint"] = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    return facts


def runtime_supported(system, os_version, versions, architecture):
    return (system == "darwin" and os_version.split(".")[0] == "26"
            and versions == VERSIONS and architecture == "applegpu_g17s")


def can_wrap_runtime(executable):
    """Wrap only a console entrypoint demonstrably owned by this interpreter."""
    resolved = shutil.which(executable)
    if resolved is None:
        return False
    script = Path(resolved)
    python = Path(sys.executable)
    try:
        if script.parent.resolve() != python.parent.resolve():
            return False
        with script.open("rb") as handle:
            first_line = handle.readline(4096).decode("utf-8").strip()
        return first_line == "#!" + str(python)
    except (OSError, UnicodeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--expected-kernel-fingerprint", required=True)
    args, remaining = parser.parse_known_args(argv)
    evidence = identity()
    if evidence["fingerprint"] != args.expected_kernel_fingerprint:
        parser.error("Prompt kernel source or dependencies changed; review a fresh serve preview.")
    import mlx.core as mx
    from mlx_lm import server

    supported = runtime_supported(sys.platform, platform.mac_ver()[0], evidence["versions"],
                                  mx.device_info().get("architecture"))
    original_load = server.load
    if supported:
        if __package__:
            from ._m5_prefill import apply
        else:
            from _m5_prefill import apply

        def load(*load_args, **kwargs):
            model, tokenizer = original_load(*load_args, **kwargs)
            # Adapter subclasses retain their own forward and training semantics.
            if not kwargs.get("adapter_path"):
                count = apply(model)
                logging.warning("M5 prompt kernel: %d MLP layers; other inputs use native MLX.", count)
            return model, tokenizer

        server.load = load
    else:
        logging.warning("M5 prompt kernel unavailable for this hardware/runtime; using native MLX.")
    previous = sys.argv
    try:
        sys.argv = [previous[0]] + remaining
        server.main()
    finally:
        sys.argv = previous
        server.load = original_load


if __name__ == "__main__":
    main()
