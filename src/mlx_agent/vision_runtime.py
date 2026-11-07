"""Source-bound image feature reuse in an isolated MLX-VLM server."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import shutil
import sys

if __package__:
    from ._vision_cache import MAX_BYTES, MAX_ENTRIES
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _vision_cache import MAX_BYTES, MAX_ENTRIES

VERSIONS = {"mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.32.2", "mlx-vlm": "0.7.4",
            "numpy": "2.5.3", "transformers": "5.18.0"}
UPSTREAM_SOURCES = {
    "server/generation.py": "c9b7a5468d4cc2660458792f4829f711f02c3c357125c29fbe78a46f881396d4",
    "models/qwen2_5_vl/qwen2_5_vl.py": "5f0c0f5928e248cb14746b59f961bb9b51181536130007db9065da065b331e13",
    "models/qwen2_5_vl/vision.py": "ba936643eb9c468a25afb56eedf5c624d86c9065e43d68fcfa3fa7a790ded83f",
    "models/qwen3_vl/qwen3_vl.py": "12f0393421918eb1cbaa6daa447d93a405762e997f688676688e5c71d987d3ee",
    "models/qwen3_vl/vision.py": "664adc00fd0aec192a2a33c52d026f36873f3baab53044bc79660d4072b34afb",
    "models/qwen3_vl/processing_qwen3_vl.py": "8f8acffe91739016a8829217e8745d66141a37b73a2fd646c3989cef05db855b",
    "models/qwen3_vl_moe/qwen3_vl_moe.py": "92c0254ce3049899c714d12eb74ef94b67a7c4b77d0d6f6aaa5bd0b88e1b9fd5",
    "models/qwen3_vl_moe/vision.py": "51ac5a6a3241c5b59b575a680878a5b6115db21e7bf53c768909c7f28d508c1c",
}


def _environment(executable):
    resolved = shutil.which(executable)
    if resolved is None:
        return None, None
    entry = Path(resolved).absolute()
    python = entry.with_name("python")
    prefix = entry.parent.parent
    try:
        with entry.open("rb") as handle:
            shebang = handle.readline(4096).decode().strip()
        sites = list((prefix/"lib").glob("python3.*/site-packages"))
        if (entry.name == "mlx_vlm.server" and python.is_file() and (prefix/"pyvenv.cfg").is_file()
                and shebang == "#!"+str(python) and len(sites) == 1):
            return python, sites[0]
    except (OSError, UnicodeError):
        pass
    return None, None


def interpreter(executable):
    python, _ = _environment(executable)
    return str(python) if python is not None else None


def identity(executable):
    """Read installed metadata and source bytes without executing the backend."""
    python, site = _environment(executable)
    versions = dict.fromkeys(VERSIONS)
    runtime_sources = {}
    if site is not None:
        try:
            for distribution in importlib.metadata.distributions(path=[str(site)]):
                name = (distribution.metadata.get("Name") or "").lower().replace("_", "-")
                if name in versions:
                    versions[name] = distribution.version
            runtime_sources = {name: hashlib.sha256((site/"mlx_vlm"/name).read_bytes()).hexdigest()
                               for name in UPSTREAM_SOURCES}
        except (OSError, ValueError, UnicodeError):
            runtime_sources = {}
    root = Path(__file__).resolve().parent
    facts = {"name": "vision-features-v1", "runtime_executable": str(executable),
             "runtime_python": str(python) if python is not None else None,
             "versions": versions, "runtime_sources": runtime_sources,
             "sources": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                         for name in ("vision_runtime.py", "_vision_cache.py")},
             "max_bytes": MAX_BYTES, "max_entries": MAX_ENTRIES}
    facts["fingerprint"] = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    return facts


def available(evidence):
    return bool(evidence["runtime_python"] and evidence["versions"] == VERSIONS
                and evidence["runtime_sources"] == UPSTREAM_SOURCES)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-executable", required=True)
    parser.add_argument("--expected-vision-fingerprint", required=True)
    args, remaining = parser.parse_known_args(argv)
    evidence = identity(args.runtime_executable)
    if evidence["fingerprint"] != args.expected_vision_fingerprint:
        parser.error("Vision runtime source or dependencies changed; review a fresh serve preview.")
    from mlx_vlm import server
    from mlx_vlm.server import generation

    original = generation.load
    if sys.platform == "darwin" and available(evidence):
        if __package__:
            from ._vision_cache import apply
        else:
            from _vision_cache import apply

        def load(*load_args, **kwargs):
            model, processor = original(*load_args, **kwargs)
            adapter = kwargs.get("adapter_path", load_args[1] if len(load_args) > 1 else None)
            enabled = not adapter and apply(model)
            logging.getLogger(__name__).info("Vision feature cache: enabled=%s", enabled)
            return model, processor

        generation.load = load
    else:
        logging.getLogger(__name__).warning("Vision feature reuse unavailable; using native MLX-VLM.")
    previous = sys.argv
    try:
        sys.argv = [previous[0]] + remaining
        server.main()
    finally:
        sys.argv = previous
        generation.load = original


if __name__ == "__main__":
    main()
