"""Source-bound, opt-in code-edit decoding for the MLX-LM server."""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

if __package__:
    from .prefill_runtime import VERSIONS, can_wrap_runtime, identity as prefill_identity
else:
    from prefill_runtime import VERSIONS, can_wrap_runtime, identity as prefill_identity

SOURCE_FILES = ("prompt_lookup_runtime.py", "_prompt_lookup.py", "prefill_runtime.py")


def identity():
    root = Path(__file__).resolve().parent
    facts = {"name": "prompt-lookup-v1", "versions": prefill_identity()["versions"],
             "sources": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                         for name in SOURCE_FILES}}
    facts["fingerprint"] = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    return facts


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--expected-decode-fingerprint", required=True)
    parser.add_argument("--m5-prefill-fingerprint")
    args, remaining = parser.parse_known_args(argv)
    evidence = identity()
    if evidence["fingerprint"] != args.expected_decode_fingerprint:
        parser.error("Code-edit runtime source or dependencies changed; review a fresh serve preview.")
    from mlx_lm import server

    restore = lambda: None
    if sys.platform == "darwin" and evidence["versions"] == VERSIONS:
        if __package__:
            from ._prompt_lookup import install
        else:
            from _prompt_lookup import install
        restore = install(server)
        logging.getLogger(__name__).warning("Code-edit acceleration: plain greedy requests run one at a time.")
    else:
        logging.getLogger(__name__).warning("Code-edit acceleration unavailable for this runtime; using native MLX.")
    previous = sys.argv
    try:
        if args.m5_prefill_fingerprint:
            if __package__:
                from .prefill_runtime import main as prefill_main
            else:
                from prefill_runtime import main as prefill_main
            prefill_main(["--expected-kernel-fingerprint", args.m5_prefill_fingerprint] + remaining)
        else:
            sys.argv = [previous[0]] + remaining
            server.main()
    finally:
        sys.argv = previous
        restore()


if __name__ == "__main__":
    main()
