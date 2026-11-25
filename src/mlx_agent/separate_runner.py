"""Run inside an audio backend's environment: split one mix into a vocal stem and the rest, print JSON."""

import argparse
import importlib
import json
import sys
import time

if __package__:
    from .runner_model_cache import load_model
else:
    from runner_model_cache import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("module", "model", "audio", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rest", default=None)
    arguments = parser.parse_args(argv)
    port = importlib.import_module(arguments.module)
    started = time.monotonic()
    model = load_model(port.load, arguments.model)
    loaded = time.monotonic()
    result = port.generate(model, arguments.audio, arguments.out, arguments.rest)
    result["load_seconds"] = round(loaded - started, 3)
    result["seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
