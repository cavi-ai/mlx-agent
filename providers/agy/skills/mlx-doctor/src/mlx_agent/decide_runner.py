"""Run inside a classification backend's environment: load one converted model, answer one request, print JSON."""

import argparse
import importlib
import json
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request", required=True)
    arguments = parser.parse_args(argv)

    with open(arguments.request, encoding="utf-8") as handle:
        request = json.load(handle)
    port = importlib.import_module(arguments.module)
    started = time.time()
    model, tokenizer = port.load(arguments.model)
    loaded = time.time()
    result = port.predict(model, tokenizer, request["state"], request["questions"])
    result["seconds"] = round(time.time() - loaded, 3)
    result["load_seconds"] = round(loaded - started, 3)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
