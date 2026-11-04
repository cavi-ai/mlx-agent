"""Evaluate one converted image-to-mesh model inside its backend environment."""

import argparse
import importlib
import json
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("module", "model", "image", "out"):
        parser.add_argument("--" + name, required=True)
    for name in ("steps", "seed", "resolution", "max-faces"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    args = parser.parse_args(argv)
    port = importlib.import_module(args.module)
    started = time.monotonic()
    model = port.load(args.model)
    load_seconds = time.monotonic() - started
    result = port.generate(model, args.image, args.out, steps=args.steps, seed=args.seed,
                           resolution=args.resolution, guidance=args.guidance,
                           max_faces=args.max_faces)
    result["load_seconds"] = round(load_seconds, 3)
    result["stage_seconds"] = {"load": result["load_seconds"], **result["stage_seconds"]}
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
