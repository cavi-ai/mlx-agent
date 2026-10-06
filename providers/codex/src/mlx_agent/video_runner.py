"""Run inside a video backend's environment: render one prompt with a converted model to an MP4, print JSON."""

import argparse
import importlib
import json
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args(argv)

    port = importlib.import_module(arguments.module)
    started = time.time()
    model = port.load(arguments.model)
    open_seconds = time.time() - started
    result = port.generate(
        model, arguments.prompt, arguments.out, width=arguments.width, height=arguments.height,
        frames=arguments.frames, fps=arguments.fps, steps=arguments.steps, seed=arguments.seed,
    )
    result["load_seconds"] = round(float(result.get("load_seconds", 0.0)) + open_seconds, 3)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
