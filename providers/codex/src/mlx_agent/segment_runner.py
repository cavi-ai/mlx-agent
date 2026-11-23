"""Execute a local prompted image-segmentation task."""

import argparse
import importlib
import json
import time

if __package__:
    from .runner_model_cache import load_model
else:
    from runner_model_cache import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('module', 'model', 'image', 'out', 'box', 'points', 'labels'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    args = parser.parse_args(argv)
    port = importlib.import_module(args.module)
    start = time.monotonic()
    model = load_model(port.load, args.model)
    loaded = time.monotonic()
    result = port.generate(model, args.image, args.out, json.loads(args.box),
                           json.loads(args.points), json.loads(args.labels), args.threshold)
    result['load_seconds'] = round(loaded - start, 3)
    result['stage_seconds'] = {'load': result['load_seconds'], 'inference': round(time.monotonic() - loaded, 3)}
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
