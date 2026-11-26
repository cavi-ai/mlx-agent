"""Run inside the isolated ACE-Step backend: format a caption and lyrics with the ACE-Step planner, print JSON.

The planner loads from a private folder of links to its files; the network is closed.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import tempfile
import time

if __package__:
    from .acestep_runner import link_folder, offline, planned
else:
    from acestep_runner import link_folder, offline, planned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("planner", "caption", "lyrics"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--bpm", type=int)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--key")
    parser.add_argument("--time-signature", type=int)
    parser.add_argument("--language")
    args = parser.parse_args(argv)
    planner_path = Path(args.planner)
    socket.socket.connect = offline
    socket.create_connection = offline
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    from acestep.inference import format_sample
    from acestep.llm_inference import LLMHandler

    # The planner's decoding is bound to every value given; it chooses the rest.
    bound = {"bpm": args.bpm, "duration": args.duration, "keyscale": args.key,
             "timesignature": str(args.time_signature) if args.time_signature else None, "language": args.language}
    with tempfile.TemporaryDirectory(prefix="mlx-agent-acestep-") as directory:
        checkpoints = Path(directory) / "checkpoints"
        checkpoints.mkdir()
        link_folder(planner_path, checkpoints / planner_path.name)
        os.environ.update(ACESTEP_CHECKPOINTS_DIR=str(checkpoints), ACESTEP_PROJECT_ROOT=directory)
        started = time.perf_counter()
        planner = LLMHandler()
        status, ok = planner.initialize(checkpoint_dir=str(checkpoints), lm_model_path=planner_path.name,
                                        backend="mlx", device="mps")
        if not ok:
            raise RuntimeError("The ACE-Step planner did not load: {0}".format(status))
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result = format_sample(planner, args.caption, args.lyrics,
                               user_metadata={name: value for name, value in bound.items() if value is not None} or None,
                               temperature=args.temperature)
        seconds = time.perf_counter() - started
    if not result.success:
        raise RuntimeError("The ACE-Step planner returned no plan: {0}".format(result.error or result.status_message))
    metadata = {"bpm": result.bpm, "keyscale": result.keyscale, "duration": result.duration,
                "timesignature": result.timesignature, "language": result.language}
    print(json.dumps(dict(planned(metadata, result.caption, True), lyrics=result.lyrics, seconds=seconds,
                          load_seconds=load_seconds)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
