"""Run inside a speech backend's environment: load one converted model, transcribe one clip, print JSON."""

import argparse
import inspect
import json
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", default=None)
    arguments = parser.parse_args(argv)

    from mlx_audio.stt.utils import load_audio, load_model

    started = time.time()
    model = load_model(arguments.model)
    samples = load_audio(arguments.audio)
    kwargs = {}
    if arguments.language and "language" in inspect.signature(model.generate).parameters:
        kwargs["language"] = arguments.language
    output = model.generate(arguments.audio, **kwargs)
    text = getattr(output, "text", output if isinstance(output, str) else "")
    print(json.dumps({
        "text": str(text).strip(),
        "seconds": round(time.time() - started, 3),
        "audio_seconds": round(int(samples.shape[0]) / 16000, 3),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
