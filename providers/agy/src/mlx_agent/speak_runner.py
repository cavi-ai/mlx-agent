"""Run inside a speech backend's environment: load one converted TTS model, synthesize one text to a WAV, print JSON."""

import argparse
import json
import sys
import time
import wave


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--lang-code", default=None)
    arguments = parser.parse_args(argv)

    import mlx.core as mx
    import numpy as np
    from mlx_audio.tts.utils import load

    started = time.time()
    model = load(arguments.model, model_type=arguments.model_type)
    load_seconds = time.time() - started

    kwargs = {"text": arguments.text, "speed": arguments.speed, "verbose": False}
    if arguments.voice is not None:
        kwargs["voice"] = arguments.voice
    if arguments.lang_code is not None:
        kwargs["lang_code"] = arguments.lang_code

    mx.reset_peak_memory()
    started = time.time()
    chunks = []
    sample_rate = getattr(model, "sample_rate", None)
    for result in model.generate(**kwargs):
        chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        sample_rate = getattr(result, "sample_rate", None) or sample_rate
    seconds = time.time() - started
    peak_memory_gb = mx.get_peak_memory() / 1e9

    if not chunks or not sample_rate:
        raise RuntimeError("the model produced no audio")
    samples = np.concatenate(chunks)
    if samples.size == 0:
        raise RuntimeError("the model produced no audio")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with open(arguments.out, "xb") as handle:
        writer = wave.open(handle, "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(int(sample_rate))
        writer.writeframes(pcm.tobytes())
        writer.close()

    audio_seconds = int(pcm.size) / int(sample_rate)
    print(json.dumps({
        "path": arguments.out,
        "sample_rate": int(sample_rate),
        "audio_seconds": round(audio_seconds, 3),
        "seconds": round(seconds, 3),
        "load_seconds": round(load_seconds, 3),
        "real_time_factor": round(seconds / audio_seconds, 4),
        "peak_memory_gb": round(peak_memory_gb, 3),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
