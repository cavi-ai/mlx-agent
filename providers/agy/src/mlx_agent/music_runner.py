"""Run inside the isolated music backend; write a new WAV and measured metrics."""

import argparse
import json
import time
import wave


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "caption", "lyrics", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    import mlx.core as mx
    import numpy as np
    from mlx_audio.music import load

    started = time.perf_counter()
    model = load(args.model)
    load_seconds = time.perf_counter() - started
    mx.reset_peak_memory()
    started = time.perf_counter()
    chunks, sample_rate = [], None
    for result in model.generate(text=args.caption, lyrics=args.lyrics, duration=args.duration, steps=args.steps, seed=args.seed):
        if sample_rate is not None and sample_rate != result.sample_rate:
            raise RuntimeError("Inconsistent sample rates")
        sample_rate = result.sample_rate
        audio = np.asarray(result.audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.ndim != 2 or audio.shape[1] not in (1, 2) or not np.isfinite(audio).all():
            raise RuntimeError("Invalid audio output")
        chunks.append(audio)
    seconds = time.perf_counter() - started
    if not chunks or not sample_rate:
        raise RuntimeError("No audio output")
    audio = np.concatenate(chunks, axis=0)
    if audio.shape[0] == 0:
        raise RuntimeError("Empty audio output")
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with open(args.out, "xb") as handle:
        with wave.open(handle, "wb") as writer:
            writer.setnchannels(audio.shape[1])
            writer.setsampwidth(2)
            writer.setframerate(int(sample_rate))
            writer.writeframes(pcm.tobytes())
    audio_seconds = audio.shape[0] / sample_rate
    print(json.dumps({"path": args.out, "sample_rate": sample_rate, "audio_seconds": audio_seconds,
                      "seconds": seconds, "load_seconds": load_seconds, "real_time_factor": seconds / audio_seconds,
                      "peak_memory_gb": mx.get_peak_memory() / 1e9}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
