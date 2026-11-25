"""Split a mix into a vocal stem and the rest with a Mel-Band-RoFormer checkpoint on mlx-audio's model."""

import numpy as np


def load(path):
    from mlx_audio.sts.models.mel_roformer import MelRoFormer

    model = MelRoFormer.from_pretrained(path)
    model.eval()
    return model


def separate(model, mixture):
    """Vocals for a [channels, samples] float32 mix.

    The model sees `chunk_size` samples at a time, `num_overlap` chunks over each sample; the chunk
    outputs are Hann-weighted and normalized by the summed weights, so every sample gets full weight.
    """
    import mlx.core as mx

    chunk = int(model.config.chunk_size)
    step = chunk // int(model.config.num_overlap)
    border = chunk - step
    length = mixture.shape[1]
    starts = range(0, border + length, step)
    padded = np.zeros((mixture.shape[0], starts[-1] + chunk), dtype=np.float32)
    padded[:, border:border + length] = mixture
    window = (np.hanning(chunk) if border else np.ones(chunk)).astype(np.float32)
    total = np.zeros_like(padded)
    weight = np.zeros(padded.shape[1], dtype=np.float32)
    for start in starts:
        vocals = model(mx.array(padded[None, :, start:start + chunk]))[0]
        total[:, start:start + chunk] += np.array(vocals.astype(mx.float32)) * window
        weight[start:start + chunk] += window
    return total[:, border:border + length] / weight[border:border + length]


def generate(model, audio, out, rest=None):
    from mlx_audio.audio_io import read, write

    rate = int(model.config.sample_rate)
    samples, _ = read(audio, always_2d=True, dtype="float32", sample_rate=rate, nchannels=2)
    mixture = np.ascontiguousarray(samples.T)
    vocals = separate(model, mixture)
    write(out, vocals.T, rate)
    result = {"path": out, "sample_rate": rate, "audio_seconds": round(mixture.shape[1] / rate, 3)}
    if rest:
        write(rest, (mixture - vocals).T, rate)
        result["rest"] = rest
    return result
