"""Split a mix into a vocal stem and the rest with a Mel-Band-RoFormer checkpoint on mlx-audio's model."""

import errno
import os
import shutil
import wave
from contextlib import ExitStack, suppress
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PCM_BLOCK_FRAMES = 65_536


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


def _stage_stem(stack, target, vocals, rate, mixture=None):
    directory = stack.enter_context(TemporaryDirectory(prefix=".mlx-separate-", dir=Path(target).parent))
    path = Path(directory) / "stem.wav"
    channels, frames = vocals.shape
    with wave.open(str(path), "wb") as wav:
        wav.setparams((channels, 2, rate, frames, "NONE", "not compressed"))
        for start in range(0, frames, PCM_BLOCK_FRAMES):
            end = start + PCM_BLOCK_FRAMES
            block = vocals[:, start:end]
            if mixture is not None:
                block = mixture[:, start:end] - block
            pcm = (np.clip(block, -1.0, 1.0) * 32767).astype(np.int16)
            wav.writeframesraw(pcm.T.tobytes())
    return path


def _remove_owned_stem(target, identity):
    with suppress(FileNotFoundError):
        if os.path.samestat(identity, os.lstat(target)):
            os.unlink(target)


def _publish_stem(staged, target):
    # A hard link publishes the completed file without replacing an existing destination.
    try:
        os.link(staged, target)
    except OSError as error:
        if error.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM):
            raise
    else:
        return staged.stat()

    # Volumes without hard links can still claim the destination with exclusive creation.
    identity = None
    try:
        with staged.open("rb") as source, open(target, "xb") as output:
            identity = os.fstat(output.fileno())
            shutil.copyfileobj(source, output)
    except OSError:
        if identity is not None:
            _remove_owned_stem(target, identity)
        raise
    return identity


def generate(model, audio, out, rest=None):
    from mlx_audio.audio_io import read

    rate = int(model.config.sample_rate)
    samples, _ = read(audio, always_2d=True, dtype="float32", sample_rate=rate, nchannels=2)
    mixture = np.ascontiguousarray(samples.T)
    vocals = separate(model, mixture)
    with ExitStack() as stack:
        staged_vocals = _stage_stem(stack, out, vocals, rate)
        staged_rest = _stage_stem(stack, rest, vocals, rate, mixture) if rest else None
        vocal_identity = _publish_stem(staged_vocals, out)
        if staged_rest is not None:
            try:
                _publish_stem(staged_rest, rest)
            except OSError:
                _remove_owned_stem(out, vocal_identity)
                raise
    result = {"path": out, "sample_rate": rate, "audio_seconds": round(mixture.shape[1] / rate, 3)}
    if rest:
        result["rest"] = rest
    return result
