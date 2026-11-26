"""Separate the sound a text prompt names from a mix with a SAM-Audio checkpoint on mlx-audio's model.

The T5 text encoder the checkpoint's config names is read from the folder of that name beside the
checkpoint, so nothing is fetched.
"""

import dataclasses
import importlib
import json
import warnings
from contextlib import ExitStack
from pathlib import Path

import numpy as np


def _stems():
    """The Mel-Band-RoFormer port's staged, non-replacing WAV writer, installed beside this port."""
    package = __name__.rpartition(".")[0]
    return importlib.import_module(package + ".mel_band_roformer" if package else "mel_band_roformer")


def text_encoder_folder(path):
    folder = Path(path)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    name = config["text_encoder"]["name"]
    encoder = folder.parent / name
    missing = [file for file in ("config.json", "model.safetensors") if not (encoder / file).is_file()]
    if missing:
        raise FileNotFoundError(
            "SAM-Audio reads its {0} text encoder from {1}, which lacks {2}; place that model folder beside "
            "the checkpoint.".format(name, encoder, ", ".join(missing))
        )
    return encoder


def _t5(folder):
    """T5 encoder weights from a local folder, configured as mlx-audio configures its hub copy."""
    import mlx.core as mx
    from mlx_audio.sts.models.sam_audio.text_encoder import T5Config, T5Encoder

    values = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    defaults = T5Config(is_gated_act=False, dense_act_fn="relu")
    config = T5Config(**{field.name: values.get(field.name, getattr(defaults, field.name))
                         for field in dataclasses.fields(T5Config)})
    model = T5Encoder(config)
    model.load_weights(list(T5Encoder.sanitize(mx.load(str(folder / "model.safetensors"))).items()))
    mx.eval(model.parameters())
    model.eval()
    return model


def load(path):
    encoder = text_encoder_folder(path)
    import transformers
    from mlx_audio.sts.models.sam_audio import SAMAudio

    # mlx-audio warns and keeps random weights when a checkpoint is incomplete.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SAMAudio.from_pretrained(path)
    failed = [str(item.message) for item in caught if Path(item.filename).parent.name == "sam_audio"]
    if failed:
        raise RuntimeError("SAM-Audio checkpoint did not load: " + "; ".join(failed))
    model.text_encoder.tokenizer = transformers.AutoTokenizer.from_pretrained(str(encoder), local_files_only=True)
    model.text_encoder.model = _t5(encoder)
    model.eval()
    return model


def fit(samples, length):
    """A [1, length] row: the decoder's output trimmed or zero-padded to the mix length."""
    row = np.asarray(samples, dtype=np.float32).reshape(1, -1)[:, :length]
    return np.pad(row, ((0, 0), (0, length - row.shape[1])))


def generate(model, audio, out, rest=None, prompt=None):
    import mlx.core as mx
    from mlx_audio.audio_io import read

    if not prompt:
        raise ValueError("SAM-Audio separates the sound a prompt names; pass a prompt.")
    rate = int(model.sample_rate)
    samples, _ = read(audio, always_2d=True, dtype="float32", sample_rate=rate, nchannels=1)
    length = samples.shape[0]
    result = model.separate_long(mx.array(np.ascontiguousarray(samples.T)[None]), [prompt])
    target = fit(np.array(result.target[0].astype(mx.float32)), length)
    residual = fit(np.array(result.residual[0].astype(mx.float32)), length) if rest else None
    stems = _stems()
    with ExitStack() as stack:
        staged_target = stems._stage_stem(stack, out, target, rate)
        staged_rest = stems._stage_stem(stack, rest, residual, rate) if rest else None
        identity = stems._publish_stem(staged_target, out)
        if staged_rest is not None:
            try:
                stems._publish_stem(staged_rest, rest)
            except OSError:
                stems._remove_owned_stem(out, identity)
                raise
    value = {"path": out, "prompt": prompt, "sample_rate": rate, "audio_seconds": round(length / rate, 3)}
    if rest:
        value["rest"] = rest
    return value
