"""Offline inputs for native MLX training on converted single-branch Wan.

Callers own compute admission and immutable source snapshots. These helpers
return tensors and content identities; they never download or publish files.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path

MAX_SOURCE_BYTES = 24 * 1024 * 1024 * 1024
MAX_VIDEO_BYTES = 512 * 1024 * 1024
MAX_DECODED_PIXELS = 32 * 1024 * 1024


def _digest_regular(path, limit):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError("Training sources must be bounded regular files.")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise ValueError("Training source changed while reading.")
        return "sha256:" + digest.hexdigest()
    finally:
        os.close(descriptor)


def training_sources(path):
    """Validate the supported converted layout and hash every consumed source."""
    root = Path(path).expanduser()
    try:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("A local converted model directory is required.")
        identities = {}
        for name in ("config.json", "model.safetensors", "t5_encoder.safetensors", "vae.safetensors"):
            identities[name] = _digest_regular(root / name, 65536 if name == "config.json" else MAX_SOURCE_BYTES)
        config = json.loads((root / "config.json").read_text())
        if (
            not isinstance(config, dict)
            or config.get("model_type") != "t2v"
            or config.get("model_version") != "2.1"
            or config.get("dual_model", False) is not False
            or config.get("in_dim", 16) != 16
            or config.get("out_dim", 16) != 16
            or config.get("vae_z_dim", 16) != 16
        ):
            raise ValueError("Only converted single-branch Wan 2.1 T2V is supported.")
        tokenizer = root / "tokenizer"
        if tokenizer.is_symlink() or not tokenizer.is_dir():
            raise ValueError("A converted local tokenizer is required.")
        files = []
        for directory, subdirs, names in os.walk(tokenizer, followlinks=False):
            for name in subdirs:
                if (Path(directory) / name).is_symlink():
                    raise ValueError("Tokenizer links are not supported.")
            for name in names:
                files.append(Path(directory) / name)
                if len(files) > 64:
                    raise ValueError("Tokenizer exceeds the supported file count.")
        if not files:
            raise ValueError("A converted local tokenizer is required.")
        for file in sorted(files):
            identities[file.relative_to(root).as_posix()] = _digest_regular(file, 64 * 1024 * 1024)
        if _digest_regular(root / "config.json", 65536) != identities["config.json"]:
            raise ValueError("Converted configuration changed while reading.")
        return config, identities
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Converted training sources are incomplete or unreadable.") from error


def _unchanged(root, identities):
    if training_sources(root)[1] != identities:
        raise ValueError("Converted training sources changed during loading.")


def _config(raw):
    from mlx_video.models.wan.config import WanModelConfig
    return WanModelConfig.from_dict(raw)


def load_training_session(path, *, targets, rank, alpha, learning_rate, seed=42):
    """Strictly load the converted backbone before freezing it for LoRA."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_video.models.wan.model import WanModel
    from .training import WanMLXTrainingSession

    root = Path(path).expanduser()
    raw, identities = training_sources(root)
    model = WanModel(_config(raw))
    quantization = raw.get("quantization")
    if quantization:
        from mlx_video.convert_wan import _quantize_predicate
        nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"],
                    class_predicate=lambda name, layer: _quantize_predicate(name, layer))
    weights = mx.load(root / "model.safetensors")
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    del weights
    _unchanged(root, identities)
    return WanMLXTrainingSession(model, targets=targets, rank=rank, alpha=alpha,
                                 learning_rate=learning_rate, seed=seed)


def validate_video_shape(shape, config):
    """Reject incompatible media without inventing crop, resize or trim rules."""
    stride = config.get("vae_stride", [4, 8, 8])
    patch = config.get("patch_size", [1, 2, 2])
    if (
        len(shape) != 4 or any(type(value) is not int for value in shape)
        or not 5 <= shape[0] <= 241 or shape[3] != 3
        or any(not 16 <= value <= 1024 for value in shape[1:3])
        or len(stride) != 3 or len(patch) != 3
        or any(type(value) is not int or value <= 0 for value in (*stride, *patch))
        or (shape[0] - 1) % stride[0]
        or ((shape[0] - 1) // stride[0] + 1) % patch[0]
        or any(shape[index + 1] % (stride[index + 1] * patch[index + 1]) for index in range(2))
        or math.prod(shape[:3]) > MAX_DECODED_PIXELS
    ):
        raise ValueError("Training video dimensions or frame count are unsupported.")


def _decode_video(path, config):
    import cv2
    import numpy as np

    reader = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not reader.isOpened():
            raise ValueError("Training video could not be opened.")
        metadata = tuple(reader.get(key) for key in (
            cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH
        ))
        if any(not math.isfinite(value) or value != int(value) for value in metadata):
            raise ValueError("Training video has invalid dimensions or frame count.")
        expected_frames, height, width = map(int, metadata)
        validate_video_shape((expected_frames, height, width, 3), config)
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            height, width, channels = frame.shape
            # Reject oversized decoding before retaining another frame.
            if channels != 3 or not 16 <= height <= 1024 or not 16 <= width <= 1024:
                raise ValueError("Training video dimensions are unsupported.")
            if len(frames) >= 241 or (len(frames) + 1) * height * width > MAX_DECODED_PIXELS:
                raise ValueError("Training video exceeds the decoded size limit.")
            if frames and frame.shape != frames[0].shape:
                raise ValueError("Training video dimensions changed.")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) != expected_frames:
            raise ValueError("Training video could not be decoded completely.")
        shape = (len(frames), *frames[0].shape) if frames else (0, 0, 0, 3)
        validate_video_shape(shape, config)
        return np.stack(frames)
    finally:
        reader.release()


@dataclass(frozen=True)
class PreparedTrainingBatch:
    clean: object
    noise: object
    context: object
    sigma: object
    source_identities: dict
    clip_digest: str
    caption_digest: str

    def tensors(self):
        return self.clean, self.noise, self.context, self.sigma


def prepare_training_batch(path, clip, caption, *, seed=42, sigma=0.5):
    """Encode one complete clip and its caption using local converted weights."""
    if (
        type(caption) is not str or not caption.strip() or len(caption) > 2000
        or any(ord(value) < 32 and value not in "\n\t" for value in caption)
        or type(seed) is not int or not 0 <= seed < 2**32
        or type(sigma) not in (int, float) or not math.isfinite(sigma) or not 0 <= sigma <= 1
    ):
        raise ValueError("Invalid training caption, seed or sigma.")
    root, clip = Path(path).expanduser(), Path(clip).expanduser()
    raw, identities = training_sources(root)
    clip_digest = _digest_regular(clip, MAX_VIDEO_BYTES)
    pixels = _decode_video(clip, raw)

    import mlx.core as mx
    import numpy as np
    from mlx_video.models.wan.loading import encode_text
    from mlx_video.models.wan.text_encoder import T5Encoder
    from mlx_video.models.wan.vae import WanVAE
    from transformers import AutoTokenizer

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise ValueError("Wan MLX training requires the Metal GPU.")
    config = _config(raw)
    encoder = T5Encoder(vocab_size=config.t5_vocab_size, dim=config.t5_dim,
                        dim_attn=config.t5_dim_attn, dim_ffn=config.t5_dim_ffn,
                        num_heads=config.t5_num_heads, num_layers=config.t5_num_layers,
                        num_buckets=config.t5_num_buckets, shared_pos=False)
    weights = mx.load(root / "t5_encoder.safetensors")
    encoder.load_weights(list(weights.items()), strict=True)
    mx.eval(encoder.parameters())
    del weights
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True,
                                               trust_remote_code=False)
    context = encode_text(encoder, tokenizer, caption, config.text_len)
    mx.eval(context)
    context = mx.array(np.asarray(context.astype(mx.float32)))[None, :, :]
    del encoder, tokenizer
    gc.collect()
    mx.clear_cache()

    vae = WanVAE(z_dim=16, encoder=True)
    weights = mx.load(root / "vae.safetensors")
    # Converted VAE files contain learned tensors; normalization is model-defined.
    for name in ("mean", "std", "inv_std"):
        weights.setdefault(name, getattr(vae, name))
    vae.load_weights(list(weights.items()), strict=True)
    mx.eval(vae.parameters())
    del weights
    video = mx.array(pixels.astype(np.float32) / 127.5 - 1).transpose(3, 0, 1, 2)[None, :, :, :, :]
    clean = vae.encode(video)
    mx.eval(clean)
    clean = mx.array(np.asarray(clean))
    del vae, video, pixels
    gc.collect()
    mx.clear_cache()
    noise = mx.random.normal(clean.shape, key=mx.random.key(seed))
    flow_sigma = mx.array([sigma], dtype=mx.float32)
    mx.eval(clean, noise, context, flow_sigma)
    if any(not mx.all(mx.isfinite(value)).item() for value in (clean, noise, context)):
        raise ValueError("Converted encoders produced non-finite training tensors.")
    _unchanged(root, identities)
    if _digest_regular(clip, MAX_VIDEO_BYTES) != clip_digest:
        raise ValueError("Training media changed during encoding.")
    return PreparedTrainingBatch(clean, noise, context, flow_sigma, identities, clip_digest,
                                  "sha256:" + hashlib.sha256(caption.encode()).hexdigest())
