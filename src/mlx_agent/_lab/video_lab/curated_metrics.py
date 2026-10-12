"""Bounded deterministic image metrics for curated validation."""

from __future__ import annotations

import hashlib
from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np
from PIL import Image

from video_lab.curated_source import DecodedFrame

METRIC_QUANTUM = Decimal("0.0001")


def quantize_metric(value: float | int | Decimal) -> str:
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError("metric must be finite")
    return format(decimal.quantize(METRIC_QUANTUM, rounding=ROUND_HALF_EVEN), "f")


def frame_array(frame: DecodedFrame) -> np.ndarray:
    expected = frame.width * frame.height * frame.channels
    if len(frame.pixels) != expected or frame.channels not in (3, 4):
        raise ValueError("decoded frame byte contract is invalid")
    return np.frombuffer(frame.pixels, dtype=np.uint8).reshape(
        frame.height, frame.width, frame.channels
    )


def perceptual_hash64(frame: DecodedFrame) -> int:
    array = frame_array(frame)
    image = Image.fromarray(array, mode="RGB" if frame.channels == 3 else "RGBA")
    grayscale = image.convert("L").resize((8, 8), resample=Image.Resampling.LANCZOS)
    values = np.asarray(grayscale, dtype=np.uint8).reshape(64)
    median = int(np.median(values))
    result = 0
    for value in values:
        result = (result << 1) | int(int(value) > median)
    return result


def hamming_distance64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def blur_variance(frame: DecodedFrame) -> float:
    rgb = frame_array(frame)[..., :3].astype(np.float64)
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(laplacian, dtype=np.float64))


def histogram_discontinuity(left: DecodedFrame, right: DecodedFrame) -> float:
    left_array = frame_array(left)[..., :3]
    right_array = frame_array(right)[..., :3]
    distances: list[float] = []
    for channel in range(3):
        left_hist = np.bincount(left_array[..., channel].reshape(-1), minlength=256)
        right_hist = np.bincount(right_array[..., channel].reshape(-1), minlength=256)
        left_norm = left_hist / left_hist.sum()
        right_norm = right_hist / right_hist.sum()
        distances.append(float(np.abs(left_norm - right_norm).sum() / 2.0))
    return sum(distances) / 3.0


def temporal_motion(left: DecodedFrame, right: DecodedFrame) -> float:
    left_array = frame_array(left)
    right_array = frame_array(right)
    if left_array.shape != right_array.shape:
        return 1.0
    difference = np.abs(left_array[..., :3].astype(np.int16) - right_array[..., :3])
    return float(np.mean(difference, dtype=np.float64) / 255.0)


def metric_policy_digest() -> str:
    policy = b"phash64:lanczos-l8-median;blur:laplacian4-edge;hist:rgb256-l1;motion:rgb-mae"
    return hashlib.sha256(policy).hexdigest()
