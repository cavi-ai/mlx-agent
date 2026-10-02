"""Streaming audio frontend: the clock windows and log-mel features of the reference decoder.

Each 80 ms clock step reads its own window (52.5 ms look-back, 2.5 ms
look-ahead) and computes a centered, reflect-padded STFT over that window
alone, so window edges shape the features exactly as in the reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.voxtral_realtime.audio import compute_mel_filters

N_FFT = 400
HOP_LENGTH = 160
LOOK_BACK_MS = 52.5
LOOK_AHEAD_MS = 2.5


def ms_to_samples(milliseconds: float, sampling_rate: int) -> int:
    samples = sampling_rate * float(milliseconds) / 1000.0
    if not samples.is_integer():
        raise ValueError("{0} ms is not a whole number of samples at {1} Hz.".format(milliseconds, sampling_rate))
    return int(samples)


@dataclass
class StreamLayout:
    """The padded stream and its clock windows: window 0 is the prefill, every later window one token."""

    stream: np.ndarray
    samples_per_token: int
    prefill_tokens: int
    windows: int
    look_back: int
    look_ahead: int

    @property
    def total_tokens(self) -> int:
        return self.prefill_tokens + self.windows - 1


def stream_layout(waveform: np.ndarray, *, samples_per_token: int, left_pad_tokens: int, num_delay_tokens: int,
                  right_pad_text_tokens: int, sampling_rate: int) -> StreamLayout:
    prefill_tokens = left_pad_tokens + num_delay_tokens + 1
    right_pad_tokens = num_delay_tokens + 1 + right_pad_text_tokens
    stream = np.concatenate([
        np.zeros(left_pad_tokens * samples_per_token, dtype=np.float32),
        waveform.astype(np.float32, copy=False),
        np.zeros(right_pad_tokens * samples_per_token, dtype=np.float32),
    ])
    look_back = ms_to_samples(LOOK_BACK_MS, sampling_rate)
    look_ahead = ms_to_samples(LOOK_AHEAD_MS, sampling_rate)
    first_end = prefill_tokens * samples_per_token
    last_end = stream.shape[0] - look_ahead
    windows = 0 if last_end < first_end else (last_end - first_end) // samples_per_token + 1
    return StreamLayout(stream, samples_per_token, prefill_tokens, windows, look_back, look_ahead)


def window_batches(layout: StreamLayout):
    """Yield (audio [rows, samples], tokens per row): the prefill window, then the one-token windows."""
    step = layout.samples_per_token
    first_end = layout.prefill_tokens * step
    yield layout.stream[None, : first_end + layout.look_ahead], layout.prefill_tokens
    if layout.windows > 1:
        starts = first_end + step * np.arange(layout.windows - 1)
        offsets = np.arange(-layout.look_back, step + layout.look_ahead)
        yield layout.stream[starts[:, None] + offsets[None, :]], 1


class LogMel:
    def __init__(self, num_mel_bins: int = 128, sampling_rate: int = 16000, global_log_mel_max: float = 1.5):
        self.filters = mx.array(compute_mel_filters(num_mel_bins, N_FFT, sampling_rate), dtype=mx.float32)
        n = mx.arange(N_FFT, dtype=mx.float32)
        self.window = 0.5 * (1.0 - mx.cos(2.0 * math.pi * n / N_FFT))
        self.floor = global_log_mel_max - 8.0

    def __call__(self, audio: np.ndarray) -> mx.array:
        """[rows, samples] waveform -> [rows, frames, mel] features (torch.stft center=True, last frame dropped)."""
        pad = N_FFT // 2
        padded = mx.array(np.pad(audio, ((0, 0), (pad, pad)), mode="reflect"), dtype=mx.float32)
        frames = 1 + (padded.shape[1] - N_FFT) // HOP_LENGTH
        indices = mx.arange(N_FFT)[None, :] + (mx.arange(frames) * HOP_LENGTH)[:, None]
        spectrum = mx.fft.rfft(padded[:, indices] * self.window, n=N_FFT, axis=-1)
        power = mx.abs(spectrum[:, :-1, :]) ** 2
        mel = power @ self.filters
        log_mel = mx.maximum(mx.log10(mx.maximum(mel, 1e-10)), self.floor)
        return (log_mel + 4.0) / 4.0


def as_waveform(samples: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
