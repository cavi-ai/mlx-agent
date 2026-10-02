"""Audio8 ASR Infinite: Voxtral Realtime audio tower + Qwen2 decoder with delay conditioning.

Port of the Edge0/Audio8-ASR-Infinite simulated-streaming greedy decoder:
one text token per 80 ms clock step, the audio of each step fused into the
token embedding, the transcription delay applied through a sinusoidal time
embedding plus a frame-length embedding that scales every decoder layer's
MLP input. Weight names are the Hugging Face checkpoint's own.
"""

from __future__ import annotations

import glob
import math
import shutil
import time
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.base import STTOutput

from .audio import LogMel, as_waveform, stream_layout, window_batches
from .config import SAMPLE_RATE, AudioConfig, ModelConfig, TextConfig

STREAMING_PAD_TOKEN = "[STREAMING_PAD]"
STREAMING_WORD_TOKEN = "[STREAMING_WORD]"
LANGUAGE_TOKENS = {"en": "[LANGUAGE_EN]", "zh": "[LANGUAGE_ZH]"}
DEFAULT_DELAY_MS = 480
RIGHT_PAD_TEXT_TOKENS = 10
ROLLING_CONTEXT_TOKENS = 375
ROLLING_TRIM_INTERVAL_TOKENS = 38
ROLLING_STABLE_PREFIX_TOKENS = 16
WINDOW_BATCH = 1024
SUPPORTING_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "generation_config.json",
)


def _rope_inv_freq(head_dim: int, theta: float) -> mx.array:
    return 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))


def _rotate(x: mx.array, angle: mx.array) -> mx.array:
    """Apply the rotate-half RoPE rotation by ``angle`` [..., positions, head_dim / 2]."""
    cos = mx.cos(mx.concatenate([angle, angle], axis=-1)).astype(x.dtype)
    sin = mx.sin(mx.concatenate([angle, angle], axis=-1)).astype(x.dtype)
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


def prepare_config(config: dict, model_path: Path) -> dict:
    """Converter hook: drop ``auto_map`` so the converted directory never points at the repository's code."""
    return {key: value for key, value in config.items() if key != "auto_map"}


def copy_supporting_files(source: Path, destination: Path) -> None:
    """Converter hook: copy the tokenizer and processor files, never the repository's Python or extra weights."""
    for name in SUPPORTING_FILES:
        for found in glob.glob(str(Path(source) / name)):
            shutil.copy(found, destination)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride)
        self.left_pad = kernel_size - stride

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(mx.pad(x, [(0, 0), (self.left_pad, 0), (0, 0)]))


class Embedder(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.conv1 = CausalConv1d(config.num_mel_bins, config.hidden_size, 3)
        self.conv2 = CausalConv1d(config.hidden_size, config.hidden_size, 3, stride=2)

    def __call__(self, features: mx.array) -> mx.array:
        return nn.gelu(self.conv2(nn.gelu(self.conv1(features))))


class EncoderAttention(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias=True)

    def __call__(self, x: mx.array, mask: mx.array, cache: dict) -> mx.array:
        """``cache`` holds this layer's un-rotated keys and values of the frames still inside the window."""
        length = x.shape[1]
        q = self.q_proj(x).reshape(1, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(1, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(1, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        if cache.get("keys") is not None:
            k = mx.concatenate([cache["keys"], k], axis=2)
            v = mx.concatenate([cache["values"], v], axis=2)
        cache["chunk_keys"], cache["chunk_values"] = k, v
        past = k.shape[2] - length
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.theta, scale=1.0, offset=past)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.theta, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(1, length, -1))


class EncoderMLP(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class EncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.self_attn = EncoderAttention(config)
        self.self_attn_layer_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layer_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = EncoderMLP(config)

    def __call__(self, x: mx.array, mask: mx.array, cache: dict) -> mx.array:
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask, cache)
        return x + self.mlp(self.final_layer_norm(x))


class AudioTower(nn.Module):
    """Causal Voxtral Realtime encoder with sliding-window attention over the 20 ms frames."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.embedder = Embedder(config)
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def encode(self, frames: mx.array, chunk: Optional[int] = None) -> mx.array:
        """[1, T, hidden] conv frames -> [1, T, hidden] hidden states, chunk by chunk.

        Keys are cached un-rotated and re-rotated from position 0 each chunk:
        attention depends only on relative offsets, so positions stay bounded
        however long the audio is.
        """
        window = self.config.sliding_window
        chunk = chunk or window
        caches = [dict() for _ in self.layers]
        outputs = []
        for start in range(0, frames.shape[1], chunk):
            x = frames[:, start : start + chunk]
            length = x.shape[1]
            past = 0 if caches[0].get("keys") is None else caches[0]["keys"].shape[2]
            query_positions = mx.arange(past, past + length)[:, None]
            key_positions = mx.arange(past + length)[None, :]
            offsets = query_positions - key_positions
            mask = (offsets >= 0) & (offsets < window)
            for layer, cache in zip(self.layers, caches):
                x = layer(x, mask, cache)
                keep = window - 1
                cache["keys"] = cache.pop("chunk_keys")[:, :, -keep:]
                cache["values"] = cache.pop("chunk_values")[:, :, -keep:]
            x = self.norm(x)
            mx.eval(x, [(cache["keys"], cache["values"]) for cache in caches])
            outputs.append(x)
        return mx.concatenate(outputs, axis=1)


class Projector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.text_config.hidden_size
        self.linear_1 = nn.Linear(config.projection_size, hidden, bias=False)
        self.linear_2 = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.gelu(self.linear_1(x)))


class TextAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias=False)

    def __call__(self, x: mx.array, cache: "TextCache", layer: int) -> mx.array:
        length = x.shape[1]
        offset = cache.length
        q = self.q_proj(x).reshape(1, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(1, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(1, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.theta, scale=1.0, offset=offset)
        k, v = cache.update(layer, k, v)
        mask = "causal" if length > 1 else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(1, length, -1))


class TextMLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class AdaRmsNorm(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, 32, bias=False)
        self.linear2 = nn.Linear(32, config.hidden_size, bias=False)

    def __call__(self, t_cond: mx.array) -> mx.array:
        return self.linear2(nn.gelu(self.linear1(t_cond)))


class TextLayer(nn.Module):
    """Qwen2 decoder layer with the realtime delay scaling before the MLP."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = TextAttention(config)
        self.mlp = TextMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ada_rms_norm = AdaRmsNorm(config)

    def __call__(self, x: mx.array, scale: mx.array, cache: "TextCache", layer: int) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cache, layer)
        return x + self.mlp(self.post_attention_layernorm(x) * scale)


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TextLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.model = TextModel(config)

    def __call__(self, embeds: mx.array, scales: List[mx.array], cache: "TextCache") -> mx.array:
        x = embeds
        for index, layer in enumerate(self.model.layers):
            x = layer(x, scales[index], cache, index)
        cache.advance(embeds.shape[1])
        return self.model.embed_tokens.as_linear(self.model.norm(x[:, -1:]))


class TextCache:
    """Decoder KV cache with the reference rolling window: a stable head plus a re-based tail."""

    def __init__(self, layers: int, head_dim: int, theta: float, context: int = ROLLING_CONTEXT_TOKENS,
                 interval: int = ROLLING_TRIM_INTERVAL_TOKENS, stable_prefix: int = ROLLING_STABLE_PREFIX_TOKENS):
        self.keys: List[Optional[mx.array]] = [None] * layers
        self.values: List[Optional[mx.array]] = [None] * layers
        self.length = 0
        self.inv_freq = _rope_inv_freq(head_dim, theta)
        self.context = context
        self.target = context - (interval - 1) if interval > 1 else context
        self.stable_prefix = min(stable_prefix, max(1, context - 1)) if context > 0 else 0

    def update(self, layer: int, keys: mx.array, values: mx.array):
        if self.keys[layer] is not None:
            keys = mx.concatenate([self.keys[layer], keys], axis=2)
            values = mx.concatenate([self.values[layer], values], axis=2)
        self.keys[layer], self.values[layer] = keys, values
        return keys, values

    def advance(self, tokens: int) -> None:
        self.length += tokens
        if self.context <= 0 or self.length <= self.context:
            return
        trim = self.length - self.target
        head = self.stable_prefix
        angle = (-float(trim) * self.inv_freq)[None, None, None, :]
        for layer, (keys, values) in enumerate(zip(self.keys, self.values)):
            tail_keys = _rotate(keys[:, :, head + trim :].astype(mx.float32), angle).astype(keys.dtype)
            self.keys[layer] = mx.concatenate([keys[:, :, :head], tail_keys], axis=2)
            self.values[layer] = mx.concatenate([values[:, :, :head], values[:, :, head + trim :]], axis=2)
        self.length = self.target


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.audio_tower = AudioTower(config.audio_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = Projector(config)
        if config.use_frame_len_embedding:
            self.frame_len_embedding = nn.Embedding(len(config.supported_frame_lens), config.text_config.hidden_size)
        if config.semantic_vad_horizons_seconds:
            self.semantic_vad_heads = [
                nn.Linear(config.text_config.hidden_size, config.semantic_vad_num_classes, bias=True)
                for _ in config.semantic_vad_horizons_seconds
            ]
        self._tokenizer = None
        self._log_mel = None

    def sanitize(self, weights: dict) -> dict:
        sanitized = {}
        for key, value in weights.items():
            if key == "language_model.lm_head.weight":
                continue
            if key.startswith("audio_tower.embedder.conv") and key.endswith(".weight") and value.shape[-1] == 3:
                value = value.transpose(0, 2, 1)
            sanitized[key] = value
        return sanitized

    def model_quant_predicate(self, path: str, module: nn.Module) -> bool:
        return path.startswith("language_model.") and "ada_rms_norm" not in path

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from tokenizers import Tokenizer

        model._tokenizer = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
        return model

    def _token_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError("The tokenizer has no {0} token.".format(token))
        return int(token_id)

    def _frame_len(self, frame_len: Optional[int]) -> int:
        frame_len = int(frame_len or self.config.supported_frame_lens[0])
        if frame_len not in self.config.supported_frame_lens:
            raise ValueError("frame_len must be one of {0}.".format(self.config.supported_frame_lens))
        return frame_len

    def delay_scales(self, num_delay_tokens: int, frame_len: int) -> List[mx.array]:
        """Per-layer MLP input scale ``1 + ada(t_cond)``, computed in the model dtype like the reference."""
        hidden = self.config.text_config.hidden_size
        dtype = self.language_model.model.norm.weight.dtype
        inv_freq = mx.exp(-math.log(10000.0) * mx.arange(hidden // 2, dtype=mx.float32) / (hidden // 2))
        phase = mx.array(float(num_delay_tokens), dtype=dtype) * inv_freq.astype(dtype)
        t_cond = mx.concatenate([mx.cos(phase), mx.sin(phase)])
        if self.config.use_frame_len_embedding:
            index = self.config.supported_frame_lens.index(frame_len)
            t_cond = t_cond + self.frame_len_embedding.weight[index].astype(dtype)
        t_cond = t_cond[None, None, :]
        return [1 + layer.ada_rms_norm(t_cond) for layer in self.language_model.model.layers]

    def audio_embeds(self, waveform: np.ndarray, *, frame_len: int, num_delay_tokens: int, left_pad_tokens: int):
        """Projected audio for every text position: the prefill window, then one token per clock step."""
        if self._log_mel is None:
            self._log_mel = LogMel(self.config.audio_config.num_mel_bins, SAMPLE_RATE)
        samples_per_token = SAMPLE_RATE * frame_len * self.config.audio_tower_frame_ms // 1000
        layout = stream_layout(
            waveform, samples_per_token=samples_per_token, left_pad_tokens=left_pad_tokens,
            num_delay_tokens=num_delay_tokens, right_pad_text_tokens=RIGHT_PAD_TEXT_TOKENS,
            sampling_rate=SAMPLE_RATE,
        )
        dtype = self.multi_modal_projector.linear_2.weight.dtype
        if not mx.issubdtype(dtype, mx.floating):
            dtype = mx.bfloat16
        conv_frames = []
        for audio, tokens in window_batches(layout):
            for start in range(0, audio.shape[0], WINDOW_BATCH):
                features = self._log_mel(audio[start : start + WINDOW_BATCH]).astype(dtype)
                frames = self.audio_tower.embedder(features)[:, -tokens * frame_len :]
                conv_frames.append(frames.reshape(1, -1, frames.shape[-1]))
                mx.eval(conv_frames[-1])
        hidden = self.audio_tower.encode(mx.concatenate(conv_frames, axis=1))
        width = self.config.audio_config.hidden_size
        grouped = hidden.reshape(layout.total_tokens, frame_len * width)
        if frame_len < self.config.max_frame_len:
            grouped = mx.pad(grouped, [(0, 0), (0, (self.config.max_frame_len - frame_len) * width)])
        embeds = self.multi_modal_projector(grouped)
        mx.eval(embeds)
        return embeds, layout

    def generate(
        self,
        audio: Union[str, Path, mx.array, np.ndarray, List[mx.array]],
        *,
        language: str = "en",
        transcription_delay_ms: Optional[int] = None,
        frame_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> STTOutput:
        language_token = LANGUAGE_TOKENS.get(str(language).strip().lower())
        if language_token is None:
            raise ValueError("Audio8 ASR Infinite transcribes en or zh, got {0!r}.".format(language))
        frame_len = self._frame_len(frame_len)
        token_ms = frame_len * self.config.audio_tower_frame_ms
        delay_ms = int(transcription_delay_ms or DEFAULT_DELAY_MS)
        if delay_ms <= 0 or delay_ms % token_ms:
            raise ValueError("transcription_delay_ms must be a positive multiple of {0} ms.".format(token_ms))
        num_delay_tokens = delay_ms // token_ms
        left_pad_tokens = self.config.left_pad_tokens(frame_len)

        start_time = time.time()
        waveform = as_waveform(self._load_audio(audio))
        embeds, layout = self.audio_embeds(
            waveform, frame_len=frame_len, num_delay_tokens=num_delay_tokens, left_pad_tokens=left_pad_tokens,
        )
        scales = self.delay_scales(num_delay_tokens, frame_len)
        text = self.config.text_config
        cache = TextCache(text.num_hidden_layers, text.head_dim, text.rope_theta)
        embed_tokens = self.language_model.model.embed_tokens
        pad_id = self._token_id(STREAMING_PAD_TOKEN)
        eos_id = int(self.config.eos_token_id)
        prompt = [int(self.config.bos_token_id), self._token_id(language_token)]
        prompt += [pad_id] * (layout.prefill_tokens - 2)

        def step(token_ids: List[int], position: int) -> mx.array:
            inputs = embed_tokens(mx.array([token_ids])) + embeds[None, position : position + len(token_ids)]
            logits = self.language_model(inputs, scales, cache)[0, -1].astype(mx.float32)
            return mx.argmax(mx.where(mx.arange(logits.shape[0]) == eos_id, -mx.inf, logits))

        limit = layout.windows if max_tokens is None else min(layout.windows, int(max_tokens))
        generated: List[int] = []
        next_token = step(prompt, 0)
        decode_start = time.time()
        for position in range(layout.prefill_tokens, layout.prefill_tokens + limit - 1):
            token = int(next_token.item())
            generated.append(token)
            next_token = step([token], position)
            mx.async_eval(next_token)
        if limit > 0:
            generated.append(int(next_token.item()))
        end_time = time.time()

        text_value = self._decode(generated)
        duration = waveform.shape[0] / SAMPLE_RATE
        if verbose:
            elapsed = end_time - decode_start
            print("Decode: {0} steps in {1:.2f}s ({2:.1f} steps/s); audio {3:.2f}s".format(
                len(generated), elapsed, len(generated) / max(elapsed, 1e-9), duration))
        total = end_time - start_time
        return STTOutput(
            text=text_value,
            segments=[{"text": text_value, "start": 0.0, "end": round(duration, 3)}],
            language=str(language).strip().lower(),
            prompt_tokens=layout.prefill_tokens,
            generation_tokens=len(generated),
            total_tokens=layout.prefill_tokens + len(generated),
            prompt_tps=layout.prefill_tokens / total if total > 0 else 0.0,
            generation_tps=len(generated) / (end_time - decode_start) if end_time > decode_start else 0.0,
            total_time=total,
        )

    def _decode(self, token_ids: List[int]) -> str:
        skipped = {
            self._token_id(STREAMING_PAD_TOKEN),
            self._token_id(STREAMING_WORD_TOKEN),
            int(self.config.bos_token_id),
            int(self.config.pad_token_id),
        }
        visible = [token for token in token_ids if token not in skipped]
        return self._tokenizer.decode(visible, skip_special_tokens=True).strip()

    @staticmethod
    def _load_audio(audio) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            from mlx_audio.stt.utils import load_audio

            return np.array(load_audio(str(audio), sr=SAMPLE_RATE))
        if isinstance(audio, list):
            audio = audio[0]
        return np.array(audio)
