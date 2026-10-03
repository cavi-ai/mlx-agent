from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mlx_audio.base import BaseModelArgs

SAMPLE_RATE = 16000


def _rope_theta(params: Optional[dict], default: float) -> float:
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])
    return default


@dataclass
class AudioConfig(BaseModelArgs):
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    head_dim: int = 64
    num_mel_bins: int = 128
    rms_norm_eps: float = 1e-5
    rope_parameters: Optional[dict] = None
    rope_theta: float = 1_000_000.0
    sliding_window: int = 750

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.rope_theta = _rope_theta(self.rope_parameters, self.rope_theta)


@dataclass
class TextConfig(BaseModelArgs):
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-6
    rope_parameters: Optional[dict] = None
    rope_theta: float = 1_000_000.0
    vocab_size: int = 151936
    tie_word_embeddings: bool = True
    model_type: str = "qwen2"

    def __post_init__(self):
        if self.model_type != "qwen2":
            raise ValueError(
                "audio8_asr_infinite supports a qwen2 text decoder, got {0!r}.".format(self.model_type)
            )
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        self.rope_theta = _rope_theta(self.rope_parameters, self.rope_theta)


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "audio8_asr_infinite"
    audio_config: Optional[AudioConfig] = None
    text_config: Optional[TextConfig] = None
    supported_frame_lens: List[int] = field(default_factory=lambda: [4, 6, 8])
    audio_tower_frame_ms: int = 20
    use_frame_len_embedding: bool = False
    projector_hidden_act: str = "gelu"
    semantic_vad_horizons_seconds: Optional[List[float]] = None
    semantic_vad_num_classes: int = 8
    streaming_n_left_pad_tokens: int = 18
    streaming_n_left_pad_tokens_by_frame_len: Optional[Dict[str, int]] = None
    weight_format_version: int = 2
    bos_token_id: int = 151644
    eos_token_id: int = 151645
    pad_token_id: int = 151643

    def __post_init__(self):
        if self.weight_format_version != 2:
            raise ValueError(
                "audio8_asr_infinite loads weight_format_version 2, got {0}.".format(self.weight_format_version)
            )
        if isinstance(self.audio_config, dict):
            self.audio_config = AudioConfig.from_dict(self.audio_config)
        if self.audio_config is None:
            self.audio_config = AudioConfig()
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.text_config is None:
            self.text_config = TextConfig()
        self.supported_frame_lens = [int(value) for value in self.supported_frame_lens]
        self.use_frame_len_embedding = bool(self.use_frame_len_embedding and len(self.supported_frame_lens) > 1)

    @property
    def max_frame_len(self) -> int:
        return max(self.supported_frame_lens)

    @property
    def projection_size(self) -> int:
        return self.audio_config.hidden_size * self.max_frame_len

    def left_pad_tokens(self, frame_len: int) -> int:
        by_frame_len = self.streaming_n_left_pad_tokens_by_frame_len or {}
        return int(by_frame_len.get(str(frame_len), self.streaming_n_left_pad_tokens))
