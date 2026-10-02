"""The bundled Audio8 ASR Infinite port; runs where mlx-audio is importable (the backend venv)."""

import importlib
import importlib.util
import math
import sys
import unittest

from mlx_agent.backends import PORTS_DIR

HAS_MLX_AUDIO = importlib.util.find_spec("mlx_audio") is not None


def load_port():
    sys.path.insert(0, str(PORTS_DIR / "mlx-audio"))
    try:
        return importlib.import_module("audio8_asr_infinite")
    finally:
        sys.path.pop(0)


def tiny_config(**overrides):
    config = {
        "model_type": "audio8_asr_infinite", "weight_format_version": 2,
        "audio_config": {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
                         "head_dim": 8, "num_mel_bins": 128, "sliding_window": 5,
                         "rope_parameters": {"rope_theta": 1000000.0}},
        "text_config": {"model_type": "qwen2", "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 2,
                        "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 40,
                        "rope_parameters": {"rope_theta": 1000000.0}},
        "supported_frame_lens": [4, 6, 8], "use_frame_len_embedding": True,
        "semantic_vad_horizons_seconds": [0.5, 1.0], "semantic_vad_num_classes": 8,
        "streaming_n_left_pad_tokens": 18, "bos_token_id": 30, "eos_token_id": 31, "pad_token_id": 32,
    }
    config.update(overrides)
    return config


class FakeTokenizer:
    ids = {"[STREAMING_PAD]": 33, "[STREAMING_WORD]": 34, "[LANGUAGE_EN]": 35, "[LANGUAGE_ZH]": 36}

    def token_to_id(self, token):
        return self.ids.get(token)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(value) for value in ids)


@unittest.skipUnless(HAS_MLX_AUDIO, "mlx-audio is not installed in this interpreter")
class Audio8PortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np

        cls.mx, cls.np = mx, np
        cls.port = load_port()
        cls.audio = importlib.import_module("audio8_asr_infinite.audio")
        cls.model_module = importlib.import_module("audio8_asr_infinite.audio8_asr_infinite")

    def test_windows_follow_the_reference_clock(self):
        np = self.np
        for samples in (0, 1, 1279, 1280, 1281, 40_000):
            layout = self.audio.stream_layout(np.zeros(samples, np.float32), samples_per_token=1280, left_pad_tokens=18,
                                              num_delay_tokens=6, right_pad_text_tokens=10, sampling_rate=16000)
            stream, expected, end = layout.stream.shape[0], 0, 25 * 1280
            while end + 40 <= stream:
                expected, end = expected + 1, end + 1280
            self.assertEqual(layout.windows, expected, samples)
            batches = list(self.audio.window_batches(layout))
            self.assertEqual(batches[0][0].shape, (1, 25 * 1280 + 40))
            self.assertEqual(batches[0][1], 25)
            if expected > 1:
                self.assertEqual(batches[1][0].shape, (expected - 1, 840 + 1280 + 40))
                self.assertTrue(np.array_equal(batches[1][0][0], layout.stream[25 * 1280 - 840: 26 * 1280 + 40]))
            self.assertEqual(layout.total_tokens, 25 + expected - 1)

    def test_log_mel_matches_a_centered_numpy_stft(self):
        np = self.np
        rng = np.random.default_rng(0)
        audio = rng.uniform(-0.5, 0.5, size=(2, 2160)).astype(np.float32)
        ours = np.array(self.audio.LogMel()(audio))
        filters = np.array(self.audio.LogMel().filters)
        padded = np.pad(audio, ((0, 0), (200, 200)), mode="reflect")
        window = 0.5 * (1 - np.cos(2 * math.pi * np.arange(400) / 400))
        frames = np.stack([padded[:, i * 160: i * 160 + 400] for i in range(1 + 2160 // 160)], axis=1)
        power = np.abs(np.fft.rfft(frames * window, n=400, axis=-1))[:, :-1] ** 2
        expected = (np.maximum(np.log10(np.maximum(power @ filters, 1e-10)), -6.5) + 4.0) / 4.0
        self.assertEqual(ours.shape, (2, 13, 128))
        self.assertLess(np.abs(ours - expected).max(), 1e-3)

    def test_chunked_encoder_equals_one_pass_with_the_sliding_window(self):
        mx = self.mx
        # CPU float32: the GPU matmul rounds differently per batch shape, which would mask the comparison.
        with mx.stream(mx.cpu):
            mx.random.seed(0)
            model = self.port.Model(self.port.ModelConfig.from_dict(tiny_config()))
            frames = mx.random.normal((1, 23, 16))
            whole = model.audio_tower.encode(frames, chunk=23)
            for chunk in (1, 2, 5, 7):
                chunked = model.audio_tower.encode(frames, chunk=chunk)
                self.assertLess(float(mx.abs(chunked - whole).max()), 1e-5, chunk)

    def test_rolling_trim_keeps_the_head_and_rebases_the_tail(self):
        mx = self.mx
        cache = self.model_module.TextCache(1, 8, 10000.0, context=10, interval=4, stable_prefix=3)
        raw = mx.random.normal((1, 1, 11, 8))
        roped = mx.fast.rope(raw, 8, traditional=False, base=10000.0, scale=1.0, offset=0)
        cache.update(0, roped, raw)
        cache.advance(11)
        self.assertEqual(cache.length, 7)
        trim = 11 - 7
        keys = cache.keys[0]
        self.assertLess(float(mx.abs(keys[:, :, :3] - roped[:, :, :3]).max()), 1e-6)
        rebased = mx.fast.rope(raw[:, :, 3 + trim:], 8, traditional=False, base=10000.0, scale=1.0, offset=3)
        self.assertLess(float(mx.abs(keys[:, :, 3:] - rebased).max()), 1e-4)
        self.assertTrue(bool(mx.array_equal(cache.values[0], mx.concatenate([raw[:, :, :3], raw[:, :, 3 + trim:]], axis=2))))

    def test_sanitize_transposes_torch_convs_once_and_drops_the_tied_head(self):
        mx = self.mx
        model = self.port.Model(self.port.ModelConfig.from_dict(tiny_config()))
        weights = {"audio_tower.embedder.conv1.weight": mx.zeros((16, 128, 3)), "language_model.lm_head.weight": mx.zeros((40, 64))}
        once = model.sanitize(weights)
        self.assertEqual(once["audio_tower.embedder.conv1.weight"].shape, (16, 3, 128))
        self.assertNotIn("language_model.lm_head.weight", once)
        self.assertEqual(model.sanitize(once)["audio_tower.embedder.conv1.weight"].shape, (16, 3, 128))
        self.assertEqual(self.port.prepare_config({"auto_map": {}, "model_type": "x"}, None), {"model_type": "x"})

    def test_quantization_covers_the_decoder_only(self):
        model = self.port.Model(self.port.ModelConfig.from_dict(tiny_config()))
        self.assertTrue(model.model_quant_predicate("language_model.model.layers.0.mlp.up_proj", None))
        self.assertTrue(model.model_quant_predicate("language_model.model.embed_tokens", None))
        self.assertFalse(model.model_quant_predicate("language_model.model.layers.0.ada_rms_norm.linear1", None))
        self.assertFalse(model.model_quant_predicate("audio_tower.layers.0.mlp.up_proj", None))
        self.assertFalse(model.model_quant_predicate("multi_modal_projector.linear_1", None))

    def test_generate_emits_one_token_per_clock_step(self):
        mx, np = self.mx, self.np
        mx.random.seed(1)
        model = self.port.Model(self.port.ModelConfig.from_dict(tiny_config()))
        model._tokenizer = FakeTokenizer()
        audio = np.random.default_rng(1).uniform(-0.1, 0.1, 16000).astype(np.float32)
        out = model.generate(audio, language="en")
        layout = self.audio.stream_layout(audio, samples_per_token=1280, left_pad_tokens=18, num_delay_tokens=6,
                                          right_pad_text_tokens=10, sampling_rate=16000)
        self.assertEqual(out.generation_tokens, layout.windows)
        self.assertEqual(out.prompt_tokens, 25)
        self.assertEqual(model.generate(audio, language="zh", max_tokens=3).generation_tokens, 3)
        for bad in ({"language": "fr"}, {"transcription_delay_ms": 100}, {"frame_len": 5}):
            with self.assertRaises(ValueError):
                model.generate(audio, **bad)


if __name__ == "__main__":
    unittest.main()
