"""Causal depth reuse against the pinned native MiniMax decoder."""
import importlib.util
from types import SimpleNamespace
import unittest

HAS_MLX = importlib.util.find_spec('mlx') is not None
HAS_AUDIO = HAS_MLX and importlib.util.find_spec('mlx_audio') is not None


@unittest.skipUnless(HAS_AUDIO, 'requires the native music backend')
class MusicDepthTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        from mlx_audio.music.models.minimax_music3.config import ModelConfig
        self.previous = mx.default_device()
        self.addCleanup(mx.set_default_device, self.previous)
        self.config = ModelConfig(hidden_size=64, depth_num_layers=2, depth_num_heads=4,
                                  depth_intermediate_size=128, audio_vocab_size=64,
                                  num_codebooks=8, vocab_size=256, audio_code_offset=32)

    def cached(self, decoder):
        from mlx_agent.music_speed import configure_depth_cache
        model = SimpleNamespace(model_type='minimax_music3', rvq_depth_decoder=decoder)
        self.assertTrue(configure_depth_cache(model))
        return model.rvq_depth_decoder

    def test_prefix_parity_on_cpu_gpu_float32_bfloat16_and_quantized_weights(self):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx_audio.music.models.minimax_music3.depth import RVQDepthDecoder
        for device in (mx.cpu, mx.gpu):
            mx.set_default_device(device)
            for dtype in (mx.float32, mx.bfloat16):
                for quantized in (False, True):
                    with self.subTest(device=device, dtype=dtype, quantized=quantized):
                        mx.random.seed(9)
                        decoder = RVQDepthDecoder(self.config)
                        decoder.apply(lambda x: x.astype(dtype))
                        if quantized:
                            nn.quantize(decoder, group_size=32, bits=8,
                                        class_predicate=lambda p,m: isinstance(m,nn.Linear))
                        values = mx.random.normal((2,8,64)).astype(dtype)
                        expected = [np.array(decoder(values[:,:n]).astype(mx.float32)) for n in range(2,9)]
                        lengths = []
                        original = decoder.layers[0].attn.to_k
                        class RecordingLinear(nn.Module):
                            def __call__(self,x):
                                lengths.append(x.shape[1])
                                return original(x)
                        decoder.layers[0].attn.to_k = RecordingLinear()
                        cached = self.cached(decoder)
                        for n,reference in zip(range(2,9),expected):
                            actual = np.array(cached(values[:,:n]).astype(mx.float32))
                            if dtype == mx.float32:
                                np.testing.assert_allclose(actual,reference,atol=2e-5,rtol=2e-4)
                            else:
                                self.assertTrue(np.isfinite(actual).all())
                                relative_rms = np.sqrt(np.mean((actual-reference)**2)/np.mean(reference**2))
                                self.assertLess(relative_rms,.02)
                        self.assertEqual(lengths,[2,1,1,1,1,1,1])

    def test_new_frames_and_nonconsecutive_prefixes_discard_old_context(self):
        import mlx.core as mx
        import numpy as np
        from mlx_audio.music.models.minimax_music3.depth import RVQDepthDecoder
        decoder = RVQDepthDecoder(self.config)
        cached = self.cached(decoder)
        for seed in (3,7):
            mx.random.seed(seed)
            values = mx.random.normal((2,8,64))
            for n in (2,3,8,2,3):
                np.testing.assert_allclose(np.array(cached(values[:,:n])),np.array(decoder(values[:,:n])),
                                           atol=2e-5,rtol=2e-4)

    def test_configuration_is_idempotent_and_other_models_are_untouched(self):
        from mlx_agent.music_speed import configure_depth_cache
        from mlx_audio.music.models.minimax_music3.depth import RVQDepthDecoder
        decoder = RVQDepthDecoder(self.config)
        model = SimpleNamespace(model_type='minimax_music3',rvq_depth_decoder=decoder)
        self.assertTrue(configure_depth_cache(model))
        wrapper = model.rvq_depth_decoder
        self.assertTrue(configure_depth_cache(model))
        self.assertIs(wrapper,model.rvq_depth_decoder)
        other = SimpleNamespace(model_type='other',rvq_depth_decoder=decoder)
        self.assertFalse(configure_depth_cache(other))
        self.assertIs(other.rvq_depth_decoder,decoder)

    def test_runtime_prefixes_do_not_become_model_parameters(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_audio.music.models.minimax_music3.depth import RVQDepthDecoder
        cached = self.cached(RVQDepthDecoder(self.config))
        before = [(k,v.shape) for k,v in tree_flatten(cached.parameters())]
        values = mx.random.normal((2,8,64))
        for n in range(2,9):mx.eval(cached(values[:,:n]))
        self.assertEqual(before,[(k,v.shape) for k,v in tree_flatten(cached.parameters())])

    def test_native_ar_codes_and_rng_match_for_float32_reference_frames(self):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx_audio.music.models.minimax_music3.ar import generate_depth_codes
        from mlx_audio.music.models.minimax_music3.depth import RVQDepthDecoder
        decoder = RVQDepthDecoder(self.config)
        cached = self.cached(decoder)
        language = SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(256,64)))
        for seed in (0,17,42):
            hidden = mx.random.normal((2,64))
            args = self.config, hidden, mx.array([3,3]), mx.random.key(seed)
            expected = generate_depth_codes(language,decoder,*args)
            actual = generate_depth_codes(language,cached,*args)
            np.testing.assert_array_equal(np.array(actual[0]),np.array(expected[0]))
            np.testing.assert_array_equal(np.array(actual[2]),np.array(expected[2]))
            np.testing.assert_allclose(np.array(actual[1]),np.array(expected[1]),atol=2e-5,rtol=2e-4)

    def test_custom_decoder_keeps_its_native_behavior(self):
        from mlx_agent.music_speed import configure_depth_cache
        from mlx_audio.music.models.minimax_music3.depth import RVQDepthDecoder
        class CustomDecoder(RVQDepthDecoder):
            def __call__(self,embeddings):
                return 'custom'
        decoder = CustomDecoder(self.config)
        model = SimpleNamespace(model_type='minimax_music3',rvq_depth_decoder=decoder)
        self.assertFalse(configure_depth_cache(model))
        self.assertIs(model.rvq_depth_decoder,decoder)
        self.assertEqual(model.rvq_depth_decoder(None),'custom')

@unittest.skipUnless(HAS_MLX, 'requires MLX on Apple Silicon')
class MusicMLXTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        import mlx.nn as nn
        self.previous = mx.default_device()
        self.addCleanup(mx.set_default_device, self.previous)
        self.config = object()

        class UniformDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.pos_embedding = nn.Embedding(16,4)
                self.pos_embedding.weight = mx.broadcast_to(mx.arange(16)[:,None],(16,4)).astype(mx.float32)/8
                self.projection = nn.Identity()
                self.audio_embeddings = nn.Embedding(4,4)
                self.audio_heads = [nn.Linear(4,4)]
                self.norm = nn.Identity()
                attention = SimpleNamespace(heads=1,head_dim=4,to_q=mx.zeros_like,to_k=mx.zeros_like,
                                            to_v=nn.Identity(),to_out=nn.Identity())
                self.layers = [SimpleNamespace(input_layernorm=nn.Identity(),attn=attention,
                    post_attention_layernorm=nn.Identity(),gate_proj=mx.zeros_like,
                    up_proj=mx.ones_like,down_proj=nn.Identity())]

        self.decoder_type = UniformDecoder

    def test_cached_prefix_matches_closed_form_causal_means(self):
        import mlx.core as mx
        import numpy as np
        from mlx_agent.music_speed import _DepthCache
        # Zero Q/K make causal attention the mean of preceding values. This
        # closed-form oracle is independent of the cached attention implementation.
        for device in (mx.cpu,mx.gpu):
            mx.set_default_device(device)
            decoder = self.decoder_type()
            calls = []
            def keys(x):
                calls.append(x.shape[1])
                return mx.zeros_like(x)
            decoder.layers[0].attn.to_k = keys
            cached = _DepthCache(decoder)
            for shift in (0,3):
                values = mx.arange(64).reshape(2,8,4).astype(mx.float32)/32+shift
                positioned = values+decoder.pos_embedding(mx.arange(8))
                expected = positioned+mx.cumsum(positioned,axis=1)/mx.arange(1,9)[None,:,None]
                calls.clear()
                for length in range(2,9):
                    rtol = np.finfo(np.float16).eps if device == mx.gpu else 2e-5
                    np.testing.assert_allclose(np.array(cached(values[:,:length])),
                                               np.array(expected[:,:length]),atol=2e-5,rtol=rtol)
                self.assertEqual(calls,[2,1,1,1,1,1,1])
            self.assertIs(cached.projection,decoder.projection)
            self.assertIs(cached.audio_embeddings,decoder.audio_embeddings)
            self.assertIs(cached.audio_heads,decoder.audio_heads)

    def test_runner_reuses_the_caches_and_exports_valid_stereo_pcm(self):
        import io
        import json
        from contextlib import redirect_stdout
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import sys
        from unittest.mock import Mock, patch
        import wave
        import mlx.core as mx
        import numpy as np
        from mlx_agent import music_runner
        from mlx_agent.music_speed import _DepthCache
        model = SimpleNamespace(model_type='minimax_music3',config=self.config,
                                rvq_depth_decoder=self.decoder_type())
        def generate(**kwargs):
            self.assertIsInstance(model.rvq_depth_decoder,_DepthCache)
            self.assertEqual((kwargs['duration'],kwargs['steps'],kwargs['seed']),(2,30,7))
            mx.eval(model._text_ids(kwargs['text'],kwargs['lyrics']))
            yield SimpleNamespace(audio=mx.array([[0.,.25],[-.5,.75]]),sample_rate=44100)
        model.generate = generate
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tokenizer.json').write_text('{}')
            encode = Mock(return_value=mx.array([[1,2],[1,2]]))
            modules = {
                'mlx_audio.music':SimpleNamespace(load=lambda path:model),
                'mlx_audio.music.models.minimax_music3.depth':SimpleNamespace(RVQDepthDecoder=self.decoder_type),
                'mlx_audio.music.models.minimax_music3.minimax_music3':SimpleNamespace(_encode_official_text_pair=encode),
                'mlx_audio.music.models.minimax_music3.prompt':SimpleNamespace(assemble_prompt=lambda c,l:c+'|'+l),
            }
            with patch.object(music_runner,'load_model',return_value=model), patch.dict(sys.modules,modules):
                for index in range(2):
                    out = root/f'{index}.wav'
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        self.assertEqual(music_runner.main(['--model',str(root),'--caption','piano',
                            '--lyrics','[instrumental]','--out',str(out),'--duration','2',
                            '--steps','30','--seed','7']),0)
                    self.assertEqual(json.loads(buffer.getvalue())['sample_rate'],44100)
                    with wave.open(str(out),'rb') as audio:
                        self.assertEqual((audio.getnchannels(),audio.getsampwidth(),audio.getnframes()),(2,2,2))
                        np.testing.assert_array_equal(np.frombuffer(audio.readframes(2),dtype='<i2'),
                                                      [0,8191,-16383,24575])
                self.assertEqual(encode.call_count,1)
