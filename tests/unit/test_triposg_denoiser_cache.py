"""Cache control and native U-shaped-transformer regression tests."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

PORT = Path(__file__).resolve().parents[2] / 'src/mlx_agent/resources/ports/mlx-3d'
sys.path.insert(0, str(PORT))


class TripoSGCachePolicyTests(unittest.TestCase):
    def cache(self, **kwargs):
        try:
            from triposg.denoiser_cache import DenoiserCache
        except ImportError:
            self.fail('TripoSG adaptive denoiser cache is unavailable')
        return DenoiserCache(steps=12, **kwargs)

    def test_warmup_streak_and_final_steps_force_full_computation(self):
        cache = self.cache()
        decisions = [cache.decide(0.0, ('image',), 100) for _ in range(12)]
        self.assertEqual(decisions, [False,False,False,True,False,True,False,True,False,True,False,False])
        self.assertEqual(cache.stats()['reused_steps'],4)

    def test_threshold_budget_nonfinite_and_context_changes(self):
        cache = self.cache(memory_limit_bytes=100)
        for _ in range(3): self.assertFalse(cache.decide(0, ('image',), 100))
        self.assertFalse(cache.decide(.1, ('image',), 100))
        self.assertFalse(cache.decide(0, ('image',), 101))
        for error in (math.nan,math.inf,-1):
            self.assertFalse(cache.decide(error, ('image',), 100))
        for _ in range(3): self.assertFalse(cache.decide(0, ('other-image',), 100))
        self.assertEqual(cache.stats()['reused_steps'],0)
        json.dumps(cache.stats(),allow_nan=False)

    def test_short_jobs_zero_threshold_and_invalid_settings(self):
        cache = self.cache(threshold=0)
        self.assertFalse(any(cache.decide(0, ('image',), 100) for _ in range(12)))
        for options in ({'threshold':math.nan},{'threshold':-1},{'memory_limit_bytes':0}):
            with self.subTest(options=options),self.assertRaises(ValueError): self.cache(**options)


HAS_MLX = importlib.util.find_spec('mlx') is not None


@unittest.skipUnless(HAS_MLX, 'native MLX requires Apple Silicon')
class TripoSGCacheNativeTests(unittest.TestCase):
    def test_pipeline_closes_failed_sessions_and_over_budget_jobs_bypass_cache(self):
        import mlx.core as mx
        from unittest.mock import patch, Mock
        from triposg import pipeline
        from triposg.denoiser_cache import DenoiserCache
        sessions=[]

        def create(*args):
            session=DenoiserCache(*args)
            session.close=Mock(wraps=session.close)
            sessions.append(session)
            return session

        config={'components':{'transformer':{'width':2048,'in_channels':4},'image_encoder_dinov2':{}},
                'preprocessor':{},'scheduler':{'_class_name':'RectifiedFlowScheduler'}}
        model=config,{'transformer':object(),'image_encoder_dinov2':object()}
        with patch.object(pipeline,'preprocess'),patch.object(pipeline,'encode',return_value=mx.zeros((1,2,4))), \
             patch.object(pipeline,'initial_latents',return_value=mx.zeros((1,2,4))), \
             patch.object(pipeline,'denoise',side_effect=RuntimeError('synthetic sampling failure')) as denoise, \
             patch('triposg.denoiser_cache.DenoiserCache',side_effect=create):
            for budget in (256,8):
                with self.assertRaisesRegex(RuntimeError,'synthetic sampling failure'):
                    pipeline.generate(model,'unused.png','unused.glb',denoiser_cache='adaptive',cache_memory_mib=budget)
            self.assertEqual(len(sessions),1)
            sessions[0].close.assert_called_once()
            self.assertFalse(sessions[0].anchors)
            self.assertIsNone(sessions[0].first_output)
            self.assertNotIn('denoiser_cache',denoise.call_args.kwargs)

    def test_reuse_skips_blocks_but_always_evaluates_the_output_head(self):
        import mlx.core as mx
        import numpy as np
        from unittest.mock import patch
        from collections import Counter
        from triposg import transformer
        from triposg.denoiser_cache import DenoiserCache
        from triposg.weights import Weights
        root=Path(__file__).parents[1]/'fixtures/triposg'
        config=json.loads((root/'parity-config.json').read_text())['transformer']
        calls=Counter()

        class RecordingWeights(Weights):
            def linear(self,name,x):
                calls[name]+=1
                return super().linear(name,x)

        with np.load(root/'transformer-weights.npz') as values:
            weights=RecordingWeights({k:mx.array(v) for k,v in values.items()})
        cache=DenoiserCache(12,threshold=100)
        try:
            with np.load(root/'parity.npz') as fixture,patch.object(transformer,'block',wraps=transformer.block) as block:
                features,latents=mx.array(fixture['features']),mx.array(fixture['latents'])
                for step in range(12):
                    output=transformer.denoise(weights,config,latents,[1000-step*20]*latents.shape[0],features,denoiser_cache=cache)
                    mx.eval(output)
                self.assertEqual(cache.stats()['reused_steps'],4)
                self.assertEqual(block.call_count,8*config['num_layers']+4)
                self.assertEqual(calls['proj_out'],12)
        finally:cache.close()

    def test_full_path_matches_reference_and_keeps_weights_unchanged(self):
        import mlx.core as mx
        import numpy as np
        from triposg.denoiser_cache import DenoiserCache
        from triposg.transformer import denoise, _CrossAttentionCache
        from triposg.weights import Weights
        root = Path(__file__).parents[1] / 'fixtures/triposg'
        config = json.loads((root/'parity-config.json').read_text())['transformer']
        with np.load(root/'transformer-weights.npz') as values:
            weights = Weights({k:mx.array(v) for k,v in values.items()})
        keys = set(weights.values)
        with np.load(root/'parity.npz') as fixture:
            features,latents = mx.array(fixture['features']),mx.array(fixture['latents'])
            times = fixture['times'].tolist()
            reference = np.array(denoise(weights,config,latents,times,features))
            cache = DenoiserCache(steps=12,threshold=0)
            try:
                actual = np.array(denoise(weights,config,latents,times,features,
                    context_cache=_CrossAttentionCache(weights,features),denoiser_cache=cache))
                np.testing.assert_array_equal(actual,reference)
                self.assertEqual(set(weights.values),keys)
            finally: cache.close()
        self.assertFalse(cache.anchors)
        self.assertIsNone(cache.first_output)

    def test_first_order_forecast_and_indicator_refresh(self):
        import mlx.core as mx
        import numpy as np
        from triposg.denoiser_cache import DenoiserCache
        cache = DenoiserCache(steps=12)
        weights,features = object(),mx.ones((1,2,4))
        before = mx.ones((1,2,4))
        for t in (12,10,7,6):
            first = before + t
            reuse = cache.begin(weights,features,before,first,t)
            output = cache.predict(first) if reuse else first+5*t
            if not reuse: cache.record(output)
            np.testing.assert_array_equal(np.array(output),np.ones((1,2,4))+6*t)
        self.assertTrue(cache.reuse)
        cache.begin(weights,features,before,before+5,5);cache.record(before+30)
        self.assertFalse(cache.begin(weights,features,before,before+104,4))
        cache.record(before+624)
        self.assertGreater(cache.stats()['refresh_reasons'].get('threshold',0),0)
        self.assertFalse(cache.begin(weights,mx.zeros_like(features),before,before+3,3))
        cache.close()
        self.assertFalse(cache.anchors)
