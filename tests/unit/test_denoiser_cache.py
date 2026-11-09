"""Portable control-policy tests for bounded, generation-local denoiser reuse."""
import math
import unittest
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('denoiser_cache_policy', Path(__file__).resolve().parents[2] /
    'src/mlx_agent/resources/ports/mlx-video/t2v/denoiser_cache.py')
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class DenoiserCachePolicyTests(unittest.TestCase):
    def policy(self, **options):
        return CACHE.DenoiserCachePolicy(steps=12, **options)

    def test_warmup_final_refreshes_and_streak_limit(self):
        policy = self.policy()
        decisions = [policy.decide(0.0, ('image',), 100) for _ in range(12)]
        self.assertEqual(decisions, [False,False,False,True,True,True,False,True,True,True,False,False])
        self.assertEqual(policy.stats()['reused_steps'],6)
        self.assertEqual(policy.stats()['full_steps'],6)

    def test_small_changes_accumulate_until_a_full_refresh(self):
        policy = self.policy(threshold=.03)
        for _ in range(3):self.assertFalse(policy.decide(0.0, ('image',), 100))
        self.assertTrue(policy.decide(.02, ('image',), 100))
        self.assertFalse(policy.decide(.02, ('image',), 100))
        self.assertTrue(policy.decide(.01, ('image',), 100))

    def test_context_changes_restart_warmup_and_never_reuse_another_shape(self):
        policy = self.policy()
        for _ in range(3):policy.decide(0.0, ('image',), 100)
        self.assertTrue(policy.decide(0.0, ('image',), 100))
        for _ in range(3):self.assertFalse(policy.decide(0.0, ('other-image',), 100))
        self.assertTrue(policy.decide(0.0, ('other-image',), 100))

    def test_over_budget_and_nonfinite_indicators_force_full_computation(self):
        policy = self.policy(memory_limit_bytes=100)
        for _ in range(5):self.assertFalse(policy.decide(0.0, ('image',), 101))
        self.assertEqual(policy.stats()['peak_cache_bytes'],0)
        for bad in (math.nan, math.inf, -1):
            self.assertFalse(policy.decide(bad, ('image',), 100))
        self.assertEqual(policy.stats()['reused_steps'],0)

    def test_zero_threshold_is_a_full_compute_reference_and_short_jobs_never_skip(self):
        policy = self.policy(threshold=0)
        self.assertFalse(any(policy.decide(0.0, ('image',), 100) for _ in range(12)))
        policy = CACHE.DenoiserCachePolicy(steps=4)
        self.assertFalse(any(policy.decide(0.0, ('image',), 100) for _ in range(4)))

    def test_parameters_are_validated_and_statistics_are_json_safe(self):
        for options in ({'threshold':float('nan')},{'threshold':-1},{'memory_limit_bytes':0},
                        {'max_reuses':0},{'warmup':0},{'final_steps':0}):
            with self.subTest(options=options),self.assertRaises(ValueError):self.policy(**options)
        import json
        json.dumps(self.policy().stats(),allow_nan=False)

    def test_feature_reservation_scales_with_shape(self):
        config = {'dim':1536, 'patch_size':[1,2,2], 'vae_stride':[4,8,8]}
        self.assertEqual(CACHE.required_cache_bytes(config,320,192,33),159252480)


try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except ImportError:
    mx = None


@unittest.skipIf(mx is None, 'native MLX is available on Apple Silicon')
class NativeDenoiserCacheTests(unittest.TestCase):
    def test_reuse_context_invalidation_and_parameter_lifetime(self):
        class Block(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.weight = mx.array(float(index + 1))
                self.cross_attn = nn.Identity()

            def __call__(self, x, **kwargs):
                return x + self.weight * (kwargs['t'].reshape(-1)[0] + kwargs.get('bend', 0))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = [Block(i) for i in range(3)]

            def __call__(self, x, **kwargs):
                for block in self.blocks:
                    x = block(x, **kwargs)
                return x

        model = Model()
        originals = model.blocks
        size = lambda: sum(a.size for _,a in tree_flatten(model.parameters()))
        before = size()
        context = mx.ones((2,2))
        kwargs = dict(context=context, seq_lens=[2,2], grid_sizes=[(1,1,2)] * 2)
        session = CACHE.WanDenoiserCache(12,1024)
        try:
            proxy = session.attach(model)
            for t in (12,10,7,6):
                kwargs['t'] = mx.array([t,t])
                result = proxy._compiled(mx.ones((2,2)), **kwargs)
                mx.eval(result)
                expected = 1 + 6*t
                self.assertEqual(result.tolist(), [[expected,expected],[expected,expected]])
            self.assertEqual(session.stats()['reused_steps'],1)
            kwargs['t'] = mx.array([5,5])
            proxy._compiled(mx.ones((2,2)), **kwargs)
            kwargs.update(t=mx.array([4,4]), bend=mx.array(100))
            result = proxy._compiled(mx.ones((2,2)), **kwargs)
            mx.eval(result)
            self.assertEqual(result.tolist(), [[625,625],[625,625]])
            self.assertFalse(session.reuse)
            self.assertGreater(session.stats()['refresh_reasons'].get('threshold', 0), 0)
            kwargs['context'] = mx.zeros((2,2))
            proxy._compiled(mx.ones((2,2)), **kwargs)
            self.assertFalse(session.reuse)
            self.assertEqual(size(), before)
        finally:
            session.close()
        self.assertIs(model.blocks, originals)
        self.assertIsNone(session.residual)
        self.assertEqual(size(), before)
