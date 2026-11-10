"""Allocator controls preserve active tensors and legacy policy defaults."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlx_agent.jit_serve import validate_memory_policy


class AllocatorCacheTests(unittest.TestCase):
    def core(self):
        state = {'cache': 12 * 1024**2, 'limit': 128 * 1024**2}
        def limit(value):
            previous = state['limit']; state['limit'] = value; return previous
        core = SimpleNamespace(set_cache_limit=Mock(side_effect=limit), synchronize=Mock(),
            clear_cache=Mock(side_effect=lambda: state.update(cache=0)),
            get_cache_memory=lambda: state['cache'], get_active_memory=lambda: 2 * 1024**2)
        return core, state

    def test_cap_reset_and_manual_reclaim_preserve_active_weights(self):
        from mlx_agent.allocator_cache import AllocatorCache
        core, state = self.core()
        with patch.dict(sys.modules, {'mlx.core': core}):
            cache = AllocatorCache()
            cache.configure({'allocator_cache_limit_mib': 4, 'cache_reclaim': 'off'})
            cache.apply_limit()
            self.assertEqual(state['limit'], 4 * 1024**2)
            self.assertEqual(state['cache'], 0)
            state['cache'] = 3 * 1024**2
            cache.reclaim('manual')
            self.assertEqual(cache.snapshot()['last_reclaim']['released_cache_bytes'], 3 * 1024**2)
            self.assertEqual(core.get_active_memory(), 2 * 1024**2)
            self.assertTrue(core.synchronize.called)
            cache.configure({'allocator_cache_limit_mib': None, 'cache_reclaim': 'off'})
            cache.apply_limit()
            self.assertEqual(state['limit'], 128 * 1024**2)

    def test_default_does_not_import_a_runtime(self):
        from mlx_agent.allocator_cache import AllocatorCache
        with patch.dict(sys.modules, {'mlx.core': None}), patch('importlib.import_module') as imports:
            cache = AllocatorCache(); cache.apply_limit()
            self.assertFalse(cache.snapshot()['available'])
            imports.assert_not_called()

    def test_policy_optional_limits_pressure_and_legacy_defaults(self):
        baseline = validate_memory_policy()
        self.assertNotIn('allocator_cache_limit_mib', baseline)
        for value in (-1, True, 1.5, 1048577):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_memory_policy({'allocator_cache_limit_mib': value})
        for value in (None, 0, 1024):
            self.assertEqual(validate_memory_policy({'allocator_cache_limit_mib': value})['allocator_cache_limit_mib'], value)
        with self.assertRaises(ValueError):validate_memory_policy({'cache_reclaim': 'pressure'})
        value = validate_memory_policy({'cache_reclaim': 'pressure', 'minimum_headroom_gb': 2, 'keep_loaded': True})
        self.assertEqual(value['cache_reclaim'], 'pressure')
        with self.assertRaises(ValueError):validate_memory_policy({'cache_reclaim': 'anything'})
