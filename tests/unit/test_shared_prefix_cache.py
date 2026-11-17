import unittest

from mlx_agent.shared_prefix_cache import eligible_prefix, PrefixCache, ScopedPrompt, SystemSegment


class Cache:
    def __init__(self, offset, nbytes=8):
        self.offset, self.nbytes = offset, nbytes


class Native:
    def __init__(self):
        self.entries = {}
        self.nbytes = 0

    def __len__(self):
        return len(self.entries)

    def insert_cache(self, model, tokens, state, **kwargs):
        self.entries[tuple(tokens)] = state
        self.nbytes = sum(sum(c.nbytes for c in s) for s in self.entries.values())

    def fetch_nearest_cache(self, model, tokens):
        for key, state in self.entries.items():
            if tokens[:len(key)] == list(key):
                import copy
                return copy.deepcopy(state), tokens[len(key):]
        return None, tokens

    def trim_to(self, **kwargs):
        if kwargs.get('n_bytes') == 0:
            self.entries.clear()
            self.nbytes = 0


class PrefixCacheTests(unittest.TestCase):
 def test_eligible_prefix_only_exact_pre_user_segment(self):
    assert eligible_prefix([1, 2, 3, 4], [[1, 2], [3, 4]], ['system', 'user'], [1, 2, 9],
                           [{'role': 'system'}, {'role': 'user'}]) == (1, 2)
    assert eligible_prefix([1, 2, 3], [[1, 2], [3]], ['system', 'user'], [1, 9],
                           [{'role': 'system'}, {'role': 'user'}]) is None
    assert eligible_prefix([1, 2, 3], [[1, 2], [3]], ['system', 'user'], [1, 2, 9],
                           [{'role': 'user'}]) is None


 def test_cache_only_admits_authorized_prefix_and_isolates_returned_state(self):
    native = Native()
    cache = PrefixCache(native, Cache, 16)
    cache.insert_cache('model', [1, 2, 3], [Cache(3)], cache_type='assistant')
    cache.insert_cache('model', [1, 2], [Cache(2)], cache_type=SystemSegment((1, 2)))
    result, rest = cache.fetch_nearest_cache('model', ScopedPrompt([1, 2, 9], (1, 2)))
    assert rest == [9]
    result[0].offset = 20
    assert native.entries[(1, 2)][0].offset == 2
    assert cache.fetch_nearest_cache('model', [5, 2, 9]) == (None, [5, 2, 9])
    cache.insert_cache('model', [1, 2], [Cache(3)], cache_type=SystemSegment((1, 2)))
    assert native.entries[(1, 2)][0].offset == 2

class PolicyTests(unittest.TestCase):
 def test_scope_requires_key_and_valid_budget(self):
    from mlx_agent.serve import plan_start, load_recipes, ServeError
    for opts in ({'shared_prefix_cache_mib': 15}, {'shared_prefix_cache_mib': 16},
                 {'shared_prefix_cache_mib': 2049, 'jit': True, 'api_key_file': '/missing'}):
        with self.assertRaises(ServeError):
            plan_start('pub/model', 'mlx_lm', load_recipes(), **opts)

 def test_extra_growth_is_reserved(self):
    from mlx_agent.memory_policy import reservation, validate
    config = {'runtime': 'mlx_lm', 'local_path': '/unused'}
    base = reservation(validate(None), config, weight_bytes=1000000)
    config['shared_prefix_cache'] = {'mib': 16}
    extended = reservation(validate(None), config, weight_bytes=1000000)
    self.assertEqual(extended['size'] - base['size'], 16 << 20)
    self.assertEqual(extended['growth_bytes'] - base['growth_bytes'], 16 << 20)

 def test_version_and_source_mismatch_rejected(self):
    from mlx_agent.shared_prefix_cache import supported, VERSIONS, UPSTREAM
    good = {'versions': dict(VERSIONS), 'sources': dict(UPSTREAM)}
    self.assertTrue(supported(good))
    good['sources']['mlx_lm/server.py'] = '0' * 64
    self.assertFalse(supported(good))

class TransportTests(unittest.TestCase):
 def test_pinned_runtime_binds_inherited_listener(self):
    import os
    import socket
    import sys
    import tempfile
    import types
    from unittest import mock
    from mlx_agent.private_serve import run_runtime
    from mlx_agent.shared_prefix_cache import VERSIONS, UPSTREAM
    calls = []
    server = types.ModuleType('mlx_lm.server')
    class Handler: pass
    server.APIHandler = Handler
    def http(host, port, generator, server_class=None, handler_class=None):
        instance = server_class((host, port), handler_class)
        calls.append(instance.socket.family)
        instance.server_close()
    def run(host, port, provider, server_class=None, handler_class=None):
        server._run_http_server(host, port, provider)
    server._run_http_server, server.run = http, run
    server.main = lambda: server.run('127.0.0.1', 0, None)
    package = types.ModuleType('mlx_lm')
    package.server = server
    facts = {'versions': VERSIONS, 'sources': UPSTREAM, 'fingerprint': 'ok'}
    with tempfile.TemporaryDirectory() as root:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(sock.close)
        sock.bind(root + '/sock')
        sock.listen()
        with mock.patch.dict(sys.modules, {'mlx_lm': package, 'mlx_lm.server': server}), \
             mock.patch('mlx_agent.shared_prefix_cache.identity_here', return_value=facts), \
             mock.patch('mlx_agent.shared_prefix_cache.install', return_value=lambda: None):
            run_runtime(root, os.dup(sock.fileno()), [], 16, 'ok')
    self.assertEqual(calls, [socket.AF_UNIX])
    self.assertIs(server._run_http_server, http)

 def test_prefix_requires_compatible_http_factory_and_restores_install(self):
    import os
    import socket
    import sys
    import tempfile
    import types
    from unittest import mock
    from mlx_agent.private_serve import run_runtime
    from mlx_agent.shared_prefix_cache import VERSIONS, UPSTREAM
    facts = {'versions': VERSIONS, 'sources': UPSTREAM, 'fingerprint': 'ok'}
    for factory in (None, lambda host, port, provider: None):
        restored = []
        server = types.ModuleType('mlx_lm.server')
        server.APIHandler = type('Handler', (), {})
        server.run = lambda host, port, provider, server_class=None, handler_class=None: None
        server.main = lambda: self.fail('unsupported factory reached runtime main')
        if factory is not None:
            server._run_http_server = factory
        package = types.ModuleType('mlx_lm')
        package.server = server
        with tempfile.TemporaryDirectory() as root:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.bind(root + '/sock')
                sock.listen()
                with mock.patch.dict(sys.modules, {'mlx_lm': package, 'mlx_lm.server': server}), \
                     mock.patch('mlx_agent.shared_prefix_cache.identity_here', return_value=facts), \
                     mock.patch('mlx_agent.shared_prefix_cache.install', return_value=lambda: restored.append(True)):
                    with self.assertRaises(ValueError):
                        run_runtime(root, os.dup(sock.fileno()), [], 16, 'ok')
        self.assertEqual(restored, [True])
