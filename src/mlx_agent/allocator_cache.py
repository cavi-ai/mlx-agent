"""Optional free-buffer controls inside one admitted resident MLX worker."""
import importlib
import sys
import time

FIELDS = frozenset(('allocator_cache_limit_mib', 'cache_reclaim'))
MODES = ('off', 'idle', 'pressure')


class CacheUnavailable(ValueError):
    pass


def policy(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - FIELDS:
        raise ValueError('Unknown allocator cache policy fields.')
    limit, mode = value.get('allocator_cache_limit_mib'), value.get('cache_reclaim', 'off')
    if limit is not None and (type(limit) is not int or not 0 <= limit <= 1048576):
        raise ValueError('Allocator cache limit must be null or an integer from 0 to 1048576 MiB.')
    if mode not in MODES:
        raise ValueError('Cache reclaim must be off, idle or pressure.')
    return {'allocator_cache_limit_mib': limit, 'cache_reclaim': mode}


def from_memory(value):
    return policy({key: value[key] for key in FIELDS if key in value})


def requested(value):
    settings = from_memory(value)
    return settings['allocator_cache_limit_mib'] is not None or settings['cache_reclaim'] != 'off'


def validate_state(value):
    if not isinstance(value, dict) or set(value) != {'available', 'applied_policy', 'last_reclaim', 'error'}:
        raise ValueError('Invalid allocator cache state.')
    if type(value['available']) is not bool or value['error'] is not None and not isinstance(value['error'], str):
        raise ValueError('Invalid allocator cache availability.')
    if value['applied_policy'] is not None:
        policy(value['applied_policy'])
    last = value['last_reclaim']
    if last is not None:
        if not isinstance(last, dict) or set(last) != {'reason', 'before_cache_bytes', 'after_cache_bytes', 'released_cache_bytes', 'captured_at'}:
            raise ValueError('Invalid allocator reclaim observation.')
        if last['reason'] not in ('manual', 'idle', 'pressure', 'policy', 'limit'):
            raise ValueError('Invalid allocator reclaim reason.')
        if any(type(last[key]) is not int or last[key] < 0 for key in ('before_cache_bytes', 'after_cache_bytes', 'released_cache_bytes')):
            raise ValueError('Invalid allocator reclaim size.')
        if type(last['captured_at']) not in (int, float) or not 0 <= last['captured_at'] < float('inf'):
            raise ValueError('Invalid allocator reclaim time.')
    return value


class AllocatorCache:
    def __init__(self):
        self.desired = policy()
        self.applied = None
        self.default_limit = None
        self.current_limit = None
        self.last_reclaim = None
        self.error = None

    @staticmethod
    def _core(required=False):
        core = sys.modules.get('mlx.core')
        if core is None and required:
            try:
                core = importlib.import_module('mlx.core')
            except (ImportError, RuntimeError) as error:
                raise CacheUnavailable('The worker MLX allocator is unavailable.') from error
        if core is None or not all(callable(getattr(core, name, None)) for name in
                ('set_cache_limit', 'get_cache_memory', 'clear_cache', 'synchronize')):
            if required:
                raise CacheUnavailable('This backend does not expose MLX allocator cache controls.')
            return None
        return core

    def configure(self, value):
        self.desired = policy(value)

    def apply_limit(self, enforce=True):
        try:
            limit = self.desired['allocator_cache_limit_mib']
            core = self._core(requested(self.desired) or self.default_limit is not None)
            if core is not None:
                if limit is not None:
                    previous = core.set_cache_limit(limit * 1024**2)
                    if self.default_limit is None:
                        self.default_limit = previous
                    self.current_limit = limit
                    if enforce and core.get_cache_memory() > limit * 1024**2:
                        self.reclaim('limit')
                elif self.current_limit is not None:
                    core.set_cache_limit(self.default_limit)
                    self.current_limit = None
            self.applied = dict(self.desired)
            self.error = None
        except (ValueError, RuntimeError, NotImplementedError) as error:
            self.error = str(error)
            raise CacheUnavailable(str(error)) from error

    def reclaim(self, reason):
        try:
            core = self._core(True)
            core.synchronize()
            before = core.get_cache_memory()
            core.clear_cache()
            after = core.get_cache_memory()
            self.last_reclaim = {'reason': reason, 'before_cache_bytes': before, 'after_cache_bytes': after,
                'released_cache_bytes': max(0, before - after), 'captured_at': time.time()}
            self.error = None
        except (ValueError, RuntimeError, NotImplementedError) as error:
            self.error = str(error)
            raise CacheUnavailable(str(error)) from error

    def snapshot(self):
        return validate_state({'available': self._core() is not None, 'applied_policy': self.applied,
                               'last_reclaim': self.last_reclaim, 'error': self.error})
