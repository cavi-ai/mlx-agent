"""Versioned optional allocator telemetry; never import an inference runtime."""
import math
import sys
import time


def validate(value):
    fields={'schema','active_bytes','cache_bytes','peak_bytes','loads','cache_hits','load_seconds','captured_at'}
    if not isinstance(value,dict) or set(value) not in (fields, fields | {'components'}) or value['schema']!='worker-memory/1':
        raise ValueError('Invalid worker memory telemetry.')
    for name in ('active_bytes','cache_bytes','peak_bytes','loads','cache_hits'):
        item=value[name]
        if item is None and name.endswith('_bytes'):continue
        if type(item) is not int or not 0<=item<=2**60:
            raise ValueError('Invalid worker allocation counter.')
    for name in ('load_seconds','captured_at'):
        seconds=value[name]
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
            raise ValueError('Invalid worker telemetry time.')
    if 'components' in value:
        from importlib import import_module
        names = import_module('mlx_agent.resources.ports.mlx-3d.trellis2.components').NAMES
        component = value['components']
        if not isinstance(component, dict) or set(component) != {'schema', 'resident', 'loads', 'load_seconds', 'checkpoint_bytes'} or component['schema'] != 'resident-components/1':
            raise ValueError('Invalid resident component telemetry.')
        resident = component['resident']
        if not isinstance(resident, list) or any(not isinstance(name, str) or name not in names for name in resident) or len(set(resident)) != len(resident):
            raise ValueError('Invalid resident components.')
        for field in ('loads', 'load_seconds', 'checkpoint_bytes'):
            values = component[field]
            if not isinstance(values, dict) or set(values) != set(resident):
                raise ValueError('Invalid resident component counters.')
            for item in values.values():
                if field == 'load_seconds':
                    if type(item) not in (int, float) or not math.isfinite(item) or item < 0:
                        raise ValueError('Invalid component load time.')
                elif type(item) is not int or not 0 <= item <= 2**60:
                    raise ValueError('Invalid component counter.')
    return value


def capture(statistics):
    core=sys.modules.get('mlx.core')
    result={'schema':'worker-memory/1','captured_at':time.time(),**statistics}
    for field,method in (('active_bytes','get_active_memory'),('cache_bytes','get_cache_memory'),('peak_bytes','get_peak_memory')):
        function=getattr(core,method,None)
        try:
            result[field]=function() if callable(function) else None
        except (RuntimeError,NotImplementedError):
            result[field]=None
    return validate(result)
