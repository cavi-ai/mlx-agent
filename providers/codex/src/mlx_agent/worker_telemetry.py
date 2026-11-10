"""Versioned optional allocator telemetry; never import an inference runtime."""
import math
import sys
import time


def validate(value):
    fields={'schema','active_bytes','cache_bytes','peak_bytes','loads','cache_hits','load_seconds','captured_at'}
    if not isinstance(value,dict) or set(value)!=fields or value['schema']!='worker-memory/1':
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
