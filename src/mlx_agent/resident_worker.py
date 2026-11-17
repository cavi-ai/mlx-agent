"""Private, serial runner protocol inside the selected backend environment."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import sys
import signal
import threading

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_agent import runner_model_cache, worker_telemetry
from mlx_agent.allocator_cache import AllocatorCache, CacheUnavailable

RUNNERS = frozenset(("decide_runner", "music_runner", "transcribe_runner", "speak_runner",
                     "generate_runner", "describe_runner", "video_runner", "mesh_runner", "embed_runner", "segment_runner"))
MAX_MESSAGE = 16 * 1024 * 1024


def apply_memory_limit(runner, limit):
    if type(limit) is not int or not 0 < limit <= 2**60:
        raise ValueError('Invalid worker memory limit.')
    if runner == 'segment_runner':
        torch = importlib.import_module('torch')
        if not torch.backends.mps.is_available():
            raise ValueError('The MPS device is unavailable.')
        recommended = torch.mps.recommended_max_memory()
        fraction = limit / recommended
        if not 0 < fraction <= 2:
            raise ValueError('Worker memory limit is outside the supported MPS range.')
        torch.mps.set_per_process_memory_fraction(fraction)
    else:
        importlib.import_module('mlx.core').set_memory_limit(limit)


class BoundedText(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > MAX_MESSAGE // 4:
            raise ValueError("Runner output exceeded its limit.")
        return super().write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-fd", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    parent = os.getppid()

    def watch_parent():
        while os.getppid() == parent:
            threading.Event().wait(.5)
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch_parent, daemon=True).start()
    runner_model_cache._enabled = True
    cache = AllocatorCache()
    runner_model_cache._allocator_cache = cache
    framework = 'mlx'
    with os.fdopen(args.response_fd, "wb", buffering=0) as response:
        response.write(json.dumps({"ready": os.getpid()}).encode() + b"\n")
        while True:
            raw = sys.stdin.buffer.readline(MAX_MESSAGE + 1)
            if not raw:
                return 0
            if len(raw) > MAX_MESSAGE or not raw.endswith(b"\n"):
                return 2
            output, errors = BoundedText(), BoundedText()
            code = 1
            try:
                request = json.loads(raw)
                if request.get('control') == 'cache':
                    if set(request) != {'control', 'policy', 'reclaim'} or request['reclaim'] not in (None, 'manual', 'pressure', 'policy'):
                        raise ValueError('Invalid allocator cache control.')
                    if not runner_model_cache.loaded():
                        raise ValueError('Cache controls require a loaded worker.')
                    cache.configure(request['policy'])
                    cache.apply_limit(enforce=request['reclaim'] is None)
                    if request['reclaim']:
                        cache.reclaim(request['reclaim'])
                    code = 0
                else:
                    if (not {'runner', 'arguments'} <= set(request)
                            or set(request) - {'runner', 'arguments', 'cache_policy', 'memory_limit_bytes'}
                            or request['runner'] not in RUNNERS):
                        raise ValueError("Unknown inference runner.")
                    limit = request.get('memory_limit_bytes')
                    framework = 'torch' if request['runner'] == 'segment_runner' else 'mlx'
                    if limit is not None:
                        apply_memory_limit(request['runner'], limit)
                    cache.configure(request.get('cache_policy', {}))
                    with redirect_stdout(output), redirect_stderr(errors):
                        module = importlib.import_module("mlx_agent." + request["runner"])
                        code = module.main(request["arguments"])
                    if code == 0:
                        try:
                            idle = cache.desired['cache_reclaim'] == 'idle'
                            cache.apply_limit(enforce=not idle)
                            if idle:
                                cache.reclaim('idle')
                        except CacheUnavailable:
                            pass  # Preserve the successful result and expose maintenance failure.
            except (Exception, SystemExit) as error:
                errors.write(str(error)[-2000:])
            message = json.dumps({"returncode": code, "stdout": output.getvalue(), "stderr": errors.getvalue(),
                                  "loaded": runner_model_cache.loaded(),
                                  "memory": worker_telemetry.capture(runner_model_cache.statistics(), framework), "cache": cache.snapshot()}, allow_nan=False).encode() + b"\n"
            if len(message) > MAX_MESSAGE:
                return 2
            response.write(message)


if __name__ == "__main__":
    raise SystemExit(main())
