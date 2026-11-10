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

RUNNERS = frozenset(("decide_runner", "music_runner", "transcribe_runner", "speak_runner",
                     "generate_runner", "describe_runner", "video_runner", "mesh_runner", "embed_runner"))
MAX_MESSAGE = 16 * 1024 * 1024


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
                if set(request) != {"runner", "arguments"} or request["runner"] not in RUNNERS:
                    raise ValueError("Unknown inference runner.")
                with redirect_stdout(output), redirect_stderr(errors):
                    module = importlib.import_module("mlx_agent." + request["runner"])
                    code = module.main(request["arguments"])
            except (Exception, SystemExit) as error:
                errors.write(str(error)[-2000:])
            message = json.dumps({"returncode": code, "stdout": output.getvalue(), "stderr": errors.getvalue(),
                                  "loaded": runner_model_cache.loaded(),
                                  "memory": worker_telemetry.capture(runner_model_cache.statistics())}, allow_nan=False).encode() + b"\n"
            if len(message) > MAX_MESSAGE:
                return 2
            response.write(message)


if __name__ == "__main__":
    raise SystemExit(main())
