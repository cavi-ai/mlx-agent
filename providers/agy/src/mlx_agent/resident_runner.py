"""Bounded, cancellable calls to one isolated backend retaining one local model."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
import threading

from .resident_worker import MAX_MESSAGE, RUNNERS
from .compute import process_identity
from .task_runtime import _stop


class ResidentRunner:
    def __init__(self, executable, model, environment=None, log_path=None, port=0):
        self.executable, self.model = str(executable), str(model)
        self.environment, self.log_path, self.port = environment, log_path, port
        self.process, self.response, self.log = None, None, None
        self.loaded = False
        self.telemetry = None
        self.cache_state = None
        self.identity = None
        self.lock = threading.RLock()

    def _receive(self, argv, timeout, cancelled):
        deadline = time.monotonic() + timeout
        chunks = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self.response, selectors.EVENT_READ)
            while True:
                if cancelled():
                    raise InterruptedError("Inference cancelled.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                if not selector.select(min(.1, remaining)):
                    continue
                chunk = os.read(self.response, 65536)
                if not chunk:
                    raise OSError("Resident inference worker exited without a result.")
                chunks.extend(chunk)
                if len(chunks) > MAX_MESSAGE:
                    raise ValueError("Runner response exceeded its limit.")
                if chunks.endswith(b"\n"):
                    return json.loads(chunks)

    def _start(self, timeout=15, cancelled=lambda: False):
        read_fd, write_fd = os.pipe()
        self.response = read_fd
        try:
            if self.log_path:
                descriptor = os.open(self.log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
                self.log = os.fdopen(descriptor, "ab", buffering=0)
            self.process = subprocess.Popen([self.executable, str(Path(__file__).with_name("resident_worker.py")),
                "--response-fd", str(write_fd), "--model", self.model, "--port", str(self.port)], stdin=subprocess.PIPE,
                stdout=self.log or subprocess.DEVNULL, stderr=subprocess.STDOUT,
                env=self.environment, pass_fds=(write_fd,), start_new_session=True, umask=0o077)
            os.close(write_fd)
            write_fd = None
            ready = self._receive(self.process.args, timeout, cancelled)
            if ready != {"ready": self.process.pid}:
                raise ValueError("The resident worker readiness identity did not match its launcher.")
            # Framework Python launchers can exec after Popen returns. Bind only
            # once the actual worker acknowledges that it reached its request loop.
            self.identity = process_identity(self.process.pid)
            if not self.identity:
                raise OSError("The ready inference worker identity is unavailable.")
        except BaseException:
            if self.process is not None:
                # No request has been sent: only this unreaped direct child can
                # exist, and it has not imported a model or spawned inference.
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            self._clear()
            raise
        finally:
            if write_fd is not None:
                os.close(write_fd)

    def run(self, argv, timeout=3600, cancelled=lambda: False, cache_policy=None, memory_limit_bytes=None, **options):
        with self.lock:
            return self._run(argv, timeout, cancelled, cache_policy, memory_limit_bytes, **options)

    def _run(self, argv, timeout, cancelled, cache_policy, memory_limit_bytes=None, **options):
        script = Path(argv[1]) if len(argv) > 1 else Path("")
        if (argv[0] != self.executable or script.parent.resolve() != Path(__file__).parent.resolve()
                or script.stem not in RUNNERS or script.suffix != ".py"
                or "--model" not in argv or argv[argv.index("--model") + 1] != self.model):
            raise ValueError("Resident requests must use the selected local model and an engine inference runner.")
        if self.process is not None and self.process.poll() is not None:
            self.close()
        started = time.monotonic()
        try:
            if cancelled():
                raise InterruptedError("Inference cancelled.")
            if self.process is None:
                self._start(timeout=min(timeout, 15), cancelled=cancelled)
            message = {"runner": script.stem, "arguments": argv[2:]}
            if cache_policy is not None:
                message['cache_policy'] = cache_policy
            if memory_limit_bytes is not None:
                if type(memory_limit_bytes) is not int or not 0 < memory_limit_bytes <= 2**60:
                    raise ValueError('Invalid worker memory limit.')
                message['memory_limit_bytes'] = memory_limit_bytes
            raw = json.dumps(message).encode() + b"\n"
            if len(raw) > MAX_MESSAGE:
                raise ValueError("Runner request exceeded its limit.")
            self.process.stdin.write(raw)
            self.process.stdin.flush()
            remaining = timeout - (time.monotonic() - started)
            result = self._receive(argv, max(0, remaining), cancelled)
            from .worker_telemetry import validate
            self.telemetry = validate(result['memory'])
            from .allocator_cache import validate_state
            self.cache_state = validate_state(result['cache']) if 'cache' in result else None
            self.loaded = result["loaded"] is True
            return subprocess.CompletedProcess(argv, result["returncode"], result["stdout"], result["stderr"])
        except BaseException:
            self.close()
            raise

    def control_cache(self, policy, reclaim=None):
        with self.lock:
            if not self.loaded or self.process is None:
                raise ValueError('Cache controls require a loaded worker.')
            message = {'control': 'cache', 'policy': policy, 'reclaim': reclaim}
            try:
                self.process.stdin.write(json.dumps(message).encode() + b'\n')
                self.process.stdin.flush()
                result = self._receive(self.process.args, 10, lambda: False)
                from .worker_telemetry import validate
                from .allocator_cache import validate_state
                self.telemetry = validate(result['memory'])
                self.cache_state = validate_state(result['cache'])
                if result['loaded'] is not True:
                    raise ValueError('Cache control changed model residency.')
                return result['returncode'] == 0
            except BaseException:
                self.close()
                raise

    def close(self):
        with self.lock:
            self._close()

    def _close(self):
        process = self.process
        if process is not None:
            known = {process.pid: self.identity} if self.identity else {}
            if not _stop(process, self.identity, known, grace=3):
                raise OSError("The owned inference process group has not stopped safely.")
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired as error:
                raise OSError("The owned inference worker exit has not been confirmed.") from error
        self._clear()

    def _clear(self):
        if self.process is not None and self.process.stdin:
            self.process.stdin.close()
        self.process = None
        self.identity = None
        if self.response is not None:
            os.close(self.response)
            self.response = None
        if self.log:
            self.log.close()
            self.log = None
        self.loaded = False
        self.telemetry = None
        self.cache_state = None
