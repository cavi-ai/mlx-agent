"""Loopback OpenAI gateway with one owned, lazily started local-model worker.

No runtime imports or downloads. Inference goes to the existing recipe's
server process; a request lease lasts through the final streamed byte.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import hmac
import http.client
import json
import math
import os
import re
import selectors
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlx_agent import chat_completions, responses_api
from mlx_agent.contextfit import CONTEXT_MAX, CONTEXT_MIN, extract_architecture, kv_cache_bytes
from mlx_agent.messages_api import (MessagesRequestError, StreamTranslator, WorkerResponseError, error_body,
                                    from_chat_response, to_chat_request)
from mlx_agent.transactions import _atomic_in_directory

MAX_BODY = 16 * 1024 * 1024
MAX_REQUESTS = 32
STORED_RESPONSES = 64


STRATA_CONTEXT = "Strata JIT admission already includes the KV cache at --strata-context-tokens."


def validate_memory_policy(policy=None):
    defaults = {"idle_timeout_seconds": 0, "keep_loaded": False, "minimum_headroom_gb": None}
    if policy is None:
        return defaults
    if not isinstance(policy, dict) or set(policy) - set(defaults) - {"context_tokens", "unload_under_pressure", "model_memory_budget_gb"}:
        raise ValueError("Unknown memory policy fields.")
    result = dict(defaults, **policy)
    # Null and absent both mean no KV term; policies without a context keep their existing form.
    context = result.pop("context_tokens", None)
    pressure = result.pop("unload_under_pressure", False)
    budget = result.pop("model_memory_budget_gb", None)
    if budget is not None:
        if type(budget) not in (int, float) or not math.isfinite(budget) or not 1 <= budget <= 1024:
            raise ValueError("Model memory budget must be a finite value from 1 to 1024 GB.")
        result["model_memory_budget_gb"] = budget
        if result["minimum_headroom_gb"] is None:
            result["minimum_headroom_gb"] = 2
    idle, keep, reserve = (result[key] for key in defaults)
    if type(idle) is not int or not 0 <= idle <= 86400:
        raise ValueError("Idle timeout must be an integer between 0 and 86400 seconds.")
    if type(keep) is not bool:
        raise ValueError("Keep loaded must be a boolean.")
    if reserve is not None and (type(reserve) not in (int, float) or not math.isfinite(reserve) or not 0 <= reserve <= 1024):
        raise ValueError("Headroom reserve must be a finite value between 0 and 1024 GB.")
    if type(pressure) is not bool:
        raise ValueError("Unload under pressure must be a boolean.")
    if pressure:
        if reserve is None or keep:
            raise ValueError("Unload under pressure requires a headroom reserve and cannot be combined with keep loaded.")
        result["unload_under_pressure"] = True
    if context is not None:
        if type(context) is not int or not CONTEXT_MIN <= context <= CONTEXT_MAX:
            raise ValueError("Context tokens must be an integer between {} and {}.".format(CONTEXT_MIN, CONTEXT_MAX))
        if reserve is None:
            raise ValueError("Context tokens require a headroom reserve.")
        result["context_tokens"] = context
    return result


def _kv_cache_bytes(local_path, context_tokens):
    """KV cache bytes at the context from the local config.json, or None when its architecture is unknown."""
    try:
        location = Path(local_path) / "config.json"
        if location.stat().st_size > 1024 * 1024:
            return None
        config = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    architecture = extract_architecture(config) or extract_architecture(config.get("text_config"))
    return kv_cache_bytes(architecture, context_tokens)


def available_memory_bytes():
    """Fresh OS headroom, or unknown. Runs on the worker-loading thread."""
    try:
        if sys.platform == "darwin":
            output = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=2, check=True).stdout
            page_size = int(re.search(r"page size of (\d+) bytes", output)[1])
            pages = [int(re.search(r"^Pages {}:\s+(\d+)".format(name), output, re.M)[1])
                     for name in ("free", "inactive")]
            return sum(pages) * page_size
        if sys.platform.startswith("linux"):
            return int(re.search(r"^MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M)[1]) * 1024
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        pass
    return None


class GatewayError(RuntimeError):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


def model_fingerprint(path):
    """File identity manifest, deliberately not a model-quality assessment."""
    root = Path(path)
    rows = []
    for item in sorted(root.rglob("*")):
        if any(part.startswith(".") for part in item.relative_to(root).parts):
            continue
        if item.is_file():
            stat = item.stat()
            rows.append([str(item.relative_to(root)), str(item.resolve()), stat.st_dev,
                         stat.st_ino, stat.st_size, stat.st_mtime_ns])
        if len(rows) > 20000:
            raise ValueError("Model directory has too many files.")
    if not root.is_dir() or not rows:
        raise ValueError("Local model files are unavailable.")
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


class ModelWorker:
    def __init__(self, config):
        self.config = config
        self.condition = threading.Condition()
        self.process = None
        self.worker_port = None
        self.worker_argv = None
        self.model_state = "unloaded"
        self.active_requests = 0
        self.last_error = None
        self.closed = False
        self.log = None
        self.memory_policy = validate_memory_policy(config.get("memory_policy"))
        self.memory_check = None
        self.resident_memory_check = None
        self.request_blocked_reason = None
        self.load_blocked_reason = None
        self.last_unload_reason = None
        self.last_activity = None
        self.activity_revision = 0
        self.policy_revision = 0
        self.memory_budget_pending = False
        self.strata_children = {}
        self.owned_session = None
        self.clock = time.monotonic
        self.memory_probe = available_memory_bytes

    def _persist(self):
        state_path = self.config.get("state_path")
        if state_path:
            path = Path(state_path)
            if path.is_symlink():
                raise OSError("Refusing a symbolic-link worker state file.")
            state = {"gateway_pid": os.getpid(), "worker_pid": self.process.pid if self.process else None,
                     "worker_argv": self.worker_argv, "model_state": self.model_state}
            _atomic_in_directory(path.parent, path.name, json.dumps(state).encode(), 0o600)

    def _refresh(self):
        if self.process and self.process.poll() is not None and self.model_state == "loaded":
            self.process = None
            if self.config.get("runtime") == "strata":
                from mlx_agent.process_tree import session_children
                self.strata_children.update(session_children(self.owned_session))
            if self.strata_children:
                from mlx_agent.process_tree import remaining
                try:
                    self.strata_children = remaining(self.strata_children)
                except (OSError, subprocess.SubprocessError):
                    pass
                if self.strata_children:
                    self.model_state = "failed"
                    self.last_error = "The Strata frontend exited while its engine may still be running."
                    self._persist()
                    return
            self.model_state = "unloaded"
            self.memory_budget_pending = False
            self._persist()

    def status(self):
        with self.condition:
            self._refresh()
            result = {"model_state": self.model_state, "active_requests": self.active_requests,
                    "worker_pid": self.process.pid if self.process and self.process.poll() is None else None,
                    "worker_port": self.worker_port if self.model_state == "loaded" else None,
                    "last_error": self.last_error, "memory_policy": dict(self.memory_policy),
                    "memory_check": self.memory_check, "load_blocked_reason": self.load_blocked_reason,
                    "resident_memory_check": self.resident_memory_check,
                    "request_blocked_reason": self.request_blocked_reason,
                    "last_unload_reason": self.last_unload_reason}
            if self.config.get("runtime") == "strata":
                result.update(memory_allocation=self.config.get("strata_memory_allocation"),
                              strata_decoding=self.config.get("strata_decoding"),
                              memory_budget_pending=self.memory_budget_pending)
            return result

    def _strata_speculation(self):
        if "strata_decoding" in self.config:
            return self.config["strata_decoding"] == "mtp"
        argv = self.config.get("worker_argv", [])
        return self.config.get("strata_speculation", "--speculation" in argv
                               and argv[argv.index("--speculation") + 1] == "on")

    def configure(self, changes):
        with self.condition:
            try:
                policy = validate_memory_policy(dict(self.memory_policy, **changes))
            except (ValueError, TypeError) as error:
                raise GatewayError(400, "invalid_policy", str(error)) from error
            if self.config.get("runtime") == "strata" and "context_tokens" in policy:
                raise GatewayError(400, "invalid_policy", STRATA_CONTEXT)
            if self.config.get("runtime") != "strata" and "model_memory_budget_gb" in policy:
                raise GatewayError(400, "invalid_policy", "Model memory budgets currently require Strata.")
            config = dict(self.config, memory_policy=policy)
            budget_changed = policy.get("model_memory_budget_gb") != self.memory_policy.get("model_memory_budget_gb")
            if budget_changed:
                if self.model_state in ("loading", "unloading"):
                    raise GatewayError(409, "model_busy", "Finish the current model transition before changing its budget.")
                from mlx_agent.strata_runtime import StrataError, read_model, memory_allocation
                budget = policy.get("model_memory_budget_gb")
                try:
                    allocation = None if budget is None else memory_allocation(read_model(config["local_path"]), budget,
                        config.get("strata_context_tokens", 131072), self._strata_speculation())
                except StrataError as error:
                    raise GatewayError(400, error.code, str(error)) from error
                argv, old = [], iter(config["worker_argv"])
                for value in old:
                    if value in ("--memory-budget-gb", "--expert-memory-gb"):
                        next(old)
                    else:
                        argv.append(value)
                if budget is not None:
                    argv += ["--memory-budget-gb", str(budget)]
                elif config.get("strata_expert_memory_gb") is not None:
                    argv += ["--expert-memory-gb", str(config["strata_expert_memory_gb"])]
                config.update(worker_argv=argv, strata_memory_allocation=allocation)
            if config.get("config_path"):
                path = Path(config["config_path"])
                try:
                    if path.is_symlink():
                        raise OSError("Refusing a symbolic-link JIT configuration.")
                    _atomic_in_directory(path.parent, path.name, json.dumps(config).encode(), 0o600)
                except OSError as error:
                    raise GatewayError(503, "policy_not_saved", str(error)) from error
            self.config.update(config)
            self.memory_policy = policy
            if budget_changed and self.model_state == "loaded":
                self.memory_budget_pending = True
            self.policy_revision += 1
            return self.status()

    def _check_memory(self):
        with self.condition:
            reserve = self.memory_policy["minimum_headroom_gb"]
            context = self.memory_policy.get("context_tokens")
            self.load_blocked_reason = None
        if reserve is None:
            return
        weights = sum(item.stat().st_size for item in Path(self.config["local_path"]).rglob("*")
                      if item.is_file() and item.suffix in (".safetensors", ".npz", ".bin", ".gguf"))
        if weights and self.config.get("adapter_path") is not None:
            weights += self.config["adapter_identity"]["adapters.safetensors"]["bytes"]
        kv = 0 if context is None else _kv_cache_bytes(self.config["local_path"], context)
        available = self.memory_probe()
        required = math.ceil(weights * 1.10 + kv + 1.5e9 + reserve * 1e9) if weights and kv is not None else None
        if self.config.get("runtime") == "strata":
            from mlx_agent.strata_runtime import read_model, memory_estimate, memory_allocation
            model = read_model(self.config["local_path"])
            context = self.config.get("strata_context_tokens", 131072)
            budget = self.memory_policy.get("model_memory_budget_gb")
            size = (memory_allocation(model, budget, context, self._strata_speculation())["estimated_resident_bytes"]
                    if budget is not None else memory_estimate(model, self.config.get("strata_expert_memory_gb"),
                                                              context, self._strata_speculation()))
            required = math.ceil(size + reserve * 1e9)
        with self.condition:
            self.memory_check = {"available_bytes": available, "required_available_bytes": required,
                                 "captured_at": time.time(), "estimated": True}
            if type(available) is not int or available < 0 or required is None:
                self.load_blocked_reason = ("The local config.json does not give the KV cache size for the policy context."
                                            if kv is None else "Memory headroom or model weight size is unavailable.")
                raise GatewayError(503, "memory_unknown", self.load_blocked_reason)
            if available < required:
                self.load_blocked_reason = "Insufficient memory headroom for estimated weights, {}runtime allowance and reserve.".format(
                    "KV cache, " if context is not None else "")
                raise GatewayError(503, "insufficient_headroom", self.load_blocked_reason)

    def _resident_check(self, policy, available, scope):
        known = type(available) is int and available >= 0
        required = math.ceil(policy["minimum_headroom_gb"] * 1e9)
        self.resident_memory_check = {
            "available_bytes": available if known else None,
            "minimum_available_bytes": required, "captured_at": time.time(),
            "scope": scope, "estimated": True,
        }
        if not known:
            return GatewayError(503, "memory_unknown", "OS memory headroom is unavailable for the resident model guard.")
        if available < required:
            return GatewayError(503, "insufficient_headroom", "Insufficient memory headroom for the configured reserve.")
        return None

    def _check_resident_memory(self, policy):
        if not policy.get("unload_under_pressure"):
            return
        # The admission already holds a request lease, so an idle reaper cannot
        # evict the worker while this bounded OS probe runs outside the lock.
        available = self.memory_probe()
        with self.condition:
            error = self._resident_check(policy, available, "request")
            self.request_blocked_reason = str(error) if error else None
        if error:
            raise error

    def acquire(self):
        with self.condition:
            if not self.condition.wait_for(lambda: self.model_state != "unloading" or self.closed, timeout=10):
                raise GatewayError(503, "model_unloading", "Model unload has not finished.")
            if self.closed:
                raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
            self._refresh()
            if self.memory_budget_pending:
                raise GatewayError(503, "model_reconfiguring", "The model is applying a memory budget; retry when it is idle.")
            if self.active_requests >= MAX_REQUESTS:
                raise GatewayError(503, "endpoint_busy", "Too many inference requests.")
            self.active_requests += 1
            self.activity_revision += 1
            policy = dict(self.memory_policy)
            self.request_blocked_reason = None
            starter = self.model_state not in ("loaded", "loading")
            if starter:
                self.model_state = "loading"
                self.last_error = None
        try:
            if starter:
                self._start()
            else:
                with self.condition:
                    self.condition.wait_for(lambda: self.model_state != "loading" or self.closed,
                                            timeout=self.config.get("worker_readiness_seconds", 60) + 5)
            with self.condition:
                if self.closed or self.model_state != "loaded":
                    raise GatewayError(503, "model_load_failed", self.last_error or "Model is not ready.")
            self._check_resident_memory(policy)
            with self.condition:
                if self.closed or self.model_state != "loaded":
                    raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
                return self.worker_port
        except BaseException:
            self.release()
            raise

    def _check_adapter(self):
        argv = self.config["worker_argv"]
        path = self.config.get("adapter_path")
        if path is None and "--adapter-path" not in argv:
            return
        from mlx_agent.lora import adapter_identity
        if (not isinstance(path, str) or "--adapter-path" not in argv
                or argv.index("--adapter-path") + 1 >= len(argv)
                or argv[argv.index("--adapter-path") + 1] != path
                or not self.config.get("adapter_identity")
                or adapter_identity(path) != self.config["adapter_identity"]):
            raise GatewayError(503, "adapter_changed", "Adapter files changed or have no reviewed identity. Stop and review a fresh serve plan.")

    def _check_training_receipt(self):
        binding = self.config.get("training_binding")
        if binding is None:
            return
        try:
            from mlx_agent.training_artifacts import read_trained_adapter
            _, current = read_trained_adapter(binding["training_receipt"])
            if current == binding:
                return
        except (OSError, ValueError, TypeError, KeyError):
            pass
        raise GatewayError(503, "training_receipt_changed", "The training receipt or its pinned artifacts changed. Stop and review a fresh serve plan.")

    def _start(self):
        try:
            if self.strata_children:
                from mlx_agent.process_tree import remaining
                self.strata_children = remaining(self.strata_children)
                if self.strata_children:
                    raise GatewayError(503, "model_unload_failed", "A previous Strata engine is still releasing memory.")
            if model_fingerprint(self.config["local_path"]) != self.config["fingerprint"]:
                raise GatewayError(503, "model_changed", "Model files changed. Stop and review a fresh serve plan.")
            self._check_training_receipt()
            self._check_adapter()
            self._check_memory()
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            argv = [str(port) if value == "{worker_port}" else value for value in self.config["worker_argv"]]
            environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
            log = open(self.config["log_path"], "ab", buffering=0) if self.config.get("log_path") else subprocess.DEVNULL
            with self.condition:
                if self.closed:
                    raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
                # Inherit the gateway's session: serve stop signals the whole
                # owned group, including a worker when the gateway is killed.
                self.process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                                stderr=subprocess.STDOUT, env=environment)
                self.log = log
                self.worker_port, self.worker_argv = port, argv
                self._persist()
                process = self.process
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            readiness_seconds = self.config.get("worker_readiness_seconds", 60)
            deadline = time.monotonic() + readiness_seconds
            while time.monotonic() < deadline:
                if process.poll() is not None or self.closed:
                    raise GatewayError(503, "model_load_failed", "Model worker exited before readiness.")
                try:
                    with opener.open("http://127.0.0.1:{}/v1/models".format(port), timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.1)
            else:
                raise GatewayError(503, "model_load_failed",
                                   "Model worker did not become ready in {} seconds.".format(readiness_seconds))
            if model_fingerprint(self.config["local_path"]) != self.config["fingerprint"]:
                raise GatewayError(503, "model_changed", "Model files changed while loading.")
            self._check_training_receipt()
            self._check_adapter()
            if self.config.get("runtime") == "strata":
                from mlx_agent.process_tree import descendants
                self.strata_children.update(descendants(process.pid))
            with self.condition:
                if self.closed:
                    raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
                self.model_state = "loaded"
                self._persist()
                self.condition.notify_all()
        except BaseException as error:
            failure = error
            try:
                self._stop_worker()
            except BaseException as cleanup_error:
                failure = GatewayError(503, "model_unload_failed", str(cleanup_error))
            with self.condition:
                self.model_state = "failed"
                self.last_error = str(failure)
                self._persist()
                self.condition.notify_all()
            if isinstance(failure, GatewayError):
                if failure is error:
                    raise
                raise failure from error
            raise GatewayError(503, "model_load_failed", str(failure)) from failure

    def release(self):
        with self.condition:
            self.active_requests -= 1
            self.activity_revision += 1
            if self.active_requests == 0:
                self.last_activity = self.clock()
            self.condition.notify_all()

    def _stop_worker(self):
        with self.condition:
            process = self.process
        if self.config.get("runtime") == "strata":
            from mlx_agent.process_tree import session_children
            self.strata_children.update(session_children(self.owned_session))
        if self.config.get("runtime") == "strata" and process and process.poll() is None:
            from mlx_agent.process_tree import descendants
            self.strata_children.update(descendants(process.pid))
        if process and process.poll() is None:
            process.terminate()
            try:
                # Strata's frontend closes its engine in three bounded 20-second stages.
                process.wait(timeout=65 if self.config.get("runtime") == "strata" else 3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self.strata_children or self.owned_session is not None:
            from mlx_agent.process_tree import remaining, session_children, terminate
            for _ in range(3):
                # A launcher can exec or create the engine while the frontend is exiting.
                self.strata_children.update(session_children(self.owned_session))
                terminate(self.strata_children)
                self.strata_children = remaining(session_children(self.owned_session))
                if not self.strata_children:
                    break
            if self.strata_children:
                raise OSError("The owned Strata session still has live engines; model reload is blocked.")
        with self.condition:
            self.process = None
            if self.log not in (None, subprocess.DEVNULL):
                self.log.close()
            self.log = None

    def unload_if_idle(self):
        with self.condition:
            self._refresh()
            policy = dict(self.memory_policy)
            budget_reload = self.memory_budget_pending
            if (self.closed or self.model_state != "loaded" or self.active_requests
                    or (not budget_reload and (policy["keep_loaded"] or self.last_activity is None))):
                return False
            idle = budget_reload or bool(policy["idle_timeout_seconds"] and
                        self.clock() - self.last_activity >= policy["idle_timeout_seconds"])
            if idle:
                self.model_state = "unloading"
            elif not policy.get("unload_under_pressure"):
                return False
            worker = self.process
            revisions = self.activity_revision, self.policy_revision
        if not idle:
            available = self.memory_probe()
            with self.condition:
                if (self.closed or self.model_state != "loaded" or self.active_requests
                        or self.process is not worker or self.memory_policy != policy
                        or revisions != (self.activity_revision, self.policy_revision)):
                    return False
                error = self._resident_check(policy, available, "idle_reclaim")
                if error is None or error.code == "memory_unknown":
                    return False
                self.model_state = "unloading"
        self._finish_unload("memory_budget" if budget_reload else "idle" if idle else "memory_pressure")
        return True

    def unload(self):
        with self.condition:
            if self.closed:
                raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
            if self.active_requests or self.model_state == "unloading":
                raise GatewayError(409, "model_busy", "Finish active requests before unloading.")
            self.model_state = "unloading"
        self._finish_unload("manual")
        return self.status()

    def _finish_unload(self, reason):
        try:
            self._stop_worker()
            with self.condition:
                self.model_state = "unloaded"
                self.memory_budget_pending = False
                self.last_error = None
                self.load_blocked_reason = None
                self.last_unload_reason = reason
                self._persist()
                self.condition.notify_all()
        except BaseException:
            with self.condition:
                self.model_state = "failed"
                self.condition.notify_all()
            raise

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self._stop_worker()
        with self.condition:
            self.model_state = "unloaded"
            self._persist()


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, config):
        self.controller = ModelWorker(config)
        self.config = config
        self.slots = threading.BoundedSemaphore(MAX_REQUESTS + 4)
        self.responses = collections.OrderedDict()
        self.responses_lock = threading.Lock()
        self.end_text = chat_completions.end_of_sequence(config["local_path"])
        super().__init__(address, GatewayHandler)

    def remember(self, response):
        """Keep the most recent stored Responses objects for this gateway process."""
        with self.responses_lock:
            self.responses[response["id"]] = response
            while len(self.responses) > STORED_RESPONSES:
                self.responses.popitem(last=False)

    def recall(self, identifier):
        with self.responses_lock:
            return self.responses.get(identifier)

    def server_bind(self):
        # HTTPServer normally performs reverse DNS here. This gateway binds
        # a numeric loopback address and must start without network discovery.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def serve_forever(self, poll_interval=0.5):
        stopped = threading.Event()

        def reclaim_idle():
            while not stopped.wait(0.5):
                try:
                    self.controller.unload_if_idle()
                except OSError:
                    # Persist/stop failures remain visible in controller state.
                    pass

        timer = threading.Thread(target=reclaim_idle, daemon=True)
        timer.start()
        try:
            super().serve_forever(poll_interval)
        finally:
            stopped.set()
            timer.join(timeout=7)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MLX-JIT"
    sys_version = ""

    def setup(self):
        self.request.settimeout(120)
        super().setup()

    def log_message(self, *args):
        pass

    def _guard(self, control=False):
        port = self.server.server_port
        if self.headers.get("Host") not in ("127.0.0.1:{}".format(port), "localhost:{}".format(port)):
            raise GatewayError(403, "invalid_host", "Use the loopback endpoint address.")
        origin = self.headers.get("Origin")
        if origin and origin not in ("http://127.0.0.1:{}".format(port), "http://localhost:{}".format(port)):
            raise GatewayError(403, "invalid_origin", "Cross-origin access is refused.")
        if control and not hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                               ("Bearer " + self.server.config["control_token"]).encode()):
            raise GatewayError(403, "control_denied", "Endpoint control authorization required.")

    def _json(self, status, value):
        encoded = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(encoded)

    def _error(self, error):
        if self.path == "/v1/messages":
            self._json(error.status, error_body(error.status, str(error)))
        elif self.path.startswith("/v1/responses"):
            self._json(error.status, responses_api.error_body(error.status, error.code, str(error)))
        else:
            self._json(error.status, {"error": {"code": error.code, "message": str(error)}})

    @staticmethod
    def _worker_failure(response):
        detail = response.read(4096)
        try:
            reason = json.loads(detail).get("error")
        except (ValueError, AttributeError):
            reason = None
        message = "Model worker returned HTTP {}".format(response.status)
        return GatewayError(502, "worker_error", message + (": {}".format(reason) if isinstance(reason, str) else "."))

    def _worker_replied(self, worker):
        """Wait without a deadline for the worker's first reply byte; False once the client disconnects."""
        with selectors.DefaultSelector() as selector:
            selector.register(worker, selectors.EVENT_READ)
            selector.register(self.connection, selectors.EVENT_READ)
            while True:
                for key, _ in selector.select():
                    if key.fileobj is worker:
                        return True
                    try:
                        if not self.connection.recv(1, socket.MSG_PEEK):
                            return False
                    except OSError:
                        return False
                    selector.unregister(self.connection)

    def _relay(self, response, translate):
        """Answer a non-streamed request from the worker's translated chat completion."""
        if response.status != 200:
            raise self._worker_failure(response)
        payload = response.read(MAX_BODY + 1)
        try:
            if len(payload) > MAX_BODY:
                raise WorkerResponseError("Model worker response exceeds 16 MiB.")
            value = translate(json.loads(payload))
        except WorkerResponseError as error:
            raise GatewayError(502, "invalid_worker_response", str(error))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise GatewayError(502, "invalid_worker_response", "Model worker returned an invalid response.")
        self._json(200, value)
        return value

    def _stream(self, response, translator):
        """Translate the worker's chat-completion chunks into client events as they arrive."""
        if response.status != 200:
            raise self._worker_failure(response)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(b"".join(translator.start()))
        try:
            while line := response.readline(MAX_BODY):
                if line.startswith(b":"):
                    # Prompt-progress comments keep the client stream active before the first token.
                    self.wfile.write(b"".join(translator.keepalive()))
                    self.wfile.flush()
                    continue
                if not line.startswith(b"data: "):
                    continue
                data = line[len(b"data: "):].strip()
                if data == b"[DONE]":
                    self.wfile.write(b"".join(translator.finish()))
                    break
                self.wfile.write(b"".join(translator.chunk(json.loads(data))))
                self.wfile.flush()
            else:
                raise WorkerResponseError("Model worker stream ended early.")
        except WorkerResponseError as error:
            self.wfile.write(b"".join(translator.fail(str(error))))
        except (ValueError, TypeError, AttributeError):
            self.wfile.write(b"".join(translator.fail("Model worker returned an invalid streamed response.")))
        self.wfile.flush()

    def do_GET(self):
        try:
            self._guard(control=self.path == "/_mlx/status")
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": self.server.config["model"], "object": "model"}]})
            elif self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/_mlx/status":
                self._json(200, self.server.controller.status())
            elif self.path.startswith("/v1/responses/"):
                stored = self.server.recall(self.path[len("/v1/responses/"):])
                if stored is None:
                    raise GatewayError(404, "not_found", "No stored response has that id.")
                self._json(200, stored)
            else:
                raise GatewayError(404, "not_found", "Unknown endpoint.")
        except GatewayError as error:
            self._error(error)

    def do_POST(self):
        connection = None
        leased = False
        response_started = False
        try:
            self._guard(control=self.path in ("/_mlx/unload", "/_mlx/policy"))
            if self.headers.get("Transfer-Encoding"):
                raise GatewayError(400, "invalid_body", "Use a bounded Content-Length request.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise GatewayError(400, "invalid_body", "Invalid Content-Length.")
            if not 0 <= length <= MAX_BODY:
                raise GatewayError(413, "body_too_large", "Request exceeds 16 MiB.")
            try:
                body = json.loads(self.rfile.read(length)) if length else {}
            except (ValueError, UnicodeError):
                raise GatewayError(400, "invalid_body", "Expected a JSON object.")
            if not isinstance(body, dict):
                raise GatewayError(400, "invalid_body", "Expected a JSON object.")
            if self.path == "/_mlx/unload":
                self._json(200, self.server.controller.unload())
                return
            if self.path == "/_mlx/policy":
                self._json(200, self.server.controller.configure(body))
                return
            translated = self.path in ("/v1/messages", "/v1/responses")
            allow_images = self.server.config.get("runtime") == "mlx-vlm"
            if self.path == "/v1/messages":
                try:
                    body = to_chat_request(body, allow_images=allow_images)
                except MessagesRequestError as error:
                    raise GatewayError(400, "invalid_request", str(error))
            elif self.path == "/v1/responses":
                try:
                    fields, body = responses_api.envelope(body), responses_api.to_chat_request(body, allow_images=allow_images)
                except responses_api.ResponsesRequestError as error:
                    raise GatewayError(400, "invalid_request", str(error))
            elif self.path not in ("/v1/chat/completions", "/v1/completions"):
                raise GatewayError(404, "not_found", "Unknown inference endpoint.")
            aliases = {self.server.config["model"], self.server.config["local_path"], "default_model"}
            local = Path(self.server.config["local_path"])
            aliases.add(local.name)
            if local.parent.name == "snapshots" and local.parent.parent.name.startswith("models--"):
                aliases.add(local.parent.parent.name[len("models--"):].replace("--", "/"))
            requested = body.get("model", "default_model")
            if not isinstance(requested, str) or requested not in aliases:
                raise GatewayError(400, "model_not_allowed", "This endpoint serves only its configured local model.")
            if any(body.get(key) not in (None, "default_model") for key in ("adapter_path", "adapter", "draft_model", "draft_adapter")):
                raise GatewayError(400, "model_not_allowed", "Request-selected adapters and draft models are refused.")
            tokens = body.get("max_tokens", self.server.config["max_tokens"])
            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
                raise GatewayError(400, "invalid_body", "max_tokens must be a positive integer.")
            body["max_tokens"] = min(tokens, self.server.config["max_tokens"])
            body["model"] = self.server.config["local_path"]
            runtime = self.server.config.get("runtime")
            # Existing receipts predate the runtime field; recognize only the
            # exact native console entrypoint for those configurations.
            legacy_mlx = runtime is None and Path(self.server.config["worker_argv"][0]).name == "mlx_lm.server"
            if runtime == "mlx_lm" or legacy_mlx:
                chat_completions.decode_tool_arguments(body)
            port = self.server.controller.acquire()
            leased = True
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            connection.request("POST", "/v1/chat/completions" if translated else self.path, json.dumps(body),
                               {"Content-Type": "application/json"})
            # MLX-LM writes nothing until an unstreamed answer is complete or a queued request starts.
            if not self._worker_replied(connection.sock):
                return
            response = connection.getresponse()

            def trimmed(value):
                chat_completions.trim_end(value, self.server.end_text)
                return value

            if self.path == "/v1/messages" and body.get("stream"):
                response_started = True
                self._stream(response, StreamTranslator(requested))
                return
            if self.path == "/v1/messages":
                self._relay(response, lambda value: from_chat_response(trimmed(value), requested))
                return
            if self.path == "/v1/responses":
                if body.get("stream"):
                    response_started = True
                    stream = responses_api.ResponseStream(requested, fields)
                    self._stream(response, stream)
                    result = stream.response
                else:
                    result = self._relay(response, lambda value: responses_api.from_chat_response(trimmed(value), requested, fields))
                if fields["store"]:
                    self.server.remember(result)
                return
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.send_header("Connection", "close")
            self.end_headers()
            response_started = True
            self.close_connection = True
            if response.status == 200 and body.get("stream") and self.path == "/v1/chat/completions":
                stream = chat_completions.ChunkStream()
                while line := response.readline(MAX_BODY):
                    self.wfile.write(stream.line(line))
                    self.wfile.flush()
                return
            if response.status == 200 and not body.get("stream"):
                payload = response.read(MAX_BODY + 1)
                self.wfile.write(chat_completions.complete_body(payload, self.server.end_text)
                                 if len(payload) <= MAX_BODY else payload)
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except GatewayError as error:
            self._error(error)
        except (OSError, http.client.HTTPException):
            if not response_started:
                self._error(GatewayError(503, "worker_unavailable", "Model worker connection failed."))
        finally:
            if connection:
                connection.close()
            if leased:
                self.server.controller.release()


def make_server(config, port):
    return GatewayServer(("127.0.0.1", port), config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = Path(args.config)
    if path.is_symlink():
        raise ValueError("Refusing a symbolic-link JIT configuration.")
    config = json.loads(path.read_text())
    if args.model != config["model"]:
        raise ValueError("JIT configuration identity changed.")
    server = make_server(config, args.port)
    if config.get("runtime") == "strata":
        if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
            server.server_close()
            raise ValueError("Launch Strata through serve start in its owned gateway session.")
        server.controller.owned_session = os.getpid()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    finally:
        server.controller.close()
        server.server_close()


if __name__ == "__main__":
    main()
