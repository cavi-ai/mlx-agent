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
import secrets
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
from mlx_agent.messages_api import (MessagesRequestError, StreamTranslator, WorkerResponseError, error_body,
                                    from_chat_response, to_chat_request)
from mlx_agent.transactions import _atomic_in_directory
from mlx_agent.residency import MemoryRegistry, MemoryAdmissionError
from mlx_agent import allocator_cache
from mlx_agent.inference_telemetry import InferenceTelemetry
from mlx_agent import inference_defaults
from mlx_agent.compute import ComputeBusy, ComputeLease, ExecutionAdmission, group_members, process_identity
from mlx_agent.identity_pass import recheck_bound_identity
from mlx_agent.memory_policy import (PolicyRejected, enforce_configure, execution_wait_seconds, reject_cache_outside_tasks,
                                     release_reason, releases_after_request, reservation, validate as validate_memory_policy)

MAX_BODY = 16 * 1024 * 1024
MAX_REQUESTS = 32
STORED_RESPONSES = 64
DRAIN_TIMEOUT_SECONDS = 30


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


def public_model_name(local_path):
    """Path-free name of a local model: its Hub repository for cache snapshots, else its directory."""
    local = Path(local_path)
    if local.parent.name == "snapshots" and local.parent.parent.name.startswith("models--"):
        return local.parent.parent.name[len("models--"):].replace("--", "/")
    return local.name


def model_aliases(config):
    """Request names accepted for an endpoint's one configured local model."""
    return {config["model"], config["local_path"], "default_model", Path(config["local_path"]).name,
            public_model_name(config["local_path"])}


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
        self.private_endpoint = None
        self.worker_argv = None
        self.model_state = "unloaded"
        self.active_requests = 0
        self.inference_telemetry = InferenceTelemetry()
        defaults = config.get('request_defaults', {})
        if not isinstance(defaults, dict) or set(defaults) - {'system_prompt', 'temperature', 'top_p'}:
            raise ValueError('Invalid request defaults.')
        self.request_defaults = inference_defaults.validate(defaults.get('system_prompt'), defaults.get('temperature'), defaults.get('top_p'))
        self.execution = ExecutionAdmission("serve:" + config.get("model", config["local_path"]), config.get("compute_state_dir"),
            shared=lambda: self.memory_registry.shared_execution_enabled())
        self.worker_group = None
        self.worker_identity = None
        self.execution_unresolved = False
        self.last_error = None
        self.memory_guard_error = None
        self.closed = False
        self.shutdown_requested = False
        self.log = None
        self.memory_policy = validate_memory_policy(config.get("memory_policy"))
        reject_cache_outside_tasks(self.memory_policy, config.get("runtime"))
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
        self.memory_registry = MemoryRegistry(config.get('memory_state_dir'))
        self.memory_token = secrets.token_hex(16)
        self.memory_reservation = None
        self.worker_memory = None
        self.cache_control_active = False
        self.workspace_bytes = 1_500_000_000
        self.growth_bytes = 0
        self.memory_materialized = False
        self.memory_transition_blocked = False

    def _persist(self):
        state_path = self.config.get("state_path")
        if state_path:
            path = Path(state_path)
            if path.is_symlink():
                raise OSError("Refusing a symbolic-link worker state file.")
            state = {"gateway_pid": os.getpid(), "worker_pid": self.process.pid if self.process else None,
                     "worker_argv": self.worker_argv, "model_state": self.model_state}
            if self.process:
                from mlx_agent.serve import _pid_start
                state["worker_started_at"] = _pid_start(self.process.pid)
            _atomic_in_directory(path.parent, path.name, json.dumps(state).encode(), 0o600)

    def _refresh(self):
        if self.process and self.process.poll() is not None and self.model_state == "loaded":
            if self.worker_group is not None:
                try:
                    self._stop_group()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    self.model_state = "failed"
                    self.last_error = str(error)
                    self._persist()
                    return
            self.process = None
            if self.private_endpoint is not None:
                self.private_endpoint.close()
                self.private_endpoint = None
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
            self._release_memory()
            self._persist()

    def status(self):
        with self.condition:
            self._refresh()
            result = {"model_state": self.model_state, "active_requests": self.active_requests,
                    "worker_pid": self.process.pid if self.process and self.process.poll() is None else None,
                    "inference_telemetry": self.inference_telemetry.snapshot(),
                    "request_defaults": inference_defaults.summary(self.request_defaults),
                    "worker_port": self.worker_port if self.model_state == "loaded" else None,
                    "last_error": self.last_error, "memory_policy": dict(self.memory_policy),
                    "execution": {"active_requests": self.execution.users, "waiting_requests": self.execution.waiting},
                    "memory_check": self.memory_check, "load_blocked_reason": self.load_blocked_reason,
                    "memory_reservation": self.memory_reservation, "worker_memory": self.worker_memory,
                    "allocator_cache": self._allocator_cache_status(),
                    "resident_memory_check": self.resident_memory_check,
                    "request_blocked_reason": self.request_blocked_reason,
                    "last_unload_reason": self.last_unload_reason,
                    "api_key_sha256": self.config.get("api_key_sha256", self.config.get("access_key_sha256"))}
            if self.config.get("private_worker"):
                result["worker_transport"] = "inherited-unix-v1"
            if self.config.get("runtime") == "strata":
                result.update(memory_allocation=self.config.get("strata_memory_allocation"),
                              strata_decoding=self.config.get("strata_decoding"),
                              memory_budget_pending=self.memory_budget_pending)
            return result

    def _allocator_cache_status(self):
        return {'supported': False, 'available': False, 'pending': False,
                'control_path': None, 'policy_fields': [], 'reclaim_modes': [], 'active': False,
                'configured_policy': allocator_cache.from_memory(self.memory_policy),
                'applied_policy': None, 'last_reclaim': None, 'error': None}

    def reclaim_cache(self):
        raise GatewayError(400, 'cache_control_unavailable', 'This runtime does not expose allocator cache controls.')

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
                enforce_configure(policy, self.config.get("runtime"), changes)
            except PolicyRejected as error:
                raise GatewayError(400, error.code, str(error)) from error
            except (ValueError, TypeError) as error:
                raise GatewayError(400, "invalid_policy", str(error)) from error
            config = dict(self.config, memory_policy=policy)
            budget_changed = policy.get("model_memory_budget_gb") != self.memory_policy.get("model_memory_budget_gb")
            if budget_changed and self.config.get("runtime") == "strata":
                if self.model_state in ("loading", "unloading"):
                    raise GatewayError(409, "model_busy", "Finish the current model transition before changing its budget.")
                from mlx_agent.strata_runtime import (StrataError, memory_allocation, memory_estimate,
                                                       memory_growth_allowance, read_model)
                budget = policy.get("model_memory_budget_gb")
                try:
                    model = read_model(config["local_path"])
                    context = config.get("strata_context_tokens", 131072)
                    speculation = self._strata_speculation()
                    allocation = None if budget is None else memory_allocation(model, budget, context, speculation)
                    if allocation is not None:
                        capacity = 0 if allocation["mode"] == "resident" else allocation["expert_memory_gb"]
                        estimate = allocation["estimated_resident_bytes"]
                    else:
                        capacity = config.get("strata_expert_memory_gb")
                        estimate = memory_estimate(model, capacity, context, speculation)
                    growth = memory_growth_allowance(model, context, capacity, speculation)
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
                config.update(worker_argv=argv, strata_memory_allocation=allocation,
                              strata_memory_estimate_bytes=estimate, strata_memory_growth_bytes=growth)
            self._set_memory_headroom(max(policy['minimum_headroom_gb'] or 0,self.memory_policy['minimum_headroom_gb'] or 0))
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
            self._set_memory_headroom(policy['minimum_headroom_gb'] or 0)
            if budget_changed and self.config.get("runtime") == "strata" and self.model_state == "loaded":
                self.memory_budget_pending = True
            self.policy_revision += 1
            return self.status()

    def _match_serving_plan(self):
        # serve imports the identity pass, so ServeError is imported when the pass runs.
        from mlx_agent.serve import ServeError
        try:
            recheck_bound_identity(self.config)
        except ServeError as error:
            raise GatewayError(503, error.code, str(error)) from error

    def _check_memory(self):
        with self.condition:
            self.load_blocked_reason = None
            policy = dict(self.memory_policy)
            config = dict(self.config)
        outcome = reservation(policy, config)
        if outcome is None:
            self.load_blocked_reason = "Model or context memory estimate is unavailable."
            raise GatewayError(503, "memory_unknown", self.load_blocked_reason)
        self.workspace_bytes = outcome["workspace_bytes"]
        self.growth_bytes = outcome["growth_bytes"]
        model_key = hashlib.sha256(json.dumps([str(Path(self.config["local_path"]).resolve()),
            self.config.get("fingerprint"), self.config.get("runtime"), self.config["worker_argv"]],
            sort_keys=True).encode()).hexdigest()
        try:
            self.memory_reservation = self.memory_registry.reserve(self.memory_token, model_key,
                outcome["reserve_bytes"], outcome["workspace_bytes"],
                lambda: math.ceil((self.memory_policy["minimum_headroom_gb"] or 0) * 1e9), self.memory_probe,
                growth_bytes=outcome["growth_bytes"])
            self.memory_check = self.memory_reservation
        except MemoryAdmissionError as error:
            self.memory_check = error.details
            self.load_blocked_reason = str(error)
            raise GatewayError(503, error.code, str(error)) from error

    def _release_memory(self):
        if self.memory_reservation is not None:
            self.memory_registry.release(self.memory_token)
            self.memory_reservation = None
        self.worker_memory = None
        self.memory_materialized = False
        self.memory_transition_blocked = False

    def _set_memory_headroom(self, gigabytes):
        if self.memory_reservation is not None:
            try:
                self.memory_reservation = self.memory_registry.set_headroom(self.memory_token, math.ceil(gigabytes * 1e9))
            except MemoryAdmissionError as error:
                raise GatewayError(503, error.code, str(error)) from error

    def _launching_memory(self):
        if self.memory_reservation is not None:
            self.memory_reservation = self.memory_registry.launching(self.memory_token)

    def _bind_memory(self, process, identity):
        if self.memory_reservation is not None:
            self.memory_reservation = self.memory_registry.bind_worker(
                self.memory_token, process.pid, identity, os.getpgid(process.pid))

    def _materialized_memory(self):
        # Readiness alone does not prove that an asynchronous model loader has
        # allocated its weights. Keep cold allocations reserved until output.
        if self.memory_reservation is not None and not self.memory_materialized and not self.memory_transition_blocked:
            try:
                self.memory_reservation = self.memory_registry.settle(self.memory_token,
                    workspace_bytes=self.workspace_bytes + self.growth_bytes)
            except MemoryAdmissionError as error:
                if error.code != "memory_registry_busy":
                    raise GatewayError(503, error.code, str(error)) from error
                self.memory_transition_blocked = True
                self.last_error = "Memory reservation remains pending: " + str(error)
                return
            self.memory_materialized = True
            self.last_error = None

    def _reserve_workspace(self):
        try:
            self.memory_check = self.memory_registry.reserve_workspace(
                self.memory_token, self.workspace_bytes, self.memory_probe)
            self.memory_reservation = self.memory_check
        except MemoryAdmissionError as error:
            self.memory_check = error.details
            raise GatewayError(503, error.code, str(error)) from error

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
        try:
            budget = self.memory_registry.snapshot().get('agent_budget')
        except MemoryAdmissionError as error:
            raise GatewayError(503, error.code, str(error)) from error
        if budget is not None and not budget.get('within_budget'):
            raise GatewayError(503, 'agent_memory_budget_exceeded',
                               'The shared agent memory budget is unavailable or exhausted.')
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

    def acquire(self, cancelled=lambda: False):
        with self.condition:
            if not self.condition.wait_for(lambda: self.model_state != "unloading" or self.closed, timeout=10):
                raise GatewayError(503, "model_unloading", "Model unload has not finished.")
            if self.closed:
                raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
            self._refresh()
            if self.execution_unresolved:
                raise GatewayError(503, "worker_unavailable", "The previous backend execution has not stopped.")
            if self.memory_budget_pending:
                raise GatewayError(503, "model_reconfiguring", "The model is applying a memory budget; retry when it is idle.")
            if self.active_requests >= MAX_REQUESTS:
                raise GatewayError(503, "endpoint_busy", "Too many inference requests.")
            self.active_requests += 1
            self.memory_transition_blocked = False
            self.activity_revision += 1
            policy = dict(self.memory_policy)
            self.request_blocked_reason = None
        admitted = False
        try:
            self.execution.acquire(execution_wait_seconds(policy), lambda: self.closed or cancelled())
            admitted = True
            with self.condition:
                starter = self.model_state not in ("loaded", "loading")
                if starter:
                    self.model_state = "loading"
                    self.last_error = None
            if starter:
                self._start()
            else:
                with self.condition:
                    self.condition.wait_for(lambda: self.model_state != "loading" or self.closed,
                                            timeout=self.config.get("worker_readiness_seconds", 60) + 5)
            with self.condition:
                if self.closed or self.model_state != "loaded":
                    raise GatewayError(503, "model_load_failed", self.last_error or "Model is not ready.")
            self.execution.bind(self.process.pid)
            self._check_resident_memory(policy)
            with self.condition:
                if self.closed or self.model_state != "loaded":
                    raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
                return self.worker_port
        except BaseException as error:
            self.release(compute=admitted)
            if isinstance(error, MemoryAdmissionError):
                raise GatewayError(503, error.code, str(error)) from error
            if isinstance(error, ComputeBusy):
                raise GatewayError(503, "compute_busy", str(error)) from error
            if isinstance(error, InterruptedError):
                raise GatewayError(503, "execution_cancelled", str(error)) from error
            raise

    def _start(self):
        try:
            if self.strata_children:
                from mlx_agent.process_tree import remaining
                self.strata_children = remaining(self.strata_children)
                if self.strata_children:
                    raise GatewayError(503, "model_unload_failed", "A previous Strata engine is still releasing memory.")
            self._match_serving_plan()
            self._check_memory()
            self._launching_memory()
            private = self.config.get("private_worker")
            if private:
                from mlx_agent.private_serve import PrivateEndpoint
                self.private_endpoint = PrivateEndpoint()
                port = 0
            else:
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    port = reservation.getsockname()[1]
            argv = [str(port) if value == "{worker_port}" else value for value in self.config["worker_argv"]]
            extra = {}
            if private:
                from mlx_agent.private_serve import worker_command
                descriptor = self.private_endpoint.listener.fileno()
                argv = worker_command(argv, self.config["local_path"], descriptor)
                extra["pass_fds"] = (descriptor,)
            environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
            splash = self.config.get("runtime") == "splash"
            if splash:
                # Splash execs its server over this worker, so it watches the gateway itself.
                environment.pop("SPLASH_API_KEY", None)
                environment["SPLASH_APP_PID"] = str(os.getpid())
            if private or self.config.get("api_key_sha256") is not None:
                from mlx_agent.protected_worker import command, WORKER_DIGEST
                self.worker_key = secrets.token_urlsafe(32)
                if splash:
                    # Splash checks its own key on every route but /health, /ready and its chat page.
                    environment["SPLASH_API_KEY"] = self.worker_key
                else:
                    environment[WORKER_DIGEST] = hashlib.sha256(self.worker_key.encode()).hexdigest()
                    if not private:
                        argv = command(argv, self.config.get("runtime") or "mlx_lm")
            from mlx_agent.owned_worker import command as owned_command
            ready_read, ready_write = os.pipe()
            gate_read, gate_write = os.pipe()
            try:
                argv = owned_command(argv, ready_write, gate_read)
                extra["pass_fds"] = (*extra.get("pass_fds", ()), ready_write, gate_read)
                log = open(self.config["log_path"], "ab", buffering=0) if self.config.get("log_path") else subprocess.DEVNULL
                with self.condition:
                    if self.closed:
                        if log != subprocess.DEVNULL:
                            log.close()
                        raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
                    self.log = log
                    self.process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, env=environment, start_new_session=True, **extra)
                    self.worker_group = self.process.pid
                    if private:
                        self.private_endpoint.listener.close()
                    self.worker_port, self.worker_argv = (None if private else port), argv
                    self._persist()
                    process = self.process
                os.close(ready_write)
                ready_write = None
                os.close(gate_read)
                gate_read = None
                with selectors.DefaultSelector() as selector:
                    selector.register(ready_read, selectors.EVENT_READ)
                    if not selector.select(5) or os.read(ready_read, 1) != b'R':
                        raise GatewayError(503, "model_load_failed", "Model worker startup was not acknowledged.")
                self.execution.bind(process.pid)
                self.worker_identity = process_identity(process.pid)
                os.write(gate_write, b'G')
            finally:
                for descriptor in (ready_read, ready_write, gate_read, gate_write):
                    if descriptor is not None:
                        os.close(descriptor)
            from mlx_agent.inference_auth import loopback_opener
            opener = loopback_opener()
            readiness_seconds = self.config.get("worker_readiness_seconds", 60)
            deadline = time.monotonic() + readiness_seconds
            while time.monotonic() < deadline:
                if process.poll() is not None or self.closed:
                    raise GatewayError(503, "model_load_failed", "Model worker exited before readiness.")
                try:
                    if private:
                        from mlx_agent.private_serve import PRIVATE_PROTOCOL
                        connection = self.connection(timeout=1)
                        try:
                            connection.request("GET", "/_mlx/private", headers={"Authorization": "Bearer " + self.worker_key})
                            response = connection.getresponse()
                            ready = response.status == 200 and response.getheader("X-MLX-Agent-Transport") == PRIVATE_PROTOCOL
                            response.read(1024)
                        finally:
                            connection.close()
                        if ready:
                            break
                    else:
                        request = urllib.request.Request("http://127.0.0.1:{}/v1/models".format(port),
                            headers={"Authorization": "Bearer " + self.worker_key} if self.config.get("api_key_sha256") is not None else {})
                        with opener.open(request, timeout=1) as response:
                            if response.status == 200:
                                break
                except OSError:
                    time.sleep(0.1)
            else:
                raise GatewayError(503, "model_load_failed",
                                   "Model worker did not become ready in {} seconds.".format(readiness_seconds))
            if splash:
                # Splash's launcher execs its server over the owned worker: bind the server that answered.
                self.worker_identity = process_identity(process.pid)
                self.execution.bind(process.pid, refresh=True)
            self._bind_memory(process,process_identity(process.pid))
            self._match_serving_plan()
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

    def _idle_execution(self, lease):
        process = self.process
        if process is None:
            return
        try:
            if self.execution_unresolved:
                raise ComputeBusy("The upstream execution did not complete.")
            if process.poll() is not None:
                raise ComputeBusy("The model worker exited.")
            members = {pid: process_identity(pid) for pid in group_members(process.pid)}
            if not all(members.values()):
                raise ComputeBusy("The idle backend group cannot be inspected.")
            lease.release_idle_process_group(process.pid, members)
        except (ComputeBusy, OSError, ValueError, subprocess.SubprocessError):
            self.execution_unresolved = True
            self._stop_worker()
            with self.condition:
                self.model_state = "unloaded"
                self._persist()

    def release(self, compute=True):
        try:
            if compute:
                self.execution.release(self._idle_execution)
        finally:
            self._release_request()

    def _release_request(self):
        with self.condition:
            self.active_requests -= 1
            self.activity_revision += 1
            if self.active_requests == 0:
                self.last_activity = self.clock()
            release = self.active_requests == 0 and self.model_state == "loaded" and releases_after_request(self.memory_policy)
            if release:
                self.model_state = "unloading"
            self.condition.notify_all()
        if release:
            self._finish_unload("request_complete")

    def connection(self, timeout=120):
        if self.private_endpoint is not None:
            return self.private_endpoint.connection(timeout)
        return http.client.HTTPConnection("127.0.0.1", self.worker_port, timeout=timeout)

    def _stop_worker(self):
        if self.worker_group is not None:
            self._stop_group()
            return
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
            if self.private_endpoint is not None:
                self.private_endpoint.close()
                self.private_endpoint = None
        self._release_memory()

    def _stop_group(self):
        with self.condition:
            process, group = self.process, self.worker_group
        if process is not None and process.poll() is None:
            if self.worker_identity is not None and process_identity(process.pid) != self.worker_identity:
                raise OSError("The model worker identity changed; shutdown is blocked.")
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=65 if self.config.get("runtime") == "strata" else 3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
        if group_members(group):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 3
            while group_members(group) and time.monotonic() < deadline:
                time.sleep(.05)
            if group_members(group):
                raise OSError("The owned model process group has not stopped.")
        with self.condition:
            self.process = None
            self.worker_group = self.worker_identity = None
            self.execution_unresolved = False
            self.strata_children = {}
            if self.log not in (None, subprocess.DEVNULL):
                self.log.close()
            self.log = None
            if self.private_endpoint is not None:
                self.private_endpoint.close()
                self.private_endpoint = None
        self._release_memory()

    def unload_if_idle(self):
        with self.condition:
            self._refresh()
            policy = dict(self.memory_policy)
            budget_reload = self.memory_budget_pending
            busy = self.closed or self.model_state != "loaded" or self.active_requests or self.cache_control_active
            idle_seconds = None if self.last_activity is None else self.clock() - self.last_activity
            reason = release_reason(policy, busy=busy, idle_seconds=idle_seconds,
                                     budget_reload=budget_reload)
            if reason is None:
                return False
            if reason != "probe":
                self.model_state = "unloading"
            worker = self.process
            revisions = self.activity_revision, self.policy_revision
        if reason == "probe":
            available = self.memory_probe()
            with self.condition:
                if (self.closed or self.model_state != "loaded" or self.active_requests
                        or self.process is not worker or self.memory_policy != policy
                        or revisions != (self.activity_revision, self.policy_revision)):
                    return False
                idle_seconds = None if self.last_activity is None else self.clock() - self.last_activity
                reason = release_reason(policy, busy=False, idle_seconds=idle_seconds,
                                         budget_reload=False, available=available)
                if reason is None:
                    return False
                self.model_state = "unloading"
        try:
            with ComputeLease("idle-unload:" + self.config.get("model", self.config["local_path"]), self.config.get("compute_state_dir")):
                self._finish_unload(reason)
        except ComputeBusy:
            with self.condition:
                self.model_state = "loaded"
                self.condition.notify_all()
            return False
        return True

    def guard_agent_budget(self):
        """Stop this owned worker on a measured breach; never signal external agents."""
        with self.condition:
            worker = self.process
            if worker is None or self.closed or self.model_state not in ('loading', 'loaded'):
                return
        try:
            snapshot = self.memory_registry.snapshot()
            budget = snapshot.get('agent_budget')
        except MemoryAdmissionError as error:
            if error.code == 'memory_registry_busy':
                return  # Admission is serialized; contention is not a measured breach.
            budget = {'status': 'unavailable', 'error': str(error)}
        if budget is None:
            return
        allocation = budget.get('allocations', {}).get(self.memory_token, {})
        ceiling = allocation.get('execution_bytes')
        job_breach = ceiling is not None and allocation.get('observed_bytes', 0) > ceiling
        if budget.get('status') == 'unavailable' or not budget.get('within_budget', False) or job_breach:
            with self.condition:
                if (self.process is not worker or self.closed
                        or self.model_state not in ('loading', 'loaded')):
                    return
                # Claim this transition before a concurrent manual or idle unload.
                self.model_state = 'unloading'
                self.memory_guard_error = (
                    'job_memory_cap_exceeded' if job_breach else 'agent_memory_budget_exceeded',
                    'Job memory cap exceeded; increase --memory-budget-gb.' if job_breach
                    else 'The shared agent memory admission was exceeded or unavailable.')
            self._halt()
            with self.condition:
                self.last_error = (('Job memory cap exceeded; increase --memory-budget-gb.' if job_breach
                    else 'Agent memory guard stopped the owned model worker.')
                    + (' ' + budget['error'] if budget.get('error') else ''))
                self.model_state = 'failed'
                # Only the worker stops; the next request passes memory admission again.
                self.closed = self.shutdown_requested
                self.condition.notify_all()
                self._persist()

    def unload(self):
        with self.condition:
            if self.closed:
                raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
            if self.active_requests or self.cache_control_active or self.model_state == "unloading":
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
            self.shutdown_requested = True
        self._halt()
        self.execution.close()
        with self.condition:
            self.model_state = "unloaded"
            self._persist()

    def _halt(self):
        """Refuse new work, cancel running work and stop the owned worker."""
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self._stop_worker()


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, config):
        if config.get("api_key_sha256") is not None:
            from mlx_agent.inference_auth import validate_digest
            validate_digest(config["api_key_sha256"])
        self.controller = ModelWorker(config)
        self.config = config
        self.slots = threading.BoundedSemaphore(MAX_REQUESTS + 4)
        self.responses = collections.OrderedDict()
        self.responses_lock = threading.Lock()
        self.end_text = chat_completions.end_of_sequence(config["local_path"])
        self.tool_parser = chat_completions.vlm_tool_parser(config["local_path"]) if config.get("runtime") == "mlx-vlm" else None
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
                    self.controller.guard_agent_budget()
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
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0] not in ("127.0.0.1:{}".format(port), "localhost:{}".format(port)):
            raise GatewayError(403, "invalid_host", "Use the loopback endpoint address.")
        origins = self.headers.get_all("Origin")
        if origins is not None and (len(origins) != 1 or origins[0] not in ("http://127.0.0.1:{}".format(port), "http://localhost:{}".format(port))):
            raise GatewayError(403, "invalid_origin", "Cross-origin access is refused.")
        if control and not hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                               ("Bearer " + self.server.config["control_token"]).encode()):
            raise GatewayError(403, "control_denied", "Endpoint control authorization required.")
        protected = self.server.config.get("api_key_sha256") is not None or self.server.config.get("access_key_file") is not None
        if not control and protected and not (self.command == "GET" and self.path == "/health"):
            from mlx_agent.serving_access import read_access_key
            from mlx_agent.inference_auth import validate_digest, authorized_headers
            try:
                from mlx_agent.private_serve import identity
                if self.server.config.get("private_worker") and self.server.config["private_worker"] != identity():
                    raise ValueError("Protected worker configuration changed.")
                digest = validate_digest(self.server.config.get("api_key_sha256", self.server.config.get("access_key_sha256")))
                if self.server.config.get("access_key_file") is not None:
                    read_access_key(self.server.config["access_key_file"], digest)
            except (OSError, ValueError, KeyError):
                raise GatewayError(503, "access_key_unavailable", "The reviewed serving key is unavailable or changed.") from None
            if not authorized_headers(self.headers, digest):
                raise GatewayError(401, "access_denied", "Inference authorization required.")

    def _json(self, status, value):
        encoded = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
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

    def _client_gone(self):
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.connection, selectors.EVENT_READ)
                if not selector.select(0):
                    return False
            return not self.connection.recv(1, socket.MSG_PEEK)
        except OSError:
            return True

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

    def _memory_output(self, payload):
        measurement = getattr(self, '_measurement', None)
        if measurement is not None:
            measurement.observe_json(payload)
        controller = self.server.controller
        if controller.memory_materialized or controller.memory_transition_blocked or controller.memory_reservation is None:
            return
        try:
            value = json.loads(payload)
        except (ValueError,TypeError):
            return
        if isinstance(value,dict):
            controller._materialized_memory()

    def _relay(self, response, translate):
        """Answer a non-streamed request from the worker's translated chat completion."""
        if response.status != 200:
            raise self._worker_failure(response)
        payload = response.read(MAX_BODY + 1)
        try:
            if len(payload) > MAX_BODY:
                raise WorkerResponseError("Model worker response exceeds 16 MiB.")
            original = json.loads(payload)
            if getattr(self, '_measurement', None) is not None:
                self._measurement.observe(original)
            value = translate(original)
        except WorkerResponseError as error:
            raise GatewayError(502, "invalid_worker_response", str(error))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise GatewayError(502, "invalid_worker_response", "Model worker returned an invalid response.")
        self.server.controller._materialized_memory()
        self._json(200, value)
        self._inference_success = True
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
        finished = False
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
                    finished = True
                    break
                if getattr(self, '_measurement', None) is not None:
                    self._measurement.observe_json(data)
                events = translator.chunk(json.loads(data))
                self.server.controller._materialized_memory()
                self.wfile.write(b"".join(events))
                self.wfile.flush()
            else:
                raise WorkerResponseError("Model worker stream ended early.")
        except WorkerResponseError as error:
            self.wfile.write(b"".join(translator.fail(str(error))))
        except (ValueError, TypeError, AttributeError):
            self.wfile.write(b"".join(translator.fail("Model worker returned an invalid streamed response.")))
        self.wfile.flush()
        self._inference_success = finished

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

    def _dispatch_control(self, body):
        """Shared unload, cache, and policy routes for chat and task endpoints."""
        if self.path == "/_mlx/unload":
            self._json(200, self.server.controller.unload())
        elif self.path == "/_mlx/cache":
            if body:
                raise GatewayError(400, "invalid_body", "Cache reclaim requires an empty object.")
            self._json(200, self.server.controller.reclaim_cache())
        elif self.path == "/_mlx/policy":
            self._json(200, self.server.controller.configure(body))
        else:
            return False
        return True

    def do_POST(self):
        self._measurement = None
        self._inference_success = False
        connection = None
        leased = False
        response_started = False
        response = upstream = None
        submitted = False
        try:
            self._guard(control=self.path in ("/_mlx/unload", "/_mlx/policy", "/_mlx/cache"))
            if self.headers.get("Transfer-Encoding"):
                raise GatewayError(400, "invalid_body", "Use a bounded Content-Length request.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise GatewayError(400, "invalid_body", "Invalid Content-Length.")
            limit = MAX_BODY
            if self.server.config.get("private_worker"):
                from mlx_agent.private_serve import MAX_BODY as PRIVATE_MAX_BODY
                limit = PRIVATE_MAX_BODY
            if not 0 <= length <= limit:
                raise GatewayError(413, "body_too_large", "Request exceeds {} bytes.".format(limit))
            try:
                body = json.loads(self.rfile.read(length)) if length else {}
            except (ValueError, UnicodeError):
                raise GatewayError(400, "invalid_body", "Expected a JSON object.")
            if not isinstance(body, dict):
                raise GatewayError(400, "invalid_body", "Expected a JSON object.")
            if self._dispatch_control(body):
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
            requested = body.get("model", "default_model")
            if not isinstance(requested, str) or requested not in model_aliases(self.server.config):
                raise GatewayError(400, "model_not_allowed", "This endpoint serves only its configured local model.")
            if any(body.get(key) not in (None, "default_model") for key in ("adapter_path", "adapter", "draft_model", "draft_adapter")):
                raise GatewayError(400, "model_not_allowed", "Request-selected adapters and draft models are refused.")
            tokens = body.get("max_tokens", self.server.config["max_tokens"])
            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
                raise GatewayError(400, "invalid_body", "max_tokens must be a positive integer.")
            body["max_tokens"] = min(tokens, self.server.config["max_tokens"])
            runtime = self.server.config.get("runtime")
            # Splash answers only to its served name.
            body["model"] = self.server.config["splash_served_model" if runtime == "splash" else "local_path"]
            # Existing receipts predate the runtime field; recognize only the
            # exact native console entrypoint for those configurations.
            legacy_mlx = runtime is None and Path(self.server.config["worker_argv"][0]).name == "mlx_lm.server"
            if runtime == "mlx_lm" or legacy_mlx:
                chat_completions.decode_tool_arguments(body)
            elif runtime == "mlx-vlm" and self.server.tool_parser and body.get("tools"):
                body.setdefault("tool_parser", self.server.tool_parser)
            body = inference_defaults.apply(body, self.server.controller.request_defaults)
            self._measurement = self.server.controller.inference_telemetry.begin()
            port = self.server.controller.acquire(cancelled=self._client_gone)
            leased = True
            connection = self.server.controller.connection(timeout=120)
            connection.request("POST", "/v1/chat/completions" if translated else self.path, json.dumps(body),
                               dict({"Content-Type": "application/json"}, **({"Authorization": "Bearer " + self.server.controller.worker_key}
                                   if self.server.config.get("private_worker") or self.server.config.get("api_key_sha256") is not None else {})))
            submitted = True
            upstream = connection.sock
            upstream.settimeout(None)
            # MLX-LM writes nothing until an unstreamed answer is complete or a queued request starts.
            if not self._worker_replied(upstream):
                return
            response = connection.getresponse()
            if self.server.config.get("private_worker"):
                from mlx_agent.private_serve import PRIVATE_PROTOCOL
                if response.getheader("X-MLX-Agent-Transport") != PRIVATE_PROTOCOL:
                    raise GatewayError(503, "private_protocol_missing", "The private model worker protocol is unavailable.")

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
                finished = False
                while line := response.readline(MAX_BODY):
                    if line.strip() == b'data: [DONE]':
                        finished = True
                    if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                        self._memory_output(line[len(b'data: '):])
                    self.wfile.write(stream.line(line))
                    self.wfile.flush()
                self._inference_success = finished
                return
            if response.status == 200 and not body.get("stream"):
                payload = response.read(MAX_BODY + 1)
                if payload:
                    self._memory_output(payload)
                self.wfile.write(chat_completions.complete_body(payload, self.server.end_text)
                                 if len(payload) <= MAX_BODY else payload)
            while chunk := response.read1(65536):
                if body.get('stream'):
                    self._measurement.feed(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
            self._inference_success = response.status == 200 and (not body.get('stream') or self._measurement.stream_done)
        except GatewayError as error:
            if not self._client_gone():
                self._error(error)
        except (OSError, http.client.HTTPException):
            if not response_started and not self._client_gone():
                self._error(GatewayError(503, "worker_unavailable", "Model worker connection failed."))
        finally:
            if self._measurement is not None:
                self._measurement.finish(success=self._inference_success)
            try:
                if leased and submitted and (response is None or not response.isclosed()):
                    try:
                        deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
                        if upstream is not None:
                            upstream.settimeout(DRAIN_TIMEOUT_SECONDS)
                        if response is None:
                            response = connection.getresponse()
                        while not response.isclosed():
                            if upstream is not None:
                                upstream.settimeout(max(.001, deadline - time.monotonic()))
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Model reply did not drain.")
                            if not response.read1(65536):
                                break
                    except (OSError, http.client.HTTPException):
                        self.server.controller.execution_unresolved = True
                        try:
                            self.server.controller._stop_worker()
                        finally:
                            with self.server.controller.condition:
                                self.server.controller.model_state = "failed" if self.server.controller.process else "unloaded"
                                self.server.controller._persist()
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
