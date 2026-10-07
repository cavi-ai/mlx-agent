"""Loopback OpenAI gateway with one owned, lazily started local-model worker.

No runtime imports or downloads. Inference goes to the existing recipe's
server process; a request lease lasts through the final streamed byte.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import json
import os
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
from mlx_agent.transactions import _atomic_in_directory

MAX_BODY = 16 * 1024 * 1024
MAX_REQUESTS = 32


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
            self.model_state = "unloaded"
            self._persist()

    def status(self):
        with self.condition:
            self._refresh()
            return {"model_state": self.model_state, "active_requests": self.active_requests,
                    "worker_pid": self.process.pid if self.process and self.process.poll() is None else None,
                    "last_error": self.last_error}

    def acquire(self):
        with self.condition:
            if not self.condition.wait_for(lambda: self.model_state != "unloading" or self.closed, timeout=10):
                raise GatewayError(503, "model_unloading", "Model unload has not finished.")
            if self.closed:
                raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
            self._refresh()
            if self.active_requests >= MAX_REQUESTS:
                raise GatewayError(503, "endpoint_busy", "Too many inference requests.")
            self.active_requests += 1
            starter = self.model_state not in ("loaded", "loading")
            if starter:
                self.model_state = "loading"
                self.last_error = None
        try:
            if starter:
                self._start()
            else:
                with self.condition:
                    self.condition.wait_for(lambda: self.model_state != "loading" or self.closed, timeout=65)
            with self.condition:
                if self.closed or self.model_state != "loaded":
                    raise GatewayError(503, "model_load_failed", self.last_error or "Model is not ready.")
                return self.worker_port
        except BaseException:
            self.release()
            raise

    def _start(self):
        try:
            if model_fingerprint(self.config["local_path"]) != self.config["fingerprint"]:
                raise GatewayError(503, "model_changed", "Model files changed. Stop and review a fresh serve plan.")
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
            deadline = time.monotonic() + 60
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
                raise GatewayError(503, "model_load_failed", "Model worker did not become ready in 60 seconds.")
            if model_fingerprint(self.config["local_path"]) != self.config["fingerprint"]:
                raise GatewayError(503, "model_changed", "Model files changed while loading.")
            with self.condition:
                if self.closed:
                    raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
                self.model_state = "loaded"
                self._persist()
                self.condition.notify_all()
        except BaseException as error:
            self._stop_worker()
            with self.condition:
                self.model_state = "failed"
                self.last_error = str(error)
                self._persist()
                self.condition.notify_all()
            if isinstance(error, GatewayError):
                raise
            raise GatewayError(503, "model_load_failed", str(error)) from error

    def release(self):
        with self.condition:
            self.active_requests -= 1
            self.condition.notify_all()

    def _stop_worker(self):
        with self.condition:
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        with self.condition:
            self.process = None
            if self.log not in (None, subprocess.DEVNULL):
                self.log.close()
            self.log = None

    def unload(self):
        with self.condition:
            if self.closed:
                raise GatewayError(503, "endpoint_stopping", "Endpoint is stopping.")
            if self.active_requests or self.model_state == "unloading":
                raise GatewayError(409, "model_busy", "Finish active requests before unloading.")
            self.model_state = "unloading"
        try:
            self._stop_worker()
            with self.condition:
                self.model_state = "unloaded"
                self.last_error = None
                self._persist()
                self.condition.notify_all()
        except BaseException:
            with self.condition:
                self.model_state = "failed"
                self.condition.notify_all()
            raise
        return self.status()

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
        super().__init__(address, GatewayHandler)

    def server_bind(self):
        # HTTPServer normally performs reverse DNS here. This gateway binds
        # a numeric loopback address and must start without network discovery.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

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
        self._json(error.status, {"error": {"code": error.code, "message": str(error)}})

    def do_GET(self):
        try:
            self._guard(control=self.path == "/_mlx/status")
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": self.server.config["model"], "object": "model"}]})
            elif self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/_mlx/status":
                self._json(200, self.server.controller.status())
            else:
                raise GatewayError(404, "not_found", "Unknown endpoint.")
        except GatewayError as error:
            self._error(error)

    def do_POST(self):
        connection = None
        leased = False
        response_started = False
        try:
            self._guard(control=self.path == "/_mlx/unload")
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
            if self.path not in ("/v1/chat/completions", "/v1/completions"):
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
            port = self.server.controller.acquire()
            leased = True
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            connection.request("POST", self.path, json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.send_header("Connection", "close")
            self.end_headers()
            response_started = True
            self.close_connection = True
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
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    finally:
        server.controller.close()
        server.server_close()


if __name__ == "__main__":
    main()
