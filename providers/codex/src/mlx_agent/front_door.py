"""Authenticated loopback front door that routes requests to discovered model routes by name."""
from __future__ import annotations

import argparse
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading
import time

from .jit_serve import GatewayError, GatewayHandler, MAX_BODY, MAX_REQUESTS, public_model_name

CHAT_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses")
EMBEDDINGS_PATH = "/v1/embeddings"
REFRESH_SECONDS = 5


def route_table(services, digest):
    """Stable name to route for live routes verifying `digest`; a name served twice maps to None."""
    table = {}
    for entry in services:
        if not (entry.get("alive") and entry.get("argv_match") and entry.get("api_key_sha256") == digest
                and (entry.get("jit") or entry.get("runtime") == "tasks") and isinstance(entry.get("port"), int)):
            continue
        name = public_model_name(entry["path"]) if entry.get("path") else entry.get("repo")
        if isinstance(name, str) and name:
            table[name] = None if name in table else entry
    return table


class FrontDoorServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, config, discover):
        super().__init__(address, FrontDoorHandler)
        self.config, self.discover = config, discover
        self.digest = config.get("api_key_sha256") or config["access_key_sha256"]
        self.slots = threading.BoundedSemaphore(MAX_REQUESTS + 4)
        self.lock = threading.Lock()
        self.table, self.loaded_at = {}, None

    def routes(self, refresh=False):
        with self.lock:
            if refresh or self.loaded_at is None or time.monotonic() - self.loaded_at >= REFRESH_SECONDS:
                self.table = route_table(self.discover().get("services", []), self.digest)
                self.loaded_at = time.monotonic()
            return self.table

    def forget(self):
        with self.lock:
            self.loaded_at = None

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


class FrontDoorHandler(GatewayHandler):
    server_version = "MLX-Front-Door"

    def do_GET(self):
        try:
            self._guard()
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    dict({"id": name, "object": "model", "owned_by": "mlx-agent"},
                         **({"task_kinds": route["task_kinds"]} if route.get("task_kinds") else {}))
                    for name, route in sorted(self.server.routes(refresh=True).items()) if route is not None]})
            else:
                raise GatewayError(404, "not_found", "Unknown endpoint.")
        except GatewayError as error:
            self._error(error)

    def do_POST(self):
        try:
            self._guard()
            if self.path not in CHAT_PATHS + (EMBEDDINGS_PATH,):
                raise GatewayError(404, "not_found", "Unknown inference endpoint.")
            if self.headers.get("Transfer-Encoding"):
                raise GatewayError(400, "invalid_body", "Use a bounded Content-Length request.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise GatewayError(400, "invalid_body", "Invalid Content-Length.") from None
            if not 0 <= length <= MAX_BODY:
                raise GatewayError(413, "body_too_large", "Request exceeds {} bytes.".format(MAX_BODY))
            payload = self.rfile.read(length)
            try:
                model = json.loads(payload).get("model")
            except (ValueError, UnicodeError, AttributeError):
                raise GatewayError(400, "invalid_body", "Expected a JSON object.") from None
            self._forward(self._route(model)["port"], payload)
        except GatewayError as error:
            self._error(error)

    def _route(self, model):
        if not isinstance(model, str):
            raise GatewayError(400, "invalid_body", "Name a model listed at /v1/models.")
        table = self.server.routes()
        if model not in table:
            table = self.server.routes(refresh=True)
        if model not in table:
            raise GatewayError(404, "model_not_found", "No route with this key serves that model.")
        route = table[model]
        if route is None:
            raise GatewayError(409, "ambiguous_model", "More than one route serves that model name.")
        typed = route.get("runtime") == "tasks"
        if (self.path == EMBEDDINGS_PATH) != typed or (typed and "embedding" not in route.get("task_kinds", ())):
            raise GatewayError(400, "unsupported_endpoint", "That model's route does not serve this endpoint.")
        return route

    def _forward(self, port, payload):
        # The guard admitted exactly one credential; the route verifies the same key digest.
        headers = {"Content-Type": "application/json"}
        for name in ("Authorization", "x-api-key"):
            if self.headers.get(name) is not None:
                headers[name] = self.headers[name]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        started = False
        try:
            connection.request("POST", self.path, payload, headers)
            connection.sock.settimeout(None)
            response = connection.getresponse()
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            if response.status == 401:
                self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            started = True
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            if not started:
                self.server.forget()
                raise GatewayError(503, "route_unavailable", "The model route is unavailable.") from None
        finally:
            connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlx-agent serve front-door", description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    key = parser.add_mutually_exclusive_group(required=True)
    key.add_argument("--api-key-file")
    key.add_argument("--api-key-sha256")
    parser.add_argument("--receipts-dir", action="append", default=[])
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    from .inference_auth import configuration_digest
    from .service_registry import discover_services
    if args.api_key_file:
        path = str(Path(args.api_key_file).absolute())
        config = {"access_key_file": path, "access_key_sha256": configuration_digest(key_file=path)}
    else:
        config = {"api_key_sha256": configuration_digest(key_sha256=args.api_key_sha256)}
    receipts = [str(Path(value).absolute()) for value in args.receipts_dir]
    server = FrontDoorServer(("127.0.0.1", args.port), config, lambda: discover_services(receipts))
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        server.serve_forever()
    finally:
        server.server_close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0
