"""Authenticated loopback front door that routes requests to discovered model routes by name."""
from __future__ import annotations

import argparse
import http.client
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading
import time

from .jit_serve import GatewayError, GatewayHandler, MAX_BODY, MAX_REQUESTS, public_model_name
from .http_relay import ClientDisconnected, RelayLimits, Reply

CHAT_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses")
EMBEDDINGS_PATH = "/v1/embeddings"
SYSTEMONE_PATH = "/v1/systemone"
# Typed-route endpoints and the task kind a route must serve to answer them.
TASK_PATHS = {EMBEDDINGS_PATH: "embedding", SYSTEMONE_PATH: "classification"}
REFRESH_SECONDS = 5
REFRESH_WAIT_SECONDS = 12
REFRESH_RETRY_SECONDS = 1
MAX_STALE_SECONDS = 30
OPERATION = "serve-front-door"


class FrontDoorError(ValueError):
    """A startup step that failed, with the recovery a host can show."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code, self.remediation = code, remediation


def route_table(services, digest):
    """Stable name to route for live routes verifying `digest`; a name served twice maps to None."""
    table = {}
    for entry in services:
        port = entry.get("port")
        if not (entry.get("alive") and entry.get("argv_match") and entry.get("api_key_sha256") == digest
                and (entry.get("jit") or entry.get("runtime") == "tasks") and type(port) is int and 1 <= port <= 65535):
            continue
        name = public_model_name(entry["path"]) if entry.get("path") else entry.get("repo")
        if isinstance(name, str) and name:
            table[name] = None if name in table else entry
    return table


class FrontDoorServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, config, discover):
        self.relay_limits = RelayLimits.from_environment()
        super().__init__(address, FrontDoorHandler)
        self.config, self.discover = config, discover
        self.digest = config.get("api_key_sha256") or config["access_key_sha256"]
        self.slots = threading.BoundedSemaphore(MAX_REQUESTS + 4)
        self.condition = threading.Condition()
        self.table, self.loaded_at = None, None
        self.attempted_at = float("-inf")
        self.refreshing = self.degraded = False

    def routes(self, *, wait=False, refresh=False):
        with self.condition:
            now = time.monotonic()
            elapsed = now - self.attempted_at
            interval = REFRESH_RETRY_SECONDS if refresh else REFRESH_SECONDS
            if not self.refreshing and elapsed >= interval:
                self.refreshing = True
                threading.Thread(target=self._refresh_routes, daemon=True).start()
            expired = self.table is None or now - self.loaded_at > MAX_STALE_SECONDS
            if self.refreshing and (wait or expired):
                self.condition.wait_for(lambda: not self.refreshing, timeout=REFRESH_WAIT_SECONDS)
                now = time.monotonic()
            if self.table is None or now - self.loaded_at > MAX_STALE_SECONDS:
                raise GatewayError(503, "route_discovery_unavailable", "Model route discovery is unavailable.")
            return self.table

    def _refresh_routes(self):
        table = None
        degraded = True
        try:
            found = self.discover()
            routes = route_table(found["services"], self.digest)
            degraded = bool(found.get("errors"))
            if routes or not degraded:
                table = routes
        except (OSError, ValueError, TypeError, KeyError):
            pass
        finally:
            with self.condition:
                self.attempted_at = time.monotonic()
                if table is not None:
                    self.table = table
                    self.loaded_at = self.attempted_at
                self.degraded = degraded
                self.refreshing = False
                self.condition.notify_all()

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
                    for name, route in sorted(self.server.routes().items()) if route is not None]})
            else:
                raise GatewayError(404, "not_found", "Unknown endpoint.")
        except GatewayError as error:
            if not self._client_gone():
                self._error(error)

    def do_POST(self):
        try:
            self._guard()
            if self.path not in CHAT_PATHS and self.path not in TASK_PATHS:
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
            self._forward(model, payload)
        except GatewayError as error:
            if not self._client_gone():
                self._error(error)

    def _route(self, model, *, refresh=False):
        if not isinstance(model, str):
            raise GatewayError(400, "invalid_body", "Name a model listed at /v1/models.")
        table = self.server.routes(wait=refresh, refresh=refresh)
        if model not in table and not refresh:
            table = self.server.routes(wait=True)
        if model not in table:
            if self.server.degraded or self.server.refreshing:
                raise GatewayError(503, "route_discovery_unavailable", "Model route discovery is incomplete.")
            raise GatewayError(404, "model_not_found", "No route with this key serves that model.")
        route = table[model]
        if route is None:
            raise GatewayError(409, "ambiguous_model", "More than one route serves that model name.")
        typed, kind = route.get("runtime") == "tasks", TASK_PATHS.get(self.path)
        if (kind is not None) != typed or (typed and kind not in route.get("task_kinds", ())):
            raise GatewayError(400, "unsupported_endpoint", "That model's route does not serve this endpoint.")
        return route

    def _connect(self, model):
        port = self._route(model)["port"]
        for attempt in range(2):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                connection.connect()
                return connection
            except OSError:
                connection.close()
                if attempt == 0 and not self._client_gone():
                    next_port = self._route(model, refresh=True)["port"]
                    if next_port != port:
                        port = next_port
                        continue
                break
        raise GatewayError(503, "route_unavailable", "The model route is unavailable.")

    def _forward(self, model, payload):
        # The guard admitted exactly one credential; the route verifies the same key digest.
        headers = {"Content-Type": "application/json"}
        for name in ("Authorization", "x-api-key"):
            if self.headers.get(name) is not None:
                headers[name] = self.headers[name]
        connection = None
        response = None
        started = False
        try:
            connection = self._connect(model)
            if self._client_gone():
                return
            connection.request("POST", self.path, payload, headers)
            response = Reply(connection, self.server.relay_limits, client=self.connection).getresponse()
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
        except ClientDisconnected:
            pass
        except TimeoutError:
            if not started and not self._client_gone():
                raise GatewayError(504, "route_timeout", "The model route reply timed out.") from None
        except (OSError, http.client.HTTPException):
            if not started and not self._client_gone():
                raise GatewayError(503, "route_unavailable", "The model route is unavailable.") from None
        finally:
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()


def _listen(args, parent):
    """Bind the listener, naming the startup step that failed."""
    from .inference_auth import configuration_digest
    from .service_registry import discover_services
    if args.watch_parent and parent <= 1:
        raise FrontDoorError("front_door_parent_unavailable", "--watch-parent requires a live parent.",
                             "Start the front door from the process that owns it.")
    try:
        if args.api_key_file:
            path = str(Path(args.api_key_file).absolute())
            config = {"access_key_file": path, "access_key_sha256": configuration_digest(key_file=path)}
        else:
            config = {"api_key_sha256": configuration_digest(key_sha256=args.api_key_sha256)}
    except (OSError, ValueError):
        raise FrontDoorError("front_door_key_unavailable", "The front door API key is unavailable.",
                             "Pass an owned private key file or a lowercase SHA-256 key digest.") from None
    receipts = [str(Path(value).absolute()) for value in args.receipts_dir]
    try:
        return FrontDoorServer(("127.0.0.1", args.port), config,
                               lambda: discover_services(receipts, routing_only=True))
    except ValueError as error:
        raise FrontDoorError("front_door_invalid_limits", str(error), "Correct the relay timeout settings.") from None
    except OSError:
        raise FrontDoorError("front_door_port_unavailable", "Port {} is unavailable on 127.0.0.1.".format(args.port),
                             "Choose another port or stop the process using it.") from None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlx-agent serve front-door", description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    key = parser.add_mutually_exclusive_group(required=True)
    key.add_argument("--api-key-file")
    key.add_argument("--api-key-sha256")
    parser.add_argument("--receipts-dir", action="append", default=[])
    parser.add_argument("--watch-parent", action="store_true", help="stop when the launching process exits")
    parser.add_argument("--json", action="store_true", help="print one result when listening or when startup fails")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    from .contracts import ResultEnvelope
    parent = os.getppid()
    try:
        server = _listen(args, parent)
    except FrontDoorError as error:
        if not args.json:
            raise
        print(json.dumps(ResultEnvelope.fail(OPERATION, error.code, str(error), error.remediation).to_dict(), indent=2))
        return 2

    def watch_parent():
        while os.getppid() == parent:
            time.sleep(.25)
        server.shutdown()

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        if args.watch_parent:
            threading.Thread(target=watch_parent, daemon=True).start()
        if args.json:
            # The listener already accepts connections; a host reads this line as readiness.
            print(json.dumps(ResultEnvelope.ok(OPERATION, {
                "state": "listening", "port": args.port, "api_base": "http://127.0.0.1:{}/v1".format(args.port),
                "pid": os.getpid()}).to_dict()), flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0
