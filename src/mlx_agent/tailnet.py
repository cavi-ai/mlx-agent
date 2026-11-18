"""Read-only Tailnet preflight and endpoint selection for managed sharing."""
from __future__ import annotations

import hashlib
import http.client
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import struct
import threading
import time

from .inference_auth import authorized_headers, validate_digest
from .jit_serve import GatewayError, GatewayHandler, MAX_BODY, public_model_name
from .serve import _argv_matches, _pid_command, _pid_start, _read_receipt
from .service_registry import discover_services


class TailnetError(ValueError):
    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code, self.remediation = code, remediation


def _error(code, message, remedy):
    raise TailnetError(code, message, remedy)


def native_executable(path):
    """Resolve and admit only Mach-O executable files, never command shims."""
    thin = {b"\xce\xfa\xed\xfe": "<", b"\xcf\xfa\xed\xfe": "<",
            b"\xfe\xed\xfa\xce": ">", b"\xfe\xed\xfa\xcf": ">"}
    fat = {b"\xca\xfe\xba\xbe": (">", False), b"\xbe\xba\xfe\xca": ("<", False),
           b"\xca\xfe\xba\xbf": (">", True), b"\xbf\xba\xfe\xca": ("<", True)}
    try:
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            return None
        with resolved.open("rb") as binary:
            size = os.fstat(binary.fileno()).st_size
            def is_executable(offset):
                if offset < 0 or offset + 16 > size:
                    return False
                binary.seek(offset)
                header = binary.read(16)
                order = thin.get(header[:4])
                return order is not None and struct.unpack(order + "I", header[12:16])[0] == 2
            magic = binary.read(4)
            if magic in thin:
                return str(resolved) if is_executable(0) else None
            if magic not in fat:
                return None
            order, wide = fat[magic]
            count_raw = binary.read(4)
            if len(count_raw) != 4:
                return None
            count = struct.unpack(order + "I", count_raw)[0]
            if not 1 <= count <= 32:
                return None
            entry_size = 32 if wide else 20
            entries = binary.read(count * entry_size)
            if len(entries) != count * entry_size:
                return None
            for index in range(count):
                entry = entries[index * entry_size:(index + 1) * entry_size]
                offset, length = struct.unpack(order + ("QQ" if wide else "II"), entry[8:24 if wide else 16])
                if length < 16 or offset + length > size or not is_executable(offset):
                    return None
            return str(resolved)
    except (OSError, ValueError, struct.error):
        return None


def executable():
    candidates = ["/Applications/Tailscale.app/Contents/MacOS/Tailscale",
                  "/Applications/Tailscale.app/Contents/MacOS/tailscale",
                  "/Applications/Tailscale.app/Contents/MacOS/Tailscale CLI",
                  shutil.which("tailscale")]
    for candidate in candidates:
        resolved = native_executable(candidate) if candidate else None
        if resolved:
            return resolved
    _error("tailscale_missing", "A native Tailscale CLI is unavailable.", "Install Tailscale and its command-line integration.")


def read_json(binary, *arguments):
    try:
        result = subprocess.run([binary, *arguments], capture_output=True, timeout=5,
                                env=dict(os.environ, TAILSCALE_BE_CLI="1"), check=True)
        if len(result.stdout) > 2_000_000:
            raise ValueError("oversized response")
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise ValueError("unexpected response")
        return value
    except (OSError, subprocess.SubprocessError, ValueError):
        _error("tailnet_unavailable", "Tailscale status is unavailable.", "Check the local Tailscale client and retry.")


def node_status(binary):
    status = read_json(binary, "status", "--json", "--peers=false")
    if status.get("BackendState") != "Running":
        _error("tailscale_disconnected", "Tailscale is disconnected.", "Connect Tailscale and retry.")
    version = str(status.get("Version", "")).split("-", 1)[0]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version) or tuple(map(int, version.split("."))) < (1, 104, 2):
        _error("tailnet_unavailable", "The Tailscale service version is not qualified.", "Update Tailscale to version 1.104.2 or later.")
    tailnet, node = status.get("CurrentTailnet") or {}, status.get("Self") or {}
    if not tailnet.get("MagicDNSEnabled"):
        _error("tailscale_dns_required", "MagicDNS is required.", "Enable MagicDNS for the tailnet.")
    host = node.get("DNSName", "").removesuffix(".").lower()
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host) or "." not in host:
        _error("tailscale_dns_required", "The node has no usable MagicDNS name.", "Check the Tailscale connection and MagicDNS.")
    if host not in [str(item).removesuffix(".").lower() for item in status.get("CertDomains", [])]:
        _error("tailscale_https_required", "Tailscale HTTPS is unavailable for this node.", "Enable HTTPS certificates for the tailnet.")
    return host


def scopes(config):
    yield config
    foreground = config.get("Foreground", {})
    if not isinstance(foreground, dict):
        _error("tailnet_unavailable", "Unsupported Tailscale Serve status.", "Update Tailscale and retry.")
    for value in foreground.values():
        if not isinstance(value, dict):
            _error("tailnet_unavailable", "Unsupported Tailscale Serve session.", "Update Tailscale and retry.")
        yield value


def occupied(config, port):
    """Detect any TCP, web, or Funnel mapping for a port in every scope."""
    port = str(port)
    for scope in scopes(config):
        tcp, web, funnel = (scope.get(name) or {} for name in ("TCP", "Web", "AllowFunnel"))
        if not all(isinstance(item, dict) for item in (tcp, web, funnel)):
            _error("tailnet_unavailable", "Unsupported Tailscale Serve mapping.", "Update Tailscale and retry.")
        if port in tcp or any(str(key).rsplit(":", 1)[-1] == port for key in (*web, *funnel)):
            return True
    return False


def _matching_descriptor(endpoint, port, runtime, service_id):
    origin = "http://127.0.0.1:{}".format(port)
    return (isinstance(endpoint, dict)
            and set(endpoint) == {"origin", "api_base", "service_id", "runtime", "state", "auth_required"}
            and endpoint.get("origin") == origin and endpoint.get("api_base") == origin + "/v1"
            and endpoint.get("runtime") == runtime and runtime in ("mlx_lm", "mlx-vlm", "strata", "splash")
            and endpoint.get("state") in ("ready", "unknown") and endpoint.get("auth_required") is True
            and isinstance(service_id, str) and re.fullmatch(r"[0-9a-f]{64}", service_id) is not None
            and endpoint.get("service_id") == service_id)


def select_endpoint(port, receipts_dir=None):
    """Bind selection to one authenticated, live, receipt-owned JIT process."""
    services = discover_services([receipts_dir] if receipts_dir else []).get("services", [])
    matches = [item for item in services if item.get("port") == port and item.get("alive") and item.get("argv_match")]
    if len(matches) != 1:
        _error("tailnet_endpoint_ineligible", "Select one live managed endpoint.", "Start an authenticated managed chat endpoint and retry.")
    entry = matches[0]
    receipt = _read_receipt(Path(entry["receipt"]))
    try:
        digest = validate_digest(entry.get("api_key_sha256"))
        if (not receipt or not entry.get("jit") or not receipt.get("process_started_at")
                or receipt["process_started_at"] != entry.get("process_started_at")
                or not isinstance(entry.get("receipts_dir"), str)
                or not _matching_descriptor(entry.get("endpoint"), port, entry.get("runtime"), entry.get("service_id"))):
            raise ValueError()
        model = public_model_name(entry["path"]) if entry.get("path") else entry.get("repo")
        if not isinstance(model, str) or not model:
            raise ValueError()
    except (ValueError, TypeError):
        _error("tailnet_endpoint_ineligible", "The endpoint lacks a verified authenticated JIT identity.", "Start an authenticated managed chat endpoint and retry.")
    return {"endpoint_port": port, "model_id": model, "pid": entry["pid"],
            "process_started_at": entry["process_started_at"], "api_key_sha256": digest,
            "receipt": entry["receipt"], "receipts_dir": entry["receipts_dir"],
            "runtime": entry["runtime"], "service_id": entry["service_id"]}


def preview(port, https_port=8443, receipts_dir=None):
    if not 1 <= port <= 65535 or not 1 <= https_port <= 65535:
        raise ValueError("Ports must be between 1 and 65535.")
    endpoint = select_endpoint(port, receipts_dir)
    binary = executable()
    qualified_version(binary)
    host = node_status(binary)
    config = read_json(binary, "serve", "status", "--json")
    if occupied(config, https_port):
        _error("tailscale_port_conflict", "The selected Tailnet port is already in use.", "Choose another HTTPS port.")
    binding = dict(endpoint, host=host, https_port=https_port)
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"state": "ready", "api_base": "https://{}:{}/v1".format(host, https_port),
            "model_id": endpoint["model_id"], "endpoint_port": port,
            "https_port": https_port, "preview_hash": digest}, binding, binary


# The relay is deliberately independent of the regular gateway's Host rules.


def endpoint_matches(binding):
    from .model_serving import resolve_receipt_endpoint
    try:
        receipt = _read_receipt(Path(binding["receipt"]))
        if not receipt:
            return False
        pid = binding["pid"]
        if not (receipt.get("pid") == pid and receipt.get("port") == binding["endpoint_port"]
                and receipt.get("runtime") == binding["runtime"] and receipt.get("jit")
                and receipt.get("process_started_at") == binding["process_started_at"]
                and receipt.get("api_key_sha256") == binding["api_key_sha256"]
                and _pid_start(pid) == binding["process_started_at"]
                and _argv_matches(receipt, _pid_command(pid))):
            return False
        item = {"receipt": binding["receipt"], "receipts_dir": binding["receipts_dir"],
                "pid": pid, "port": binding["endpoint_port"], "runtime": binding["runtime"],
                "process_started_at": binding["process_started_at"], "api_key_sha256": binding["api_key_sha256"],
                "alive": True, "argv_match": True}
        endpoint = resolve_receipt_endpoint(item)
        return _matching_descriptor(endpoint, binding["endpoint_port"], binding["runtime"], binding["service_id"])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError):
        return False


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, binding):
        super().__init__(("127.0.0.1", 0), RelayHandler)
        self.binding = binding
        self.admitted = threading.Event()
        self.generation = threading.BoundedSemaphore(1)
        self.slots = threading.BoundedSemaphore(8)
        self.connections = set()
        self.connection_lock = threading.Lock()

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def close_upstreams(self):
        with self.connection_lock:
            for sock in tuple(self.connections):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()


MAX_SSE_LINE = 1024 * 1024


def _chat_response(raw, model):
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or "error" in value or not isinstance(value.get("choices"), list):
            raise ValueError()
        value["model"] = model
        return value
    except (ValueError, UnicodeError, TypeError):
        raise GatewayError(502, "upstream_error", "The shared model response was invalid.") from None


def _sse_line(line, model):
    ending = b"\r\n" if line.endswith(b"\r\n") else b"\n"
    content = line[:-len(ending)]
    if not content:
        return ending
    if content.startswith(b":"):
        return b": keepalive" + ending
    if not content.startswith(b"data:"):
        raise ValueError("Invalid event stream.")
    data = content[5:].lstrip(b" ")
    if data == b"[DONE]":
        return b"data: [DONE]" + ending
    try:
        value = json.loads(data)
        if (not isinstance(value, dict) or "error" in value or
                not (isinstance(value.get("choices"), list) or isinstance(value.get("usage"), dict))):
            raise ValueError()
        if "model" in value:
            value["model"] = model
        return b"data: " + json.dumps(value, separators=(",", ":")).encode() + ending
    except (ValueError, UnicodeError, TypeError):
        raise ValueError("Invalid event stream.") from None


class RelayHandler(GatewayHandler):
    server_version = "MLX-Tailnet"

    def _guard(self, control=False):
        binding = self.server.binding
        authority = "{}:{}".format(binding["host"], binding["https_port"])
        hosts = self.headers.get_all("Host", [])
        allowed = (authority, binding["host"]) if binding["https_port"] == 443 else (authority,)
        if len(hosts) != 1 or hosts[0] not in allowed or self.headers.get_all("Origin") is not None:
            raise GatewayError(403, "invalid_host", "The sharing authority is required.")
        if not self.server.admitted.is_set() or not endpoint_matches(binding):
            raise GatewayError(503, "route_unavailable", "The shared endpoint is unavailable.")
        if not authorized_headers(self.headers, binding["api_key_sha256"]):
            raise GatewayError(401, "access_denied", "Inference authorization required.")

    def do_GET(self):
        try:
            self._guard()
            if self.path != "/v1/models":
                raise GatewayError(404, "not_found", "Unknown endpoint.")
            self._json(200, {"object": "list", "data": [{"id": self.server.binding["model_id"],
                       "object": "model", "owned_by": "mlx-agent"}]})
        except GatewayError as error:
            self._error(error)

    def do_POST(self):
        try:
            self._guard()
            if self.path != "/v1/chat/completions":
                raise GatewayError(404, "not_found", "Unknown endpoint.")
            lengths = self.headers.get_all("Content-Length", [])
            if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
                    or len(lengths[0]) > 8 or self.headers.get_all("Transfer-Encoding") is not None):
                raise GatewayError(400, "invalid_body", "Use one bounded Content-Length.")
            length = int(lengths[0])
            if not 0 < length <= MAX_BODY:
                raise GatewayError(413, "body_too_large", "Request exceeds the body limit.")
            payload = self.rfile.read(length)
            try:
                value = json.loads(payload)
                if len(payload) != length or not isinstance(value, dict) or value.get("model") != self.server.binding["model_id"]:
                    raise ValueError()
            except (ValueError, UnicodeError):
                raise GatewayError(400, "invalid_body", "Select the shared model in a JSON object.") from None
            if not self.server.generation.acquire(blocking=False):
                raise GatewayError(429, "busy", "The remote generation slot is busy.")
            try:
                self._forward(payload, value.get("stream") is True)
            finally:
                self.server.generation.release()
        except GatewayError as error:
            self._error(error)

    def _forward(self, payload, streaming):
        if not self.server.admitted.is_set() or not endpoint_matches(self.server.binding):
            raise GatewayError(503, "route_unavailable", "The shared endpoint is unavailable.")
        headers = {"Content-Type": "application/json"}
        for name in ("Authorization", "x-api-key"):
            if self.headers.get(name) is not None:
                headers[name] = self.headers[name]
        conn = http.client.HTTPConnection("127.0.0.1", self.server.binding["endpoint_port"], timeout=10)
        started = False
        sock = None
        response = None
        try:
            conn.connect()
            sock = conn.sock
            with self.server.connection_lock:
                if not self.server.admitted.is_set():
                    raise GatewayError(503, "route_unavailable", "The shared endpoint is unavailable.")
                self.server.connections.add(sock)
            if not self.server.admitted.is_set() or not endpoint_matches(self.server.binding):
                raise GatewayError(503, "route_unavailable", "The shared endpoint is unavailable.")
            conn.request("POST", self.path, payload, headers)
            deadline = time.monotonic() + 300
            response = conn.getresponse()
            if not 200 <= response.status < 300:
                raise GatewayError(response.status, "upstream_error", "The shared model request failed.")
            if not streaming:
                sock.settimeout(max(0.01, deadline - time.monotonic()))
                raw = response.read(MAX_BODY + 1)
                if len(raw) > MAX_BODY:
                    raise GatewayError(502, "upstream_error", "The shared model response was invalid.")
                value = _chat_response(raw, self.server.binding["model_id"])
                self._json(response.status, value)
                started = True
                return
            self.send_response(response.status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            started = True
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                sock.settimeout(remaining)
                line = response.readline(MAX_SSE_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_SSE_LINE or not line.endswith(b"\n"):
                    break
                try:
                    clean = _sse_line(line, self.server.binding["model_id"])
                except ValueError:
                    break
                self.wfile.write(clean)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            if not started:
                raise GatewayError(503, "route_unavailable", "The shared endpoint is unavailable.") from None
        finally:
            with self.server.connection_lock:
                self.server.connections.discard(sock)
            if response is not None:
                response.close()
            conn.close()


def qualified_version(binary):
    import sys
    if sys.platform != "darwin":
        _error("tailnet_unavailable", "Managed sharing is unavailable on this platform.", "Use a qualified macOS Tailscale client.")
    version = read_json(binary, "version", "--json").get("majorMinorPatch", "")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version) or tuple(map(int, version.split("."))) < (1, 104, 2):
        _error("tailnet_unavailable", "This Tailscale version is not qualified for managed sharing.", "Update Tailscale to version 1.104.2 or later.")


def matching_session(config, binding, relay_port):
    """Require exactly one private foreground mapping, with no other port owner."""
    authority = "{}:{}".format(binding["host"], binding["https_port"])
    port = str(binding["https_port"])
    foreground = config.get("Foreground", {})
    if not isinstance(foreground, dict) or occupied({k: v for k, v in config.items() if k != "Foreground"}, port):
        return False
    found = 0
    for session in foreground.values():
        if not isinstance(session, dict):
            return False
        if not occupied(session, port):
            continue
        if (session.get("TCP", {}).get(port) != {"HTTPS": True}
                or session.get("Web", {}).get(authority) != {"Handlers": {"/": {"Proxy": "http://127.0.0.1:{}".format(relay_port)}}}
                or any(str(key).rsplit(":", 1)[-1] == port for key in session.get("AllowFunnel", {}))
                or any(key != authority and str(key).rsplit(":", 1)[-1] == port for key in session.get("Web", {}))):
            return False
        found += 1
    return found == 1


def owned_mapping_present(config, binding, relay_port):
    authority = "{}:{}".format(binding["host"], binding["https_port"])
    target = "http://127.0.0.1:{}".format(relay_port)
    for scope in scopes(config):
        web = scope.get("Web") or {}
        if not isinstance(web, dict):
            _error("tailnet_unavailable", "Unsupported Tailscale Serve mapping.", "Check Tailscale Serve status.")
        website = web.get(authority, {})
        if not isinstance(website, dict) or not isinstance(website.get("Handlers", {}), dict):
            _error("tailnet_unavailable", "Unsupported Tailscale Serve mapping.", "Check Tailscale Serve status.")
        handlers = website.get("Handlers", {})
        if any(isinstance(value, dict) and value.get("Proxy") == target for value in handlers.values()):
            return True
    return False


def state_root():
    from .service_registry import registry_root
    return registry_root().parent / "tailnet"


def read_state():
    import stat
    path = state_root() / "session.json"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_size > 16384:
                raise ValueError()
            value = json.load(source)
            if not isinstance(value, dict):
                raise ValueError()
            return value
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError):
        _error("tailnet_unavailable", "The sharing session record is unavailable.", "Inspect the local sharing session and retry.")


def owner_matches(record):
    from .compute import process_birth_identity
    from .service_registry import _process_arguments
    try:
        return (isinstance(record["pid"], int) and record["pid"] > 1 and
                bool(record["process_started_at"]) and
                process_birth_identity(record["pid"]) == record["process_started_at"] and
                _process_arguments(record["pid"]) == record["argv"])
    except (KeyError, OSError, ValueError, TypeError):
        return False


def stale_cleaned(record):
    """Only retire a dead supervisor after its guard and mapping are gone."""
    from .compute import process_birth_identity
    from .service_registry import _process_arguments
    try:
        guard = record["guard_pid"]
        alive = (process_birth_identity(guard) == record["guard_started_at"] and
                 _process_arguments(guard) == record["guard_argv"])
        return (not alive and not owned_mapping_present(read_json(record["binary"], "serve", "status", "--json"),
                                                        record["binding"], record["relay_port"]))
    except (KeyError, OSError, ValueError, TypeError):
        return False


def status():
    record = read_state()
    if record is None:
        return {"state": "off"}
    if not owner_matches(record):
        if stale_cleaned(record):
            return {"state": "off"}
        _error("tailnet_unavailable", "The recorded sharing supervisor is no longer active.", "Check Tailscale Serve status and retry.")
    try:
        binding, binary, port = record["binding"], record["binary"], record["relay_port"]
        if (not endpoint_matches(binding) or node_status(binary) != binding["host"]
                or not matching_session(read_json(binary, "serve", "status", "--json"), binding, port)):
            raise ValueError()
        return record["public"]
    except (KeyError, ValueError, TypeError):
        _error("tailnet_unavailable", "The managed sharing session changed.", "Stop sharing and check Tailscale status.")


def stop(expected_share_id=None):
    import signal
    record = read_state()
    if record is None:
        return {"state": "off"}
    if expected_share_id is not None and record.get("public", {}).get("share_id") != expected_share_id:
        _error("tailnet_session_changed", "The sharing session changed.", "Refresh the sharing status and retry.")
    if not owner_matches(record):
        if stale_cleaned(record):
            return {"state": "off"}
        _error("tailnet_unavailable", "The recorded sharing supervisor is no longer active.", "Check Tailscale Serve status and retry.")
    os.kill(record["pid"], signal.SIGTERM)
    for _ in range(100):
        time.sleep(0.1)
        if not owner_matches(record):
            if owned_mapping_present(read_json(record["binary"], "serve", "status", "--json"), record["binding"], record["relay_port"]):
                _error("tailnet_unavailable", "The sharing port remains occupied after stopping.", "Inspect Tailscale Serve status.")
            return {"state": "off"}
    _error("tailnet_unavailable", "The sharing supervisor did not exit.", "Inspect the local sharing session.")


def serve_child(binary, https_port, relay_port):
    """Do not start Serve until the supervisor has bound this child's identity."""
    import sys
    os.write(sys.stdout.fileno(), b"R")
    if os.read(sys.stdin.fileno(), 1) != b"1":
        return 2
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(null, 1)
    os.close(null)
    command = [binary, "serve", "--yes", "--https={}".format(https_port),
               "http://127.0.0.1:{}".format(relay_port)]
    os.execve(binary, command, dict(os.environ, TAILSCALE_BE_CLI="1"))


def guard(binary, https_port, relay_port, listener_fd):
    """Hold the relay port and supervise an acknowledged foreground Serve child."""
    import select
    import signal
    import sys
    from .compute import process_birth_identity
    from .service_registry import _process_arguments
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())
    helper = [sys.executable, "-m", "mlx_agent.tailnet", "--serve-child", binary,
              str(https_port), str(relay_port)]
    expected = [binary, "serve", "--yes", "--https={}".format(https_port),
                "http://127.0.0.1:{}".format(relay_port)]
    child = subprocess.Popen(helper, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
    try:
        readable, _, _ = select.select([child.stdout], [], [], 2.5)
        if not readable or os.read(child.stdout.fileno(), 1) != b"R":
            raise RuntimeError("The sharing helper did not become ready.")
        child.stdout.close()
        birth = process_birth_identity(child.pid)
        argv = _process_arguments(child.pid)
        print(json.dumps({"pid": child.pid, "birth": birth, "argv": argv, "expected": expected}), flush=True)
        acknowledged = False
        while child.poll() is None and not stopping.is_set():
            readable, _, _ = select.select([sys.stdin.buffer], [], [], 0.5)
            if readable:
                value = os.read(sys.stdin.fileno(), 1)
                if not value:
                    break
                if value == b"1" and not acknowledged:
                    child.stdin.write(b"1")
                    child.stdin.flush()
                    child.stdin.close()
                    acknowledged = True
    finally:
        if not child.stdout.closed:
            child.stdout.close()
        if not child.stdin.closed:
            child.stdin.close()
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        os.close(listener_fd)
    return child.returncode


def run_session(public, binding, binary, emit_ready, watch_parent=False, share_id=None, parent=None):
    import fcntl
    import select
    import signal
    import sys
    import tempfile
    import uuid
    from .compute import process_birth_identity
    from .service_registry import _process_arguments
    from .transactions import _assert_safe_directory, _atomic_in_directory

    parent = os.getppid() if parent is None else parent
    if watch_parent and (parent <= 1 or os.getppid() != parent):
        _error("tailnet_unavailable", "The supervising parent has exited.", "Start sharing from the application again.")
    root = state_root()
    _assert_safe_directory(root, create=True)
    fd = os.open(root / "session.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _error("tailnet_unavailable", "A sharing session is already active.", "Stop the current session first.")
        existing = read_state()
        if existing is not None:
            if not owner_matches(existing) and stale_cleaned(existing):
                (root / "session.json").unlink()
            else:
                _error("tailnet_unavailable", "A previous sharing record remains.", "Inspect Tailscale Serve status before retrying.")
        relay = RelayServer(binding)
        guard_process = None
        child_identity = None
        worker = None
        stopping = threading.Event()
        previous = {}
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, lambda *_: stopping.set())
            if not endpoint_matches(binding) or node_status(binary) != binding["host"]:
                _error("tailnet_preview_changed", "The reviewed endpoint or node changed.", "Preview sharing again.")
            if occupied(read_json(binary, "serve", "status", "--json"), binding["https_port"]):
                _error("tailscale_port_conflict", "The selected Tailnet port is already in use.", "Choose another HTTPS port.")
            with tempfile.TemporaryFile() as errors:
                guard_process = subprocess.Popen([sys.executable, "-m", "mlx_agent.tailnet", "--guard", binary,
                    str(binding["https_port"]), str(relay.server_port), str(relay.socket.fileno())],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, pass_fds=(relay.socket.fileno(),),
                    env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent)))
                if not select.select([guard_process.stdout], [], [], 5)[0]:
                    _error("tailnet_unavailable", "The sharing guard did not start.", "Check Tailscale Serve status and retry.")
                child_identity = json.loads(guard_process.stdout.readline(16384))
                if (not child_identity.get("birth") or not child_identity.get("argv") or
                        process_birth_identity(child_identity["pid"]) != child_identity["birth"] or
                        _process_arguments(child_identity["pid"]) != child_identity["argv"]):
                    _error("tailnet_unavailable", "The sharing child identity is unavailable.", "Check Tailscale Serve status and retry.")
                guard_process.stdin.write(b"1")
                guard_process.stdin.flush()
                for _ in range(100):
                    if guard_process.poll() is not None:
                        errors.seek(0)
                        detail = errors.read(4096)
                        if b"listener already exists" in detail:
                            _error("tailscale_port_conflict", "The selected Tailnet port is already in use.", "Choose another HTTPS port.")
                        _error("tailnet_unavailable", "The Tailscale sharing session exited.", "Check Tailscale Serve status and retry.")
                    if matching_session(read_json(binary, "serve", "status", "--json"), binding, relay.server_port):
                        break
                    if stopping.wait(0.1):
                        _error("tailnet_unavailable", "Sharing was stopped during startup.", "Start sharing again if needed.")
                else:
                    _error("tailnet_unavailable", "The Tailscale mapping could not be confirmed.", "Check Tailscale Serve status and retry.")
                if not endpoint_matches(binding) or (watch_parent and os.getppid() != parent):
                    _error("tailnet_preview_changed", "The endpoint or supervising parent changed.", "Preview sharing again.")
                identity = process_birth_identity(os.getpid())
                argv = _process_arguments(os.getpid())
                if not identity or not argv:
                    _error("tailnet_unavailable", "The sharing supervisor identity is unavailable.", "Retry sharing.")
                sharing = dict(public, state="sharing", share_id=str(uuid.UUID(share_id)) if share_id else str(uuid.uuid4()))
                sharing.pop("preview_hash", None)
                guard_started_at = process_birth_identity(guard_process.pid)
                guard_argv = _process_arguments(guard_process.pid)
                if not guard_started_at or not guard_argv:
                    _error("tailnet_unavailable", "The sharing guard identity is unavailable.", "Retry sharing.")
                record = {"guard_pid": guard_process.pid, "guard_started_at": guard_started_at, "guard_argv": guard_argv,
                          "pid": os.getpid(), "process_started_at": identity, "argv": argv,
                          "binding": binding, "binary": binary, "relay_port": relay.server_port, "public": sharing}
                _atomic_in_directory(root, "session.json", json.dumps(record).encode(), 0o600)
                relay.admitted.set()
                worker = threading.Thread(target=relay.serve_forever, daemon=True)
                worker.start()
                emit_ready(sharing)
                while not stopping.wait(1):
                    if guard_process.poll() is not None or (watch_parent and os.getppid() != parent):
                        break
                    try:
                        if (not endpoint_matches(binding) or node_status(binary) != binding["host"] or
                                not matching_session(read_json(binary, "serve", "status", "--json"), binding, relay.server_port)):
                            break
                    except TailnetError:
                        break
        finally:
            try:
                relay.admitted.clear()
                relay.close_upstreams()
                if worker:
                    relay.shutdown()
                    worker.join(timeout=3)
                if guard_process:
                    try:
                        guard_process.stdin.close()
                    except BrokenPipeError:
                        pass
                    try:
                        guard_process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        guard_process.terminate()
                        guard_process.wait(timeout=5)
                    guard_process.stdout.close()
                    if child_identity:
                        child_pid = child_identity["pid"]
                        def same_child():
                            return (process_birth_identity(child_pid) == child_identity["birth"] and
                                    _process_arguments(child_pid) in (child_identity["argv"], child_identity["expected"]))
                        if same_child():
                            os.kill(child_pid, signal.SIGINT)
                            for _ in range(50):
                                if not same_child():
                                    break
                                time.sleep(0.1)
                            else:
                                if same_child():
                                    os.kill(child_pid, signal.SIGKILL)
                                for _ in range(50):
                                    if not same_child():
                                        break
                                    time.sleep(0.1)
                                else:
                                    _error("tailnet_unavailable", "The sharing child did not exit.", "Inspect Tailscale Serve status.")
                if guard_process and owned_mapping_present(read_json(binary, "serve", "status", "--json"), binding, relay.server_port):
                    _error("tailnet_unavailable", "The sharing port remains occupied after cleanup.", "Inspect Tailscale Serve status.")
                existing = read_state()
                if existing and existing.get("pid") == os.getpid():
                    (root / "session.json").unlink()
            finally:
                relay.server_close()
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
    finally:
        os.close(fd)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5 and sys.argv[1] == "--serve-child":
        raise SystemExit(serve_child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
    if len(sys.argv) == 6 and sys.argv[1] == "--guard":
        raise SystemExit(guard(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])))
    raise SystemExit(2)
