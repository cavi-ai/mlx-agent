"""Supervised MLX-LM inference over an inherited Unix listener, never TCP."""
from __future__ import annotations

import argparse
import errno
import hashlib
import inspect
import io
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from http.server import ThreadingHTTPServer
from socketserver import BaseServer

PRIVATE_PROTOCOL = "inherited-unix-v1"
MAX_BODY = 1024 * 1024


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout=120):
        super().__init__("127.0.0.1", timeout=timeout)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        try:
            self.sock.connect(self.path)
        except BaseException:
            self.sock.close()
            self.sock = None
            raise


def worker_command(native, model, descriptor):
    python = runtime_python(native[0])
    return [python, str(Path(__file__).resolve()), "--runtime-child", "--model", model,
            "--socket-fd", str(descriptor), "--"] + native[1:]


class PrivateEndpoint:
    """One supervisor-owned listener; no new TCP port or credential in argv."""
    def __init__(self):
        # Keep sockaddr_un below Darwin's length limit even with a long TMPDIR.
        self.directory = tempfile.TemporaryDirectory(prefix="mlx-worker-", dir="/tmp")
        self.path = str(Path(self.directory.name) / "worker.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.listener.bind(self.path)
            os.chmod(self.path, 0o600)
            self.listener.listen(16)
        except BaseException:
            self.close()
            raise

    def connection(self, timeout=120):
        return UnixHTTPConnection(self.path, timeout)

    def close(self):
        self.listener.close()
        self.directory.cleanup()


def identity():
    return {"protocol": PRIVATE_PROTOCOL,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def private_server_class(descriptor):
    listener = socket.socket(fileno=descriptor)
    try:
        if listener.family != socket.AF_UNIX or listener.type != socket.SOCK_STREAM or not listener.getsockname():
            raise ValueError("A bound Unix stream socket inherited from the supervisor is required.")
        try:
            if not listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN):
                raise ValueError("The inherited Unix socket is not listening.")
        except OSError as error:
            # Darwin does not expose SO_ACCEPTCONN on Unix sockets. A bound
            # Unix stream still cannot become a TCP listener; accept fails if
            # the supervisor supplied a bound but non-listening descriptor.
            if error.errno != errno.ENOPROTOOPT:
                raise
    except (OSError, ValueError):
        listener.close()
        raise ValueError("A listening Unix socket inherited from the supervisor is required.") from None

    class PrivateServer(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False

        def __init__(self, address, handler):
            # Upstream may set address_family after resolving its dummy address.
            # BaseServer creates no socket; adopt only the validated Unix FD.
            BaseServer.__init__(self, listener.getsockname(), handler)
            self.socket = listener
            self.server_name, self.server_port = "private", 0

        def get_request(self):
            connection, _ = self.socket.accept()
            return connection, ("private", 0)

    return PrivateServer


def guarded_handler(upstream, model, max_tokens=8192, aliases=()):
    class PrivateHandler(upstream):
        def setup(self):
            self.request.settimeout(120)
            super().setup()

        def log_message(self, *args):
            pass

        def _set_cors_headers(self):
            pass

        def end_headers(self):
            self.send_header("X-MLX-Agent-Transport", PRIVATE_PROTOCOL)
            super().end_headers()

        def _reply(self, status, value):
            raw = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(raw)

        def _guard(self):
            hosts = self.headers.get_all("Host", [])
            if len(hosts) != 1 or hosts[0] not in ("127.0.0.1", "localhost") or self.headers.get_all("Origin") is not None:
                self._reply(403, {"error": "Private supervisor requests only."})
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            if self.path == "/_mlx/private":
                self._reply(200, {"protocol": PRIVATE_PROTOCOL})
            elif self.path == "/v1/models":
                self._reply(200, {"data": [{"id": model}]})
            else:
                self._reply(404, {"error": "Unknown private endpoint."})

        def do_OPTIONS(self):
            self._reply(405, {"error": "Browser preflights are not supported."})

        def do_POST(self):
            if not self._guard():
                return
            if self.path != "/v1/chat/completions":
                self._reply(404, {"error": "Only configured-model chat inference is supported."})
                return
            if (self.headers.get_all("Content-Type", []) != ["application/json"]):
                self._reply(415, {"error": "Use application/json."})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or self.headers.get_all("Transfer-Encoding") is not None or not lengths[0].isascii() or not lengths[0].isdigit():
                self._reply(400, {"error": "Use one bounded Content-Length."})
                return
            if len(lengths[0]) > 8:
                self._reply(413, {"error": "Request exceeds the private worker limit."})
                return
            length = int(lengths[0])
            if not 0 < length <= MAX_BODY:
                self._reply(413, {"error": "Request exceeds the private worker limit."})
                return
            try:
                raw = self.rfile.read(length)
                body = json.loads(raw)
                if len(raw) != length or not isinstance(body, dict):
                    raise ValueError()
                if body.get("model", "default_model") not in ("default_model", model, *aliases):
                    raise ValueError()
                if any(body.get(key) is not None for key in ("adapter_path", "adapter", "draft_model", "draft_adapter", "trust_remote_code")):
                    raise ValueError()
                messages = body.get("messages")
                if not isinstance(messages, list) or any(not isinstance(m, dict) or not isinstance(m.get("content"), str)
                                                        or m.get("role") not in ("system", "user", "assistant", "tool") for m in messages):
                    raise ValueError()
                tokens = body.get("max_tokens", max_tokens)
                if type(tokens) is not int or tokens <= 0 or type(body.get("stream", False)) is not bool:
                    raise ValueError()
                completion_tokens = body.get("max_completion_tokens", tokens)
                if type(completion_tokens) is not int or completion_tokens <= 0:
                    raise ValueError()
            except (ValueError, UnicodeError, TypeError):
                self._reply(400, {"error": "Invalid fixed-model text chat request."})
                return
            body["model"] = model
            body["max_tokens"] = min(tokens, completion_tokens, max_tokens)
            body.pop("max_completion_tokens", None)
            encoded = json.dumps(body).encode()
            original = self.rfile
            self.rfile = io.BytesIO(encoded)
            del self.headers["Content-Length"]
            self.headers["Content-Length"] = str(len(encoded))
            try:
                super().do_POST()
            finally:
                self.rfile = original

    return PrivateHandler


def runtime_python(executable):
    """Resolve only the Python interpreter named by a venv-owned MLX entrypoint."""
    sibling = Path(sys.executable).parent / "mlx_lm.server"
    resolved = str(sibling) if executable == "mlx_lm.server" and sibling.is_file() else shutil.which(executable)
    if resolved is None:
        raise ValueError("mlx_lm.server is not installed.")
    entry = Path(resolved).resolve()
    if entry.name != "mlx_lm.server" or not (entry.parent.parent / "pyvenv.cfg").is_file():
        raise ValueError("mlx_lm.server must belong to an explicit Python virtual environment.")
    with entry.open("rb") as source:
        first = source.readline(4096).decode().strip()
    for name in ("python", "python3"):
        python = entry.parent / name
        if python.is_file() and first == "#!" + str(python):
            return str(python)
    raise ValueError("The MLX entrypoint does not identify its virtual environment interpreter.")


def run_runtime(model, descriptor, forwarded):
    # Adopt the listener before importing a runtime that could load weights.
    server_class = private_server_class(descriptor)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    from mlx_lm import server
    original_run = server.run
    parameters = inspect.signature(original_run).parameters
    if "server_class" not in parameters or "handler_class" not in parameters:
        raise ValueError("This MLX-LM version cannot serve an inherited private listener.")
    local = Path(model).expanduser()
    if not local.is_dir():
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model, "config.json")
        if not isinstance(cached, str):
            raise ValueError("Prepare the model in the local cache before managed serving.")
        local = Path(cached).parent
    local = local.resolve()
    maximum = 8192
    if "--max-tokens" in forwarded:
        maximum = int(forwarded[forwarded.index("--max-tokens") + 1])
        if not 1 <= maximum <= 65536:
            raise ValueError("Invalid private worker token limit.")
    handler = guarded_handler(server.APIHandler, str(local), maximum, aliases=(model,))

    def private_run(*args, **kwargs):
        if len(args) > 3:
            raise ValueError("Unsupported MLX-LM server initialization.")
        return original_run(*args, **dict(kwargs, server_class=server_class, handler_class=handler))

    previous = sys.argv
    server.run = private_run
    try:
        # Locked arguments come last: forwarded serving options cannot select a
        # different model, host, or public listener.
        sys.argv = [previous[0]] + forwarded + ["--model", str(local), "--host", "127.0.0.1", "--port", "0", "--log-level", "ERROR"]
        server.main()
    finally:
        server.run = original_run
        sys.argv = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--socket-fd", type=int)
    parser.add_argument("--runtime-executable", default="mlx_lm.server")
    parser.add_argument("--runtime-child", action="store_true", help=argparse.SUPPRESS)
    args, forwarded = parser.parse_known_args(argv)
    if args.describe:
        if forwarded or args.model is not None or args.socket_fd is not None:
            parser.error("--describe does not start a worker or accept model arguments")
        print(json.dumps(identity()))
        return 0
    if not args.model or args.socket_fd is None or args.socket_fd < 3:
        parser.error("a model and supervisor-owned --socket-fd >= 3 are required")
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if not args.runtime_child:
        python = runtime_python(args.runtime_executable)
        os.set_inheritable(args.socket_fd, True)
        command = [python, str(Path(__file__).resolve()), "--runtime-child", "--model", args.model,
                   "--socket-fd", str(args.socket_fd), "--"] + forwarded
        os.execv(python, command)
    run_runtime(args.model, args.socket_fd, forwarded)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError):
        # Runtime errors may include private paths or request content.
        print("Private MLX worker unavailable; verify the installed runtime and local model.", file=sys.stderr)
        raise SystemExit(1)
