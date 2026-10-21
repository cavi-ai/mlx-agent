"""One owned coding session: extension host, private client server, one prompt and its explicit export."""

import base64
import json
import os
import secrets
import selectors
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

from .capture_platform import capture_process_identity, stop_owned
from .coding_capture_contracts import DEFAULT_LIMITS
from .coding_capture_plugins import ExtensionHost, verify_extensions
from .coding_capture_process import export_session, run_client
from .task_contracts import TaskError

READY_SECONDS = 30
MAX_API_BYTES = 1024 * 1024


def _unavailable(message):
    return TaskError("capture_client_unavailable", message, "Inspect the owned server diagnostics and start a new capture.")


class OwnedSession:
    """Bootstrap the frozen extensions, start the private server, run one prompt, export it, stop everything.

    Use as a context manager. Every client command rechecks the frozen extensions and the server's process
    identity; leaving the context stops the server and the extension host by their recorded identities.
    """

    def __init__(self, manifest, profile_root, configuration, project_path, model, environment=None,
                 artifact_bytes=DEFAULT_LIMITS["artifact_bytes"], on_process=None, cancelled=None,
                 host_factory=ExtensionHost):
        self.manifest = manifest
        self.profile_root, self.project_path = Path(profile_root), Path(project_path)
        self.configuration, self.model, self.environment = configuration, model, environment
        self.artifact_bytes = artifact_bytes
        self.on_process = on_process or (lambda role, identity: None)
        self.cancelled = cancelled or (lambda: False)
        self._host_factory = host_factory
        self.host = self.server = self.server_url = None
        self._identity = self._errors = self._client_environment = self._headers = None

    def __enter__(self):
        self.host = self._host_factory(Path(self.manifest["client_path"]), self.manifest, self.profile_root,
                                       self.configuration, self.project_path, self.environment)
        self.host.__enter__()
        try:
            self._start_server()
            self._await_extensions()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_):
        self.close()

    def _start_server(self):
        password = secrets.token_urlsafe(32)
        environment = dict(self.host.profile["environment"], OPENCODE_SERVER_PASSWORD=password)
        descriptor = os.open(self.profile_root / "server-stderr.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._errors = os.fdopen(descriptor, "wb")
        self.server = subprocess.Popen(
            [self.manifest["client_path"], "serve", "--print-logs", "--stdio", "--hostname", "127.0.0.1", "--port", "0"],
            env=environment, cwd=self.project_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._errors, pass_fds=self.host.server_fds, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(self.server.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + READY_SECONDS
        try:
            while self.server_url is None and time.monotonic() < deadline and self.server.poll() is None:
                if selector.select(min(1, max(0, deadline - time.monotonic()))):
                    try:
                        self.server_url = json.loads(self.server.stdout.readline(8192))["url"]
                    except (ValueError, KeyError, TypeError):
                        continue
        finally:
            selector.close()
        if not isinstance(self.server_url, str):
            raise _unavailable("The owned client server did not report its address.")
        # A launcher may exec again before serving; the identity is bound once the server answers.
        self._identity = capture_process_identity(self.server.pid)
        self.on_process("private_server", self._identity)
        self._headers = {"Authorization": "Basic " + base64.b64encode(("opencode:" + password).encode()).decode(),
                         "x-opencode-directory": str(self.project_path)}
        self._client_environment = environment
        self._client_environment.pop("MLX_AGENT_CAPTURE_EXTENSION_FD", None)

    def _await_extensions(self):
        expected = {bundle["id"] for bundle in self.manifest["bundles"]}
        deadline = time.monotonic() + READY_SECONDS
        while True:
            inventory = self.api("/api/plugin")["data"]
            active = {item["id"] for item in inventory if item["state"]["status"] == "active"}
            failed = {item.get("id") for item in inventory if item["state"]["status"] == "failed"}
            if expected <= active:
                return
            if expected & failed or time.monotonic() >= deadline:
                raise _unavailable("The owned client server did not activate every frozen extension.")
            time.sleep(0.05)

    def api(self, path):
        """One authenticated read from the owned server, bounded to 1 MiB of JSON."""
        with urlopen(Request(self.server_url + path, headers=self._headers), timeout=READY_SECONDS) as response:
            return json.loads(response.read(MAX_API_BYTES))

    def check_context(self):
        verify_extensions(self.manifest)
        if capture_process_identity(self.server.pid) != self._identity:
            raise TaskError("capture_context_changed", "The owned client server changed.",
                            "Start a new capture; no prompt was sent to another server.")

    def run(self, prompt, storage, on_event=None):
        """Run `prompt` through the owned server and export its session; both retain evidence in `storage`."""
        command = {"client_path": self.manifest["client_path"], "project_path": str(self.project_path),
                   "environment": dict(self._client_environment), "server_url": self.server_url, "model": self.model,
                   "storage": storage, "artifact_bytes": self.artifact_bytes, "check_context": self.check_context,
                   "on_process": self.on_process, "cancelled": self.cancelled}
        run = run_client(command, prompt, on_event or (lambda _: None))
        export = export_session(command, run["session_id"], storage) if run["error"] is None else None
        return {"run": run, "export": export}

    def close(self):
        try:
            if self.server is not None:
                identity = self._identity
                if identity is None and self.server.poll() is None:
                    # Not yet serving: the unreaped child still holds its PID, so its current identity is ours.
                    identity = capture_process_identity(self.server.pid)
                stop_owned(self.server, identity)
        finally:
            for stream in (getattr(self.server, "stdin", None), getattr(self.server, "stdout", None), self._errors):
                if stream is not None and not stream.closed:
                    stream.close()
            if self.host is not None:
                self.host.__exit__(None, None, None)
