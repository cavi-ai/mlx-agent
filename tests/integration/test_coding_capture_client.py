"""Opt-in installed-client proof with existing extensions and a deterministic API."""

import base64
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from uuid import uuid4

from mlx_agent.capture_platform import CaptureVolume, capture_process_identity
from mlx_agent.coding_capture_client import OpenCodeEvents, validate_session_export
from mlx_agent.coding_capture_plugins import ExtensionHost, freeze_extensions

CLIENT = os.environ.get("MLX_AGENT_OPENCODE_TEST_CLIENT")
CONFIG = os.environ.get("MLX_AGENT_OPENCODE_TEST_CONFIG")


class FixtureAPI(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.requests.append({"path": self.path, "authorization": self.headers.get("Authorization")})
        if self.path == "/api/v0/models":
            body = {"data": [{"id": "fixture-chat", "type": "llm", "arch": "llama"},
                              {"id": "fixture-asr", "type": "llm", "arch": "whisper"}]}
        elif self.path == "/api/tags":
            body = {"models": [{"name": "fixture-chat:latest"}, {"name": "fixture-embed"}]}
        else:
            self.send_error(404)
            return
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 1024 * 1024 or self.path != "/v1/chat/completions":
            self.send_error(400)
            return
        body = json.loads(self.rfile.read(length))
        self.requests.append({"path": self.path, "session": self.headers.get("x-opencode-session-id"), "body": body})
        self.send_response(200)
        if body.get("stream"):
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, finish in (({"role": "assistant", "content": "Compatibility fixture accepted."}, None), ({}, "stop")):
                chunk = {"id": "chatcmpl_fixture", "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": "local", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            raw = json.dumps({"id": "chatcmpl_fixture", "object": "chat.completion", "created": int(time.time()),
                              "model": "local", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Fixture"},
                                                              "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)


@unittest.skipUnless(CLIENT and CONFIG, "Select existing OpenCode and extension assets explicitly.")
class OwnedClientCompatibilityTests(unittest.TestCase):
    def test_existing_extensions_owned_server_run_and_explicit_export(self):
        source_root = Path(CONFIG)
        declared = json.loads((source_root / "opencode.json").read_text())
        self.assertFalse(declared.get("plugins") or declared.get("plugin"), "Package operations require rejection before bootstrap.")
        sources = sorted(str(p) for p in (source_root / "plugins").glob("*.ts"))
        self.assertTrue(sources)
        FixtureAPI.requests = []
        api = ThreadingHTTPServer(("127.0.0.1", 0), FixtureAPI)
        threading.Thread(target=api.serve_forever, daemon=True).start()
        self.addCleanup(api.server_close)
        self.addCleanup(api.shutdown)
        endpoint = "http://127.0.0.1:" + str(api.server_port)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            identifier = str(uuid4())
            volume = root / identifier
            CaptureVolume.create(volume, identifier, 128 * 1024 * 1024 - 65536)
            try:
                with CaptureVolume.open_owned(volume, readonly=False) as mounted:
                    project, profile = mounted / "project", mounted / "profile"
                    project.mkdir(mode=0o700)
                    profile.mkdir(mode=0o700)
                    manifest = freeze_extensions(Path(CLIENT), sources, mounted / "frozen")
                    for directory in (profile / "data", profile / "cache", profile / "data/opencode", profile / "cache/opencode"):
                        directory.mkdir(mode=0o700)
                    (profile / "data/opencode/auth.json").write_text(json.dumps({"lmstudio": {"key": "fixture-lmstudio-key"}}))
                    (profile / "cache/opencode/models.json").write_text(json.dumps({"lmstudio": {"models": {"stale": {}}}}))
                    toolbin = mounted / "bin"
                    toolbin.mkdir(mode=0o700)
                    graph = toolbin / "code-review-graph"
                    graph.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$CAPTURE_GRAPH_LOG\"\n")
                    graph.chmod(0o700)
                    configuration = {"permission": {"bash": "ask", "edit": "ask"}, "lsp": False, "autoupdate": False,
                                     "model": "capture/local", "small_model": "capture/local", "provider": {
                                         "capture": {"npm": "@ai-sdk/openai-compatible", "name": "Fixture", "options": {
                                             "baseURL": endpoint + "/v1", "apiKey": "fixture-key"}, "models": {
                                                 "local": {"name": "Fixture", "limit": {"context": 200000, "output": 128}}}},
                                         "lmstudio": {"npm": "@ai-sdk/openai-compatible"},
                                         "ollama": {"npm": "@ai-sdk/openai-compatible"}}}
                    environment = dict(os.environ, PATH=str(toolbin) + os.pathsep + os.environ["PATH"],
                                       LMSTUDIO_BASE_URL=endpoint, OLLAMA_BASE_URL=endpoint,
                                       CAPTURE_GRAPH_LOG=str(mounted / "graph-commands.log"))
                    environment.pop("LMSTUDIO_API_KEY", None)
                    environment.pop("LM_API_TOKEN", None)
                    with ExtensionHost(Path(CLIENT), manifest, profile, configuration, project, environment) as host:
                        receipt = json.loads(Path(host.profile["receipt_path"]).read_text())
                        self.assertIn("mlx_agent_command", [tool["name"] for item in receipt["plugins"]
                                                           for tool in item["description"]["tools"]])
                        self.assertEqual(len(FixtureAPI.requests), 2)
                        lmstudio = next(item for item in FixtureAPI.requests if item["path"] == "/api/v0/models")
                        self.assertEqual(lmstudio["authorization"], "Bearer fixture-lmstudio-key")
                        self.assertFalse(any(item["path"].startswith("/v1") for item in FixtureAPI.requests))
                        effective = json.loads(Path(host.profile["effective_config_path"]).read_text())
                        self.assertEqual(effective["permission"], configuration["permission"])
                        self.assertEqual(set(effective["provider"]["lmstudio"]["models"]), {"fixture-chat"})
                        self.assertIn("stale", effective["provider"]["lmstudio"]["blacklist"])
                        self.run_owned_server(host, project, profile, manifest)
            finally:
                CaptureVolume.remove(volume)

    def run_owned_server(self, host, project, profile, manifest):
        environment = dict(host.profile["environment"], OPENCODE_SERVER_PASSWORD="fixture-private-password")
        with (profile / "server-stderr.log").open("wb") as errors:
            server = subprocess.Popen([CLIENT, "serve", "--print-logs", "--stdio", "--hostname", "127.0.0.1", "--port", "0"],
                                      env=environment, cwd=project, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=errors, pass_fds=host.server_fds, start_new_session=True)
            try:
                identity = capture_process_identity(server.pid)
                selector = selectors.DefaultSelector()
                selector.register(server.stdout, selectors.EVENT_READ)
                deadline, address = time.monotonic() + 30, None
                while time.monotonic() < deadline and server.poll() is None:
                    if selector.select(min(1, max(0, deadline - time.monotonic()))):
                        line = server.stdout.readline(8192)
                        try:
                            address = json.loads(line)["url"]
                            break
                        except (ValueError, KeyError):
                            continue
                selector.close()
                self.assertIsNotNone(address, (profile / "server-stderr.log").read_text()[-4096:])
                headers = {"Authorization": "Basic " + base64.b64encode(b"opencode:fixture-private-password").decode(),
                           "x-opencode-directory": str(project)}
                def get(path):
                    with urlopen(Request(address + path, headers=headers), timeout=30) as response:
                        return json.loads(response.read(1024 * 1024))
                expected = {item["id"] for item in manifest["bundles"]}
                activation_deadline = time.monotonic() + 30
                while True:
                    inventory = get("/api/plugin")["data"]
                    accepted = {item["id"] for item in inventory if item["state"]["status"] == "active"}
                    failed = {item.get("id") for item in inventory if item["state"]["status"] == "failed"}
                    if expected <= accepted or expected & failed or time.monotonic() >= activation_deadline:
                        break
                    time.sleep(0.05)
                self.assertTrue(expected <= accepted,
                                (inventory, (profile / "server-stderr.log").read_text()[-4096:]))
                self.assertTrue(get("/api/config"))
                agents = get("/api/agent")["data"]
                self.assertTrue(agents)
                self.assertFalse(any(item["path"].startswith("/v1") for item in FixtureAPI.requests))
                cli_environment = dict(environment)
                cli_environment.pop("MLX_AGENT_CAPTURE_EXTENSION_FD", None)
                run = subprocess.run([CLIENT, "run", "--server", address, "--format", "json", "--model", "capture/local",
                                      "--title", "Compatibility fixture", "Reply with fixture text."],
                                     env=cli_environment, cwd=project, capture_output=True, timeout=45)
                self.assertEqual(run.returncode, 0, (run.stdout.decode()[-4096:], run.stderr.decode()[-4096:]))
                events = OpenCodeEvents(lambda _: None, artifact_bytes=33554432)
                events.feed(run.stdout)
                events.finish()
                self.assertFalse(events.has_error, run.stdout.decode()[-4096:])
                export = subprocess.run([CLIENT, "session", "export", events.session_id, "--server", address],
                                        env=cli_environment, cwd=project, capture_output=True, timeout=30)
                self.assertEqual(export.returncode, 0, export.stderr.decode()[-4096:])
                try:
                    session = validate_session_export(export.stdout, events.session_id, "capture/local")
                except Exception as error:
                    types = sorted({item.get("type", "<missing>") for item in json.loads(export.stdout).get("messages", [])})
                    self.fail(str(error) + "; observed message types: " + repr(types))
                self.assertEqual(session["info"]["id"], events.session_id)
                self.assertEqual(session["info"]["location"]["directory"], str(project))
                requests = [item for item in FixtureAPI.requests if item["path"].startswith("/v1")]
                self.assertTrue(requests)
                self.assertTrue(all(item["session"] == events.session_id for item in requests))
                tool_names = sorted({tool.get("function", {}).get("name", "<missing>")
                                     for item in requests for tool in item["body"].get("tools", [])})
                self.assertIn("mlx_agent_command", tool_names,
                              (tool_names, [(item.get("name"), item.get("permission")) for item in agents],
                               (profile / "server-stderr.log").read_text()[-8192:]))
                self.assertEqual(capture_process_identity(server.pid), identity)
                print(json.dumps({"schema": "coding-capture-client-proof/1", "session_id": events.session_id,
                                  "session_export_sha256": hashlib.sha256(export.stdout).hexdigest(),
                                  "client_sha256": next(item["sha256"] for item in manifest["inputs"]
                                                        if item["path"] == manifest["client_path"]),
                                  "extensions": len(expected), "inference_requests": len(requests),
                                  "project_matches": True, "session_headers_match": True,
                                  "legacy_tool_registered": True}, sort_keys=True))
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
                server.stdin.close()
                server.stdout.close()
