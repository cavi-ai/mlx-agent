"""Opt-in installed-client proof with existing extensions and a deterministic API."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

from mlx_agent.capture_platform import CaptureVolume, capture_process_identity
from mlx_agent.coding_capture import OwnedSession
from mlx_agent.coding_capture_plugins import freeze_extensions

CLIENT = os.environ.get("MLX_AGENT_OPENCODE_TEST_CLIENT")
CONFIG = os.environ.get("MLX_AGENT_OPENCODE_TEST_CONFIG")


class FixtureArtifacts:
    def __init__(self, root):
        self.root = root
        root.mkdir(mode=0o700)

    def write_artifact(self, name, raw):
        with (self.root / name).open("xb") as output:
            os.chmod(output.fileno(), 0o600)
            output.write(raw)
        return {"name": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


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
                    processes = {}
                    deadline = time.monotonic() + 120
                    with OwnedSession(manifest, profile, configuration, project, "capture/local", environment,
                                      artifact_bytes=33554432,
                                      on_process=lambda role, value: processes.update({role: value}),
                                      cancelled=lambda: time.monotonic() >= deadline) as session:
                        self.check_bootstrap(session, configuration)
                        self.run_prompt(session, project, profile, manifest, processes)
                    self.assertIsNotNone(session.server.poll())
            finally:
                CaptureVolume.remove(volume)

    def check_bootstrap(self, session, configuration):
        receipt = json.loads(Path(session.host.profile["receipt_path"]).read_text())
        self.assertIn("mlx_agent_command", [tool["name"] for item in receipt["plugins"]
                                           for tool in item["description"]["tools"]])
        # The extension bootstrap lists local models first; no inference request leaves before the prompt.
        bootstrap = FixtureAPI.requests[:2]
        self.assertEqual(sorted(item["path"] for item in bootstrap), ["/api/tags", "/api/v0/models"])
        lmstudio = next(item for item in bootstrap if item["path"] == "/api/v0/models")
        self.assertEqual(lmstudio["authorization"], "Bearer fixture-lmstudio-key")
        self.assertFalse(any(item["path"].startswith("/v1") for item in FixtureAPI.requests))
        effective = json.loads(Path(session.host.profile["effective_config_path"]).read_text())
        self.assertEqual(effective["permission"], configuration["permission"])
        self.assertEqual(set(effective["provider"]["lmstudio"]["models"]), {"fixture-chat"})
        self.assertIn("stale", effective["provider"]["lmstudio"]["blacklist"])

    def run_prompt(self, session, project, profile, manifest, processes):
        self.assertTrue(session.api("/api/config"))
        agents = session.api("/api/agent")["data"]
        self.assertTrue(agents)
        self.assertFalse(any(item["path"].startswith("/v1") for item in FixtureAPI.requests))
        storage = FixtureArtifacts(profile.parent / "artifacts")
        outcome = session.run("Reply with fixture text.", storage)
        run, export = outcome["run"], outcome["export"]
        self.assertIsNone(run["error"], run)
        self.assertEqual(run["exit_code"], 0)
        self.assertFalse(run["has_error"])
        self.assertIsNone(export["error"], export)
        exported = export["session"]
        self.assertEqual(set(processes), {"private_server", "client", "exporter"})
        self.assertEqual(exported["info"]["id"], run["session_id"])
        self.assertEqual(exported["info"]["location"]["directory"], str(project))
        requests = [item for item in FixtureAPI.requests if item["path"].startswith("/v1")]
        self.assertTrue(requests)
        self.assertTrue(all(item["session"] == run["session_id"] for item in requests))
        tool_names = sorted({tool.get("function", {}).get("name", "<missing>")
                             for item in requests for tool in item["body"].get("tools", [])})
        self.assertIn("mlx_agent_command", tool_names,
                      (tool_names, [(item.get("name"), item.get("permission")) for item in agents],
                       (profile / "server-stderr.log").read_text()[-8192:]))
        self.assertEqual(capture_process_identity(session.server.pid), processes["private_server"])
        print(json.dumps({"schema": "coding-capture-client-proof/1", "session_id": run["session_id"],
                          "session_export_sha256": hashlib.sha256((storage.root / "session.json").read_bytes()).hexdigest(),
                          "client_sha256": next(item["sha256"] for item in manifest["inputs"]
                                                if item["path"] == manifest["client_path"]),
                          "extensions": len(manifest["bundles"]), "inference_requests": len(requests),
                          "project_matches": True, "session_headers_match": True,
                          "legacy_tool_registered": True}, sort_keys=True))
