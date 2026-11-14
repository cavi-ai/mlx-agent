import io
import json
import os
import signal
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from jsonschema import Draft202012Validator, FormatChecker

from mlx_agent import cli
from mlx_agent.serve import (ServeError, load_recipes, plan_start, resolve_serve_runtime, start_serve,
                             status_serve, stop_serve)
from mlx_agent.splash_runtime import SplashError, read_target
from .test_gguf import write_gguf

SCHEMA = Path(__file__).resolve().parents[2] / "schemas/serve-receipt.schema.json"
SHARD = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{0}-of-00002.gguf"
# Splash's serving contract: its key from SPLASH_API_KEY, and only its served name as a request model.
SPLASH_SERVER = '''
import argparse, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
p = argparse.ArgumentParser()
p.add_argument('command'); p.add_argument('--model'); p.add_argument('--model-dir')
p.add_argument('--offline', action='store_true'); p.add_argument('--host'); p.add_argument('--port', type=int)
p.add_argument('--max-memory')
a = p.parse_args()
key = os.environ.get('SPLASH_API_KEY')
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def reply(self, status, value):
        self.send_response(status); self.send_header('Content-Type', 'application/json'); self.end_headers()
        self.wfile.write(json.dumps(value).encode())
    def authorized(self):
        return key is not None and self.headers.get('Authorization') == 'Bearer ' + key
    def do_GET(self):
        self.reply(200 if self.authorized() else 401, {'object': 'list', 'data': [{'id': a.model}]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if not self.authorized():
            return self.reply(401, {'error': 'invalid or missing API key'})
        if body.get('model') not in (None, '', a.model):
            return self.reply(404, {'error': 'model not found'})
        self.reply(200, {'model': body['model'], 'stops_with': os.environ.get('SPLASH_APP_PID'), 'model_dir': a.model_dir,
                         'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'hi'}, 'finish_reason': 'stop'}]})
class Server(ThreadingHTTPServer):
    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
Server((a.host, a.port), Handler).serve_forever()
'''
# Like Splash's launcher, exec the server in place of the launcher process.
SPLASH_LAUNCHER = '''
import os, sys
from pathlib import Path
server = str(Path(__file__).with_name('server.py'))
os.execv(sys.executable, [sys.executable, '-u', server, *sys.argv[1:]])
'''


class SplashServeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.receipts = str(self.root / "receipts")
        self.launcher = self.install("Splash 1.6.0")

    def install(self, version_line):
        launcher = self.root / "bin/splash"
        launcher.parent.mkdir(exist_ok=True)
        launcher.write_text("#!/bin/sh\nprintf '%s\\n' '{0}'\n".format(version_line))
        launcher.chmod(0o755)
        environment = patch.dict(os.environ, {"MLX_AGENT_SPLASH": str(launcher)})
        environment.start()
        self.addCleanup(environment.stop)
        return str(launcher)

    def strata_profile(self):
        profile = self.root / "Qwen3.8-Flash-Next-Q2_0"
        (profile / "Q2_0").mkdir(parents=True)
        (profile / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "serving_runtime": "strata"}))
        (profile / "strata-model.json").write_text(json.dumps(
            {"schema_version": 1, "gguf": "Q2_0/" + SHARD.format(1), "mtp": None}))
        for part in (1, 2):
            write_gguf(profile / "Q2_0" / SHARD.format(part), architecture="qwen4exp")
        (profile / "tokenizer").mkdir()
        for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja"):
            (profile / "tokenizer" / name).write_text("{}")
        return profile

    def draft(self):
        draft = self.root / "draft"
        draft.mkdir()
        (draft / "config.json").write_text(json.dumps({"dflash_config": {"block_size": 16}}))
        (draft / "model.safetensors").write_bytes(b"draft weights")
        return draft

    def plan(self, path, **options):
        return plan_start(None, "splash", load_recipes(), path=str(path), receipts_dir=self.receipts, **options)

    def start(self, plan, spawn=None):
        return start_serve(plan, receipts_dir=self.receipts, confirm=True, preview_hash=plan["preview_hash"],
                           which=lambda executable: executable, port_free=lambda port: True,
                           spawn=spawn or (lambda argv, log_path: 4242), readiness=lambda url, deadline: True)

    def code(self, call):
        with self.assertRaises(ServeError) as error:
            call()
        return error.exception.code

    def test_strata_profile_serves_its_gguf_through_the_reviewed_launcher(self):
        profile = self.strata_profile()
        self.assertEqual(resolve_serve_runtime(None, str(profile), "splash", load_recipes()), ("splash", self.launcher))
        plan = self.plan(profile, splash_max_memory_gb=64, splash_max_context_tokens=32768, thinking="off")
        self.assertEqual(plan["argv"], [
            self.launcher, "serve", "--model", "local/Qwen3.8-Flash-Next-Q2_0",
            "--model-dir", str(profile / "Q2_0" / SHARD.format(1)), "--offline", "--host", "127.0.0.1",
            "--port", "8086", "--max-memory", "64000000000", "--max-context", "32768",
            "--default-reasoning-effort", "none"])
        self.assertEqual((plan["splash_source"], plan["splash_version"], plan["weight_format"], plan["thinking"]),
                         ("strata_profile", "1.6.0", "gguf", "off"))
        self.assertEqual(plan["readiness"], "http://127.0.0.1:8086/ready")
        other = self.plan(profile, splash_max_memory_gb=48, splash_max_context_tokens=32768, thinking="off")
        self.assertNotEqual(plan["preview_hash"], other["preview_hash"])

    def test_split_gguf_requires_every_part_and_names_the_model(self):
        first = write_gguf(self.root / "gguf" / SHARD.format(1), architecture="qwen4exp")
        self.assertEqual(self.code(lambda: self.plan(first)), "model_not_local")
        write_gguf(self.root / "gguf" / SHARD.format(2), architecture="qwen4exp")
        plan = self.plan(first)
        self.assertEqual(plan["splash_target"], str(first))
        self.assertEqual(plan["splash_served_model"], "local/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0")
        self.assertIsNone(plan["splash_source"])

    def test_served_name_follows_the_splash_model_rule(self):
        target = write_gguf(self.root / "my model--v2..final.gguf")
        self.assertEqual(self.plan(target)["splash_served_model"], "local/my-model-v2.final")

    def test_quantized_mlx_directory_is_served_and_unquantized_is_refused(self):
        model = self.root / "mlx"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"weights")
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        self.assertEqual(self.code(lambda: self.plan(model)), "splash_model_invalid")
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "quantization": {"bits": 4}}))
        plan = self.plan(model)
        self.assertEqual((plan["weight_format"], plan["splash_target"]), ("mlx", str(model)))

    def test_scan_reports_the_family_the_splash_engine_finds_for_each_quantized_output(self):
        models = self.root / "models"
        for name, width in (("Qwen3.8-27B-4bit", 5120), ("Other-9B-4bit", 4096)):
            (models / name).mkdir(parents=True)
            (models / name / "model.safetensors").write_bytes(b"weights")
            (models / name / "config.json").write_text(json.dumps(
                {"model_type": "qwen3_5", "text_config": {"hidden_size": width}, "quantization": {"bits": 4}}))

        def scan():
            out = io.StringIO()
            with patch("mlx_agent.gguf.default_gguf_roots", return_value=[]), redirect_stdout(out):
                self.assertEqual(cli.main(["convert", "scan", "--mlx-root", str(models), "--receipts-dir",
                                           self.receipts, "--no-signature", "--json"]), 0)
            return {item["name"]: item.get("splash") for item in json.loads(out.getvalue())["data"]["outputs"]}

        self.assertEqual(scan(), {"Other-9B-4bit": None, "Qwen3.8-27B-4bit": None})
        engine = self.root / "libexec/engine/splash"
        engine.parent.mkdir(parents=True)
        engine.write_text("#!/bin/sh\nif grep -q '\"hidden_size\": 5120' \"$4\"; then echo '{\"family\": \"Qwen3.8-27B\"}'; "
                          "else echo 'error: no supported model has this architecture' >&2; exit 1; fi\n")
        engine.chmod(0o755)
        self.assertEqual(scan(), {
            "Other-9B-4bit": {"family": None, "reason": "no supported model has this architecture"},
            "Qwen3.8-27B-4bit": {"family": "Qwen3.8-27B"},
        })

    def test_cached_mlx_repository_is_served_under_its_repository_id(self):
        repo, hf = "mlx-community/Qwen3.8-27B-4bit", str(self.root / "hf")
        self.assertEqual(self.code(lambda: plan_start(repo, "splash", load_recipes(), hf_cache=hf)), "model_not_local")
        entry = self.root / "hf/models--mlx-community--Qwen3.8-27B-4bit"
        snapshot = entry / "snapshots/abc123"
        snapshot.mkdir(parents=True)
        (entry / "refs").mkdir()
        (entry / "refs/main").write_text("abc123")
        (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64}}))
        (snapshot / "model.safetensors").write_bytes(b"weights")
        self.assertEqual(resolve_serve_runtime(repo, None, "splash", load_recipes(), hf_cache=hf), ("splash", self.launcher))
        plan = plan_start(repo, "splash", load_recipes(), hf_cache=hf, receipts_dir=self.receipts)
        self.assertEqual(plan["argv"][2:6], ["--model", repo, "--model-dir", str(snapshot)])
        receipt = self.start(plan)["receipt"]
        validator = Draft202012Validator(json.loads(SCHEMA.read_text()), format_checker=FormatChecker())
        self.assertEqual(list(validator.iter_errors(receipt)), [])
        self.assertEqual((receipt["repo"], receipt["splash_served_model"]), (repo, repo))
        command = "python -u -P -m server.server prepared --model {0} --port 8086".format(repo)
        self.assertTrue(status_serve(self.receipts, pid_alive=lambda pid: True, pid_command=lambda pid: command)[0]["argv_match"])
        (snapshot / "model.safetensors").write_bytes(b"other weights")
        self.assertEqual(self.code(lambda: self.start(plan)), "preview_stale")
        with self.assertRaises(SplashError) as error:
            read_target(snapshot, served_model="pub/name--copy")
        self.assertEqual(error.exception.code, "invalid_repo")

    def test_launcher_must_report_a_release_with_split_gguf_support(self):
        target = write_gguf(self.root / "model.gguf")
        self.install("Splash 1.3.9")
        self.assertEqual(self.code(lambda: self.plan(target)), "splash_version_unsupported")
        self.install("Splash (source checkout)")
        self.assertEqual(self.code(lambda: self.plan(target)), "splash_not_installed")
        with patch.dict(os.environ, {"MLX_AGENT_SPLASH": str(self.root / "missing")}):
            self.assertEqual(self.code(lambda: self.plan(target)), "splash_not_installed")

    def test_unsupported_selections_are_refused(self):
        target = write_gguf(self.root / "model.gguf")
        adapter = self.root / "adapter"
        adapter.mkdir()
        self.assertEqual(self.code(lambda: self.plan(target, jit=True)), "invalid_arguments")
        self.assertEqual(self.code(lambda: self.plan(target, adapter_path=str(adapter))), "unsupported_runtime")
        self.assertEqual(self.code(lambda: plan_start("pub/model", "mlx_lm", load_recipes(),
                                                      splash_max_context_tokens=8192)), "invalid_arguments")
        for options in ({"splash_max_memory_gb": 0.5}, {"splash_max_memory_gb": True},
                        {"splash_max_context_tokens": 1_010_001}, {"splash_max_context_tokens": 511}):
            with self.subTest(options=options):
                self.assertEqual(self.code(lambda: self.plan(target, **options)), "invalid_arguments")

    def test_draft_is_validated_and_bound_to_the_preview(self):
        target = write_gguf(self.root / "model.gguf")
        draft = self.draft()
        plan = self.plan(target, splash_draft_model=str(draft))
        self.assertEqual(plan["argv"][plan["argv"].index("--draft-model") + 1], str(draft))
        (draft / "model.safetensors").write_bytes(b"other draft weights")
        spawn = Mock()
        self.assertEqual(self.code(lambda: self.start(plan, spawn)), "preview_stale")
        spawn.assert_not_called()
        (draft / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        self.assertEqual(self.code(lambda: self.plan(target, splash_draft_model=str(draft))), "splash_draft_invalid")

    def test_confirm_refuses_target_or_launcher_changed_after_preview(self):
        profile = self.strata_profile()
        plan = self.plan(profile)
        (profile / "Q2_0" / SHARD.format(2)).write_bytes(b"changed")
        spawn = Mock()
        self.assertEqual(self.code(lambda: self.start(plan, spawn)), "preview_stale")
        target = write_gguf(self.root / "model.gguf")
        plan = self.plan(target)
        Path(self.launcher).write_text("#!/bin/sh\nprintf '%s\\n' 'Splash 1.6.1'\n")
        self.assertEqual(self.code(lambda: self.start(plan, spawn)), "preview_stale")
        spawn.assert_not_called()

    def test_receipt_owns_the_server_the_launcher_execs(self):
        profile = self.strata_profile()
        plan = self.plan(profile)
        receipt = self.start(plan)["receipt"]
        validator = Draft202012Validator(json.loads(SCHEMA.read_text()), format_checker=FormatChecker())
        self.assertEqual(list(validator.iter_errors(receipt)), [])
        self.assertEqual((receipt["runtime"], receipt["splash_served_model"], receipt["splash_version"]),
                         ("splash", "local/Qwen3.8-Flash-Next-Q2_0", "1.6.0"))
        server = ("/opt/homebrew/Cellar/splash/1.6.0/libexec/runtime/bin/python -u -P -m server.server "
                  "/Users/me/Library/Application Support/Splash/models/prepared --model {0} "
                  "--binary /opt/homebrew/Cellar/splash/1.6.0/libexec/engine/splash --port {1} --host 127.0.0.1")
        own = server.format("local/Qwen3.8-Flash-Next-Q2_0", 8086)
        foreign = server.format("local/other", 8086)
        status = lambda command: status_serve(self.receipts, pid_alive=lambda pid: True,
                                              pid_command=lambda pid: command)[0]["argv_match"]
        self.assertEqual((status(own), status(foreign)), (True, False))
        signals = []
        self.assertEqual(self.code(lambda: stop_serve(8086, self.receipts, pid_alive=lambda pid: True,
                                                      pid_command=lambda pid: foreign,
                                                      terminate=lambda pid, sig: signals.append(sig))),
                         "pid_argv_mismatch")
        self.assertEqual(signals, [])

    def test_load_on_request_serves_splash_behind_the_endpoint_key(self):
        import hashlib
        import http.client
        import sys
        import threading
        from mlx_agent.jit_serve import make_server
        from mlx_agent.worker_transport import gateway_config
        profile = self.strata_profile()
        libexec = Path(self.launcher).resolve().parents[1] / "libexec"
        python, launcher = libexec / "python/bin/python3", libexec / "install/launcher.py"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        launcher.parent.mkdir(parents=True)
        launcher.write_text(SPLASH_LAUNCHER)
        launcher.with_name("server.py").write_text(SPLASH_SERVER)
        token = "client-" + "s" * 40
        plan = self.plan(profile, jit=True, api_key_sha256=hashlib.sha256(token.encode()).hexdigest(),
                         splash_max_memory_gb=3)
        served = "local/Qwen3.8-Flash-Next-Q2_0"
        worker = plan["worker_argv"]
        self.assertEqual(worker[:2], [str(python), str(launcher)])
        self.assertEqual(worker[worker.index("--model") + 1], served)
        receipt = self.start(plan)["receipt"]
        validator = Draft202012Validator(json.loads(SCHEMA.read_text()), format_checker=FormatChecker())
        self.assertEqual(list(validator.iter_errors(receipt)), [])
        gateway_command = " ".join(plan["argv"])
        self.assertTrue(status_serve(self.receipts, pid_alive=lambda pid: True,
                                     pid_command=lambda pid: gateway_command)[0]["argv_match"])

        state = self.root.resolve()
        config = gateway_config(plan, control_token="control-" + "c" * 32, state_path=str(state / "state.json"))
        config.update(memory_state_dir=str(state / "memory"), compute_state_dir=str(state / "compute"))
        server = make_server(config, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        for cleanup in (lambda: thread.join(2), server.server_close, server.controller.close, server.shutdown):
            self.addCleanup(cleanup)

        def request(method, path, body=None, key=token):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
            try:
                connection.request(method, path, json.dumps(body) if body is not None else None,
                                   {"Authorization": "Bearer " + key} if key else {})
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()

        chat = {"model": profile.name, "messages": [{"role": "user", "content": "Hello"}]}
        self.assertEqual(request("POST", "/_mlx/policy", {"minimum_headroom_gb": 0}, key=config["control_token"])[0], 200)
        server.controller.memory_probe = lambda: 3_000_000_000 - 1
        status, body = request("POST", "/v1/chat/completions", chat)
        self.assertEqual((status, body["error"]["code"]), (503, "insufficient_headroom"))
        self.assertEqual(server.controller.status()["memory_check"]["required_available_bytes"], 3_000_000_000)
        server.controller.memory_probe = lambda: 64_000_000_000
        self.assertEqual(request("POST", "/v1/chat/completions", chat, key=None)[0], 401)
        for _ in range(2):
            status, body = request("POST", "/v1/chat/completions", chat)
            self.assertEqual(status, 200, body)
            self.assertEqual((body["model"], body["stops_with"], body["model_dir"]),
                             (served, str(os.getpid()), str(profile / "Q2_0" / SHARD.format(1))))

    def test_stop_waits_for_the_engine_grace_period_before_killing(self):
        plan = self.plan(write_gguf(self.root / "model.gguf"))
        self.start(plan)
        now, signals = [0.0], []
        command = "python -u -P -m server.server prepared --model local/model --port 8086"

        def sleep(seconds):
            now[0] += seconds

        result = stop_serve(8086, self.receipts, pid_alive=lambda pid: now[0] < 12, pid_command=lambda pid: command,
                            terminate=lambda pid, sig: signals.append(sig), clock=lambda: now[0], sleep=sleep)
        self.assertEqual((result["status"], result.get("forced"), signals), ("stopped", None, [signal.SIGTERM]))
        self.assertFalse((Path(self.receipts) / "8086.json").exists())


if __name__ == "__main__":
    unittest.main()
