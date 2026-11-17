import hashlib
import io
import json
import os
import socket
import struct
import subprocess
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mlx_agent import cli
from mlx_agent.backends import INSTALL_MARKER
from mlx_agent.speak import SPEAK_RUNNER, SpeakError, plan_speak, run_speak

from .backend_fixtures import synthetic_manifests, synthetic_registries


FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def result_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


RESULT = {
    "path": "", "sample_rate": 24000, "audio_seconds": 2.5, "seconds": 0.5, "load_seconds": 1.2,
    "real_time_factor": 0.2, "peak_memory_gb": 0.9,
}


class SpeakTests(unittest.TestCase):
    def setUp(self):
        from tests.unit.ownership_fixtures import isolate
        isolate(self)
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifests = synthetic_manifests()
        self.registries = synthetic_registries(self.manifests, installed=("mlx-audio",))
        venv = self.root / "backends" / "mlx-audio"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / INSTALL_MARKER).write_text(json.dumps({"id": "mlx-audio", "version": self.manifests["mlx-audio"]["version"]}), encoding="utf-8")
        self.out = str(self.root / "speech.wav")

    def model(self, model_type="kokoro", name=None):
        directory = self.root / (name or "model-{0}".format(len(list(self.root.glob("model-*")))))
        directory.mkdir(parents=True)
        config = {"model_type": model_type} if model_type else {"sample_rate": 24000}
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return directory

    def plan(self, path, **kwargs):
        return plan_speak(path, kwargs.pop("text", "Hello there."), kwargs.pop("out", self.out),
                          manifests=self.manifests, registries=self.registries, root=self.root / "backends", **kwargs)

    def test_plan_runs_the_runner_with_the_text_as_one_token(self):
        plan = self.plan(self.model(), text="Hi; rm -rf /\nsecond line", voice="af_heart", speed=1.25, lang_code="en-US")
        argv = plan["argv"]
        self.assertEqual(argv[:2], [str(self.root / "backends" / "mlx-audio" / "bin" / "python"), str(SPEAK_RUNNER)])
        self.assertIn("--text=Hi; rm -rf /\nsecond line", argv)
        self.assertEqual(argv[argv.index("--model-type") + 1], "kokoro")
        self.assertEqual([argv[argv.index(flag) + 1] for flag in ("--out", "--speed", "--voice", "--lang-code")],
                         [self.out, "1.25", "af_heart", "en-US"])
        self.assertEqual((plan["backend"], plan["voice"], plan["speed"], plan["lang_code"]), ("mlx-audio", "af_heart", 1.25, "en-US"))

    def test_plan_passes_text_that_looks_like_a_flag_as_its_own_value(self):
        self.assertIn("--text=--speed 9", self.plan(self.model(), text="--speed 9")["argv"])
        self.assertIn("--text=-x", self.plan(self.model(), text="-x")["argv"])

    def test_plan_omits_voice_and_language_unless_given(self):
        argv = self.plan(self.model())["argv"]
        self.assertNotIn("--voice", argv)
        self.assertNotIn("--lang-code", argv)
        self.assertEqual(argv[argv.index("--speed") + 1], "1.0")

    def test_kokoro_uses_voice_assets_from_the_selected_local_model(self):
        model = self.model()
        (model / "voices").mkdir()
        for name in ("af_heart", "bf_emma"):
            (model / "voices" / (name + ".safetensors")).write_bytes(b"voice")
        for voice in (None, "bf_emma", "bf_emma.safetensors"):
            with self.subTest(voice=voice):
                plan = self.plan(model, voice=voice)
                filename = "af_heart.safetensors" if voice is None else voice.removesuffix(".safetensors") + ".safetensors"
                argv = plan["argv"]
                self.assertIn("--voice", argv)
                self.assertEqual(argv[argv.index("--voice") + 1], str(model / "voices" / filename))
                self.assertEqual(plan["voice"], voice)

    def test_plan_names_the_architecture_from_the_directory_when_config_has_no_model_type(self):
        snapshot = self.model(model_type=None, name="models--mlx-community--Kokoro-82M-bf16/snapshots/abc123")
        argv = self.plan(snapshot)["argv"]
        self.assertEqual(argv[argv.index("--model-type") + 1], "kokoro")
        argv = self.plan(self.model(model_type=None, name="kokoro-82m-bf16"))["argv"]
        self.assertEqual(argv[argv.index("--model-type") + 1], "kokoro")
        with self.assertRaises(SpeakError) as caught:
            self.plan(self.model(model_type=None, name="unnamed-model"))
        self.assertEqual(caught.exception.code, "not_tts_model")

    def test_plan_refusals(self):
        model = self.model()
        existing = self.root / "exists.wav"
        existing.write_bytes(b"x")
        broken = self.root / "broken"
        broken.mkdir()
        (broken / "config.json").write_text("[1]", encoding="utf-8")
        cases = [
            (lambda: self.plan(self.model("whisper")), "not_tts_model"),
            (lambda: self.plan(self.model("qwen2_vl")), "not_tts_model"),
            (lambda: self.plan(self.root / "missing"), "model_unreadable"),
            (lambda: self.plan(broken), "model_unreadable"),
            (lambda: self.plan(model, text=""), "invalid_arguments"),
            (lambda: self.plan(model, text="   "), "invalid_arguments"),
            (lambda: self.plan(model, text="x" * 2001), "invalid_arguments"),
            (lambda: self.plan(model, text="bell\x07"), "invalid_arguments"),
            (lambda: self.plan(model, text=None), "invalid_arguments"),
            (lambda: self.plan(model, out="relative.wav"), "invalid_arguments"),
            (lambda: self.plan(model, out=str(self.root / "x.mp3")), "invalid_arguments"),
            (lambda: self.plan(model, out=str(existing)), "invalid_arguments"),
            (lambda: self.plan(model, out=str(self.root / "nowhere" / "x.wav")), "invalid_arguments"),
            (lambda: self.plan(model, voice=""), "invalid_arguments"),
            (lambda: self.plan(model, voice="../etc/passwd"), "invalid_arguments"),
            (lambda: self.plan(model, voice="v" * 65), "invalid_arguments"),
            (lambda: self.plan(model, speed=0.49), "invalid_arguments"),
            (lambda: self.plan(model, speed=2.01), "invalid_arguments"),
            (lambda: self.plan(model, speed=float("nan")), "invalid_arguments"),
            (lambda: self.plan(model, speed=True), "invalid_arguments"),
            (lambda: self.plan(model, speed="1.0"), "invalid_arguments"),
            (lambda: self.plan(model, lang_code=""), "invalid_arguments"),
            (lambda: self.plan(model, lang_code="en us"), "invalid_arguments"),
        ]
        for call, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(SpeakError) as caught:
                    call()
                self.assertEqual(caught.exception.code, code)
        for speed in (0.5, 2.0, 1):
            self.assertEqual(self.plan(model, speed=speed)["speed"], float(speed))
        registries = synthetic_registries(self.manifests)
        with self.assertRaises(SpeakError) as caught:
            plan_speak(model, "x", self.out, manifests=self.manifests, registries=registries, root=self.root / "elsewhere")
        self.assertEqual(caught.exception.code, "backend_not_installed")

    def test_run_reads_the_last_json_line_and_classifies_failures(self):
        plan = self.plan(self.model(), voice="af_heart")
        line = json.dumps(dict(RESULT, path=self.out))
        seen = []
        with mock.patch("mlx_agent.inference_runner.sync_ports", return_value=[]) as sync:
            result = run_speak(plan, manifests=self.manifests, root=self.root / "backends",
                               runner=lambda argv, **kw: seen.append((argv, kw)) or Completed(stdout="loading\n" + line + "\n"))
            self.assertEqual(sync.call_args.args[0]["id"], "mlx-audio")
            self.assertEqual(seen[0][0], plan["argv"])
            self.assertEqual(seen[0][1]["stdin"], subprocess.DEVNULL)
            self.assertEqual(seen[0][1]["timeout"], 600)
            self.assertEqual(set(result), set(result_fixture("convert-speak.json")))
            self.assertEqual((result["schema"], result["path"], result["sample_rate"], result["real_time_factor"]), ("speak/1", self.out, 24000, 0.2))
            self.assertEqual((result["voice"], result["backend"]), ("af_heart", "mlx-audio"))

            noisy = "Traceback\n  File x\nRuntimeError: bad api_key=sk-secret123456 here\n"
            with self.assertRaises(SpeakError) as caught:
                run_speak(plan, manifests=self.manifests, root=self.root / "backends",
                          runner=lambda argv, **kw: Completed(returncode=1, stderr=noisy))
            self.assertEqual(caught.exception.code, "speak_failed")
            self.assertIn("RuntimeError", str(caught.exception))
            self.assertNotIn("sk-secret123456", str(caught.exception))

            with self.assertRaises(SpeakError) as caught:
                run_speak(plan, manifests=self.manifests, root=self.root / "backends",
                          runner=lambda argv, **kw: Completed(returncode=0, stdout="no json here\n"))
            self.assertEqual(caught.exception.code, "speak_failed")

            def slow(argv, **kwargs):
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

            with self.assertRaises(SpeakError) as caught:
                run_speak(plan, manifests=self.manifests, root=self.root / "backends", runner=slow, timeout=5)
            self.assertEqual(caught.exception.code, "speak_timeout")
            self.assertIn("exceeded 5 s", str(caught.exception))

    def test_cli_speaks_through_the_plan_and_emits_the_envelope(self):
        model = self.model()
        line = json.dumps(dict(RESULT, path=self.out))
        captured = []
        real = run_speak

        def run(plan, timeout):
            return real(plan, manifests=self.manifests, root=self.root / "backends", timeout=timeout,
                        runner=lambda argv, **kw: captured.append((argv, kw)) or Completed(stdout=line + "\n"))

        argv = ["convert", "speak", "--path", str(model), "--text", "Hello there.", "--out", self.out,
                "--voice", "af_heart", "--speed", "1.5", "--lang-code", "a", "--timeout", "42", "--json"]
        with mock.patch("mlx_agent.speak.load_manifests", return_value=self.manifests), \
                mock.patch("mlx_agent.speak.load_registries", return_value=self.registries), \
                mock.patch("mlx_agent.speak.is_installed", return_value=True), \
                mock.patch("mlx_agent.inference_runner.sync_ports", return_value=[]), \
                mock.patch("mlx_agent.speak.run_speak", run):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(argv)
            payload = json.loads(buffer.getvalue())
            self.assertEqual((code, payload["operation"], payload["status"]), (0, "convert-speak", "ok"))
            self.assertEqual(payload["data"]["path"], self.out)
            self.assertEqual(payload["data"]["real_time_factor"], 0.2)
            self.assertEqual(captured[0][1]["timeout"], 42)
            self.assertEqual(captured[0][0][captured[0][0].index("--speed") + 1], "1.5")

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(argv[:argv.index("--speed") + 1] + ["9"] + argv[argv.index("--speed") + 2:])
            payload = json.loads(buffer.getvalue())
            self.assertEqual((code, payload["status"], payload["error"]["code"]), (2, "error", "invalid_arguments"))

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(argv[:argv.index("--timeout") + 1] + ["0"] + argv[argv.index("--timeout") + 2:])
            self.assertEqual((code, json.loads(buffer.getvalue())["error"]["code"]), (2, "convert_failed"))

    def test_cli_speaks_on_the_endpoint_that_already_serves_the_model(self):
        model = self.model()
        wav, key = b"RIFF" + bytes(60), "k" * 40
        state, requests = {"busy": False, "reset": False}, []

        class Endpoint(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def answer(self, status, body, content_type="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers.get("Authorization"), body))
                if state["busy"]:
                    return self.answer(409, {"error": {"code": "model_busy", "message": "Finish the active model job."}})
                self.answer(202, {"id": "job-1", "state": "queued"})

            def do_GET(self):
                if self.path == "/v1/tasks/job-1":
                    self.answer(200, {"id": "job-1", "state": "succeeded"})
                elif self.path == "/v1/tasks/job-1/result":
                    reference = {"bytes": len(wav), "sha256": hashlib.sha256(wav).hexdigest()}
                    self.answer(200, {"result": dict(RESULT, path="/endpoint/output.wav", output_artifact=reference)})
                elif self.path == "/v1/tasks/job-1/artifacts/output" and state["reset"]:
                    # The connection drops after the first bytes of the artifact.
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(wav)))
                    self.end_headers()
                    self.wfile.write(wav[:20])
                    self.wfile.flush()
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    self.connection.close()
                    self.close_connection = True
                elif self.path == "/v1/tasks/job-1/artifacts/output":
                    self.answer(200, wav, "audio/wav")
                else:
                    self.answer(404, {"error": {"code": "not_found", "message": "Unknown task endpoint."}})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]
        receipts = self.root / "receipts" / ".mlx-agent-receipts" / "serve"
        receipts.mkdir(parents=True)
        for path, text in ((self.root / "access.key", key), (receipts / "{0}.jit-config.json".format(port), json.dumps(
                {"access_key_file": str(self.root / "access.key"), "api_key_sha256": hashlib.sha256(key.encode()).hexdigest()}))):
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as target:
                target.write(text)
        service = {"runtime": "tasks", "alive": True, "argv_match": True, "path": str(model), "port": port,
                   "receipts_dir": str(self.root / "receipts"), "task_kinds": ["speak"]}
        argv = ["convert", "speak", "--path", str(model), "--text", "Hello there.", "--out", self.out,
                "--voice", "af_heart", "--speed", "1.5", "--lang-code", "a", "--json"]
        with mock.patch("mlx_agent.speak.load_manifests", return_value=self.manifests), \
                mock.patch("mlx_agent.speak.load_registries", return_value=self.registries), \
                mock.patch("mlx_agent.speak.is_installed", return_value=True), \
                mock.patch("mlx_agent.service_registry.discover_services", return_value={"services": [service]}), \
                mock.patch("mlx_agent.speak.run_speak", side_effect=AssertionError("loaded a second copy")):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(argv)
            payload = json.loads(buffer.getvalue())
            self.assertEqual((code, payload["status"]), (0, "ok"))
            self.assertEqual((payload["data"]["path"], payload["data"]["served"]), (self.out, {"port": port, "task_id": "job-1"}))
            self.assertEqual(Path(self.out).read_bytes(), wav)
            self.assertEqual(requests, [("/v1/tasks", "Bearer " + key, {"kind": "speak", "parameters": {
                "text": "Hello there.", "voice": "af_heart", "speed": 1.5, "lang_code": "a"}})])

            state["busy"] = True
            busy_out = str(self.root / "busy.wav")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(argv[:argv.index("--out") + 1] + [busy_out] + argv[argv.index("--out") + 2:])
            self.assertEqual((code, json.loads(buffer.getvalue())["error"]["code"]), (2, "model_busy"))
            self.assertFalse(Path(busy_out).exists())

            # The endpoint stops speech jobs after run_speak's 600 s; a longer --timeout is refused before submission.
            state["busy"] = False
            long_out = str(self.root / "long.wav")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(argv[:argv.index("--out") + 1] + [long_out] + argv[argv.index("--out") + 2:]
                                + ["--timeout", "700"])
            self.assertEqual((code, json.loads(buffer.getvalue())["error"]["code"]), (2, "served_timeout_limit"))
            self.assertEqual(len(requests), 2)
            self.assertFalse(Path(long_out).exists())

            state["reset"] = True
            cut_out = str(self.root / "cut.wav")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(argv[:argv.index("--out") + 1] + [cut_out] + argv[argv.index("--out") + 2:])
            self.assertFalse(Path(cut_out).exists())
            self.assertEqual((code, json.loads(buffer.getvalue())["error"]["code"]), (2, "artifact_incomplete"))


if __name__ == "__main__":
    unittest.main()
