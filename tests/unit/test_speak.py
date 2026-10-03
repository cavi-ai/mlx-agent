import io
import json
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mlx_agent import cli
from mlx_agent.backends import INSTALL_MARKER
from mlx_agent.speak import SPEAK_RUNNER, SpeakError, plan_speak, run_speak

from .backend_fixtures import synthetic_manifests, synthetic_registries


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


RESULT = {
    "path": "", "sample_rate": 24000, "audio_seconds": 2.5, "seconds": 0.5, "load_seconds": 1.2,
    "real_time_factor": 0.2, "peak_memory_gb": 0.9,
}


class SpeakTests(unittest.TestCase):
    def setUp(self):
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
        self.assertEqual(argv[argv.index("--text") + 1], "Hi; rm -rf /\nsecond line")
        self.assertEqual(argv[argv.index("--model-type") + 1], "kokoro")
        self.assertEqual([argv[argv.index(flag) + 1] for flag in ("--out", "--speed", "--voice", "--lang-code")],
                         [self.out, "1.25", "af_heart", "en-US"])
        self.assertEqual((plan["backend"], plan["voice"], plan["speed"], plan["lang_code"]), ("mlx-audio", "af_heart", 1.25, "en-US"))

    def test_plan_omits_voice_and_language_unless_given(self):
        argv = self.plan(self.model())["argv"]
        self.assertNotIn("--voice", argv)
        self.assertNotIn("--lang-code", argv)
        self.assertEqual(argv[argv.index("--speed") + 1], "1.0")

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
        with mock.patch("mlx_agent.speak.sync_ports", return_value=[]) as sync:
            result = run_speak(plan, manifests=self.manifests, root=self.root / "backends",
                               runner=lambda argv, **kw: seen.append((argv, kw)) or Completed(stdout="loading\n" + line + "\n"))
            self.assertEqual(sync.call_args.args[0]["id"], "mlx-audio")
            self.assertEqual(seen[0][0], plan["argv"])
            self.assertEqual(seen[0][1]["stdin"], subprocess.DEVNULL)
            self.assertEqual(seen[0][1]["timeout"], 600)
            self.assertEqual(
                sorted(result),
                sorted(["schema", "backend", "model", "voice", "speed", "lang_code", "path", "sample_rate", "audio_seconds",
                        "seconds", "load_seconds", "real_time_factor", "peak_memory_gb"]),
            )
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
            self.assertEqual(caught.exception.code, "speak_failed")
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
                mock.patch("mlx_agent.speak.sync_ports", return_value=[]), \
                mock.patch("mlx_agent.cli.run_speak", run):
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


if __name__ == "__main__":
    unittest.main()
