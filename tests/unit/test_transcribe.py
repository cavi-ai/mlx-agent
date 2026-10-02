import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.backends import INSTALL_MARKER
from mlx_agent.transcribe import TRANSCRIBE_RUNNER, TranscribeError, plan_transcribe, run_transcribe

from .backend_fixtures import synthetic_manifests, synthetic_registries


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class TranscribeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifests = synthetic_manifests()
        self.registries = synthetic_registries(self.manifests, installed=("mlx-audio",))
        venv = self.root / "backends" / "mlx-audio"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / INSTALL_MARKER).write_text(json.dumps({"id": "mlx-audio", "version": "0.0-test"}), encoding="utf-8")
        self.audio = self.root / "clip.wav"
        self.audio.write_bytes(b"RIFF")

    def model(self, model_type):
        directory = self.root / model_type
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
        return directory

    def plan(self, path, **kwargs):
        return plan_transcribe(path, kwargs.pop("audio", self.audio), manifests=self.manifests,
                               registries=self.registries, root=self.root / "backends", **kwargs)

    def test_plan_runs_the_runner_with_the_speech_backend_python(self):
        plan = self.plan(self.model("whisper"), language="en")
        self.assertEqual(plan["backend"], "mlx-audio")
        self.assertEqual(plan["argv"][:2], [str(self.root / "backends" / "mlx-audio" / "bin" / "python"), str(TRANSCRIBE_RUNNER)])
        self.assertEqual(plan["argv"][-2:], ["--language", "en"])
        self.assertNotIn("--trust-remote-code", plan["argv"])

    def test_plan_refusals(self):
        cases = [
            (lambda: self.plan(self.model("qwen2")), "not_speech_model"),
            (lambda: self.plan(self.root / "missing"), "model_unreadable"),
            (lambda: self.plan(self.model("voxtral"), audio=self.root / "none.wav"), "audio_not_found"),
            (lambda: self.plan(self.model("glmasr"), language="EN; rm"), "invalid_arguments"),
        ]
        for call, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(TranscribeError) as caught:
                    call()
                self.assertEqual(caught.exception.code, code)
        registries = synthetic_registries(self.manifests)
        with self.assertRaises(TranscribeError) as caught:
            plan_transcribe(self.model("voxtral_realtime"), self.audio, manifests=self.manifests,
                            registries=registries, root=self.root / "elsewhere")
        self.assertEqual(caught.exception.code, "backend_not_installed")

    def test_run_reads_the_last_json_line(self):
        plan = self.plan(self.model("whisper"))
        seen = []

        def runner(argv, **kwargs):
            seen.append((argv, kwargs))
            return Completed(stdout='loading...\n{"text": "hello there", "seconds": 1.5, "audio_seconds": 2.0}\n')

        result = run_transcribe(plan, manifests=self.manifests, root=self.root / "backends", runner=runner)
        self.assertEqual((result["schema"], result["text"], result["seconds"]), ("transcribe/1", "hello there", 1.5))
        self.assertEqual(seen[0][0], plan["argv"])
        self.assertEqual(seen[0][1]["stdin"], subprocess.DEVNULL)
        self.assertNotIn("HF_TOKEN", seen[0][1]["env"])

    def test_run_failures_are_classified(self):
        plan = self.plan(self.model("whisper"))
        failing = lambda argv, **kwargs: Completed(returncode=1, stderr="Traceback\nValueError: bad weights\n")
        with self.assertRaises(TranscribeError) as caught:
            run_transcribe(plan, manifests=self.manifests, root=self.root / "backends", runner=failing)
        self.assertEqual(caught.exception.code, "transcribe_failed")
        self.assertIn("bad weights", str(caught.exception))

        def slow(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        with self.assertRaises(TranscribeError) as caught:
            run_transcribe(plan, manifests=self.manifests, root=self.root / "backends", runner=slow, timeout=5)
        self.assertEqual(caught.exception.code, "transcribe_timeout")


if __name__ == "__main__":
    unittest.main()
