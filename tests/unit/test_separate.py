import io
import json
import subprocess
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mlx_agent import separate_runner
from mlx_agent.backends import INSTALL_MARKER, load_manifests, load_registries
from mlx_agent.separate import SEPARATE_RUNNER, SeparateError, plan_separate, run_separate


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class SeparateTests(unittest.TestCase):
    def setUp(self):
        from tests.unit.ownership_fixtures import isolate
        isolate(self)
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.manifests = load_manifests()
        self.registries = load_registries(self.manifests, root=self.root / "unused", find_spec=lambda name: None)
        venv = self.root / "backends" / "mlx-audio"
        self.package = venv / "lib" / "python3.12" / "site-packages" / "mlx_audio"
        self.package.mkdir(parents=True)
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        marker = {"id": "mlx-audio", "version": self.manifests["mlx-audio"]["version"]}
        (venv / INSTALL_MARKER).write_text(json.dumps(marker), encoding="utf-8")
        self.audio = self.root / "mix.wav"
        self.audio.write_bytes(b"RIFF")
        self.out = self.root / "vocals.wav"
        self.rest = self.root / "rest.wav"

    def model(self, model_type):
        directory = self.root / model_type
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
        return directory

    def plan(self, path, **kwargs):
        return plan_separate(path, kwargs.pop("audio", self.audio), kwargs.pop("out", self.out),
                             manifests=self.manifests, registries=self.registries, root=self.root / "backends", **kwargs)

    def test_plan_runs_the_audio_backend_port_for_a_mel_band_roformer_checkpoint(self):
        separator = self.model("mel_band_roformer")
        plan = self.plan(separator, rest=self.rest)
        python = str(self.root / "backends" / "mlx-audio" / "bin" / "python")
        self.assertEqual(plan["backend"], "mlx-audio")
        self.assertEqual(plan["argv"], [
            python, str(SEPARATE_RUNNER), "--module", "mlx_audio.mlx_agent_ports.mel_band_roformer",
            "--model", str(separator), "--audio", str(self.audio), "--out", str(self.out), "--rest", str(self.rest),
        ])
        alone = self.plan(separator)
        self.assertNotIn("--rest", alone["argv"])
        self.assertIsNone(alone["rest"])

    def test_plan_refusals(self):
        separator = self.model("mel_band_roformer")
        self.out.write_bytes(b"RIFF")
        cases = [
            (lambda: self.plan(self.model("whisper"), out=self.root / "a.wav"), "not_separation_model"),
            (lambda: self.plan(self.root / "missing"), "model_unreadable"),
            (lambda: self.plan(separator, audio=self.root / "none.wav", out=self.root / "a.wav"), "audio_not_found"),
            (lambda: self.plan(separator), "invalid_arguments"),
            (lambda: self.plan(separator, out="vocals.wav"), "invalid_arguments"),
            (lambda: self.plan(separator, out=self.root / "a.mp3"), "invalid_arguments"),
            (lambda: self.plan(separator, out=self.root / "a.wav", rest=self.root / "a.wav"), "invalid_arguments"),
            (lambda: self.plan(separator, out=self.root / "no" / "a.wav"), "invalid_arguments"),
        ]
        for call, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(SeparateError) as caught:
                    call()
                self.assertEqual(caught.exception.code, code)
        with self.assertRaises(SeparateError) as caught:
            plan_separate(separator, self.audio, self.root / "a.wav", manifests=self.manifests,
                          registries=self.registries, root=self.root / "elsewhere")
        self.assertEqual(caught.exception.code, "backend_not_installed")

    def test_run_returns_both_stems(self):
        plan = self.plan(self.model("mel_band_roformer"), rest=self.rest)
        printed = {"path": str(self.out), "rest": str(self.rest), "sample_rate": 44100, "audio_seconds": 12.0,
                   "seconds": 9.5, "load_seconds": 1.2}
        runner = lambda argv, **kwargs: Completed(stdout="loading\n" + json.dumps(printed) + "\n")
        result = run_separate(plan, manifests=self.manifests, root=self.root / "backends", runner=runner)
        self.assertEqual((result["schema"], result["path"], result["rest"], result["audio_seconds"]),
                         ("separate/1", str(self.out), str(self.rest), 12.0))
        self.assertTrue((self.package / "mlx_agent_ports" / "mel_band_roformer" / "__init__.py").is_file())
        missing_rest = lambda argv, **kwargs: Completed(stdout=json.dumps(dict(printed, rest=None)) + "\n")
        with self.assertRaises(SeparateError) as caught:
            run_separate(plan, manifests=self.manifests, root=self.root / "backends", runner=missing_rest)
        self.assertEqual(caught.exception.code, "separate_failed")

        def slow(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        with self.assertRaises(SeparateError) as caught:
            run_separate(plan, manifests=self.manifests, root=self.root / "backends", runner=slow, timeout=5)
        self.assertEqual(caught.exception.code, "separate_timeout")


class SeparateRunnerTests(unittest.TestCase):
    def test_the_runner_loads_the_port_model_and_prints_its_stems(self):
        port = types.ModuleType("separation_port")
        port.load = lambda path: {"weights": path}
        port.generate = lambda model, audio, out, rest: {"path": out, "rest": rest, "model": model["weights"], "audio": audio}
        buffer = io.StringIO()
        with mock.patch.dict(sys.modules, {"separation_port": port}), redirect_stdout(buffer):
            code = separate_runner.main(["--module", "separation_port", "--model", "/m", "--audio", "/mix.wav",
                                         "--out", "/v.wav", "--rest", "/r.wav"])
        result = json.loads(buffer.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual((result["path"], result["rest"], result["model"], result["audio"]), ("/v.wav", "/r.wav", "/m", "/mix.wav"))
        self.assertIn("load_seconds", result)


if __name__ == "__main__":
    unittest.main()
