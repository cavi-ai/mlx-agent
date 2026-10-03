import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mlx_agent.backends import INSTALL_MARKER, load_manifests, with_ports
from mlx_agent.generate import GENERATE_RUNNER, GenerateError, plan_generate, run_generate

from .backend_fixtures import synthetic_manifests, synthetic_registries


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifests = dict(synthetic_manifests(), mflux=load_manifests()["mflux"])
        self.registries = self.ported(synthetic_registries(self.manifests, installed=("mflux",)))
        venv = self.root / "backends" / "mflux"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / INSTALL_MARKER).write_text(json.dumps({"id": "mflux", "version": self.manifests["mflux"]["version"]}), encoding="utf-8")
        self.out = str(self.root / "render.png")

    def ported(self, registries):
        for backend_id, entry in registries.items():
            entry["registry"] = with_ports(entry["registry"], self.manifests[backend_id])
        return registries

    def model(self, model_type):
        directory = self.root / "model-{0}".format(len(list(self.root.glob("model-*"))))
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
        return directory

    def plan(self, path, **kwargs):
        return plan_generate(path, kwargs.pop("prompt", "a red apple"), kwargs.pop("out", self.out),
                             manifests=self.manifests, registries=self.registries, root=self.root / "backends", **kwargs)

    def test_plan_runs_the_runner_with_the_port_module_and_prompt_as_one_token(self):
        plan = self.plan(self.model("qwen_image_21"), prompt="a red apple; rm -rf /", width=512, height=768, steps=12, seed=7)
        self.assertEqual(plan["argv"][:4], [str(self.root / "backends" / "mflux" / "bin" / "python"), str(GENERATE_RUNNER),
                                            "--module", "mflux.mlx_agent_ports.qwen_image_21"])
        argv = plan["argv"]
        self.assertEqual(argv[argv.index("--prompt") + 1], "a red apple; rm -rf /")
        self.assertEqual([argv[argv.index(flag) + 1] for flag in ("--width", "--height", "--steps", "--seed")], ["512", "768", "12", "7"])

    def test_plan_refusals(self):
        model = self.model("qwen_image_21")
        existing = self.root / "exists.png"
        existing.write_bytes(b"x")
        cases = [
            (lambda: self.plan(self.model("whisper")), "not_image_model"),
            (lambda: self.plan(self.root / "missing"), "model_unreadable"),
            (lambda: self.plan(model, prompt=""), "invalid_arguments"),
            (lambda: self.plan(model, prompt="x" * 2001), "invalid_arguments"),
            (lambda: self.plan(model, prompt="bell\x07"), "invalid_arguments"),
            (lambda: self.plan(model, out="relative.png"), "invalid_arguments"),
            (lambda: self.plan(model, out=str(self.root / "x.jpg")), "invalid_arguments"),
            (lambda: self.plan(model, out=str(existing)), "invalid_arguments"),
            (lambda: self.plan(model, width=500), "invalid_arguments"),
            (lambda: self.plan(model, height=4096), "invalid_arguments"),
            (lambda: self.plan(model, steps=0), "invalid_arguments"),
            (lambda: self.plan(model, seed=-1), "invalid_arguments"),
        ]
        for call, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(GenerateError) as caught:
                    call()
                self.assertEqual(caught.exception.code, code)
        registries = self.ported(synthetic_registries(self.manifests))
        with self.assertRaises(GenerateError) as caught:
            plan_generate(model, "x", self.out, manifests=self.manifests, registries=registries, root=self.root / "elsewhere")
        self.assertEqual(caught.exception.code, "backend_not_installed")

    def test_run_reads_the_last_json_line_and_classifies_failures(self):
        plan = self.plan(self.model("qwen_image_21"))
        line = json.dumps({"path": self.out, "width": 1024, "height": 1024, "steps": 40, "seed": 42, "seconds": 70.1, "pixel_std": 61.2})
        seen = []
        with mock.patch("mlx_agent.generate.sync_ports", return_value=[]) as sync:
            result = run_generate(plan, manifests=self.manifests, root=self.root / "backends",
                                  runner=lambda argv, **kw: seen.append(kw) or Completed(stdout="step 1/40\n" + line + "\n"))
            self.assertEqual(sync.call_args.args[0]["id"], "mflux")
            self.assertEqual((result["schema"], result["path"], result["pixel_std"]), ("generate/1", self.out, 61.2))
            self.assertEqual(seen[0]["stdin"], subprocess.DEVNULL)
            with self.assertRaises(GenerateError) as caught:
                run_generate(plan, manifests=self.manifests, root=self.root / "backends",
                             runner=lambda argv, **kw: Completed(returncode=1, stderr="MemoryError\n"))
            self.assertEqual(caught.exception.code, "generate_failed")

            def slow(argv, **kwargs):
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

            with self.assertRaises(GenerateError) as caught:
                run_generate(plan, manifests=self.manifests, root=self.root / "backends", runner=slow, timeout=5)
            self.assertEqual(caught.exception.code, "generate_timeout")


if __name__ == "__main__":
    unittest.main()
