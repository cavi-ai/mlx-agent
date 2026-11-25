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
        from tests.unit.ownership_fixtures import isolate
        isolate(self)
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

    def model(self, model_type, **config):
        directory = self.root / "model-{0}".format(len(list(self.root.glob("model-*"))))
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(dict(config, model_type=model_type)), encoding="utf-8")
        return directory

    def still(self, name):
        path = self.root / name
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return str(path)

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
        self.assertNotIn("--image", argv)

    def test_reference_stills_reach_the_runner_in_order(self):
        stills = [self.still("sheet.png"), self.still("pose.jpg")]
        argv = self.plan(self.model("qwen_image_21", max_references=10), images=stills)["argv"]
        self.assertEqual([argv[index + 1] for index, flag in enumerate(argv) if flag == "--image"], stills)

    def test_reference_checkpoint_requires_two_steps_even_without_images(self):
        vision = self.model("qwen_image_21", max_references=10)
        self.assertIn("1", self.plan(self.model("qwen_image_21"), steps=1)["argv"])
        with self.assertRaises(GenerateError) as caught:
            self.plan(vision, steps=1)
        self.assertEqual(caught.exception.code, "invalid_arguments")

    def test_reference_refusals(self):
        vision = self.model("qwen_image_21", max_references=1)
        sheet = self.still("sheet.png")
        cases = [
            (lambda: self.plan(self.model("qwen_image_21"), images=[sheet]), "references_unsupported"),
            (lambda: self.plan(vision, images=[sheet, sheet]), "references_unsupported"),
            (lambda: self.plan(vision, images=["sheet.png"]), "invalid_arguments"),
            (lambda: self.plan(vision, images=[str(self.root / "missing.png")]), "invalid_arguments"),
            (lambda: self.plan(vision, images=[str(vision / "config.json")]), "invalid_arguments"),
        ]
        for call, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(GenerateError) as caught:
                    call()
                self.assertEqual(caught.exception.code, code)

    def test_cli_refuses_a_reference_for_a_text_only_checkpoint(self):
        import contextlib
        import io

        from mlx_agent.cli import main

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["convert", "generate", "--path", str(self.model("qwen_image_21")), "--prompt", "the same figure",
                         "--out", self.out, "--image", self.still("sheet.png"), "--json"])
        self.assertNotEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "references_unsupported")

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
        with mock.patch("mlx_agent.inference_runner.sync_ports", return_value=[]) as sync:
            result = run_generate(plan, manifests=self.manifests, root=self.root / "backends",
                                  runner=lambda argv, **kw: seen.append(kw) or Completed(stdout="step 1/40\n" + line + "\n"))
            self.assertEqual(sync.call_args.args[0]["id"], "mflux")
            self.assertEqual((result["schema"], result["path"], result["pixel_std"]), ("generate/1", self.out, 61.2))
            self.assertEqual(seen[0]["stdin"], subprocess.DEVNULL)
            with self.assertRaises(GenerateError) as caught:
                run_generate(plan, manifests=self.manifests, root=self.root / "backends",
                             runner=lambda argv, **kw: Completed(returncode=1, stderr="MemoryError\n"))
            self.assertEqual(caught.exception.code, "generate_failed")

            with self.assertRaises(GenerateError) as caught:
                run_generate(plan, manifests=self.manifests, root=self.root / "backends", runner=lambda argv, **kw:
                             Completed(returncode=3, stdout=json.dumps({"error": "invalid_arguments", "detail": "Unreadable reference."})))
            self.assertEqual(caught.exception.code, "invalid_arguments")
            self.assertIn("Unreadable reference.", str(caught.exception))

            def slow(argv, **kwargs):
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

            with self.assertRaises(GenerateError) as caught:
                run_generate(plan, manifests=self.manifests, root=self.root / "backends", runner=slow, timeout=5)
            self.assertEqual(caught.exception.code, "generate_timeout")


if __name__ == "__main__":
    unittest.main()
