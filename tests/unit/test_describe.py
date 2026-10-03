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
from mlx_agent.describe import DESCRIBE_RUNNER, DescribeError, plan_describe, run_describe

from .backend_fixtures import synthetic_manifests, synthetic_registries


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


RESULT = {
    "text": "A red circle on white.", "prompt_tokens": 286, "generation_tokens": 8, "prompt_tps": 164.8,
    "generation_tps": 211.2, "peak_memory_gb": 3.77, "seconds": 1.8, "load_seconds": 0.5,
}


class DescribeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifests = synthetic_manifests()
        self.registries = synthetic_registries(self.manifests, installed=("mlx-vlm",))
        venv = self.root / "backends" / "mlx-vlm"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / INSTALL_MARKER).write_text(json.dumps({"id": "mlx-vlm", "version": self.manifests["mlx-vlm"]["version"]}), encoding="utf-8")
        self.image = self.media("picture.png")
        self.video = self.media("clip.mp4")

    def media(self, name):
        path = self.root / name
        path.write_bytes(b"x")
        return str(path)

    def model(self, model_type="qwen2_vl"):
        directory = self.root / "model-{0}".format(len(list(self.root.glob("model-*"))))
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
        return directory

    def plan(self, path, **kwargs):
        kwargs.setdefault("image", self.image if "video" not in kwargs else None)
        return plan_describe(path, kwargs.pop("prompt", "What is in this image?"),
                             manifests=self.manifests, registries=self.registries, root=self.root / "backends", **kwargs)

    def test_plan_image_runs_the_runner_with_the_prompt_as_one_token(self):
        plan = self.plan(self.model(), prompt="What? ; rm -rf /\nsecond", max_tokens=64, temperature=0.5, max_pixels=1000000)
        argv = plan["argv"]
        self.assertEqual(argv[:2], [str(self.root / "backends" / "mlx-vlm" / "bin" / "python"), str(DESCRIBE_RUNNER)])
        self.assertIn("--prompt=What? ; rm -rf /\nsecond", argv)
        self.assertIn("--prompt=-x", self.plan(self.model(), prompt="-x")["argv"])
        self.assertEqual([argv[argv.index(flag) + 1] for flag in ("--image", "--max-tokens", "--temperature", "--max-pixels")],
                         [self.image, "64", "0.5", "1000000"])
        self.assertNotIn("--video", argv)
        self.assertNotIn("--fps", argv)
        self.assertEqual((plan["backend"], plan["model_type"], plan["input"]), ("mlx-vlm", "qwen2_vl", {"kind": "image", "path": self.image}))

    def test_plan_video_carries_fps_with_a_default_and_no_max_pixels_unless_given(self):
        plan = self.plan(self.model(), image=None, video=self.video)
        argv = plan["argv"]
        self.assertEqual((argv[argv.index("--video") + 1], argv[argv.index("--fps") + 1]), (self.video, "1.0"))
        self.assertNotIn("--image", argv)
        self.assertNotIn("--max-pixels", argv)
        self.assertEqual(plan["input"], {"kind": "video", "path": self.video})
        argv = self.plan(self.model(), image=None, video=self.media("clip.MOV"), fps=2.5)["argv"]
        self.assertEqual(argv[argv.index("--fps") + 1], "2.5")

    def test_plan_requires_exactly_one_of_image_or_video(self):
        model = self.model()
        for kwargs in ({"image": None}, {"image": self.image, "video": self.video}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(DescribeError) as caught:
                    plan_describe(model, "q", manifests=self.manifests, registries=self.registries, root=self.root / "backends", **kwargs)
                self.assertEqual(caught.exception.code, "invalid_arguments")
                self.assertIn("exactly one", str(caught.exception))

    def test_plan_refusals(self):
        model = self.model()
        broken = self.root / "broken"
        broken.mkdir()
        (broken / "config.json").write_text("[1]", encoding="utf-8")
        cases = [
            (lambda: self.plan(self.model("whisper")), "not_vision_model"),
            (lambda: self.plan(self.model("qwen2")), "not_vision_model"),
            (lambda: self.plan(self.model("kokoro")), "not_vision_model"),
            (lambda: self.plan(self.root / "missing"), "model_unreadable"),
            (lambda: self.plan(broken), "model_unreadable"),
            (lambda: self.plan(model, prompt=""), "invalid_arguments"),
            (lambda: self.plan(model, prompt="x" * 4001), "invalid_arguments"),
            (lambda: self.plan(model, prompt="bell\x07"), "invalid_arguments"),
            (lambda: self.plan(model, prompt=None), "invalid_arguments"),
            (lambda: self.plan(model, image=str(self.root / "gone.png")), "invalid_arguments"),
            (lambda: self.plan(model, image=self.media("notes.txt")), "invalid_arguments"),
            (lambda: self.plan(model, image=self.video), "invalid_arguments"),
            (lambda: self.plan(model, image=None, video=self.image), "invalid_arguments"),
            (lambda: self.plan(model, image=None, video=str(self.root / "gone.mp4")), "invalid_arguments"),
            (lambda: self.plan(model, max_tokens=0), "invalid_arguments"),
            (lambda: self.plan(model, max_tokens=4097), "invalid_arguments"),
            (lambda: self.plan(model, max_tokens=True), "invalid_arguments"),
            (lambda: self.plan(model, temperature=-0.1), "invalid_arguments"),
            (lambda: self.plan(model, temperature=2.5), "invalid_arguments"),
            (lambda: self.plan(model, temperature=float("nan")), "invalid_arguments"),
            (lambda: self.plan(model, fps=1.0), "invalid_arguments"),
            (lambda: self.plan(model, image=None, video=self.video, fps=0.05), "invalid_arguments"),
            (lambda: self.plan(model, image=None, video=self.video, fps=8.5), "invalid_arguments"),
            (lambda: self.plan(model, image=None, video=self.video, fps=True), "invalid_arguments"),
            (lambda: self.plan(model, max_pixels=0), "invalid_arguments"),
            (lambda: self.plan(model, max_pixels=True), "invalid_arguments"),
            (lambda: self.plan(model, max_pixels=100000001), "invalid_arguments"),
        ]
        for call, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(DescribeError) as caught:
                    call()
                self.assertEqual(caught.exception.code, code)
        for suffix in (".png", ".jpg", ".JPEG", ".webp"):
            self.assertEqual(self.plan(model, image=self.media("p" + suffix))["input"]["kind"], "image")
        for suffix in (".mp4", ".mov", ".m4v"):
            self.assertEqual(self.plan(model, image=None, video=self.media("v" + suffix))["input"]["kind"], "video")
        registries = synthetic_registries(self.manifests)
        with self.assertRaises(DescribeError) as caught:
            plan_describe(model, "q", image=self.image, manifests=self.manifests, registries=registries, root=self.root / "elsewhere")
        self.assertEqual(caught.exception.code, "backend_not_installed")

    def test_run_reads_the_last_json_line_and_classifies_failures(self):
        plan = self.plan(self.model())
        line = json.dumps(RESULT)
        seen = []
        with mock.patch("mlx_agent.describe.sync_ports", return_value=[]) as sync:
            result = run_describe(plan, manifests=self.manifests, root=self.root / "backends",
                                  runner=lambda argv, **kw: seen.append((argv, kw)) or Completed(stdout="Fetching\n" + line + "\n"))
            self.assertEqual(sync.call_args.args[0]["id"], "mlx-vlm")
            self.assertEqual(seen[0][0], plan["argv"])
            self.assertEqual((seen[0][1]["stdin"], seen[0][1]["timeout"]), (subprocess.DEVNULL, 900))
            self.assertEqual(
                sorted(result),
                sorted(["schema", "backend", "model", "text", "prompt_tokens", "generation_tokens", "prompt_tps",
                        "generation_tps", "peak_memory_gb", "seconds", "load_seconds", "input"]),
            )
            self.assertEqual((result["schema"], result["text"], result["generation_tokens"]), ("describe/1", "A red circle on white.", 8))
            self.assertEqual(result["input"], {"kind": "image", "path": self.image})

            noisy = "Traceback\nValueError: bad token=hf_abcdefghijklmnop here\n"
            with self.assertRaises(DescribeError) as caught:
                run_describe(plan, manifests=self.manifests, root=self.root / "backends",
                             runner=lambda argv, **kw: Completed(returncode=1, stderr=noisy))
            self.assertEqual(caught.exception.code, "describe_failed")
            self.assertIn("ValueError", str(caught.exception))
            self.assertNotIn("hf_abcdefghijklmnop", str(caught.exception))

            with self.assertRaises(DescribeError) as caught:
                run_describe(plan, manifests=self.manifests, root=self.root / "backends",
                             runner=lambda argv, **kw: Completed(returncode=0, stdout="nothing\n"))
            self.assertEqual(caught.exception.code, "describe_failed")

            def slow(argv, **kwargs):
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

            with self.assertRaises(DescribeError) as caught:
                run_describe(plan, manifests=self.manifests, root=self.root / "backends", runner=slow, timeout=5)
            self.assertEqual(caught.exception.code, "describe_failed")
            self.assertIn("exceeded 5 s", str(caught.exception))

    def test_run_refuses_video_naming_the_model_type(self):
        plan = self.plan(self.model("moondream3"), image=None, video=self.video)
        refusal = json.dumps({"error": "video_unsupported", "model_type": "moondream3", "detail": "no decoder"})
        with mock.patch("mlx_agent.describe.sync_ports", return_value=[]):
            with self.assertRaises(DescribeError) as caught:
                run_describe(plan, manifests=self.manifests, root=self.root / "backends",
                             runner=lambda argv, **kw: Completed(returncode=3, stdout=refusal + "\n"))
            self.assertEqual(caught.exception.code, "video_unsupported")
            self.assertIn("moondream3", str(caught.exception))
            self.assertIn("no decoder", str(caught.exception))
            refusal = json.dumps({"error": "not_vision_model", "model_type": "moondream3"})
            with self.assertRaises(DescribeError) as caught:
                run_describe(self.plan(self.model("moondream3")), manifests=self.manifests, root=self.root / "backends",
                             runner=lambda argv, **kw: Completed(returncode=3, stdout=refusal + "\n"))
            self.assertEqual(caught.exception.code, "not_vision_model")

    def test_cli_describes_through_the_plan_and_emits_the_envelope(self):
        model = self.model()
        captured = []
        real = run_describe

        def run(plan, timeout):
            return real(plan, manifests=self.manifests, root=self.root / "backends", timeout=timeout,
                        runner=lambda argv, **kw: captured.append((argv, kw)) or Completed(stdout=json.dumps(RESULT) + "\n"))

        base = ["convert", "describe", "--path", str(model), "--prompt", "What is it?", "--json"]
        patches = (
            mock.patch("mlx_agent.describe.load_manifests", return_value=self.manifests),
            mock.patch("mlx_agent.describe.load_registries", return_value=self.registries),
            mock.patch("mlx_agent.describe.is_installed", return_value=True),
            mock.patch("mlx_agent.describe.sync_ports", return_value=[]),
            mock.patch("mlx_agent.cli.run_describe", run),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        def invoke(*extra):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(base + list(extra))
            return code, json.loads(buffer.getvalue())

        code, payload = invoke("--image", self.image, "--max-tokens", "32", "--timeout", "42")
        self.assertEqual((code, payload["operation"], payload["status"]), (0, "convert-describe", "ok"))
        self.assertEqual((payload["data"]["text"], payload["data"]["input"]["kind"]), ("A red circle on white.", "image"))
        self.assertEqual(captured[-1][1]["timeout"], 42)
        self.assertEqual(captured[-1][0][captured[-1][0].index("--max-tokens") + 1], "32")

        code, payload = invoke("--video", self.video, "--fps", "2", "--max-pixels", "50000")
        argv = captured[-1][0]
        self.assertEqual((code, payload["data"]["input"]["kind"]), (0, "video"))
        self.assertEqual([argv[argv.index(flag) + 1] for flag in ("--fps", "--max-pixels")], ["2.0", "50000"])

        code, payload = invoke("--image", self.image, "--video", self.video)
        self.assertEqual((code, payload["status"], payload["error"]["code"]), (2, "error", "invalid_arguments"))
        code, payload = invoke()
        self.assertEqual((code, payload["error"]["code"]), (2, "invalid_arguments"))
        code, payload = invoke("--image", self.image, "--timeout", "0")
        self.assertEqual((code, payload["error"]["code"]), (2, "convert_failed"))


if __name__ == "__main__":
    unittest.main()
