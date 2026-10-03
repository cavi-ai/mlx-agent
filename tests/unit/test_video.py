import io
import json
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mlx_agent import cli
from mlx_agent.backends import INSTALL_MARKER, load_manifests, with_ports
from mlx_agent.video import VIDEO_RUNNER, VideoError, plan_video, run_video

from .backend_fixtures import synthetic_manifests, synthetic_registries


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


RESULT = {
    "path": "", "width": 128, "height": 96, "frames": 5, "fps": 16, "duration_seconds": 0.312, "steps": 4, "seed": 7,
    "seconds": 12.5, "load_seconds": 30.1, "seconds_per_frame": 2.5, "peak_memory_gb": 21.4, "pixel_std": 55.2,
}
WAN_CONFIG = {"model_type": "t2v", "patch_size": [1, 2, 2], "vae_stride": [4, 8, 8], "sample_steps": 50, "sample_fps": 16}


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifests = dict(synthetic_manifests(), **{"mlx-video": load_manifests()["mlx-video"]})
        self.registries = synthetic_registries(self.manifests, installed=("mlx-video",))
        for backend_id, entry in self.registries.items():
            entry["registry"] = with_ports(entry["registry"], self.manifests[backend_id])
        venv = self.root / "backends" / "mlx-video"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / INSTALL_MARKER).write_text(json.dumps({"id": "mlx-video", "version": self.manifests["mlx-video"]["version"]}), encoding="utf-8")
        self.out = str(self.root / "clip.mp4")

    def model(self, config=None):
        directory = self.root / "model-{0}".format(len(list(self.root.glob("model-*"))))
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(WAN_CONFIG if config is None else config), encoding="utf-8")
        return directory

    def plan(self, path, **kwargs):
        return plan_video(path, kwargs.pop("prompt", "a red ball"), kwargs.pop("out", self.out),
                          manifests=self.manifests, registries=self.registries, root=self.root / "backends", **kwargs)

    def test_plan_runs_the_runner_with_the_port_module_and_prompt_as_one_token(self):
        plan = self.plan(self.model(), prompt="-a red ball; rm -rf /", width=128, height=96, frames=9, fps=12, steps=4, seed=7)
        argv = plan["argv"]
        self.assertEqual(argv[:4], [str(self.root / "backends" / "mlx-video" / "bin" / "python"), str(VIDEO_RUNNER),
                                    "--module", "mlx_video.mlx_agent_ports.t2v"])
        self.assertIn("--prompt=-a red ball; rm -rf /", argv)
        self.assertEqual([argv[argv.index(flag) + 1] for flag in ("--width", "--height", "--frames", "--fps", "--steps", "--seed", "--out")],
                         ["128", "96", "9", "12", "4", "7", self.out])
        self.assertEqual((plan["backend"], plan["frames"], plan["fps"]), ("mlx-video", 9, 12))

    def test_defaults_come_from_the_model_config(self):
        plan = self.plan(self.model(dict(WAN_CONFIG, sample_steps=30, sample_fps=24)))
        self.assertEqual((plan["steps"], plan["fps"], plan["width"], plan["height"], plan["frames"]), (30, 24, 832, 480, 81))
        bare = self.plan(self.model({"model_type": "t2v"}))
        self.assertEqual((bare["steps"], bare["fps"]), (50, 16))

    def test_bounds_follow_the_model_strides(self):
        wide = self.model(dict(WAN_CONFIG, vae_stride=[4, 16, 16]))
        self.plan(wide, width=640, height=352)
        with self.assertRaises(VideoError) as caught:
            self.plan(wide, width=624, height=352)
        self.assertEqual(caught.exception.code, "invalid_arguments")
        self.plan(self.model(), width=624, height=352)
        with self.assertRaises(VideoError) as caught:
            self.plan(self.model(dict(WAN_CONFIG, max_area=100000)), width=832, height=480)
        self.assertEqual(caught.exception.code, "invalid_arguments")

    def test_plan_refusals(self):
        model = self.model()
        existing = self.root / "exists.mp4"
        existing.write_bytes(b"x")
        cases = [
            (lambda: self.plan(self.model({"model_type": "whisper"})), "not_video_model"),
            (lambda: self.plan(self.root / "missing"), "model_unreadable"),
            (lambda: self.plan(model, prompt=""), "invalid_arguments"),
            (lambda: self.plan(model, prompt="x" * 2001), "invalid_arguments"),
            (lambda: self.plan(model, prompt="bell\x07"), "invalid_arguments"),
            (lambda: self.plan(model, out="relative.mp4"), "invalid_arguments"),
            (lambda: self.plan(model, out=str(self.root / "x.gif")), "invalid_arguments"),
            (lambda: self.plan(model, out=str(existing)), "invalid_arguments"),
            (lambda: self.plan(model, out=str(self.root / "nope" / "x.mp4")), "invalid_arguments"),
            (lambda: self.plan(model, width=500), "invalid_arguments"),
            (lambda: self.plan(model, width=32), "invalid_arguments"),
            (lambda: self.plan(model, height=4096), "invalid_arguments"),
            (lambda: self.plan(model, frames=16), "invalid_arguments"),
            (lambda: self.plan(model, frames=1), "invalid_arguments"),
            (lambda: self.plan(model, frames=245), "invalid_arguments"),
            (lambda: self.plan(model, fps=0), "invalid_arguments"),
            (lambda: self.plan(model, fps=61), "invalid_arguments"),
            (lambda: self.plan(model, steps=0), "invalid_arguments"),
            (lambda: self.plan(model, steps=101), "invalid_arguments"),
            (lambda: self.plan(model, seed=-1), "invalid_arguments"),
            (lambda: self.plan(model, seed=True), "invalid_arguments"),
        ]
        for call, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(VideoError) as caught:
                    call()
                self.assertEqual(caught.exception.code, code)
        registries = synthetic_registries(self.manifests)
        for backend_id, entry in registries.items():
            entry["registry"] = with_ports(entry["registry"], self.manifests[backend_id])
        with self.assertRaises(VideoError) as caught:
            plan_video(model, "x", self.out, manifests=self.manifests, registries=registries, root=self.root / "elsewhere")
        self.assertEqual(caught.exception.code, "backend_not_installed")

    def test_run_reads_the_last_json_line_and_classifies_failures(self):
        plan = self.plan(self.model())
        line = json.dumps(dict(RESULT, path=self.out))
        seen = []
        with mock.patch("mlx_agent.video.sync_ports", return_value=[]) as sync:
            result = run_video(plan, manifests=self.manifests, root=self.root / "backends",
                               runner=lambda argv, **kw: seen.append(kw) or Completed(stdout="Diffusion 1/4\n" + line + "\n"))
            self.assertEqual(sync.call_args.args[0]["id"], "mlx-video")
            self.assertEqual((result["schema"], result["path"], result["pixel_std"], result["seconds_per_frame"]), ("video/1", self.out, 55.2, 2.5))
            self.assertEqual(set(result) - {"schema", "backend", "model"}, set(RESULT))
            self.assertEqual(seen[0]["stdin"], subprocess.DEVNULL)
            with self.assertRaises(VideoError) as caught:
                run_video(plan, manifests=self.manifests, root=self.root / "backends",
                          runner=lambda argv, **kw: Completed(returncode=1, stderr="Traceback\nMemoryError: bad api_key=sk-secret123456 here\n"))
            self.assertEqual(caught.exception.code, "video_failed")
            self.assertIn("MemoryError", str(caught.exception))
            self.assertNotIn("sk-secret123456", str(caught.exception))
            with self.assertRaises(VideoError) as caught:
                run_video(plan, manifests=self.manifests, root=self.root / "backends",
                          runner=lambda argv, **kw: Completed(returncode=0, stdout="no json here\n"))
            self.assertEqual(caught.exception.code, "video_failed")

            def slow(argv, **kwargs):
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

            with self.assertRaises(VideoError) as caught:
                run_video(plan, manifests=self.manifests, root=self.root / "backends", runner=slow, timeout=5)
            self.assertEqual(caught.exception.code, "video_failed")
            self.assertIn("exceeded 5 s", str(caught.exception))

    def test_cli_renders_through_the_plan_and_emits_the_envelope(self):
        model = self.model()
        line = json.dumps(dict(RESULT, path=self.out))
        captured = []
        real = run_video

        def run(plan, timeout):
            return real(plan, manifests=self.manifests, root=self.root / "backends", timeout=timeout,
                        runner=lambda argv, **kw: captured.append((argv, kw)) or Completed(stdout=line + "\n"))

        argv = ["convert", "video", "--path", str(model), "--prompt", "A red ball bouncing", "--out", self.out,
                "--width", "128", "--height", "96", "--frames", "5", "--fps", "16", "--steps", "4", "--seed", "7",
                "--timeout", "42", "--json"]
        with mock.patch("mlx_agent.video.load_manifests", return_value=self.manifests), \
                mock.patch("mlx_agent.video.load_registries", return_value=self.registries), \
                mock.patch("mlx_agent.video.is_installed", return_value=True), \
                mock.patch("mlx_agent.video.sync_ports", return_value=[]), \
                mock.patch("mlx_agent.cli.run_video", run):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(argv)
            payload = json.loads(buffer.getvalue())
            self.assertEqual((code, payload["operation"], payload["status"]), (0, "convert-video", "ok"))
            self.assertEqual((payload["data"]["path"], payload["data"]["frames"], payload["data"]["pixel_std"]), (self.out, 5, 55.2))
            self.assertEqual(captured[0][1]["timeout"], 42)
            self.assertEqual(captured[0][0][captured[0][0].index("--frames") + 1], "5")

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(argv[:argv.index("--frames") + 1] + ["6"] + argv[argv.index("--frames") + 2:])
            payload = json.loads(buffer.getvalue())
            self.assertEqual((code, payload["status"], payload["error"]["code"]), (2, "error", "invalid_arguments"))

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(argv[:argv.index("--timeout") + 1] + ["0"] + argv[argv.index("--timeout") + 2:])
            self.assertEqual((code, json.loads(buffer.getvalue())["error"]["code"]), (2, "convert_failed"))


if __name__ == "__main__":
    unittest.main()
