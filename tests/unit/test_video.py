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
from mlx_agent import video_runner

from .backend_fixtures import synthetic_manifests, synthetic_registries


FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def result_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


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

    def test_adaptive_cache_options_and_model_boundary(self):
        model = self.model(dict(WAN_CONFIG, model_version="2.1", dual_model=False))
        argv = self.plan(model, denoiser_cache="adaptive", cache_memory_mib=64)["argv"]
        self.assertEqual(argv[-4:], ["--denoiser-cache", "adaptive", "--cache-memory-mib", "64"])
        self.assertNotIn("--denoiser-cache", self.plan(model)["argv"])
        for options in ({"denoiser_cache": "invalid"}, {"cache_memory_mib": True},
                        {"cache_memory_mib": 7}, {"cache_memory_mib": 4097}):
            with self.subTest(options=options), self.assertRaises(VideoError):
                self.plan(model, **options)
        for config in (dict(WAN_CONFIG, model_version="2.2"),
                       dict(WAN_CONFIG, model_version="2.1", dual_model=True)):
            with self.subTest(config=config), self.assertRaises(VideoError):
                self.plan(self.model(config), denoiser_cache="adaptive")

    def test_cache_statistics_survive_the_result_envelope(self):
        model = self.model(dict(WAN_CONFIG, model_version="2.1"))
        plan = self.plan(model, denoiser_cache="adaptive")
        stats = {"mode": "adaptive", "reused_steps": 12, "full_steps": 38}
        with mock.patch("mlx_agent.inference_runner.sync_ports"):
            result = run_video(plan, manifests=self.manifests, root=self.root / "backends",
                               runner=lambda *a, **k: Completed(stdout=json.dumps(dict(RESULT, path=self.out, denoiser_cache=stats))))
        self.assertEqual(result["denoiser_cache"], stats)

    def test_runner_forwards_adaptive_options(self):
        port = mock.Mock()
        port.generate.return_value = {"path": self.out}
        args = ["--module", "test_port", "--model", "/model", "--prompt", "A pan.", "--out", self.out,
                "--width", "320", "--height", "192", "--frames", "33", "--fps", "16", "--steps", "50",
                "--seed", "42", "--denoiser-cache", "adaptive", "--cache-memory-mib", "64"]
        with mock.patch.object(video_runner.importlib, "import_module", return_value=port), \
                mock.patch.object(video_runner, "load_model", return_value=mock.Mock()), redirect_stdout(io.StringIO()):
            self.assertEqual(video_runner.main(args), 0)
        self.assertEqual(port.generate.call_args.kwargs["denoiser_cache"], "adaptive")
        self.assertEqual(port.generate.call_args.kwargs["cache_memory_mib"], 64)

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
        with mock.patch("mlx_agent.inference_runner.sync_ports", return_value=[]) as sync:
            result = run_video(plan, manifests=self.manifests, root=self.root / "backends",
                               runner=lambda argv, **kw: seen.append(kw) or Completed(stdout="Diffusion 1/4\n" + line + "\n"))
            self.assertEqual(sync.call_args.args[0]["id"], "mlx-video")
            self.assertEqual((result["schema"], result["path"], result["pixel_std"], result["seconds_per_frame"]), ("video/1", self.out, 55.2, 2.5))
            self.assertEqual(set(result), set(result_fixture("convert-video.json")))
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
            self.assertEqual(caught.exception.code, "video_timeout")
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
                mock.patch("mlx_agent.inference_runner.sync_ports", return_value=[]), \
                mock.patch("mlx_agent.video.run_video", run):
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


LTX_CONFIG = {"model_type": "ltx2", "model_version": "2.3", "vae_stride": [8, 32, 32], "patch_size": [1, 2, 2],
              "sample_fps": 24, "sample_steps": 8, "spatial_upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"}


class ConditionedVideoTests(unittest.TestCase):
    """A first-frame image and a driving audio file reach the LTX-2.3 port; text-only models refuse them."""

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        bundled = load_manifests()
        self.manifests = dict(synthetic_manifests(), **{name: bundled[name] for name in ("mlx-video", "ltx-video")})
        self.registries = synthetic_registries(self.manifests, installed=("mlx-video", "ltx-video"))
        for backend_id, entry in self.registries.items():
            entry["registry"] = with_ports(entry["registry"], self.manifests[backend_id])
        for backend_id in ("mlx-video", "ltx-video"):
            venv = self.root / "backends" / backend_id
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").write_text("", encoding="utf-8")
            (venv / INSTALL_MARKER).write_text(
                json.dumps({"id": backend_id, "version": self.manifests[backend_id]["version"]}), encoding="utf-8")
        self.image, self.audio = self.root / "first.png", self.root / "speech.wav"
        self.image.write_bytes(b"png")
        self.audio.write_bytes(b"wav")

    def model(self, config):
        directory = self.root / "model-{0}".format(len(list(self.root.glob("model-*"))))
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return directory

    def plan(self, config, **kwargs):
        return plan_video(self.model(config), "a person speaks to the camera", str(self.root / "clip.mp4"),
                          manifests=self.manifests, registries=self.registries, root=self.root / "backends", **kwargs)

    def test_image_and_audio_reach_the_ltx_port(self):
        plan = self.plan(LTX_CONFIG, width=576, height=1024, frames=97, image=str(self.image), audio=str(self.audio),
                         audio_start=1.5)
        argv = plan["argv"]
        self.assertEqual((plan["backend"], argv[argv.index("--module") + 1]), ("ltx-video", "mlx_video.mlx_agent_ports.ltx2"))
        self.assertEqual([argv[argv.index(flag) + 1] for flag in ("--image", "--audio", "--audio-start", "--fps", "--steps")],
                         [str(self.image), str(self.audio), "1.5", "24", "8"])
        self.assertEqual((plan["image"], plan["audio"], plan["audio_start"]), (str(self.image), str(self.audio), 1.5))

    def test_ltx_sizes_align_to_the_half_size_stage(self):
        with self.assertRaises(VideoError) as caught:
            self.plan(LTX_CONFIG, width=600, height=1024, frames=97)
        self.assertIn("multiple of 64", str(caught.exception))

    def test_text_only_models_refuse_an_image_or_audio(self):
        with self.assertRaises(VideoError) as caught:
            self.plan(WAN_CONFIG, width=128, height=96, frames=9, image=str(self.image))
        self.assertEqual(caught.exception.code, "invalid_arguments")
        with self.assertRaises(VideoError):
            self.plan(LTX_CONFIG, width=576, height=1024, frames=97, audio_start=2.0)

    def test_runner_hands_the_conditioning_to_the_port(self):
        port = mock.Mock()
        port.generate.return_value = {"path": "/clip.mp4"}
        args = ["--module", "test_port", "--model", "/model", "--prompt", "A speech.", "--out", "/clip.mp4",
                "--width", "576", "--height", "1024", "--frames", "97", "--fps", "24", "--steps", "8", "--seed", "7",
                "--image", "/first.png", "--audio", "/speech.wav", "--audio-start", "1.5"]
        with mock.patch.object(video_runner.importlib, "import_module", return_value=port), \
                mock.patch.object(video_runner, "load_model", return_value=mock.Mock()), redirect_stdout(io.StringIO()):
            self.assertEqual(video_runner.main(args), 0)
        keywords = port.generate.call_args.kwargs
        self.assertEqual((keywords["image"], keywords["audio"], keywords["audio_start"]), ("/first.png", "/speech.wav", 1.5))


def _ltx_converter():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "src/mlx_agent/resources/ports/ltx-video/ltx2/convert.py"
    spec = importlib.util.spec_from_file_location("ltx2_convert", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# LTX-2.3 snapshots declare 6 residual layers at block 4 and a doubling at block 7; the weights hold
# 4 layers and a 1024 -> 1024 compression (a 128-channel convolution over a 2x2x2 space-to-depth).
PUBLISHED_BLOCKS = [["res_x", {"num_layers": 4}], ["compress_space_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 6}], ["compress_time_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 6}], ["compress_all_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 2}], ["compress_all_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 2}]]


def _encoder_shapes():
    shapes = {"conv_in.conv.weight": (128, 3, 3, 3, 48)}
    for block, count in {0: 4, 2: 6, 4: 4, 6: 2, 8: 2}.items():
        for layer in range(count):
            shapes["down_blocks.{0}.res_blocks.{1}.conv1.conv.weight".format(block, layer)] = (1, 3, 3, 3, 1)
    for block, out in {1: 64, 3: 256, 5: 128, 7: 128}.items():
        shapes["down_blocks.{0}.conv.conv.weight".format(block)] = (out, 3, 3, 3, 1)
    return shapes


class LtxEncoderBlocksTests(unittest.TestCase):
    def test_published_encoder_blocks_are_replaced_by_the_ones_the_weights_hold(self):
        blocks = _ltx_converter().encoder_blocks(PUBLISHED_BLOCKS, _encoder_shapes())
        self.assertEqual(blocks[4], ["res_x", {"num_layers": 4}])
        self.assertEqual(blocks[7], ["compress_all_res", {"multiplier": 1}])
        self.assertEqual(blocks[:4], PUBLISHED_BLOCKS[:4])


class LtxConvertTests(unittest.TestCase):
    def test_conversion_leaves_the_fetched_sources_unchanged(self):
        from mlx_agent.source_evidence import identities

        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        base, encoder, out = root / "ltx", root / "gemma", root / "out"
        files = {"transformer/config.json": "{}", "transformer/model.safetensors": "transformer",
                 "text_projections/model.safetensors": "projections", "vae/decoder/config.json": "{}",
                 "vae/encoder/model.safetensors": "encoder", "audio_vae/encoder/config.json": "{}",
                 "vocoder/config.json": "{}", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors": "upscaler",
                 "vae/encoder/config.json": json.dumps({"encoder_blocks": PUBLISHED_BLOCKS})}
        text = {"config.json": json.dumps({"text_config": {"hidden_size": 3840, "num_hidden_layers": 48}}),
                "model.safetensors": "gemma", "tokenizer.json": "{}", "tokenizer_config.json": "{}", "tokenizer.model": "spm"}
        for folder, contents in ((base, files), (encoder, text)):
            for name, data in contents.items():
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_text(data, encoding="utf-8")
        before = identities(base), identities(encoder)
        module = _ltx_converter()
        with mock.patch.object(module, "_shapes", return_value=_encoder_shapes()), redirect_stdout(io.StringIO()):
            module.convert(str(base), str(out), {module.TEXT_ENCODER: str(encoder)})
        self.assertEqual((identities(base), identities(encoder)), before)
        self.assertEqual((out / "transformer/model.safetensors").read_text(encoding="utf-8"), "transformer")
        self.assertEqual((out / "text_encoder/model.safetensors").read_text(encoding="utf-8"), "gemma")
        self.assertEqual((out / "tokenizer/tokenizer.model").read_text(encoding="utf-8"), "spm")
        fixed = json.loads((out / "vae/encoder/config.json").read_text(encoding="utf-8"))["encoder_blocks"]
        self.assertEqual(fixed[7], ["compress_all_res", {"multiplier": 1}])
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["model_type"], config["spatial_upscaler"]),
                         ("ltx2", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"))


if __name__ == "__main__":
    unittest.main()
