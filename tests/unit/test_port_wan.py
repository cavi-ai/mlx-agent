"""The bundled Wan text-to-video port: converter refusals and model validation run in any interpreter."""

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mlx_agent.backends import PORTS_DIR

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_HUB = importlib.util.find_spec("huggingface_hub") is not None
PORT = PORTS_DIR / "mlx-video"


def load_port():
    if str(PORT) not in sys.path:
        sys.path.insert(0, str(PORT))
    import importlib

    return importlib.import_module("t2v.convert"), importlib.import_module("t2v.t2v")


class WanPortTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.convert, self.t2v = load_port()

    def checkpoint(self, config=None, drop=(), name="checkpoint"):
        base = self.root / name
        (base / "google" / "umt5-xxl").mkdir(parents=True)
        (base / "config.json").write_text(json.dumps({"model_type": "t2v"} if config is None else config), encoding="utf-8")
        for file in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "diffusion_pytorch_model.safetensors"):
            (base / file).write_bytes(b"x")
        for file in self.convert.TOKENIZER_FILES:
            (base / "google" / "umt5-xxl" / file).write_bytes(b"x")
        for file in drop:
            (base / file).unlink()
        return base

    def refusal(self, base, out="out"):
        with self.assertRaises(SystemExit) as caught:
            self.convert.convert(str(base), str(self.root / out))
        return str(caught.exception)

    def test_the_converter_refuses_other_checkpoints_before_it_loads_anything(self):
        if HAS_HUB:
            self.assertIn("not a checkpoint directory", self.refusal(self.root / "missing"))
        self.assertIn("not a Wan checkpoint", self.refusal(self.checkpoint(drop=("config.json",), name="no-config")))
        self.assertIn("'i2v' checkpoint", self.refusal(self.checkpoint({"model_type": "i2v"}, name="i2v")))
        self.assertIn("models_t5_umt5-xxl-enc-bf16.pth", self.refusal(self.checkpoint(drop=("models_t5_umt5-xxl-enc-bf16.pth",), name="no-t5")))
        self.assertIn("Wan2.1_VAE.pth", self.refusal(self.checkpoint(drop=("Wan2.1_VAE.pth",), name="no-vae")))
        self.assertIn("diffusion_pytorch_model*.safetensors", self.refusal(self.checkpoint(drop=("diffusion_pytorch_model.safetensors",), name="no-dit")))
        broken = self.checkpoint(name="no-tokenizer")
        (broken / "google" / "umt5-xxl" / "spiece.model").unlink()
        self.assertIn("spiece.model", self.refusal(broken))
        occupied = self.root / "occupied"
        occupied.mkdir()
        (occupied / "model.safetensors").write_bytes(b"x")
        self.assertIn("not empty", self.refusal(self.checkpoint(name="full"), out="occupied"))

    def test_load_requires_a_complete_converted_directory(self):
        directory = self.root / "converted"
        directory.mkdir()
        with self.assertRaises(ValueError):
            self.t2v.load(directory)
        (directory / "config.json").write_text(json.dumps({"model_type": "t2v"}), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            self.t2v.load(directory)
        self.assertIn("model.safetensors", str(caught.exception))
        for file in ("t5_encoder.safetensors", "vae.safetensors", "model.safetensors"):
            (directory / file).write_bytes(b"x")
        self.assertEqual(self.t2v.load(directory).config["model_type"], "t2v")
        (directory / "config.json").write_text(json.dumps({"model_type": "t2v", "dual_model": True}), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            self.t2v.load(directory)
        self.assertIn("low_noise_model.safetensors", str(caught.exception))

    def fake_backend(self, hub_names):
        """Stand-ins for the backend modules ``generate`` imports; the fake pipeline loads its tokenizer the way mlx-video does."""
        class AutoTokenizer:
            received = []

            @classmethod
            def from_pretrained(cls, name, *arguments, **keywords):
                cls.received.append(name)
                return name

        class Video:
            shape = (3, 4, 6, 3)

        wan = types.ModuleType("mlx_video.generate_wan")
        for loader in self.t2v.LOADERS:
            setattr(wan, loader, lambda *arguments, **keywords: None)
        wan.requests = []
        wan.save_video = lambda *arguments, **keywords: self.fail("the unpatched save_video ran")

        def generate_video(**keywords):
            wan.requests.append(keywords)
            for name in hub_names:
                AutoTokenizer.from_pretrained(name)
            wan.save_video(Video(), keywords["output_path"], fps=16)

        wan.generate_video = generate_video
        core = types.ModuleType("mlx.core")
        core.reset_peak_memory = lambda: None
        core.get_peak_memory = lambda: 2_000_000_000
        package = types.ModuleType("mlx")
        package.core = core
        video_package = types.ModuleType("mlx_video")
        video_package.generate_wan = wan
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = AutoTokenizer
        modules = {"mlx": package, "mlx.core": core, "mlx_video": video_package,
                   "mlx_video.generate_wan": wan, "transformers": transformers}
        return modules, AutoTokenizer, wan

    def run_generate(self, hub_names, with_tokenizer=True):
        converted = self.root / "converted"
        converted.mkdir(parents=True)
        if with_tokenizer:
            (converted / "tokenizer").mkdir()
        modules, tokenizer, wan = self.fake_backend(hub_names)
        out = str(self.root / "videos" / "clip.mp4")
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(self.t2v, "_encode") as encode, \
                mock.patch.object(self.t2v, "_pixel_std", return_value=12.5):
            result = self.t2v.generate(self.t2v.Model(converted, {"model_type": "t2v"}), "a red square", out,
                                       width=6, height=4, frames=3, fps=8, steps=2, seed=7)
        return converted, out, result, tokenizer, wan, encode

    def test_generate_redirects_the_hub_tokenizer_to_the_converted_copy_and_writes_the_out_path(self):
        converted, out, result, tokenizer, wan, encode = self.run_generate(["google/umt5-xxl", "other/repo"])
        self.assertEqual(tokenizer.received, [str(converted / "tokenizer"), "other/repo"])
        encode.assert_called_once()
        self.assertEqual(encode.call_args.args[1:], (out, 8))
        self.assertEqual(wan.requests[0]["output_path"], out)
        self.assertEqual(wan.requests[0]["model_dir"], str(converted))
        self.assertEqual((result["path"], result["width"], result["height"], result["frames"], result["fps"]),
                         (out, 6, 4, 3, 8))
        self.assertEqual((result["pixel_std"], result["peak_memory_gb"], result["steps"], result["seed"]), (12.5, 2.0, 2, 7))

    def test_generate_leaves_the_hub_tokenizer_alone_when_the_converted_copy_is_missing(self):
        _, _, _, tokenizer, _, _ = self.run_generate(["google/umt5-xxl"], with_tokenizer=False)
        self.assertEqual(tokenizer.received, ["google/umt5-xxl"])

    @unittest.skipUnless(HAS_NUMPY, "numpy is not installed in this interpreter")
    def test_pixel_std_tells_a_blank_video_from_a_varied_one(self):
        import numpy as np

        blank = np.full((3, 4, 4, 3), 128, dtype=np.uint8)
        varied = np.zeros((2, 2, 2, 3), dtype=np.uint8)
        varied[1] = 255
        self.assertEqual(self.t2v._pixel_std(blank), 0.0)
        self.assertAlmostEqual(self.t2v._pixel_std(varied), float(varied.astype(np.float64).std()), places=6)


if __name__ == "__main__":
    unittest.main()
