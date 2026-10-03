"""The bundled Wan text-to-video port: converter refusals and model validation run in any interpreter."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
