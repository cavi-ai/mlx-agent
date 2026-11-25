"""The bundled Qwen-Image 2.1 port; runs where mflux is importable (the backend venv)."""

import importlib
import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.backends import PORTS_DIR
from mlx_agent.generate_runner import main as generate_main

HAS_MFLUX = importlib.util.find_spec("mflux") is not None


def load_port():
    sys.path.insert(0, str(PORTS_DIR / "mflux"))
    try:
        return importlib.import_module("qwen_image_21.qwen_image_21"), importlib.import_module("qwen_image_21.convert")
    finally:
        sys.path.pop(0)


@unittest.skipUnless(HAS_MFLUX, "mflux is not installed in this interpreter")
class QwenImagePortTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx

        self.mx = mx
        self.port, self.convert = load_port()

    def tree(self):
        mx = self.mx
        mx.random.seed(0)
        block = lambda: {"attn": {name: {"weight": mx.random.normal((8, 6)).astype(mx.bfloat16)} for name in ("to_q", "to_k", "to_v")}
                         | {"to_out": [{"weight": mx.random.normal((8, 6)).astype(mx.bfloat16)}]}}
        return {"transformer_blocks": [block(), block()]}

    def test_merge_is_base_plus_scale_b_at_a_rounded_to_the_weight_dtype(self):
        mx = self.mx
        tree = self.tree()
        original = tree["transformer_blocks"][1]["attn"]["to_out"][0]["weight"]
        a, b = mx.random.normal((2, 6)).astype(mx.bfloat16), mx.random.normal((8, 2)).astype(mx.bfloat16)
        lora = {"transformer_blocks.1.attn.to_out.0.lora_A.default.weight": a,
                "transformer_blocks.1.attn.to_out.0.lora_B.default.weight": b}
        self.assertEqual(self.port.merge_lora(tree, lora, 0.5), 1)
        merged = tree["transformer_blocks"][1]["attn"]["to_out"][0]["weight"]
        expected = (original.astype(mx.float32) + 0.5 * (b.astype(mx.float32) @ a.astype(mx.float32))).astype(mx.bfloat16)
        self.assertEqual(merged.dtype, mx.bfloat16)
        self.assertTrue(mx.array_equal(merged, expected).item())
        self.assertFalse(mx.array_equal(tree["transformer_blocks"][0]["attn"]["to_out"][0]["weight"], merged).item())

    def test_merge_refuses_a_bad_lora(self):
        mx = self.mx
        a, b = mx.zeros((2, 6)), mx.zeros((8, 2))
        cases = [
            {"transformer_blocks.0.attn.to_q.lora_A.default.weight": a},
            {"transformer_blocks.0.attn.to_q.lora_A.default.weight": mx.zeros((2, 5)), "transformer_blocks.0.attn.to_q.lora_B.default.weight": b},
            {"norm_out.linear.lora_A.default.weight": a, "norm_out.linear.lora_B.default.weight": b},
            {},
        ]
        for lora in cases:
            with self.subTest(keys=sorted(lora)):
                with self.assertRaises(ValueError):
                    self.port.merge_lora(self.tree(), lora, 1.0)

    def test_convert_refuses_another_pipeline_or_a_used_destination(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base").mkdir()
            (root / "base" / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}), encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                self.convert.convert(str(root / "base"), str(root / "out"), quantize=True, q_bits=8)
            self.assertIn("FluxPipeline", str(caught.exception))
            (root / "used").mkdir()
            (root / "used" / "keep").write_text("x", encoding="utf-8")
            with self.assertRaises(SystemExit):
                self.convert.convert(str(root / "base"), str(root / "used"))

    def edit_result(self, width, height, prompt, seed, num_inference_steps, guidance, image_paths=()):
        from mflux.models.common.config import ModelConfig
        from mflux.models.qwen21.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
        from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
        from mflux.utils.generated_image import GeneratedImage
        from PIL import Image

        QwenImage21LatentCreator.validate(width, height, num_inference_steps, len(image_paths))
        image = Image.new("RGBA", (width, height), "red")
        image.putpixel((8, 0), (0, 0, 255, 255))
        return GeneratedImage(
            image, ModelConfig.qwen_image_21(), seed, prompt, num_inference_steps, guidance,
            self.mx.bfloat16, 8, 0.1, width=width, height=height,
            image_paths=[QwenImage21Edit._recorded_path(reference) for reference in image_paths],
        )

    def test_text_only_edit_checkpoint_writes_the_requested_size(self):
        from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
        from PIL import Image

        with TemporaryDirectory() as directory:
            output = Path(directory) / "render.png"
            model = QwenImage21Edit.__new__(QwenImage21Edit)
            with patch.object(QwenImage21Edit, "generate_image", side_effect=self.edit_result):
                result = self.port.generate(model, "test", str(output), width=784, height=512, steps=2)
            with Image.open(output) as image:
                self.assertEqual(image.size, (784, 512))
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 255, 255))
                metadata = json.loads(image.getexif().get_ifd(0x8769)[0x9286][8:])
            self.assertEqual((metadata["width"], metadata["height"]), (784, 512))
            self.assertEqual((result["width"], result["height"], result["steps"]), (784, 512, 2))

    def test_reference_png_does_not_embed_source_paths(self):
        from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
        from PIL import Image

        with TemporaryDirectory() as directory:
            reference = Path(directory) / "private-reference.png"
            output = Path(directory) / "render.png"
            Image.new("RGB", (32, 32)).save(reference)
            model = QwenImage21Edit.__new__(QwenImage21Edit)
            with patch.object(QwenImage21Edit, "generate_image", side_effect=self.edit_result):
                result = self.port.generate(model, "test", str(output), width=256, height=256, steps=2, images=[reference])
            with Image.open(output) as image:
                metadata = json.loads(image.getexif().get_ifd(0x8769)[0x9286][8:])
            self.assertNotIn(reference.name, json.dumps(metadata))
            self.assertEqual(result["images"], [str(reference)])

    def test_runner_refuses_invalid_reference_before_model_load(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jpg"
            reference.write_text("not an image", encoding="utf-8")
            argv = ["--module", "qwen_image_21", "--model", str(root / "model"), "--prompt", "test",
                    "--out", str(root / "out.png"), "--width", "256", "--height", "256", "--steps", "2",
                    "--seed", "42", "--image", str(reference)]
            output = io.StringIO()
            with patch("mlx_agent.generate_runner.load_model", side_effect=AssertionError("model loaded")), redirect_stdout(output):
                self.assertNotEqual(generate_main(argv), 0)
            refusal = json.loads(output.getvalue())
            self.assertEqual(refusal["error"], "invalid_arguments")
            self.assertIn(reference.name, refusal["detail"])


if __name__ == "__main__":
    unittest.main()
