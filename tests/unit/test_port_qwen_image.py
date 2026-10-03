"""The bundled Qwen-Image 2.1 port; runs where mflux is importable (the backend venv)."""

import importlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.backends import PORTS_DIR

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


if __name__ == "__main__":
    unittest.main()
