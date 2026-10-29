import unittest
import platform
import sys
from types import SimpleNamespace
from unittest.mock import patch

from mlx_agent.prefill_runtime import identity, runtime_supported

class M5PrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Discovery must leave the portable core independent of the GPU runtime.
        global mx, nn, MLP, PrefillLinear, apply, product, supported
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm.models.qwen2 import MLP
            from mlx_agent._m5_prefill import PrefillLinear, apply, product, supported
        except ImportError:
            raise unittest.SkipTest("Requires the supported M5/MLX runtime")
        if not runtime_supported(sys.platform, platform.mac_ver()[0], identity()["versions"],
                                 mx.device_info().get("architecture")):
            raise unittest.SkipTest("Requires the supported M5/MLX runtime")

    def tensors(self, rows, unaligned=False):
        mx.random.seed(23 + rows)
        if unaligned:
            x = mx.random.normal((rows * 128 + 1,)).astype(mx.float16)[1:].reshape(rows, 128)
        else:
            x = mx.random.normal((rows, 128)).astype(mx.float16)
        weights = mx.quantize(mx.random.normal((256, 128)).astype(mx.float16), group_size=64, bits=4)
        return x, weights

    def test_tail_tiles_strided_and_unaligned_inputs(self):
        for rows in (128, 129, 145, 191, 192, 193, 257):
            for layout in ("contiguous", "unaligned", "strided"):
                with self.subTest(rows=rows, layout=layout):
                    x, weights = self.tensors(rows, unaligned=layout == "unaligned")
                    if layout == "strided":
                        x = mx.stack([x, x], axis=-1)[..., 0]
                    expected = mx.quantized_matmul(x, *weights, transpose=True, group_size=64, bits=4)
                    actual = product(x, *weights)
                    relative = mx.linalg.norm(actual.astype(mx.float32) - expected.astype(mx.float32)) / mx.linalg.norm(expected.astype(mx.float32))
                    self.assertLess(relative.item(), 0.001)

    def test_short_sequences_are_exactly_native(self):
        for rows in (1, 17, 63, 64, 65):
            x, weights = self.tensors(rows)
            expected = mx.quantized_matmul(x, *weights, transpose=True, group_size=64, bits=4)
            self.assertTrue(mx.array_equal(product(x, *weights), expected).item())

    def test_empty_dimensions_and_wrong_layout_are_rejected(self):
        for rows, width, outputs in ((128, 0, 128), (128, 128, 0), (0, 128, 128), (128, 128, 65)):
            with self.subTest(rows=rows, width=width, outputs=outputs), self.assertRaises(ValueError):
                product(mx.zeros((rows, width), mx.float16), mx.zeros((outputs, width // 8), mx.uint32),
                        mx.zeros((outputs, width // 64), mx.float16), mx.zeros((outputs, width // 64), mx.float16))

    def model(self):
        mlp = MLP(128, 256)
        nn.quantize(mlp, group_size=64, bits=4)
        mlp.set_dtype(mx.float16)
        mlp.eval()
        mx.eval(mlp.parameters())
        return SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)])

    def test_adapter_keeps_storage_decode_and_inference_semantics(self):
        model = self.model()
        original = model.layers[0].mlp
        self.assertEqual(apply(model), 1)
        adapted = model.layers[0].mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            a, b = getattr(original, name), getattr(adapted, name)
            self.assertIsInstance(b, PrefillLinear)
            for key in ("weight", "scales", "biases"):
                self.assertIs(getattr(a, key), getattr(b, key))
        x, _ = self.tensors(1)
        self.assertTrue(mx.array_equal(original(x), adapted(x)).item())
        self.assertEqual(apply(model), 0)

    def test_bad_scales_training_and_compile_failure_leave_model_native(self):
        model = self.model()
        original = model.layers[0].mlp
        original.gate_proj.scales = original.gate_proj.scales[:, :1]
        self.assertFalse(supported(original.gate_proj))
        self.assertEqual(apply(model), 0)
        model = self.model()
        original = model.layers[0].mlp
        original.train()
        self.assertEqual(apply(model), 0)
        original.eval()
        with patch("mlx_agent._m5_prefill.product", side_effect=RuntimeError("unsupported compiler")):
            self.assertEqual(apply(model), 0)
        self.assertIs(model.layers[0].mlp, original)


if __name__ == "__main__":
    program = unittest.main(exit=False)
    if program.result.testsRun == 0 and program.result.skipped:
        raise SystemExit(77)
    raise SystemExit(not program.result.wasSuccessful())
