"""Opt-in numerical coverage for the pinned backend's Prism packed kernels.

Run with the installed mlx-vlm backend Python and MLX_AGENT_PRISM_KERNELS=1.
No checkpoint download or model-repository code is needed.
"""
import os
import unittest


def reference_rotation(value, block, signs, inverse=False):
    """CPU butterfly reference independent of the Metal Hadamard operator."""
    import numpy as np
    x = np.array(value.tolist(), dtype=np.float32)
    shape = x.shape
    sign = np.array(signs.tolist(), dtype=np.float32)
    if not inverse:
        x *= sign
    x = x.reshape(-1, block)
    width = 1
    while width < block:
        pairs = x.reshape(-1, block // (2 * width), 2, width)
        left, right = pairs[:, :, 0].copy(), pairs[:, :, 1].copy()
        pairs[:, :, 0], pairs[:, :, 1] = left + right, left - right
        width *= 2
    x = x.reshape(shape) / np.sqrt(block)
    if inverse:
        x *= sign
    return x


@unittest.skipUnless(os.environ.get("MLX_AGENT_PRISM_KERNELS") == "1", "requires the installed MLX-VLM Metal backend")
class PrismKernelTests(unittest.TestCase):
    def test_packed_linear_matches_dequantized_rotated_reference(self):
        import mlx.core as mx
        from mlx_vlm.models.prism_hadamard_qwen35.prism_hadamard_qwen35 import (
            HadamardQuantizedLinear,
        )
        for block in (0, 512, 1024, 2048, 4096):
            with self.subTest(block=block):
                width = max(block, 128)
                dense = mx.sin(mx.arange(4 * width).reshape(4, width).astype(mx.float32))
                layer = HadamardQuantizedLinear(width, 4, block)
                layer.weight, layer.scales, layer.biases = mx.quantize(dense, group_size=128, bits=2)
                x = mx.cos(mx.arange(width).astype(mx.float32)).reshape(1, width)
                if block:
                    layer.signs = mx.where(mx.arange(width) % 2, -1., 1.)
                    rotated = mx.array(reference_rotation(x, block, layer.signs), dtype=mx.float32)
                else:
                    rotated = x
                reference = rotated @ mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=128, bits=2).T
                self.assertTrue(mx.allclose(layer(x), reference, atol=1e-3, rtol=1e-3).item())

    def test_embedding_applies_inverse_rotation_to_selected_rows(self):
        import mlx.core as mx
        from mlx_vlm.models.prism_hadamard_qwen35.prism_hadamard_qwen35 import (
            HadamardQuantizedEmbedding,
        )
        layer = HadamardQuantizedEmbedding(512, 4, 512)
        dense = mx.sin(mx.arange(4 * 512).reshape(4, 512).astype(mx.float32))
        layer.weight, layer.scales, layer.biases = mx.quantize(dense, group_size=128, bits=2)
        layer.signs = mx.where(mx.arange(512) % 2, -1., 1.)
        indices = mx.array([[3, 1]])
        rows = mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=128, bits=2)[indices].astype(mx.float16)
        reference = mx.array(reference_rotation(rows, 512, layer.signs, inverse=True), dtype=mx.float16)
        self.assertTrue(mx.allclose(layer(indices), reference, atol=1e-3, rtol=1e-3).item())
