"""Inference-only Qwen2 affine Q4 prefill; imported only by the guarded runtime."""

import copy
import logging
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen2 import MLP


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mlx_agent_m5_q4_prefill",
        input_names=["activations", "packed", "scales", "biases"],
        output_names=["output"],
        header=Path(__file__).with_name("resources").joinpath("m5_prefill.metal").read_text(),
        source="""
            const int rows = activations_shape[0];
            threadgroup half weights[2 * 64 * 72];
            mlx_agent_nax::qmm_t_nax_rows16_impl<half, 64, 4>(
                packed, scales, biases, activations, output, weights, K, N, rows,
                threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
        """,
    )


def product(activations, packed, scales, biases):
    """Evaluate a supported matrix product; never dispatch empty device buffers."""
    if activations.ndim != 2 or packed.ndim != 2:
        raise ValueError("M5 prefill expects two-dimensional matrices")
    rows, width = activations.shape
    outputs = packed.shape[0]
    if (rows <= 0 or width <= 0 or outputs <= 0 or width % 64 or outputs % 64
            or activations.dtype != mx.float16 or packed.dtype != mx.uint32
            or packed.shape != (outputs, width // 8)
            or scales.shape != (outputs, width // 64) or biases.shape != scales.shape
            or scales.dtype != mx.float16 or biases.dtype != mx.float16):
        raise ValueError("Unsupported M5 prefill matrix layout")
    if rows < 128:
        return mx.quantized_matmul(activations, packed, scales, biases,
                                   transpose=True, group_size=64, bits=4)
    return _kernel()(
        inputs=[activations, packed, scales, biases], template=[("K", width), ("N", outputs)],
        grid=(128 * (outputs // 64), (rows + 63) // 64, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows, outputs)], output_dtypes=[mx.float16],
    )[0]


def supported(projection):
    if (type(projection) is not nn.QuantizedLinear or "bias" in projection
            or projection.group_size != 64 or projection.bits != 4 or projection.mode != "affine"
            or projection.weight.ndim != 2 or projection.biases is None):
        return False
    outputs, packed_width = projection.weight.shape
    return (outputs > 0 and packed_width > 0 and outputs % 64 == 0 and packed_width % 8 == 0
            and projection.weight.dtype == mx.uint32
            and projection.scales.shape == (outputs, packed_width // 8)
            and projection.biases.shape == projection.scales.shape
            and projection.scales.dtype == mx.float16 and projection.biases.dtype == mx.float16)


class PrefillLinear(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight, self.scales, self.biases = original.weight, original.scales, original.biases
        self.freeze()
        self.train(original.training)

    def __call__(self, x):
        rows = x.size // x.shape[-1]
        if not self.training and x.dtype == mx.float16 and rows >= 128:
            y = product(x.reshape(rows, x.shape[-1]), self.weight, self.scales, self.biases)
            return y.reshape(*x.shape[:-1], self.weight.shape[0])
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                   transpose=True, group_size=64, bits=4)


def apply(model):
    """Keep original storage and replace only supported inference MLP projections."""
    candidates = []
    names = ("gate_proj", "up_proj", "down_proj")
    for layer in getattr(model, "layers", ()):
        mlp = getattr(layer, "mlp", None)
        if type(mlp) is MLP and not mlp.training and all(supported(getattr(mlp, n)) for n in names):
            candidates.append((layer, mlp))
    # Compile every selected shape before changing the model. A compiler/runtime
    # without this MPP implementation retains the original model in its entirety.
    warmed = set()
    try:
        for _, mlp in candidates:
            for name in names:
                p = getattr(mlp, name)
                shape = p.weight.shape
                if shape not in warmed:
                    x = mx.zeros((128, shape[1] * 8), dtype=mx.float16)
                    mx.eval(product(x, p.weight, p.scales, p.biases))
                    warmed.add(shape)
    except RuntimeError as error:
        logging.warning("M5 prompt kernel could not compile; using native MLX: %s", error)
        return 0
    for layer, mlp in candidates:
        adapted = copy.copy(mlp)
        for name in names:
            setattr(adapted, name, PrefillLinear(getattr(mlp, name)))
        layer.mlp = adapted
    return len(candidates)
