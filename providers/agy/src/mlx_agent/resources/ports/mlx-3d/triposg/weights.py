"""Checkpoint operations shared by the native TripoSG components."""

import mlx.core as mx
import mlx.nn as nn


class Weights:
    def __init__(self, values, bits=8, group_size=64):
        self.values = values
        self.bits = bits
        self.group_size = group_size

    def __getitem__(self, name):
        return self.values[name]

    def linear(self, name, x):
        weight = self.values[name + ".weight"]
        if name + ".scales" in self.values:
            y = mx.quantized_matmul(x, weight, self.values[name + ".scales"],
                                    self.values[name + ".biases"], transpose=True,
                                    group_size=self.group_size, bits=self.bits)
        else:
            y = x @ weight.T
        bias = self.values.get(name + ".bias")
        return y if bias is None else y + bias

    def norm(self, name, x, eps=1e-5):
        # FP32LayerNorm in the reference normalizes in float32 and casts back.
        y = mx.fast.layer_norm(x.astype(mx.float32), self[name + ".weight"].astype(mx.float32),
                               self[name + ".bias"].astype(mx.float32), eps)
        return y.astype(x.dtype)

    def rms(self, name, x):
        return mx.fast.rms_norm(x, self[name + ".weight"], 1e-6)

    def feed_forward(self, name, x):
        return self.linear(name + ".net.2", nn.gelu(self.linear(name + ".net.0.proj", x)))
