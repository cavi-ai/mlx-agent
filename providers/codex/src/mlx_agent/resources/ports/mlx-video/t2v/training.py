"""Native MLX LoRA optimization for a converted single-branch Wan backbone.

The caller owns dataset preparation, compute admission, and durable artifacts.
This runtime never changes converted model files or writes checkpoint paths.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import asdict

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


class TrainableLoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha, key):
        super().__init__()
        out_features, in_features = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            in_features = in_features * 32 // linear.bits
        self.linear = linear
        self.scale = alpha / rank
        bound = 1 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(
            low=-bound, high=bound, shape=(rank, in_features), key=key
        )
        self.lora_b = mx.zeros((out_features, rank))

    def __call__(self, x):
        baseline = self.linear(x)
        update = (x.astype(mx.float32) @ self.lora_a.T) @ self.lora_b.T
        return baseline + (self.scale * update).astype(baseline.dtype)


def _fingerprint(model):
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(model.config), sort_keys=True).encode())
    for name, value in sorted(tree_flatten(model.parameters())):
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(str(value.shape).encode() + b"\0")
        material = value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
        digest.update(np.asarray(material).tobytes())
    return "sha256:" + digest.hexdigest()


class WanMLXTrainingSession:
    def __init__(self, model, *, targets, rank, alpha, learning_rate, seed=42):
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            raise ValueError("Wan MLX training requires the Metal GPU.")
        if (
            type(rank) is not int
            or not 1 <= rank <= 256
            or type(alpha) is not int
            or not 1 <= alpha <= 4096
            or type(seed) is not int
            or not 0 <= seed < 2**32
            or type(learning_rate) not in (int, float)
            or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 1
        ):
            raise ValueError("Invalid MLX LoRA configuration.")
        if (
            model.config.model_type != "t2v"
            or model.config.model_version != "2.1"
            or model.config.dual_model
            or model.config.in_dim != model.config.out_dim
        ):
            raise ValueError("Only converted single-branch Wan 2.1 T2V is supported.")
        if (
            type(targets) is not tuple
            or not targets
            or targets != tuple(sorted(set(targets)))
            or any(type(name) is not str or not name for name in targets)
        ):
            raise ValueError("Targets must be explicit, sorted and unique.")
        modules = dict(model.named_modules())
        if any(
            name not in modules
            or not isinstance(modules[name], (nn.Linear, nn.QuantizedLinear))
            for name in targets
        ):
            raise ValueError("An unsupported LoRA target was requested.")
        self.base_identity = _fingerprint(model)
        self.model = model
        self.targets = targets
        self.rank, self.alpha = rank, alpha
        self.steps = 0
        self._failed = False
        self._metadata = {
            "schema": "wan-mlx-training-state/1",
            "base": self.base_identity,
            "targets": list(targets),
            "rank": rank,
            "alpha": alpha,
            "learning_rate": float(learning_rate),
            "seed": seed,
        }
        keys = mx.random.split(mx.random.key(seed), len(targets))
        replacements = [
            (name, TrainableLoRALinear(modules[name], rank, alpha, keys[index]))
            for index, name in enumerate(targets)
        ]
        model.freeze()
        model.update_modules(tree_unflatten(replacements))
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0)
        self.optimizer.init(model.trainable_parameters())
        self._require_trainable_boundary()
        self._gradient = nn.value_and_grad(model, self._loss)

    def _require_trainable_boundary(self):
        expected = {
            name + suffix for name in self.targets for suffix in (".lora_a", ".lora_b")
        }
        actual = dict(tree_flatten(self.model.trainable_parameters()))
        if set(actual) != expected:
            raise ValueError("The frozen base boundary changed.")
        return actual

    def _loss(self, model, clean, noise, context, sigma):
        mixture = (1 - sigma[:, None, None, None, None]) * clean + sigma[
            :, None, None, None, None
        ] * noise
        patch = model.config.patch_size
        seq_len = math.prod(
            clean.shape[index + 2] // patch[index] for index in range(3)
        )
        outputs = model(
            list(mixture),
            sigma * model.config.num_train_timesteps,
            list(context),
            seq_len,
        )
        prediction = mx.stack(outputs)
        return mx.mean(mx.square(prediction.astype(mx.float32) - (noise - clean)))

    def train_step(self, clean, noise, context, sigma):
        if self._failed:
            raise ValueError(
                "Restore a valid checkpoint after a failed optimizer update."
            )
        self._require_trainable_boundary()
        patch = self.model.config.patch_size
        if (
            clean.ndim != 5
            or clean.shape != noise.shape
            or clean.shape[0] != 1
            or clean.shape[1] != self.model.config.in_dim
            or any(
                value <= 0 or value % stride
                for value, stride in zip(clean.shape[2:], patch)
            )
            or context.ndim != 3
            or context.shape[0] != clean.shape[0]
            or not 0 < context.shape[1] <= self.model.config.text_len
            or context.shape[2] != self.model.config.text_dim
            or sigma.shape != (clean.shape[0],)
        ):
            raise ValueError("Invalid Wan training batch shape.")
        for value in (clean, noise, context, sigma):
            if (
                value.dtype not in (mx.float16, mx.bfloat16, mx.float32)
                or not mx.all(mx.isfinite(value)).item()
            ):
                raise ValueError(
                    "Training inputs must be finite floating-point tensors."
                )
        if not mx.all((sigma >= 0) & (sigma <= 1)).item():
            raise ValueError("Flow-matching sigma must be between zero and one.")
        loss, gradients = self._gradient(self.model, clean, noise, context, sigma)
        flat = dict(tree_flatten(gradients))
        if set(flat) != set(self._require_trainable_boundary()):
            raise ValueError("Unexpected optimizer parameters.")
        norm = mx.sqrt(
            sum(mx.sum(mx.square(value.astype(mx.float32))) for value in flat.values())
        )
        mx.eval(loss, norm, gradients)
        if not math.isfinite(loss.item()) or not math.isfinite(norm.item()):
            raise ValueError("Non-finite Wan training measurement.")
        try:
            self.optimizer.update(self.model, gradients)
            mx.eval(self.model.trainable_parameters(), self.optimizer.state)
        except Exception:
            self._failed = True
            raise
        self.steps += 1
        return {"step": self.steps, "loss": loss.item(), "gradient_norm": norm.item()}

    def adapter_weights(self):
        parameters = self._require_trainable_boundary()
        result = {}
        for name in self.targets:
            result[name + ".lora_A.weight"] = parameters[name + ".lora_a"]
            result[name + ".lora_B.weight"] = parameters[name + ".lora_b"]
            result[name + ".alpha"] = mx.array(self.alpha, dtype=mx.float32)
        return result

    def _state(self):
        result = {
            "adapter." + key: value
            for key, value in tree_flatten(self.model.trainable_parameters())
        }
        result.update(
            {
                "optimizer." + key: value
                for key, value in tree_flatten(self.optimizer.state)
            }
        )
        result["completed_steps"] = mx.array(self.steps, dtype=mx.uint64)
        return result

    def checkpoint_bytes(self):
        if self._failed:
            raise ValueError("A failed optimizer update cannot be checkpointed.")
        self._require_trainable_boundary()
        values = self._state()
        metadata = json.dumps(self._metadata, sort_keys=True).encode()
        values["metadata"] = mx.array(list(metadata), dtype=mx.uint8)
        output = io.BytesIO()
        mx.savez(output, **values)
        payload = output.getvalue()
        if len(payload) > MAX_CHECKPOINT_BYTES:
            raise ValueError("MLX checkpoint exceeds its supported size.")
        return payload

    def restore_checkpoint(self, payload):
        if type(payload) is not bytes or len(payload) > MAX_CHECKPOINT_BYTES:
            raise ValueError("Invalid MLX checkpoint.")
        expected = self._state()
        expected["metadata"] = mx.zeros(
            (len(json.dumps(self._metadata, sort_keys=True).encode()),), dtype=mx.uint8
        )
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
                if (
                    set(names) != {key + ".npy" for key in expected}
                    or len(names) != len(expected)
                    or any(
                        entry.compress_type != zipfile.ZIP_STORED for entry in entries
                    )
                    or sum(entry.file_size for entry in entries) > MAX_CHECKPOINT_BYTES
                ):
                    raise ValueError
                for entry in entries:
                    reference = expected[entry.filename[:-4]]
                    if entry.file_size > reference.size * reference.dtype.size + 4096:
                        raise ValueError
            restored = mx.load(io.BytesIO(payload), format="npz")
            for key, reference in expected.items():
                value = restored[key]
                if value.shape != reference.shape or value.dtype != reference.dtype:
                    raise ValueError
                if (
                    value.dtype in (mx.float16, mx.bfloat16, mx.float32)
                    and not mx.all(mx.isfinite(value)).item()
                ):
                    raise ValueError
            metadata = bytes(restored["metadata"].tolist()).decode()
            if json.loads(metadata) != self._metadata:
                raise ValueError
            step = int(restored["completed_steps"].item())
            if step != int(restored["optimizer.step"].item()):
                raise ValueError
            parameters = [
                (key[len("adapter.") :], value)
                for key, value in restored.items()
                if key.startswith("adapter.")
            ]
            optimizer = [
                (key[len("optimizer.") :], value)
                for key, value in restored.items()
                if key.startswith("optimizer.")
            ]
        except Exception as error:
            raise ValueError("Invalid or incompatible MLX checkpoint.") from error
        self.model.update(tree_unflatten(parameters))
        self.optimizer.state = tree_unflatten(optimizer)
        self.steps = step
        self._failed = False
        self._require_trainable_boundary()
