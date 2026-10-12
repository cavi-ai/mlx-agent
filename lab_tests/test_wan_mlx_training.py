"""Real MLX gradient and checkpoint checks for converted Wan training."""

import importlib.util
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_video.models.wan.model")
from mlx.utils import tree_flatten
from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.model import WanModel

PORT = (
    Path(__file__).resolve().parents[1]
    / "src/mlx_agent/resources/ports/mlx-video/t2v/training.py"
)


def runtime():
    assert PORT.is_file(), "The converted-MLX training runtime is missing."
    spec = importlib.util.spec_from_file_location("wan_mlx_training", PORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model():
    config = WanModelConfig(
        model_version="2.1",
        dual_model=False,
        dim=128,
        ffn_dim=256,
        num_heads=2,
        num_layers=1,
        text_dim=32,
        text_len=8,
        freq_dim=32,
    )
    mx.random.seed(7)
    result = WanModel(config)
    nn.quantize(
        result,
        group_size=64,
        bits=4,
        class_predicate=lambda path, layer: (
            isinstance(layer, nn.Linear) and layer.weight.shape[-1] % 64 == 0
        ),
    )
    mx.eval(result.parameters())
    return result


def batch():
    return (
        mx.zeros((1, 16, 1, 4, 4)),
        mx.random.normal((1, 16, 1, 4, 4), key=mx.random.key(11)) * 0.25,
        mx.zeros((1, 4, 32)),
        mx.array([0.5]),
    )


def test_real_quantized_wan_step_changes_only_lora_parameters():
    module = runtime()
    backbone = model()
    original = {
        key: mx.array(value) for key, value in tree_flatten(backbone.parameters())
    }
    session = module.WanMLXTrainingSession(
        backbone, targets=("blocks.0.self_attn.q",), rank=4, alpha=4, learning_rate=1e-3
    )
    measured = session.train_step(*batch())
    assert measured["gradient_norm"] > 0
    assert measured["loss"] >= 0
    weights = dict(tree_flatten(backbone.parameters()))
    assert mx.any(weights["blocks.0.self_attn.q.lora_b"] != 0).item()
    for key, expected in original.items():
        actual_key = key.replace(
            "blocks.0.self_attn.q.", "blocks.0.self_attn.q.linear."
        )
        assert mx.array_equal(weights[actual_key], expected).item(), key


def test_persisted_optimizer_restores_exact_training_continuation(tmp_path):
    module = runtime()
    first = module.WanMLXTrainingSession(
        model(), targets=("blocks.0.self_attn.q",), rank=4, alpha=4, learning_rate=1e-3
    )
    first.train_step(*batch())
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(first.checkpoint_bytes())
    expected = first.train_step(*batch())
    restored = module.WanMLXTrainingSession(
        model(), targets=("blocks.0.self_attn.q",), rank=4, alpha=4, learning_rate=1e-3
    )
    restored.restore_checkpoint(checkpoint.read_bytes())
    actual = restored.train_step(*batch())
    assert actual["step"] == 2
    assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-6)
    for key, value in first.adapter_weights().items():
        assert mx.allclose(value, restored.adapter_weights()[key], atol=1e-6).item()


def test_invalid_targets_leave_converted_model_unchanged():
    module = runtime()
    backbone = model()
    original = {
        key: mx.array(value) for key, value in tree_flatten(backbone.parameters())
    }
    with pytest.raises(ValueError):
        module.WanMLXTrainingSession(
            backbone,
            targets=("blocks.0.self_attn.q", "missing"),
            rank=4,
            alpha=4,
            learning_rate=1e-3,
        )
    after = dict(tree_flatten(backbone.parameters()))
    assert after.keys() == original.keys()
    for key, value in original.items():
        assert mx.array_equal(after[key], value).item()


def test_exported_adapter_is_accepted_by_the_native_inference_loader(tmp_path):
    from mlx_video.lora.apply import apply_loras_to_model
    from mlx_video.lora.loader import load_lora_weights

    module = runtime()
    session = module.WanMLXTrainingSession(
        model(), targets=("blocks.0.self_attn.q",), rank=4, alpha=8, learning_rate=1e-3
    )
    session.train_step(*batch())
    path = tmp_path / "adapter.safetensors"
    mx.save_safetensors(path, session.adapter_weights())
    restored = load_lora_weights(path)
    assert set(restored) == {"blocks.0.self_attn.q"}
    assert restored["blocks.0.self_attn.q"].rank == 4
    assert restored["blocks.0.self_attn.q"].alpha == 8
    independent = model()
    apply_loras_to_model(
        independent, {"blocks.0.self_attn.q": [(restored["blocks.0.self_attn.q"], 1.0)]}
    )
    _clean, noise, context, sigma = batch()
    expected = session.model(list(noise), sigma * 1000, list(context), 4)[0]
    actual = independent(list(noise), sigma * 1000, list(context), 4)[0]
    assert mx.allclose(expected, actual, atol=1e-4, rtol=1e-4).item()


def test_checkpoint_with_different_base_is_rejected_before_state_changes():
    module = runtime()
    first = module.WanMLXTrainingSession(
        model(), targets=("blocks.0.self_attn.q",), rank=4, alpha=4, learning_rate=1e-3
    )
    first.train_step(*batch())
    different = model()
    different.config.sample_shift += 1
    second = module.WanMLXTrainingSession(
        different,
        targets=("blocks.0.self_attn.q",),
        rank=4,
        alpha=4,
        learning_rate=1e-3,
    )
    before = second.checkpoint_bytes()
    with pytest.raises(ValueError):
        second.restore_checkpoint(first.checkpoint_bytes())
    assert second.checkpoint_bytes() == before
    assert second.steps == 0


def test_nonfinite_batch_is_rejected_before_optimizer_update():
    module = runtime()
    session = module.WanMLXTrainingSession(
        model(), targets=("blocks.0.self_attn.q",), rank=4, alpha=4, learning_rate=1e-3
    )
    clean, noise, context, sigma = batch()
    before = session.checkpoint_bytes()
    with pytest.raises(ValueError):
        session.train_step(clean, noise * float("nan"), context, sigma)
    assert session.checkpoint_bytes() == before
