from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from video_lab.compilers.wan_curated import (
    WanCompilerComponents,
    WanCompilerLimits,
    WanCompilerSnapshot,
    WanCuratedBatchCompiler,
    encode_wan_curated_source,
)
from video_lab.curated_models import (
    SequenceTransformRecipeV1,
    StructuredCaptionV1,
    render_caption_v1,
)
from video_lab.dataset_compiler import (
    TransformInputReceipt,
    TransformReceipt,
    transformed_array_digest,
)
from video_lab.domain import canonical_json
from video_lab.trainers.wan_batch import (
    WAN_BATCH_TENSORS,
    validate_wan_batch_bytes,
    validated_safetensors_header,
    validated_safetensors_header_prefix,
)
from video_lab.trainers.wan_diffusers_runtime import validate_wan_batch_artifact
from video_lab.trainers.wan_factory import (
    WanPinnedLocalRoot,
    WanSnapshotCopyLimits,
    require_component_on_mps,
    root_manifest_digest,
    secure_snapshot_pinned_root,
)


class _Tokenizer:
    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        assert kwargs == {"padding": True, "return_tensors": "pt", "truncation": True}
        assert texts == [render_caption_v1(_caption())]
        return {"input_ids": torch.tensor([[7, 11]], dtype=torch.int64)}


class _Encoder:
    hf_device_map = {"": "mps"}
    device = "mps"

    def parameters(self):
        return iter((_Placed(),))

    def buffers(self):
        return iter((_Placed(),))

    def __call__(self, *, input_ids: torch.Tensor) -> tuple[torch.Tensor]:
        assert torch.equal(input_ids, torch.tensor([[7, 11]]))
        return (torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16),)


class _Placed:
    device = "mps"
    dtype = "torch.bfloat16"


class _Vae(_Encoder):
    config = SimpleNamespace(scaling_factor=2.0)

    def encode(self, video: torch.Tensor) -> SimpleNamespace:
        assert video.shape == (1, 3, 2, 2, 2)
        assert torch.equal(video[0, :, 0, 0, 0], torch.tensor([-1.0, -1.0, -1.0]))
        latent = torch.tensor([[[[[0.25]]], [[[0.5]]]]], dtype=torch.bfloat16)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latent))


class _Scheduler:
    config = SimpleNamespace(num_train_timesteps=1000)


def _caption() -> StructuredCaptionV1:
    return StructuredCaptionV1(
        character_token="test_person", immutable_attributes=("blue eyes",), action="walking"
    )


def _recipe(**updates: object) -> SequenceTransformRecipeV1:
    values: dict[str, object] = {
        "recipe_id": "wan_recipe",
        "output_width": 2,
        "output_height": 2,
        "output_frame_count": 2,
        "temporal_stride_numerator": 1,
        "temporal_stride_denominator": 1,
        "resize_mode": "center_crop",
        "interpolation": "nearest",
        "padding_color": (0, 0, 0),
        "seed": 19,
    }
    values.update(updates)
    return SequenceTransformRecipeV1.model_validate(values)


def _root(tmp_path: Path, name: str) -> WanPinnedLocalRoot:
    root = tmp_path / name
    root.mkdir(parents=True)
    (root / "config.json").write_text(canonical_json({"name": name}))
    return WanPinnedLocalRoot(root, "a" * 40, root_manifest_digest(root))


def _snapshot(tmp_path: Path) -> WanCompilerSnapshot:
    return WanCompilerSnapshot(
        tokenizer=_root(tmp_path, "tokenizer"),
        text_encoder=_root(tmp_path, "encoder"),
        vae=_root(tmp_path, "vae"),
        scheduler=_root(tmp_path, "scheduler"),
    )


def _source(frames: np.ndarray | None = None) -> bytes:
    values = np.zeros((2, 2, 2, 3), dtype=np.uint8) if frames is None else frames
    values[1, ...] = 255
    bindings = tuple(
        TransformInputReceipt(
            frame_id=f"frame_{index}",
            input_pixel_digest=f"{index + 1:064x}",
            crop_rectangle=(0, 0, 2, 2),
            output_shape=frame.shape,
            output_digest=transformed_array_digest(frame),
        )
        for index, frame in enumerate(values)
    )
    return encode_wan_curated_source(
        values,
        transform_receipt=TransformReceipt(
            recipe_id="wan_recipe",
            seed=19,
            inputs=bindings,
            decoded_output_digest=transformed_array_digest(values),
        ),
        caption=_caption(),
        source_digest="b" * 64,
        validation_evidence={"fixture": "b" * 64},
        selected_sequence_ids=("sequence_fixture",),
        selected_frame_source_digests=tuple(
            (binding.frame_id, "c" * 64) for binding in bindings
        ),
    )


def _compiler(tmp_path: Path, **updates: Any) -> WanCuratedBatchCompiler:
    components = WanCompilerComponents(_Tokenizer(), _Encoder(), _Vae(), _Scheduler())
    return WanCuratedBatchCompiler(
        snapshot=_snapshot(tmp_path),
        component_loader=lambda _: components,
        **updates,
    )


def _compiler_recipe(**updates: object) -> bytes:
    payload: dict[str, object] = {
        "schema_version": "wan-curated-compiler-recipe.v1",
        "transform_recipe": _recipe().model_dump(mode="json"),
        "timestep": 750,
        "timestep_boundary": 500,
        "latent_channels": 2,
        "vae_scaling_factor": 2.0,
    }
    payload.update(updates)
    return canonical_json(payload).encode()


def test_compiler_builds_exact_flowmatch_batch_and_evidence(tmp_path: Path) -> None:
    outputs = _compiler(tmp_path).compile_batch(_source(), _compiler_recipe())
    assert [(item.role, item.media_type) for item in outputs] == [
        ("batch", "application/vnd.safetensors"),
        ("manifest", "application/json"),
        ("provenance", "application/json"),
    ]
    batch, manifest, provenance = (item.data for item in outputs)
    tensors = validate_wan_batch_bytes(manifest, batch, torch_module=torch)
    clean = torch.tensor([[[[[0.5]]], [[[1.0]]]]], dtype=torch.bfloat16)
    noise_seed = int.from_bytes(
        hashlib.sha256(b"wan-flow-noise.v1\0" + b"19").digest()[:8], "big"
    ) & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    noise = torch.randn(clean.shape, dtype=clean.dtype, generator=generator)
    sigma = torch.tensor(0.75, dtype=torch.bfloat16)
    assert torch.equal(tensors["clean_latents"], clean)
    assert torch.equal(tensors["noise_latents"], noise)
    assert torch.equal(tensors["hidden_states"], sigma * noise + (1 - sigma) * clean)
    assert torch.equal(tensors["target"], noise - clean)
    assert torch.equal(tensors["timestep"], torch.tensor([750.0]))
    assert torch.equal(tensors["transformer_selector"], torch.tensor([0]))
    assert torch.equal(
        tensors["encoder_hidden_states"],
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16),
    )
    evidence = json.loads(provenance)
    assert evidence["source_digest"] == "b" * 64
    assert evidence["rendered_caption"] == render_caption_v1(_caption())
    assert evidence["token_ids"] == [[7, 11]]
    assert evidence["recipe_digest"] == hashlib.sha256(_compiler_recipe()).hexdigest()
    assert set(evidence["component_digests"]) == {"tokenizer", "text_encoder", "vae", "scheduler"}


def test_fresh_compilers_are_byte_identical_and_preserve_rng(tmp_path: Path) -> None:
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    first = _compiler(tmp_path / "a").compile_batch(_source(), _compiler_recipe())
    observed = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    second = _compiler(tmp_path / "b").compile_batch(_source(), _compiler_recipe())
    assert first == second
    assert observed[0] == expected[0]
    assert np.array_equal(observed[1][1], expected[1][1])
    assert torch.equal(observed[2], expected[2])


def test_stochastic_vae_is_seeded_independently_of_caller_and_seed_changes_bytes(
    tmp_path: Path,
) -> None:
    class StochasticVae(_Vae):
        def encode(self, video: torch.Tensor) -> SimpleNamespace:
            del video
            return SimpleNamespace(
                latent_dist=SimpleNamespace(
                    sample=lambda: torch.randn((1, 2, 1, 1, 1), dtype=torch.bfloat16)
                )
            )

    components = WanCompilerComponents(_Tokenizer(), _Encoder(), StochasticVae(), _Scheduler())
    compiler = WanCuratedBatchCompiler(
        snapshot=_snapshot(tmp_path), component_loader=lambda _: components
    )
    torch.manual_seed(1)
    random.seed(2)
    np.random.seed(3)
    first = compiler.compile_batch(_source(), _compiler_recipe())
    torch.manual_seed(900)
    random.seed(901)
    np.random.seed(902)
    second = compiler.compile_batch(_source(), _compiler_recipe())
    changed_source = json.loads(_source())
    changed_receipt = changed_source["transform_receipt"]
    changed_receipt["seed"] = 20
    changed_source["transform_receipt_digest"] = hashlib.sha256(
        canonical_json(changed_receipt).encode()
    ).hexdigest()
    assert first == second
    assert (
        compiler.compile_batch(
            canonical_json(changed_source).encode(),
            _compiler_recipe(transform_recipe=_recipe(seed=20).model_dump(mode="json")),
        )
        != first
    )


@pytest.mark.parametrize(
    ("recipe_update", "error"),
    [
        ({"vae_scaling_factor": 1.0}, "wan_compiler_vae_scale_invalid"),
        ({"timestep_boundary": 1001}, "wan_compiler_selector_invalid"),
        ({"latent_channels": 3}, "wan_compiler_latent_shape_invalid"),
    ],
)
def test_compiler_rejects_scientific_contract_mismatch(
    tmp_path: Path, recipe_update: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _compiler(tmp_path).compile_batch(_source(), _compiler_recipe(**recipe_update))


def test_compiler_rejects_nonfinite_and_invalid_placement(tmp_path: Path) -> None:
    bad_vae = _Vae()
    bad_vae.encode = lambda video: SimpleNamespace(  # type: ignore[method-assign]
        latent_dist=SimpleNamespace(
            sample=lambda: torch.tensor([[[[[float("nan")]]], [[[0.5]]]]], dtype=torch.bfloat16)
        )
    )
    components = WanCompilerComponents(_Tokenizer(), _Encoder(), bad_vae, _Scheduler())
    compiler = WanCuratedBatchCompiler(
        snapshot=_snapshot(tmp_path), component_loader=lambda _: components
    )
    with pytest.raises(ValueError, match="wan_compiler_tensor_invalid"):
        compiler.compile_batch(_source(), _compiler_recipe())
    broken = _Encoder()
    broken.device = "cpu"  # type: ignore[misc]
    components = WanCompilerComponents(_Tokenizer(), broken, _Vae(), _Scheduler())
    compiler = WanCuratedBatchCompiler(
        snapshot=_snapshot(tmp_path / "placement"), component_loader=lambda _: components
    )
    with pytest.raises(ValueError, match="wan_runtime_placement_invalid"):
        compiler.compile_batch(_source(), _compiler_recipe())


def test_compiler_rejects_source_and_resource_mutations(tmp_path: Path) -> None:
    compiler = _compiler(tmp_path, limits=WanCompilerLimits(max_source_bytes=100))
    with pytest.raises(ValueError, match="wan_compiler_budget_exceeded"):
        compiler.compile_batch(_source(), _compiler_recipe())
    source = json.loads(_source())
    source["frame_digest"] = "0" * 64
    with pytest.raises(ValueError, match="wan_compiler_source_invalid"):
        _compiler(tmp_path / "mutation").compile_batch(
            canonical_json(source).encode(), _compiler_recipe()
        )


def test_shared_validator_rejects_output_mutation_and_rogue_tensor(tmp_path: Path) -> None:
    outputs = _compiler(tmp_path).compile_batch(_source(), _compiler_recipe())
    batch, manifest = outputs[0].data, outputs[1].data
    with pytest.raises(ValueError, match="wan_training_batch_invalid"):
        validate_wan_batch_bytes(manifest, batch[:-1] + bytes([batch[-1] ^ 1]), torch_module=torch)
    header_size = int.from_bytes(batch[:8], "little")
    header = json.loads(batch[8 : 8 + header_size])
    header["rogue"] = header["target"]
    encoded = canonical_json(header).encode()
    rogue = len(encoded).to_bytes(8, "little") + encoded + batch[8 + header_size :]
    with pytest.raises(ValueError, match="wan_training_batch_invalid"):
        validate_wan_batch_bytes(manifest, rogue, torch_module=torch)


def test_shared_header_authority_accepts_full_and_exact_prefix(tmp_path: Path) -> None:
    batch = _compiler(tmp_path).compile_batch(_source(), _compiler_recipe())[0].data
    header_size = int.from_bytes(batch[:8], "little")
    full = validated_safetensors_header(batch, expected_names=WAN_BATCH_TENSORS, batch_size=1)
    prefix = validated_safetensors_header_prefix(
        batch[: 8 + header_size],
        total_size=len(batch),
        expected_names=WAN_BATCH_TENSORS,
        batch_size=1,
    )
    assert full == prefix


def test_shared_header_authority_rejects_truncated_and_oversized(tmp_path: Path) -> None:
    batch = _compiler(tmp_path).compile_batch(_source(), _compiler_recipe())[0].data
    header_size = int.from_bytes(batch[:8], "little")
    with pytest.raises(ValueError, match="wan_training_batch_invalid"):
        validated_safetensors_header(
            batch[: 8 + header_size], expected_names=WAN_BATCH_TENSORS, batch_size=1
        )
    with pytest.raises(ValueError, match="wan_training_batch_invalid"):
        validated_safetensors_header_prefix(
            batch[: 8 + header_size],
            total_size=17 * 1024 * 1024 * 1024,
            expected_names=WAN_BATCH_TENSORS,
            batch_size=1,
        )


def test_default_loader_is_lazy_private_and_strictly_local(tmp_path: Path) -> None:
    calls: list[tuple[str, Path, dict[str, object]]] = []

    class _Loadable:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> object:
            candidate = Path(path)
            assert candidate.exists()
            assert "wan-compiler-" in candidate.as_posix()
            calls.append((cls.__name__, candidate, kwargs))
            return cls()

        def to(self, **kwargs: object) -> object:
            assert kwargs == {"device": "mps", "dtype": torch.bfloat16}
            return self

    transformers = SimpleNamespace(AutoTokenizer=_Loadable, UMT5EncoderModel=_Loadable)
    diffusers = SimpleNamespace(
        AutoencoderKLWan=_Loadable, FlowMatchEulerDiscreteScheduler=_Loadable
    )
    imported: list[str] = []

    def importer(name: str) -> object:
        imported.append(name)
        return {"torch": torch, "transformers": transformers, "diffusers": diffusers}[name]

    compiler = WanCuratedBatchCompiler(snapshot=_snapshot(tmp_path), importer=importer)
    assert imported == []
    compiler._load_components(compiler._snapshot)  # type: ignore[attr-defined]
    assert imported == ["torch", "transformers", "diffusers"]
    assert len(calls) == 4
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert all(not call[1].exists() for call in calls)


def test_snapshot_replacement_fails_before_serialization(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)

    def replacing_loader(_: WanCompilerSnapshot) -> WanCompilerComponents:
        (snapshot.vae.path / "config.json").write_text('{"replaced":true}')
        return WanCompilerComponents(_Tokenizer(), _Encoder(), _Vae(), _Scheduler())

    compiler = WanCuratedBatchCompiler(snapshot=snapshot, component_loader=replacing_loader)
    with pytest.raises(ValueError, match="wan_compiler_snapshot_replaced"):
        compiler.compile_batch(_source(), _compiler_recipe())


def test_frame_mutation_with_updated_raw_digest_cannot_retain_receipt(tmp_path: Path) -> None:
    source = json.loads(_source())
    raw = bytearray(__import__("base64").b64decode(source["frames_base64"]))
    raw[0] ^= 1
    source["frames_base64"] = __import__("base64").b64encode(raw).decode()
    source["frame_digest"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="wan_compiler_source_invalid"):
        _compiler(tmp_path).compile_batch(canonical_json(source).encode(), _compiler_recipe())


def test_rng_is_restored_when_stochastic_vae_fails(tmp_path: Path) -> None:
    class FailingVae(_Vae):
        def encode(self, video: torch.Tensor) -> SimpleNamespace:
            del video

            def fail() -> torch.Tensor:
                torch.randn(4)
                random.random()
                np.random.random()
                raise RuntimeError("expected")

            return SimpleNamespace(latent_dist=SimpleNamespace(sample=fail))

    components = WanCompilerComponents(_Tokenizer(), _Encoder(), FailingVae(), _Scheduler())
    compiler = WanCuratedBatchCompiler(
        snapshot=_snapshot(tmp_path), component_loader=lambda _: components
    )
    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with pytest.raises(RuntimeError, match="expected"):
        compiler.compile_batch(_source(), _compiler_recipe())
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_realistic_component_placement_without_hf_map_and_edge_cases() -> None:
    parameter_only = SimpleNamespace(
        device="mps", parameters=lambda: iter((_Placed(),)), buffers=lambda: iter(())
    )
    buffer_only = SimpleNamespace(
        device="mps", parameters=lambda: iter(()), buffers=lambda: iter((_Placed(),))
    )
    both = SimpleNamespace(
        device="mps", parameters=lambda: iter((_Placed(),)), buffers=lambda: iter((_Placed(),))
    )
    require_component_on_mps(parameter_only)
    require_component_on_mps(buffer_only)
    require_component_on_mps(both)
    empty = SimpleNamespace(device="mps", parameters=lambda: iter(()), buffers=lambda: iter(()))
    with pytest.raises(ValueError, match="wan_runtime_placement_invalid"):
        require_component_on_mps(empty)
    authorized_empty = SimpleNamespace(
        device="mps",
        dtype="torch.bfloat16",
        parameters=lambda: iter(()),
        buffers=lambda: iter(()),
    )
    require_component_on_mps(authorized_empty, parameterless_kind="scheduler")
    with pytest.raises(ValueError, match="wan_runtime_placement_invalid"):
        require_component_on_mps(
            SimpleNamespace(
                device="cpu",
                dtype="torch.bfloat16",
                parameters=lambda: iter(()),
                buffers=lambda: iter(()),
            ),
            parameterless_kind="scheduler",
        )
    mixed = SimpleNamespace(
        parameters=lambda: iter((_Placed(), SimpleNamespace(device="cpu", dtype="torch.bfloat16"))),
        buffers=lambda: iter(()),
    )
    with pytest.raises(ValueError, match="wan_runtime_placement_invalid"):
        require_component_on_mps(mixed)


def test_secure_snapshot_rejects_symlink_and_resource_limits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_bytes(b"1234")
    pinned = WanPinnedLocalRoot(source, "a" * 40, root_manifest_digest(source))
    with pytest.raises(ValueError, match="wan_factory_snapshot_invalid"):
        secure_snapshot_pinned_root(
            pinned,
            tmp_path / "too-small",
            limits=WanSnapshotCopyLimits(max_files=1, max_bytes=3, max_depth=1),
        )
    (source / "link").symlink_to(source / "file")
    with pytest.raises(ValueError):
        root_manifest_digest(source)


def test_manifest_rejects_fifo_socket_and_symlink_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    normal = tmp_path / "normal"
    normal.mkdir()
    (normal / "file").write_bytes(b"safe")
    assert root_manifest_digest(normal).startswith("sha256:")

    fifo_root = tmp_path / "fifo"
    fifo_root.mkdir()
    os.mkfifo(fifo_root / "leaf")
    started = time.monotonic()
    with pytest.raises(ValueError, match="wan_factory_root_invalid"):
        root_manifest_digest(fifo_root)
    assert time.monotonic() - started < 1.0

    socket_root = tmp_path / "socket"
    socket_root.mkdir()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        monkeypatch.chdir(socket_root)
        try:
            server.bind("leaf")
        except PermissionError:
            pytest.skip("managed sandbox denies AF_UNIX bind")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="wan_factory_root_invalid"):
            root_manifest_digest(socket_root)
    finally:
        monkeypatch.chdir(tmp_path)
        server.close()

    link_root = tmp_path / "link"
    link_root.mkdir()
    (link_root / "target").write_bytes(b"safe")
    (link_root / "leaf").symlink_to(link_root / "target")
    with pytest.raises(ValueError, match="wan_factory_root_invalid"):
        root_manifest_digest(link_root)


def test_secure_snapshot_rejects_replacement_and_incomplete_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    leaf = source / "file"
    leaf.write_bytes(b"1234")
    pinned = WanPinnedLocalRoot(source, "a" * 40, root_manifest_digest(source))
    real_pread = __import__("os").pread

    def replacing_pread(fd: int, size: int, offset: int) -> bytes:
        data = real_pread(fd, size, offset)
        leaf.write_bytes(b"5678")
        return data

    monkeypatch.setattr("os.pread", replacing_pread)
    with pytest.raises(ValueError, match="wan_factory_snapshot_invalid"):
        secure_snapshot_pinned_root(pinned, tmp_path / "replaced")
    assert not (tmp_path / "replaced").exists()
    monkeypatch.setattr("os.pread", real_pread)
    leaf.write_bytes(b"1234")
    real_write = __import__("os").write
    monkeypatch.setattr("os.write", lambda fd, data: max(0, real_write(fd, data[:1])))
    with pytest.raises(ValueError, match="wan_factory_snapshot_invalid"):
        secure_snapshot_pinned_root(pinned, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()


def test_runtime_accepts_semantic_json_while_compiler_emits_canonical_json(tmp_path: Path) -> None:
    outputs = _compiler(tmp_path).compile_batch(_source(), _compiler_recipe())
    batch, canonical_manifest = outputs[0].data, outputs[1].data
    assert canonical_json(json.loads(canonical_manifest)).encode() == canonical_manifest
    tensor_path = tmp_path / "batch.safetensors"
    manifest_path = tmp_path / "manifest.json"
    tensor_path.write_bytes(batch)
    manifest_path.write_text(json.dumps(json.loads(canonical_manifest), indent=2))
    assert validate_wan_batch_artifact(manifest_path, tensor_path) == batch
