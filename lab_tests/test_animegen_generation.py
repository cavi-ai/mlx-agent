from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lab_tests.test_capabilities import manifest
from video_lab.adapter_protocol import AdapterExecutionError, GenerationRequest
from video_lab.adapters.animegen_mps import (
    AnimeGenMpsAdapter,
    GenerationProfile,
    build_generation_invocation,
    classify_repeatability,
    select_minimum_profile,
)
from video_lab.domain import Manifest, canonical_json


def test_minimum_profile_selects_smallest_valid_dimensions_and_frames() -> None:
    profile = select_minimum_profile(
        profiles=(
            GenerationProfile(512, 288, 17, 8),
            GenerationProfile(256, 144, 33, 8),
            GenerationProfile(128, 80, 17, 8),
        ),
        width_multiple=16,
        height_multiple=16,
    )
    assert profile == GenerationProfile(width=128, height=80, num_frames=17, fps=8)


def test_minimum_profile_never_synthesizes_or_accepts_invalid_tuple() -> None:
    assert select_minimum_profile(
        profiles=(GenerationProfile(256, 144, 17, 8), GenerationProfile(128, 80, 33, 8)),
        width_multiple=16,
        height_multiple=16,
    ) == GenerationProfile(128, 80, 33, 8)
    with pytest.raises(ValueError):
        select_minimum_profile(
            profiles=(GenerationProfile(130, 80, 16, 8),),
            width_multiple=16,
            height_multiple=16,
        )


def test_repeatability_distinguishes_bytes_frames_and_nondeterminism() -> None:
    common = dict(
        manifest_hash="a" * 64,
        source_fingerprint="b" * 64,
        runtime_fingerprint="c" * 64,
        placement_fingerprint="d" * 64,
    )
    bitwise = classify_repeatability(
        first_artifact_sha256="1" * 64,
        second_artifact_sha256="1" * 64,
        first_frame_sha256=("2" * 64,) * 17,
        second_frame_sha256=("2" * 64,) * 17,
        **common,
    )
    assert bitwise.classification == "bitwise"
    assert bitwise.matching_frame_count == 17
    assert bitwise.differing_frame_count == 0
    assert bitwise.matching_frame_ratio == pytest.approx(1.0)

    decoded_frame = classify_repeatability(
        first_artifact_sha256="1" * 64,
        second_artifact_sha256="3" * 64,
        first_frame_sha256=("2" * 64,) * 17,
        second_frame_sha256=("2" * 64,) * 17,
        **common,
    )
    assert decoded_frame.classification == "decoded_frame"
    assert decoded_frame.matching_frame_count == 17
    assert decoded_frame.differing_frame_count == 0
    assert decoded_frame.matching_frame_ratio == pytest.approx(1.0)

    numerically_nondeterministic = classify_repeatability(
        first_artifact_sha256="1" * 64,
        second_artifact_sha256="3" * 64,
        first_frame_sha256=("2" * 64,) * 15 + ("3" * 64,) * 2,
        second_frame_sha256=("2" * 64,) * 15 + ("4" * 64,) * 2,
        **common,
    )
    assert numerically_nondeterministic.classification == "numerically_nondeterministic"
    assert numerically_nondeterministic.matching_frame_count == 15
    assert numerically_nondeterministic.differing_frame_count == 2
    assert numerically_nondeterministic.matching_frame_ratio == pytest.approx(15 / 17)


def test_repeatability_is_invalid_when_provenance_differs() -> None:
    evidence = classify_repeatability(
        first_artifact_sha256="1" * 64,
        second_artifact_sha256="1" * 64,
        first_frame_sha256=("2" * 64,),
        second_frame_sha256=("2" * 64,),
        manifest_hash="a" * 64,
        second_manifest_hash="e" * 64,
        source_fingerprint="b" * 64,
        runtime_fingerprint="c" * 64,
        placement_fingerprint="d" * 64,
    )
    assert evidence.classification == "invalid"


def test_generation_invocation_preserves_manifest_controls_and_separates_rng_device() -> None:
    requested = manifest().model_copy(
        update={
            "prompt": "anime hero",
            "negative_prompt": "blur",
            "seed": 42,
            "width": 512,
            "height": 288,
            "num_frames": 17,
            "fps": 8,
            "inference_steps": 4,
            "guidance_scale": 3.5,
            "scheduler": "flow-match-euler-shift-3",
            "dtype": "bfloat16",
            "quantization": "layerwise-fp8",
            "memory_strategy": "model_cpu_offload",
        }
    )
    generator = object()
    invocation = build_generation_invocation(requested, generator=generator)
    assert invocation.pipeline_kwargs == {
        "prompt": "anime hero",
        "negative_prompt": "blur",
        "height": 288,
        "width": 512,
        "num_frames": 17,
        "num_inference_steps": 4,
        "guidance_scale": 3.5,
        "generator": generator,
    }
    assert invocation.controls["seed"] == 42
    assert invocation.controls["scheduler"] == "flow-match-euler-shift-3"
    assert invocation.controls["dtype"] == "bfloat16"
    assert invocation.controls["memory_strategy"] == "model_cpu_offload"
    assert invocation.controls["execution_placement"] == "local_hybrid:mps"
    assert invocation.controls["generator_device"] == "cpu"


class ProductionRuntime:
    def __init__(self) -> None:
        self.seed: int | None = None
        self.kwargs: dict[str, object] = {}
        self.observed_storage_dtype: str | None = None
        self.invoke_called = False
        self.encode_called = False

    def prepare(self, requested: Manifest) -> tuple[object, dict[str, object]]:
        return object(), {
            "scheduler": requested.scheduler,
            "dtype": requested.dtype,
            "quantization": requested.quantization,
            "compute_dtype": "bfloat16",
            "storage_dtype": self.observed_storage_dtype
            or ("bfloat16" if requested.quantization == "bf16" else "float8_e4m3fn"),
            "loras": tuple(item.model_dump(mode="json") for item in requested.loras),
            "memory_strategy": requested.memory_strategy,
            "source_fingerprint": "a" * 64,
            "placement_fingerprint": "b" * 64,
            "execution_placement": "local_hybrid:mps",
        }

    def cpu_generator(self, seed: int) -> object:
        self.seed = seed
        return object()

    def invoke(self, pipeline: object, **kwargs: object) -> object:
        self.invoke_called = True
        self.kwargs = kwargs
        callback = kwargs["callback_on_step_end"]
        assert callable(callback)
        callback(pipeline, 0, object(), {})
        return [object()] * 17

    def encode(self, frames: object, path: Path, fps: int) -> None:
        self.encode_called = True
        path.write_bytes(b"encoded")

    def runtime_versions(self) -> dict[str, str]:
        return {
            "python": "test",
            "dist:diffusers": "test",
            "dist:otherwise-uncurated": "1.0",
            "dist:torch": "test",
            "video-lab": "source-sha256:" + "1" * 64,
        }


def test_real_adapter_executes_production_shaped_runtime_and_returns_result(tmp_path: Path) -> None:
    runtime = ProductionRuntime()
    requested = manifest().model_copy(
        update={
            "adapter_id": "diffusers.animegen-t2v.wan2.2-a14b.mps.v1",
            "adapter_version": "1",
            "model_id": "aidealab/AnimeGen-T2V",
            "width": 512,
            "height": 288,
            "num_frames": 17,
            "fps": 8,
            "scheduler": "flow-match-euler-shift-3",
            "dtype": "bfloat16",
            "quantization": "layerwise-fp8",
            "backend": "mps",
            "device": "mps",
            "memory_strategy": "model_cpu_offload",
            "runtime_versions": runtime.runtime_versions(),
        }
    )
    result = AnimeGenMpsAdapter(runtime=runtime, authorized_profile=(512, 288, 17)).generate(
        GenerationRequest(
            job_id="job", attempt_id="attempt", manifest=requested, scratch_root=tmp_path
        ),
        lambda _update: None,
        lambda: False,
    )
    assert result.effective_manifest == requested
    assert result.video_path.read_bytes() == b"encoded"
    assert runtime.seed == requested.seed
    assert runtime.kwargs["num_inference_steps"] == requested.inference_steps
    assert runtime.kwargs["callback_on_step_end"]
    assert result.adapter_telemetry["observed_configuration"]["source_fingerprint"] == "a" * 64


@pytest.mark.parametrize("storage_dtype", ["float8_e4m3fn", "float32"])
def test_adapter_rejects_bf16_manifest_when_observed_storage_is_not_bf16(
    tmp_path: Path, storage_dtype: str
) -> None:
    runtime = ProductionRuntime()
    requested = manifest().model_copy(
        update={
            "adapter_id": "diffusers.animegen-t2v.wan2.2-a14b.mps.v1",
            "adapter_version": "1",
            "model_id": "aidealab/AnimeGen-T2V",
            "width": 512,
            "height": 288,
            "num_frames": 17,
            "fps": 8,
            "scheduler": "flow-match-euler-shift-3",
            "dtype": "bfloat16",
            "quantization": "bf16",
            "backend": "mps",
            "device": "mps",
            "memory_strategy": "model_cpu_offload",
            "runtime_versions": runtime.runtime_versions(),
        }
    )
    runtime.observed_storage_dtype = storage_dtype

    with pytest.raises(AdapterExecutionError, match="observed AnimeGen configuration"):
        AnimeGenMpsAdapter(runtime=runtime, authorized_profile=(512, 288, 17)).generate(
            GenerationRequest(
                job_id="job", attempt_id="attempt", manifest=requested, scratch_root=tmp_path
            ),
            lambda _update: None,
            lambda: False,
        )


def test_adapter_rejects_uncurated_distribution_drift_before_invoke_or_encode(
    tmp_path: Path,
) -> None:
    runtime = ProductionRuntime()
    requested = manifest().model_copy(
        update={
            "adapter_id": "diffusers.animegen-t2v.wan2.2-a14b.mps.v1",
            "adapter_version": "1",
            "model_id": "aidealab/AnimeGen-T2V",
            "width": 512,
            "height": 288,
            "num_frames": 17,
            "fps": 8,
            "scheduler": "flow-match-euler-shift-3",
            "dtype": "bfloat16",
            "quantization": "bf16",
            "backend": "mps",
            "device": "mps",
            "memory_strategy": "model_cpu_offload",
            "runtime_versions": {
                **runtime.runtime_versions(),
                "dist:otherwise-uncurated": "2.0",
            },
        }
    )

    with pytest.raises(AdapterExecutionError, match="runtime versions"):
        AnimeGenMpsAdapter(runtime=runtime, authorized_profile=(512, 288, 17)).generate(
            GenerationRequest(
                job_id="job", attempt_id="attempt", manifest=requested, scratch_root=tmp_path
            ),
            lambda _update: None,
            lambda: False,
        )

    assert runtime.invoke_called is False
    assert runtime.encode_called is False
    assert not (tmp_path / "output.mp4").exists()


def test_authorized_adapter_rejects_fps_and_manifest_hash_drift(tmp_path: Path) -> None:
    requested = manifest().model_copy(
        update={
            "adapter_id": "diffusers.animegen-t2v.wan2.2-a14b.mps.v1",
            "adapter_version": "1",
            "model_id": "aidealab/AnimeGen-T2V",
            "width": 512,
            "height": 288,
            "num_frames": 17,
            "fps": 8,
            "scheduler": "flow-match-euler-shift-3",
            "dtype": "bfloat16",
            "quantization": "layerwise-fp8",
            "backend": "mps",
            "device": "mps",
            "memory_strategy": "model_cpu_offload",
        }
    )
    digest = hashlib.sha256(canonical_json(requested).encode()).hexdigest()

    def request(value: Manifest) -> GenerationRequest:
        return GenerationRequest(
            job_id="job", attempt_id="attempt", manifest=value, scratch_root=tmp_path
        )

    adapter = AnimeGenMpsAdapter(
        runtime=ProductionRuntime(),
        authorized_profile=(512, 288, 17),
        authorized_manifest=requested,
        authorized_verdict="experimental_local_hybrid",
        authorized_manifest_hash=digest,
    )
    with pytest.raises(AdapterExecutionError):
        adapter.generate(
            request(requested.model_copy(update={"fps": 9})), lambda _: None, lambda: False
        )
    bad_hash = AnimeGenMpsAdapter(
        runtime=ProductionRuntime(),
        authorized_profile=(512, 288, 17),
        authorized_manifest=requested,
        authorized_verdict="experimental_local_hybrid",
        authorized_manifest_hash="0" * 64,
    )
    with pytest.raises(AdapterExecutionError, match="hash"):
        bad_hash.generate(request(requested), lambda _: None, lambda: False)


def _standard_authorized_adapter(requested: Manifest) -> AnimeGenMpsAdapter:
    return AnimeGenMpsAdapter(
        runtime=ProductionRuntime(),
        authorized_profile=(512, 288, 17),
        exact_profile_only=False,
        authorized_manifest=requested,
        authorized_verdict="standard_lab_qualified",
        authorized_manifest_hash=hashlib.sha256(canonical_json(requested).encode()).hexdigest(),
    )


def test_standard_authorization_freezes_schema_version(tmp_path: Path) -> None:
    requested = manifest().model_copy(update={"width": 512, "height": 288, "num_frames": 17})
    adapter = _standard_authorized_adapter(requested)
    drifted = requested.model_copy(update={"schema_version": "manifest.v2"})
    with pytest.raises(AdapterExecutionError, match="frozen controls"):
        adapter.generate(
            GenerationRequest(
                job_id="job", attempt_id="attempt", manifest=drifted, scratch_root=tmp_path
            ),
            lambda _: None,
            lambda: False,
        )


def test_standard_authorization_freezes_protocol_version(tmp_path: Path) -> None:
    requested = manifest().model_copy(update={"width": 512, "height": 288, "num_frames": 17})
    adapter = _standard_authorized_adapter(requested)
    drifted = requested.model_copy(update={"protocol_version": "ab.v2"})
    with pytest.raises(AdapterExecutionError, match="frozen controls"):
        adapter.generate(
            GenerationRequest(
                job_id="job", attempt_id="attempt", manifest=drifted, scratch_root=tmp_path
            ),
            lambda _: None,
            lambda: False,
        )
