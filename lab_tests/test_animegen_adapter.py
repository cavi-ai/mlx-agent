from __future__ import annotations

import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_tests.test_capabilities import manifest as base_manifest
from video_lab.adapter_protocol import AdapterExecutionError, GenerationCancelled
from video_lab.adapters.animegen_mps import (
    AnimeGenMpsAdapter,
    AnimeGenStageExecutor,
    AnimeGenStorageMode,
    DiffusersAnimeGenGenerationRuntime,
    PlacementPolicyError,
    _DiffusersRuntime,
    storage_mode_for_manifest,
)
from video_lab.artifacts import ArtifactStore
from video_lab.capabilities import validate_manifest_support
from video_lab.database import Database
from video_lab.domain import LoraBinding
from video_lab.feasibility_models import FeasibilityStage
from video_lab.source_inventory import (
    ANIMEGEN_REQUIRED_FILES,
    ANIMEGEN_SOURCE_REPOS,
    ResolvedSource,
    SourceFile,
    canonical_source_digest,
)


def frozen_sources():
    revision = "a" * 40

    def source(component: str, repo: str, files: tuple[str, ...]) -> ResolvedSource:
        return ResolvedSource(
            component,
            repo,
            "v1",
            revision,
            "apache-2.0",
            tuple(SourceFile(name, 1, local_sha256="c" * 64) for name in files),
        )

    return {
        key: source(key, ANIMEGEN_SOURCE_REPOS[key], ANIMEGEN_REQUIRED_FILES[key])
        for key in ANIMEGEN_REQUIRED_FILES
    }


def test_animegen_storage_modes_are_supported_and_unknown_storage_is_rejected() -> None:
    adapter = AnimeGenMpsAdapter(
        runtime=SimpleNamespace(), authorized_profile=(128, 80, 17), exact_profile_only=False
    )
    common = {
        "adapter_id": adapter.adapter_id,
        "model_id": "aidealab/AnimeGen-T2V",
        "dtype": "bfloat16",
        "scheduler": "flow-match-euler-shift-3",
        "memory_strategy": "model_cpu_offload",
        "width": 128,
        "height": 80,
    }
    bf16_manifest = base_manifest().model_copy(update={**common, "quantization": "bf16"})
    fp8_manifest = base_manifest().model_copy(update={**common, "quantization": "layerwise-fp8"})

    assert not validate_manifest_support(bf16_manifest, adapter.capabilities())
    assert not validate_manifest_support(fp8_manifest, adapter.capabilities())
    assert storage_mode_for_manifest(bf16_manifest) is AnimeGenStorageMode.BF16
    assert storage_mode_for_manifest(fp8_manifest) is AnimeGenStorageMode.LAYERWISE_FP8
    for quantization in ("fp8", "float16", "none"):
        unknown_manifest = bf16_manifest.model_copy(update={"quantization": quantization})
        with pytest.raises(AdapterExecutionError, match="unsupported transformer storage"):
            storage_mode_for_manifest(unknown_manifest)


def test_repeatability_evidence_records_decoded_frame_match_metrics() -> None:
    from video_lab.adapters.animegen_mps import classify_repeatability

    evidence = classify_repeatability(
        first_artifact_sha256="1" * 64,
        second_artifact_sha256="2" * 64,
        first_frame_sha256=("3" * 64,) * 15 + ("4" * 64,) * 2,
        second_frame_sha256=("3" * 64,) * 15 + ("5" * 64,) * 2,
        manifest_hash="6" * 64,
        source_fingerprint="7" * 64,
        runtime_fingerprint="8" * 64,
        placement_fingerprint="9" * 64,
    )

    assert evidence.matching_frame_count == 15
    assert evidence.differing_frame_count == 2
    assert evidence.matching_frame_ratio == pytest.approx(15 / 17)


class DType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"torch.{self.name}"


class TorchStub:
    bfloat16 = DType("bfloat16")
    float8_e4m3fn = DType("float8_e4m3fn")

    class backends:
        class mps:
            is_available = staticmethod(lambda: True)
            is_built = staticmethod(lambda: True)

    class mps:
        empty_cache = staticmethod(lambda: None)

    empty = staticmethod(lambda *_args, **_kwargs: object())


class Component:
    def __init__(self, name: str, device: str = "cpu", *, hook: bool = True) -> None:
        self.name = name
        self.dtype = TorchStub.bfloat16
        self._parameters = [
            SimpleNamespace(device=device, dtype=TorchStub.float8_e4m3fn, requires_grad=False),
            SimpleNamespace(device=device, dtype=TorchStub.bfloat16),
        ]
        self.want_hook = hook
        self._hf_hook = None
        self.cast_calls: list[dict[str, object]] = []

    def parameters(self):
        return iter(self._parameters)

    def enable_layerwise_casting(self, **kwargs: object) -> None:
        self.cast_calls.append(kwargs)


class Pipeline:
    def __init__(self, components: dict[str, Component]) -> None:
        self.components = components
        self.offload_calls: list[str] = []
        self.lora_loads: list[tuple[str, str, str]] = []
        self.active = ["lightning_high", "lightning_low"]
        self.peft_layer = SimpleNamespace(
            scaling={},
            lora_alpha={"lightning_high": 4.0, "lightning_low": 16.0},
            r={"lightning_high": 8, "lightning_low": 8},
        )

    def load_lora_weights(
        self,
        repo: str,
        *,
        weight_name: str,
        adapter_name: str,
        local_files_only: bool,
    ) -> None:
        assert local_files_only and "/snapshots/" in repo
        self.lora_loads.append((repo, weight_name, adapter_name))

    def set_adapters(self, names: list[str], adapter_weights: list[float]) -> None:
        self.peft_layer.scaling = {
            name: (self.peft_layer.lora_alpha[name] / self.peft_layer.r[name]) * weight
            for name, weight in zip(names, adapter_weights, strict=True)
        }

    def get_active_adapters(self):
        return self.active

    def named_modules(self):
        return (("transformer.block.lora", self.peft_layer),)

    def enable_model_cpu_offload(self, *, device: str) -> None:
        self.offload_calls.append(device)
        for component in self.components.values():
            if component.want_hook:
                component._hf_hook = SimpleNamespace(execution_device="mps:0")


class Runtime:
    def __init__(self, *, bad_hook: bool = False) -> None:
        self.calls: list[tuple[str, object]] = []
        self.components = {
            name: Component(name, hook=not (bad_hook and name == "transformer"))
            for name in ("text_encoder", "text_encoder_2", "vae", "transformer", "transformer_2")
        }
        self.tokenizer = SimpleNamespace(name="tokenizer")
        self.pipeline = Pipeline(self.components)

    def load_text_stack(self, repo: str, **kwargs: object):
        self.calls.append(("text_stack", (repo, kwargs)))
        return {"tokenizer": self.tokenizer, "text_encoder": self.components["text_encoder"]}

    def load_vae(self, repo: str, *, subfolder: str, torch_dtype: object, **kwargs: object):
        self.calls.append(("vae", (repo, subfolder, torch_dtype, kwargs)))
        return self.components["vae"]

    def load_transformer_file(self, path: Path, *, torch_dtype: object, **kwargs: object):
        name = (
            "transformer"
            if not any(call[0] == "transformer" for call in self.calls)
            else "transformer_2"
        )
        self.calls.append(("transformer", (path, torch_dtype, kwargs)))
        return self.components[name]

    def load_scheduler_config(self, repo: str, *, subfolder: str, **kwargs: object):
        self.calls.append(("scheduler_config", (repo, subfolder, kwargs)))
        return {"shift": 1.0}

    def scheduler_from_config(self, config: object, *, shift: float):
        self.calls.append(("scheduler_from_config", (config, shift)))
        return SimpleNamespace(config={"shift": shift})

    def assemble_pipeline(self, repo: str, **components: object):
        assert components["tokenizer"] is self.tokenizer
        assert components["text_encoder"] is self.components["text_encoder"]
        assert "text_encoder_2" not in components
        self.calls.append(("pipeline", (repo, components)))
        return self.pipeline

    def observe_adapters(self, pipeline: Pipeline):
        active = pipeline.get_active_adapters()
        module = dict(pipeline.named_modules())["transformer.block.lora"]
        return {
            name: module.scaling[name] / (module.lora_alpha[name] / module.r[name])
            for name in active
        }


def test_diffusers_runtime_uses_pipeline_declared_fast_tokenizer(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Loader:
        def __init__(self, name: str) -> None:
            self.name = name

        def from_pretrained(self, _repo: str, **kwargs: object) -> object:
            calls.append((self.name, kwargs))
            return object()

    module = SimpleNamespace(
        AutoTokenizer=Loader("tokenizer"),
        UMT5EncoderModel=Loader("text_encoder"),
    )
    monkeypatch.setattr(
        "video_lab.adapters.animegen_mps.importlib.import_module", lambda _name: module
    )

    _DiffusersRuntime().load_text_stack("/frozen/snapshot", local_files_only=True)

    assert calls[0] == (
        "tokenizer",
        {
            "subfolder": "tokenizer",
            "local_files_only": True,
            "use_fast": True,
        },
    )
    assert calls[1] == (
        "text_encoder",
        {"subfolder": "text_encoder", "local_files_only": True},
    )


def test_diffusers_runtime_observes_loras_on_transformer_components() -> None:
    layer = SimpleNamespace(
        scaling={"high": 1.0, "low": 2.0},
        lora_alpha={"high": 4.0, "low": 8.0},
        r={"high": 8, "low": 8},
    )
    transformer = SimpleNamespace(named_modules=lambda: (("block.lora", layer),))
    pipeline = SimpleNamespace(
        transformer=transformer,
        transformer_2=transformer,
        get_active_adapters=lambda: ("high", "low"),
    )

    observed = _DiffusersRuntime().observe_adapters(pipeline)

    assert observed == {"high": 2.0, "low": 2.0}


def test_uses_real_api_shaped_stages_and_observed_evidence(tmp_path: Path) -> None:
    runtime = Runtime()
    evidence = AnimeGenStageExecutor(
        cache_root=tmp_path, torch=TorchStub, runtime=runtime, sources=frozen_sources()
    ).execute(cancelled=lambda: False)
    assert [name for name, _ in runtime.calls] == [
        "text_stack",
        "vae",
        "transformer",
        "transformer",
        "scheduler_config",
        "scheduler_from_config",
        "pipeline",
    ]
    assert runtime.calls[5][1] == ({"shift": 1.0}, 3.0)
    for index in (0, 1, 2, 3, 4, 6):
        source = runtime.calls[index][1][0]
        assert "/snapshots/" in str(source)
    assert [Path(repo).name for repo, _weight, _name in runtime.pipeline.lora_loads] == [
        "a" * 40,
        "a" * 40,
    ]
    assert all(
        not isinstance(value, str) or not value.startswith(("aidealab/", "Wan-AI/", "lightx2v/"))
        for _name, call in runtime.calls
        for value in (call if isinstance(call, tuple) else ())
    )
    for name in ("transformer", "transformer_2"):
        assert runtime.components[name].cast_calls == [
            {
                "storage_dtype": TorchStub.float8_e4m3fn,
                "compute_dtype": TorchStub.bfloat16,
            }
        ]
    for index in (2, 3):
        assert runtime.calls[index][1][2]["config"].endswith("/" + "a" * 40)
        assert runtime.calls[index][1][2]["subfolder"] == "transformer"
    assert runtime.pipeline.offload_calls == ["mps"]
    assert evidence.details["observed"]["compute_dtype"] == "bfloat16"
    assert evidence.details["observed"]["storage_dtype"] == "float8_e4m3fn"
    assert evidence.details["observed"]["casting_calls"] == 2
    assert evidence.details["observed"]["active_loras"] == {
        "lightning_high": 2.0,
        "lightning_low": 1.0,
    }
    assert len(evidence.details["placement"]["sample"]) <= 16
    assert [stage["stage"] for stage in evidence.details["stages"]] == [
        "text_stack",
        "vae",
        "transformer_high",
        "transformer_low",
        "layerwise_casting",
        "layerwise_casting",
        "scheduler_config",
        "scheduler_construct",
        "assemble_pipeline",
        "apply_loras",
        "enable_offload",
    ]
    assert evidence.details["stages"][1]["placement"]["counts"] == {"cpu": 2}


def test_production_generation_runtime_assembles_frozen_task4_pipeline(tmp_path: Path) -> None:
    runtime = Runtime()
    sources = frozen_sources()
    requested = base_manifest().model_copy(
        update={
            "model_id": "aidealab/AnimeGen-T2V",
            "model_revision": "a" * 40,
            "scheduler": "flow-match-euler-shift-3",
            "dtype": "bfloat16",
            "quantization": "layerwise-fp8",
            "memory_strategy": "model_cpu_offload",
            "loras": (
                LoraBinding(
                    model_id="lightx2v/Wan2.2-Lightning",
                    revision="a" * 40,
                    checksum=canonical_source_digest(sources["lightning-high"]),
                    weight=2.0,
                ),
                LoraBinding(
                    model_id="lightx2v/Wan2.2-Lightning",
                    revision="a" * 40,
                    checksum=canonical_source_digest(sources["lightning-low"]),
                    weight=1.0,
                ),
            ),
            "source_checksums": {
                component: canonical_source_digest(source) for component, source in sources.items()
            },
        }
    )
    pipeline, observed = DiffusersAnimeGenGenerationRuntime(
        cache_root=tmp_path,
        sources=sources,
        runtime=runtime,
        torch=TorchStub,
    ).prepare(requested)

    assert pipeline is runtime.pipeline
    assert runtime.pipeline.offload_calls == ["mps"]
    assert observed["execution_placement"] == "local_hybrid:mps"
    assert observed["loras"] == tuple(item.model_dump(mode="json") for item in requested.loras)
    assert len(str(observed["source_fingerprint"])) == 64
    assert len(str(observed["placement_fingerprint"])) == 64


def test_production_generation_runtime_reuses_pipeline_for_generation_only_changes(
    tmp_path: Path,
) -> None:
    runtime = Runtime()
    sources = frozen_sources()
    requested = base_manifest().model_copy(
        update={
            "model_id": "aidealab/AnimeGen-T2V",
            "model_revision": "a" * 40,
            "scheduler": "flow-match-euler-shift-3",
            "dtype": "bfloat16",
            "quantization": "layerwise-fp8",
            "memory_strategy": "model_cpu_offload",
            "loras": (
                LoraBinding(
                    model_id="lightx2v/Wan2.2-Lightning",
                    revision="a" * 40,
                    checksum=canonical_source_digest(sources["lightning-high"]),
                    weight=2.0,
                ),
                LoraBinding(
                    model_id="lightx2v/Wan2.2-Lightning",
                    revision="a" * 40,
                    checksum=canonical_source_digest(sources["lightning-low"]),
                    weight=1.0,
                ),
            ),
            "source_checksums": {
                component: canonical_source_digest(source)
                for component, source in sources.items()
            },
        }
    )
    subject = DiffusersAnimeGenGenerationRuntime(
        cache_root=tmp_path,
        sources=sources,
        runtime=runtime,
        torch=TorchStub,
    )

    first_pipeline, first_observed = subject.prepare(requested)
    assembly_calls = tuple(runtime.calls)
    second_pipeline, second_observed = subject.prepare(
        requested.model_copy(
            update={
                "prompt": "a different controlled prompt",
                "seed": requested.seed + 1,
                "guidance_scale": requested.guidance_scale + 0.25,
            }
        )
    )

    assert second_pipeline is first_pipeline
    assert tuple(runtime.calls) == assembly_calls
    assert second_observed == first_observed
    assert second_observed is not first_observed
    assert isinstance(second_observed, dict)
    second_observed["scheduler"] = "tampered-by-caller"

    _, third_observed = subject.prepare(requested)

    assert third_observed == first_observed


def test_bf16_stage_executor_preserves_bf16_storage_without_layerwise_casting(
    tmp_path: Path,
) -> None:
    runtime = Runtime()
    for name in ("transformer", "transformer_2"):
        runtime.components[name]._parameters[0].dtype = TorchStub.bfloat16

    evidence = AnimeGenStageExecutor(
        cache_root=tmp_path,
        torch=TorchStub,
        runtime=runtime,
        sources=frozen_sources(),
        storage_mode=AnimeGenStorageMode.BF16,
    ).execute(cancelled=lambda: False)

    assert [call[1][1] for call in runtime.calls if call[0] == "transformer"] == [
        TorchStub.bfloat16,
        TorchStub.bfloat16,
    ]
    assert runtime.components["transformer"].cast_calls == []
    assert runtime.components["transformer_2"].cast_calls == []
    assert evidence.details["observed"] == {
        "quantization": "bf16",
        "compute_dtype": "bfloat16",
        "storage_dtype": "bfloat16",
        "active_loras": {"lightning_high": 2.0, "lightning_low": 1.0},
        "casting_calls": 0,
    }
    assert "layerwise_casting" not in [stage["stage"] for stage in evidence.details["stages"]]


def test_production_generation_runtime_assembles_bf16_without_layerwise_casting(
    tmp_path: Path,
) -> None:
    runtime = Runtime()
    sources = frozen_sources()
    for name in ("transformer", "transformer_2"):
        runtime.components[name]._parameters[0].dtype = TorchStub.bfloat16
    requested = base_manifest().model_copy(
        update={
            "model_id": "aidealab/AnimeGen-T2V",
            "model_revision": "a" * 40,
            "scheduler": "flow-match-euler-shift-3",
            "dtype": "bfloat16",
            "quantization": "bf16",
            "memory_strategy": "model_cpu_offload",
            "loras": (
                LoraBinding(
                    model_id="lightx2v/Wan2.2-Lightning",
                    revision="a" * 40,
                    checksum=canonical_source_digest(sources["lightning-high"]),
                    weight=2.0,
                ),
                LoraBinding(
                    model_id="lightx2v/Wan2.2-Lightning",
                    revision="a" * 40,
                    checksum=canonical_source_digest(sources["lightning-low"]),
                    weight=1.0,
                ),
            ),
            "source_checksums": {
                component: canonical_source_digest(source) for component, source in sources.items()
            },
        }
    )

    pipeline, observed = DiffusersAnimeGenGenerationRuntime(
        cache_root=tmp_path,
        sources=sources,
        runtime=runtime,
        torch=TorchStub,
    ).prepare(requested)

    assert pipeline is runtime.pipeline
    assert [call[1][1] for call in runtime.calls if call[0] == "transformer"] == [
        TorchStub.bfloat16,
        TorchStub.bfloat16,
    ]
    assert runtime.components["transformer"].cast_calls == []
    assert runtime.components["transformer_2"].cast_calls == []
    assert observed["quantization"] == "bf16"
    assert observed["compute_dtype"] == "bfloat16"
    assert observed["storage_dtype"] == "bfloat16"
    transformer_calls = [call[1] for call in runtime.calls if call[0] == "transformer"]
    wan_snapshot = tmp_path / "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers" / "snapshots" / ("a" * 40)
    assert [call[2] for call in transformer_calls] == [
        {
            "config": str(wan_snapshot),
            "subfolder": "transformer",
            "local_files_only": True,
        },
        {
            "config": str(wan_snapshot),
            "subfolder": "transformer",
            "local_files_only": True,
        },
    ]
    transformer_config_file = next(
        item
        for item in sources["wan-base"].files
        if item.relative_path == "transformer/config.json"
    )
    assert observed["transformer_config_identity"] == {
        "repo_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "revision": "a" * 40,
        "subfolder": "transformer",
        "relative_path": "transformer/config.json",
        "sha256": transformer_config_file.local_sha256,
        "local_files_only": True,
    }


def test_generation_runtime_accepts_real_shaped_numpy_frame_batch(tmp_path: Path) -> None:
    import numpy as np

    subject = DiffusersAnimeGenGenerationRuntime(
        cache_root=tmp_path,
        sources=frozen_sources(),
        runtime=Runtime(),
        torch=TorchStub,
    )
    frames = np.zeros((1, 17, 80, 128, 3), dtype=np.uint8)

    result = subject.invoke(lambda **_kwargs: SimpleNamespace(frames=frames))

    assert result.shape == (17, 80, 128, 3)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "frames",
    [
        None,
        [],
        pytest.param("empty-outer", id="numpy-empty-outer"),
        pytest.param("empty-first-video", id="numpy-empty-first-video"),
    ],
)
def test_generation_runtime_rejects_missing_or_empty_frame_batch(
    tmp_path: Path, frames: object
) -> None:
    if frames == "empty-outer":
        import numpy as np

        frames = np.zeros((0, 17, 80, 128, 3), dtype=np.uint8)
    elif frames == "empty-first-video":
        import numpy as np

        frames = np.zeros((1, 0, 80, 128, 3), dtype=np.uint8)
    subject = DiffusersAnimeGenGenerationRuntime(
        cache_root=tmp_path,
        sources=frozen_sources(),
        runtime=Runtime(),
        torch=TorchStub,
    )

    with pytest.raises(AdapterExecutionError, match="returned no frames"):
        subject.invoke(lambda **_kwargs: SimpleNamespace(frames=frames))


@pytest.mark.parametrize("dtype", [TorchStub.float8_e4m3fn, DType("float32")])
def test_bf16_generation_runtime_rejects_non_bf16_transformer_storage(
    tmp_path: Path, dtype: object
) -> None:
    runtime = Runtime()
    sources = frozen_sources()
    for name in ("transformer", "transformer_2"):
        runtime.components[name]._parameters[0].dtype = dtype
    requested = base_manifest().model_copy(
        update={
            "model_id": "aidealab/AnimeGen-T2V",
            "model_revision": "a" * 40,
            "scheduler": "flow-match-euler-shift-3",
            "dtype": "bfloat16",
            "quantization": "bf16",
            "memory_strategy": "model_cpu_offload",
            "loras": (
                LoraBinding(
                    model_id="lightx2v/Wan2.2-Lightning",
                    revision="a" * 40,
                    checksum=canonical_source_digest(sources["lightning-high"]),
                    weight=2.0,
                ),
                LoraBinding(
                    model_id="lightx2v/Wan2.2-Lightning",
                    revision="a" * 40,
                    checksum=canonical_source_digest(sources["lightning-low"]),
                    weight=1.0,
                ),
            ),
            "source_checksums": {
                component: canonical_source_digest(source) for component, source in sources.items()
            },
        }
    )

    with pytest.raises(PlacementPolicyError, match="bf16"):
        DiffusersAnimeGenGenerationRuntime(
            cache_root=tmp_path,
            sources=sources,
            runtime=runtime,
            torch=TorchStub,
        ).prepare(requested)


def test_cpu_transformer_requires_inspectable_accelerate_offload_hook(tmp_path: Path) -> None:
    with pytest.raises(PlacementPolicyError, match="offload hook"):
        AnimeGenStageExecutor(
            cache_root=tmp_path,
            torch=TorchStub,
            runtime=Runtime(bad_hook=True),
            sources=frozen_sources(),
        ).execute(cancelled=lambda: False)


@pytest.mark.parametrize("dtype", [TorchStub.bfloat16, DType("float32")])
def test_rejects_missing_fp8_or_unsupported_storage_dtype(tmp_path: Path, dtype: object) -> None:
    runtime = Runtime()
    for name in ("transformer", "transformer_2"):
        runtime.components[name]._parameters[0].dtype = dtype
    with pytest.raises(PlacementPolicyError, match="dtype"):
        AnimeGenStageExecutor(
            cache_root=tmp_path, torch=TorchStub, runtime=runtime, sources=frozen_sources()
        ).execute(cancelled=lambda: False)


def test_cleanup_releases_local_references(tmp_path: Path) -> None:
    runtime = Runtime()
    refs = [weakref.ref(item) for item in runtime.components.values()]
    subject = AnimeGenStageExecutor(
        cache_root=tmp_path, torch=TorchStub, runtime=runtime, sources=frozen_sources()
    )
    with pytest.raises(GenerationCancelled):
        subject.execute(cancelled=lambda: True)
    assert subject.retained_component_count == 0
    assert all(ref() is not None for ref in refs)  # owned only by injected runtime, never executor


@pytest.mark.parametrize(
    ("stage", "call"),
    [
        (FeasibilityStage.LOAD_TEXT_STACK, "text_stack"),
        (FeasibilityStage.LOAD_VAE, "vae"),
        (FeasibilityStage.LOAD_HIGH_TRANSFORMER, "transformer"),
        (FeasibilityStage.LOAD_LOW_TRANSFORMER, "transformer"),
    ],
)
def test_protocol_component_stage_loads_only_requested_component(
    tmp_path: Path, stage: FeasibilityStage, call: str
) -> None:
    runtime = Runtime()
    subject = AnimeGenStageExecutor(
        cache_root=tmp_path, torch=TorchStub, runtime=runtime, sources=frozen_sources()
    )

    evidence = subject.execute_protocol_stage(stage, cancelled=lambda: False)

    assert evidence.details["stage"] == stage.value
    assert [item[0] for item in runtime.calls] == [call]
    assert evidence.details["placement"]["parameter_count"] > 0
    assert evidence.details["placement_policy"] == "declared_cpu_staging"
    if "transformer" in stage.value:
        assert evidence.details["configuration"] == {
            "compute_dtype": "bfloat16",
            "storage_dtype": "float8_e4m3fn",
            "casting_applied": True,
        }
    assert subject.retained_component_count == 0


def test_component_stage_persists_warmup_failure_at_exact_stage(tmp_path: Path) -> None:
    class FailingWarmupTorch(TorchStub):
        empty = staticmethod(
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("warmup failed"))
        )

    subject = AnimeGenStageExecutor(
        cache_root=tmp_path,
        torch=FailingWarmupTorch,
        runtime=Runtime(),
        sources=frozen_sources(),
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        subject.execute_protocol_stage(FeasibilityStage.LOAD_TEXT_STACK, cancelled=lambda: False)


def test_generation_prepare_persists_warmup_failure_before_assembly(tmp_path: Path) -> None:
    class FailingWarmupTorch(TorchStub):
        empty = staticmethod(
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("warmup failed"))
        )

    subject = DiffusersAnimeGenGenerationRuntime(
        cache_root=tmp_path,
        torch=FailingWarmupTorch,
        runtime=Runtime(),
        sources=frozen_sources(),
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        subject.prepare(base_manifest())


def test_protocol_component_stage_rejects_uninspectable_device(tmp_path: Path) -> None:
    runtime = Runtime()
    runtime.components["vae"]._parameters[0].device = "meta"
    subject = AnimeGenStageExecutor(
        cache_root=tmp_path, torch=TorchStub, runtime=runtime, sources=frozen_sources()
    )

    with pytest.raises(PlacementPolicyError, match="uninspectable placement"):
        subject.execute_protocol_stage(FeasibilityStage.LOAD_VAE, cancelled=lambda: False)


def test_protocol_transformer_rejects_ignored_casting(tmp_path: Path) -> None:
    runtime = Runtime()
    transformer = runtime.components["transformer"]
    transformer._parameters = [
        SimpleNamespace(device="cpu", dtype=DType("float32"), requires_grad=False)
    ]
    subject = AnimeGenStageExecutor(
        cache_root=tmp_path, torch=TorchStub, runtime=runtime, sources=frozen_sources()
    )

    with pytest.raises(PlacementPolicyError, match="ignored bf16/FP8"):
        subject.execute_protocol_stage(
            FeasibilityStage.LOAD_HIGH_TRANSFORMER, cancelled=lambda: False
        )


def test_cancelled_generation_never_crosses_artifact_store_boundary(tmp_path: Path) -> None:
    subject = AnimeGenStageExecutor(
        cache_root=tmp_path, torch=TorchStub, runtime=Runtime(), sources=frozen_sources()
    )
    calls: list[str] = []
    store = SimpleNamespace(
        stage=lambda *_: calls.append("stage"), commit=lambda *_: calls.append("commit")
    )

    def invoke(callback):
        partial = tmp_path / "output.partial.mp4"
        partial.write_bytes(b"partial")
        callback(object(), 0, object(), {})
        return partial

    with pytest.raises(GenerationCancelled):
        subject.execute_generation(
            invoke=invoke,
            scratch_root=tmp_path,
            cancelled=lambda: True,
            artifact_store=store,
            job_id="job",
        )
    assert calls == []
    assert not (tmp_path / "output.partial.mp4").exists()


def test_cancel_after_artifact_stage_aborts_staged_handle(tmp_path: Path) -> None:
    subject = AnimeGenStageExecutor(
        cache_root=tmp_path, torch=TorchStub, runtime=Runtime(), sources=frozen_sources()
    )
    calls: list[str] = []
    store = SimpleNamespace(
        stage=lambda *_: calls.append("stage") or "handle",
        commit=lambda *_: calls.append("commit"),
        discard=lambda *_: calls.append("discard"),
    )
    checks = iter((False, False, True))

    def invoke(_callback):
        partial = tmp_path / "output.partial.mp4"
        partial.write_bytes(b"partial")
        return partial

    with pytest.raises(GenerationCancelled):
        subject.execute_generation(
            invoke=invoke,
            scratch_root=tmp_path,
            cancelled=lambda: next(checks),
            artifact_store=store,
            job_id="job",
        )
    assert calls == ["stage", "discard"]


def test_real_artifact_store_is_empty_when_cancelled_after_stage(
    database: Database, tmp_path: Path
) -> None:
    subject = AnimeGenStageExecutor(
        cache_root=tmp_path, torch=TorchStub, runtime=Runtime(), sources=frozen_sources()
    )
    store = ArtifactStore(database, tmp_path / "artifacts")
    checks = iter((False, False, True))

    def invoke(_callback):
        partial = tmp_path / "output.partial.mp4"
        partial.write_bytes(b"partial")
        return partial

    with pytest.raises(GenerationCancelled):
        subject.execute_generation(
            invoke=invoke,
            scratch_root=tmp_path,
            cancelled=lambda: next(checks),
            artifact_store=store,
            job_id="job",
        )
    assert not tuple(store.staging_root.glob("*"))
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0


@pytest.mark.parametrize("cancel_at", range(12))
def test_cancellation_between_all_assembly_operations(tmp_path: Path, cancel_at: int) -> None:
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        result = checks == cancel_at
        checks += 1
        return result

    with pytest.raises(GenerationCancelled):
        AnimeGenStageExecutor(
            cache_root=tmp_path, torch=TorchStub, runtime=Runtime(), sources=frozen_sources()
        ).execute(cancelled=cancelled)
