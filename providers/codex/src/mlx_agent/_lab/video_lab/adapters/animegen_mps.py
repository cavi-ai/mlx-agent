from __future__ import annotations

import gc
import hashlib
import importlib
import json
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast

from video_lab.adapter_protocol import (
    AdapterExecutionError,
    CancellationCheck,
    GenerationCancelled,
    GenerationProgress,
    GenerationRequest,
    GenerationResult,
    ProgressCallback,
)
from video_lab.capabilities import AdapterCapabilities, validate_manifest_support
from video_lab.domain import Manifest, canonical_json
from video_lab.feasibility_models import FeasibilityStage, StageEvidence
from video_lab.feasibility_telemetry import mark_mps_context_ready
from video_lab.runtime_identity import animegen_runtime_versions, runtime_fingerprint
from video_lab.source_inventory import (
    ANIMEGEN_REQUIRED_FILES,
    ANIMEGEN_SOURCE_REPOS,
    ResolvedSource,
    canonical_source_digest,
)

ADAPTER_ID = "diffusers.animegen-t2v.wan2.2-a14b.mps.v1"
_PLACEMENT_SAMPLE_LIMIT = 16
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
STANDARD_VARIABLE_FIELDS = frozenset(
    {
        "prompt",
        "negative_prompt",
        "seed",
        "width",
        "height",
        "num_frames",
        "inference_steps",
        "guidance_scale",
    }
)


class AnimeGenStorageMode(StrEnum):
    LAYERWISE_FP8 = "layerwise-fp8"
    BF16 = "bf16"


def storage_mode_for_manifest(manifest: Manifest) -> AnimeGenStorageMode:
    try:
        return AnimeGenStorageMode(manifest.quantization)
    except ValueError as error:
        raise AdapterExecutionError("unsupported transformer storage") from error


@dataclass(frozen=True)
class GenerationProfile:
    width: int
    height: int
    num_frames: int
    fps: int


@dataclass(frozen=True)
class RepeatabilityEvidence:
    classification: str
    artifact_sha256: tuple[str, str]
    decoded_frame_sha256: tuple[tuple[str, ...], tuple[str, ...]]
    matching_frame_count: int
    differing_frame_count: int
    matching_frame_ratio: float
    manifest_hash: str
    source_fingerprint: str
    runtime_fingerprint: str
    placement_fingerprint: str


@dataclass(frozen=True)
class GenerationInvocation:
    pipeline_kwargs: Mapping[str, object]
    controls: Mapping[str, object]


class AnimeGenGenerationRuntime(Protocol):
    def prepare(self, requested: Manifest) -> tuple[object, Mapping[str, object]]: ...
    def cpu_generator(self, seed: int) -> object: ...
    def invoke(self, pipeline: object, **kwargs: object) -> object: ...
    def encode(self, frames: object, path: Path, fps: int) -> None: ...
    def runtime_versions(self) -> Mapping[str, str]: ...


class AnimeGenMpsAdapter:
    adapter_id = ADAPTER_ID
    adapter_version = "1"

    def __init__(
        self,
        *,
        runtime: AnimeGenGenerationRuntime,
        authorized_profile: tuple[int, int, int],
        exact_profile_only: bool = True,
        expected_fingerprints: tuple[str, str, str] | None = None,
        authorized_manifest: Manifest | None = None,
        authorized_verdict: str | None = None,
        authorized_manifest_hash: str | None = None,
    ) -> None:
        self._generation_runtime = runtime
        self._authorized_profile = authorized_profile
        self._exact_profile_only = exact_profile_only
        self._expected_fingerprints = expected_fingerprints
        self._authorized_manifest = authorized_manifest
        self._authorized_verdict = authorized_verdict
        self._authorized_manifest_hash = authorized_manifest_hash

    def capabilities(self) -> AdapterCapabilities:
        width, height, frames = self._authorized_profile
        return AdapterCapabilities(
            adapter_id=self.adapter_id,
            model_patterns=("aidealab/AnimeGen-T2V",),
            backends=("mps",),
            dtypes=("bfloat16",),
            quantizations=tuple(mode.value for mode in AnimeGenStorageMode),
            schedulers=("flow-match-euler-shift-3",),
            min_width=width if self._exact_profile_only else 16,
            max_width=width,
            width_multiple=16,
            min_height=height if self._exact_profile_only else 16,
            max_height=height,
            height_multiple=16,
            min_frames=frames if self._exact_profile_only else 17,
            max_frames=frames,
            frame_rule="16n+1",
            supports_lora=True,
            min_fps=(1 if self._authorized_manifest is None else self._authorized_manifest.fps),
            max_fps=(120 if self._authorized_manifest is None else self._authorized_manifest.fps),
        )

    def probe(self) -> bool:
        return True

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback,
        cancelled: CancellationCheck,
    ) -> GenerationResult:
        manifest = request.manifest
        self._validate_authorized_request(manifest)
        violations = validate_manifest_support(manifest, self.capabilities())
        if violations:
            raise AdapterExecutionError(
                "unsupported manifest: "
                + "; ".join(f"{item.field}: {item.reason}" for item in violations)
            )
        scratch = request.scratch_root
        partial = scratch / "output.partial.mp4"
        final = scratch / "output.mp4"
        scratch.mkdir(parents=True, exist_ok=True)
        partial.unlink(missing_ok=True)
        final.unlink(missing_ok=True)
        try:
            if cancelled():
                raise GenerationCancelled("AnimeGen generation cancelled")
            pipeline, observed = self._generation_runtime.prepare(manifest)
            self._validate_observed_configuration(manifest, observed)
            runtime_versions = self._generation_runtime.runtime_versions()
            if dict(runtime_versions) != dict(manifest.runtime_versions):
                raise AdapterExecutionError("runtime versions differ from frozen manifest")
            observed_runtime_fingerprint = runtime_fingerprint(runtime_versions)
            if (
                self._expected_fingerprints is not None
                and (
                    observed.get("source_fingerprint"),
                    observed_runtime_fingerprint,
                    observed.get("placement_fingerprint"),
                )
                != self._expected_fingerprints
            ):
                raise AdapterExecutionError("runtime evidence differs from authorized run")
            generator = self._generation_runtime.cpu_generator(manifest.seed)
            invocation = build_generation_invocation(manifest, generator=generator)

            def callback(pipe: object, step: int, timestep: object, kwargs: object) -> object:
                if cancelled():
                    raise GenerationCancelled("AnimeGen generation cancelled")
                progress(
                    GenerationProgress(
                        phase="generating",
                        completed=step + 1,
                        total=manifest.inference_steps,
                        message=f"completed diffusion step {step + 1}",
                        elapsed_seconds=0,
                    )
                )
                return kwargs

            frames = self._generation_runtime.invoke(
                pipeline, **invocation.pipeline_kwargs, callback_on_step_end=callback
            )
            self._generation_runtime.encode(frames, partial, manifest.fps)
            if cancelled():
                raise GenerationCancelled("AnimeGen generation cancelled")
            if not partial.is_file() or partial.stat().st_size == 0:
                raise AdapterExecutionError("AnimeGen encoder produced no MP4")
            partial.replace(final)
            return GenerationResult(
                video_path=final,
                effective_manifest=manifest,
                frame_count=manifest.num_frames,
                fps=manifest.fps,
                width=manifest.width,
                height=manifest.height,
                adapter_version=self.adapter_version,
                runtime_versions=runtime_versions,
                adapter_telemetry={
                    "generator_device": "cpu",
                    "execution_placement": observed["execution_placement"],
                    "observed_configuration": dict(observed),
                },
            )
        except (GenerationCancelled, AdapterExecutionError):
            partial.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            raise
        except Exception as error:
            partial.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            raise AdapterExecutionError(
                f"AnimeGen generation failed: {type(error).__name__}: {str(error)[:500]}"
            ) from error

    @staticmethod
    def _validate_observed_configuration(
        manifest: Manifest, observed: Mapping[str, object]
    ) -> None:
        storage_mode = storage_mode_for_manifest(manifest)
        expected = {
            "scheduler": manifest.scheduler,
            "dtype": manifest.dtype,
            "quantization": manifest.quantization,
            "compute_dtype": manifest.dtype,
            "storage_dtype": (
                "float8_e4m3fn" if storage_mode is AnimeGenStorageMode.LAYERWISE_FP8 else "bfloat16"
            ),
            "loras": tuple(item.model_dump(mode="json") for item in manifest.loras),
            "memory_strategy": manifest.memory_strategy,
            "execution_placement": "local_hybrid:mps",
        }
        if any(observed.get(key) != value for key, value in expected.items()):
            raise AdapterExecutionError("observed AnimeGen configuration differs from manifest")
        for key in ("source_fingerprint", "placement_fingerprint"):
            value = observed.get(key)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise AdapterExecutionError(f"observed {key} is invalid")

    def _validate_authorized_request(self, manifest: Manifest) -> None:
        baseline = self._authorized_manifest
        if baseline is None:
            return
        baseline_hash = hashlib.sha256(canonical_json(baseline).encode()).hexdigest()
        if baseline_hash != self._authorized_manifest_hash:
            raise AdapterExecutionError("authorized manifest hash is invalid")
        if self._authorized_verdict == "experimental_local_hybrid":
            if manifest != baseline:
                raise AdapterExecutionError("experimental request differs from authorized manifest")
            return
        requested = manifest.model_dump(mode="json")
        authorized = baseline.model_dump(mode="json")
        for field in STANDARD_VARIABLE_FIELDS:
            requested.pop(field)
            authorized.pop(field)
        if requested != authorized:
            raise AdapterExecutionError("standard request drifts from authorized frozen controls")


def build_generation_invocation(manifest: Manifest, *, generator: object) -> GenerationInvocation:
    return GenerationInvocation(
        pipeline_kwargs={
            "prompt": manifest.prompt,
            "negative_prompt": manifest.negative_prompt,
            "height": manifest.height,
            "width": manifest.width,
            "num_frames": manifest.num_frames,
            "num_inference_steps": manifest.inference_steps,
            "guidance_scale": manifest.guidance_scale,
            "generator": generator,
        },
        controls={
            "seed": manifest.seed,
            "scheduler": manifest.scheduler,
            "dtype": manifest.dtype,
            "quantization": manifest.quantization,
            "loras": tuple(item.model_dump(mode="json") for item in manifest.loras),
            "memory_strategy": manifest.memory_strategy,
            "execution_placement": "local_hybrid:mps",
            "generator_device": "cpu",
        },
    )


def select_minimum_profile(
    *,
    profiles: tuple[GenerationProfile, ...],
    width_multiple: int,
    height_multiple: int,
) -> GenerationProfile:
    valid = tuple(
        profile
        for profile in profiles
        if profile.width > 0
        and profile.height > 0
        and profile.fps > 0
        and profile.width % width_multiple == 0
        and profile.height % height_multiple == 0
        and profile.num_frames >= 17
        and (profile.num_frames - 1) % 16 == 0
    )
    if not valid or len(valid) != len(profiles):
        raise ValueError("pipeline configuration has no valid minimum generation profile")
    return min(valid, key=lambda item: (item.width * item.height, item.num_frames, item.fps))


def classify_repeatability(
    *,
    first_artifact_sha256: str,
    second_artifact_sha256: str,
    first_frame_sha256: tuple[str, ...],
    second_frame_sha256: tuple[str, ...],
    manifest_hash: str,
    source_fingerprint: str,
    runtime_fingerprint: str,
    placement_fingerprint: str,
    second_manifest_hash: str | None = None,
    second_source_fingerprint: str | None = None,
    second_runtime_fingerprint: str | None = None,
    second_placement_fingerprint: str | None = None,
) -> RepeatabilityEvidence:
    hashes = (
        first_artifact_sha256,
        second_artifact_sha256,
        manifest_hash,
        source_fingerprint,
        runtime_fingerprint,
        placement_fingerprint,
        *first_frame_sha256,
        *second_frame_sha256,
    )
    provenance_matches = (
        (second_manifest_hash or manifest_hash) == manifest_hash
        and (second_source_fingerprint or source_fingerprint) == source_fingerprint
        and (second_runtime_fingerprint or runtime_fingerprint) == runtime_fingerprint
        and (second_placement_fingerprint or placement_fingerprint) == placement_fingerprint
    )
    valid = all(_SHA256.fullmatch(value) for value in hashes) and bool(first_frame_sha256)
    valid = valid and len(first_frame_sha256) == len(second_frame_sha256) and provenance_matches
    if not valid:
        classification = "invalid"
        matching_frame_count = 0
        differing_frame_count = 0
        matching_frame_ratio = 0.0
    else:
        matching_frame_count = sum(
            left == right
            for left, right in zip(first_frame_sha256, second_frame_sha256, strict=True)
        )
        differing_frame_count = len(first_frame_sha256) - matching_frame_count
        matching_frame_ratio = matching_frame_count / len(first_frame_sha256)
        if first_artifact_sha256 == second_artifact_sha256:
            classification = "bitwise"
        elif first_frame_sha256 == second_frame_sha256:
            classification = "decoded_frame"
        else:
            classification = "numerically_nondeterministic"
    return RepeatabilityEvidence(
        classification=classification,
        artifact_sha256=(first_artifact_sha256, second_artifact_sha256),
        decoded_frame_sha256=(first_frame_sha256, second_frame_sha256),
        matching_frame_count=matching_frame_count,
        differing_frame_count=differing_frame_count,
        matching_frame_ratio=matching_frame_ratio,
        manifest_hash=manifest_hash,
        source_fingerprint=source_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        placement_fingerprint=placement_fingerprint,
    )


@dataclass(frozen=True)
class AnimeGenAssembly:
    animegen_repo: str = "aidealab/AnimeGen-T2V"
    wan_repo: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    scheduler_shift: float = 3.0
    compute_dtype: str = "bfloat16"
    storage_dtype: str = "float8_e4m3fn"
    loras: tuple[tuple[str, str, float], ...] = (
        (
            "lightx2v/Wan2.2-Lightning",
            "Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
            2.0,
        ),
        (
            "lightx2v/Wan2.2-Lightning",
            "Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",
            1.0,
        ),
    )


ANIMEGEN_ASSEMBLY = AnimeGenAssembly()


@dataclass(frozen=True)
class CastingObservation:
    storage_dtype: str
    compute_dtype: str


class PlacementPolicyError(RuntimeError):
    """Observed execution state does not prove the declared strategy."""


class AnimeGenRuntime(Protocol):
    def load_text_stack(self, repo: str, **kwargs: object) -> Mapping[str, object]: ...
    def load_vae(self, repo: str, **kwargs: object) -> object: ...
    def load_transformer_file(self, path: Path, **kwargs: object) -> object: ...
    def load_scheduler_config(self, repo: str, **kwargs: object) -> object: ...
    def scheduler_from_config(self, config: object, *, shift: float) -> object: ...
    def assemble_pipeline(self, repo: str, **components: object) -> object: ...
    def observe_adapters(self, pipeline: object) -> Mapping[str, float]: ...


class _DiffusersRuntime:
    """Lazy real-library boundary; its signatures mirror upstream public APIs."""

    def load_text_stack(self, repo: str, **kwargs: object) -> Mapping[str, object]:
        transformers = importlib.import_module("transformers")
        tokenizer_kwargs = {**kwargs, "use_fast": True}
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            repo, subfolder="tokenizer", **tokenizer_kwargs
        )
        text_encoder = transformers.UMT5EncoderModel.from_pretrained(
            repo, subfolder="text_encoder", **kwargs
        )
        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_vae(self, repo: str, **kwargs: object) -> object:
        cls = importlib.import_module("diffusers").AutoencoderKLWan
        return cls.from_pretrained(repo, **kwargs)

    def load_transformer_file(self, path: Path, **kwargs: object) -> object:
        cls = importlib.import_module("diffusers").WanTransformer3DModel
        return cls.from_single_file(path, **kwargs)

    def load_scheduler_config(self, repo: str, **kwargs: object) -> object:
        cls = importlib.import_module("diffusers").FlowMatchEulerDiscreteScheduler
        return cls.load_config(repo, **kwargs)

    def scheduler_from_config(self, config: object, *, shift: float) -> object:
        cls = importlib.import_module("diffusers").FlowMatchEulerDiscreteScheduler
        return cls.from_config(config, shift=shift)

    def assemble_pipeline(self, repo: str, **components: object) -> object:
        cls = importlib.import_module("diffusers").WanPipeline
        return cls.from_pretrained(repo, **components)

    def observe_adapters(self, pipeline: object) -> Mapping[str, float]:
        peft_pipeline = cast(Any, pipeline)
        active = tuple(peft_pipeline.get_active_adapters())
        values: dict[str, set[float]] = {name: set() for name in active}
        for component_name in ("transformer", "transformer_2"):
            component = getattr(peft_pipeline, component_name, None)
            for _name, module in getattr(component, "named_modules", lambda: ())():
                scaling = getattr(module, "scaling", None)
                lora_alpha = getattr(module, "lora_alpha", None)
                ranks = getattr(module, "r", None)
                if isinstance(scaling, Mapping):
                    for adapter in active:
                        if (
                            adapter in scaling
                            and isinstance(lora_alpha, Mapping)
                            and isinstance(ranks, Mapping)
                            and adapter in lora_alpha
                            and adapter in ranks
                            and float(ranks[adapter]) != 0
                        ):
                            base = float(lora_alpha[adapter]) / float(ranks[adapter])
                            values[adapter].add(float(scaling[adapter]) / base)
        if any(len(scales) != 1 for scales in values.values()):
            raise PlacementPolicyError("PEFT adapter scales are not uniquely observable")
        return {name: next(iter(scales)) for name, scales in values.items()}


class DiffusersAnimeGenGenerationRuntime:
    """Lazy production runtime assembled from the frozen Task 4 source inventory."""

    def __init__(
        self,
        *,
        cache_root: Path,
        sources: Mapping[str, ResolvedSource],
        runtime: AnimeGenRuntime | None = None,
        torch: Any | None = None,
    ) -> None:
        self._cache_root = cache_root
        self._sources = sources
        self._runtime = runtime or _DiffusersRuntime()
        self._torch = torch
        self._executor = AnimeGenStageExecutor(
            cache_root=cache_root,
            sources=sources,
            runtime=self._runtime,
            torch=torch,
        )
        self._prepared: tuple[str, object, dict[str, object]] | None = None

    def prepare(self, requested: Manifest) -> tuple[object, Mapping[str, object]]:
        torch = self._executor._warm_mps()
        storage_mode = storage_mode_for_manifest(requested)
        if self._executor.storage_mode is not storage_mode:
            self._executor = AnimeGenStageExecutor(
                cache_root=self._cache_root,
                sources=self._sources,
                runtime=self._runtime,
                torch=self._torch,
                storage_mode=storage_mode,
            )
            torch = self._executor._warm_mps()
        sources = self._executor._validated_sources()
        self._validate_manifest_bindings(requested, sources)
        compatibility = requested.model_dump(mode="json")
        for field in STANDARD_VARIABLE_FIELDS:
            compatibility.pop(field)
        compatibility_digest = hashlib.sha256(
            canonical_json(compatibility).encode()
        ).hexdigest()
        if self._prepared is not None and self._prepared[0] == compatibility_digest:
            return self._prepared[1], deepcopy(self._prepared[2])
        animegen = sources["animegen"]
        wan = sources["wan-base"]
        wan_snapshot = self._executor._snapshot_dir(wan)
        common = {"local_files_only": True, "torch_dtype": torch.bfloat16}
        text = self._executor._runtime.load_text_stack(str(wan_snapshot), **common)
        vae = self._executor._runtime.load_vae(str(wan_snapshot), subfolder="vae", **common)
        high = self._executor._runtime.load_transformer_file(
            self._executor._snapshot_file(animegen, "high_noise.safetensors"),
            config=str(wan_snapshot),
            subfolder="transformer",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        low = self._executor._runtime.load_transformer_file(
            self._executor._snapshot_file(animegen, "low_noise.safetensors"),
            config=str(wan_snapshot),
            subfolder="transformer",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        casting_observations: list[CastingObservation] = []
        for transformer in (high, low):
            observation = self._executor._configure_transformer_storage(transformer)
            if observation is not None:
                casting_observations.append(observation)
        config = self._executor._runtime.load_scheduler_config(
            str(wan_snapshot), subfolder="scheduler", local_files_only=True
        )
        scheduler = self._executor._runtime.scheduler_from_config(config, shift=3.0)
        pipeline = self._executor._runtime.assemble_pipeline(
            str(wan_snapshot),
            **dict(text),
            vae=vae,
            transformer=high,
            transformer_2=low,
            scheduler=scheduler,
            local_files_only=True,
        )
        self._executor._apply_and_observe_loras(pipeline)
        offload = getattr(pipeline, "enable_model_cpu_offload", None)
        if not callable(offload):
            raise PlacementPolicyError("model CPU offload is unavailable")
        offload(device="mps")
        placement = self._executor._observe_placement(pipeline)
        transformer_config_file = next(
            (item for item in wan.files if item.relative_path == "transformer/config.json"),
            None,
        )
        if transformer_config_file is None or not (
            transformer_config_sha256 := (
                transformer_config_file.local_sha256 or transformer_config_file.upstream_sha256
            )
        ):
            raise AdapterExecutionError("pinned Wan transformer config identity is unavailable")
        transformer_config_identity = {
            "repo_id": wan.repo_id,
            "revision": wan.revision,
            "subfolder": "transformer",
            "relative_path": "transformer/config.json",
            "sha256": transformer_config_sha256,
            "local_files_only": True,
        }
        source_payload: dict[str, object] = {
            key: {
                "repo": source.repo_id,
                "revision": source.revision,
                "files": tuple(
                    (
                        item.relative_path,
                        item.local_sha256 or item.upstream_sha256,
                        item.size,
                    )
                    for item in source.files
                ),
            }
            for key, source in sorted(sources.items())
        }
        source_payload["observed_transformer_config"] = transformer_config_identity
        source_fingerprint = hashlib.sha256(
            json.dumps(source_payload, sort_keys=True).encode()
        ).hexdigest()
        placement_fingerprint = hashlib.sha256(
            json.dumps(placement, sort_keys=True).encode()
        ).hexdigest()
        configuration = self._executor._observe_configuration(
            pipeline, high, low, tuple(casting_observations)
        )
        observed_scales = self._executor._runtime.observe_adapters(pipeline)
        observed_loras = tuple(
            {
                "model_id": sources[source_name].repo_id,
                "revision": sources[source_name].revision,
                "checksum": canonical_source_digest(sources[source_name]),
                "weight": observed_scales[adapter_name],
            }
            for source_name, (_repo, _filename, _weight), adapter_name in zip(
                ("lightning-high", "lightning-low"),
                ANIMEGEN_ASSEMBLY.loras,
                ("lightning_high", "lightning_low"),
                strict=True,
            )
        )
        observed = {
            "scheduler": "flow-match-euler-shift-3",
            "dtype": "bfloat16",
            "quantization": configuration["quantization"],
            "compute_dtype": configuration["compute_dtype"],
            "storage_dtype": configuration["storage_dtype"],
            "transformer_config_identity": transformer_config_identity,
            "loras": observed_loras,
            "memory_strategy": "model_cpu_offload",
            "source_fingerprint": source_fingerprint,
            "placement_fingerprint": placement_fingerprint,
            "execution_placement": "local_hybrid:mps",
        }
        self._prepared = (compatibility_digest, pipeline, deepcopy(observed))
        return pipeline, observed

    @staticmethod
    def _validate_manifest_bindings(
        requested: Manifest, sources: Mapping[str, ResolvedSource]
    ) -> None:
        storage_mode_for_manifest(requested)
        if (
            requested.model_id != ANIMEGEN_ASSEMBLY.animegen_repo
            or requested.model_revision != sources["animegen"].revision
            or requested.scheduler != "flow-match-euler-shift-3"
            or requested.dtype != ANIMEGEN_ASSEMBLY.compute_dtype
            or requested.memory_strategy != "model_cpu_offload"
        ):
            raise AdapterExecutionError("manifest differs from frozen AnimeGen assembly")
        expected: list[tuple[str, str, str, float]] = []
        expected_source_checksums = {
            component: canonical_source_digest(source) for component, source in sources.items()
        }
        if dict(requested.source_checksums) != expected_source_checksums:
            raise AdapterExecutionError("manifest checksums differ from frozen sources")
        for source_name, (_repo, _filename, weight) in zip(
            ("lightning-high", "lightning-low"), ANIMEGEN_ASSEMBLY.loras, strict=True
        ):
            source = sources[source_name]
            expected.append(
                (source.repo_id, source.revision, expected_source_checksums[source_name], weight)
            )
        actual = [
            (item.model_id, item.revision, item.checksum, item.weight) for item in requested.loras
        ]
        if actual != expected:
            raise AdapterExecutionError("manifest LoRA bindings differ from frozen sources")

    def cpu_generator(self, seed: int) -> object:
        return self._executor._get_torch().Generator(device="cpu").manual_seed(seed)

    def invoke(self, pipeline: object, **kwargs: object) -> object:
        output = cast(Any, pipeline)(**kwargs)
        frames = getattr(output, "frames", None)
        if frames is None or len(frames) == 0 or len(frames[0]) == 0:
            raise AdapterExecutionError("AnimeGen pipeline returned no frames")
        return frames[0]

    def encode(self, frames: object, path: Path, fps: int) -> None:
        importlib.import_module("diffusers.utils").export_to_video(frames, str(path), fps=fps)

    def runtime_versions(self) -> Mapping[str, str]:
        return animegen_runtime_versions()


class ArtifactBoundary(Protocol):
    def stage(self, source: Path, media_type: str) -> object: ...
    def commit(self, staged: object, job_id: str | None) -> object: ...
    def discard(self, staged: object) -> None: ...


GenerationInvoker = Callable[[Callable[[object, int, object, object], object]], Path]


class AnimeGenStageExecutor:
    def __init__(
        self,
        *,
        cache_root: Path,
        torch: Any | None = None,
        runtime: AnimeGenRuntime | None = None,
        sources: Mapping[str, ResolvedSource] | None = None,
        storage_mode: AnimeGenStorageMode = AnimeGenStorageMode.LAYERWISE_FP8,
    ) -> None:
        self._cache_root = Path(cache_root)
        self._torch = torch
        self._runtime = runtime or _DiffusersRuntime()
        self._sources = sources
        self._storage_mode = storage_mode
        self._retained: list[object] = []

    @property
    def retained_component_count(self) -> int:
        return len(self._retained)

    @property
    def storage_mode(self) -> AnimeGenStorageMode:
        return self._storage_mode

    def _get_torch(self) -> Any:
        if self._torch is None:
            self._torch = importlib.import_module("torch")
        return self._torch

    def _warm_mps(self) -> Any:
        torch = self._get_torch()
        if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
            raise PlacementPolicyError("MPS is unavailable")
        torch.empty(1, device="mps")
        mark_mps_context_ready(torch)
        return torch

    def execute_protocol_stage(
        self, stage: FeasibilityStage, *, cancelled: CancellationCheck
    ) -> StageEvidence:
        """Measure one independently resumable component-load stage."""
        if stage is FeasibilityStage.ASSEMBLE_PIPELINE:
            return self.execute(cancelled=cancelled)
        loaders: dict[FeasibilityStage, Callable[[], object]] = {
            FeasibilityStage.LOAD_TEXT_STACK: self._protocol_text_stack,
            FeasibilityStage.LOAD_VAE: self._protocol_vae,
            FeasibilityStage.LOAD_HIGH_TRANSFORMER: lambda: self._protocol_transformer(
                "high_noise.safetensors"
            ),
            FeasibilityStage.LOAD_LOW_TRANSFORMER: lambda: self._protocol_transformer(
                "low_noise.safetensors"
            ),
        }
        loader = loaders.get(stage)
        if loader is None:
            raise ValueError(f"unsupported AnimeGen component stage: {stage.value}")
        torch = self._warm_mps()
        self._checkpoint(cancelled)
        started = monotonic()
        component = loader()
        self._retained.append(component)
        try:
            self._checkpoint(cancelled)
            observation = self._protocol_observation(stage, component)
            return StageEvidence(
                measurements={"duration_seconds": monotonic() - started},
                details=cast(
                    Any,
                    {
                        "stage": stage.value,
                        "local_files_only": True,
                        **observation,
                    },
                ),
            )
        finally:
            self._retained.clear()
            component = None
            gc.collect()
            try:
                torch.mps.empty_cache()
            except (AttributeError, RuntimeError):
                pass

    def _protocol_observation(
        self, stage: FeasibilityStage, component: object
    ) -> dict[str, object]:
        placement = dict(self._component_summary(stage.value, component))
        count = placement.get("parameter_count")
        counts = placement.get("counts")
        if (
            not isinstance(count, int)
            or count <= 0
            or not isinstance(counts, Mapping)
            or any(device not in {"cpu", "mps"} for device in counts)
        ):
            raise PlacementPolicyError(f"uninspectable placement for {stage.value}")
        evidence: dict[str, object] = {
            "placement": placement,
            "placement_policy": (
                "declared_cpu_staging" if counts.get("cpu", 0) else "mps_resident"
            ),
        }
        if stage in {
            FeasibilityStage.LOAD_HIGH_TRANSFORMER,
            FeasibilityStage.LOAD_LOW_TRANSFORMER,
        }:
            compute = _dtype_name(getattr(component, "dtype", "unknown"))
            storage = {
                _dtype_name(getattr(parameter, "dtype", "unknown"))
                for parameter in getattr(component, "parameters", lambda: ())()
            }
            if self._storage_mode is AnimeGenStorageMode.LAYERWISE_FP8:
                fp8 = storage & {"float8_e4m3fn", "float8_e5m2"}
                if (
                    compute != "bfloat16"
                    or not fp8
                    or storage
                    - {
                        "bfloat16",
                        "float8_e4m3fn",
                        "float8_e5m2",
                    }
                ):
                    raise PlacementPolicyError("isolated transformer ignored bf16/FP8 casting")
                storage_dtype = sorted(fp8)[0]
            elif compute != "bfloat16" or storage != {"bfloat16"}:
                raise PlacementPolicyError("isolated transformer does not retain bf16 storage")
            else:
                storage_dtype = "bfloat16"
            evidence["configuration"] = {
                "compute_dtype": compute,
                "storage_dtype": storage_dtype,
                "casting_applied": self._storage_mode is AnimeGenStorageMode.LAYERWISE_FP8,
            }
        return evidence

    def _protocol_text_stack(self) -> object:
        wan = self._validated_sources()["wan-base"]
        return self._runtime.load_text_stack(
            str(self._snapshot_dir(wan)),
            local_files_only=True,
            torch_dtype=self._get_torch().bfloat16,
        )

    def _protocol_vae(self) -> object:
        wan = self._validated_sources()["wan-base"]
        return self._runtime.load_vae(
            str(self._snapshot_dir(wan)),
            subfolder="vae",
            local_files_only=True,
            torch_dtype=self._get_torch().bfloat16,
        )

    def _protocol_transformer(self, filename: str) -> object:
        animegen = self._validated_sources()["animegen"]
        wan = self._validated_sources()["wan-base"]
        transformer = self._runtime.load_transformer_file(
            self._snapshot_file(animegen, filename),
            config=str(self._snapshot_dir(wan)),
            subfolder="transformer",
            local_files_only=True,
            torch_dtype=self._get_torch().bfloat16,
        )
        self._configure_transformer_storage(transformer)
        return transformer

    def _configure_transformer_storage(self, transformer: object) -> CastingObservation | None:
        if self._storage_mode is AnimeGenStorageMode.BF16:
            self._require_bf16_transformer_storage(transformer)
            return None
        casting = getattr(transformer, "enable_layerwise_casting", None)
        if not callable(casting):
            raise PlacementPolicyError("layerwise FP8 casting is unsupported")
        casting(
            storage_dtype=self._get_torch().float8_e4m3fn,
            compute_dtype=self._get_torch().bfloat16,
        )
        return CastingObservation("float8_e4m3fn", "bfloat16")

    @staticmethod
    def _require_bf16_transformer_storage(transformer: object) -> None:
        storage = {
            _dtype_name(getattr(parameter, "dtype", "unknown"))
            for parameter in getattr(transformer, "parameters", lambda: ())()
        }
        if storage != {"bfloat16"}:
            raise PlacementPolicyError("bf16 transformer storage must be exactly bfloat16")

    def execute(self, *, cancelled: CancellationCheck) -> StageEvidence:
        torch = self._warm_mps()
        common = {
            "local_files_only": True,
            "torch_dtype": torch.bfloat16,
        }
        stages: list[dict[str, object]] = []
        sources = self._validated_sources()
        animegen = sources["animegen"]
        wan = sources["wan-base"]
        wan_snapshot = self._snapshot_dir(wan)
        self._retained.clear()
        try:
            text = self._load_stage(
                "text_stack",
                stages,
                cancelled,
                lambda: self._runtime.load_text_stack(str(wan_snapshot), **common),
            )
            vae = self._load_stage(
                "vae",
                stages,
                cancelled,
                lambda: self._runtime.load_vae(str(wan_snapshot), subfolder="vae", **common),
            )
            high = self._load_stage(
                "transformer_high",
                stages,
                cancelled,
                lambda: self._runtime.load_transformer_file(
                    self._snapshot_file(animegen, "high_noise.safetensors"),
                    config=str(wan_snapshot),
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                ),
            )
            low = self._load_stage(
                "transformer_low",
                stages,
                cancelled,
                lambda: self._runtime.load_transformer_file(
                    self._snapshot_file(animegen, "low_noise.safetensors"),
                    config=str(wan_snapshot),
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                ),
            )
            casting_observations: list[CastingObservation] = []
            for transformer in (high, low):
                self._checkpoint(cancelled)
                cast_started = monotonic()
                observation = self._configure_transformer_storage(transformer)
                if observation is not None:
                    casting_observations.append(observation)
                    stages.append(
                        {
                            "stage": "layerwise_casting",
                            "duration_seconds": monotonic() - cast_started,
                        }
                    )
            self._checkpoint(cancelled)
            config_started = monotonic()
            config = self._runtime.load_scheduler_config(
                str(wan_snapshot),
                subfolder="scheduler",
                local_files_only=True,
            )
            stages.append(
                {"stage": "scheduler_config", "duration_seconds": monotonic() - config_started}
            )
            self._checkpoint(cancelled)
            scheduler_started = monotonic()
            scheduler = self._runtime.scheduler_from_config(config, shift=3.0)
            stages.append(
                {
                    "stage": "scheduler_construct",
                    "duration_seconds": monotonic() - scheduler_started,
                    "configuration": {"shift": 3.0},
                }
            )
            self._checkpoint(cancelled)
            assembled = monotonic()
            pipeline = self._runtime.assemble_pipeline(
                str(wan_snapshot),
                **dict(cast(Mapping[str, object], text)),
                vae=vae,
                transformer=high,
                transformer_2=low,
                scheduler=scheduler,
                local_files_only=True,
            )
            self._retained.append(pipeline)
            stages.append(
                {"stage": "assemble_pipeline", "duration_seconds": monotonic() - assembled}
            )
            self._checkpoint(cancelled)
            lora_started = monotonic()
            self._apply_and_observe_loras(pipeline)
            stages.append({"stage": "apply_loras", "duration_seconds": monotonic() - lora_started})
            self._checkpoint(cancelled)
            offload_started = monotonic()
            offload = getattr(pipeline, "enable_model_cpu_offload", None)
            if not callable(offload):
                raise PlacementPolicyError("model CPU offload is unavailable")
            offload(device="mps")
            stages.append(
                {"stage": "enable_offload", "duration_seconds": monotonic() - offload_started}
            )
            placement = self._observe_placement(pipeline)
            observed = self._observe_configuration(pipeline, high, low, tuple(casting_observations))
            self._checkpoint(cancelled)
            return StageEvidence(
                measurements={"component_stage_count": float(len(stages))},
                details=cast(
                    Any,
                    {"stages": tuple(stages), "placement": placement, "observed": observed},
                ),
            )
        finally:
            self._retained.clear()
            text = vae = high = low = pipeline = None  # type: ignore[assignment,unused-ignore]
            gc.collect()
            try:
                torch.mps.empty_cache()
            except (AttributeError, RuntimeError):
                pass

    def _validated_sources(self) -> Mapping[str, ResolvedSource]:
        if self._sources is None:
            raise PlacementPolicyError("frozen source inventory is required")
        required = {"animegen", "wan-base", "lightning-high", "lightning-low"}
        if set(self._sources) != required:
            raise PlacementPolicyError("frozen source inventory is incomplete")
        for key, source in self._sources.items():
            if len(source.revision) != 40 or any(
                char not in "0123456789abcdef" for char in source.revision
            ):
                raise PlacementPolicyError("source revision must be an exact 40-character commit")
            component = key
            repo = ANIMEGEN_SOURCE_REPOS[key]
            files = set(ANIMEGEN_REQUIRED_FILES[key])
            actual_files = {item.relative_path for item in source.files}
            if source.component != component or source.repo_id != repo or not files <= actual_files:
                raise PlacementPolicyError(f"frozen source identity mismatch for {key}")
        return self._sources

    def _snapshot_file(self, source: ResolvedSource, filename: str) -> Path:
        return self._snapshot_dir(source) / filename

    def _snapshot_dir(self, source: ResolvedSource) -> Path:
        repo = source.repo_id.replace("/", "--")
        return self._cache_root / f"models--{repo}" / "snapshots" / source.revision

    def _load_stage(
        self,
        name: str,
        stages: list[dict[str, object]],
        cancelled: CancellationCheck,
        loader: Callable[[], object],
    ) -> object:
        self._checkpoint(cancelled)
        started = monotonic()
        component = loader()
        self._retained.append(component)
        stages.append(
            {
                "stage": name,
                "duration_seconds": monotonic() - started,
                "placement": self._component_summary(name, component),
            }
        )
        return component

    def execute_generation(
        self,
        *,
        invoke: GenerationInvoker,
        scratch_root: Path,
        cancelled: CancellationCheck,
        artifact_store: ArtifactBoundary,
        job_id: str,
    ) -> object:
        partial = scratch_root / "output.partial.mp4"
        final = scratch_root / "output.mp4"
        partial.unlink(missing_ok=True)
        final.unlink(missing_ok=True)
        staged: object | None = None

        def callback(_pipe: object, _step: int, _time: object, kwargs: object) -> object:
            self._checkpoint(cancelled)
            return kwargs

        try:
            produced = invoke(callback)
            self._checkpoint(cancelled)
            if produced != partial or not partial.is_file() or not partial.stat().st_size:
                raise RuntimeError("invalid partial artifact")
            partial.replace(final)
            self._checkpoint(cancelled)
            staged = artifact_store.stage(final, "video/mp4")
            self._checkpoint(cancelled)
            return artifact_store.commit(staged, job_id)
        except BaseException:
            if staged is not None:
                artifact_store.discard(staged)
            partial.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            raise
        finally:
            try:
                self._get_torch().mps.empty_cache()
            except (AttributeError, RuntimeError):
                pass

    @staticmethod
    def _checkpoint(cancelled: CancellationCheck) -> None:
        if cancelled():
            raise GenerationCancelled("AnimeGen execution cancelled")

    def _apply_and_observe_loras(self, pipeline: object) -> None:
        load = getattr(pipeline, "load_lora_weights", None)
        set_adapters = getattr(pipeline, "set_adapters", None)
        if not callable(load) or not callable(set_adapters):
            raise PlacementPolicyError("LoRA APIs unavailable")
        names = ["lightning_high", "lightning_low"]
        source_names = ("lightning-high", "lightning-low")
        for (_repo, weight, _), name, source_name in zip(
            ANIMEGEN_ASSEMBLY.loras, names, source_names, strict=True
        ):
            source = self._sources[source_name]  # type: ignore[index]
            load(
                str(self._snapshot_dir(source)),
                weight_name=weight,
                adapter_name=name,
                local_files_only=True,
            )
        set_adapters(names, adapter_weights=[2.0, 1.0])

    @staticmethod
    def _component_summary(name: str, component: object) -> Mapping[str, object]:
        members = component.values() if isinstance(component, Mapping) else (component,)
        counts: dict[str, int] = {}
        sample: list[dict[str, str]] = []
        total = 0
        for member in members:
            for parameter in getattr(member, "parameters", lambda: ())():
                device = str(getattr(parameter, "device", "unknown")).split(":", 1)[0]
                counts[device] = counts.get(device, 0) + 1
                total += 1
                if len(sample) < _PLACEMENT_SAMPLE_LIMIT:
                    sample.append({"device": device})
        return {
            "component": name,
            "parameter_count": total,
            "counts": counts,
            "sample": tuple(sample),
            "truncated": total > len(sample),
        }

    def _observe_placement(self, pipeline: object) -> Mapping[str, object]:
        components = getattr(pipeline, "components", None)
        if not isinstance(components, Mapping):
            raise PlacementPolicyError("pipeline placement is uninspectable")
        sample: list[dict[str, str]] = []
        counts = {"mps": 0, "cpu_offloaded": 0}
        for name, component in components.items():
            for index, parameter in enumerate(getattr(component, "parameters", lambda: ())()):
                device = str(getattr(parameter, "device", "unknown"))
                if device.startswith("mps"):
                    counts["mps"] += 1
                elif device.startswith("cpu"):
                    hook = getattr(component, "_hf_hook", None)
                    execution_device = str(getattr(hook, "execution_device", ""))
                    if not execution_device.startswith("mps"):
                        raise PlacementPolicyError(
                            f"CPU component {name} lacks an MPS offload hook"
                        )
                    counts["cpu_offloaded"] += 1
                else:
                    raise PlacementPolicyError(f"uninspectable placement for {name}")
                if len(sample) < _PLACEMENT_SAMPLE_LIMIT:
                    sample.append({"name": f"{name}.parameters[{index}]", "device": device})
        return {
            "counts": counts,
            "sample": tuple(sample),
            "truncated": sum(counts.values()) > len(sample),
        }

    def _observe_configuration(
        self,
        pipeline: object,
        high: object,
        low: object,
        casting_observations: tuple[CastingObservation, ...],
    ) -> Mapping[str, object]:
        compute = {_dtype_name(getattr(item, "dtype", "unknown")) for item in (high, low)}
        storage: set[str] = set()
        for item in (high, low):
            for parameter in getattr(item, "parameters", lambda: ())():
                storage.add(_dtype_name(getattr(parameter, "dtype", "unknown")))
        observed_loras = dict(self._runtime.observe_adapters(pipeline))
        if self._storage_mode is AnimeGenStorageMode.LAYERWISE_FP8:
            allowed = {"bfloat16", "float8_e4m3fn", "float8_e5m2"}
            fp8 = storage & {"float8_e4m3fn", "float8_e5m2"}
            casting_valid = len(casting_observations) == 2 and all(
                item == CastingObservation("float8_e4m3fn", "bfloat16")
                for item in casting_observations
            )
            if not casting_valid or compute != {"bfloat16"} or not fp8 or storage - allowed:
                raise PlacementPolicyError("observed compute/storage dtype does not match bf16/FP8")
            storage_dtype = sorted(fp8)[0]
        elif compute != {"bfloat16"} or storage != {"bfloat16"} or casting_observations:
            raise PlacementPolicyError("observed compute/storage dtype does not match bf16 storage")
        else:
            storage_dtype = "bfloat16"
        if observed_loras != {"lightning_high": 2.0, "lightning_low": 1.0}:
            raise PlacementPolicyError("active LoRA names or scales do not match")
        return {
            "quantization": self._storage_mode.value,
            "compute_dtype": "bfloat16",
            "storage_dtype": storage_dtype,
            "active_loras": observed_loras,
            "casting_calls": len(casting_observations),
        }


def _dtype_name(value: object) -> str:
    name = str(value).lower().removeprefix("torch.")
    return {"bf16": "bfloat16", "fp8": "float8_e4m3fn"}.get(name, name)
