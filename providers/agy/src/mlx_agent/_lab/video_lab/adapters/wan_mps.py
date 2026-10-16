from __future__ import annotations

import hashlib
import importlib
import math
import os
import platform
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from video_lab.adapter_protocol import (
    AdapterExecutionError,
    CancellationCheck,
    GenerationCancelled,
    GenerationProgress,
    GenerationRequest,
    GenerationResult,
    ProbePhase,
    ProbeReport,
    ProbeRequest,
    ProgressCallback,
)
from video_lab.capabilities import AdapterCapabilities, validate_manifest_support
from video_lab.domain import Manifest

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REQUESTED_REVISION = "main"
_MAX_DIAGNOSTIC_LENGTH = 1000


@dataclass(frozen=True)
class _Metadata:
    requested_revision: str
    resolved_revision: str
    file_sizes: Mapping[str, int]


class _Pipeline(Protocol):
    def to(self, device: str) -> _Pipeline: ...
    def __call__(self, **kwargs: object) -> object: ...


PipelineLoader = Callable[..., _Pipeline]
Exporter = Callable[[object, str, int], None]
DeviceInspector = Callable[[object], Mapping[str, str]]
VideoInspector = Callable[[Path], Mapping[str, object]]
FrameCounter = Callable[[Path, int], int]


def _load_pipeline(model_id: str, **kwargs: object) -> _Pipeline:
    diffusers = importlib.import_module("diffusers")
    return diffusers.WanPipeline.from_pretrained(model_id, **kwargs)  # type: ignore[no-any-return]


def _export_video(frames: object, path: str, fps: int) -> None:
    importlib.import_module("diffusers.utils").export_to_video(frames, path, fps=fps)


def _inspect_video(path: Path) -> Mapping[str, object]:
    return cast(Mapping[str, object], importlib.import_module("imageio.v3").immeta(path))


def _count_video_frames(path: Path, limit: int) -> int:
    count = 0
    for _frame in importlib.import_module("imageio.v3").imiter(path):
        count += 1
        if count >= limit:
            break
    return count


def _sanitize_diagnostic(value: object, sensitive_paths: tuple[Path, ...] = ()) -> str:
    text = str(value)
    for match in re.findall(r"https?://[^\s]+", text):
        parsed = urlsplit(match)
        host = parsed.hostname or "redacted-host"
        if parsed.port:
            host = f"{host}:{parsed.port}"
        clean = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        text = text.replace(match, clean)
    text = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer <redacted>", text)
    text = re.sub(r"\bhf_[A-Za-z0-9_-]+", "hf_<redacted>", text)
    homes = {str(Path.home())}
    for variable in ("HF_HOME", "HF_HUB_CACHE"):
        if os.environ.get(variable):
            homes.add(str(Path(os.environ[variable]).expanduser()))
    for home in sorted(homes, key=len, reverse=True):
        text = text.replace(home, "<home>")
    for path in sorted((str(item.expanduser()) for item in sensitive_paths), key=len, reverse=True):
        text = text.replace(path, "<cache>")
    return text[:_MAX_DIAGNOSTIC_LENGTH]


def _error_diagnostics(
    error: BaseException, sensitive_paths: tuple[Path, ...] = ()
) -> Mapping[str, object]:
    return {
        "error_type": type(error).__name__,
        "error": _sanitize_diagnostic(error, sensitive_paths),
    }


class WanMpsAdapter:
    adapter_id = "diffusers.wan2.1-t2v-1.3b.mps.v1"
    adapter_version = "1"

    def __init__(
        self,
        *,
        _torch: Any | None = None,
        _pipeline_loader: PipelineLoader = _load_pipeline,
        _exporter: Exporter = _export_video,
        _hf_api: object | None = None,
        _device_inspector: DeviceInspector | None = None,
        _cache_root: Path | None = None,
        _video_inspector: VideoInspector = _inspect_video,
        _frame_counter: FrameCounter = _count_video_frames,
    ) -> None:
        self._torch = _torch
        self._pipeline_loader = _pipeline_loader
        self._exporter = _exporter
        self._hf_api = _hf_api
        self._device_inspector = _device_inspector or self._inspect_devices
        self._cache_root = _cache_root
        self._video_inspector = _video_inspector
        self._frame_counter = _frame_counter
        self._metadata: _Metadata | None = None

    def _get_torch(self) -> Any:
        if self._torch is None:
            self._torch = importlib.import_module("torch")
        return self._torch

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_id=self.adapter_id,
            model_patterns=(MODEL_ID,),
            backends=("mps",),
            dtypes=("bfloat16",),
            quantizations=("none",),
            schedulers=("unipc",),
            min_width=16,
            max_width=4096,
            width_multiple=16,
            min_height=16,
            max_height=4096,
            height_multiple=16,
            min_frames=17,
            max_frames=129,
            frame_rule="16n+1",
            supports_lora=False,
        )

    def probe(self) -> bool:
        return self.probe_phase("environment").ok

    def probe_phase(
        self, phase: ProbePhase | str, request: ProbeRequest | None = None
    ) -> ProbeReport:
        try:
            if phase == "environment":
                torch = self._get_torch()
                diagnostics = {
                    "python": platform.python_version(),
                    "system": platform.system(),
                    "machine": platform.machine(),
                    "torch": str(torch.__version__),
                    "diffusers": self._optional_version("diffusers"),
                    "mps_available": bool(torch.backends.mps.is_available()),
                    "mps_built": bool(torch.backends.mps.is_built()),
                }
                ok = (diagnostics["system"], diagnostics["machine"]) == ("Darwin", "arm64")
                return ProbeReport(
                    phase,
                    bool(ok and diagnostics["mps_available"] and diagnostics["mps_built"]),
                    diagnostics,
                )
            if phase == "metadata":
                metadata = self._resolve_metadata()
                return ProbeReport(phase, True, self._metadata_diagnostics(metadata))
            if phase == "cache":
                metadata = self._resolve_metadata()
                root = self._effective_cache_root()
                snapshot = (
                    root
                    / "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
                    / "snapshots"
                    / metadata.resolved_revision
                )
                missing_files = tuple(
                    name for name in metadata.file_sizes if not (snapshot / name).is_file()
                )
                missing_bytes = sum(metadata.file_sizes[name] for name in missing_files)
                return ProbeReport(
                    phase,
                    not missing_files,
                    {
                        **self._metadata_diagnostics(metadata),
                        "cache_root": str(root),
                        "snapshot_path": str(snapshot),
                        "missing_files": missing_files,
                        "missing_bytes": missing_bytes,
                    },
                )
            if phase == "load":
                metadata = self._resolve_metadata()
                self._load(metadata.resolved_revision)
                return ProbeReport(
                    phase, True, {**self._metadata_diagnostics(metadata), "execution_device": "mps"}
                )
            if phase == "generate":
                if request is None:
                    raise RuntimeError("generate probe requires an explicit ProbeRequest")
                profile = request.generation.manifest
                if (profile.width, profile.height, profile.num_frames, profile.inference_steps) != (
                    832,
                    480,
                    17,
                    4,
                ):
                    raise RuntimeError(
                        "generate probe requires profile 832x480, 17 frames, 4 steps"
                    )
                result: GenerationResult | None = None
                try:
                    result = self.generate(request.generation, request.progress, request.cancelled)
                    manifest = request.generation.manifest
                    video = self._video_inspector(result.video_path)
                    size: tuple[object, ...] = tuple(cast(Any, video.get("size", ())))
                    frame_count = self._validated_frame_count(
                        result.video_path, video.get("nframes"), manifest.num_frames
                    )
                    fps = float(cast(Any, video.get("fps", 0)))
                    valid = (
                        result.video_path.is_file()
                        and result.video_path.stat().st_size > 0
                        and size == (manifest.width, manifest.height)
                        and frame_count == manifest.num_frames
                        and math.isfinite(fps)
                        and round(fps) == manifest.fps
                    )
                    if not valid:
                        raise RuntimeError("generate probe produced invalid MP4 metadata")
                    return ProbeReport(
                        phase,
                        True,
                        {
                            "video_path": str(result.video_path),
                            "width": result.width,
                            "height": result.height,
                            "frame_count": result.frame_count,
                            "fps": result.fps,
                            "container_metadata": dict(video),
                        },
                    )
                except Exception:
                    scratch = request.generation.scratch_root
                    self._remove(scratch / "output.partial.mp4", scratch / "output.mp4")
                    if result is not None:
                        self._remove(result.video_path)
                    raise
            raise ValueError(f"unknown probe phase: {phase}")
        except Exception as error:
            paths = (self._cache_root,) if self._cache_root is not None else ()
            return ProbeReport(phase, False, _error_diagnostics(error, paths))

    def _validated_frame_count(self, path: Path, reported: object, expected: int) -> int:
        if isinstance(reported, int | float) and not isinstance(reported, bool):
            numeric = float(reported)
            if math.isfinite(numeric) and numeric > 0 and numeric.is_integer():
                return int(numeric)
        return self._frame_counter(path, expected + 1)

    def _resolve_metadata(self, revision: str = REQUESTED_REVISION) -> _Metadata:
        if self._metadata is not None and self._metadata.requested_revision == revision:
            return self._metadata
        api = self._hf_api
        if api is None:
            api = importlib.import_module("huggingface_hub").HfApi()
        info = cast(Any, api).model_info(MODEL_ID, revision=revision, files_metadata=True)
        resolved = str(info.sha or "").strip()
        if revision != REQUESTED_REVISION and resolved != revision:
            raise AdapterExecutionError("Pinned model revision did not resolve exactly.")
        if not resolved:
            raise RuntimeError("Hub metadata returned an empty resolved revision")
        sizes = {str(item.rfilename): int(item.size or 0) for item in list(info.siblings or [])}
        if not sizes:
            raise RuntimeError("Hub metadata returned no model files")
        self._metadata = _Metadata(revision, resolved, sizes)
        return self._metadata

    @staticmethod
    def _metadata_diagnostics(metadata: _Metadata) -> Mapping[str, object]:
        return {
            "requested_revision": metadata.requested_revision,
            "resolved_revision": metadata.resolved_revision,
            "file_sizes": dict(metadata.file_sizes),
            "download_bytes": sum(metadata.file_sizes.values()),
        }

    def _effective_cache_root(self) -> Path:
        if self._cache_root is not None:
            return self._cache_root.expanduser()
        if os.environ.get("HF_HUB_CACHE"):
            return Path(os.environ["HF_HUB_CACHE"]).expanduser()
        if os.environ.get("HF_HOME"):
            return Path(os.environ["HF_HOME"]).expanduser() / "hub"
        try:
            return Path(importlib.import_module("huggingface_hub.constants").HF_HUB_CACHE)
        except ImportError:
            return Path.home() / ".cache" / "huggingface" / "hub"

    @staticmethod
    def _optional_version(module: str) -> str:
        try:
            return str(importlib.import_module(module).__version__)
        except (ImportError, AttributeError):
            return "not-installed"

    def _load(self, resolved_revision: str) -> _Pipeline:
        torch = self._get_torch()
        if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
            raise AdapterExecutionError("MPS is unavailable; CPU fallback is forbidden")
        pipeline = self._pipeline_loader(
            MODEL_ID,
            revision=resolved_revision,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=False,
        )
        pipeline.to("mps")
        self._assert_execution_device(pipeline)
        return pipeline

    @staticmethod
    def _inspect_devices(pipeline: object) -> Mapping[str, str]:
        devices: dict[str, str] = {}
        execution_device = getattr(pipeline, "device", getattr(pipeline, "_execution_device", None))
        if execution_device is not None:
            devices["pipeline.device"] = str(execution_device)
        for collection in ("parameters", "buffers"):
            getter = getattr(pipeline, collection, None)
            if callable(getter):
                for index, item in enumerate(getter()):
                    devices[f"pipeline.{collection}[{index}]"] = str(
                        getattr(item, "device", "unknown")
                    )
        components = getattr(pipeline, "components", {})
        for name, component in components.items():
            if component is None:
                devices[name] = "non-executing"
                continue
            found = False
            explicit = getattr(component, "device", None)
            if explicit is not None:
                devices[f"{name}.device"] = str(explicit)
                found = True
            for collection in ("parameters", "buffers"):
                getter = getattr(component, collection, None)
                if callable(getter):
                    for index, item in enumerate(getter()):
                        devices[f"{name}.{collection}[{index}]"] = str(
                            getattr(item, "device", "unknown")
                        )
                        found = True
            if not found:
                kind = type(component).__name__.lower()
                devices[name] = (
                    "non-executing"
                    if any(label in kind for label in ("tokenizer", "scheduler", "processor"))
                    else "uninspectable"
                )
        if not devices:
            devices["pipeline"] = "uninspectable"
        return devices

    def _assert_execution_device(self, pipeline: _Pipeline) -> None:
        for name, device in self._device_inspector(pipeline).items():
            if device == "non-executing":
                continue
            if not device.startswith("mps"):
                raise AdapterExecutionError(
                    f"execution-device fallback detected in {name}: {device}"
                )

    def _verify_pinned_weights(self, manifest: Manifest, metadata: _Metadata) -> None:
        if manifest.model_revision == REQUESTED_REVISION:
            return
        root = (
            self._effective_cache_root()
            / ("models--" + MODEL_ID.replace("/", "--"))
            / "snapshots"
            / metadata.resolved_revision
        )
        digest = hashlib.sha256()
        files = sorted(root.rglob("*.safetensors"))
        if not files:
            raise AdapterExecutionError("Pinned model weights are unavailable.")
        for path in files:
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            digest.update(str(path.relative_to(root)).encode() + b"\0" + checksum.encode())
        if "sha256:" + digest.hexdigest() != manifest.model_checksum:
            raise AdapterExecutionError("Pinned model weight content changed.")

    def generate(
        self, request: GenerationRequest, progress: ProgressCallback, cancelled: CancellationCheck
    ) -> GenerationResult:
        violations = validate_manifest_support(request.manifest, self.capabilities())
        if (
            request.manifest.model_revision != REQUESTED_REVISION
            and re.fullmatch(r"[0-9a-f]{40}", request.manifest.model_revision) is None
        ):
            raise AdapterExecutionError(
                "unsupported manifest: model_revision must be a pinned revision "
                f"or {REQUESTED_REVISION}"
            )
        if violations:
            detail = ", ".join(f"{item.field}: {item.reason}" for item in violations)
            raise AdapterExecutionError(f"unsupported manifest: {detail}")
        scratch = request.scratch_root
        partial = scratch / "output.partial.mp4"
        final = scratch / "output.mp4"
        scratch.mkdir(parents=True, exist_ok=True)
        self._remove(partial, final)
        started = monotonic()
        try:
            if cancelled():
                raise GenerationCancelled("generation cancelled")
            manifest = request.manifest
            if "study-recipes" in manifest.runtime_versions:
                from mlx_agent.study_recipes import verify
                sizes = verify(manifest, self._effective_cache_root())
                metadata = _Metadata(manifest.model_revision, manifest.model_revision, sizes)
            else:
                metadata = self._resolve_metadata(manifest.model_revision)
                self._verify_pinned_weights(manifest, metadata)
            pipeline = self._load(metadata.resolved_revision)
            generator = self._get_torch().Generator(device="cpu").manual_seed(manifest.seed)

            def on_step_end(_pipe: object, step: int, _time: object, kwargs: object) -> object:
                if cancelled():
                    raise GenerationCancelled("generation cancelled")
                progress(
                    GenerationProgress(
                        phase="generating",
                        completed=step + 1,
                        total=manifest.inference_steps,
                        message=f"completed diffusion step {step + 1}",
                        elapsed_seconds=monotonic() - started,
                    )
                )
                self._assert_execution_device(pipeline)
                return kwargs

            output = pipeline(
                prompt=manifest.prompt,
                negative_prompt=manifest.negative_prompt,
                height=manifest.height,
                width=manifest.width,
                num_frames=manifest.num_frames,
                num_inference_steps=manifest.inference_steps,
                guidance_scale=manifest.guidance_scale,
                generator=generator,
                callback_on_step_end=on_step_end,
            )
            self._assert_execution_device(pipeline)
            frames = getattr(output, "frames", None)
            if frames is None or len(frames) == 0 or frames[0] is None or len(frames[0]) == 0:
                raise AdapterExecutionError("pipeline returned invalid video frames")
            self._exporter(frames[0], str(partial), manifest.fps)
            if not partial.is_file() or partial.stat().st_size == 0:
                raise AdapterExecutionError("video exporter produced an invalid artifact")
            with partial.open("rb") as artifact:
                os.fsync(artifact.fileno())
            if cancelled():
                raise GenerationCancelled("generation cancelled")
            partial.replace(final)
            if "study-recipes" in manifest.runtime_versions:
                from mlx_agent.study_recipes import verify
                verify(manifest, self._effective_cache_root())
            else:
                self._verify_pinned_weights(manifest, metadata)
            return GenerationResult(
                video_path=final,
                effective_manifest=manifest,
                frame_count=manifest.num_frames,
                fps=manifest.fps,
                width=manifest.width,
                height=manifest.height,
                adapter_version=self.adapter_version,
                runtime_versions={
                    "python": platform.python_version(),
                    "torch": str(self._get_torch().__version__),
                    "diffusers": self._optional_version("diffusers"),
                },
                adapter_telemetry={
                    "execution_device": "mps",
                    "rng_device": "cpu",
                    "requested_revision": metadata.requested_revision,
                    "resolved_revision": metadata.resolved_revision,
                },
            )
        except GenerationCancelled:
            self._remove(partial, final)
            raise
        except Exception as error:
            self._remove(partial, final)
            paths = (self._cache_root,) if self._cache_root is not None else ()
            sanitized = _sanitize_diagnostic(error, paths)
            raise AdapterExecutionError(
                f"Wan MPS generation failed: {type(error).__name__}: {sanitized}"
            ) from error

    @staticmethod
    def _remove(*paths: Path) -> None:
        for path in paths:
            path.unlink(missing_ok=True)
