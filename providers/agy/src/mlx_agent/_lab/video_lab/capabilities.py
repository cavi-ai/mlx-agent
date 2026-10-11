from fnmatch import fnmatch
from typing import Literal

from pydantic import Field

from video_lab.domain import FrozenModel, Manifest


class AdapterCapabilities(FrozenModel):
    adapter_id: str
    model_patterns: tuple[str, ...]
    backends: tuple[str, ...]
    dtypes: tuple[str, ...]
    quantizations: tuple[str, ...]
    schedulers: tuple[str, ...]
    min_width: int = Field(gt=0)
    max_width: int = Field(gt=0)
    width_multiple: int = Field(gt=0)
    min_height: int = Field(gt=0)
    max_height: int = Field(gt=0)
    height_multiple: int = Field(gt=0)
    min_frames: int = Field(gt=0)
    max_frames: int = Field(gt=0)
    frame_rule: Literal["any", "16n+1"]
    supports_lora: bool
    min_fps: int = Field(default=1, gt=0)
    max_fps: int = Field(default=120, gt=0)


class CapabilityViolation(FrozenModel):
    field: str
    value: object
    reason: str


def validate_manifest_support(
    manifest: Manifest, capabilities: AdapterCapabilities
) -> tuple[CapabilityViolation, ...]:
    violations: list[CapabilityViolation] = []

    def reject(field: str, value: object, reason: str) -> None:
        violations.append(CapabilityViolation(field=field, value=value, reason=reason))

    if manifest.adapter_id != capabilities.adapter_id:
        reject("adapter_id", manifest.adapter_id, f"expected adapter {capabilities.adapter_id}")
    if not any(fnmatch(manifest.model_id, pattern) for pattern in capabilities.model_patterns):
        reject("model_id", manifest.model_id, "model is not supported by this adapter")
    for field, setting, allowed in (
        ("backend", manifest.backend, capabilities.backends),
        ("dtype", manifest.dtype, capabilities.dtypes),
        ("quantization", manifest.quantization, capabilities.quantizations),
        ("scheduler", manifest.scheduler, capabilities.schedulers),
    ):
        if setting not in allowed:
            reject(field, setting, f"expected one of {allowed}")
    for axis, dimension, low, high, multiple in (
        (
            "width",
            manifest.width,
            capabilities.min_width,
            capabilities.max_width,
            capabilities.width_multiple,
        ),
        (
            "height",
            manifest.height,
            capabilities.min_height,
            capabilities.max_height,
            capabilities.height_multiple,
        ),
    ):
        if not low <= dimension <= high or dimension % multiple:
            reject(axis, dimension, f"expected {low}..{high} in multiples of {multiple}")
    if not capabilities.min_frames <= manifest.num_frames <= capabilities.max_frames:
        reject("num_frames", manifest.num_frames, "frame count outside adapter range")
    if capabilities.frame_rule == "16n+1" and (manifest.num_frames - 1) % 16:
        reject("num_frames", manifest.num_frames, "frame count must equal 16n+1")
    if manifest.loras and not capabilities.supports_lora:
        reject("loras", manifest.loras, "adapter does not support LoRA bindings")
    if not capabilities.min_fps <= manifest.fps <= capabilities.max_fps:
        reject("fps", manifest.fps, f"expected {capabilities.min_fps}..{capabilities.max_fps}")
    return tuple(violations)
