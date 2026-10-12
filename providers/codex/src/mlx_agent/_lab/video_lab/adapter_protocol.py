import json
import math
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from pydantic import Field, field_serializer, field_validator

from video_lab.capabilities import AdapterCapabilities
from video_lab.domain import FrozenModel, Manifest


class GenerationRequest(FrozenModel):
    job_id: str
    attempt_id: str
    manifest: Manifest
    scratch_root: Path


class GenerationProgress(FrozenModel):
    phase: str
    completed: int = Field(ge=0)
    total: int = Field(gt=0)
    message: str
    elapsed_seconds: float = Field(ge=0)


class GenerationResult(FrozenModel):
    video_path: Path
    effective_manifest: Manifest
    frame_count: int = Field(gt=0)
    fps: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    adapter_version: str
    runtime_versions: Mapping[str, str]
    adapter_telemetry: Mapping[str, object]

    @field_validator("runtime_versions")
    @classmethod
    def freeze_runtime_versions(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_validator("adapter_telemetry")
    @classmethod
    def freeze_adapter_telemetry(cls, value: object) -> Mapping[str, object]:
        frozen = _freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise TypeError("adapter telemetry must be a mapping")
        return frozen

    @field_serializer("runtime_versions")
    def serialize_runtime_versions(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @field_serializer("adapter_telemetry")
    def serialize_adapter_telemetry(self, value: Mapping[str, object]) -> dict[str, object]:
        serialized = _serialize_json(value)
        if not isinstance(serialized, dict):
            raise TypeError("adapter telemetry must serialize to a mapping")
        return serialized


def _freeze_json(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("unsupported non-finite JSON number")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("adapter telemetry mapping keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, Set):
        frozen = tuple(_freeze_json(item) for item in value)
        return tuple(sorted(frozen, key=_json_order))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"unsupported adapter telemetry value: {type(value).__name__}")


def _serialize_json(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("unsupported non-finite JSON number")
    if isinstance(value, Mapping):
        return {str(key): _serialize_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize_json(item) for item in value]
    return value


def _json_order(value: object) -> str:
    return json.dumps(_serialize_json(value), sort_keys=True)


class GenerationCancelled(Exception):
    """Raised when generation is cancelled before artifact finalization."""


class AdapterExecutionError(Exception):
    """Raised when an adapter cannot complete generation."""


ProgressCallback = Callable[[GenerationProgress], None]
CancellationCheck = Callable[[], bool]


class ProbePhase(StrEnum):
    ENVIRONMENT = "environment"
    METADATA = "metadata"
    CACHE = "cache"
    LOAD = "load"
    GENERATE = "generate"


@dataclass(frozen=True)
class ProbeReport:
    phase: ProbePhase | str
    ok: bool
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class ProbeRequest:
    generation: GenerationRequest
    progress: ProgressCallback
    cancelled: CancellationCheck


@runtime_checkable
class GenerationAdapter(Protocol):
    adapter_id: str

    def capabilities(self) -> AdapterCapabilities: ...

    def probe(self) -> bool: ...

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback,
        cancelled: CancellationCheck,
    ) -> GenerationResult: ...


@runtime_checkable
class StagedProbeAdapter(GenerationAdapter, Protocol):
    def probe_phase(
        self, phase: ProbePhase, request: ProbeRequest | None = None
    ) -> ProbeReport: ...
