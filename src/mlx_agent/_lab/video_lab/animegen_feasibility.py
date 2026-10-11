from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from video_lab.adapter_protocol import AdapterExecutionError
from video_lab.adapters.animegen_mps import ADAPTER_ID as ANIMEGEN_ADAPTER_ID
from video_lab.adapters.animegen_mps import (
    AnimeGenStageExecutor,
    AnimeGenStorageMode,
    storage_mode_for_manifest,
)
from video_lab.domain import Manifest
from video_lab.feasibility import (
    InvalidEvidenceError,
    StageDefinition,
    StageExecutionContext,
    StageResult,
)
from video_lab.feasibility_models import (
    FeasibilityConfiguration,
    FeasibilityStage,
    StageEvidence,
    StageOutcome,
)
from video_lab.runtime_identity import animegen_runtime_versions
from video_lab.source_inventory import (
    ANIMEGEN_REQUIRED_FILES,
    ANIMEGEN_SOURCE_REPOS,
    SourceInventory,
    SourceInventoryClient,
    SourceInventoryError,
    SourceSpec,
    canonical_source_digest,
    inventory_from_evidence,
)

ANIMEGEN_BF16_PROTOCOL_VERSION = "animegen-feasibility.v2"
ANIMEGEN_FROZEN_PROMPT = "anime character walking"
ANIMEGEN_FROZEN_NEGATIVE_PROMPT = ""
_GIB = 1024**3
_MEMORY_POLICY = "max(24GiB,20pct_total)"


def canonical_animegen_configuration(configuration: FeasibilityConfiguration) -> dict[str, Any]:
    if configuration.adapter_id != ANIMEGEN_ADAPTER_ID:
        raise ValueError("configuration adapter_id is not the AnimeGen candidate")
    if configuration.protocol_version != ANIMEGEN_BF16_PROTOCOL_VERSION:
        raise ValueError("configuration protocol_version is not the AnimeGen BF16 protocol")
    if set(configuration.sources) != set(ANIMEGEN_SOURCE_REPOS):
        raise ValueError("configuration must contain the canonical AnimeGen sources")
    for component, repo_id in ANIMEGEN_SOURCE_REPOS.items():
        if configuration.sources[component].id != repo_id:
            raise ValueError(f"source repository mismatch: {component}")
    profiles = configuration.manifest_profiles.model_dump(mode="json")
    minimum = Manifest.model_validate(profiles["minimum"])
    standard = Manifest.model_validate(profiles["standard"])
    for name, manifest in (("minimum", minimum), ("standard", standard)):
        animegen = configuration.sources["animegen"]
        if (
            manifest.adapter_id != configuration.adapter_id
            or manifest.adapter_version != "1"
            or manifest.model_id != animegen.id
            or manifest.model_revision != animegen.revision
            or manifest.model_checksum != animegen.checksum.removeprefix("sha256:")
            or dict(manifest.runtime_versions) != dict(configuration.runtime_versions)
            or manifest.hardware_fingerprint != configuration.hardware_fingerprint
            or manifest.os_fingerprint != configuration.os_fingerprint
            or manifest.memory_strategy != configuration.memory_strategy
            or manifest.scheduler != "flow-match-euler-shift-3"
            or manifest.dtype != "bfloat16"
            or manifest.backend != "mps"
            or manifest.device != "mps"
            or manifest.num_frames < 17
            or (manifest.num_frames - 1) % 16
            or manifest.width % 16
            or manifest.height % 16
        ):
            raise ValueError(f"{name} manifest does not match canonical AnimeGen configuration")
        try:
            storage_mode = storage_mode_for_manifest(manifest)
        except AdapterExecutionError as error:
            raise ValueError(f"{name} manifest uses unsupported transformer storage") from error
        if storage_mode is not AnimeGenStorageMode.BF16:
            raise ValueError(f"{name} manifest does not match canonical AnimeGen configuration")
        expected_loras = tuple(
            (
                configuration.sources[component].id,
                configuration.sources[component].revision,
                configuration.sources[component].checksum.removeprefix("sha256:"),
                weight,
            )
            for component, weight in (("lightning-high", 2.0), ("lightning-low", 1.0))
        )
        actual_loras = tuple(
            (item.model_id, item.revision, item.checksum, item.weight) for item in manifest.loras
        )
        if actual_loras != expected_loras:
            raise ValueError(f"{name} manifest LoRA bindings are not canonical")
        expected_checksums = {
            component: identity.checksum.removeprefix("sha256:")
            for component, identity in configuration.sources.items()
        }
        if dict(manifest.source_checksums) != expected_checksums:
            raise ValueError(f"{name} manifest source checksums are not canonical")
    frozen_controls = {
        "num_frames": 17,
        "fps": 8,
        "inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": 7,
        "prompt": ANIMEGEN_FROZEN_PROMPT,
        "negative_prompt": ANIMEGEN_FROZEN_NEGATIVE_PROMPT,
    }
    for name, manifest, dimensions in (
        ("minimum", minimum, (128, 80)),
        ("standard", standard, (512, 288)),
    ):
        if (manifest.width, manifest.height) != dimensions or any(
            getattr(manifest, field) != value for field, value in frozen_controls.items()
        ):
            raise ValueError(f"{name} manifest does not match the frozen AnimeGen BF16 recipe")
    minimum_identity = minimum.model_dump(mode="json")
    standard_identity = standard.model_dump(mode="json")
    for field in ("width", "height"):
        minimum_identity.pop(field)
        standard_identity.pop(field)
    if minimum_identity != standard_identity:
        raise ValueError("minimum and standard manifests may differ only in dimensions")
    payload = configuration.model_dump(mode="json")
    payload["manifest_profiles"] = {
        "minimum": minimum.model_dump(mode="json"),
        "standard": standard.model_dump(mode="json"),
    }
    return payload


def evaluate_preassembly_preflight(
    *,
    configuration: Mapping[str, object],
    source_digests: Mapping[str, str],
    storage: Mapping[str, int],
    observation: Mapping[str, object],
    stage: FeasibilityStage,
) -> StageResult:
    """Compare one complete pre-assembly observation with the persisted run identity."""
    configured_sources = configuration.get("sources")
    if not isinstance(configured_sources, Mapping):
        raise InvalidEvidenceError("persisted preflight source identity is missing")
    expected_sources = {
        str(component): str(identity.get("checksum", "")).removeprefix("sha256:")
        for component, identity in configured_sources.items()
        if isinstance(identity, Mapping)
    }
    expected_runtime = configuration.get("runtime_versions")
    observed_runtime = observation.get("runtime_versions")
    mps = observation.get("mps")
    memory = observation.get("unified_memory")
    if (
        not isinstance(expected_runtime, Mapping)
        or not isinstance(observed_runtime, Mapping)
        or not isinstance(mps, Mapping)
        or not isinstance(memory, Mapping)
    ):
        raise InvalidEvidenceError("preflight observation is incomplete")
    requested = {
        "runtime_versions": dict(expected_runtime),
        "source_digests": expected_sources,
        "hardware_fingerprint": configuration.get("hardware_fingerprint"),
        "os_fingerprint": configuration.get("os_fingerprint"),
        "mps": {"available": True, "built": True},
        "disk": {"required_bytes": storage.get("required_bytes")},
        "unified_memory": {
            "required_available_bytes": memory.get("required_available_bytes"),
            "policy": memory.get("policy"),
        },
    }
    observed = {
        "runtime_versions": dict(observed_runtime),
        "source_digests": dict(source_digests),
        "hardware_fingerprint": observation.get("hardware_fingerprint"),
        "os_fingerprint": observation.get("os_fingerprint"),
        "mps": dict(mps),
        "disk": dict(storage),
        "unified_memory": dict(memory),
    }
    mismatches = tuple(
        name
        for name, expected, actual in (
            ("runtime", requested["runtime_versions"], observed["runtime_versions"]),
            ("source", requested["source_digests"], observed["source_digests"]),
            (
                "hardware",
                requested["hardware_fingerprint"],
                observed["hardware_fingerprint"],
            ),
            ("os", requested["os_fingerprint"], observed["os_fingerprint"]),
        )
        if expected != actual
    )
    details: dict[str, object] = {
        "stage": stage.value,
        "requested": requested,
        "observed": observed,
        "identity_mismatches": mismatches,
    }
    evidence = StageEvidence(details=cast(Any, {"preassembly_preflight": details}))
    if mismatches:
        return StageResult(
            outcome=StageOutcome.INVALID_EVIDENCE,
            evidence=evidence,
            diagnostic_code="preflight_identity_drift",
            diagnostic_message="persisted and observed preflight identity differ",
        )
    if mps.get("available") is not True or mps.get("built") is not True:
        return StageResult(
            outcome=StageOutcome.INCOMPATIBLE,
            evidence=evidence,
            diagnostic_code="mps_unavailable",
            diagnostic_message="MPS is unavailable or not built",
        )
    required_disk = storage.get("required_bytes")
    free_disk = storage.get("free_bytes")
    if (
        not isinstance(required_disk, int)
        or not isinstance(free_disk, int)
        or free_disk < required_disk
    ):
        return StageResult(
            outcome=StageOutcome.RESOURCE_EXHAUSTED,
            evidence=evidence,
            diagnostic_code="insufficient_disk_headroom",
            diagnostic_message="insufficient free disk headroom",
        )
    required_memory = memory.get("required_available_bytes")
    available_memory = memory.get("available_bytes")
    if (
        not isinstance(required_memory, int)
        or not isinstance(available_memory, int)
        or available_memory < required_memory
    ):
        return StageResult(
            outcome=StageOutcome.RESOURCE_EXHAUSTED,
            evidence=evidence,
            diagnostic_code="insufficient_unified_memory_headroom",
            diagnostic_message="insufficient unified-memory headroom",
        )
    return StageResult.passed(evidence)


def production_system_observation() -> Mapping[str, object]:
    """Observe only privacy-safe fields needed by the local M5 qualification identity."""
    import psutil  # type: ignore[import-untyped]
    import torch

    hardware = json.loads(
        subprocess.run(
            ["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["SPHardwareDataType"][0]
    chip = re.sub(r"[^a-z0-9]+", "-", str(hardware["chip_type"]).lower()).strip("-")
    memory_label = re.sub(r"\s+", "", str(hardware["physical_memory"]).lower())
    cores = re.search(r"\d+", str(hardware["number_processors"]))
    if cores is None:
        raise RuntimeError("hardware core count is unavailable")
    hardware_fingerprint = f"{chip}-{memory_label}-{cores.group()}c"
    product_version = subprocess.run(
        ["/usr/bin/sw_vers", "-productVersion"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    build_version = subprocess.run(
        ["/usr/bin/sw_vers", "-buildVersion"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    os_fingerprint = f"macos-{product_version}-{build_version}-{platform.machine()}"
    virtual_memory = psutil.virtual_memory()
    required_available = max(24 * _GIB, (int(virtual_memory.total) + 4) // 5)
    return {
        "runtime_versions": animegen_runtime_versions(),
        "hardware_fingerprint": hardware_fingerprint,
        "os_fingerprint": os_fingerprint,
        "mps": {
            "available": bool(torch.backends.mps.is_available()),
            "built": bool(torch.backends.mps.is_built()),
        },
        "unified_memory": {
            "total_bytes": int(virtual_memory.total),
            "available_bytes": int(virtual_memory.available),
            "required_available_bytes": required_available,
            "policy": _MEMORY_POLICY,
        },
    }


class AnimeGenFeasibilityExecutor:
    """Executes source and isolated component stages from persisted evidence."""

    def __init__(
        self,
        cache_root: Path | None,
        inventory_client: SourceInventoryClient | None,
        component_factory: Callable[..., AnimeGenStageExecutor] | None = None,
        component_options: Mapping[str, object] | None = None,
        inventory_evidence: Mapping[str, object] | None = None,
        system_observer: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.inventory_client = inventory_client
        self.component_factory = component_factory or AnimeGenStageExecutor
        self.component_options = dict(component_options or {})
        self.inventory_evidence = inventory_evidence
        self.system_observer = system_observer or production_system_observation

    def execute(self, context: StageExecutionContext, stage: StageDefinition) -> StageResult:
        current = stage.stage
        if current is FeasibilityStage.RESOLVE_SOURCES:
            return StageResult.passed(
                StageEvidence(details=cast(Any, {"sources": self._source_specs(context)}))
            )
        if current is FeasibilityStage.INVENTORY_SOURCES:
            client = self._client()
            inventory = (
                client.resolve(self._specs(context))
                if self.inventory_evidence is None
                else self._validated_imported_inventory(context, client)
            )
            return StageResult.passed(
                StageEvidence(details=cast(Any, {"inventory": inventory.export_evidence()}))
            )
        if current is FeasibilityStage.STORAGE_PREFLIGHT:
            return self.preflight(context, stage)
        if current is FeasibilityStage.DOWNLOAD_SOURCES:
            inventory = self._inventory(context, FeasibilityStage.INVENTORY_SOURCES)
            downloaded = (
                inventory.download()
                if self.inventory_evidence is None
                else self._client().verify_cached(inventory)
            )
            digests = {
                source.component: canonical_source_digest(source) for source in downloaded.sources
            }
            return StageResult.passed(
                StageEvidence(
                    details=cast(
                        Any,
                        {
                            "inventory": downloaded.export_evidence(),
                            "canonical_source_digests": digests,
                        },
                    )
                )
            )
        if current in {
            FeasibilityStage.LOAD_TEXT_STACK,
            FeasibilityStage.LOAD_VAE,
            FeasibilityStage.LOAD_HIGH_TRANSFORMER,
            FeasibilityStage.LOAD_LOW_TRANSFORMER,
            FeasibilityStage.ASSEMBLE_PIPELINE,
        }:
            inventory = self.validated_download_inventory(context)
            sources = {source.component: source for source in inventory.sources}
            executor = self.component_factory(
                cache_root=self._cache(),
                sources=sources,
                runtime=cast(Any, self.component_options.get("runtime")),
                torch=self.component_options.get("torch"),
                storage_mode=self._storage_mode(context),
            )
            return StageResult.passed(
                executor.execute_protocol_stage(current, cancelled=lambda: False)
            )
        if current is FeasibilityStage.PUBLISH_VERDICT:
            return StageResult.passed(StageEvidence(details={"published": True}))
        raise SourceInventoryError(f"stage requires generation composition: {current.value}")

    def preflight(self, context: StageExecutionContext, stage: StageDefinition) -> StageResult:
        source_stage = (
            FeasibilityStage.INVENTORY_SOURCES
            if stage.stage
            in {FeasibilityStage.STORAGE_PREFLIGHT, FeasibilityStage.DOWNLOAD_SOURCES}
            else FeasibilityStage.DOWNLOAD_SOURCES
        )
        inventory = self._inventory(context, source_stage)
        if source_stage is FeasibilityStage.DOWNLOAD_SOURCES:
            inventory = self._client().verify_cached(inventory)
        source_digests = {
            source.component: canonical_source_digest(source) for source in inventory.sources
        }
        cache_root = self._cache()
        cache_root.mkdir(parents=True, exist_ok=True)
        free_bytes = int(shutil.disk_usage(cache_root).free)
        safety_margin = max(20 * _GIB, (inventory.expected_bytes + 4) // 5)
        storage = {
            "expected_bytes": inventory.expected_bytes,
            "safety_margin_bytes": safety_margin,
            "required_bytes": inventory.expected_bytes + safety_margin,
            "free_bytes": free_bytes,
        }
        try:
            observation = self.system_observer()
        except Exception as error:
            return StageResult(
                outcome=StageOutcome.INVALID_EVIDENCE,
                evidence=StageEvidence(
                    details={
                        "preassembly_preflight": {
                            "stage": stage.stage.value,
                            "observation_error": type(error).__name__,
                        }
                    }
                ),
                diagnostic_code="preflight_observation_failed",
                diagnostic_message="complete preflight observation failed",
            )
        return evaluate_preassembly_preflight(
            configuration=cast(Mapping[str, object], context.run.configuration),
            source_digests=source_digests,
            storage=storage,
            observation=observation,
            stage=stage.stage,
        )

    @staticmethod
    def _storage_mode(context: StageExecutionContext) -> AnimeGenStorageMode:
        profiles = context.run.configuration.get("manifest_profiles")
        if not isinstance(profiles, Mapping):
            raise SourceInventoryError("persisted manifest profiles are missing")
        modes = {
            storage_mode_for_manifest(Manifest.model_validate(profiles.get(name)))
            for name in ("minimum", "standard")
        }
        if len(modes) != 1:
            raise SourceInventoryError("manifest profiles use inconsistent transformer storage")
        return modes.pop()

    def _client(self) -> SourceInventoryClient:
        if self.inventory_client is None:
            raise SourceInventoryError("explicit cache is required for source operation")
        return self.inventory_client

    def _cache(self) -> Path:
        if self.cache_root is None:
            raise SourceInventoryError("explicit cache is required for local source operation")
        return self.cache_root

    def _source_specs(self, context: StageExecutionContext) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "component": spec.component,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "required_files": spec.required_files,
            }
            for spec in self._specs(context)
        )

    def _specs(self, context: StageExecutionContext) -> tuple[SourceSpec, ...]:
        configured = context.run.configuration.get("sources")
        if not isinstance(configured, Mapping):
            raise SourceInventoryError("persisted source identities are missing")
        specs: list[SourceSpec] = []
        for component, repo_id in ANIMEGEN_SOURCE_REPOS.items():
            identity = configured.get(component)
            if not isinstance(identity, Mapping) or identity.get("id") != repo_id:
                raise SourceInventoryError(f"persisted source identity mismatch: {component}")
            specs.append(
                SourceSpec(
                    component,
                    repo_id,
                    str(identity.get("revision", "")),
                    ANIMEGEN_REQUIRED_FILES[component],
                )
            )
        return tuple(specs)

    def _inventory(
        self, context: StageExecutionContext, source_stage: FeasibilityStage
    ) -> SourceInventory:
        attempt = next(
            (
                item
                for item in reversed(context.attempts)
                if item.stage is source_stage and item.evidence is not None
            ),
            None,
        )
        if attempt is None or attempt.evidence is None:
            raise SourceInventoryError(f"persisted {source_stage.value} evidence is missing")
        evidence = attempt.evidence.details.get("inventory")
        if not isinstance(evidence, Mapping):
            raise SourceInventoryError(f"persisted {source_stage.value} inventory is missing")
        return inventory_from_evidence(evidence, self._client())

    def _validated_imported_inventory(
        self, context: StageExecutionContext, client: SourceInventoryClient
    ) -> SourceInventory:
        assert self.inventory_evidence is not None
        specs = self._specs(context)
        inventory = client.refresh_cached(
            inventory_from_evidence(self.inventory_evidence, client), specs
        )
        if len(inventory.sources) != len(specs):
            raise SourceInventoryError("imported inventory source set does not match configuration")
        for source, spec in zip(inventory.sources, specs, strict=True):
            if (
                source.component != spec.component
                or source.repo_id != spec.repo_id
                or source.requested_revision != spec.revision
                or source.revision != spec.revision
                or tuple(item.relative_path for item in source.files) != spec.required_files
                or any(
                    item.upstream_sha256 is None and item.local_sha256 is None
                    for item in source.files
                )
            ):
                raise SourceInventoryError(
                    f"imported inventory does not match configured source: {spec.component}"
                )
        return inventory

    def validated_download_inventory(self, context: StageExecutionContext) -> SourceInventory:
        inventory = self._inventory(context, FeasibilityStage.DOWNLOAD_SOURCES)
        digests = {
            source.component: canonical_source_digest(source) for source in inventory.sources
        }
        configured = context.run.configuration.get("sources")
        profiles = context.run.configuration.get("manifest_profiles")
        if not isinstance(configured, Mapping) or not isinstance(profiles, Mapping):
            raise SourceInventoryError("persisted checksum configuration is missing")
        expected: dict[str, str] = {}
        for component in ANIMEGEN_SOURCE_REPOS:
            identity = configured.get(component)
            if not isinstance(identity, Mapping):
                raise SourceInventoryError(f"persisted checksum is missing: {component}")
            expected[component] = str(identity.get("checksum", "")).removeprefix("sha256:")
        if digests != expected:
            raise InvalidEvidenceError("downloaded canonical source checksum mismatch")
        for name in ("minimum", "standard"):
            manifest = Manifest.model_validate(profiles.get(name))
            if dict(manifest.source_checksums) != digests:
                raise InvalidEvidenceError(f"{name} manifest source checksum mismatch")
            if manifest.model_checksum != digests["animegen"]:
                raise InvalidEvidenceError(f"{name} manifest model checksum mismatch")
            expected_loras = tuple(
                (ANIMEGEN_SOURCE_REPOS[component], digests[component])
                for component in ("lightning-high", "lightning-low")
            )
            actual_loras = tuple((item.model_id, item.checksum) for item in manifest.loras)
            if actual_loras != expected_loras:
                raise InvalidEvidenceError(f"{name} manifest LoRA checksum mismatch")
        return inventory


def production_inventory_client(cache_root: Path) -> SourceInventoryClient:
    """Build Hub seams lazily; snapshot acquisition remains download-stage-only."""
    from huggingface_hub import HfApi, snapshot_download

    def download(**kwargs: object) -> str:
        download_impl = cast(Any, snapshot_download)
        return str(download_impl(**kwargs))

    return SourceInventoryClient(cache_root, HfApi().model_info, download)


def production_component_options() -> Mapping[str, object]:
    return {}


def production_generation_runtime(cache_root: Path, sources: Mapping[str, object]) -> object:
    from video_lab.adapters.animegen_mps import DiffusersAnimeGenGenerationRuntime

    return DiffusersAnimeGenGenerationRuntime(cache_root=cache_root, sources=cast(Any, sources))


__all__ = [
    "AnimeGenFeasibilityExecutor",
    "canonical_animegen_configuration",
    "evaluate_preassembly_preflight",
    "production_component_options",
    "production_generation_runtime",
    "production_inventory_client",
    "production_system_observation",
]
