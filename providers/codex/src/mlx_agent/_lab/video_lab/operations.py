from __future__ import annotations

from typing import Any, Literal

from video_lab.adapter_registry import AdapterRegistry
from video_lab.capabilities import AdapterCapabilities
from video_lab.domain import ExperimentDraft, FrozenModel, Manifest
from video_lab.services import ExperimentService, ExperimentView


class ManifestFieldSpec(FrozenModel):
    path: str
    label: str
    value_type: Literal["string", "integer", "number", "boolean"]
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    rule: Literal["16n+1"] | None = None


class AdapterCatalogEntry(FrozenModel):
    adapter_id: str
    display_name: str
    capabilities: AdapterCapabilities
    fields: tuple[ManifestFieldSpec, ...]
    source_model: str | None = None
    frame_rule: str | None = None
    declared_profiles: tuple[str, ...] = ()
    generation_enabled: bool = True


class ExperimentPreview(FrozenModel):
    variant_a: Manifest
    variant_b: Manifest
    variable_path: str
    changed_paths: tuple[str, ...]


def _field_specs(capabilities: AdapterCapabilities) -> tuple[ManifestFieldSpec, ...]:
    return (
        ManifestFieldSpec(path="prompt", label="Prompt", value_type="string"),
        ManifestFieldSpec(path="negative_prompt", label="Negative prompt", value_type="string"),
        ManifestFieldSpec(path="seed", label="Seed", value_type="integer"),
        ManifestFieldSpec(
            path="width",
            label="Width",
            value_type="integer",
            minimum=capabilities.min_width,
            maximum=capabilities.max_width,
            step=capabilities.width_multiple,
        ),
        ManifestFieldSpec(
            path="height",
            label="Height",
            value_type="integer",
            minimum=capabilities.min_height,
            maximum=capabilities.max_height,
            step=capabilities.height_multiple,
        ),
        ManifestFieldSpec(
            path="num_frames",
            label="Frames",
            value_type="integer",
            minimum=capabilities.min_frames,
            maximum=capabilities.max_frames,
            rule=None if capabilities.frame_rule == "any" else capabilities.frame_rule,
        ),
        ManifestFieldSpec(path="fps", label="FPS", value_type="integer", minimum=1),
        ManifestFieldSpec(
            path="inference_steps", label="Inference steps", value_type="integer", minimum=1
        ),
        ManifestFieldSpec(
            path="guidance_scale", label="Guidance scale", value_type="number", minimum=0
        ),
        ManifestFieldSpec(
            path="scheduler",
            label="Scheduler",
            value_type="string",
            options=capabilities.schedulers,
        ),
    )


class LaboratoryOperations:
    def __init__(self, registry: AdapterRegistry, service: ExperimentService) -> None:
        self.registry = registry
        self.service = service

    @staticmethod
    def catalog_for(registry: AdapterRegistry) -> tuple[AdapterCatalogEntry, ...]:
        return tuple(
            AdapterCatalogEntry(
                adapter_id=metadata.adapter_id,
                display_name=metadata.display_name,
                capabilities=capabilities,
                fields=_field_specs(capabilities),
                source_model=metadata.source_model,
                frame_rule=metadata.frame_rule,
                declared_profiles=metadata.declared_profiles,
                generation_enabled=metadata.generation_enabled,
            )
            for metadata, capabilities in registry.catalog_metadata()
        )

    def catalog(self) -> tuple[AdapterCatalogEntry, ...]:
        return self.catalog_for(self.registry)

    def preview(
        self, draft: ExperimentDraft | dict[str, Any], adapter_id: str
    ) -> ExperimentPreview:
        canonical = ExperimentDraft.model_validate(draft)
        pair = self.service.preview(canonical, self.registry.capabilities(adapter_id))
        return ExperimentPreview.model_validate(pair.model_dump(mode="json"))

    def create(self, draft: ExperimentDraft | dict[str, Any], adapter_id: str) -> ExperimentView:
        canonical = ExperimentDraft.model_validate(draft)
        return self.service.create(canonical, self.registry.capabilities(adapter_id))
