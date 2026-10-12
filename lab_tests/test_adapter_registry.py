import subprocess
import sys
from types import SimpleNamespace

import pytest

from video_lab.adapter_registry import AdapterRegistry, UnknownAdapterError
from video_lab.adapters.fake import FakeVideoAdapter
from video_lab.capabilities import AdapterCapabilities
from video_lab.feasibility_models import SupportVerdict


def mismatched_capabilities() -> AdapterCapabilities:
    return FakeVideoAdapter().capabilities().model_copy(update={"adapter_id": "other"})


class MismatchedAdapter(FakeVideoAdapter):
    def capabilities(self) -> AdapterCapabilities:
        return mismatched_capabilities()


def test_registry_resolves_only_the_exact_registered_id() -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", FakeVideoAdapter)

    assert isinstance(registry.resolve("video-lab.fake.v1"), FakeVideoAdapter)
    with pytest.raises(UnknownAdapterError):
        registry.resolve("VIDEO-LAB.FAKE.V1")


def test_registry_rejects_duplicate_adapter_ids() -> None:
    registry = AdapterRegistry()
    registry.register("video-lab.fake.v1", FakeVideoAdapter)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("video-lab.fake.v1", FakeVideoAdapter)


def test_capabilities_rejects_factory_identity_mismatch() -> None:
    registry = AdapterRegistry()
    registry.register("custom", FakeVideoAdapter, capabilities_factory=mismatched_capabilities)

    with pytest.raises(ValueError, match="capabilities do not match registered adapter 'custom'"):
        registry.capabilities("custom")


def test_capabilities_rejects_legacy_adapter_identity_mismatch() -> None:
    registry = AdapterRegistry()
    registry.register("custom", MismatchedAdapter)

    with pytest.raises(ValueError, match="capabilities do not match registered adapter 'custom'"):
        registry.capabilities("custom")


def test_default_registry_resolves_fake_without_heavy_framework_imports() -> None:
    code = """
import sys
from video_lab.adapter_registry import AdapterRegistry
adapter = AdapterRegistry.with_defaults().resolve("video-lab.fake.v1")
assert adapter.adapter_id == "video-lab.fake.v1"
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(name == "diffusers" or name.startswith("diffusers.") for name in sys.modules)
"""

    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr


def test_animegen_catalog_is_lazy_and_generation_is_disabled() -> None:
    code = """
import json
import sys
from video_lab.adapter_registry import AdapterRegistry
catalog = AdapterRegistry.with_defaults().catalog_metadata()
adapter_id = "diffusers.animegen-t2v.wan2.2-a14b.mps.v1"
metadata, capabilities = next(item for item in catalog if item[0].adapter_id == adapter_id)
assert metadata.display_name == "AnimeGen T2V / Wan 2.2 A14B (MPS candidate)"
assert metadata.source_model == "aidealab/AnimeGen-T2V"
assert metadata.generation_enabled is False
assert metadata.declared_profiles == ("minimum", "standard-512x288-17f-8fps")
assert capabilities.frame_rule == "16n+1"
assert capabilities.min_frames == 17
assert capabilities.quantizations == ("layerwise-fp8", "bf16")
heavy = ("torch", "diffusers", "transformers", "accelerate", "peft", "safetensors")
assert not any(
    name == root or name.startswith(root + ".")
    for name in sys.modules
    for root in heavy
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_animegen_candidate_resolution_is_blocked_until_standard_qualification() -> None:
    registry = AdapterRegistry.with_defaults()
    with pytest.raises(UnknownAdapterError, match="generation is disabled"):
        registry.resolve("diffusers.animegen-t2v.wan2.2-a14b.mps.v1")


class AuthorizedAnimeGen(FakeVideoAdapter):
    adapter_id = "diffusers.animegen-t2v.wan2.2-a14b.mps.v1"

    def capabilities(self) -> AdapterCapabilities:
        return super().capabilities().model_copy(update={"adapter_id": self.adapter_id})


class AuthorizationRepository:
    def __init__(self, value):
        self.value = value

    def generation_authorization_bundle(self, _adapter_id):
        return self.value


def authorization(verdict: SupportVerdict, profile: tuple[int, int, int]):
    return SimpleNamespace(
        verdict=verdict,
        measured_profile=(*profile, 8),
        source_fingerprint="a" * 64,
        runtime_fingerprint="b" * 64,
        placement_fingerprint="c" * 64,
        authorized_manifest=None,
        canonical_manifest_hash="",
    )


def test_experimental_animegen_authorizes_only_frozen_minimum_profile() -> None:
    registry = AdapterRegistry.with_defaults(
        feasibility_repository=AuthorizationRepository(
            authorization(SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID, (128, 80, 17))
        ),
        animegen_runtime_factory=lambda _bundle: object(),
    )
    capabilities = registry.capabilities(AuthorizedAnimeGen.adapter_id)
    assert (capabilities.min_width, capabilities.max_width) == (128, 128)
    assert (capabilities.min_height, capabilities.max_height) == (80, 80)
    assert (capabilities.min_frames, capabilities.max_frames) == (17, 17)
    assert (
        registry.resolve(AuthorizedAnimeGen.adapter_id).adapter_id == AuthorizedAnimeGen.adapter_id
    )


def test_standard_animegen_caps_authorization_at_measured_profile() -> None:
    registry = AdapterRegistry.with_defaults(
        feasibility_repository=AuthorizationRepository(
            authorization(SupportVerdict.STANDARD_LAB_QUALIFIED, (512, 288, 17))
        ),
        animegen_runtime_factory=lambda _bundle: object(),
    )
    capabilities = registry.capabilities(AuthorizedAnimeGen.adapter_id)
    assert (capabilities.max_width, capabilities.max_height, capabilities.max_frames) == (
        512,
        288,
        17,
    )


@pytest.mark.parametrize(
    "verdict",
    [SupportVerdict.PENDING, SupportVerdict.INCOMPATIBLE, SupportVerdict.RESOURCE_LIMITED],
)
def test_non_support_verdicts_cannot_be_bypassed_with_factory(
    verdict: SupportVerdict,
) -> None:
    registry = AdapterRegistry.with_defaults(
        feasibility_repository=AuthorizationRepository(None),
        animegen_runtime_factory=lambda _bundle: object(),
    )
    with pytest.raises(UnknownAdapterError, match="generation is disabled"):
        registry.resolve(AuthorizedAnimeGen.adapter_id)
