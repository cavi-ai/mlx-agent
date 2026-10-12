from collections.abc import Callable
from typing import Protocol

from video_lab.adapter_protocol import GenerationAdapter
from video_lab.adapters.animegen_mps import AnimeGenStorageMode
from video_lab.capabilities import AdapterCapabilities
from video_lab.domain import FrozenModel
from video_lab.feasibility_models import SupportVerdict
from video_lab.feasibility_repository import GenerationAuthorizationBundle

AdapterFactory = Callable[[], GenerationAdapter]
CapabilitiesFactory = Callable[[], AdapterCapabilities]


class FeasibilityAuthorizationRepository(Protocol):
    def generation_authorization_bundle(
        self, adapter_id: str
    ) -> GenerationAuthorizationBundle | None: ...


class AdapterMetadata(FrozenModel):
    adapter_id: str
    display_name: str
    source_model: str | None = None
    frame_rule: str | None = None
    declared_profiles: tuple[str, ...] = ()
    generation_enabled: bool = True


MetadataFactory = Callable[[], AdapterMetadata]


class _Registration:
    def __init__(
        self,
        factory: AdapterFactory,
        metadata_factory: MetadataFactory | None,
        capabilities_factory: CapabilitiesFactory | None,
    ) -> None:
        self.factory = factory
        self.metadata_factory = metadata_factory
        self.capabilities_factory = capabilities_factory


class UnknownAdapterError(LookupError):
    """Raised when an exact adapter ID is not registered."""


class AdapterRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, _Registration] = {}

    def register(
        self,
        adapter_id: str,
        factory: AdapterFactory,
        *,
        metadata_factory: MetadataFactory | None = None,
        capabilities_factory: CapabilitiesFactory | None = None,
    ) -> None:
        if adapter_id in self._registrations:
            raise ValueError(f"adapter {adapter_id!r} is already registered")
        self._registrations[adapter_id] = _Registration(
            factory, metadata_factory, capabilities_factory
        )

    def resolve(self, adapter_id: str) -> GenerationAdapter:
        try:
            registration = self._registrations[adapter_id]
        except KeyError as error:
            raise UnknownAdapterError(f"unknown adapter: {adapter_id}") from error
        return registration.factory()

    def catalog_metadata(self) -> tuple[tuple[AdapterMetadata, AdapterCapabilities], ...]:
        catalog: list[tuple[AdapterMetadata, AdapterCapabilities]] = []
        for adapter_id, registration in self._registrations.items():
            if registration.metadata_factory is None or registration.capabilities_factory is None:
                continue
            metadata = registration.metadata_factory()
            capabilities = registration.capabilities_factory()
            if metadata.adapter_id != adapter_id:
                raise ValueError(
                    f"catalog metadata does not match registered adapter {adapter_id!r}"
                )
            self._validate_capabilities_identity(adapter_id, capabilities)
            catalog.append((metadata, capabilities))
        return tuple(catalog)

    def capabilities(self, adapter_id: str) -> AdapterCapabilities:
        try:
            registration = self._registrations[adapter_id]
        except KeyError as error:
            raise UnknownAdapterError(f"unknown adapter: {adapter_id}") from error
        capabilities = (
            registration.factory().capabilities()
            if registration.capabilities_factory is None
            else registration.capabilities_factory()
        )
        self._validate_capabilities_identity(adapter_id, capabilities)
        return capabilities

    @staticmethod
    def _validate_capabilities_identity(adapter_id: str, capabilities: AdapterCapabilities) -> None:
        if capabilities.adapter_id != adapter_id:
            raise ValueError(f"capabilities do not match registered adapter {adapter_id!r}")

    @classmethod
    def with_defaults(
        cls,
        *,
        feasibility_repository: FeasibilityAuthorizationRepository | None = None,
        animegen_runtime_factory: Callable[[GenerationAuthorizationBundle], object] | None = None,
    ) -> "AdapterRegistry":
        registry = cls()

        def fake_factory() -> GenerationAdapter:
            from video_lab.adapters.fake import FakeVideoAdapter

            return FakeVideoAdapter()

        def fake_capabilities() -> AdapterCapabilities:
            return AdapterCapabilities(
                adapter_id="video-lab.fake.v1",
                model_patterns=("*",),
                backends=("cpu",),
                dtypes=("uint8",),
                quantizations=("none",),
                schedulers=("fake",),
                min_width=1,
                max_width=4096,
                width_multiple=1,
                min_height=1,
                max_height=4096,
                height_multiple=1,
                min_frames=17,
                max_frames=17,
                frame_rule="16n+1",
                supports_lora=False,
            )

        registry.register(
            "video-lab.fake.v1",
            fake_factory,
            metadata_factory=lambda: AdapterMetadata(
                adapter_id="video-lab.fake.v1", display_name="Fake video"
            ),
            capabilities_factory=fake_capabilities,
        )

        def wan_mps_factory() -> GenerationAdapter:
            from video_lab.adapters.wan_mps import WanMpsAdapter

            return WanMpsAdapter()

        def wan_mps_capabilities() -> AdapterCapabilities:
            return AdapterCapabilities(
                adapter_id="diffusers.wan2.1-t2v-1.3b.mps.v1",
                model_patterns=("Wan-AI/Wan2.1-T2V-1.3B-Diffusers",),
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

        registry.register(
            "diffusers.wan2.1-t2v-1.3b.mps.v1",
            wan_mps_factory,
            metadata_factory=lambda: AdapterMetadata(
                adapter_id="diffusers.wan2.1-t2v-1.3b.mps.v1", display_name="Wan 2.1 T2V 1.3B (MPS)"
            ),
            capabilities_factory=wan_mps_capabilities,
        )

        animegen_id = "diffusers.animegen-t2v.wan2.2-a14b.mps.v1"
        authorization_bundle = (
            None
            if feasibility_repository is None
            else feasibility_repository.generation_authorization_bundle(animegen_id)
        )
        animegen_verdict = (
            SupportVerdict.PENDING if authorization_bundle is None else authorization_bundle.verdict
        )
        animegen_profile = (
            None if authorization_bundle is None else authorization_bundle.measured_profile[:3]
        )

        def blocked_animegen_factory() -> GenerationAdapter:
            raise UnknownAdapterError(
                f"generation is disabled for candidate adapter: {animegen_id}"
            )

        def animegen_capabilities() -> AdapterCapabilities:
            if animegen_verdict is SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID:
                if animegen_profile is None:
                    raise ValueError("experimental AnimeGen authorization requires minimum profile")
                min_width, min_height, min_frames = animegen_profile
                max_width, max_height, max_frames = min_width, min_height, min_frames
            elif animegen_verdict is SupportVerdict.STANDARD_LAB_QUALIFIED:
                if animegen_profile is None:
                    raise ValueError("standard AnimeGen authorization requires measured profile")
                min_width, min_height, min_frames = 16, 16, 17
                max_width, max_height, max_frames = animegen_profile
            else:
                min_width, min_height, min_frames = 16, 16, 17
                max_width, max_height, max_frames = 4096, 4096, 129
            return AdapterCapabilities(
                adapter_id=animegen_id,
                model_patterns=("aidealab/AnimeGen-T2V",),
                backends=("mps",),
                dtypes=("bfloat16",),
                quantizations=tuple(mode.value for mode in AnimeGenStorageMode),
                schedulers=("flow-match-euler-shift-3",),
                min_width=min_width,
                max_width=max_width,
                width_multiple=16,
                min_height=min_height,
                max_height=max_height,
                height_multiple=16,
                min_frames=min_frames,
                max_frames=max_frames,
                frame_rule="16n+1",
                supports_lora=True,
                min_fps=(
                    1 if authorization_bundle is None else authorization_bundle.measured_profile[3]
                ),
                max_fps=(
                    120
                    if authorization_bundle is None
                    else authorization_bundle.measured_profile[3]
                ),
            )

        registry.register(
            animegen_id,
            (
                (
                    lambda: _authorized_animegen_adapter(
                        animegen_runtime_factory, authorization_bundle
                    )
                )
                if animegen_runtime_factory is not None
                and authorization_bundle is not None
                and animegen_verdict
                in (
                    SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID,
                    SupportVerdict.STANDARD_LAB_QUALIFIED,
                )
                else blocked_animegen_factory
            ),
            metadata_factory=lambda: AdapterMetadata(
                adapter_id=animegen_id,
                display_name="AnimeGen T2V / Wan 2.2 A14B (MPS candidate)",
                source_model="aidealab/AnimeGen-T2V",
                frame_rule="16n+1",
                declared_profiles=("minimum", "standard-512x288-17f-8fps"),
                generation_enabled=animegen_verdict
                in (
                    SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID,
                    SupportVerdict.STANDARD_LAB_QUALIFIED,
                ),
            ),
            capabilities_factory=animegen_capabilities,
        )
        return registry


def _authorized_animegen_adapter(
    runtime_factory: Callable[[GenerationAuthorizationBundle], object],
    authorization: GenerationAuthorizationBundle | None,
) -> GenerationAdapter:
    from typing import cast

    from video_lab.adapters.animegen_mps import AnimeGenGenerationRuntime, AnimeGenMpsAdapter

    if authorization is None:
        raise UnknownAdapterError("AnimeGen generation is not authorized")
    verdict, profile = authorization.verdict, authorization.measured_profile[:3]
    return AnimeGenMpsAdapter(
        runtime=cast(AnimeGenGenerationRuntime, runtime_factory(authorization)),
        authorized_profile=profile,
        exact_profile_only=verdict is SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID,
        expected_fingerprints=(
            authorization.source_fingerprint,
            authorization.runtime_fingerprint,
            authorization.placement_fingerprint,
        ),
        authorized_manifest=authorization.authorized_manifest,
        authorized_verdict=authorization.verdict.value,
        authorized_manifest_hash=authorization.canonical_manifest_hash,
    )
