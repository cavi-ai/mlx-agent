from video_lab.capabilities import AdapterCapabilities, validate_manifest_support
from video_lab.domain import Manifest


def manifest() -> Manifest:
    return Manifest(
        model_id="org/model",
        model_revision="r1",
        model_checksum="sha256:x",
        adapter_id="test",
        adapter_version="1",
        runtime_versions={},
        prompt="walk",
        seed=1,
        width=512,
        height=288,
        num_frames=17,
        fps=8,
        inference_steps=4,
        guidance_scale=1.0,
        scheduler="euler",
        dtype="float16",
        quantization="none",
        backend="mps",
        device="mps",
        memory_strategy="unified",
        hardware_fingerprint="m5",
        os_fingerprint="macos",
    )


def test_reports_every_unsupported_setting_without_mutation() -> None:
    original = manifest()
    capabilities = AdapterCapabilities(
        adapter_id="test",
        model_patterns=("org/*",),
        backends=("mps",),
        dtypes=("bfloat16",),
        quantizations=("none",),
        schedulers=("flow",),
        min_width=256,
        max_width=1024,
        width_multiple=16,
        min_height=256,
        max_height=1024,
        height_multiple=16,
        min_frames=1,
        max_frames=81,
        frame_rule="16n+1",
        supports_lora=False,
    )

    violations = validate_manifest_support(original, capabilities)

    assert {item.field for item in violations} == {"dtype", "scheduler", "loras"} - {"loras"}
    assert original == manifest()


def test_rejects_manifest_for_a_different_adapter() -> None:
    selected = AdapterCapabilities(
        adapter_id="other",
        model_patterns=("org/*",),
        backends=("mps",),
        dtypes=("float16",),
        quantizations=("none",),
        schedulers=("euler",),
        min_width=256,
        max_width=1024,
        width_multiple=16,
        min_height=256,
        max_height=1024,
        height_multiple=16,
        min_frames=1,
        max_frames=81,
        frame_rule="16n+1",
        supports_lora=False,
    )

    assert validate_manifest_support(manifest(), selected)[0].model_dump() == {
        "field": "adapter_id",
        "value": "test",
        "reason": "expected adapter other",
    }
