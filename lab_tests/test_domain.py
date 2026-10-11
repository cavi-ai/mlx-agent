import json
from typing import Any, cast

import pytest
from pydantic import ValidationError

from video_lab.domain import Manifest, canonical_json, new_id


def make_manifest() -> Manifest:
    return Manifest(
        model_id="org/model",
        model_revision="abc123",
        model_checksum="sha256:weights",
        adapter_id="diffusers-mps",
        adapter_version="0.1.0",
        runtime_versions={"python": "3.12.0"},
        prompt="a blue-haired character walking — 蒼",
        negative_prompt="flicker",
        seed=42,
        width=512,
        height=288,
        num_frames=17,
        fps=8,
        inference_steps=4,
        guidance_scale=1.0,
        scheduler="flow-match-euler",
        dtype="float16",
        quantization="none",
        backend="mps",
        device="mps",
        memory_strategy="unified",
        hardware_fingerprint="apple-m5-max-128gb",
        os_fingerprint="macos-test",
    )


def test_manifest_is_immutable_and_canonical() -> None:
    manifest = make_manifest()

    assert canonical_json(manifest).startswith('{"adapter_id":"diffusers-mps"')
    assert manifest.model_copy(update={"inference_steps": 8}).inference_steps == 8
    assert new_id("exp").startswith("exp_")


def test_manifest_rejects_top_level_and_nested_mutation() -> None:
    manifest = make_manifest()

    with pytest.raises(ValidationError, match="Instance is frozen"):
        manifest.prompt = "changed"

    with pytest.raises(TypeError):
        cast(Any, manifest.runtime_versions)["python"] = "3.13.0"


def test_manifest_copy_validates_and_freezes_runtime_version_updates() -> None:
    source = {"python": "3.12.1"}

    copied = make_manifest().model_copy(update={"runtime_versions": source})
    source["python"] = "3.13.0"

    assert copied.runtime_versions["python"] == "3.12.1"
    assert copied.model_copy(update={"inference_steps": 8}).inference_steps == 8
    with pytest.raises(TypeError):
        cast(Any, copied.runtime_versions)["python"] = "3.13.0"


def test_canonical_json_is_sorted_compact_stable_and_preserves_unicode() -> None:
    manifest = make_manifest()

    serialized = canonical_json(manifest)

    assert serialized == canonical_json(manifest)
    assert list(json.loads(serialized)) == sorted(manifest.model_dump(mode="json"))
    assert serialized.startswith('{"adapter_id":"diffusers-mps","adapter_version":"0.1.0"')
    assert '": "' not in serialized
    assert '", "' not in serialized
    assert "— 蒼" in serialized
    assert canonical_json({"z": 1, "a": 2}) == '{"a":2,"z":1}'
