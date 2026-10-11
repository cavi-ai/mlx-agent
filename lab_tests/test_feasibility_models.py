from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from math import inf, nan

import pytest
from pydantic import ValidationError

from lab_tests.test_capabilities import manifest as base_manifest
from video_lab.adapters.animegen_mps import ADAPTER_ID
from video_lab.animegen_feasibility import canonical_animegen_configuration
from video_lab.domain import LoraBinding
from video_lab.feasibility_models import (
    FeasibilityConfiguration,
    FeasibilityRun,
    FeasibilityStage,
    StageAttempt,
    StageEvidence,
    StageOutcome,
    SupportVerdict,
)
from video_lab.source_inventory import ANIMEGEN_SOURCE_REPOS

FINGERPRINT = "a" * 64


def test_canonical_animegen_configuration_requires_bf16_storage() -> None:
    revision = "a" * 40
    checksum = "1" * 64
    sources = {
        component: {"id": repo, "revision": revision, "checksum": f"sha256:{checksum}"}
        for component, repo in ANIMEGEN_SOURCE_REPOS.items()
    }

    def profile(width: int, height: int, quantization: str) -> dict[str, object]:
        return (
            base_manifest()
            .model_copy(
                update={
                    "model_id": sources["animegen"]["id"],
                    "model_revision": revision,
                    "model_checksum": checksum,
                    "adapter_id": ADAPTER_ID,
                    "runtime_versions": {"python": "3.12"},
                    "source_checksums": {component: checksum for component in sources},
                    "prompt": "anime character walking",
                    "seed": 7,
                    "width": width,
                    "height": height,
                    "scheduler": "flow-match-euler-shift-3",
                    "dtype": "bfloat16",
                    "quantization": quantization,
                    "loras": tuple(
                        LoraBinding(
                            model_id=sources[name]["id"],
                            revision=revision,
                            checksum=checksum,
                            weight=weight,
                        )
                        for name, weight in (("lightning-high", 2.0), ("lightning-low", 1.0))
                    ),
                    "memory_strategy": "model_cpu_offload",
                    "hardware_fingerprint": "apple-silicon",
                    "os_fingerprint": "macos-arm64",
                }
            )
            .model_dump(mode="json")
        )

    def configuration(quantization: str) -> FeasibilityConfiguration:
        return FeasibilityConfiguration(
            adapter_id=ADAPTER_ID,
            protocol_version="animegen-feasibility.v2",
            sources=sources,
            runtime_versions={"python": "3.12"},
            hardware_fingerprint="apple-silicon",
            os_fingerprint="macos-arm64",
            manifest_profiles={
                "minimum": profile(128, 80, quantization),
                "standard": profile(512, 288, quantization),
            },
            memory_strategy="model_cpu_offload",
        )

    assert (
        canonical_animegen_configuration(configuration("bf16"))["manifest_profiles"]["minimum"][
            "quantization"
        ]
        == "bf16"
    )
    with pytest.raises(ValueError, match="canonical AnimeGen configuration"):
        canonical_animegen_configuration(configuration("layerwise-fp8"))


def test_protocol_enums_have_closed_exact_values() -> None:
    assert [stage.value for stage in FeasibilityStage] == [
        "resolve_sources",
        "inventory_sources",
        "storage_preflight",
        "download_sources",
        "load_text_stack",
        "load_vae",
        "load_high_transformer",
        "load_low_transformer",
        "assemble_pipeline",
        "generate_minimum_1",
        "generate_minimum_2",
        "qualify_standard",
        "validate_controlled_pair",
        "publish_verdict",
    ]
    assert [outcome.value for outcome in StageOutcome] == [
        "passed",
        "incompatible",
        "resource_exhausted",
        "interrupted",
        "invalid_evidence",
        "unexpected_failure",
    ]
    assert [verdict.value for verdict in SupportVerdict] == [
        "pending",
        "incompatible",
        "resource_limited",
        "experimental_local_hybrid",
        "standard_lab_qualified",
    ]


def test_stage_evidence_deeply_freezes_inputs_and_serializes_as_json_values() -> None:
    details: dict[str, object] = {"placement": {"vae": "mps"}, "files": ["one", {"n": 2}]}
    measurements = {"elapsed_seconds": 1.25, "peak_bytes": 4.0}

    evidence = StageEvidence(measurements=measurements, details=details)
    details["placement"] = {"vae": "cpu"}
    measurements["elapsed_seconds"] = 99

    assert evidence.model_dump(mode="json") == {
        "measurements": {"elapsed_seconds": 1.25, "peak_bytes": 4.0},
        "details": {"placement": {"vae": "mps"}, "files": ["one", {"n": 2}]},
    }
    with pytest.raises(TypeError):
        evidence.details["changed"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        evidence.details["placement"]["vae"] = "cpu"  # type: ignore[index]
    with pytest.raises(TypeError):
        evidence.details["files"][1]["n"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        StageEvidence().measurements["changed"] = 1  # type: ignore[index]


@pytest.mark.parametrize("measurement", [-0.1, nan, inf, -inf])
def test_stage_evidence_rejects_negative_or_non_finite_measurements(measurement: float) -> None:
    with pytest.raises(ValidationError):
        StageEvidence(measurements={"bad": measurement})


def test_run_and_attempt_are_frozen_validate_fingerprints_and_normalize_utc() -> None:
    local_time = datetime(2026, 7, 16, 10, 30, tzinfo=timezone(timedelta(hours=-4)))
    configuration: dict[str, object] = {"profile": {"frames": 17}, "loras": ["high", "low"]}
    run = FeasibilityRun(
        id="feasibility_1",
        adapter_id="adapter.test",
        protocol_version="animegen-feasibility.v1",
        input_fingerprint=FINGERPRINT,
        configuration=configuration,
        verdict=SupportVerdict.PENDING,
        created_at=local_time,
        updated_at=local_time,
    )
    attempt = StageAttempt(
        id="stage_attempt_1",
        run_id=run.id,
        stage=FeasibilityStage.RESOLVE_SOURCES,
        ordinal=1,
        input_fingerprint=FINGERPRINT,
        outcome=None,
        started_at=local_time,
    )
    configuration["profile"] = {"frames": 1}

    assert run.created_at.tzinfo is UTC
    assert run.updated_at.tzinfo is UTC
    assert run.configuration["profile"]["frames"] == 17  # type: ignore[index]
    assert attempt.started_at.tzinfo is UTC
    with pytest.raises(ValidationError):
        run.model_copy(update={"input_fingerprint": "not-a-sha256"})
    with pytest.raises(ValidationError):
        attempt.model_copy(update={"input_fingerprint": "A" * 64})
    with pytest.raises(ValidationError, match="timezone-aware"):
        run.model_copy(update={"created_at": datetime(2026, 7, 16)})
    with pytest.raises(ValidationError):
        FeasibilityRun.model_validate({**run.model_dump(), "unknown": True})


def test_attempt_requires_consistent_terminal_fields_and_bounded_diagnostics() -> None:
    now = datetime.now(UTC)
    active = StageAttempt(
        id="stage_attempt_1",
        run_id="feasibility_1",
        stage=FeasibilityStage.RESOLVE_SOURCES,
        ordinal=1,
        input_fingerprint=FINGERPRINT,
        outcome=None,
        started_at=now,
    )
    with pytest.raises(ValidationError):
        active.model_copy(update={"diagnostic_code": "x"})
    with pytest.raises(ValidationError):
        active.model_copy(
            update={"outcome": StageOutcome.PASSED, "finished_at": now, "diagnostic_code": "x" * 65}
        )
    with pytest.raises(ValidationError):
        active.model_copy(
            update={
                "outcome": StageOutcome.PASSED,
                "finished_at": now,
                "diagnostic_message": "x" * 1001,
            }
        )
    with pytest.raises(ValidationError):
        active.model_copy(update={"outcome": StageOutcome.PASSED})
