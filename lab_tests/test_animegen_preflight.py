from __future__ import annotations

from collections.abc import Mapping

import pytest

import video_lab.animegen_feasibility as animegen_feasibility
from video_lab.feasibility_models import FeasibilityStage, StageOutcome


def _configuration() -> dict[str, object]:
    return {
        "runtime_versions": {"python": "3.12", "video-lab": "source-sha256:abc"},
        "hardware_fingerprint": "apple-m5-max-128gb-18c",
        "os_fingerprint": "macos-26.5.2-25F84-arm64",
        "sources": {
            "animegen": {"checksum": f"sha256:{'a' * 64}"},
            "wan-base": {"checksum": f"sha256:{'b' * 64}"},
        },
    }


def _observation() -> dict[str, object]:
    return {
        "runtime_versions": {"python": "3.12", "video-lab": "source-sha256:abc"},
        "hardware_fingerprint": "apple-m5-max-128gb-18c",
        "os_fingerprint": "macos-26.5.2-25F84-arm64",
        "mps": {"available": True, "built": True},
        "unified_memory": {
            "total_bytes": 128 * 1024**3,
            "available_bytes": 80 * 1024**3,
            "required_available_bytes": 26 * 1024**3,
            "policy": "max(24GiB,20pct_total)",
        },
    }


def _evaluate(
    *,
    configuration: Mapping[str, object] | None = None,
    source_digests: Mapping[str, str] | None = None,
    storage: Mapping[str, int] | None = None,
    observation: Mapping[str, object] | None = None,
):
    evaluator = getattr(animegen_feasibility, "evaluate_preassembly_preflight", None)
    assert callable(evaluator), "unified preassembly evaluator is missing"
    return evaluator(
        configuration=configuration or _configuration(),
        source_digests=source_digests or {"animegen": "a" * 64, "wan-base": "b" * 64},
        storage=storage
        or {
            "expected_bytes": 72 * 1024**3,
            "safety_margin_bytes": 20 * 1024**3,
            "required_bytes": 92 * 1024**3,
            "free_bytes": 1_000 * 1024**3,
        },
        observation=observation or _observation(),
        stage=FeasibilityStage.LOAD_TEXT_STACK,
    )


def test_unified_preassembly_preflight_records_complete_requested_and_observed_identity() -> None:
    result = _evaluate()

    assert result.outcome is StageOutcome.PASSED
    evidence = result.evidence.details["preassembly_preflight"]
    assert evidence["stage"] == "load_text_stack"
    assert set(evidence["requested"]) == {
        "runtime_versions",
        "source_digests",
        "hardware_fingerprint",
        "os_fingerprint",
        "mps",
        "disk",
        "unified_memory",
    }
    assert set(evidence["observed"]) == set(evidence["requested"])
    assert evidence["observed"]["mps"] == {"available": True, "built": True}
    assert evidence["observed"]["disk"]["free_bytes"] == 1_000 * 1024**3


@pytest.mark.parametrize("drift", ["runtime", "source", "hardware", "os"])
def test_unified_preassembly_preflight_classifies_resume_identity_drift_as_invalid(
    drift: str,
) -> None:
    configuration = _configuration()
    observation = _observation()
    sources = {"animegen": "a" * 64, "wan-base": "b" * 64}
    if drift == "runtime":
        observation["runtime_versions"] = {"python": "3.13"}
    elif drift == "source":
        sources["wan-base"] = "c" * 64
    elif drift == "hardware":
        observation["hardware_fingerprint"] = "apple-m5-max-64gb-18c"
    else:
        observation["os_fingerprint"] = "macos-drifted"

    result = _evaluate(
        configuration=configuration,
        source_digests=sources,
        observation=observation,
    )

    assert result.outcome is StageOutcome.INVALID_EVIDENCE
    assert result.diagnostic_code == "preflight_identity_drift"
    assert result.evidence.details["preassembly_preflight"]["identity_mismatches"] == (drift,)


@pytest.mark.parametrize(
    ("storage", "memory", "code"),
    [
        ({"required_bytes": 100, "free_bytes": 99}, None, "insufficient_disk_headroom"),
        (
            None,
            {
                "total_bytes": 128,
                "available_bytes": 25,
                "required_available_bytes": 26,
                "policy": "max(24GiB,20pct_total)",
            },
            "insufficient_unified_memory_headroom",
        ),
    ],
)
def test_unified_preassembly_preflight_classifies_headroom_as_resource_exhaustion(
    storage: Mapping[str, int] | None,
    memory: Mapping[str, object] | None,
    code: str,
) -> None:
    observation = _observation()
    if memory is not None:
        observation["unified_memory"] = memory

    result = _evaluate(storage=storage, observation=observation)

    assert result.outcome is StageOutcome.RESOURCE_EXHAUSTED
    assert result.diagnostic_code == code


def test_unified_preassembly_preflight_classifies_missing_mps_as_incompatible() -> None:
    observation = _observation()
    observation["mps"] = {"available": False, "built": True}

    result = _evaluate(observation=observation)

    assert result.outcome is StageOutcome.INCOMPATIBLE
    assert result.diagnostic_code == "mps_unavailable"
