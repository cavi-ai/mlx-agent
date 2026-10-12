from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from lab_tests.test_capabilities import manifest as base_manifest
from video_lab.database import Database
from video_lab.domain import canonical_json
from video_lab.feasibility import (
    FeasibilityProtocol,
    FeasibilityService,
    GenerationQualificationExecutor,
    InvalidEvidenceError,
    ResourceExhaustedError,
    StageDefinition,
    StageExecutionContext,
    StageResult,
    build_generation_feasibility_service,
    canonical_fingerprint,
    classify_exception,
    sanitize_diagnostic,
    validate_controlled_pair_evidence,
)
from video_lab.feasibility_models import (
    FeasibilityStage,
    StageEvidence,
    StageOutcome,
    SupportVerdict,
)
from video_lab.feasibility_repository import FeasibilityRepository
from video_lab.normal_generation import CommittedGenerationEvidence
from video_lab.source_inventory import SourceInventoryError

STAGES = [stage.value for stage in FeasibilityStage]


class Executor:
    def __init__(self, results: Mapping[FeasibilityStage, StageResult | Exception] | None = None):
        self.results = dict(results or {})
        self.calls: list[FeasibilityStage] = []

    def execute(self, context: StageExecutionContext, stage: StageDefinition) -> StageResult:
        self.calls.append(stage.stage)
        result = self.results.get(stage.stage, StageResult.passed())
        if isinstance(result, BaseException):
            raise result
        return result


def config(**changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "adapter_id": " Diffusers.AnimeGen ",
        "protocol_version": "v1",
        "sources": {
            "model": {
                "id": "animegen/model",
                "revision": "a" * 40,
                "checksum": f"sha256:{'1' * 64}",
            }
        },
        "runtime_versions": {"torch": "2.7", "python": "3.12"},
        "hardware_fingerprint": "apple-m2-24gb",
        "os_fingerprint": "macos-15-arm64",
        "manifest_profiles": {
            "minimum": {"width": 128, "height": 80, "frames": 17, "fps": 8},
            "standard": {"width": 512, "height": 288, "frames": 17, "fps": 8},
        },
        "memory_strategy": "sequential_offload",
    }
    value.update(changes)
    return value


def service(
    database: Database, executor: Executor | None = None
) -> tuple[FeasibilityService, Executor]:
    actual = executor or Executor()
    return FeasibilityService(FeasibilityRepository(database), actual), actual


def advance(instance: FeasibilityService, run_id: str, count: int) -> None:
    for _ in range(count):
        instance.resume(run_id)


def generation_evidence(profile: str = "minimum", manifest: str = "f" * 64) -> StageEvidence:
    return StageEvidence(
        details={
            "artifact": {
                "format": "mp4",
                "valid": True,
                "sha256": "a" * 64,
                "decoded_frame_sha256": ["b" * 64] * 17,
                "codec": "h264",
                "duration_seconds": 2.125,
                "width": 512 if profile == "standard" else 128,
                "height": 288 if profile == "standard" else 80,
                "num_frames": 17,
                "fps": 8,
            },
            "canonical_manifest_hash": manifest,
            "profile": profile,
            "placement": "local_hybrid",
            "source_fingerprint": "c" * 64,
            "runtime_fingerprint": "d" * 64,
            "placement_fingerprint": "e" * 64,
            "fallback_violations": [],
        }
    )


def test_protocol_has_exact_order_and_prerequisites() -> None:
    protocol = FeasibilityProtocol.default()
    assert [definition.stage.value for definition in protocol.stages] == STAGES
    assert protocol.stages[0].prerequisites == ()
    for previous, current in zip(protocol.stages, protocol.stages[1:], strict=False):
        assert current.prerequisites == (previous.stage,)


def test_every_stage_persists_bounded_live_telemetry(database: Database) -> None:
    instance, _ = service(database)
    run = instance.start(config())

    status = instance.resume(run.id)
    instance.resume(run.id)

    evidence = status.attempts[-1].evidence
    assert evidence is not None
    assert evidence.measurements["stage_wall_seconds"] >= 0
    telemetry = evidence.details["telemetry"]
    assert isinstance(telemetry, Mapping)
    assert isinstance(telemetry["pid"], int)
    assert telemetry["sample_count"] <= 3
    assert len(telemetry["samples"]) == telemetry["sample_count"]
    assert set(telemetry["rss_bytes"]) == {"current", "peak", "availability"}
    assert set(telemetry["mps_bytes"]) == {
        "allocated_before",
        "allocated_endpoint_max",
        "allocated_after",
        "driver_before",
        "driver_endpoint_max",
        "driver_after",
        "availability",
    }


def test_failed_stage_retains_live_telemetry(database: Database) -> None:
    executor = Executor({FeasibilityStage.RESOLVE_SOURCES: RuntimeError("boom")})
    instance, _ = service(database, executor)
    run = instance.start(config())

    status = instance.resume(run.id)

    attempt = status.attempts[-1]
    assert attempt.outcome is StageOutcome.UNEXPECTED_FAILURE
    assert attempt.evidence is not None
    assert attempt.evidence.measurements["stage_wall_seconds"] >= 0
    assert "telemetry" in attempt.evidence.details


def test_status_derives_verdict_without_persisting(database: Database) -> None:
    instance, _ = service(database)
    run = instance.start(config())
    repository = instance.repository
    active = repository.start_attempt(run.id, FeasibilityStage.RESOLVE_SOURCES)
    repository.finish_attempt(active.id, StageOutcome.INCOMPATIBLE, StageEvidence())

    before = repository.get_run(run.id)
    status = instance.status(run.id)
    after = repository.get_run(run.id)

    assert status.verdict is SupportVerdict.INCOMPATIBLE
    assert before == after
    assert after.verdict is SupportVerdict.PENDING


def test_product_composition_root_constructs_generation_qualification_executor(
    database: Database,
) -> None:
    instance = build_generation_feasibility_service(
        repository=FeasibilityRepository(database),
        delegate=Executor(),
        experiments=object(),
        jobs=object(),
        artifacts=object(),
        worker=object(),
    )
    assert isinstance(instance.executor, GenerationQualificationExecutor)


def test_resume_executes_one_stage_and_blocks_after_failed_prerequisite(database: Database) -> None:
    executor = Executor(
        {
            FeasibilityStage.RESOLVE_SOURCES: StageResult(
                outcome=StageOutcome.INCOMPATIBLE, diagnostic_code="unsupported_runtime"
            )
        }
    )
    instance, _ = service(database, executor)
    run = instance.start(config())

    instance.resume(run.id)
    instance.resume(run.id)

    assert executor.calls == [FeasibilityStage.RESOLVE_SOURCES]
    assert len(instance.status(run.id).attempts) == 1
    assert instance.status(run.id).verdict is SupportVerdict.INCOMPATIBLE


def test_resume_rechecks_preassembly_guard_and_persists_drift_before_executor(
    database: Database,
) -> None:
    class GuardedExecutor(Executor):
        def __init__(self) -> None:
            super().__init__()
            self.preflight_calls: list[FeasibilityStage] = []

        def preflight(self, context: StageExecutionContext, stage: StageDefinition) -> StageResult:
            self.preflight_calls.append(stage.stage)
            evidence = StageEvidence(
                details={"preassembly_preflight": {"stage": stage.stage.value}}
            )
            if stage.stage is FeasibilityStage.LOAD_TEXT_STACK:
                return StageResult(
                    outcome=StageOutcome.INVALID_EVIDENCE,
                    evidence=evidence,
                    diagnostic_code="preflight_identity_drift",
                    diagnostic_message="runtime identity drift",
                )
            return StageResult.passed(evidence)

    executor = GuardedExecutor()
    instance = FeasibilityService(FeasibilityRepository(database), executor)
    run = instance.start(config())

    advance(instance, run.id, 5)

    status = instance.status(run.id)
    assert executor.preflight_calls == [
        FeasibilityStage.DOWNLOAD_SOURCES,
        FeasibilityStage.LOAD_TEXT_STACK,
    ]
    assert FeasibilityStage.LOAD_TEXT_STACK not in executor.calls
    assert status.attempts[-1].stage is FeasibilityStage.LOAD_TEXT_STACK
    assert status.attempts[-1].outcome is StageOutcome.INVALID_EVIDENCE
    assert status.attempts[-1].evidence is not None
    assert status.attempts[-1].evidence.details["preassembly_preflight"]["stage"] == (
        "load_text_stack"
    )


def test_terminal_is_idempotent_and_interrupted_stage_is_retried(database: Database) -> None:
    executor = Executor({FeasibilityStage.RESOLVE_SOURCES: KeyboardInterrupt()})
    instance, _ = service(database, executor)
    run = instance.start(config())
    instance.resume(run.id)
    assert instance.status(run.id).attempts[-1].outcome is StageOutcome.INTERRUPTED

    executor.results.clear()
    instance.resume(run.id)
    assert executor.calls == [FeasibilityStage.RESOLVE_SOURCES] * 2
    advance(instance, run.id, len(STAGES) - 1)
    before = len(executor.calls)
    instance.resume(run.id)
    assert len(executor.calls) == before


def test_fingerprint_is_canonical_and_changed_input_creates_new_run(database: Database) -> None:
    left = config()
    right = dict(reversed(list(left.items())))
    assert canonical_fingerprint(left) == canonical_fingerprint(right)

    instance, _ = service(database)
    first = instance.start(left)
    same = instance.start(right)
    changed = instance.start(config(memory_strategy="model_offload"))
    assert first.id == same.id
    assert first.input_fingerprint != changed.input_fingerprint
    assert first.adapter_id == "diffusers.animegen"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("protocol_version"),
        lambda value: value.__setitem__("protocol", value.pop("protocol_version")),
        lambda value: value.__setitem__("unexpected_identity", "x"),
        lambda value: value.__setitem__("platform", {"hardware": "m2"}),
        lambda value: value.__setitem__("sources", {}),
        lambda value: value["sources"]["model"].pop("id"),
        lambda value: value["sources"]["model"].__setitem__("revision", "main"),
        lambda value: value["sources"]["model"].__setitem__("revision", "A" * 40),
        lambda value: value["sources"]["model"].__setitem__("checksum", "sha256:bad"),
        lambda value: value.__setitem__("runtime_versions", {}),
        lambda value: value["manifest_profiles"].pop("minimum"),
        lambda value: value["manifest_profiles"].pop("standard"),
        lambda value: value["manifest_profiles"].__setitem__("minimum", {}),
    ],
)
def test_start_rejects_incomplete_aliased_extra_or_mutable_identity(
    database: Database, mutation: Any
) -> None:
    candidate = config()
    mutation(candidate)
    instance, _ = service(database)
    with pytest.raises(ValueError):
        instance.start(candidate)


def test_fingerprint_rejects_unvalidated_configuration() -> None:
    candidate = config()
    candidate.pop("hardware_fingerprint")
    with pytest.raises(ValueError):
        canonical_fingerprint(candidate)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("adapter_id", "   "),
        lambda value: value.__setitem__("protocol_version", "\t"),
        lambda value: value.__setitem__("hardware_fingerprint", "  "),
        lambda value: value.__setitem__("os_fingerprint", "\n"),
        lambda value: value.__setitem__("memory_strategy", "   "),
        lambda value: value.__setitem__("sources", {"  ": next(iter(value["sources"].values()))}),
        lambda value: value["sources"]["model"].__setitem__("id", "  "),
        lambda value: value.__setitem__("runtime_versions", {"  ": "2.7"}),
        lambda value: value.__setitem__("runtime_versions", {"torch": "  "}),
        lambda value: value["manifest_profiles"]["minimum"].__setitem__("dtype", "  "),
        lambda value: value["manifest_profiles"]["standard"].__setitem__("  ", "fp16"),
    ],
)
def test_identity_strings_reject_whitespace_after_normalization(
    database: Database, mutation: Any
) -> None:
    candidate = config()
    mutation(candidate)
    instance, _ = service(database)
    with pytest.raises(ValueError):
        instance.start(candidate)


def test_identity_strings_are_trimmed_before_fingerprinting(database: Database) -> None:
    candidate = config()
    candidate["adapter_id"] = "  Diffusers.AnimeGen  "
    candidate["protocol_version"] = "  v1  "
    candidate["sources"] = {" model ": {**candidate["sources"]["model"], "id": " animegen/model "}}
    candidate["runtime_versions"] = {" torch ": " 2.7 "}
    candidate["hardware_fingerprint"] = " apple-m2-24gb "
    candidate["os_fingerprint"] = " macos-15-arm64 "
    candidate["memory_strategy"] = " sequential_offload "
    run, _ = service(database)
    stored = run.start(candidate).model_dump(mode="json")["configuration"]
    assert stored["adapter_id"] == "diffusers.animegen"
    assert stored["protocol_version"] == "v1"
    assert stored["sources"]["model"]["id"] == "animegen/model"
    assert stored["runtime_versions"] == {"torch": "2.7"}


@pytest.mark.parametrize(
    "outcome",
    [StageOutcome.INCOMPATIBLE, StageOutcome.RESOURCE_EXHAUSTED, StageOutcome.INVALID_EVIDENCE],
)
def test_terminal_failures_are_not_retried_without_changed_configuration(
    database: Database, outcome: StageOutcome
) -> None:
    executor = Executor({FeasibilityStage.RESOLVE_SOURCES: StageResult(outcome=outcome)})
    instance, _ = service(database, executor)
    run = instance.start(config())
    instance.resume(run.id)
    instance.resume(run.id)
    assert executor.calls == [FeasibilityStage.RESOLVE_SOURCES]


def test_invalid_evidence_is_terminal_incompatible_and_non_supporting(
    database: Database,
) -> None:
    executor = Executor(
        {FeasibilityStage.RESOLVE_SOURCES: StageResult(outcome=StageOutcome.INVALID_EVIDENCE)}
    )
    instance, _ = service(database, executor)
    run = instance.start(config())

    status = instance.resume(run.id)

    assert status.verdict is SupportVerdict.INCOMPATIBLE
    assert instance.repository.get_run(run.id).verdict is SupportVerdict.INCOMPATIBLE
    assert instance.repository.generation_authorization_bundle(run.adapter_id) is None
    assert executor.calls == [FeasibilityStage.RESOLVE_SOURCES]


def test_verdict_rules(database: Database) -> None:
    results = {
        FeasibilityStage.GENERATE_MINIMUM_1: StageResult.passed(generation_evidence()),
        FeasibilityStage.GENERATE_MINIMUM_2: StageResult.passed(generation_evidence()),
        FeasibilityStage.QUALIFY_STANDARD: StageResult.passed(
            generation_evidence("standard", "9" * 64)
        ),
        FeasibilityStage.VALIDATE_CONTROLLED_PAIR: StageResult.passed(
            StageEvidence(
                details={
                    "valid": True,
                    "changed_paths": ["guidance_scale"],
                    "variable_path": "guidance_scale",
                    "pair_seed": 8,
                    "controlled_values": [1.0, 3.5],
                }
            )
        ),
    }
    instance, _ = service(database, Executor(results))
    run = instance.start(config())
    advance(instance, run.id, STAGES.index("generate_minimum_2") + 1)
    assert instance.status(run.id).verdict is SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID
    assert instance.repository.get_run(run.id).verdict is SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID
    advance(instance, run.id, 2)
    assert instance.status(run.id).verdict is SupportVerdict.STANDARD_LAB_QUALIFIED
    # Stage metadata is only a display projection. Without persisted pair jobs,
    # artifact bytes, completion probes, and blind pins it cannot upgrade authorization.
    assert instance.repository.get_run(run.id).verdict is SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID


@pytest.mark.parametrize(
    "pair_details",
    [
        {
            "valid": True,
            "changed_paths": ["inference_steps"],
            "variable_path": "inference_steps",
            "pair_seed": 8,
            "controlled_values": [4, 5],
        },
        {
            "valid": True,
            "changed_paths": ["guidance_scale"],
            "variable_path": "guidance_scale",
            "pair_seed": 8,
            "controlled_values": [1.0, 2.0],
        },
        {
            "valid": True,
            "changed_paths": ["guidance_scale"],
            "variable_path": "guidance_scale",
            "pair_seed": 7,
            "controlled_values": [1.0, 3.5],
        },
    ],
)
def test_verdict_rejects_noncanonical_controlled_pair_evidence(
    database: Database, pair_details: dict[str, object]
) -> None:
    results = {
        FeasibilityStage.GENERATE_MINIMUM_1: StageResult.passed(generation_evidence()),
        FeasibilityStage.GENERATE_MINIMUM_2: StageResult.passed(generation_evidence()),
        FeasibilityStage.QUALIFY_STANDARD: StageResult.passed(
            generation_evidence("standard", "9" * 64)
        ),
        FeasibilityStage.VALIDATE_CONTROLLED_PAIR: StageResult.passed(
            StageEvidence(details=pair_details)
        ),
    }
    instance, _ = service(database, Executor(results))
    run = instance.start(config())

    advance(instance, run.id, len(STAGES) - 1)

    assert instance.status(run.id).verdict is SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID
    assert instance.repository.get_run(run.id).verdict is SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID


def test_incomplete_generation_evidence_cannot_authorize_adapter(database: Database) -> None:
    incomplete = StageEvidence(
        details={
            "artifact": {"format": "mp4", "valid": True},
            "canonical_manifest_hash": "manifest-a",
            "placement": "local_hybrid",
            "fallback_violations": [],
        }
    )
    results = {
        FeasibilityStage.GENERATE_MINIMUM_1: StageResult.passed(incomplete),
        FeasibilityStage.GENERATE_MINIMUM_2: StageResult.passed(incomplete),
    }
    instance, _ = service(database, Executor(results))
    run = instance.start(config())
    advance(instance, run.id, STAGES.index("generate_minimum_2") + 1)
    assert instance.status(run.id).verdict is SupportVerdict.INCOMPATIBLE


def test_controlled_pair_requires_exactly_guidance_scale_diff() -> None:
    baseline = {"width": 512, "height": 288, "num_frames": 17, "fps": 8, "guidance_scale": 1.0}
    alternate = {**baseline, "guidance_scale": 3.5}
    evidence = validate_controlled_pair_evidence(
        baseline, alternate, variable_path="guidance_scale"
    )
    assert evidence.details["valid"] is True
    assert evidence.details["changed_paths"] == ("guidance_scale",)

    with pytest.raises(InvalidEvidenceError):
        validate_controlled_pair_evidence(
            baseline, {**alternate, "seed": 2}, variable_path="guidance_scale"
        )


@dataclass
class PairRunner:
    calls: list[tuple[object, object, str]]

    def run_one(self, manifest: object) -> CommittedGenerationEvidence:
        raise AssertionError(f"unexpected single generation: {manifest}")

    def run_pair(
        self, baseline: object, candidate: object, *, variable_path: str
    ) -> tuple[CommittedGenerationEvidence, CommittedGenerationEvidence]:
        self.calls.append((baseline, candidate, variable_path))
        evidence = (
            CommittedGenerationEvidence(
                experiment_id="experiment-1",
                job_id="job-baseline",
                artifact_sha256="a" * 64,
                artifact={},
                source_fingerprint="b" * 64,
                runtime_fingerprint="c" * 64,
                placement_fingerprint="d" * 64,
                placement="local_hybrid",
            ),
            CommittedGenerationEvidence(
                experiment_id="experiment-1",
                job_id="job-candidate",
                artifact_sha256="e" * 64,
                artifact={},
                source_fingerprint="b" * 64,
                runtime_fingerprint="c" * 64,
                placement_fingerprint="d" * 64,
                placement="local_hybrid",
            ),
        )
        return evidence


@dataclass
class BlindReviewStarter:
    calls: list[tuple[str, str]]

    def start(self, experiment_id: str, reviewer_id: str) -> object:
        self.calls.append((experiment_id, reviewer_id))
        return type("BlindSession", (), {"session_id": "review-session-1"})()


def test_guidance_pair_evidence_is_auditable_but_blinded_before_review(database: Database) -> None:
    standard = base_manifest().model_copy(
        update={"width": 512, "height": 288, "num_frames": 17, "fps": 8, "guidance_scale": 2.0}
    )
    manifest_profiles = dict(config()["manifest_profiles"])
    manifest_profiles["standard"] = standard.model_dump(mode="json")
    run = FeasibilityService(FeasibilityRepository(database), Executor()).start(
        config(manifest_profiles=manifest_profiles)
    )
    runner = PairRunner([])
    reviews = BlindReviewStarter([])
    result = GenerationQualificationExecutor(Executor(), runner, reviews).execute(
        StageExecutionContext(run, ()),
        StageDefinition(FeasibilityStage.VALIDATE_CONTROLLED_PAIR),
    )

    assert runner.calls == [
        (
            standard.model_copy(update={"seed": 8, "guidance_scale": 1.0}),
            standard.model_copy(update={"seed": 8, "guidance_scale": 3.5}),
            "guidance_scale",
        )
    ]
    assert standard.seed == 1
    assert result.evidence.details["changed_paths"] == ("guidance_scale",)
    assert result.evidence.details["pair_seed"] == 8
    assert result.evidence.details["experiment_id"] == "experiment-1"
    assert result.evidence.details["job_ids"] == ("job-baseline", "job-candidate")
    assert result.evidence.details["artifact_sha256"] == ("a" * 64, "e" * 64)
    assert result.evidence.details["controlled_values"] == (1.0, 3.5)
    assert reviews.calls == [("experiment-1", f"feasibility:{run.id}")]
    presentation = result.evidence.details["review_identity"]
    assert presentation == {
        "experiment_id": "experiment-1",
        "session_id": "review-session-1",
        "state": "blinded_randomized",
    }
    assert "guidance" not in json.dumps(dict(presentation))


def test_atomic_authorization_bundle_never_mixes_newer_pending_run_sources(
    database: Database,
) -> None:
    source = {
        "component": "animegen",
        "repo_id": "supported/source",
        "requested_revision": "v1",
        "revision": "a" * 40,
        "license": "apache-2.0",
        "files": ({"relative_path": "config.json", "size": 1, "local_sha256": "1" * 64},),
    }
    minimum = base_manifest().model_copy(
        update={"width": 128, "height": 80, "num_frames": 17, "fps": 8}
    )
    standard = minimum.model_copy(update={"width": 512, "height": 288})
    minimum_hash = hashlib.sha256(canonical_json(minimum).encode()).hexdigest()
    results = {
        FeasibilityStage.DOWNLOAD_SOURCES: StageResult.passed(
            StageEvidence(details={"inventory": {"expected_bytes": 1, "sources": (source,)}})
        ),
        FeasibilityStage.GENERATE_MINIMUM_1: StageResult.passed(
            generation_evidence(manifest=minimum_hash)
        ),
        FeasibilityStage.GENERATE_MINIMUM_2: StageResult.passed(
            generation_evidence(manifest=minimum_hash)
        ),
    }
    instance, _ = service(database, Executor(results))
    supported_config = config()
    supported_config["manifest_profiles"] = {
        "minimum": minimum.model_dump(mode="json"),
        "standard": standard.model_dump(mode="json"),
    }
    supported = instance.start(supported_config)
    advance(instance, supported.id, STAGES.index("generate_minimum_2") + 1)
    assert instance.status(supported.id).verdict is SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID

    pending = instance.start(config(memory_strategy="newer-pending"))
    advance(instance, pending.id, STAGES.index("inventory_sources") + 1)
    bundle = instance.repository.generation_authorization_bundle(supported.adapter_id)

    assert bundle is not None
    assert bundle.run_id == supported.id
    assert bundle.resolved_sources["animegen"].repo_id == "supported/source"


def test_latest_authorized_profile_accepts_experimental_and_rejects_corruption(
    database: Database,
) -> None:
    minimum = base_manifest().model_copy(
        update={"width": 128, "height": 80, "num_frames": 17, "fps": 8}
    )
    minimum_hash = hashlib.sha256(canonical_json(minimum).encode()).hexdigest()
    results = {
        FeasibilityStage.GENERATE_MINIMUM_1: StageResult.passed(
            generation_evidence(manifest=minimum_hash)
        ),
        FeasibilityStage.GENERATE_MINIMUM_2: StageResult.passed(
            generation_evidence(manifest=minimum_hash)
        ),
    }
    instance, _ = service(database, Executor(results))
    configured = config()
    configured["manifest_profiles"]["minimum"] = minimum.model_dump(mode="json")
    run = instance.start(configured)
    advance(instance, run.id, STAGES.index("generate_minimum_2") + 1)

    assert instance.repository.latest_authorized_profile(run.adapter_id) == (
        SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID,
        (128, 80, 17),
    )

    corrupted = instance.repository.get_run(run.id).model_dump(mode="json")["configuration"]
    corrupted["manifest_profiles"]["minimum"]["width"] = 129
    with database.connect() as connection:
        connection.execute(
            "UPDATE feasibility_runs SET configuration_json = ? WHERE id = ?",
            (json.dumps(corrupted, sort_keys=True, separators=(",", ":")), run.id),
        )
    assert instance.repository.latest_authorized_profile(run.adapter_id) is None


def test_mismatched_manifest_or_fallback_is_invalid(database: Database) -> None:
    results = {
        FeasibilityStage.GENERATE_MINIMUM_1: StageResult.passed(generation_evidence()),
        FeasibilityStage.GENERATE_MINIMUM_2: StageResult.passed(
            StageEvidence(
                details={
                    **generation_evidence().model_dump(mode="json")["details"],
                    "canonical_manifest_hash": "other",
                    "fallback_violations": ["cpu"],
                }
            )
        ),
    }
    instance, _ = service(database, Executor(results))
    run = instance.start(config())
    advance(instance, run.id, 11)
    assert instance.status(run.id).verdict is SupportVerdict.INCOMPATIBLE


@pytest.mark.parametrize(
    ("error", "outcome", "code"),
    [
        (
            RuntimeError("MPS backend operator aten::foo is not implemented"),
            StageOutcome.INCOMPATIBLE,
            "mps_operator_unsupported",
        ),
        (
            RuntimeError("Undefined type Float8_e4m3fn"),
            StageOutcome.INCOMPATIBLE,
            "mps_operator_unsupported",
        ),
        (
            ResourceExhaustedError("allocation failed at /Volumes/cache/model"),
            StageOutcome.RESOURCE_EXHAUSTED,
            "resource_exhausted",
        ),
        (
            RuntimeError("MPS out of memory allocating 20 GB"),
            StageOutcome.RESOURCE_EXHAUSTED,
            "resource_exhausted",
        ),
        (
            SourceInventoryError("insufficient free space for frozen source inventory"),
            StageOutcome.RESOURCE_EXHAUSTED,
            "resource_exhausted",
        ),
        (KeyboardInterrupt(), StageOutcome.INTERRUPTED, "execution_cancelled"),
        (
            InvalidEvidenceError("checksum mismatch /private/tmp/model"),
            StageOutcome.INVALID_EVIDENCE,
            "checksum_mismatch",
        ),
        (
            ValueError("token=secret /Users/alice/.cache/huggingface/x"),
            StageOutcome.UNEXPECTED_FAILURE,
            "unexpected_failure",
        ),
    ],
)
def test_exception_classification_and_recursive_sanitization(
    error: BaseException, outcome: StageOutcome, code: str
) -> None:
    result = classify_exception(error)
    assert result.outcome is outcome
    assert result.diagnostic_code == code
    sanitized = sanitize_diagnostic(
        {"nested": [str(error), {"token": "secret", "path": "/Users/x"}]}
    )
    rendered = repr(sanitized)
    assert "/Users" not in rendered and "/private" not in rendered and "/Volumes" not in rendered
    assert "secret" not in rendered and ".cache" not in rendered


def test_sanitization_redacts_sensitive_keys_paths_with_spaces_and_tilde() -> None:
    sanitized = sanitize_diagnostic(
        {
            "/Volumes/private/token=abc": "keep this explanation",
            "nested": [
                "/Users/Alice Smith/Secret/file.txt failed safely",
                {
                    "api_key=abc": "~/Library/Application Support/Secret/file",
                    "message": "password is hunter2",
                },
            ],
        }
    )
    rendered = json.dumps(sanitized, sort_keys=True)
    for forbidden in (
        "/Volumes",
        "/Users",
        "Alice Smith",
        "Secret/file",
        "token=abc",
        "api_key=abc",
        "~/Library",
        "hunter2",
    ):
        assert forbidden not in rendered
    assert "keep this explanation" in rendered


@pytest.mark.parametrize(
    "credential",
    [
        "Authorization: Bearer opaque-value-123",
        "Bearer opaque-value-123",
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        "hf_abcdefghijklmnopqrstuvwxyz123456",
        "ghp_abcdefghijklmnopqrstuvwxyz123456",
        "github_pat_abcdefghijklmnopqrstuvwxyz123456",
    ],
)
def test_sanitization_redacts_unlabelled_credentials_in_nested_keys_and_values(
    credential: str,
) -> None:
    sanitized = sanitize_diagnostic(
        {"ordinary": "useful explanation", "nested": [{credential: f"failed with {credential}"}]}
    )
    rendered = json.dumps(sanitized, sort_keys=True)
    assert credential not in rendered
    assert "opaque-value-123" not in rendered
    assert "abcdefghijklmnopqrstuvwxyz123456" not in rendered
    assert "useful explanation" in rendered
