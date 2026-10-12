from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from video_lab.animegen_pair_contract import (
    ANIMEGEN_CONTROLLED_GUIDANCE_VALUES,
    ANIMEGEN_CONTROLLED_PAIR_SEED,
)
from video_lab.comparison import build_controlled_pair
from video_lab.domain import ExperimentDraft, Manifest, canonical_json, new_id, utc_now
from video_lab.feasibility_models import (
    FeasibilityConfiguration,
    FeasibilityRun,
    FeasibilityStage,
    StageAttempt,
    StageEvidence,
    StageOutcome,
    StageResult,
    SupportVerdict,
)
from video_lab.feasibility_repository import FeasibilityRepository
from video_lab.feasibility_telemetry import StageTelemetryCapture
from video_lab.normal_generation import CommittedGenerationEvidence

_PATH_START = re.compile(r"(?<!\w)(?:/Users|/private|/Volumes)/|(?<!\w)~/")
_CACHE = re.compile(r"\S*(?:\.cache|cache/)\S*", re.IGNORECASE)
_SECRET = re.compile(
    r"(?i)(?:token|secret|password|api[_-]?key)\b"
    r"(?:\s*(?:[=:]|\bis\b)?\s*[^\s,;]+|/[^\s,;]+)?"
)
_CREDENTIAL = re.compile(
    r"(?i)(?:authorization\s*:\s*)?bearer\s+[^\s,;]+"
    r"|sk-(?:proj-)?[A-Za-z0-9_-]{16,}"
    r"|hf_[A-Za-z0-9_-]{16,}"
    r"|ghp_[A-Za-z0-9_-]{16,}"
    r"|github_pat_[A-Za-z0-9_-]{16,}"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StageDefinition:
    stage: FeasibilityStage
    prerequisites: tuple[FeasibilityStage, ...] = ()


@dataclass(frozen=True)
class FeasibilityProtocol:
    stages: tuple[StageDefinition, ...]

    @classmethod
    def default(cls) -> FeasibilityProtocol:
        definitions: list[StageDefinition] = []
        previous: FeasibilityStage | None = None
        for stage in FeasibilityStage:
            definitions.append(StageDefinition(stage, () if previous is None else (previous,)))
            previous = stage
        return cls(tuple(definitions))


@dataclass(frozen=True)
class StageExecutionContext:
    run: FeasibilityRun
    attempts: tuple[StageAttempt, ...]


@dataclass(frozen=True)
class FeasibilityStatus:
    run: FeasibilityRun
    attempts: tuple[StageAttempt, ...]
    verdict: SupportVerdict


class StageExecutor(Protocol):
    def execute(self, context: StageExecutionContext, stage: StageDefinition) -> StageResult: ...


class NormalGenerationRunner(Protocol):
    """Queues and completes a manifest through the normal worker/artifact path."""

    def run_one(self, manifest: Manifest) -> CommittedGenerationEvidence: ...
    def run_pair(
        self,
        baseline: Manifest,
        candidate: Manifest,
        *,
        variable_path: str,
    ) -> tuple[CommittedGenerationEvidence, CommittedGenerationEvidence]: ...


class BlindReviewStarter(Protocol):
    """Creates the persisted, randomized review identity for a generated pair."""

    def start(self, experiment_id: str, reviewer_id: str) -> Any: ...


class GenerationQualificationExecutor:
    """Executes generation feasibility stages using persisted normal-path jobs."""

    def __init__(
        self,
        delegate: StageExecutor,
        runner: NormalGenerationRunner,
        reviews: BlindReviewStarter | None = None,
    ) -> None:
        self._delegate = delegate
        self._runner = runner
        self._reviews = reviews

    def preflight(self, context: StageExecutionContext, stage: StageDefinition) -> StageResult:
        preflight = getattr(self._delegate, "preflight", None)
        if not callable(preflight):
            return StageResult.passed()
        return cast(StageResult, preflight(context, stage))

    def execute(self, context: StageExecutionContext, stage: StageDefinition) -> StageResult:
        if stage.stage not in {
            FeasibilityStage.GENERATE_MINIMUM_1,
            FeasibilityStage.GENERATE_MINIMUM_2,
            FeasibilityStage.QUALIFY_STANDARD,
            FeasibilityStage.VALIDATE_CONTROLLED_PAIR,
        }:
            return self._delegate.execute(context, stage)
        profiles = context.run.configuration.get("manifest_profiles")
        if not isinstance(profiles, Mapping):
            raise InvalidEvidenceError("persisted manifest profiles are missing")
        if stage.stage is FeasibilityStage.VALIDATE_CONTROLLED_PAIR:
            standard = Manifest.model_validate(profiles.get("standard"))
            baseline = standard.model_copy(
                update={
                    "seed": ANIMEGEN_CONTROLLED_PAIR_SEED,
                    "guidance_scale": ANIMEGEN_CONTROLLED_GUIDANCE_VALUES[0],
                }
            )
            pair = build_controlled_pair(
                ExperimentDraft(
                    baseline=baseline,
                    variable_path="guidance_scale",
                    alternate_value=ANIMEGEN_CONTROLLED_GUIDANCE_VALUES[1],
                )
            )
            first_committed, second_committed = self._runner.run_pair(
                pair.variant_a, pair.variant_b, variable_path=pair.variable_path
            )
            experiment_id = first_committed.experiment_id
            if experiment_id is None or second_committed.experiment_id != experiment_id:
                raise InvalidEvidenceError("controlled pair has no shared persisted experiment")
            if self._reviews is None:
                raise InvalidEvidenceError("controlled pair has no persisted blind review service")
            review = self._reviews.start(experiment_id, f"feasibility:{context.run.id}")
            session_id = getattr(review, "session_id", None)
            if not isinstance(session_id, str) or not session_id:
                raise InvalidEvidenceError("controlled pair has no persisted blind review session")
            evidence = validate_controlled_pair_evidence(
                pair.variant_a.model_dump(mode="json"),
                pair.variant_b.model_dump(mode="json"),
                variable_path=pair.variable_path,
            )
            return StageResult.passed(
                StageEvidence(
                    details={
                        **evidence.model_dump(mode="json")["details"],
                        "job_ids": (
                            first_committed.job_id,
                            second_committed.job_id,
                        ),
                        "artifact_sha256": (
                            first_committed.artifact_sha256,
                            second_committed.artifact_sha256,
                        ),
                        "experiment_id": experiment_id,
                        "variable_path": pair.variable_path,
                        "pair_seed": ANIMEGEN_CONTROLLED_PAIR_SEED,
                        "controlled_values": (
                            pair.variant_a.guidance_scale,
                            pair.variant_b.guidance_scale,
                        ),
                        "review_identity": {
                            "experiment_id": experiment_id,
                            "session_id": session_id,
                            "state": "blinded_randomized",
                        },
                    }
                )
            )
        name = "standard" if stage.stage is FeasibilityStage.QUALIFY_STANDARD else "minimum"
        manifest = Manifest.model_validate(profiles.get(name))
        committed_evidence = self._runner.run_one(manifest)
        committed = committed_evidence.stage_details()
        manifest_hash = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
        repeatability: dict[str, object] = {}
        if stage.stage is FeasibilityStage.GENERATE_MINIMUM_2:
            first_attempt = next(
                (
                    attempt
                    for attempt in reversed(context.attempts)
                    if attempt.stage is FeasibilityStage.GENERATE_MINIMUM_1
                    and attempt.evidence is not None
                ),
                None,
            )
            if first_attempt is None or first_attempt.evidence is None:
                raise InvalidEvidenceError("first minimum generation evidence is missing")
            from video_lab.adapters.animegen_mps import classify_repeatability

            first_artifact = first_attempt.evidence.details.get("artifact")
            second_artifact = committed.get("artifact")
            if not isinstance(first_artifact, Mapping) or not isinstance(second_artifact, Mapping):
                raise InvalidEvidenceError("committed generation artifact evidence is missing")
            first_frames = first_artifact.get("decoded_frame_sha256")
            second_frames = second_artifact.get("decoded_frame_sha256")
            if not isinstance(first_frames, tuple) or not isinstance(second_frames, tuple):
                raise InvalidEvidenceError("decoded frame evidence is missing")
            classified = classify_repeatability(
                first_artifact_sha256=str(first_artifact.get("sha256", "")),
                second_artifact_sha256=str(second_artifact.get("sha256", "")),
                first_frame_sha256=cast(tuple[str, ...], first_frames),
                second_frame_sha256=cast(tuple[str, ...], second_frames),
                manifest_hash=manifest_hash,
                source_fingerprint=str(committed.get("source_fingerprint", "")),
                runtime_fingerprint=str(committed.get("runtime_fingerprint", "")),
                placement_fingerprint=str(committed.get("placement_fingerprint", "")),
                second_manifest_hash=manifest_hash,
                second_source_fingerprint=str(
                    first_attempt.evidence.details.get("source_fingerprint", "")
                ),
                second_runtime_fingerprint=str(
                    first_attempt.evidence.details.get("runtime_fingerprint", "")
                ),
                second_placement_fingerprint=str(
                    first_attempt.evidence.details.get("placement_fingerprint", "")
                ),
            )
            repeatability = {
                "repeatability": {
                    "classification": classified.classification,
                    "artifact_sha256": classified.artifact_sha256,
                    "decoded_frame_sha256": classified.decoded_frame_sha256,
                    "matching_frame_count": classified.matching_frame_count,
                    "differing_frame_count": classified.differing_frame_count,
                    "matching_frame_ratio": classified.matching_frame_ratio,
                }
            }
        return StageResult.passed(
            StageEvidence(
                details=cast(
                    Any,
                    {
                        **committed,
                        "canonical_manifest_hash": manifest_hash,
                        "profile": name,
                        **repeatability,
                    },
                )
            )
        )


class ResourceExhaustedError(RuntimeError):
    pass


class InvalidEvidenceError(RuntimeError):
    pass


def _canonical_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    validated = FeasibilityConfiguration.model_validate(configuration)
    return validated.model_dump(mode="json")


def canonical_fingerprint(configuration: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonical_configuration(configuration),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sanitize_text(value: str) -> str:
    path = _PATH_START.search(value)
    if path is not None:
        value = value[: path.start()] + "<path>"
    value = _CACHE.sub("<cache>", value)
    value = _CREDENTIAL.sub("<credential>", value)
    value = _SECRET.sub("<redacted>", value)
    return value[:1000]


def sanitize_diagnostic(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, Mapping):
        return {
            _sanitize_text(str(item_key)): sanitize_diagnostic(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_diagnostic(item) for item in value]
    return value


def classify_exception(error: BaseException) -> StageResult:
    message = _sanitize_text(str(error) or type(error).__name__)
    lowered = str(error).lower()
    if isinstance(error, (KeyboardInterrupt, InterruptedError)):
        return StageResult(
            outcome=StageOutcome.INTERRUPTED,
            diagnostic_code="execution_cancelled",
            diagnostic_message=message,
        )
    if isinstance(error, InvalidEvidenceError):
        code = "checksum_mismatch" if "checksum" in lowered else "invalid_evidence"
        return StageResult(
            outcome=StageOutcome.INVALID_EVIDENCE, diagnostic_code=code, diagnostic_message=message
        )
    if isinstance(error, ResourceExhaustedError) or any(
        term in lowered
        for term in (
            "out of memory",
            "allocation failed",
            "cannot allocate",
            "insufficient free space",
            "disk headroom",
            "unified-memory headroom",
        )
    ):
        return StageResult(
            outcome=StageOutcome.RESOURCE_EXHAUSTED,
            diagnostic_code="resource_exhausted",
            diagnostic_message=message,
        )
    if (
        "mps" in lowered
        and any(term in lowered for term in ("not implemented", "unsupported", "operator"))
    ) or "undefined type float8" in lowered:
        return StageResult(
            outcome=StageOutcome.INCOMPATIBLE,
            diagnostic_code="mps_operator_unsupported",
            diagnostic_message=message,
        )
    return StageResult(
        outcome=StageOutcome.UNEXPECTED_FAILURE,
        diagnostic_code="unexpected_failure",
        diagnostic_message=message,
    )


class FeasibilityService:
    def __init__(
        self,
        repository: FeasibilityRepository,
        executor: StageExecutor,
        protocol: FeasibilityProtocol | None = None,
    ) -> None:
        self.repository = repository
        self.executor = executor
        self.protocol = protocol or FeasibilityProtocol.default()

    def start(self, configuration: Mapping[str, Any]) -> FeasibilityRun:
        canonical = _canonical_configuration(configuration)
        fingerprint = canonical_fingerprint(canonical)
        with self.repository.database.connect() as connection:
            row = connection.execute(
                "SELECT id FROM feasibility_runs WHERE input_fingerprint = ? "
                "ORDER BY created_at LIMIT 1",
                (fingerprint,),
            ).fetchone()
        if row is not None:
            return self.repository.get_run(str(row["id"]))
        now = utc_now()
        run = FeasibilityRun(
            id=new_id("feasibility"),
            adapter_id=str(canonical["adapter_id"]),
            protocol_version=str(canonical["protocol_version"]),
            input_fingerprint=fingerprint,
            configuration=canonical,
            created_at=now,
            updated_at=now,
        )
        return self.repository.create_run(run)

    def status(self, run_id: str) -> FeasibilityStatus:
        run = self.repository.get_run(run_id)
        attempts = self.repository.list_attempts(run_id)
        verdict = derive_display_verdict(attempts)
        return FeasibilityStatus(run, attempts, verdict)

    def _persisted_status(self, run_id: str) -> FeasibilityStatus:
        status = self.status(run_id)
        if status.run.verdict is not status.verdict:
            try:
                self.repository.set_verdict(run_id, status.verdict)
            except ValueError:
                if status.verdict is not SupportVerdict.STANDARD_LAB_QUALIFIED:
                    raise
            status = self.status(run_id)
        return status

    def resume(self, run_id: str) -> FeasibilityStatus:
        self.repository.recover_interrupted(run_id)
        current = self.status(run_id)
        latest = _latest_attempts(current.attempts)
        for definition in self.protocol.stages:
            prior = latest.get(definition.stage)
            if prior is not None and prior.outcome is StageOutcome.PASSED:
                continue
            if prior is not None and prior.outcome not in (
                StageOutcome.INTERRUPTED,
                StageOutcome.UNEXPECTED_FAILURE,
            ):
                return self._persisted_status(run_id)
            if any(
                latest.get(required) is None or latest[required].outcome is not StageOutcome.PASSED
                for required in definition.prerequisites
            ):
                return self._persisted_status(run_id)
            active = self.repository.start_attempt(run_id, definition.stage)
            telemetry = StageTelemetryCapture.start(definition.stage)
            try:
                context = StageExecutionContext(current.run, current.attempts)
                preflight = getattr(self.executor, "preflight", None)
                preflight_result = (
                    cast(StageResult, preflight(context, definition))
                    if callable(preflight)
                    and definition.stage
                    not in {
                        FeasibilityStage.RESOLVE_SOURCES,
                        FeasibilityStage.INVENTORY_SOURCES,
                        FeasibilityStage.STORAGE_PREFLIGHT,
                        FeasibilityStage.PUBLISH_VERDICT,
                    }
                    else StageResult.passed()
                )
                if preflight_result.outcome is not StageOutcome.PASSED:
                    result = preflight_result
                else:
                    executed = self.executor.execute(context, definition)
                    result = _merge_stage_evidence(preflight_result, executed)
            except BaseException as error:
                result = classify_exception(error)
            result = result.model_copy(update={"evidence": telemetry.merge(result.evidence)})
            result = _sanitize_result(result)
            self.repository.finish_attempt(
                active.id,
                result.outcome,
                result.evidence,
                result.diagnostic_code,
                result.diagnostic_message,
            )
            return self._persisted_status(run_id)
        return self._persisted_status(run_id)


def _merge_stage_evidence(preflight: StageResult, executed: StageResult) -> StageResult:
    measurements = {
        **preflight.evidence.measurements,
        **executed.evidence.measurements,
    }
    details = {
        **preflight.evidence.model_dump(mode="json")["details"],
        **executed.evidence.model_dump(mode="json")["details"],
    }
    return executed.model_copy(
        update={"evidence": StageEvidence(measurements=measurements, details=details)}
    )


def _sanitize_result(result: StageResult) -> StageResult:
    message = (
        None if result.diagnostic_message is None else _sanitize_text(result.diagnostic_message)
    )
    details = sanitize_diagnostic(result.evidence.model_dump(mode="json")["details"])
    evidence = StageEvidence(measurements=result.evidence.measurements, details=details)
    return result.model_copy(update={"evidence": evidence, "diagnostic_message": message})


def _latest_attempts(attempts: tuple[StageAttempt, ...]) -> dict[FeasibilityStage, StageAttempt]:
    latest: dict[FeasibilityStage, StageAttempt] = {}
    for attempt in attempts:
        if attempt.stage not in latest or attempt.ordinal > latest[attempt.stage].ordinal:
            latest[attempt.stage] = attempt
    return latest


def _valid_generation(
    attempt: StageAttempt | None, manifest: str | None = None
) -> tuple[bool, str | None]:
    if attempt is None or attempt.outcome is not StageOutcome.PASSED or attempt.evidence is None:
        return False, manifest
    details = attempt.evidence.details
    artifact = details.get("artifact")
    frame_hashes = artifact.get("decoded_frame_sha256") if isinstance(artifact, Mapping) else None
    duration = artifact.get("duration_seconds") if isinstance(artifact, Mapping) else None
    frame_count = artifact.get("num_frames") if isinstance(artifact, Mapping) else None
    valid_artifact = (
        isinstance(artifact, Mapping)
        and artifact.get("format") == "mp4"
        and artifact.get("valid") is True
        and artifact.get("codec") in ("h264", "avc1")
        and isinstance(duration, int | float)
        and float(duration) > 0
        and isinstance(artifact.get("sha256"), str)
        and _SHA256.fullmatch(str(artifact["sha256"])) is not None
        and isinstance(frame_hashes, Sequence)
        and not isinstance(frame_hashes, str | bytes | bytearray)
        and len(frame_hashes) == artifact.get("num_frames")
        and all(isinstance(item, str) and _SHA256.fullmatch(item) for item in frame_hashes)
        and isinstance(artifact.get("width"), int)
        and isinstance(artifact.get("height"), int)
        and isinstance(frame_count, int)
        and isinstance(artifact.get("fps"), int)
        and frame_count >= 17
        and (frame_count - 1) % 16 == 0
    )
    value = details.get("canonical_manifest_hash")
    valid_manifest = (
        isinstance(value, str)
        and _SHA256.fullmatch(value) is not None
        and (manifest is None or value == manifest)
    )
    fingerprints = tuple(
        details.get(key)
        for key in ("source_fingerprint", "runtime_fingerprint", "placement_fingerprint")
    )
    valid = (
        valid_artifact
        and valid_manifest
        and all(isinstance(item, str) and _SHA256.fullmatch(item) for item in fingerprints)
        and details.get("placement") == "local_hybrid"
        and details.get("fallback_violations") in ((), [])
    )
    return valid, value if isinstance(value, str) else manifest


def derive_display_verdict(
    attempts: tuple[StageAttempt, ...] | list[StageAttempt],
) -> SupportVerdict:
    """Project stage metadata for display; never grants generation authorization."""
    latest = _latest_attempts(tuple(attempts))
    outcomes = {attempt.outcome for attempt in latest.values()}
    if StageOutcome.INVALID_EVIDENCE in outcomes:
        return SupportVerdict.INCOMPATIBLE
    if StageOutcome.INCOMPATIBLE in outcomes:
        return SupportVerdict.INCOMPATIBLE
    if StageOutcome.RESOURCE_EXHAUSTED in outcomes:
        return SupportVerdict.RESOURCE_LIMITED
    first, manifest = _valid_generation(latest.get(FeasibilityStage.GENERATE_MINIMUM_1))
    second, _ = _valid_generation(latest.get(FeasibilityStage.GENERATE_MINIMUM_2), manifest)
    if not first or not second:
        if latest.get(FeasibilityStage.GENERATE_MINIMUM_2) is not None:
            return SupportVerdict.INCOMPATIBLE
        return SupportVerdict.PENDING
    standard_attempt = latest.get(FeasibilityStage.QUALIFY_STANDARD)
    standard, _ = _valid_generation(standard_attempt)
    standard_artifact = (
        standard_attempt.evidence.details.get("artifact")
        if standard_attempt is not None and standard_attempt.evidence is not None
        else None
    )
    standard = (
        standard
        and isinstance(standard_artifact, Mapping)
        and (
            standard_artifact.get("width"),
            standard_artifact.get("height"),
            standard_artifact.get("num_frames"),
            standard_artifact.get("fps"),
        )
        == (512, 288, 17, 8)
    )
    minimum_details = latest[FeasibilityStage.GENERATE_MINIMUM_1].evidence
    standard_details = standard_attempt.evidence if standard_attempt is not None else None
    if minimum_details is None or standard_details is None:
        standard = False
    else:
        identity_keys = (
            "source_fingerprint",
            "runtime_fingerprint",
            "placement_fingerprint",
        )
        standard = standard and all(
            minimum_details.details.get(key) == standard_details.details.get(key)
            for key in identity_keys
        )
    pair = latest.get(FeasibilityStage.VALIDATE_CONTROLLED_PAIR)
    changed_paths = (
        pair.evidence.details.get("changed_paths")
        if pair is not None and pair.evidence is not None
        else None
    )
    variable_path = (
        pair.evidence.details.get("variable_path")
        if pair is not None and pair.evidence is not None
        else None
    )
    controlled_values = (
        pair.evidence.details.get("controlled_values")
        if pair is not None and pair.evidence is not None
        else None
    )
    pair_seed = (
        pair.evidence.details.get("pair_seed")
        if pair is not None and pair.evidence is not None
        else None
    )
    if (
        standard
        and pair is not None
        and pair.outcome is StageOutcome.PASSED
        and pair.evidence is not None
        and pair.evidence.details.get("valid") is True
        and isinstance(changed_paths, tuple)
        and changed_paths == ("guidance_scale",)
        and variable_path == "guidance_scale"
        and pair_seed == ANIMEGEN_CONTROLLED_PAIR_SEED
        and controlled_values == ANIMEGEN_CONTROLLED_GUIDANCE_VALUES
    ):
        return SupportVerdict.STANDARD_LAB_QUALIFIED
    return SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID


def validate_controlled_pair_evidence(
    baseline: Mapping[str, Any],
    alternate: Mapping[str, Any],
    *,
    variable_path: str,
) -> StageEvidence:
    keys = set(baseline) | set(alternate)
    changed = tuple(sorted(key for key in keys if baseline.get(key) != alternate.get(key)))
    if changed != (variable_path,):
        raise InvalidEvidenceError(f"controlled pair must differ only in {variable_path}")
    return StageEvidence(details={"valid": True, "changed_paths": changed})


def build_generation_feasibility_service(
    *,
    repository: FeasibilityRepository,
    delegate: StageExecutor,
    experiments: Any,
    jobs: Any,
    artifacts: Any,
    worker: Any,
) -> FeasibilityService:
    """Product composition root for resumable generation qualification stages."""
    from video_lab.normal_generation import RepositoryWorkerGenerationRunner
    from video_lab.review_service import ReviewService
    from video_lab.reviews import ReviewRepository

    runner = RepositoryWorkerGenerationRunner(
        experiments=experiments,
        jobs=jobs,
        artifacts=artifacts,
        worker=worker,
    )
    reviews = ReviewService(
        experiments=experiments,
        jobs=jobs,
        artifacts=artifacts,
        reviews=ReviewRepository(repository.database),
    )
    return FeasibilityService(
        repository,
        GenerationQualificationExecutor(delegate, runner, reviews),
    )


__all__ = [
    "FeasibilityProtocol",
    "FeasibilityService",
    "InvalidEvidenceError",
    "ResourceExhaustedError",
    "StageDefinition",
    "StageExecutionContext",
    "StageResult",
    "canonical_fingerprint",
    "build_generation_feasibility_service",
    "classify_exception",
    "derive_display_verdict",
    "sanitize_diagnostic",
    "validate_controlled_pair_evidence",
]
