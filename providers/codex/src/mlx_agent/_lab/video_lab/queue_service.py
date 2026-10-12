from __future__ import annotations

from collections.abc import Callable
from typing import Any

from video_lab.artifacts import ArtifactStore
from video_lab.domain import FrozenModel, JobState, RunEvent, TelemetrySample
from video_lab.repositories import ExperimentRepository, JobRepository, StoredExperiment
from video_lab.review_service import ReviewService

_EVIDENCE_LIMIT = 20
_TEXT_LIMIT = 240
_ACTIVE_STATES = {
    state.value
    for state in JobState
    if state not in {JobState.QUEUED, JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}
}


class EvidenceView(FrozenModel):
    sequence: int
    kind: str
    phase: str | None = None
    message: str | None = None


class TelemetrySummary(FrozenModel):
    elapsed_seconds: float
    process_rss_bytes: int
    system_used_bytes: int
    system_available_bytes: int
    swap_used_bytes: int
    thermal_pressure: str | None


class ArtifactView(FrozenModel):
    id: str
    media_type: str
    byte_size: int


class JobProgressView(FrozenModel):
    id: str
    variant_label: str
    state: str
    updated_at: str
    attempt_count: int
    recovery_required: bool
    progress: dict[str, object] | None
    evidence: tuple[EvidenceView, ...]
    telemetry: TelemetrySummary | None
    artifacts: tuple[ArtifactView, ...]
    can_cancel: bool


class ExperimentQueueView(FrozenModel):
    id: str
    variable_label: str
    reviewable: bool
    reviewability_reasons: tuple[str, ...]
    review_url: str | None
    jobs: tuple[JobProgressView, ...]


class QueueSnapshot(FrozenModel):
    schema_version: str = "queue.v1"
    experiments: tuple[ExperimentQueueView, ...]
    worker: dict[str, object]


class QueueService:
    def __init__(
        self,
        experiments: ExperimentRepository,
        jobs: JobRepository,
        artifacts: ArtifactStore,
        reviews: ReviewService,
        *,
        worker_status: Callable[[], dict[str, object]] | None = None,
    ) -> None:
        self.experiments = experiments
        self.jobs = jobs
        self.artifacts = artifacts
        self.reviews = reviews
        self.worker_status = worker_status

    def snapshot(self) -> QueueSnapshot:
        eligibility = {item.experiment_id: item for item in self.reviews.queue()}
        experiments = tuple(
            self._experiment(experiment, eligibility[experiment.id].reason_codes)
            for experiment in self.experiments.list()
        )
        return QueueSnapshot(experiments=experiments, worker=self._worker())

    def _worker(self) -> dict[str, object]:
        if self.worker_status is None:
            return {"state": "unavailable"}
        status = self.worker_status()
        return {
            key: status[key]
            for key in ("state", "worker_id", "job_ids", "exit_code")
            if key in status
        }

    def _experiment(
        self, experiment: StoredExperiment, reasons: tuple[str, ...]
    ) -> ExperimentQueueView:
        baseline = experiment.variants[0].manifest
        alternate = experiment.variants[1].manifest
        left = _value_at(baseline.model_dump(mode="json"), experiment.variable_path)
        right = _value_at(alternate.model_dump(mode="json"), experiment.variable_path)
        labels = {variant.id: variant.label for variant in experiment.variants}
        jobs = tuple(
            self._job(job.id, labels[job.variant_id])
            for job in self.jobs.list_for_experiment(experiment.id)
        )
        reviewable = not reasons
        return ExperimentQueueView(
            id=experiment.id,
            variable_label=f"{experiment.variable_path}: {left} -> {right}",
            reviewable=reviewable,
            reviewability_reasons=reasons,
            review_url="/api/reviews" if reviewable else None,
            jobs=jobs,
        )

    def _job(self, job_id: str, variant_label: str) -> JobProgressView:
        job = self.jobs.get(job_id)
        attempts = self.jobs.attempts_for_job(job.id)
        events = self.jobs.events_for_job(job.id)
        latest_attempt = attempts[-1] if attempts else None
        telemetry = (
            self.jobs.telemetry_for_attempt(latest_attempt.id) if latest_attempt is not None else ()
        )
        progress = _latest_progress(events)
        return JobProgressView(
            id=job.id,
            variant_label=variant_label,
            state=job.state.value,
            updated_at=job.updated_at.isoformat(),
            attempt_count=len(attempts),
            recovery_required=bool(
                latest_attempt
                and latest_attempt.status.value == "active"
                and job.state.value in _ACTIVE_STATES
            ),
            progress=progress,
            evidence=tuple(_evidence(event) for event in events[-_EVIDENCE_LIMIT:]),
            telemetry=_telemetry(telemetry[-1]) if telemetry else None,
            artifacts=tuple(
                ArtifactView(id=item.id, media_type=item.media_type, byte_size=item.byte_size)
                for item in self.artifacts.list_for_job(job.id)
            ),
            can_cancel=job.state.value == "queued" or job.state.value in _ACTIVE_STATES,
        )


def _value_at(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for component in path.split("."):
        value = value[int(component)] if isinstance(value, list) else value[component]
    return value


def _latest_progress(events: tuple[RunEvent, ...]) -> dict[str, object] | None:
    for event in reversed(events):
        if event.kind == "progress":
            return {
                key: event.payload[key]
                for key in ("phase", "step", "total")
                if key in event.payload
            }
    return None


def _evidence(event: RunEvent) -> EvidenceView:
    phase = event.payload.get("phase")
    message = event.payload.get("message")
    safe_message = None
    if isinstance(message, str):
        safe_message = message.rsplit("/", maxsplit=1)[-1][:_TEXT_LIMIT]
    return EvidenceView(
        sequence=event.id,
        kind=event.kind[:80],
        phase=phase[:80] if isinstance(phase, str) else None,
        message=safe_message,
    )


def _telemetry(sample: TelemetrySample) -> TelemetrySummary:
    return TelemetrySummary(
        elapsed_seconds=sample.elapsed_seconds,
        process_rss_bytes=sample.process_rss_bytes,
        system_used_bytes=sample.system_used_bytes,
        system_available_bytes=sample.system_available_bytes,
        swap_used_bytes=sample.swap_used_bytes,
        thermal_pressure=sample.thermal_pressure,
    )
