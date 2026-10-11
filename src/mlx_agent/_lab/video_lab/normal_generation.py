from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from video_lab.artifacts import ArtifactStore
from video_lab.comparison import build_controlled_pair
from video_lab.domain import ExperimentDraft, JobState, Manifest
from video_lab.repositories import ExperimentRepository, JobRepository
from video_lab.runtime_identity import runtime_fingerprint
from video_lab.worker import Worker


@dataclass(frozen=True)
class CommittedGenerationEvidence:
    job_id: str
    artifact_sha256: str
    artifact: dict[str, object]
    source_fingerprint: str
    runtime_fingerprint: str
    placement_fingerprint: str
    placement: str
    fallback_violations: tuple[str, ...] = ()
    experiment_id: str | None = None

    def stage_details(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "artifact_sha256": self.artifact_sha256,
            "artifact": self.artifact,
            "source_fingerprint": self.source_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "placement_fingerprint": self.placement_fingerprint,
            "placement": self.placement,
            "fallback_violations": self.fallback_violations,
        }


class RepositoryWorkerGenerationRunner:
    """Runs manifests through persisted experiment/jobs, Worker, and ArtifactStore."""

    def __init__(
        self,
        *,
        experiments: ExperimentRepository,
        jobs: JobRepository,
        artifacts: ArtifactStore,
        worker: Worker,
    ) -> None:
        self._experiments = experiments
        self._jobs = jobs
        self._artifacts = artifacts
        self._worker = worker

    def run_one(self, manifest: Manifest) -> CommittedGenerationEvidence:
        job = self._experiments.create_single_generation(manifest)
        outcome = self._worker.run_once(job.id)
        if outcome.status != "completed" or outcome.job_id != job.id:
            raise RuntimeError(
                f"normal generation job failed: {outcome.error_code}: "
                f"{self._failure_detail(job.id)}"
            )
        return self._load(job.id, manifest)

    def run_pair(
        self,
        baseline: Manifest,
        candidate: Manifest,
        *,
        variable_path: str,
    ) -> tuple[CommittedGenerationEvidence, CommittedGenerationEvidence]:
        pair = build_controlled_pair(
            ExperimentDraft(
                baseline=baseline,
                variable_path=variable_path,
                alternate_value=_value_at_path(candidate, variable_path),
            )
        )
        if pair.variant_b != candidate:
            raise ValueError(f"controlled pair differs outside {variable_path}")
        experiment, jobs = self._experiments.create_with_jobs(pair)
        for expected in jobs:
            outcome = self._worker.run_once(expected.id)
            if outcome.status != "completed" or outcome.job_id != expected.id:
                raise RuntimeError(
                    f"normal generation job failed: {outcome.error_code}: "
                    f"{self._failure_detail(expected.id)}"
                )
        return tuple(
            self._load(
                job.id,
                self._jobs.manifest_for_job(job.id),
                experiment_id=experiment.id,
            )
            for job in jobs
        )  # type: ignore[return-value]

    def _failure_detail(self, job_id: str) -> str:
        events = self._jobs.events_for_job(job_id)
        message = None if not events else events[-1].payload.get("message")
        return str(message or "failure detail unavailable")

    def _load(
        self, job_id: str, manifest: Manifest, *, experiment_id: str | None = None
    ) -> CommittedGenerationEvidence:
        if self._jobs.manifest_for_job(job_id) != manifest:
            raise RuntimeError("committed job manifest differs from request")
        if self._jobs.get(job_id).state is not JobState.COMPLETED:
            raise RuntimeError("normal generation job did not complete")
        artifacts = self._artifacts.list_for_job(job_id)
        if len(artifacts) != 1 or artifacts[0].media_type != "video/mp4":
            raise RuntimeError("normal generation job lacks one committed MP4")
        completed = self._jobs.events_for_job(job_id)[-1]
        if completed.kind != "job_completed":
            raise RuntimeError("normal generation job lacks completion evidence")
        payload = completed.payload
        telemetry = payload.get("adapter_telemetry")
        if not isinstance(telemetry, Mapping):
            raise RuntimeError("normal generation job lacks adapter telemetry")
        observed = telemetry.get("observed_configuration")
        if not isinstance(observed, Mapping):
            raise RuntimeError("normal generation job lacks observed configuration")
        decoded = payload.get("decoded_frame_sha256")
        if not isinstance(decoded, Sequence) or isinstance(decoded, str | bytes | bytearray):
            raise RuntimeError("normal generation job lacks decoded frame evidence")
        decoded_hashes = tuple(str(value) for value in decoded)
        artifact = artifacts[0]
        return CommittedGenerationEvidence(
            job_id=job_id,
            artifact_sha256=artifact.sha256,
            artifact={
                "format": "mp4",
                "valid": True,
                "sha256": artifact.sha256,
                "decoded_frame_sha256": decoded_hashes,
                "codec": payload.get("codec", "h264"),
                "duration_seconds": payload.get("duration_seconds", 0.0),
                "width": payload["width"],
                "height": payload["height"],
                "num_frames": payload["frame_count"],
                "fps": payload["fps"],
            },
            source_fingerprint=str(observed["source_fingerprint"]),
            runtime_fingerprint=runtime_fingerprint(
                cast(Mapping[str, str], payload["runtime_versions"])
            ),
            placement_fingerprint=str(observed["placement_fingerprint"]),
            placement="local_hybrid",
            fallback_violations=tuple(cast(list[str], telemetry.get("fallback_violations", []))),
            experiment_id=experiment_id,
        )


def _value_at_path(manifest: Manifest, variable_path: str) -> object:
    value: object = manifest.model_dump(mode="json")
    for part in variable_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"unknown variable path: {variable_path}")
        value = value[part]
    return value
