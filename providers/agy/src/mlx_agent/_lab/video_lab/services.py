from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from video_lab.artifacts import ArtifactStore
from video_lab.capabilities import (
    AdapterCapabilities,
    CapabilityViolation,
    validate_manifest_support,
)
from video_lab.comparison import ControlledPair, build_controlled_pair
from video_lab.database import Database
from video_lab.domain import ExperimentDraft, Job, JobState, Manifest
from video_lab.repositories import ExperimentRepository, JobRepository, StoredExperiment


class ValidationFailure(ValueError):
    def __init__(self, violations: tuple[CapabilityViolation, ...]) -> None:
        super().__init__("manifest is not supported by the selected adapter")
        self.violations = violations


@dataclass(frozen=True)
class ExperimentView:
    experiment: StoredExperiment
    jobs: tuple[Job, ...]


class ExperimentService:
    def __init__(
        self,
        experiments: ExperimentRepository,
        jobs: JobRepository,
        artifacts: ArtifactStore,
    ) -> None:
        self.experiments = experiments
        self.jobs = jobs
        self.artifacts = artifacts

    @classmethod
    def from_paths(cls, database_path: Path, artifact_root: Path) -> ExperimentService:
        database = Database(database_path)
        database.migrate()
        return cls(
            ExperimentRepository(database),
            JobRepository(database),
            ArtifactStore(database, artifact_root),
        )

    def create(self, draft: ExperimentDraft, capabilities: AdapterCapabilities) -> ExperimentView:
        pair = self.preview(draft, capabilities)
        experiment, jobs = self.experiments.create_with_jobs(pair)
        return ExperimentView(experiment=experiment, jobs=jobs)

    def create_adaptive(
        self,
        draft: ExperimentDraft,
        capabilities: AdapterCapabilities,
        *,
        study_id: str,
        optimizer_output_digest: str,
    ) -> ExperimentView:
        pair = self.preview(draft, capabilities)
        experiment, jobs = self.experiments.create_adaptive_with_jobs(
            pair,
            study_id=study_id,
            optimizer_output_digest=optimizer_output_digest,
        )
        return ExperimentView(experiment=experiment, jobs=jobs)

    def create_validation(
        self,
        draft: ExperimentDraft,
        capabilities: AdapterCapabilities,
        *,
        campaign_id: str,
        trial_id: str,
    ) -> ExperimentView:
        pair = self.preview(draft, capabilities)
        experiment, jobs = self.experiments.create_validation_with_jobs(
            pair, campaign_id=campaign_id, trial_id=trial_id
        )
        return ExperimentView(experiment=experiment, jobs=jobs)

    def preview(self, draft: ExperimentDraft, capabilities: AdapterCapabilities) -> ControlledPair:
        self._validate(draft.baseline, capabilities)
        pair = build_controlled_pair(draft)
        self._validate(pair.variant_b, capabilities)
        return pair

    def get(self, experiment_id: str) -> ExperimentView:
        experiment = self.experiments.get(experiment_id)
        return ExperimentView(
            experiment=experiment,
            jobs=self.jobs.list_for_experiment(experiment_id),
        )

    def get_adaptive(self, experiment_id: str) -> ExperimentView:
        experiment = self.experiments.get_adaptive(experiment_id)
        return ExperimentView(
            experiment=experiment,
            jobs=self.jobs.list_for_experiment(experiment_id),
        )

    def get_validation(self, experiment_id: str) -> ExperimentView:
        experiment = self.experiments.get_validation(experiment_id)
        return ExperimentView(
            experiment=experiment,
            jobs=self.jobs.list_for_experiment(experiment_id),
        )

    def list(self, limit: int = 50) -> tuple[ExperimentView, ...]:
        return tuple(self.get(experiment.id) for experiment in self.experiments.list(limit))

    def list_adaptive_for_study(self, study_id: str) -> tuple[ExperimentView, ...]:
        return tuple(
            self.get_adaptive(experiment.id)
            for experiment in self.experiments.list_adaptive_for_study(study_id)
        )

    def requeue_adaptive_failures(self, study_id: str, experiment_id: str) -> tuple[Job, ...]:
        experiment = self.get_adaptive(experiment_id)
        return self.jobs.requeue_failed_adaptive(
            study_id,
            experiment_id,
            tuple(job.id for job in experiment.jobs),
        )

    def transition_job(self, job_id: str, target: JobState) -> Job:
        return self.jobs.transition(job_id, target)

    @staticmethod
    def _validate(manifest: Manifest, capabilities: AdapterCapabilities) -> None:
        violations = validate_manifest_support(manifest, capabilities)
        if violations:
            raise ValidationFailure(violations)
