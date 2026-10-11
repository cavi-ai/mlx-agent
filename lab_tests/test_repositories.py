from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from lab_tests.test_capabilities import manifest
from lab_tests.test_validation_repository import definition, event, panel, seed_stage_1
from video_lab.comparison import build_controlled_pair
from video_lab.database import Database
from video_lab.domain import ExperimentDraft, JobState
from video_lab.repositories import (
    ExperimentRepository,
    JobRepository,
    RecordNotFound,
    ValidationExperimentConflict,
    validation_experiment_id,
)
from video_lab.states import InvalidTransition
from video_lab.validation_models import ValidationCampaignState
from video_lab.validation_repository import ValidationRepository


def test_round_trips_experiment_and_queued_jobs(database: Database) -> None:
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    experiments = ExperimentRepository(database)
    jobs = JobRepository(database)
    stored = experiments.create(pair)
    created_jobs = jobs.create_for_experiment(stored.id)

    assert experiments.get(stored.id) == stored
    assert experiments.list() == (stored,)
    assert [job.state for job in created_jobs] == [JobState.QUEUED, JobState.QUEUED]
    assert jobs.list_for_experiment(stored.id) == created_jobs


def test_list_omits_single_variant_qualification_records(database: Database) -> None:
    experiments = ExperimentRepository(database)
    experiments.create_single_generation(manifest())
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    stored = experiments.create(pair)

    assert experiments.list() == (stored,)


def test_transitions_job_through_state_machine(database: Database) -> None:
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    stored = ExperimentRepository(database).create(pair)
    jobs = JobRepository(database)
    job = jobs.create_for_experiment(stored.id)[0]

    transitioned = jobs.transition(job.id, JobState.PREPARING)

    assert transitioned.state is JobState.PREPARING
    assert jobs.get(job.id) == transitioned
    with pytest.raises(InvalidTransition):
        jobs.transition(job.id, JobState.COMPLETED)


def test_missing_records_raise_typed_error(database: Database) -> None:
    with pytest.raises(RecordNotFound):
        ExperimentRepository(database).get("exp_missing")
    with pytest.raises(RecordNotFound):
        JobRepository(database).get("job_missing")
    with pytest.raises(RecordNotFound):
        JobRepository(database).create_for_experiment("exp_missing")


def test_validation_experiment_creation_is_deterministic_idempotent_and_sealed(database) -> None:
    seed_stage_1(database)
    campaigns = ValidationRepository(database)
    campaign = campaigns.create_or_get_bound(definition("validation_campaign"), reviewer_id="owner")
    validation_panel = panel(campaign)
    campaigns.preregister_or_verify(campaign.id, validation_panel)
    campaigns.append_event_or_verify(
        event(
            campaign.id,
            2,
            ValidationCampaignState.PREREGISTERED,
            ValidationCampaignState.GENERATING,
        )
    )
    trial_id = next(item.id for item in validation_panel.trials if not item.hidden_repeat)
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="guidance_scale", alternate_value=2.0)
    )
    experiments = ExperimentRepository(database)
    barrier = Barrier(2)

    def create():
        barrier.wait()
        return ExperimentRepository(Database(database.path)).create_validation_with_jobs(
            pair, campaign_id=campaign.id, trial_id=trial_id
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        created = tuple(executor.map(lambda _: create(), range(2)))
    (first, first_jobs), (second, second_jobs) = created

    assert first.id == validation_experiment_id(campaign.id, trial_id)
    assert (first, first_jobs) == (second, second_jobs)
    assert len(first_jobs) == 2
    assert experiments.list() == ()
    with database.connect() as connection:
        binding = connection.execute(
            "SELECT campaign_id, trial_id, experiment_id "
            "FROM validation_experiment_bindings WHERE experiment_id=?",
            (first.id,),
        ).fetchone()
    assert tuple(binding) == (campaign.id, trial_id, first.id)
    missing_id = validation_experiment_id(campaign.id, "missing_trial")
    with pytest.raises(ValidationExperimentConflict):
        experiments.create_validation_with_jobs(
            pair, campaign_id=campaign.id, trial_id="missing_trial"
        )
    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM experiments WHERE id=?", (missing_id,)
            ).fetchone()[0]
            == 0
        )
    with pytest.raises(RecordNotFound):
        experiments.get(first.id)
