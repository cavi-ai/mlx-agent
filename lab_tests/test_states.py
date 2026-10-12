import pytest

from video_lab.domain import JobState
from video_lab.states import InvalidTransition, transition_job


def test_allows_forward_progress_and_terminal_failure() -> None:
    assert transition_job(JobState.QUEUED, JobState.PREPARING) is JobState.PREPARING
    assert transition_job(JobState.GENERATING, JobState.FAILED) is JobState.FAILED


def test_rejects_transition_out_of_terminal_state() -> None:
    with pytest.raises(InvalidTransition):
        transition_job(JobState.COMPLETED, JobState.GENERATING)
