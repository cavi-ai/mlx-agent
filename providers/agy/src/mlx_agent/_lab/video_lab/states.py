from video_lab.domain import JobState


class InvalidTransition(ValueError):
    pass


ALLOWED: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.PREPARING, JobState.CANCELLED, JobState.FAILED}),
    JobState.PREPARING: frozenset(
        {JobState.DOWNLOADING, JobState.LOADING, JobState.CANCELLED, JobState.FAILED}
    ),
    JobState.DOWNLOADING: frozenset({JobState.LOADING, JobState.CANCELLED, JobState.FAILED}),
    JobState.LOADING: frozenset({JobState.GENERATING, JobState.CANCELLED, JobState.FAILED}),
    JobState.GENERATING: frozenset({JobState.ENCODING, JobState.CANCELLED, JobState.FAILED}),
    JobState.ENCODING: frozenset({JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED}),
    JobState.COMPLETED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
}


def transition_job(current: JobState, target: JobState) -> JobState:
    if target not in ALLOWED[current]:
        raise InvalidTransition(f"cannot transition job from {current} to {target}")
    return target
