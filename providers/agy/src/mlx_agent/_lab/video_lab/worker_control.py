from __future__ import annotations

from typing import Protocol

from video_lab.domain import Job


class SupervisorBusy(RuntimeError):
    """Raised when persisted or in-memory worker state blocks a launch."""


class SupervisorShutdownError(RuntimeError):
    """Raised when an owned worker cannot be reaped within shutdown bounds."""


class WorkerControl(Protocol):
    def status(self) -> dict[str, object]: ...
    def run_next(self, worker_id: str) -> dict[str, object]: ...
    def cancel(self, job_id: str) -> Job: ...
    def recover(self, worker_id: str) -> tuple[Job, ...]: ...
    def close(self) -> None: ...
