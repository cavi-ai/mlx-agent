from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from video_lab.database import Database
from video_lab.worker_control import SupervisorShutdownError
from video_lab.worker_supervisor import SupervisorBusy, WorkerSupervisor


class FakeProcess:
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self.returncode: int | None = None
        self.release = threading.Event()
        self.stdout = BytesIO(stdout.encode())
        self.stderr = BytesIO(stderr.encode())
        self.pid = 4242

    def communicate(self) -> tuple[str, str]:
        self.release.wait(timeout=2)
        self.returncode = self._returncode
        return self._stdout, self._stderr

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if not self.release.wait(timeout):
            raise subprocess.TimeoutExpired("worker", timeout)
        self.returncode = self._returncode
        return self._returncode


class StreamProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> None:
        self.stdout = BytesIO(stdout)
        self.stderr = BytesIO(stderr)
        self.returncode: int | None = None
        self._final_returncode = returncode
        self.pid = 4242
        self.release = threading.Event()
        self.terminated = threading.Event()
        self.wait_calls = 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if not self.release.wait(timeout):
            raise subprocess.TimeoutExpired("worker", timeout)
        self.returncode = self._final_returncode
        return self._final_returncode

    def poll(self) -> int | None:
        return self.returncode


def _supervisor(tmp_path: Path, launcher: Any) -> WorkerSupervisor:
    database = Database(tmp_path / "lab.sqlite")
    database.migrate()
    return WorkerSupervisor(
        database_path=database.path,
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        launcher=launcher,
    )


def test_launch_is_explicit_isolated_and_environment_is_allowlisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    process = FakeProcess(stdout='{"status":"idle"}\n')

    def launcher(argv: list[str], **kwargs: object) -> FakeProcess:
        calls.append((argv, kwargs))
        return process

    monkeypatch.setenv("VIDEO_LAB_SECRET", "must-not-leak")
    supervisor = _supervisor(tmp_path, launcher)
    started = supervisor.run_next("worker-one")

    argv, options = calls[0]
    assert argv == [
        sys.executable,
        "-m",
        "video_lab.worker",
        "--database",
        str(tmp_path / "lab.sqlite"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--scratch",
        str(tmp_path / "scratch"),
        "--worker-id",
        "worker-one",
    ]
    assert options["shell"] is False
    assert options["start_new_session"] is True
    assert options["stdout"] is subprocess.PIPE
    assert options["stderr"] is subprocess.PIPE
    assert options["text"] is False
    assert "VIDEO_LAB_SECRET" not in options["env"]  # type: ignore[operator]
    assert started == {"state": "running", "worker_id": "worker-one"}
    process.release.set()


def test_duplicate_launch_is_rejected_thread_safely_and_completion_cleans_up(
    tmp_path: Path,
) -> None:
    process = FakeProcess(stdout='{"status":"completed","job_id":"job-1"}\n')
    replacement = FakeProcess(stdout='{"status":"idle"}\n')
    processes = iter((process, replacement))
    supervisor = _supervisor(tmp_path, lambda _argv, **_kwargs: next(processes))
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def launch() -> None:
        barrier.wait()
        try:
            supervisor.run_next("worker-one")
            outcomes.append("started")
        except SupervisorBusy:
            outcomes.append("busy")

    threads = [threading.Thread(target=launch) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["busy", "started"]
    process.release.set()
    assert supervisor.wait(timeout=2)["state"] == "completed"
    assert supervisor.status()["result"] == {"status": "completed", "job_id": "job-1"}
    assert supervisor._monitor is None
    assert supervisor._process_group_id is None
    assert supervisor.run_next("worker-two") == {"state": "running", "worker_id": "worker-two"}
    assert supervisor._process_group_id == replacement.pid
    replacement.release.set()
    supervisor.wait(timeout=2)
    assert supervisor._process_group_id is None


def test_output_is_bounded_and_abnormal_exit_is_reported(tmp_path: Path) -> None:
    process = FakeProcess(stdout="x" * 10_000, stderr="y" * 10_000, returncode=9)
    supervisor = WorkerSupervisor(
        database_path=tmp_path / "lab.sqlite",
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        launcher=lambda _argv, **_kwargs: process,
        output_limit=128,
    )
    supervisor.run_next("worker-one")
    process.release.set()

    status = supervisor.wait(timeout=2)
    assert status["state"] == "failed"
    assert status["exit_code"] == 9
    assert len(status["stdout"]) == 128
    assert len(status["stderr"]) == 128
    assert status["result"] is None


def test_cancel_and_recover_delegate_repository_semantics(database, tmp_path: Path) -> None:
    from lab_tests.test_run_repository import _create_one_queued_job

    repository, job_id = _create_one_queued_job(database)
    claimed = repository.claim_next("worker-one")
    assert claimed is not None
    supervisor = WorkerSupervisor(
        database_path=database.path,
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
    )

    cancelled = supervisor.cancel(job_id)
    assert cancelled.state.value == "cancelled"
    recovered = supervisor.recover("worker-one")
    assert recovered == ()


def test_noisy_streams_are_incrementally_drained_into_bounded_tails(tmp_path: Path) -> None:
    script = (
        "import sys; "
        "sys.stdout.buffer.write(b'x' * 2_000_000); sys.stdout.buffer.flush(); "
        "sys.stderr.buffer.write(b'y' * 2_000_000); sys.stderr.buffer.flush(); "
        'sys.stdout.buffer.write(b\'\\n{"status":"completed","job_id":"job-1"}\\n\')'
    )

    def launch_noisy(_argv: list[str], **options: Any):
        return subprocess.Popen([sys.executable, "-c", script], **options)

    supervisor = _supervisor(tmp_path, launch_noisy)

    supervisor.run_next("worker-one")
    status = supervisor.wait(timeout=5)

    assert len(status["stdout"]) <= 16 * 1024
    assert len(status["stderr"]) <= 16 * 1024
    assert status["result"] == {"status": "completed", "job_id": "job-1"}


def test_close_is_idempotent_blocks_launches_and_terminates_process_group(tmp_path: Path) -> None:
    process = StreamProcess(b"")
    signals: list[tuple[int, int]] = []

    def signal_group(pid: int, signal_number: int) -> None:
        signals.append((pid, signal_number))
        process.release.set()

    supervisor = WorkerSupervisor(
        database_path=tmp_path / "lab.sqlite",
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        launcher=lambda _argv, **_kwargs: process,
        group_signaler=signal_group,
    )
    supervisor.run_next("worker-one")

    threads = [threading.Thread(target=supervisor.close) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert signals
    assert supervisor.status()["state"] == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        supervisor.run_next("worker-two")


def test_close_escalates_to_process_group_kill_after_bounded_wait(tmp_path: Path) -> None:
    process = StreamProcess(b"")
    signals: list[int] = []

    def signal_group(_pid: int, signal_number: int) -> None:
        signals.append(signal_number)
        if signal_number == signal.SIGKILL:
            process.release.set()

    supervisor = WorkerSupervisor(
        database_path=tmp_path / "lab.sqlite",
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        launcher=lambda _argv, **_kwargs: process,
        group_signaler=signal_group,
        shutdown_timeout=0.01,
    )
    supervisor.run_next("worker-one")

    supervisor.close()

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert supervisor.status() == {"state": "closed"}


def test_close_treats_process_lookup_race_as_exited_and_waits_for_reap(tmp_path: Path) -> None:
    process = StreamProcess(b"")

    def process_disappeared(_pid: int, _signal_number: int) -> None:
        process.release.set()
        raise ProcessLookupError

    supervisor = WorkerSupervisor(
        database_path=tmp_path / "lab.sqlite",
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        launcher=lambda _argv, **_kwargs: process,
        group_signaler=process_disappeared,
        shutdown_timeout=0.1,
    )
    supervisor.run_next("worker-one")

    supervisor.close()

    assert process.wait_calls == 1
    assert supervisor.status() == {"state": "closed"}


def test_close_retains_ownership_when_monitor_survives_kill_and_allows_retry(
    tmp_path: Path,
) -> None:
    process = StreamProcess(b"")
    supervisor = WorkerSupervisor(
        database_path=tmp_path / "lab.sqlite",
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        launcher=lambda _argv, **_kwargs: process,
        group_signaler=lambda _pid, _signal_number: None,
        shutdown_timeout=0.01,
    )
    supervisor.run_next("worker-one")

    with pytest.raises(SupervisorShutdownError):
        supervisor.close()

    assert supervisor.status()["state"] == "shutdown_failed"
    assert supervisor._process is process
    assert supervisor._process_group_id == process.pid
    assert supervisor._monitor is not None and supervisor._monitor.is_alive()
    with pytest.raises(RuntimeError, match="closing"):
        supervisor.run_next("worker-two")

    process.release.set()
    supervisor.close()
    assert supervisor.status() == {"state": "closed"}
    assert supervisor._process_group_id is None


def test_close_kills_stored_process_group_after_leader_exit_with_inherited_pipes(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "child-ready"
    child_script = (
        "import os,signal,time,pathlib; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    script = (
        "import subprocess,sys,time,pathlib; "
        f"ready=pathlib.Path({str(ready)!r}); "
        f"child=subprocess.Popen([sys.executable,'-c',{child_script!r}]); "
        "\nwhile not ready.exists(): time.sleep(0.01)"
    )

    def launch_group(_argv: list[str], **options: Any):
        return subprocess.Popen([sys.executable, "-c", script], **options)

    supervisor = WorkerSupervisor(
        database_path=tmp_path / "lab.sqlite",
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        launcher=launch_group,
        shutdown_timeout=0.05,
    )
    supervisor.run_next("worker-one")
    process = supervisor._process
    assert process is not None
    pgid = process.pid
    try:
        deadline = time.monotonic() + 5
        while (not ready.exists() or process.poll() is None) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        assert process.poll() == 0
        assert supervisor._monitor is not None and supervisor._monitor.is_alive()

        supervisor.close()

        assert supervisor.status() == {"state": "closed"}
        assert supervisor._process is None
        assert supervisor._monitor is None
        assert supervisor._process_group_id is None
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_persisted_active_claim_requires_recovery_after_supervisor_restart(
    database, tmp_path: Path
) -> None:
    from lab_tests.test_run_repository import _create_one_queued_job

    repository, job_id = _create_one_queued_job(database)
    assert repository.claim_next("stable-worker") is not None
    supervisor = WorkerSupervisor(
        database_path=database.path,
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
    )

    assert supervisor.status() == {
        "state": "recovery_required",
        "worker_id": "stable-worker",
        "job_ids": [job_id],
    }
    with pytest.raises(SupervisorBusy, match="recovery"):
        supervisor.run_next("stable-worker")
    assert [job.id for job in supervisor.recover("stable-worker")] == [job_id]
    assert supervisor.status() == {"state": "idle"}
