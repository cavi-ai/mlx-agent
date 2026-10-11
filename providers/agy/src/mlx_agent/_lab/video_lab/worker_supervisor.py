from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from video_lab.database import Database
from video_lab.domain import Job
from video_lab.repositories import JobRepository
from video_lab.worker_control import SupervisorBusy, SupervisorShutdownError

_DEFAULT_OUTPUT_LIMIT = 16 * 1024
_ALLOWED_ENVIRONMENT = ("PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
_READ_SIZE = 4096


class _Process(Protocol):
    stdout: BinaryIO | None
    stderr: BinaryIO | None
    returncode: int | None
    pid: int

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...


Launcher = Callable[..., _Process]
GroupSignaler = Callable[[int, int], None]


class _ByteTail:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._value = bytearray()

    def append(self, chunk: bytes) -> None:
        if len(chunk) >= self.limit:
            self._value[:] = chunk[-self.limit :]
            return
        overflow = len(self._value) + len(chunk) - self.limit
        if overflow > 0:
            del self._value[:overflow]
        self._value.extend(chunk)

    def bytes(self) -> bytes:
        return bytes(self._value)


class WorkerSupervisor:
    """Own at most one isolated worker subprocess for a local web application."""

    def __init__(
        self,
        *,
        database_path: Path | str,
        artifact_root: Path | str,
        scratch_root: Path | str,
        launcher: Launcher | None = None,
        environment: Mapping[str, str] | None = None,
        output_limit: int = _DEFAULT_OUTPUT_LIMIT,
        group_signaler: GroupSignaler | None = None,
        shutdown_timeout: float = 2.0,
    ) -> None:
        if output_limit < 1:
            raise ValueError("output_limit must be positive")
        if shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be positive")
        self.database_path = Path(database_path)
        self.artifact_root = Path(artifact_root)
        self.scratch_root = Path(scratch_root)
        self._launcher: Launcher = launcher or subprocess.Popen  # type: ignore[assignment]
        source_environment = os.environ if environment is None else environment
        self._environment = {
            key: source_environment[key]
            for key in _ALLOWED_ENVIRONMENT
            if key in source_environment
        }
        self._output_limit = output_limit
        self._group_signaler = group_signaler or _signal_process_group
        self._shutdown_timeout = shutdown_timeout
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._process: _Process | None = None
        self._process_group_id: int | None = None
        self._monitor: threading.Thread | None = None
        self._closed = False
        self._closing = False
        self._status: dict[str, object] = {"state": "idle"}
        self._refresh_persisted()

    def status(self) -> dict[str, object]:
        with self._lock:
            if self._process is None and not self._closed:
                self._refresh_persisted_locked()
            return dict(self._status)

    def run_next(self, worker_id: str) -> dict[str, object]:
        worker_id = worker_id.strip()
        if not worker_id:
            raise ValueError("worker_id must not be blank")
        with self._lock:
            if self._closed:
                raise RuntimeError("worker supervisor is closed")
            if self._closing:
                raise RuntimeError("worker supervisor is closing")
            self._refresh_persisted_locked()
            if self._status.get("state") == "recovery_required":
                raise SupervisorBusy("persisted worker recovery is required")
            if self._process is not None:
                raise SupervisorBusy("a worker process is already running")
            process = self._launcher(
                self._worker_argv(worker_id),
                shell=False,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=dict(self._environment),
            )
            self._process = process
            self._process_group_id = process.pid
            self._status = {"state": "running", "worker_id": worker_id}
            monitor = threading.Thread(
                target=self._collect,
                args=(process, worker_id),
                name="video-lab-worker-monitor",
                daemon=True,
            )
            self._monitor = monitor
            monitor.start()
            return dict(self._status)

    def wait(self, timeout: float | None = None) -> dict[str, object]:
        with self._lock:
            monitor = self._monitor
        if monitor is not None:
            monitor.join(timeout)
            if monitor.is_alive():
                raise TimeoutError("worker process is still running")
        return self.status()

    def cancel(self, job_id: str) -> Job:
        return self._repository().request_cancellation(job_id)

    def recover(self, worker_id: str) -> tuple[Job, ...]:
        worker_id = worker_id.strip()
        if not worker_id:
            raise ValueError("worker_id must not be blank")
        recovered = self._repository().recover_interrupted(worker_id)
        with self._lock:
            if self._process is None and not self._closed:
                self._refresh_persisted_locked()
        return recovered

    def close(self) -> None:
        with self._close_lock:
            with self._lock:
                if self._closed:
                    return
                self._closing = True
                process = self._process
                process_group_id = self._process_group_id
                monitor = self._monitor
                self._status = {"state": "shutting_down"}
            try:
                if (
                    process is not None
                    and process_group_id is not None
                    and (process.poll() is None or (monitor is not None and monitor.is_alive()))
                ):
                    self._signal_if_present(process_group_id, signal.SIGTERM)
                if monitor is not None:
                    monitor.join(self._shutdown_timeout)
                if monitor is not None and monitor.is_alive():
                    if process is None:
                        raise SupervisorShutdownError("worker monitor has no owned process")
                    if process_group_id is None:
                        raise SupervisorShutdownError("worker process group ownership is missing")
                    self._signal_if_present(process_group_id, signal.SIGKILL)
                    monitor.join(self._shutdown_timeout)
                if monitor is not None and monitor.is_alive():
                    raise SupervisorShutdownError("worker monitor did not stop after SIGKILL")
            except SupervisorShutdownError:
                with self._lock:
                    self._status = {"state": "shutdown_failed"}
                raise
            except Exception as error:
                with self._lock:
                    self._status = {"state": "shutdown_failed"}
                raise SupervisorShutdownError("worker shutdown failed") from error
            finally:
                with self._lock:
                    if self._status.get("state") == "shutdown_failed":
                        self._closing = True
            with self._lock:
                if (
                    self._process is not None
                    or self._process_group_id is not None
                    or self._monitor is not None
                ):
                    self._status = {"state": "shutdown_failed"}
                    raise SupervisorShutdownError("worker ownership remains after shutdown")
                self._closed = True
                self._closing = False
                self._status = {"state": "closed"}

    def _signal_if_present(self, pid: int, signal_number: int) -> None:
        try:
            self._group_signaler(pid, signal_number)
        except ProcessLookupError:
            return
        except OSError as error:
            if error.errno == errno.ESRCH:
                return
            raise

    def _repository(self) -> JobRepository:
        database = Database(self.database_path)
        database.migrate()
        return JobRepository(database)

    def _refresh_persisted(self) -> None:
        with self._lock:
            self._refresh_persisted_locked()

    def _refresh_persisted_locked(self) -> None:
        if self._process is not None or self._closed:
            return
        leases = self._repository().active_worker_leases()
        if leases:
            owners = {lease.worker_id for lease in leases}
            self._status = {
                "state": "recovery_required",
                "worker_id": next(iter(owners)) if len(owners) == 1 else "multiple",
                "job_ids": [lease.job_id for lease in leases],
            }
        elif self._status.get("state") == "recovery_required":
            self._status = {"state": "idle"}

    def _worker_argv(self, worker_id: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "video_lab.worker",
            "--database",
            str(self.database_path),
            "--artifacts",
            str(self.artifact_root),
            "--scratch",
            str(self.scratch_root),
            "--worker-id",
            worker_id,
        ]

    def _collect(self, process: _Process, worker_id: str) -> None:
        stdout_tail = _ByteTail(self._output_limit)
        stderr_tail = _ByteTail(self._output_limit)
        drains = [
            threading.Thread(target=_drain, args=(process.stdout, stdout_tail), daemon=True),
            threading.Thread(target=_drain, args=(process.stderr, stderr_tail), daemon=True),
        ]
        for drain in drains:
            drain.start()
        completed: dict[str, object] | None = None
        reaped = False
        try:
            exit_code = process.wait()
            reaped = True
            for drain in drains:
                drain.join()
            stdout = stdout_tail.bytes().decode("utf-8", errors="replace")
            stderr = stderr_tail.bytes().decode("utf-8", errors="replace")
            result = _last_json_object(stdout)
            state = "completed" if exit_code == 0 and result is not None else "failed"
            completed = {
                "state": state,
                "worker_id": worker_id,
                "exit_code": exit_code,
                "result": result,
                "stdout": stdout,
                "stderr": stderr,
            }
        finally:
            current = threading.current_thread()
            with self._lock:
                if self._process is process and reaped:
                    if completed is not None and not self._closing and not self._closed:
                        self._status = completed
                    self._process = None
                    self._process_group_id = None
                if self._monitor is current:
                    self._monitor = None


def _drain(stream: BinaryIO | None, tail: _ByteTail) -> None:
    if stream is None:
        return
    try:
        while chunk := stream.read(_READ_SIZE):
            tail.append(chunk)
    except ValueError:
        pass
    finally:
        stream.close()


def _signal_process_group(pid: int, signal_number: int) -> None:
    os.killpg(pid, signal_number)


def _last_json_object(output: str) -> dict[str, object] | None:
    for line in reversed(output.splitlines()):
        try:
            value: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("status"), str):
            return {str(key): item for key, item in value.items()}
    return None
