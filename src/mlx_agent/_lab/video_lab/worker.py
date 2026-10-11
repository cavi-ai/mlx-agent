from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from time import monotonic
from typing import Literal

import imageio_ffmpeg  # type: ignore[import-untyped]

from video_lab.adapter_protocol import (
    AdapterExecutionError,
    GenerationAdapter,
    GenerationCancelled,
    GenerationProgress,
    GenerationRequest,
    GenerationResult,
)
from video_lab.adapter_registry import AdapterRegistry, UnknownAdapterError
from video_lab.artifacts import ArtifactStore
from video_lab.capabilities import validate_manifest_support
from video_lab.database import Database
from video_lab.domain import Job, JobState, RunAttemptStatus
from video_lab.feasibility_repository import FeasibilityRepository
from video_lab.repositories import JobRepository
from video_lab.telemetry import TelemetrySampler

WorkerStatus = Literal["idle", "completed", "failed", "cancelled"]


@dataclass(frozen=True)
class WorkerOutcome:
    status: WorkerStatus
    job_id: str | None = None
    attempt_id: str | None = None
    error_code: str | None = None


class InvalidGenerationOutput(ValueError):
    """Raised when an adapter result is not the promised MP4."""


class UnsupportedManifest(ValueError):
    """Raised when an exact adapter cannot execute the stored manifest."""


class WorkerConfigurationError(RuntimeError):
    """Raised when a required local worker dependency is unavailable."""


class Worker:
    def __init__(
        self,
        *,
        repository: JobRepository,
        artifact_store: ArtifactStore,
        adapter_registry: AdapterRegistry,
        scratch_root: Path,
        worker_id: str,
        telemetry_sampler: TelemetrySampler | None = None,
        ffprobe_resolver: Callable[[], str | None] | None = None,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be blank")
        self.repository = repository
        self.artifact_store = artifact_store
        self.adapter_registry = adapter_registry
        self.scratch_root = scratch_root
        self.worker_id = worker_id
        self.telemetry_sampler = telemetry_sampler or TelemetrySampler()
        self.ffprobe_resolver = ffprobe_resolver or (lambda: shutil.which("ffprobe"))
        self._ffprobe_executable: str | None = None
        self._adapters: dict[str, GenerationAdapter] = {}

    def recover(self) -> tuple[Job, ...]:
        return self.repository.recover_interrupted(self.worker_id)

    def run_until_idle(self, max_jobs: int | None = None) -> tuple[WorkerOutcome, ...]:
        if max_jobs is not None and max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
        outcomes: list[WorkerOutcome] = []
        while max_jobs is None or len(outcomes) < max_jobs:
            outcome = self.run_once()
            if outcome.status == "idle":
                return tuple(outcomes)
            outcomes.append(outcome)
            if outcome.error_code == "worker_configuration_error":
                return tuple(outcomes)
        return tuple(outcomes)

    def run_once(self, job_id: str | None = None) -> WorkerOutcome:
        claimed = self.repository.claim_next(self.worker_id, job_id)
        if claimed is None:
            return WorkerOutcome(status="idle")

        job_id = claimed.job.id
        attempt_id = claimed.attempt.id
        started = monotonic()
        try:
            self.repository.append_event(
                job_id,
                attempt_id,
                "worker_process_started",
                {"worker_id": self.worker_id, "process_id": os.getpid()},
            )
            self._sample_best_effort(job_id, attempt_id, started)
            self._ffprobe_executable = self.ffprobe_resolver()
            if self._ffprobe_executable is None:
                raise WorkerConfigurationError("ffprobe is required on PATH")
            manifest = self.repository.manifest_for_job(job_id)
            adapter = self._adapters.get(manifest.adapter_id)
            if adapter is None:
                adapter = self.adapter_registry.resolve(manifest.adapter_id)
                self._adapters[manifest.adapter_id] = adapter
            violations = validate_manifest_support(manifest, adapter.capabilities())
            if violations:
                summary = "; ".join(
                    f"{violation.field}: {violation.reason}" for violation in violations
                )
                raise UnsupportedManifest(summary)
            adapter_version = getattr(adapter, "adapter_version", None)
            if adapter_version != manifest.adapter_version:
                raise UnsupportedManifest(
                    f"adapter version mismatch: expected {manifest.adapter_version}, "
                    f"got {adapter_version}"
                )
            if self._cancelled(job_id):
                raise GenerationCancelled("cancellation requested before generation")

            request = GenerationRequest(
                job_id=job_id,
                attempt_id=attempt_id,
                manifest=manifest,
                scratch_root=self.scratch_root / attempt_id,
            )

            def progress(update: GenerationProgress) -> None:
                if self._cancelled(job_id):
                    raise GenerationCancelled("cancellation requested")
                self._advance_for_progress(job_id, update.phase)
                self.repository.append_event(
                    job_id,
                    attempt_id,
                    "generation_progress",
                    {
                        "phase": update.phase,
                        "completed": update.completed,
                        "total": update.total,
                        "message": update.message,
                        "elapsed_seconds": update.elapsed_seconds,
                    },
                )
                self._sample_best_effort(job_id, attempt_id, started)

            result = adapter.generate(request, progress, lambda: self._cancelled(job_id))
            if result.effective_manifest != manifest:
                raise InvalidGenerationOutput("effective manifest differs from requested manifest")
            if result.adapter_version != manifest.adapter_version:
                raise InvalidGenerationOutput("result adapter version differs from manifest")
            self._ensure_generating(job_id)
            self.repository.transition(job_id, JobState.ENCODING)
            self.repository.append_event(
                job_id,
                attempt_id,
                "encoding_started",
                {"video_path": result.video_path.name},
            )
            self._sample_best_effort(job_id, attempt_id, started)
            media_evidence = self._verify_result(result, request.scratch_root)
            if self._cancelled(job_id):
                raise GenerationCancelled("cancellation requested before artifact registration")
            staged = self.artifact_store.stage(result.video_path, "video/mp4")
            if self._cancelled(job_id):
                staged.path.unlink(missing_ok=True)
                raise GenerationCancelled("cancellation requested before artifact registration")
            artifact = self.artifact_store.commit(staged, None)
            self._sample_best_effort(job_id, attempt_id, started)
            self.repository.finalize_attempt(
                job_id,
                attempt_id,
                job_state=JobState.COMPLETED,
                attempt_status=RunAttemptStatus.COMPLETED,
                artifact_id=artifact.id,
                error_code=None,
                error_message=None,
                event_kind="job_completed",
                event_payload={
                    "artifact_id": artifact.id,
                    "artifact_sha256": artifact.sha256,
                    "decoded_frame_sha256": media_evidence["decoded_frame_sha256"],
                    "decoded_rgb_aggregate_sha256": media_evidence["decoded_rgb_aggregate_sha256"],
                    "codec": media_evidence["codec"],
                    "duration_seconds": media_evidence["duration_seconds"],
                    "frame_count": result.frame_count,
                    "fps": result.fps,
                    "width": result.width,
                    "height": result.height,
                    "runtime_versions": dict(result.runtime_versions),
                    "adapter_telemetry": result.model_dump(mode="json")["adapter_telemetry"],
                },
            )
            return WorkerOutcome("completed", job_id, attempt_id)
        except GenerationCancelled as error:
            return self._finish_error(
                job_id,
                attempt_id,
                started,
                status="cancelled",
                error_code="cancelled",
                error=error,
            )
        except UnknownAdapterError as error:
            return self._finish_error(
                job_id,
                attempt_id,
                started,
                status="failed",
                error_code="adapter_not_found",
                error=error,
            )
        except UnsupportedManifest as error:
            return self._finish_error(
                job_id,
                attempt_id,
                started,
                status="failed",
                error_code="unsupported_manifest",
                error=error,
            )
        except WorkerConfigurationError as error:
            return self._finish_error(
                job_id,
                attempt_id,
                started,
                status="failed",
                error_code="worker_configuration_error",
                error=error,
            )
        except AdapterExecutionError as error:
            return self._finish_error(
                job_id,
                attempt_id,
                started,
                status="failed",
                error_code="adapter_execution_failed",
                error=error,
            )
        except InvalidGenerationOutput as error:
            return self._finish_error(
                job_id,
                attempt_id,
                started,
                status="failed",
                error_code="invalid_output",
                error=error,
            )
        except Exception as error:
            return self._finish_error(
                job_id,
                attempt_id,
                started,
                status="failed",
                error_code="unexpected_error",
                error=error,
            )

    def _finish_error(
        self,
        job_id: str,
        attempt_id: str,
        started: float,
        *,
        status: Literal["failed", "cancelled"],
        error_code: str,
        error: Exception,
    ) -> WorkerOutcome:
        self._sample_best_effort(job_id, attempt_id, started)
        cancelled = status == "cancelled" or self.repository.get(job_id).state is JobState.CANCELLED
        terminal_state = JobState.CANCELLED if cancelled else JobState.FAILED
        attempt_status = RunAttemptStatus.CANCELLED if cancelled else RunAttemptStatus.FAILED
        message = str(error) or error.__class__.__name__
        self.repository.finalize_attempt(
            job_id,
            attempt_id,
            job_state=terminal_state,
            attempt_status=attempt_status,
            artifact_id=None,
            error_code="cancelled" if cancelled else error_code,
            error_message=message,
            event_kind="job_cancelled" if cancelled else "job_failed",
            event_payload={
                "error_code": "cancelled" if cancelled else error_code,
                "message": message,
            },
        )
        outcome_status: Literal["failed", "cancelled"] = "cancelled" if cancelled else "failed"
        return WorkerOutcome(
            outcome_status, job_id, attempt_id, "cancelled" if cancelled else error_code
        )

    def _sample_best_effort(self, job_id: str, attempt_id: str, started: float) -> None:
        try:
            self.repository.append_telemetry(
                self.telemetry_sampler.sample(attempt_id, max(0.0, monotonic() - started))
            )
        except Exception as error:
            try:
                self.repository.append_event(
                    job_id,
                    attempt_id,
                    "telemetry_error",
                    {"error_type": error.__class__.__name__, "message": str(error)},
                )
            except Exception:
                pass

    def _cancelled(self, job_id: str) -> bool:
        return self.repository.get(job_id).state is JobState.CANCELLED

    def _advance_for_progress(self, job_id: str, phase: str) -> None:
        state = self.repository.get(job_id).state
        if phase == "downloading" and state is JobState.PREPARING:
            self.repository.transition(job_id, JobState.DOWNLOADING)
        elif phase == "loading":
            if state is JobState.PREPARING:
                self.repository.transition(job_id, JobState.LOADING)
            elif state is JobState.DOWNLOADING:
                self.repository.transition(job_id, JobState.LOADING)
        elif phase == "generating":
            self._ensure_generating(job_id)

    def _ensure_generating(self, job_id: str) -> None:
        state = self.repository.get(job_id).state
        if state is JobState.PREPARING:
            state = self.repository.transition(job_id, JobState.LOADING).state
        elif state is JobState.DOWNLOADING:
            state = self.repository.transition(job_id, JobState.LOADING).state
        if state is JobState.LOADING:
            self.repository.transition(job_id, JobState.GENERATING)
        elif state is not JobState.GENERATING:
            raise InvalidGenerationOutput(f"cannot enter generation from {state}")

    def _verify_result(self, result: GenerationResult, scratch_root: Path) -> dict[str, object]:
        if result.video_path.suffix.lower() != ".mp4" or result.video_path.is_symlink():
            raise InvalidGenerationOutput("adapter output must be a regular .mp4 file")
        try:
            root = scratch_root.resolve(strict=True)
            path = result.video_path.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise InvalidGenerationOutput("adapter output is missing") from error
        if not path.is_relative_to(root) or path.suffix.lower() != ".mp4" or not path.is_file():
            raise InvalidGenerationOutput("adapter output escapes attempt scratch root")
        if path.stat().st_size == 0:
            raise InvalidGenerationOutput("adapter output is missing or empty")
        try:
            ffprobe = self._ffprobe_executable
            if ffprobe is None:
                raise WorkerConfigurationError("ffprobe is required on PATH")
            probe = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=format_name,duration:format_tags=major_brand:"
                    "stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames",
                    "-of",
                    "json",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(probe.stdout)
            format_names = set(str(payload["format"]["format_name"]).lower().split(","))
            if not format_names.intersection({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}):
                raise InvalidGenerationOutput("adapter output container is not MP4")
            duration = float(payload["format"].get("duration", 0.0))
            if not math.isfinite(duration) or duration <= 0:
                raise InvalidGenerationOutput("adapter output duration must be positive")
            streams = [
                stream
                for stream in payload.get("streams", [])
                if stream.get("codec_type") == "video"
            ]
            if len(streams) != 1:
                raise InvalidGenerationOutput("adapter output must contain one video stream")
            stream = streams[0]
            if str(stream.get("codec_name", "")).lower() not in {"h264", "avc1"}:
                raise InvalidGenerationOutput("adapter output video codec must be H.264")
            if (int(stream["width"]), int(stream["height"])) != (result.width, result.height):
                raise InvalidGenerationOutput("output stream dimensions do not match")
            reported_frames = str(stream.get("nb_frames", ""))
            if not math.isclose(
                float(Fraction(stream["avg_frame_rate"])), result.fps, abs_tol=0.01
            ):
                raise InvalidGenerationOutput("output stream fps does not match")
            reader = imageio_ffmpeg.read_frames(path)
            metadata = next(reader)
            decoded_hashes = tuple(hashlib.sha256(frame).hexdigest() for frame in reader)
            reader.close()
            frame_count, _seconds = imageio_ffmpeg.count_frames_and_secs(path)
        except InvalidGenerationOutput:
            raise
        except Exception as error:
            raise InvalidGenerationOutput("adapter output is not a readable MP4") from error
        codec = str(metadata.get("codec", ""))
        size = metadata.get("size")
        fps = float(metadata.get("fps", 0.0))
        if not codec or size != (result.width, result.height):
            raise InvalidGenerationOutput("output dimensions or codec metadata do not match")
        if frame_count != result.frame_count or not math.isclose(fps, result.fps, abs_tol=0.01):
            raise InvalidGenerationOutput("output frame count or fps does not match")
        if reported_frames not in {"", "N/A"} and int(reported_frames) != len(decoded_hashes):
            raise InvalidGenerationOutput("output stream frame count does not match")
        if len(decoded_hashes) != result.frame_count:
            raise InvalidGenerationOutput("decoded frame count does not match")
        return {
            "decoded_frame_sha256": decoded_hashes,
            "decoded_rgb_aggregate_sha256": hashlib.sha256(
                "".join(decoded_hashes).encode("ascii")
            ).hexdigest(),
            "duration_seconds": duration,
            "codec": str(stream.get("codec_name", "")).lower(),
        }


def _nonblank_worker_id(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("worker ID must not be blank")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="video-lab-worker",
        description=(
            "Run queued jobs. The default processes one job; --until-idle keeps the worker "
            "and adapter alive. --worker-id must remain stable across restarts and be unique "
            "among simultaneous workers."
        ),
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--animegen-cache", type=Path)
    parser.add_argument("--until-idle", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument(
        "--worker-id",
        required=True,
        type=_nonblank_worker_id,
        help="stable restart identity, unique among simultaneously running workers",
    )
    args = parser.parse_args(argv)
    if args.max_jobs is not None and not args.until_idle:
        parser.error("--max-jobs requires --until-idle")
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max-jobs must be positive")
    database = Database(args.database)
    database.migrate()
    artifact_store = ArtifactStore(database, args.artifacts)
    feasibility = FeasibilityRepository(database, artifact_store)
    animegen_id = "diffusers.animegen-t2v.wan2.2-a14b.mps.v1"
    authorization_bundle = feasibility.generation_authorization_bundle(animegen_id)
    runtime_factory = None
    if args.animegen_cache is not None and authorization_bundle is not None:

        def runtime_factory(bundle: object) -> object:
            from video_lab.adapters.animegen_mps import DiffusersAnimeGenGenerationRuntime
            from video_lab.feasibility_repository import GenerationAuthorizationBundle

            if not isinstance(bundle, GenerationAuthorizationBundle):
                raise TypeError("AnimeGen authorization bundle is required")

            return DiffusersAnimeGenGenerationRuntime(
                cache_root=args.animegen_cache, sources=bundle.resolved_sources
            )

    worker = Worker(
        repository=JobRepository(database),
        artifact_store=artifact_store,
        adapter_registry=AdapterRegistry.with_defaults(
            feasibility_repository=feasibility,
            animegen_runtime_factory=runtime_factory,
        ),
        scratch_root=args.scratch,
        worker_id=args.worker_id,
    )
    worker.recover()
    if args.until_idle:
        outcomes = worker.run_until_idle(max_jobs=args.max_jobs)
        counts = {
            status: sum(outcome.status == status for outcome in outcomes)
            for status in ("completed", "failed", "cancelled")
        }
        status = "completed" if counts["failed"] == counts["cancelled"] == 0 else "failed"
        print(
            json.dumps(
                {
                    "status": status,
                    "processed": len(outcomes),
                    **counts,
                    "outcomes": [outcome.__dict__ for outcome in outcomes],
                },
                separators=(",", ":"),
            )
        )
        return 0 if status == "completed" else 1
    outcome = worker.run_once()
    print(json.dumps(outcome.__dict__, separators=(",", ":")))
    return 0 if outcome.status in {"idle", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
