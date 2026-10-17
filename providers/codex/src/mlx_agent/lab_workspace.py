"""Native and agent projections over the same experiment services."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import uuid
import sys

from video_lab.adapter_registry import AdapterRegistry
from video_lab.artifacts import ArtifactStore, ArtifactError
from video_lab.database import Database
from video_lab.domain import DisplayPreference, ExperimentDraft, RubricScore
from video_lab.evidence_service import EvidenceService
from video_lab.feasibility_repository import FeasibilityRepository
from video_lab.repositories import ExperimentRepository, JobRepository
from video_lab.review_service import ReviewService
from video_lab.reviews import ReviewRepository
from video_lab.services import ExperimentService
from video_lab.worker import Worker
from mlx_agent.compute import ComputeLease
from video_lab.local_authority import RetainedDirectoryAuthority

_FIELDS = {
    "recipe-catalog": set(), "recipe-prepare": {"settings"},
    "adopt-preview": {"source_database", "source_artifacts", "destination"},
    "adopt": {"source_database", "source_artifacts", "destination", "preview_hash"},
    "study-list": set(), "study-create": {"manifest"},
    "study-show": {"study_id"}, "study-resume": {"study_id"}, "study-run": {"study_id"},
    "study-submit": {"study_id", "session_id", "preference"},
    "training-preview": {"receipt", "prompt", "max_tokens", "prompt_suite"},
    "training-evaluate": {"receipt", "prompt", "max_tokens", "prompt_suite", "preview_hash"},
    "training-review": {"evaluation_id", "artifact_sha256", "prompt_id", "preference", "notes"},
    "training-history": set(), "training-inspect": {"evaluation_id"},
    "snapshot": set(), "preview": {"draft"}, "create": {"draft"},
    "inspect": {"experiment_id"}, "run": {"job_id", "worker_id"},
    "cancel": {"job_id"}, "recover": {"worker_id"},
    "review-start": {"experiment_id"}, "review-media": {"session_id"},
    "review-submit": {"session_id", "preference", "scores", "notes"},
    "reveal": {"session_id"},
}
_MAX_REQUEST = 1024 * 1024

def validate_request(request):
    if not isinstance(request, dict) or request.get("operation") not in _FIELDS:
        raise ValueError("Unknown workspace operation.")
    operation = request["operation"]
    if set(request) - (_FIELDS[operation] | {"operation"}):
        raise ValueError("Unexpected workspace fields.")
    for field in ("experiment_id", "job_id", "worker_id", "session_id", "study_id", "evaluation_id", "prompt_id"):
        if field in request and (not isinstance(request[field], str) or not request[field].strip()
                                 or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", request[field])):
            raise ValueError("Invalid " + field)
    required = {
        "recipe-prepare": {"settings"},
        "adopt-preview": {"source_database", "source_artifacts", "destination"},
        "adopt": {"source_database", "source_artifacts", "destination", "preview_hash"},
        "study-create": {"manifest"},
        "study-show": {"study_id"}, "study-resume": {"study_id"}, "study-run": {"study_id"},
        "study-submit": {"study_id", "session_id", "preference"},
        "training-inspect": {"evaluation_id"},
        "training-preview": {"receipt"},
        "training-evaluate": {"receipt", "preview_hash"},
        "training-review": {"evaluation_id", "artifact_sha256", "prompt_id", "preference"},
        "inspect": {"experiment_id"}, "cancel": {"job_id"}, "run": {"job_id"},
        "review-start": {"experiment_id"}, "review-media": {"session_id"},
        "review-submit": {"session_id", "preference", "scores"}, "reveal": {"session_id"},
    }.get(operation, set())
    if not required <= set(request):
        raise ValueError("Missing workspace fields.")
    if operation in ("training-preview", "training-evaluate"):
        from mlx_agent.training_evaluation import validate_suite, _prompt
        if "prompt_suite" in request:
            if "prompt" in request or "max_tokens" in request:
                raise ValueError("Choose either a prompt suite or a single prompt.")
            validate_suite(request["prompt_suite"])
        else:
            _prompt(request.get("prompt"), request.get("max_tokens", 128))
    for field in ("source_database", "source_artifacts", "destination", "preview_hash"):
        if field in request and (not isinstance(request[field], str) or not request[field]
                or len(request[field]) > 4096 or any(ord(c) < 32 for c in request[field])):
            raise ValueError("Invalid " + field)
    if operation in ("preview", "create"):
        ExperimentDraft.model_validate(request.get("draft"))
    if operation == "recipe-prepare":
        from mlx_agent.study_recipes import validate_settings
        validate_settings(request["settings"])
    if operation == "review-submit":
        DisplayPreference(request.get("preference"))
        for item in request.get("scores", []):
            RubricScore.model_validate(item)
    return request

def execute(database_path, artifact_root, reviewer, request):
    request = validate_request(request)
    if not reviewer.strip():
        raise ValueError("A reviewer identity is required.")
    operation = request["operation"]
    if operation.startswith("recipe-"):
        from mlx_agent import study_recipes
        if operation == "recipe-catalog":
            return {"recipes": study_recipes.catalog()}
        return {"manifest": study_recipes.prepare(request["settings"])}
    if operation in ("adopt-preview", "adopt"):
        from mlx_agent.lab_adoption import preview, adopt
        arguments = [request["source_database"], request["source_artifacts"], request["destination"]]
        return preview(*arguments) if operation == "adopt-preview" else adopt(*arguments, request["preview_hash"])
    service = ExperimentService.from_paths(database_path, artifact_root)
    database = service.experiments.database
    reviews = ReviewService(service.experiments, service.jobs, service.artifacts, ReviewRepository(database))
    registry = AdapterRegistry.with_defaults(feasibility_repository=FeasibilityRepository(database))
    evidence = EvidenceService(database, service.artifacts)
    if operation.startswith("study-"):
        from mlx_agent.lab_studies import execute as study_execute
        return study_execute(service, reviews, registry, reviewer, request, lambda: Worker(
            repository=service.jobs, artifact_store=service.artifacts, adapter_registry=registry,
            scratch_root=artifact_root / ".scratch", worker_id="native-study-" + reviewer))
    if operation.startswith("training-"):
        from mlx_agent import training_evaluation
        if operation == "training-preview":
            return {"plan": training_evaluation.preview(request["receipt"], request.get("prompt"), request.get("max_tokens", 128), request.get("prompt_suite"))}
        if operation == "training-review":
            return training_evaluation.review(database, service.artifacts, reviewer, request)
        if operation == "training-history":
            return {"evaluations": training_evaluation.history(database)}
        if operation == "training-inspect":
            return training_evaluation.inspect(database, service.artifacts, request["evaluation_id"])
        return training_evaluation.run(database, service.artifacts, request)
    if operation == "snapshot":
        history = evidence.list(reviewer_id=reviewer).model_dump(mode="json")
        jobs = []
        for item in history["items"]:
            for job in service.get(item["id"]).jobs:
                attempts = service.jobs.attempts_for_job(job.id)
                jobs.append({"id": job.id, "experiment_id": item["id"],
                             "state": job.state.value, "error": ("Generation failed; inspect operational evidence after review." if item["sealed"] else attempts[-1].error_message) if attempts and attempts[-1].error_message else None})
        return {"history": history, "queue": jobs}
    if operation in ("preview", "create"):
        draft = ExperimentDraft.model_validate(request["draft"])
        capabilities = registry.capabilities(draft.baseline.adapter_id)
        pair = service.preview(draft, capabilities)
        if operation == "preview":
            return {"changed_paths": list(pair.changed_paths)}
        view = service.create(draft, capabilities)
        return {"experiment_id": view.experiment.id, "job_ids": [job.id for job in view.jobs]}
    if operation == "inspect":
        return evidence.detail(request["experiment_id"], reviewer).model_dump(mode="json")
    if operation == "cancel":
        job = service.jobs.request_cancellation(request["job_id"])
        return {"job_id": job.id, "state": job.state.value}
    if operation in ("run", "recover"):
        worker = Worker(repository=service.jobs, artifact_store=service.artifacts,
                        adapter_registry=registry, scratch_root=artifact_root / ".scratch",
                        worker_id=request.get("worker_id", "workbench"))
        if operation == "recover":
            with ComputeLease("lab-recovery"):
                return {"jobs": [job.model_dump(mode="json") for job in worker.recover()]}
        with ComputeLease("lab-generation"):
            return asdict(worker.run_once(request.get("job_id")))
    if operation == "review-start":
        view = reviews.start(request["experiment_id"], reviewer)
        return {"session_id": view.session_id, "prompt": view.prompt}
    if operation == "review-media":
        outputs = {}
        for position in ("left", "right"):
            artifact = reviews.artifact_for_display(request["session_id"], position, reviewer)
            parent = RetainedDirectoryAuthority.capture(artifact_root.absolute() / ".native-reviews")
            parent.close()
            folder = artifact_root.absolute() / ".native-reviews" / request["session_id"]
            authority = RetainedDirectoryAuthority.capture(folder)
            temporary = position + "-" + uuid.uuid4().hex + ".tmp"
            target_name = position + ".mp4"
            target = folder / target_name
            try:
                try:
                    target_stat = os.stat(target_name, dir_fd=authority.descriptor, follow_symlinks=False)
                    if not stat.S_ISREG(target_stat.st_mode):
                        raise ValueError("Review media must be a regular file.")
                except FileNotFoundError:
                    pass
                with database.connect() as connection:
                    with service.artifacts.open_verified_with_connection(
                        connection, artifact.id, sha256=artifact.sha256,
                        byte_size=artifact.byte_size, media_type=artifact.media_type,
                    ) as source:
                        digest = hashlib.sha256()
                        with os.fdopen(authority.open_exclusive(temporary), "wb") as stream:
                            while chunk := source.read(1024 * 1024):
                                stream.write(chunk)
                                digest.update(chunk)
                            stream.flush()
                            os.fsync(stream.fileno())
                        service.artifacts.reverify_open_handle(connection, source)
                        if digest.hexdigest() != artifact.sha256:
                            raise ValueError("Review artifact changed.")
                authority.reverify()
                os.replace(temporary, target_name, src_dir_fd=authority.descriptor,
                           dst_dir_fd=authority.descriptor)
                os.fsync(authority.descriptor)
                authority.reverify()
            finally:
                try:
                    authority.unlink(temporary, missing_ok=True)
                finally:
                    authority.close()
            outputs[position] = str(target)
        return {"session_id": request["session_id"], **outputs}
    if operation == "review-submit":
        result = reviews.submit(request["session_id"], reviewer,
                                DisplayPreference(request["preference"]),
                                tuple(RubricScore.model_validate(item) for item in request["scores"]),
                                request.get("notes"))
        return result.model_dump(mode="json")
    if operation == "reveal":
        return reviews.reveal(request["session_id"], reviewer).model_dump(mode="json")
    raise ValueError("Unsupported operation.")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--reviewer-id", default="local-operator")
    parser.add_argument("workspace", choices=["workspace"])
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.request.is_symlink() or not args.request.is_file() or args.request.stat().st_size > _MAX_REQUEST:
            raise ValueError("Request must be a bounded regular file.")
        request = validate_request(json.loads(args.request.read_text()))
        data = execute(args.database, args.artifacts, args.reviewer_id, request)
        print(json.dumps({"schema_version": "workspace.v1", "ok": True, "data": data}))
        return 0
    except (ValueError, TypeError, KeyError, LookupError, PermissionError, OSError, sqlite3.Error, ArtifactError) as error:
        print(json.dumps({"schema_version": "workspace.v1", "ok": False,
                          "error": {"code": "workspace_rejected", "message": str(error)}}))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
