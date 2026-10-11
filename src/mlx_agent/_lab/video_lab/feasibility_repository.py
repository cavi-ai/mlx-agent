from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from video_lab.animegen_pair_contract import (
    ANIMEGEN_CONTROLLED_GUIDANCE_VALUES,
    ANIMEGEN_CONTROLLED_PAIR_SEED,
)
from video_lab.artifacts import ArtifactError, ArtifactStore
from video_lab.database import Database
from video_lab.domain import Manifest, canonical_json, new_id, utc_now
from video_lab.feasibility_models import (
    FeasibilityRun,
    FeasibilityStage,
    StageAttempt,
    StageEvidence,
    StageOutcome,
    SupportVerdict,
)
from video_lab.runtime_identity import runtime_fingerprint
from video_lab.source_inventory import ResolvedSource, SourceFile


class FeasibilityRunNotFound(LookupError):
    pass


class StageAttemptNotFound(LookupError):
    pass


class ActiveStageAttemptError(RuntimeError):
    pass


class StageAttemptFinishedError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationAuthorizationBundle:
    run_id: str
    verdict: SupportVerdict
    measured_profile: tuple[int, int, int, int]
    canonical_manifest_hash: str
    source_fingerprint: str
    runtime_fingerprint: str
    placement_fingerprint: str
    resolved_sources: Mapping[str, ResolvedSource]
    authorized_manifest: Manifest


class FeasibilityRepository:
    def __init__(self, database: Database, artifacts: ArtifactStore | None = None) -> None:
        self.database = database
        self.artifacts = artifacts

    def create_run(self, run: FeasibilityRun) -> FeasibilityRun:
        validated = FeasibilityRun.model_validate(run.model_dump(round_trip=True))
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO feasibility_runs("
                "id, adapter_id, protocol_version, input_fingerprint, configuration_json, "
                "verdict, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    validated.id,
                    validated.adapter_id,
                    validated.protocol_version,
                    validated.input_fingerprint,
                    canonical_json(validated.model_dump(mode="json")["configuration"]),
                    validated.verdict.value,
                    validated.created_at.isoformat(),
                    validated.updated_at.isoformat(),
                ),
            )
        return validated

    def get_run(self, run_id: str) -> FeasibilityRun:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM feasibility_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise FeasibilityRunNotFound(f"feasibility run not found: {run_id}")
        return _run_from_row(row)

    def set_verdict(self, run_id: str, verdict: SupportVerdict) -> None:
        if verdict in (
            SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID,
            SupportVerdict.STANDARD_LAB_QUALIFIED,
        ):
            from video_lab.feasibility import derive_display_verdict

            attempts = self.list_attempts(run_id)
            if derive_display_verdict(attempts) is not verdict:
                raise ValueError("support verdict is not proven by immutable stage evidence")
            if verdict is SupportVerdict.STANDARD_LAB_QUALIFIED:
                run = self.get_run(run_id)
                with self.database.connect() as connection:
                    if not self._validated_standard_pair(connection, run, attempts):
                        raise ValueError(
                            "standard support verdict is not proven by persisted "
                            "generation evidence"
                        )
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE feasibility_runs SET verdict = ?, updated_at = ? WHERE id = ?",
                (verdict.value, utc_now().isoformat(), run_id),
            )

    def latest_authorized_profile(
        self, adapter_id: str
    ) -> tuple[SupportVerdict, tuple[int, int, int]] | None:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM feasibility_runs WHERE adapter_id = ? "
                "AND verdict IN (?, ?) ORDER BY updated_at DESC, id DESC",
                (
                    adapter_id,
                    SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID.value,
                    SupportVerdict.STANDARD_LAB_QUALIFIED.value,
                ),
            ).fetchall()
        from video_lab.feasibility import derive_display_verdict

        for row in rows:
            run = _run_from_row(row)
            all_attempts = self.list_attempts(run.id)
            derived = derive_display_verdict(all_attempts)
            if derived not in (
                SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID,
                SupportVerdict.STANDARD_LAB_QUALIFIED,
            ):
                continue
            profiles = run.configuration.get("manifest_profiles")
            if not isinstance(profiles, Mapping):
                continue
            name = "standard" if derived is SupportVerdict.STANDARD_LAB_QUALIFIED else "minimum"
            profile = profiles.get(name)
            if not isinstance(profile, Mapping):
                continue
            width, height, frames = (
                profile.get("width"),
                profile.get("height"),
                profile.get("num_frames", profile.get("frames")),
            )
            if isinstance(width, int) and isinstance(height, int) and isinstance(frames, int):
                stage = (
                    FeasibilityStage.QUALIFY_STANDARD
                    if derived is SupportVerdict.STANDARD_LAB_QUALIFIED
                    else FeasibilityStage.GENERATE_MINIMUM_2
                )
                profile_attempts = self.list_attempts(run.id, stage)
                evidence = profile_attempts[-1].evidence if profile_attempts else None
                if evidence is None:
                    continue
                artifact = evidence.details.get("artifact")
                if not isinstance(artifact, Mapping):
                    continue
                if (
                    artifact.get("width"),
                    artifact.get("height"),
                    artifact.get("num_frames"),
                ) != (width, height, frames):
                    continue
                configured_fps = profile.get("fps")
                if isinstance(configured_fps, int) and artifact.get("fps") != configured_fps:
                    continue
                try:
                    configured_manifest = Manifest.model_validate(profile)
                except ValueError:
                    configured_manifest = None
                if configured_manifest is not None:
                    expected_hash = hashlib.sha256(
                        canonical_json(configured_manifest).encode()
                    ).hexdigest()
                    if evidence.details.get("canonical_manifest_hash") != expected_hash:
                        continue
                if derived is SupportVerdict.STANDARD_LAB_QUALIFIED:
                    with self.database.connect() as connection:
                        if not self._validated_standard_pair(connection, run, all_attempts):
                            continue
                return derived, (width, height, frames)
        return None

    def generation_authorization_bundle(
        self, adapter_id: str
    ) -> GenerationAuthorizationBundle | None:
        from video_lab.feasibility import derive_display_verdict

        with self.database.connect() as connection:
            connection.execute("BEGIN")
            rows = connection.execute(
                "SELECT * FROM feasibility_runs WHERE adapter_id = ? "
                "ORDER BY updated_at DESC, id DESC",
                (adapter_id,),
            ).fetchall()
            for row in rows:
                run = _run_from_row(row)
                attempt_rows = connection.execute(
                    "SELECT * FROM feasibility_stage_attempts WHERE run_id = ? "
                    "ORDER BY started_at, stage, ordinal, id",
                    (run.id,),
                ).fetchall()
                attempts = tuple(_attempt_from_row(item) for item in attempt_rows)
                derived = derive_display_verdict(attempts)
                if run.verdict is not derived or derived not in (
                    SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID,
                    SupportVerdict.STANDARD_LAB_QUALIFIED,
                ):
                    continue
                stage = (
                    FeasibilityStage.QUALIFY_STANDARD
                    if derived is SupportVerdict.STANDARD_LAB_QUALIFIED
                    else FeasibilityStage.GENERATE_MINIMUM_2
                )
                generation = next(
                    (
                        item
                        for item in reversed(attempts)
                        if item.stage is stage and item.evidence is not None
                    ),
                    None,
                )
                download = next(
                    (
                        item
                        for item in reversed(attempts)
                        if item.stage is FeasibilityStage.DOWNLOAD_SOURCES
                        and item.outcome is StageOutcome.PASSED
                        and item.evidence is not None
                    ),
                    None,
                )
                if generation is None or generation.evidence is None or download is None:
                    continue
                artifact = generation.evidence.details.get("artifact")
                if not isinstance(artifact, Mapping):
                    continue
                measured = tuple(
                    artifact.get(key) for key in ("width", "height", "num_frames", "fps")
                )
                if not all(isinstance(value, int) for value in measured):
                    continue
                profiles = run.configuration.get("manifest_profiles")
                profile_name = (
                    "standard" if derived is SupportVerdict.STANDARD_LAB_QUALIFIED else "minimum"
                )
                if not isinstance(profiles, Mapping):
                    continue
                try:
                    authorized_manifest = Manifest.model_validate(profiles.get(profile_name))
                except ValueError:
                    continue
                expected_hash = hashlib.sha256(
                    canonical_json(authorized_manifest).encode()
                ).hexdigest()
                if (
                    measured
                    != (
                        authorized_manifest.width,
                        authorized_manifest.height,
                        authorized_manifest.num_frames,
                        authorized_manifest.fps,
                    )
                    or generation.evidence.details.get("profile") != profile_name
                ):
                    continue
                sources = _resolved_sources_from_attempt(download)
                if sources is None:
                    continue
                manifest_hash = generation.evidence.details.get("canonical_manifest_hash")
                fingerprints = tuple(
                    generation.evidence.details.get(key)
                    for key in (
                        "source_fingerprint",
                        "runtime_fingerprint",
                        "placement_fingerprint",
                    )
                )
                if not isinstance(manifest_hash, str) or not all(
                    isinstance(value, str) for value in fingerprints
                ):
                    continue
                if manifest_hash != expected_hash:
                    continue
                standard_unproven = (
                    derived is SupportVerdict.STANDARD_LAB_QUALIFIED
                    and not self._validated_standard_pair(connection, run, attempts)
                )
                if standard_unproven:
                    continue
                connection.commit()
                return GenerationAuthorizationBundle(
                    run_id=run.id,
                    verdict=derived,
                    measured_profile=measured,
                    canonical_manifest_hash=manifest_hash,
                    source_fingerprint=cast(str, fingerprints[0]),
                    runtime_fingerprint=cast(str, fingerprints[1]),
                    placement_fingerprint=cast(str, fingerprints[2]),
                    resolved_sources=sources,
                    authorized_manifest=authorized_manifest,
                )
            connection.commit()
        return None

    def _validated_standard_pair(
        self,
        connection: sqlite3.Connection,
        run: FeasibilityRun,
        attempts: tuple[StageAttempt, ...],
    ) -> bool:
        """Validate standard qualification against persisted jobs, bytes, probes, and pins."""
        if self.artifacts is None:
            return False
        profiles = run.configuration.get("manifest_profiles")
        if not isinstance(profiles, Mapping):
            return False
        try:
            standard = Manifest.model_validate(profiles.get("standard"))
        except ValueError:
            return False
        if (
            standard.prompt,
            standard.negative_prompt,
            standard.seed,
            standard.width,
            standard.height,
            standard.num_frames,
            standard.fps,
            standard.inference_steps,
            standard.guidance_scale,
            standard.dtype,
            standard.quantization,
            standard.scheduler,
            standard.memory_strategy,
        ) != (
            "anime character walking",
            "",
            7,
            512,
            288,
            17,
            8,
            4,
            1.0,
            "bfloat16",
            "bf16",
            "flow-match-euler-shift-3",
            "model_cpu_offload",
        ):
            return False
        latest = _latest_attempts(attempts)
        standard_attempt = latest.get(FeasibilityStage.QUALIFY_STANDARD)
        pair_attempt = latest.get(FeasibilityStage.VALIDATE_CONTROLLED_PAIR)
        if (
            standard_attempt is None
            or standard_attempt.evidence is None
            or standard_attempt.outcome is not StageOutcome.PASSED
            or pair_attempt is None
            or pair_attempt.evidence is None
            or pair_attempt.outcome is not StageOutcome.PASSED
        ):
            return False
        standard_details = standard_attempt.evidence.details
        identities = tuple(
            standard_details.get(key)
            for key in ("source_fingerprint", "runtime_fingerprint", "placement_fingerprint")
        )
        if not all(isinstance(value, str) and len(value) == 64 for value in identities):
            return False
        standard_job_id = standard_details.get("job_id")
        standard_sha256 = standard_details.get("artifact_sha256")
        if not isinstance(standard_job_id, str) or not isinstance(standard_sha256, str):
            return False
        if not self._validated_generation_job(
            connection,
            standard_job_id,
            standard_sha256,
            standard,
            cast(tuple[str, str, str], identities),
        ):
            return False

        details = pair_attempt.evidence.details
        if (
            details.get("valid") is not True
            or details.get("changed_paths") != ("guidance_scale",)
            or details.get("variable_path") != "guidance_scale"
            or details.get("pair_seed") != ANIMEGEN_CONTROLLED_PAIR_SEED
            or details.get("controlled_values") != ANIMEGEN_CONTROLLED_GUIDANCE_VALUES
        ):
            return False
        experiment_id = details.get("experiment_id")
        job_ids = details.get("job_ids")
        artifact_sha256 = details.get("artifact_sha256")
        review_identity = details.get("review_identity")
        if (
            not isinstance(experiment_id, str)
            or not isinstance(job_ids, tuple)
            or len(job_ids) != 2
            or not all(isinstance(value, str) for value in job_ids)
            or not isinstance(artifact_sha256, tuple)
            or len(artifact_sha256) != 2
            or not all(isinstance(value, str) for value in artifact_sha256)
            or not isinstance(review_identity, Mapping)
        ):
            return False
        session_id = review_identity.get("session_id")
        if (
            review_identity.get("experiment_id") != experiment_id
            or review_identity.get("state") != "blinded_randomized"
            or not isinstance(session_id, str)
        ):
            return False
        experiment = connection.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if (
            experiment is None
            or str(experiment["schema_version"]) != standard.schema_version
            or str(experiment["variable_path"]) != "guidance_scale"
            or json.loads(str(experiment["changed_paths_json"])) != ["guidance_scale"]
        ):
            return False
        variants = connection.execute(
            "SELECT variants.*, jobs.id AS job_id, jobs.state FROM variants "
            "JOIN jobs ON jobs.variant_id = variants.id "
            "WHERE variants.experiment_id = ? ORDER BY variants.label",
            (experiment_id,),
        ).fetchall()
        if len(variants) != 2 or tuple(str(row["label"]) for row in variants) != ("A", "B"):
            return False
        expected_manifests = (
            standard.model_copy(
                update={
                    "seed": ANIMEGEN_CONTROLLED_PAIR_SEED,
                    "guidance_scale": ANIMEGEN_CONTROLLED_GUIDANCE_VALUES[0],
                }
            ),
            standard.model_copy(
                update={
                    "seed": ANIMEGEN_CONTROLLED_PAIR_SEED,
                    "guidance_scale": ANIMEGEN_CONTROLLED_GUIDANCE_VALUES[1],
                }
            ),
        )
        if tuple(str(row["job_id"]) for row in variants) != job_ids:
            return False
        for index, (row, manifest) in enumerate(zip(variants, expected_manifests, strict=True)):
            if str(row["state"]) != "completed":
                return False
            try:
                persisted_manifest = Manifest.model_validate_json(str(row["manifest_json"]))
            except ValueError:
                return False
            if persisted_manifest != manifest or not self._validated_generation_job(
                connection,
                cast(str, job_ids[index]),
                cast(str, artifact_sha256[index]),
                manifest,
                cast(tuple[str, str, str], identities),
            ):
                return False
        session = connection.execute(
            "SELECT * FROM review_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            return False
        pins = {
            (
                str(session["left_variant_id"]),
                str(session["left_job_id"]),
                str(session["left_artifact_id"]),
                str(session["left_artifact_sha256"]),
            ),
            (
                str(session["right_variant_id"]),
                str(session["right_job_id"]),
                str(session["right_artifact_id"]),
                str(session["right_artifact_sha256"]),
            ),
        }
        expected_pins: set[tuple[str, str, str, str]] = set()
        for row, job_id, digest in zip(variants, job_ids, artifact_sha256, strict=True):
            artifacts = self.artifacts.list_for_job(cast(str, job_id))
            if len(artifacts) != 1:
                return False
            expected_pins.add(
                (str(row["id"]), cast(str, job_id), artifacts[0].id, cast(str, digest))
            )
        return (
            str(session["experiment_id"]) == experiment_id
            and str(session["reviewer_id"]) == f"feasibility:{run.id}"
            and pins == expected_pins
        )

    def _validated_generation_job(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        artifact_sha256: str,
        manifest: Manifest,
        identities: tuple[str, str, str],
    ) -> bool:
        if self.artifacts is None:
            return False
        job = connection.execute(
            "SELECT jobs.state, variants.manifest_json FROM jobs "
            "JOIN variants ON variants.id = jobs.variant_id WHERE jobs.id = ?",
            (job_id,),
        ).fetchone()
        if job is None or str(job["state"]) != "completed":
            return False
        try:
            persisted_manifest = Manifest.model_validate_json(str(job["manifest_json"]))
        except ValueError:
            return False
        if persisted_manifest != manifest:
            return False
        attempts = connection.execute(
            "SELECT * FROM run_attempts WHERE job_id = ? ORDER BY number", (job_id,)
        ).fetchall()
        if (
            len(attempts) != 1
            or str(attempts[0]["status"]) != "completed"
            or attempts[0]["artifact_id"] is None
        ):
            return False
        registered = self.artifacts.list_for_job(job_id)
        if (
            len(registered) != 1
            or registered[0].media_type != "video/mp4"
            or registered[0].id != str(attempts[0]["artifact_id"])
            or registered[0].sha256 != artifact_sha256
        ):
            return False
        try:
            self.artifacts.resolve(registered[0].id)
        except ArtifactError:
            return False
        events = connection.execute(
            "SELECT * FROM run_events WHERE job_id = ? AND attempt_id = ? "
            "AND kind = 'job_completed' ORDER BY id",
            (job_id, str(attempts[0]["id"])),
        ).fetchall()
        if len(events) != 1:
            return False
        try:
            payload = json.loads(str(events[0]["payload_json"]))
        except (TypeError, json.JSONDecodeError):
            return False
        decoded = payload.get("decoded_frame_sha256")
        aggregate = payload.get("decoded_rgb_aggregate_sha256")
        if (
            payload.get("artifact_id") != registered[0].id
            or payload.get("artifact_sha256") != artifact_sha256
            or payload.get("codec") not in ("h264", "avc1")
            or not isinstance(payload.get("duration_seconds"), int | float)
            or float(payload["duration_seconds"]) <= 0
            or (
                payload.get("width"),
                payload.get("height"),
                payload.get("frame_count"),
                payload.get("fps"),
            )
            != (manifest.width, manifest.height, manifest.num_frames, manifest.fps)
            or not isinstance(decoded, list)
            or len(decoded) != manifest.num_frames
            or not all(isinstance(item, str) and len(item) == 64 for item in decoded)
            or not isinstance(aggregate, str)
            or len(aggregate) != 64
        ):
            return False
        runtime_versions = payload.get("runtime_versions")
        telemetry = payload.get("adapter_telemetry")
        if not isinstance(runtime_versions, Mapping) or not isinstance(telemetry, Mapping):
            return False
        observed = telemetry.get("observed_configuration")
        if not isinstance(observed, Mapping):
            return False
        source, runtime, placement = identities
        expected_observed = {
            "scheduler": manifest.scheduler,
            "dtype": manifest.dtype,
            "quantization": manifest.quantization,
            "compute_dtype": manifest.dtype,
            "storage_dtype": ("bfloat16" if manifest.quantization == "bf16" else "float8_e4m3fn"),
            "loras": [item.model_dump(mode="json") for item in manifest.loras],
            "memory_strategy": manifest.memory_strategy,
            "execution_placement": "local_hybrid:mps",
            "source_fingerprint": source,
            "placement_fingerprint": placement,
        }
        return (
            runtime_fingerprint(cast(Mapping[str, str], runtime_versions)) == runtime
            and all(observed.get(key) == value for key, value in expected_observed.items())
            and telemetry.get("fallback_violations", []) == []
        )

    def latest_resolved_sources(self, adapter_id: str) -> dict[str, ResolvedSource] | None:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM feasibility_runs WHERE adapter_id = ? "
                "ORDER BY updated_at DESC, id DESC",
                (adapter_id,),
            ).fetchall()
        for row in rows:
            from video_lab.feasibility import derive_display_verdict

            if derive_display_verdict(self.list_attempts(str(row["id"]))) not in (
                SupportVerdict.EXPERIMENTAL_LOCAL_HYBRID,
                SupportVerdict.STANDARD_LAB_QUALIFIED,
            ):
                continue
            attempts = self.list_attempts(str(row["id"]), FeasibilityStage.DOWNLOAD_SOURCES)
            passed = next(
                (
                    attempt
                    for attempt in reversed(attempts)
                    if attempt.outcome is StageOutcome.PASSED and attempt.evidence is not None
                ),
                None,
            )
            if passed is not None:
                resolved = _resolved_sources_from_attempt(passed)
                if resolved is not None:
                    return resolved
        return None

    def start_attempt(
        self,
        run_id: str,
        stage: FeasibilityStage,
        input_fingerprint: str | None = None,
    ) -> StageAttempt:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                run = connection.execute(
                    "SELECT input_fingerprint FROM feasibility_runs WHERE id = ?", (run_id,)
                ).fetchone()
                if run is None:
                    raise FeasibilityRunNotFound(f"feasibility run not found: {run_id}")
                run_fingerprint = str(run["input_fingerprint"])
                fingerprint = run_fingerprint if input_fingerprint is None else input_fingerprint
                if run_fingerprint != fingerprint:
                    raise ValueError("attempt input fingerprint must match its run")
                active = connection.execute(
                    "SELECT 1 FROM feasibility_stage_attempts "
                    "WHERE run_id = ? AND stage = ? AND outcome IS NULL",
                    (run_id, stage.value),
                ).fetchone()
                if active is not None:
                    raise ActiveStageAttemptError(
                        f"active attempt already exists for {run_id}/{stage.value}"
                    )
                ordinal = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(ordinal), 0) + 1 "
                        "FROM feasibility_stage_attempts WHERE run_id = ? AND stage = ?",
                        (run_id, stage.value),
                    ).fetchone()[0]
                )
                attempt = StageAttempt(
                    id=new_id("feasibility_attempt"),
                    run_id=run_id,
                    stage=stage,
                    ordinal=ordinal,
                    input_fingerprint=fingerprint,
                    started_at=utc_now(),
                )
                connection.execute(
                    "INSERT INTO feasibility_stage_attempts("
                    "id, run_id, stage, ordinal, input_fingerprint, started_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        attempt.id,
                        attempt.run_id,
                        attempt.stage.value,
                        attempt.ordinal,
                        attempt.input_fingerprint,
                        attempt.started_at.isoformat(),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return attempt

    def finish_attempt(
        self,
        attempt_id: str,
        outcome: StageOutcome,
        evidence: StageEvidence,
        diagnostic_code: str | None = None,
        diagnostic_message: str | None = None,
    ) -> StageAttempt:
        finished_at = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM feasibility_stage_attempts WHERE id = ?", (attempt_id,)
                ).fetchone()
                if row is None:
                    raise StageAttemptNotFound(f"stage attempt not found: {attempt_id}")
                if row["outcome"] is not None:
                    raise StageAttemptFinishedError(f"stage attempt already finished: {attempt_id}")
                candidate = _attempt_from_row(row).model_copy(
                    update={
                        "outcome": outcome,
                        "finished_at": finished_at,
                        "evidence": evidence,
                        "diagnostic_code": diagnostic_code,
                        "diagnostic_message": diagnostic_message,
                    }
                )
                cursor = connection.execute(
                    "UPDATE feasibility_stage_attempts SET outcome = ?, finished_at = ?, "
                    "evidence_json = ?, diagnostic_code = ?, diagnostic_message = ? "
                    "WHERE id = ? AND outcome IS NULL",
                    (
                        candidate.outcome.value if candidate.outcome is not None else None,
                        candidate.finished_at.isoformat() if candidate.finished_at else None,
                        canonical_json(evidence),
                        candidate.diagnostic_code,
                        candidate.diagnostic_message,
                        attempt_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StageAttemptFinishedError(f"stage attempt already finished: {attempt_id}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return candidate

    def list_attempts(
        self, run_id: str, stage: FeasibilityStage | None = None
    ) -> tuple[StageAttempt, ...]:
        with self.database.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM feasibility_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if exists is None:
                raise FeasibilityRunNotFound(f"feasibility run not found: {run_id}")
            query = "SELECT * FROM feasibility_stage_attempts WHERE run_id = ?"
            parameters: tuple[str, ...] = (run_id,)
            if stage is not None:
                query += " AND stage = ?"
                parameters = (run_id, stage.value)
            query += " ORDER BY started_at, stage, ordinal, id"
            rows = connection.execute(query, parameters).fetchall()
        return tuple(_attempt_from_row(row) for row in rows)

    def recover_interrupted(self, run_id: str | None = None) -> tuple[StageAttempt, ...]:
        finished_at = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if run_id is not None:
                    exists = connection.execute(
                        "SELECT 1 FROM feasibility_runs WHERE id = ?", (run_id,)
                    ).fetchone()
                    if exists is None:
                        raise FeasibilityRunNotFound(f"feasibility run not found: {run_id}")
                query = "SELECT * FROM feasibility_stage_attempts WHERE outcome IS NULL"
                parameters: tuple[str, ...] = ()
                if run_id is not None:
                    query += " AND run_id = ?"
                    parameters = (run_id,)
                query += " ORDER BY started_at, stage, ordinal, id"
                rows = connection.execute(query, parameters).fetchall()
                evidence = StageEvidence()
                recovered: list[StageAttempt] = []
                for row in rows:
                    attempt = _attempt_from_row(row).model_copy(
                        update={
                            "outcome": StageOutcome.INTERRUPTED,
                            "finished_at": finished_at,
                            "evidence": evidence,
                            "diagnostic_code": "process_interrupted",
                            "diagnostic_message": "active attempt interrupted before completion",
                        }
                    )
                    connection.execute(
                        "UPDATE feasibility_stage_attempts SET outcome = ?, finished_at = ?, "
                        "evidence_json = ?, diagnostic_code = ?, diagnostic_message = ? "
                        "WHERE id = ? AND outcome IS NULL",
                        (
                            attempt.outcome.value if attempt.outcome is not None else None,
                            finished_at.isoformat(),
                            canonical_json(evidence),
                            attempt.diagnostic_code,
                            attempt.diagnostic_message,
                            attempt.id,
                        ),
                    )
                    recovered.append(attempt)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return tuple(recovered)


def _latest_attempts(attempts: tuple[StageAttempt, ...]) -> dict[FeasibilityStage, StageAttempt]:
    latest: dict[FeasibilityStage, StageAttempt] = {}
    for attempt in attempts:
        if attempt.stage not in latest or attempt.ordinal > latest[attempt.stage].ordinal:
            latest[attempt.stage] = attempt
    return latest


def _run_from_row(row: sqlite3.Row) -> FeasibilityRun:
    return FeasibilityRun(
        id=str(row["id"]),
        adapter_id=str(row["adapter_id"]),
        protocol_version=str(row["protocol_version"]),
        input_fingerprint=str(row["input_fingerprint"]),
        configuration=json.loads(str(row["configuration_json"])),
        verdict=SupportVerdict(str(row["verdict"])),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )


def _attempt_from_row(row: sqlite3.Row) -> StageAttempt:
    outcome = row["outcome"]
    evidence_json = row["evidence_json"]
    return StageAttempt(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        stage=FeasibilityStage(str(row["stage"])),
        ordinal=int(row["ordinal"]),
        input_fingerprint=str(row["input_fingerprint"]),
        outcome=None if outcome is None else StageOutcome(str(outcome)),
        started_at=datetime.fromisoformat(str(row["started_at"])),
        finished_at=(
            None if row["finished_at"] is None else datetime.fromisoformat(str(row["finished_at"]))
        ),
        evidence=(
            None if evidence_json is None else StageEvidence.model_validate_json(str(evidence_json))
        ),
        diagnostic_code=(None if row["diagnostic_code"] is None else str(row["diagnostic_code"])),
        diagnostic_message=(
            None if row["diagnostic_message"] is None else str(row["diagnostic_message"])
        ),
    )


def _resolved_sources_from_attempt(
    attempt: StageAttempt,
) -> dict[str, ResolvedSource] | None:
    evidence = attempt.evidence
    inventory = None if evidence is None else evidence.details.get("inventory")
    values = inventory.get("sources") if isinstance(inventory, Mapping) else None
    if not isinstance(values, tuple):
        return None
    resolved: dict[str, ResolvedSource] = {}
    for value in values:
        if not isinstance(value, Mapping):
            return None
        files_value = value.get("files")
        if not isinstance(files_value, tuple):
            return None
        files = tuple(
            SourceFile(
                relative_path=str(item["relative_path"]),
                size=int(cast(Any, item["size"])),
                upstream_sha256=(
                    None if item.get("upstream_sha256") is None else str(item["upstream_sha256"])
                ),
                local_sha256=(
                    None if item.get("local_sha256") is None else str(item["local_sha256"])
                ),
            )
            for item in files_value
            if isinstance(item, Mapping)
        )
        source = ResolvedSource(
            component=str(value["component"]),
            repo_id=str(value["repo_id"]),
            requested_revision=str(value["requested_revision"]),
            revision=str(value["revision"]),
            license=str(value["license"]),
            files=files,
        )
        resolved[source.component] = source
    return resolved if len(resolved) == len(values) else None
