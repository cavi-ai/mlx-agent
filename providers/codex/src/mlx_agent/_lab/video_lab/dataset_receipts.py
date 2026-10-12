"""Durable, fenced evidence for trainer-neutral dataset compilation."""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import timedelta
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from video_lab.database import Database
from video_lab.domain import canonical_json, utc_now


class CompilationConflict(RuntimeError):
    """A compilation identity or live claim did not match exactly."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class InputBinding(_FrozenModel):
    role: Literal["source_manifest", "transform_recipe"]
    artifact_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=0)
    media_type: str = Field(min_length=1)


class DatasetCompilationRequest(_FrozenModel):
    schema_version: Literal["dataset-compilation-request.v1"] = "dataset-compilation-request.v1"
    operation_key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    compiler_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source: InputBinding
    recipe: InputBinding
    output_roles: tuple[str, ...] = Field(min_length=1)
    reproduces_receipt_id: str | None = None

    @model_validator(mode="after")
    def validate_roles(self) -> DatasetCompilationRequest:
        if self.source.role != "source_manifest" or self.recipe.role != "transform_recipe":
            raise ValueError("input roles must be exact")
        if self.output_roles != tuple(sorted(set(self.output_roles))):
            raise ValueError("output roles must be sorted and unique")
        return self

    @property
    def digest(self) -> str:
        payload = canonical_json(self.model_dump(mode="json")).encode()
        return hashlib.sha256(payload).hexdigest()


class CompilationOutput(_FrozenModel):
    role: str
    artifact_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=0)
    media_type: str


class DatasetCompilationReceipt(_FrozenModel):
    schema_version: Literal["dataset-compilation-receipt.v1"] = "dataset-compilation-receipt.v1"
    receipt_id: str
    operation_id: str
    operation_key: str
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_id: str
    attempt: int = Field(gt=0)
    source: InputBinding
    recipe: InputBinding
    outputs: tuple[CompilationOutput, ...]
    reproduces_receipt_id: str | None = None
    created_at: str


class CompilationOperation(_FrozenModel):
    operation_id: str
    operation_key: str
    request: DatasetCompilationRequest
    request_digest: str
    status: Literal["pending", "running", "retryable", "completed", "failed", "cancelled"]
    attempt: int
    claim_token: str | None
    lease_expires_at: str | None
    receipt_id: str | None
    cleanup_failures: tuple[CleanupFailureEvidence, ...] = ()


class CleanupFailureEvidence(_FrozenModel):
    attempt: int
    claim_token: str
    code: str
    failure_digest: str
    created_at: str


class CompilationClaim(_FrozenModel):
    operation_id: str
    operation_key: str
    request_digest: str
    claimed: bool
    attempt: int
    claimant_id: str | None
    claim_token: str | None


class DatasetCompilationRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def begin(self, request: DatasetCompilationRequest) -> CompilationOperation:
        request_json = canonical_json(request.model_dump(mode="json"))
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT request_digest FROM dataset_compilation_operations WHERE operation_key=?",
                (request.operation_key,),
            ).fetchone()
            if existing is not None:
                if str(existing["request_digest"]) != request.digest:
                    raise CompilationConflict("compilation request digest conflict")
                connection.commit()
                return self.get(request.operation_key)
            operation_id = f"dco_{uuid.uuid4().hex}"
            connection.execute(
                "INSERT INTO dataset_compilation_operations "
                "(id,operation_key,compiler_id,request_digest,request_json,status,attempt,"
                "claimant_id,claim_token,lease_expires_at,receipt_id,created_at,updated_at) "
                "VALUES (?,?,?,?,?,'pending',0,NULL,NULL,NULL,NULL,?,?)",
                (
                    operation_id,
                    request.operation_key,
                    request.compiler_id,
                    request.digest,
                    request_json,
                    now,
                    now,
                ),
            )
            for binding in (request.source, request.recipe):
                connection.execute(
                    "INSERT INTO dataset_compilation_inputs VALUES (?,?,?,?,?,?,?)",
                    (
                        operation_id,
                        binding.role,
                        binding.artifact_id,
                        binding.sha256,
                        binding.byte_size,
                        binding.media_type,
                        now,
                    ),
                )
            self._event(connection, operation_id, 1, 0, "created", request.digest, now)
            connection.commit()
        return self.get(request.operation_key)

    def get(self, operation_key: str) -> CompilationOperation:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_compilation_operations WHERE operation_key=?",
                (operation_key,),
            ).fetchone()
            failures = connection.execute(
                "SELECT attempt,claim_token,code,failure_digest,created_at "
                "FROM dataset_compilation_failures WHERE operation_id=? AND code='cleanup_failed' "
                "ORDER BY attempt",
                (row["id"],) if row is not None else ("",),
            ).fetchall()
        if row is None:
            raise KeyError("compilation operation not found")
        return self._operation(row, failures)

    def claim(
        self, operation_key: str, *, claimant_id: str, lease_seconds: int
    ) -> CompilationClaim:
        if not claimant_id or lease_seconds <= 0:
            raise ValueError("claimant and positive lease are required")
        token = uuid.uuid4().hex
        now = utc_now()
        lease = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE dataset_compilation_operations SET status='running',attempt=attempt+1,"
                "claimant_id=?,claim_token=?,lease_expires_at=?,updated_at=? WHERE operation_key=? "
                "AND (status IN ('pending','retryable') OR (status='running' AND "
                "julianday(lease_expires_at)<=julianday('now')))",
                (claimant_id, token, lease, now.isoformat(), operation_key),
            )
            row = connection.execute(
                "SELECT * FROM dataset_compilation_operations WHERE operation_key=?",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise KeyError("compilation operation not found")
            if cursor.rowcount == 1:
                sequence = int(
                    connection.execute(
                        "SELECT count(*)+1 FROM dataset_compilation_events WHERE operation_id=?",
                        (row["id"],),
                    ).fetchone()[0]
                )
                self._event(
                    connection,
                    str(row["id"]),
                    sequence,
                    int(row["attempt"]),
                    "claimed",
                    str(row["request_digest"]),
                    now.isoformat(),
                )
            connection.commit()
        return CompilationClaim(
            operation_id=str(row["id"]),
            operation_key=operation_key,
            request_digest=str(row["request_digest"]),
            claimed=cursor.rowcount == 1,
            attempt=int(row["attempt"]),
            claimant_id=claimant_id if cursor.rowcount == 1 else None,
            claim_token=token if cursor.rowcount == 1 else None,
        )

    def complete(
        self,
        claim: CompilationClaim,
        *,
        outputs: dict[str, tuple[str, str, int, str]],
        receipt_digest: str,
        receipt_json: str,
    ) -> None:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self.complete_with_connection(
                connection,
                claim,
                outputs=outputs,
                receipt_digest=receipt_digest,
                receipt_json=receipt_json,
            )
            connection.commit()

    def complete_with_connection(
        self,
        connection: sqlite3.Connection,
        claim: CompilationClaim,
        *,
        outputs: dict[str, tuple[str, str, int, str]],
        receipt_digest: str,
        receipt_json: str,
    ) -> None:
        if not claim.claimed or claim.claim_token is None or claim.claimant_id is None:
            raise CompilationConflict("compilation completion requires a live claim")
        now = utc_now().isoformat()
        row = connection.execute(
            "SELECT request_json FROM dataset_compilation_operations WHERE id=? "
            "AND status='running' "
            "AND attempt=? AND claimant_id=? AND claim_token=? "
            "AND julianday(lease_expires_at)>julianday('now')",
            (claim.operation_id, claim.attempt, claim.claimant_id, claim.claim_token),
        ).fetchone()
        if row is None:
            raise CompilationConflict("compilation completion claim mismatch")
        request = DatasetCompilationRequest.model_validate_json(
            str(row["request_json"]), strict=True
        )
        if tuple(sorted(outputs)) != request.output_roles:
            raise CompilationConflict("compilation output roles mismatch")
        receipt = DatasetCompilationReceipt.model_validate_json(receipt_json, strict=True)
        expected_outputs = tuple(
            CompilationOutput(
                role=role,
                artifact_id=outputs[role][0],
                sha256=outputs[role][1],
                byte_size=outputs[role][2],
                media_type=outputs[role][3],
            )
            for role in request.output_roles
        )
        canonical_receipt = canonical_json(receipt.model_dump(mode="json"))
        if (
            hashlib.sha256(canonical_receipt.encode()).hexdigest() != receipt_digest
            or canonical_receipt != receipt_json
            or receipt.operation_id != claim.operation_id
            or receipt.operation_key != request.operation_key
            or receipt.compiler_id != request.compiler_id
            or receipt.attempt != claim.attempt
            or receipt.request_digest != claim.request_digest
            or receipt.source != request.source
            or receipt.recipe != request.recipe
            or receipt.outputs != expected_outputs
            or receipt.reproduces_receipt_id != request.reproduces_receipt_id
        ):
            raise CompilationConflict("compilation receipt binding mismatch")
        receipt_id = receipt.receipt_id
        for role in request.output_roles:
            artifact_id, sha256, byte_size, media_type = outputs[role]
            connection.execute(
                "INSERT INTO dataset_compilation_outputs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    receipt_id,
                    claim.operation_id,
                    role,
                    artifact_id,
                    sha256,
                    byte_size,
                    media_type,
                    claim.attempt,
                    now,
                ),
            )
        cursor = connection.execute(
            "UPDATE dataset_compilation_operations SET status='completed',claimant_id=NULL,"
            "claim_token=NULL,lease_expires_at=NULL,receipt_id=?,receipt_digest=?,receipt_json=?,"
            "updated_at=? WHERE id=? AND status='running' AND attempt=? AND claim_token=? "
            "AND julianday(lease_expires_at)>julianday('now')",
            (
                receipt_id,
                receipt_digest,
                receipt_json,
                now,
                claim.operation_id,
                claim.attempt,
                claim.claim_token,
            ),
        )
        if cursor.rowcount != 1:
            raise CompilationConflict("compilation completion claim mismatch")
        sequence = int(
            connection.execute(
                "SELECT count(*)+1 FROM dataset_compilation_events WHERE operation_id=?",
                (claim.operation_id,),
            ).fetchone()[0]
        )
        self._event(
            connection,
            claim.operation_id,
            sequence,
            claim.attempt,
            "completed",
            receipt_digest,
            now,
        )
        if request.reproduces_receipt_id is not None:
            connection.execute(
                "INSERT INTO dataset_compilation_reproductions VALUES (?,?,?,?)",
                (receipt_id, request.reproduces_receipt_id, claim.operation_id, now),
            )

    def fail(self, claim: CompilationClaim, *, code: str, failure_digest: str) -> None:
        if not claim.claimed or claim.claim_token is None:
            raise CompilationConflict("compilation failure requires a live claim")
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO dataset_compilation_failures VALUES (?,?,?,?,?,?)",
                (claim.operation_id, claim.attempt, claim.claim_token, code, failure_digest, now),
            )
            cursor = connection.execute(
                "UPDATE dataset_compilation_operations SET status='failed',claimant_id=NULL,"
                "claim_token=NULL,lease_expires_at=NULL,updated_at=? WHERE id=? "
                "AND status='running' "
                "AND attempt=? AND claim_token=? "
                "AND julianday(lease_expires_at)>julianday('now')",
                (now, claim.operation_id, claim.attempt, claim.claim_token),
            )
            if cursor.rowcount != 1:
                raise CompilationConflict("compilation failure claim mismatch")
            connection.commit()

    def cancel(self, claim: CompilationClaim, *, evidence_digest: str) -> None:
        if not claim.claimed or claim.claim_token is None:
            raise CompilationConflict("compilation cancellation requires a live claim")
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE dataset_compilation_operations SET status='cancelled',claimant_id=NULL,"
                "claim_token=NULL,lease_expires_at=NULL,updated_at=? WHERE id=? "
                "AND status='running' "
                "AND attempt=? AND claim_token=? "
                "AND julianday(lease_expires_at)>julianday('now')",
                (now, claim.operation_id, claim.attempt, claim.claim_token),
            )
            if cursor.rowcount != 1:
                raise CompilationConflict("compilation cancellation claim mismatch")
            sequence = int(
                connection.execute(
                    "SELECT count(*)+1 FROM dataset_compilation_events WHERE operation_id=?",
                    (claim.operation_id,),
                ).fetchone()[0]
            )
            self._event(
                connection,
                claim.operation_id,
                sequence,
                claim.attempt,
                "cancelled",
                evidence_digest,
                now,
            )
            connection.commit()

    def retryable_failure(self, claim: CompilationClaim, *, code: str, failure_digest: str) -> None:
        if not claim.claimed or claim.claim_token is None:
            raise CompilationConflict("compilation retryable failure requires a live claim")
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO dataset_compilation_failures VALUES (?,?,?,?,?,?)",
                (claim.operation_id, claim.attempt, claim.claim_token, code, failure_digest, now),
            )
            cursor = connection.execute(
                "UPDATE dataset_compilation_operations SET status='retryable',claimant_id=NULL,"
                "claim_token=NULL,lease_expires_at=NULL,updated_at=? WHERE id=? "
                "AND status='running' AND attempt=? AND claim_token=? "
                "AND julianday(lease_expires_at)>julianday('now')",
                (now, claim.operation_id, claim.attempt, claim.claim_token),
            )
            if cursor.rowcount != 1:
                raise CompilationConflict("compilation retryable failure claim mismatch")
            sequence = int(
                connection.execute(
                    "SELECT count(*)+1 FROM dataset_compilation_events WHERE operation_id=?",
                    (claim.operation_id,),
                ).fetchone()[0]
            )
            self._event(
                connection,
                claim.operation_id,
                sequence,
                claim.attempt,
                "cleanup_failed",
                failure_digest,
                now,
            )
            connection.commit()

    def receipt(self, receipt_id: str) -> DatasetCompilationReceipt:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_compilation_operations WHERE receipt_id=? "
                "AND status='completed'",
                (receipt_id,),
            ).fetchone()
            inputs = connection.execute(
                "SELECT * FROM dataset_compilation_inputs WHERE operation_id=? ORDER BY role",
                (row["id"],) if row is not None else ("",),
            ).fetchall()
            outputs = connection.execute(
                "SELECT * FROM dataset_compilation_outputs WHERE receipt_id=? ORDER BY role",
                (receipt_id,),
            ).fetchall()
        if row is None:
            raise KeyError("compilation receipt not found")
        receipt_json = str(row["receipt_json"])
        receipt = DatasetCompilationReceipt.model_validate_json(receipt_json, strict=True)
        request = DatasetCompilationRequest.model_validate_json(
            str(row["request_json"]), strict=True
        )
        durable_inputs = {
            str(item["role"]): InputBinding(
                role=cast(Literal["source_manifest", "transform_recipe"], str(item["role"])),
                artifact_id=str(item["artifact_id"]),
                sha256=str(item["sha256"]),
                byte_size=int(item["byte_size"]),
                media_type=str(item["media_type"]),
            )
            for item in inputs
        }
        durable_outputs = tuple(
            CompilationOutput(
                role=str(item["role"]),
                artifact_id=str(item["artifact_id"]),
                sha256=str(item["sha256"]),
                byte_size=int(item["byte_size"]),
                media_type=str(item["media_type"]),
            )
            for item in outputs
        )
        canonical_receipt = canonical_json(receipt.model_dump(mode="json"))
        if (
            canonical_receipt != receipt_json
            or hashlib.sha256(canonical_receipt.encode()).hexdigest() != str(row["receipt_digest"])
            or receipt.operation_id != str(row["id"])
            or receipt.operation_key != str(row["operation_key"])
            or receipt.compiler_id != str(row["compiler_id"])
            or receipt.attempt != int(row["attempt"])
            or receipt.request_digest != str(row["request_digest"])
            or request.digest != receipt.request_digest
            or durable_inputs
            != {"source_manifest": receipt.source, "transform_recipe": receipt.recipe}
            or durable_outputs != receipt.outputs
        ):
            raise CompilationConflict("durable compilation receipt mismatch")
        return receipt

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        operation_id: str,
        sequence: int,
        attempt: int,
        kind: str,
        digest: str,
        created_at: str,
    ) -> None:
        connection.execute(
            "INSERT INTO dataset_compilation_events VALUES (?,?,?,?,?,?)",
            (operation_id, sequence, attempt, kind, digest, created_at),
        )

    @staticmethod
    def _operation(
        row: sqlite3.Row, failures: list[sqlite3.Row] | tuple[sqlite3.Row, ...] = ()
    ) -> CompilationOperation:
        return CompilationOperation(
            operation_id=str(row["id"]),
            operation_key=str(row["operation_key"]),
            request=DatasetCompilationRequest.model_validate_json(
                str(row["request_json"]), strict=True
            ),
            request_digest=str(row["request_digest"]),
            status=cast(
                Literal["pending", "running", "retryable", "completed", "failed", "cancelled"],
                str(row["status"]),
            ),
            attempt=int(row["attempt"]),
            claim_token=row["claim_token"],
            lease_expires_at=row["lease_expires_at"],
            receipt_id=row["receipt_id"],
            cleanup_failures=tuple(
                CleanupFailureEvidence(
                    attempt=int(item["attempt"]),
                    claim_token=str(item["claim_token"]),
                    code=str(item["code"]),
                    failure_digest=str(item["failure_digest"]),
                    created_at=str(item["created_at"]),
                )
                for item in failures
            ),
        )
