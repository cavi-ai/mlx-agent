from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import sqlite3
import stat
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

from video_lab.database import Database
from video_lab.domain import utc_now

_HASH_CHUNK_SIZE = 1024 * 1024
_T = TypeVar("_T")


class ArtifactError(Exception):
    """Base error for artifact storage failures."""


class ArtifactIntegrityError(ArtifactError):
    """Stored or staged bytes do not match their registered digest."""


class ArtifactNotFoundError(ArtifactError):
    """An artifact id has not been registered."""


@dataclass(frozen=True)
class StagedArtifact:
    path: Path
    sha256: str
    byte_size: int
    media_type: str


@dataclass(frozen=True)
class Artifact:
    id: str
    job_id: str | None
    sha256: str
    byte_size: int
    media_type: str
    relative_path: str
    created_at: str


@dataclass(frozen=True)
class VerifiedArtifactHandle:
    """Path-free immutable descriptor held for the caller transaction lifetime."""

    artifact: Artifact
    descriptor: int
    relative_parts: tuple[str, ...]
    component_identities: tuple[tuple[int, int], ...]
    identity: tuple[int, int, int, int]

    def read(self, size: int = -1) -> bytes:
        return os.read(self.descriptor, self.artifact.byte_size if size < 0 else size)


class ArtifactStore:
    def __init__(self, database: Database, root: Path) -> None:
        self.database = database
        self.root = root.resolve()
        self.staging_root = self.root / ".staging"

    def stage(self, source: Path, media_type: str) -> StagedArtifact:
        self.staging_root.mkdir(parents=True, exist_ok=True)
        staged_path = self.staging_root / uuid.uuid4().hex
        digest = hashlib.sha256()
        byte_size = 0
        try:
            with source.open("rb") as source_file, staged_path.open("xb") as staged_file:
                while chunk := source_file.read(_HASH_CHUNK_SIZE):
                    staged_file.write(chunk)
                    digest.update(chunk)
                    byte_size += len(chunk)
        except Exception:
            staged_path.unlink(missing_ok=True)
            raise
        return StagedArtifact(staged_path, digest.hexdigest(), byte_size, media_type)

    def stage_bytes(self, data: bytes, media_type: str) -> StagedArtifact:
        """Stage already-produced bytes without exposing a mutable caller path."""
        self.staging_root.mkdir(parents=True, exist_ok=True)
        staged_path = self.staging_root / uuid.uuid4().hex
        try:
            with staged_path.open("xb") as staged_file:
                staged_file.write(data)
                staged_file.flush()
                os.fsync(staged_file.fileno())
        except Exception:
            staged_path.unlink(missing_ok=True)
            raise
        return StagedArtifact(staged_path, hashlib.sha256(data).hexdigest(), len(data), media_type)

    def commit_bytes_at_retained_root(
        self,
        data: bytes,
        media_type: str,
        root_descriptor: int,
        job_id: str | None = None,
        *,
        reverify: Callable[[], None],
    ) -> Artifact:
        """Publish through a retained root or compensate every visible side effect."""
        return self._commit_retained_bytes_transaction(
            data=data,
            media_type=media_type,
            root_descriptor=root_descriptor,
            job_id=job_id,
            reverify=reverify,
        )

    def _commit_retained_bytes_transaction(
        self,
        *,
        data: bytes,
        media_type: str,
        root_descriptor: int,
        job_id: str | None,
        reverify: Callable[[], None],
    ) -> Artifact:
        """Keep retained bytes and database registration recoverable until verified."""

        if (
            type(data) is not bytes
            or type(media_type) is not str
            or not media_type
            or not callable(reverify)
        ):
            raise ArtifactIntegrityError("retained artifact input is invalid")
        self._reverify_retained_root(reverify)
        try:
            root_info = os.fstat(root_descriptor)
        except OSError:
            raise ArtifactIntegrityError("retained artifact root is invalid") from None
        if not stat.S_ISDIR(root_info.st_mode):
            raise ArtifactIntegrityError("retained artifact root is invalid")
        digest = hashlib.sha256(data).hexdigest()
        relative_path = str(Path(digest[:2]) / digest[2:4] / digest)
        with ExitStack() as descriptors:
            locks = self._open_retained_directory(root_descriptor, ".locks", descriptors)
            lock_descriptor = os.open(
                f"{digest}.lock",
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=locks,
            )
            descriptors.callback(os.close, lock_descriptor)
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            descriptors.callback(fcntl.flock, lock_descriptor, fcntl.LOCK_UN)
            first = self._open_retained_directory(root_descriptor, digest[:2], descriptors)
            second = self._open_retained_directory(first, digest[2:4], descriptors)
            destination_created = False
            artifact_created = False
            association_created = False
            commit_attempted = False
            temporary_name = f".{digest}.{uuid.uuid4().hex}.tmp"
            artifact: Artifact
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                rollback_errors: list[BaseException] = []
                try:
                    row = connection.execute(
                        "SELECT * FROM artifacts WHERE sha256 = ? AND media_type = ?",
                        (digest, media_type),
                    ).fetchone()
                    if row is not None:
                        artifact = _artifact_from_row(row)
                        self._verify_retained_file(second, digest, artifact)
                    else:
                        try:
                            existing = os.open(
                                digest,
                                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                                dir_fd=second,
                            )
                        except FileNotFoundError:
                            existing = -1
                        if existing >= 0:
                            try:
                                payload = os.pread(existing, len(data) + 1, 0)
                            finally:
                                os.close(existing)
                            if payload != data:
                                raise ArtifactIntegrityError(
                                    "content-addressed destination integrity check failed"
                                )
                        else:
                            temporary = os.open(
                                temporary_name,
                                os.O_WRONLY
                                | os.O_CREAT
                                | os.O_EXCL
                                | getattr(os, "O_NOFOLLOW", 0),
                                0o600,
                                dir_fd=second,
                            )
                            try:
                                view = memoryview(data)
                                while view:
                                    written = os.write(temporary, view)
                                    if written <= 0:
                                        raise ArtifactIntegrityError(
                                            "retained artifact short write"
                                        )
                                    view = view[written:]
                                os.fsync(temporary)
                            finally:
                                os.close(temporary)
                            os.rename(
                                temporary_name,
                                digest,
                                src_dir_fd=second,
                                dst_dir_fd=second,
                            )
                            destination_created = True
                            os.fsync(second)
                        artifact = Artifact(
                            id=f"art_{uuid.uuid4().hex}",
                            job_id=None,
                            sha256=digest,
                            byte_size=len(data),
                            media_type=media_type,
                            relative_path=relative_path,
                            created_at=utc_now().isoformat(),
                        )
                        connection.execute(
                            "INSERT INTO artifacts(id, job_id, sha256, byte_size, media_type, "
                            "relative_path, created_at) VALUES (?, NULL, ?, ?, ?, ?, ?)",
                            (
                                artifact.id,
                                artifact.sha256,
                                artifact.byte_size,
                                artifact.media_type,
                                artifact.relative_path,
                                artifact.created_at,
                            ),
                        )
                        artifact_created = True
                    association_created = self._associate_retained_with_connection(
                        connection, artifact.id, job_id
                    )
                    self._reverify_retained_root(reverify)
                    self._before_database_commit()
                    commit_attempted = True
                    connection.commit()
                except BaseException as primary:
                    try:
                        connection.rollback()
                    except BaseException as rollback_error:
                        rollback_errors.append(rollback_error)
                    if commit_attempted:
                        try:
                            self._compensate_retained_database(
                                artifact_id=artifact.id,
                                job_id=job_id,
                                artifact_created=artifact_created,
                                association_created=association_created,
                            )
                        except BaseException as rollback_error:
                            rollback_errors.append(rollback_error)
                    rollback_errors.extend(
                        self._remove_retained_publication(
                            second,
                            digest,
                            temporary_name,
                            destination_created=destination_created,
                        )
                    )
                    if rollback_errors:
                        self._raise_retained_rollback_failure(primary, rollback_errors)
                    raise
            try:
                self._reverify_retained_root(reverify)
            except ArtifactIntegrityError as primary:
                rollback_errors = []
                try:
                    self._compensate_retained_database(
                        artifact_id=artifact.id,
                        job_id=job_id,
                        artifact_created=artifact_created,
                        association_created=association_created,
                    )
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
                rollback_errors.extend(
                    self._remove_retained_publication(
                        second,
                        digest,
                        temporary_name,
                        destination_created=destination_created,
                    )
                )
                if rollback_errors:
                    self._raise_retained_rollback_failure(primary, rollback_errors)
                raise
            return replace(artifact, job_id=job_id)

    @staticmethod
    def _reverify_retained_root(reverify: Callable[[], None]) -> None:
        try:
            reverify()
        except Exception:
            raise ArtifactIntegrityError(
                "retained artifact authority verification failed"
            ) from None

    @staticmethod
    def _associate_retained_with_connection(
        connection: sqlite3.Connection, artifact_id: str, job_id: str | None
    ) -> bool:
        if job_id is None:
            return False
        cursor = connection.execute(
            "INSERT OR IGNORE INTO job_artifacts(job_id, artifact_id, created_at) "
            "VALUES (?, ?, ?)",
            (job_id, artifact_id, utc_now().isoformat()),
        )
        return cursor.rowcount == 1

    def _compensate_retained_database(
        self,
        *,
        artifact_id: str,
        job_id: str | None,
        artifact_created: bool,
        association_created: bool,
    ) -> None:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if association_created and job_id is not None:
                    connection.execute(
                        "DELETE FROM job_artifacts WHERE job_id = ? AND artifact_id = ?",
                        (job_id, artifact_id),
                    )
                if artifact_created:
                    connection.execute(
                        "DELETE FROM job_artifacts WHERE artifact_id = ?", (artifact_id,)
                    )
                    connection.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _remove_retained_publication(
        parent: int,
        digest: str,
        temporary_name: str,
        *,
        destination_created: bool,
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        names = ((digest, destination_created), (temporary_name, True))
        removed = False
        for name, should_remove in names:
            if not should_remove:
                continue
            try:
                os.unlink(name, dir_fd=parent)
                removed = True
            except FileNotFoundError:
                pass
            except BaseException as error:
                errors.append(error)
        if removed:
            try:
                os.fsync(parent)
            except BaseException as error:
                errors.append(error)
        return errors

    @staticmethod
    def _raise_retained_rollback_failure(
        primary: BaseException, errors: list[BaseException]
    ) -> None:
        normalized_primary = (
            ArtifactIntegrityError(str(primary))
            if isinstance(primary, ArtifactIntegrityError)
            else ArtifactIntegrityError("retained artifact publication failed")
        )
        normalized_rollbacks = tuple(
            ArtifactIntegrityError("retained artifact rollback operation failed")
            for _error in errors
        )
        raise ArtifactIntegrityError("retained artifact rollback failed") from BaseExceptionGroup(
            "retained artifact rollback errors",
            (normalized_primary, *normalized_rollbacks),
        )

    def _before_database_commit(self) -> None:
        """Test seam immediately after pre-commit authority verification."""

    @staticmethod
    def _open_retained_directory(
        parent: int, name: str, stack: ExitStack
    ) -> int:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            pass
        descriptor = os.open(
            name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        stack.callback(os.close, descriptor)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ArtifactIntegrityError("retained artifact directory is invalid")
        return descriptor

    @staticmethod
    def _verify_retained_file(
        parent: int, name: str, artifact: Artifact
    ) -> None:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        try:
            info = os.fstat(descriptor)
            payload = os.pread(descriptor, artifact.byte_size + 1, 0)
        finally:
            os.close(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or len(payload) != artifact.byte_size
            or hashlib.sha256(payload).hexdigest() != artifact.sha256
        ):
            raise ArtifactIntegrityError("retained artifact integrity check failed")

    def discard(self, staged: StagedArtifact) -> None:
        try:
            path = staged.path.resolve(strict=False)
            root = self.staging_root.resolve(strict=False)
        except OSError as error:
            raise ArtifactIntegrityError("staged artifact path cannot be verified") from error
        if not path.is_relative_to(root) or path.parent != root:
            raise ArtifactIntegrityError("staged artifact path is outside staging root")
        path.unlink(missing_ok=True)

    def commit(self, staged: StagedArtifact, job_id: str | None) -> Artifact:
        if not staged.path.resolve().is_relative_to(self.staging_root.resolve()):
            raise ArtifactIntegrityError("staged artifact path is outside staging root")
        actual_sha256, actual_size = _hash_file(staged.path)
        if actual_sha256 != staged.sha256 or actual_size != staged.byte_size:
            raise ArtifactIntegrityError("staged artifact integrity check failed")

        locks_root = self.root / ".locks"
        locks_root.mkdir(parents=True, exist_ok=True)
        lock_path = locks_root / f"{staged.sha256}.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                return self._commit_locked(staged, job_id)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def commit_batch(
        self,
        staged_artifacts: tuple[StagedArtifact, ...],
        persist: Callable[[sqlite3.Connection, tuple[Artifact, ...]], _T],
        *,
        verify_artifacts: tuple[Artifact, ...] = (),
    ) -> _T:
        """Atomically register content and persist its owning domain records."""
        for staged in staged_artifacts:
            self._verify_staged(staged)

        with ExitStack() as locks:
            for sha256 in sorted({staged.sha256 for staged in staged_artifacts}):
                locks.enter_context(self._digest_lock(sha256))
            created_destinations: list[Path] = []
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    artifacts = []
                    for staged in staged_artifacts:
                        artifact, destination_created = self._register_with_connection(
                            connection, staged
                        )
                        artifacts.append(artifact)
                        if destination_created is not None:
                            created_destinations.append(destination_created)
                    with ExitStack() as verified:
                        handles = tuple(
                            verified.enter_context(
                                self.open_verified_with_connection(
                                    connection,
                                    artifact.id,
                                    sha256=artifact.sha256,
                                    byte_size=artifact.byte_size,
                                    media_type=artifact.media_type,
                                )
                            )
                            for artifact in (*artifacts, *verify_artifacts)
                        )
                        result = persist(connection, tuple(artifacts))
                        for handle in handles:
                            self.reverify_open_handle(connection, handle)
                        for handle in handles:
                            self.reverify_open_handle(connection, handle)
                        connection.commit()
                except Exception:
                    connection.rollback()
                    for destination in created_destinations:
                        destination.unlink(missing_ok=True)
                    raise

        for staged in staged_artifacts:
            staged.path.unlink(missing_ok=True)
        return result

    def _verify_staged(self, staged: StagedArtifact) -> None:
        if not staged.path.resolve().is_relative_to(self.staging_root.resolve()):
            raise ArtifactIntegrityError("staged artifact path is outside staging root")
        actual_sha256, actual_size = _hash_file(staged.path)
        if actual_sha256 != staged.sha256 or actual_size != staged.byte_size:
            raise ArtifactIntegrityError("staged artifact integrity check failed")

    @contextmanager
    def _digest_lock(self, sha256: str) -> Iterator[None]:
        locks_root = self.root / ".locks"
        locks_root.mkdir(parents=True, exist_ok=True)
        lock_path = locks_root / f"{sha256}.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _register_with_connection(
        self, connection: sqlite3.Connection, staged: StagedArtifact
    ) -> tuple[Artifact, Path | None]:
        row = connection.execute(
            "SELECT * FROM artifacts WHERE sha256 = ? AND media_type = ?",
            (staged.sha256, staged.media_type),
        ).fetchone()
        if row is not None:
            artifact = _artifact_from_row(row)
            self._verify_file(self._safe_path(artifact.relative_path), artifact)
            return artifact, None

        relative_path = str(Path(staged.sha256[:2]) / staged.sha256[2:4] / staged.sha256)
        destination = self._safe_path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_created: Path | None = None
        if destination.exists():
            destination_sha256, destination_size = _hash_file(destination)
            if destination_sha256 != staged.sha256 or destination_size != staged.byte_size:
                raise ArtifactIntegrityError("content-addressed destination integrity check failed")
        else:
            placement_temp = destination.parent / f".{staged.sha256}.{uuid.uuid4().hex}.tmp"
            try:
                shutil.copyfile(staged.path, placement_temp)
                placement_sha256, placement_size = _hash_file(placement_temp)
                if placement_sha256 != staged.sha256 or placement_size != staged.byte_size:
                    raise ArtifactIntegrityError("placement copy integrity check failed")
                os.replace(placement_temp, destination)
                destination_created = destination
            finally:
                placement_temp.unlink(missing_ok=True)

        artifact = Artifact(
            id=f"art_{uuid.uuid4().hex}",
            job_id=None,
            sha256=staged.sha256,
            byte_size=staged.byte_size,
            media_type=staged.media_type,
            relative_path=relative_path,
            created_at=utc_now().isoformat(),
        )
        try:
            connection.execute(
                "INSERT INTO artifacts(id, job_id, sha256, byte_size, media_type, "
                "relative_path, created_at) VALUES (?, NULL, ?, ?, ?, ?, ?)",
                (
                    artifact.id,
                    artifact.sha256,
                    artifact.byte_size,
                    artifact.media_type,
                    artifact.relative_path,
                    artifact.created_at,
                ),
            )
        except Exception:
            if destination_created is not None:
                destination_created.unlink(missing_ok=True)
            raise
        return artifact, destination_created

    def _commit_locked(self, staged: StagedArtifact, job_id: str | None) -> Artifact:
        existing = self._find_by_content(staged.sha256, staged.media_type)
        if existing is not None:
            self._verify_file(self._safe_path(existing.relative_path), existing)
            self._associate(existing.id, job_id)
            staged.path.unlink(missing_ok=True)
            return replace(existing, job_id=job_id)

        relative_path = str(Path(staged.sha256[:2]) / staged.sha256[2:4] / staged.sha256)
        destination = self._safe_path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_created = False
        if destination.exists():
            destination_sha256, destination_size = _hash_file(destination)
            if destination_sha256 != staged.sha256 or destination_size != staged.byte_size:
                raise ArtifactIntegrityError("content-addressed destination integrity check failed")
        else:
            placement_temp = destination.parent / f".{staged.sha256}.{uuid.uuid4().hex}.tmp"
            try:
                shutil.copyfile(staged.path, placement_temp)
                placement_sha256, placement_size = _hash_file(placement_temp)
                if placement_sha256 != staged.sha256 or placement_size != staged.byte_size:
                    raise ArtifactIntegrityError("placement copy integrity check failed")
                os.replace(placement_temp, destination)
                destination_created = True
            finally:
                placement_temp.unlink(missing_ok=True)

        artifact = Artifact(
            id=f"art_{uuid.uuid4().hex}",
            job_id=None,
            sha256=staged.sha256,
            byte_size=staged.byte_size,
            media_type=staged.media_type,
            relative_path=relative_path,
            created_at=utc_now().isoformat(),
        )
        try:
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """INSERT INTO artifacts(
                         id, job_id, sha256, byte_size, media_type, relative_path, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        artifact.id,
                        artifact.job_id,
                        artifact.sha256,
                        artifact.byte_size,
                        artifact.media_type,
                        artifact.relative_path,
                        artifact.created_at,
                    ),
                )
                self._associate_with_connection(connection, artifact.id, job_id)
                connection.commit()
        except sqlite3.IntegrityError:
            existing = self._find_by_content(staged.sha256, staged.media_type)
            if existing is None:
                if destination_created:
                    destination.unlink(missing_ok=True)
                raise
            self._associate(existing.id, job_id)
            staged.path.unlink(missing_ok=True)
            return replace(existing, job_id=job_id)
        except Exception:
            if destination_created:
                destination.unlink(missing_ok=True)
            raise
        staged.path.unlink(missing_ok=True)
        return replace(artifact, job_id=job_id)

    def list_for_job(self, job_id: str) -> tuple[Artifact, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT artifacts.* FROM artifacts "
                "JOIN job_artifacts ON job_artifacts.artifact_id = artifacts.id "
                "WHERE job_artifacts.job_id = ? ORDER BY job_artifacts.created_at, artifacts.id",
                (job_id,),
            ).fetchall()
        return tuple(replace(_artifact_from_row(row), job_id=job_id) for row in rows)

    def get(self, artifact_id: str) -> Artifact:
        """Return registered artifact metadata without bypassing integrity verification."""
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise ArtifactNotFoundError(f"artifact not found: {artifact_id}")
        artifact = _artifact_from_row(row)
        self.resolve(artifact.id)
        return artifact

    def _associate(self, artifact_id: str, job_id: str | None) -> None:
        if job_id is None:
            return
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._associate_with_connection(connection, artifact_id, job_id)
            connection.commit()

    @staticmethod
    def _associate_with_connection(
        connection: sqlite3.Connection, artifact_id: str, job_id: str | None
    ) -> None:
        if job_id is not None:
            connection.execute(
                "INSERT OR IGNORE INTO job_artifacts(job_id, artifact_id, created_at) "
                "VALUES (?, ?, ?)",
                (job_id, artifact_id, utc_now().isoformat()),
            )

    def resolve(self, artifact_id: str) -> Path:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
            if row is None:
                raise ArtifactNotFoundError("artifact not found")
            artifact = _artifact_from_row(row)
            return self.resolve_with_connection(
                connection,
                artifact_id,
                sha256=artifact.sha256,
                byte_size=artifact.byte_size,
                media_type=artifact.media_type,
            )

    def resolve_with_connection(
        self,
        connection: sqlite3.Connection,
        artifact_id: str,
        *,
        sha256: str,
        byte_size: int,
        media_type: str,
    ) -> Path:
        """Resolve verified content without taking ownership of the caller's transaction."""
        self._require_owned_connection(connection)
        row = connection.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise ArtifactNotFoundError("artifact not found")
        artifact = _artifact_from_row(row)
        expected = (sha256, byte_size, media_type)
        if _artifact_descriptor(artifact) != expected:
            raise ArtifactIntegrityError("artifact descriptor mismatch")

        path = self._safe_path(artifact.relative_path)
        self._verify_file(path, artifact)

        stable_row = connection.execute(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        if stable_row is None or _artifact_descriptor(_artifact_from_row(stable_row)) != expected:
            raise ArtifactIntegrityError("artifact descriptor changed during verification")
        return path

    @contextmanager
    def open_verified_with_connection(
        self,
        connection: sqlite3.Connection,
        artifact_id: str,
        *,
        sha256: str,
        byte_size: int,
        media_type: str,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[VerifiedArtifactHandle]:
        """Hold the exact verified inode open without owning the caller transaction."""
        self._require_owned_connection(connection)
        row = connection.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if row is None:
            raise ArtifactNotFoundError("artifact not found")
        artifact = _artifact_from_row(row)
        if _artifact_descriptor(artifact) != (sha256, byte_size, media_type):
            raise ArtifactIntegrityError("artifact descriptor mismatch")
        descriptor = -1
        try:
            descriptor, relative_parts, component_identities = self._open_canonical_artifact(
                artifact.relative_path
            )
            opened = os.fstat(descriptor)
            identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            if not stat.S_ISREG(opened.st_mode):
                raise ArtifactIntegrityError("artifact is not a regular file")
            handle = VerifiedArtifactHandle(
                artifact, descriptor, relative_parts, component_identities, identity
            )
            if cancelled is None:
                self.reverify_open_handle(connection, handle)
            else:
                self.reverify_open_handle(connection, handle, cancelled=cancelled)
            yield handle
        except OSError:
            raise ArtifactIntegrityError("artifact cannot be opened safely") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def reverify_open_handle(
        self,
        connection: sqlite3.Connection,
        handle: VerifiedArtifactHandle,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Rehash a held descriptor and its canonical database binding."""
        self._require_owned_connection(connection)
        current = os.fstat(handle.descriptor)
        identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        callback = cancelled or (lambda: False)
        digest, size = _hash_descriptor(handle.descriptor, cancelled=callback)
        canonical_descriptor = -1
        try:
            canonical_descriptor, relative_parts, component_identities = (
                self._open_canonical_artifact(handle.artifact.relative_path)
            )
            canonical = os.fstat(canonical_descriptor)
            canonical_identity = (
                canonical.st_dev,
                canonical.st_ino,
                canonical.st_size,
                canonical.st_mtime_ns,
            )
            canonical_digest, canonical_size = _hash_descriptor(
                canonical_descriptor, cancelled=callback
            )
        except OSError:
            raise ArtifactIntegrityError("artifact canonical path changed") from None
        finally:
            if canonical_descriptor >= 0:
                os.close(canonical_descriptor)
        row = connection.execute(
            "SELECT * FROM artifacts WHERE id=?", (handle.artifact.id,)
        ).fetchone()
        if (
            identity != handle.identity
            or relative_parts != handle.relative_parts
            or component_identities != handle.component_identities
            or canonical_identity != handle.identity
            or digest != handle.artifact.sha256
            or size != handle.artifact.byte_size
            or canonical_digest != handle.artifact.sha256
            or canonical_size != handle.artifact.byte_size
            or row is None
            or _artifact_descriptor(_artifact_from_row(row))
            != _artifact_descriptor(handle.artifact)
        ):
            raise ArtifactIntegrityError("artifact changed during verification")

    def _open_canonical_artifact(
        self, relative_path: str
    ) -> tuple[int, tuple[str, ...], tuple[tuple[int, int], ...]]:
        relative = Path(relative_path)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ArtifactIntegrityError("artifact path is invalid")
        current = -1
        leaf = -1
        try:
            current = os.open(
                self.root,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            root_stat = os.fstat(current)
            identities = [(root_stat.st_dev, root_stat.st_ino)]
            for part in relative.parts[:-1]:
                next_descriptor = os.open(
                    part,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current,
                )
                opened = os.fstat(next_descriptor)
                identities.append((opened.st_dev, opened.st_ino))
                os.close(current)
                current = next_descriptor
            leaf = os.open(
                relative.parts[-1],
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
            opened_leaf = os.fstat(leaf)
            identities.append((opened_leaf.st_dev, opened_leaf.st_ino))
            os.close(current)
            current = -1
            return leaf, relative.parts, tuple(identities)
        except OSError:
            if leaf >= 0:
                os.close(leaf)
            if current >= 0:
                os.close(current)
            raise ArtifactIntegrityError("artifact canonical path changed") from None

    def _require_owned_connection(self, connection: sqlite3.Connection) -> None:
        try:
            rows = connection.execute("PRAGMA database_list").fetchall()
        except sqlite3.Error:
            raise ArtifactIntegrityError("connection is closed") from None
        main = next((str(row[2]) for row in rows if str(row[1]) == "main"), "")
        if not main or Path(main).resolve() != self.database.path.resolve():
            raise ArtifactIntegrityError("connection belongs to a foreign database")

    def _find_by_content(self, sha256: str, media_type: str) -> Artifact | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE sha256 = ? AND media_type = ?",
                (sha256, media_type),
            ).fetchone()
        return None if row is None else _artifact_from_row(row)

    def _safe_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ArtifactIntegrityError("artifact path escapes artifact root")
        internal_roots = ((self.root / ".locks").resolve(), (self.root / ".staging").resolve())
        if any(path.is_relative_to(internal_root) for internal_root in internal_roots):
            raise ArtifactIntegrityError("database references internal artifact path")
        return path

    @staticmethod
    def _verify_file(path: Path, artifact: Artifact) -> None:
        try:
            sha256, byte_size = _hash_file(path)
        except FileNotFoundError as error:
            raise ArtifactIntegrityError("registered artifact file is missing") from error
        if sha256 != artifact.sha256 or byte_size != artifact.byte_size:
            raise ArtifactIntegrityError("registered artifact integrity check failed")


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
            byte_size += len(chunk)
    return digest.hexdigest(), byte_size


def _hash_descriptor(
    descriptor: int, *, cancelled: Callable[[], bool] = lambda: False
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        if cancelled():
            raise ArtifactIntegrityError("artifact verification cancelled")
        chunk = os.read(descriptor, _HASH_CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), size


def _artifact_from_row(row: sqlite3.Row) -> Artifact:
    return Artifact(
        id=str(row["id"]),
        job_id=None if row["job_id"] is None else str(row["job_id"]),
        sha256=str(row["sha256"]),
        byte_size=int(row["byte_size"]),
        media_type=str(row["media_type"]),
        relative_path=str(row["relative_path"]),
        created_at=str(row["created_at"]),
    )


def _artifact_descriptor(artifact: Artifact) -> tuple[str, int, str]:
    return artifact.sha256, artifact.byte_size, artifact.media_type
