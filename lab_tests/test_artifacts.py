import fcntl
import hashlib
import multiprocessing
import os
import sqlite3
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty

import pytest

import video_lab.artifacts as artifacts_module
from lab_tests.test_capabilities import manifest
from lab_tests.test_services import capabilities
from video_lab.artifacts import (
    Artifact,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactStore,
    StagedArtifact,
)
from video_lab.database import Database
from video_lab.domain import ExperimentDraft
from video_lab.local_authority import RetainedDirectoryAuthority
from video_lab.services import ExperimentService


def _commit_in_process(
    database_path: str,
    root: str,
    source: str,
    job_id: str | None,
    results: Queue[tuple[str, str]],
) -> None:
    store = ArtifactStore(Database(Path(database_path)), Path(root))
    staged = store.stage(Path(source), "video/mp4")
    artifact = store.commit(staged, job_id=job_id)
    results.put((artifact.id, str(staged.path)))


def test_commits_verified_content_once(database: Database, tmp_path: Path) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"video-bytes")
    store = ArtifactStore(database, tmp_path / "artifacts")

    first = store.commit(store.stage(source, "video/mp4"), job_id=None)
    second = store.commit(store.stage(source, "video/mp4"), job_id=None)

    assert first.id == second.id
    assert first.sha256 == "79fd615a866fe7f9eb4da8d9c41ab57e3bd48056df42fd2c13e4d461a87afbe3"
    assert store.resolve(first.id).read_bytes() == b"video-bytes"


def test_retained_root_byte_commit_writes_through_verified_descriptor(
    database: Database, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    store = ArtifactStore(database, root)
    authority = RetainedDirectoryAuthority.capture(root)
    try:
        artifact = store.commit_bytes_at_retained_root(
            b"retained-payload",
            "application/octet-stream",
            authority.descriptor,
            reverify=authority.reverify,
        )
    finally:
        authority.close()

    assert (root / artifact.relative_path).read_bytes() == b"retained-payload"
    with database.connect() as connection:
        row = connection.execute(
            "SELECT sha256, byte_size, media_type, relative_path FROM artifacts "
            "WHERE id = ?",
            (artifact.id,),
        ).fetchone()
    assert tuple(row) == (
        artifact.sha256,
        len(b"retained-payload"),
        "application/octet-stream",
        artifact.relative_path,
    )


def test_artifact_root_swap_during_commit_rolls_back_bytes_and_row(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "retained-artifacts"
    root.mkdir()
    detached = tmp_path / "detached-artifacts"
    authority = RetainedDirectoryAuthority.capture(
        root, error_code="artifact_root_changed"
    )
    store = ArtifactStore(database, root)

    def replace_root_path() -> None:
        root.rename(detached)
        root.mkdir()

    monkeypatch.setattr(
        store, "_before_database_commit", replace_root_path, raising=False
    )
    try:
        with pytest.raises(ArtifactIntegrityError):
            store.commit_bytes_at_retained_root(
                b"evidence",
                "application/json",
                authority.descriptor,
                reverify=authority.reverify,
            )
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
        assert not any(path.is_file() for path in detached.rglob("*") if ".locks" not in path.parts)
        assert not any(path.is_file() for path in root.rglob("*"))
    finally:
        if root.exists():
            for path in sorted(root.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            root.rmdir()
        if detached.exists():
            detached.rename(root)
        authority.close()


def test_retained_root_commit_deduplicates_and_preserves_job_associations(
    database: Database, tmp_path: Path
) -> None:
    service = ExperimentService.from_paths(database.path, tmp_path / "artifacts")
    jobs = service.create(
        ExperimentDraft(
            baseline=manifest(), variable_path="inference_steps", alternate_value=8
        ),
        capabilities(),
    ).jobs
    authority = RetainedDirectoryAuthority.capture(service.artifacts.root)
    try:
        first = service.artifacts.commit_bytes_at_retained_root(
            b"shared-retained-evidence",
            "application/json",
            authority.descriptor,
            jobs[0].id,
            reverify=authority.reverify,
        )
        second = service.artifacts.commit_bytes_at_retained_root(
            b"shared-retained-evidence",
            "application/json",
            authority.descriptor,
            jobs[1].id,
            reverify=authority.reverify,
        )
        repeated = service.artifacts.commit_bytes_at_retained_root(
            b"shared-retained-evidence",
            "application/json",
            authority.descriptor,
            jobs[1].id,
            reverify=authority.reverify,
        )
    finally:
        authority.close()

    assert first.id == second.id == repeated.id
    assert service.artifacts.list_for_job(jobs[0].id) == (first,)
    assert service.artifacts.list_for_job(jobs[1].id) == (second,)
    assert (service.artifacts.root / first.relative_path).read_bytes() == (
        b"shared-retained-evidence"
    )
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM job_artifacts").fetchone()[0] == 2


def test_retained_root_precommit_authority_failure_removes_bytes_and_row(
    database: Database, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    authority = RetainedDirectoryAuthority.capture(root)
    store = ArtifactStore(database, root)
    calls = 0

    def fail_precommit_reverify() -> None:
        nonlocal calls
        calls += 1
        authority.reverify()
        if calls == 2:
            raise ValueError("forced authority failure")

    try:
        with pytest.raises(
            ArtifactIntegrityError, match="retained artifact authority verification failed"
        ):
            store.commit_bytes_at_retained_root(
                b"precommit",
                "application/json",
                authority.descriptor,
                reverify=fail_precommit_reverify,
            )
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
        assert not any(
            path.is_file() for path in root.rglob("*") if ".locks" not in path.parts
        )
    finally:
        authority.close()


def test_retained_root_database_failure_removes_bytes_and_row(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    authority = RetainedDirectoryAuthority.capture(root)
    store = ArtifactStore(database, root)
    monkeypatch.setattr(
        store,
        "_before_database_commit",
        lambda: (_ for _ in ()).throw(RuntimeError("database commit failed")),
    )
    try:
        with pytest.raises(RuntimeError, match="^database commit failed$"):
            store.commit_bytes_at_retained_root(
                b"database-failure",
                "application/json",
                authority.descriptor,
                reverify=authority.reverify,
            )
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
        assert not any(
            path.is_file() for path in root.rglob("*") if ".locks" not in path.parts
        )
    finally:
        authority.close()


def test_retained_root_write_failure_removes_staging_and_database_state(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    authority = RetainedDirectoryAuthority.capture(root)
    store = ArtifactStore(database, root)
    real_fsync = artifacts_module.os.fsync
    fsync_calls = 0

    def fail_staging_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 1:
            raise OSError("forced retained stage failure")
        real_fsync(descriptor)

    monkeypatch.setattr(artifacts_module.os, "fsync", fail_staging_fsync)
    try:
        with pytest.raises(OSError, match="^forced retained stage failure$"):
            store.commit_bytes_at_retained_root(
                b"stage-failure",
                "application/json",
                authority.descriptor,
                reverify=authority.reverify,
            )
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
        assert not any(
            path.is_file() for path in root.rglob("*") if ".locks" not in path.parts
        )
    finally:
        authority.close()


def test_retained_root_postrename_fsync_failure_removes_file_row_and_receipt(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    authority = RetainedDirectoryAuthority.capture(root)
    store = ArtifactStore(database, root)
    real_fsync = artifacts_module.os.fsync
    fsync_calls = 0
    receipts: list[Artifact] = []

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("forced post-rename directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(artifacts_module.os, "fsync", fail_directory_fsync)
    try:
        with pytest.raises(
            OSError, match="^forced post-rename directory fsync failure$"
        ):
            receipts.append(
                store.commit_bytes_at_retained_root(
                    b"post-rename-failure",
                    "application/json",
                    authority.descriptor,
                    reverify=authority.reverify,
                )
            )
        assert receipts == []
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
        assert not any(
            path.is_file() for path in root.rglob("*") if ".locks" not in path.parts
        )
    finally:
        authority.close()


def test_retained_root_commit_then_raise_compensates_visible_row_association_and_file(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ExperimentService.from_paths(database.path, tmp_path / "service-artifacts")
    job = service.create(
        ExperimentDraft(
            baseline=manifest(), variable_path="inference_steps", alternate_value=8
        ),
        capabilities(),
    ).jobs[0]
    root = tmp_path / "artifacts"
    authority = RetainedDirectoryAuthority.capture(root)
    store = ArtifactStore(database, root)
    real_connect = database.connect
    first_connection = True

    class CommitThenRaise:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def __enter__(self) -> "CommitThenRaise":
            self.connection.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self.connection.__exit__(*args)  # type: ignore[arg-type]

        def __getattr__(self, name: str) -> object:
            return getattr(self.connection, name)

        def commit(self) -> None:
            self.connection.commit()
            raise RuntimeError("commit returned an ambiguous failure")

        def rollback(self) -> None:
            self.connection.rollback()

        def execute(self, *args: object) -> sqlite3.Cursor:
            return self.connection.execute(*args)  # type: ignore[arg-type]

    def connect() -> sqlite3.Connection | CommitThenRaise:
        nonlocal first_connection
        connection = real_connect()
        if first_connection:
            first_connection = False
            return CommitThenRaise(connection)
        return connection

    monkeypatch.setattr(database, "connect", connect)
    try:
        with pytest.raises(
            RuntimeError, match="^commit returned an ambiguous failure$"
        ):
            store.commit_bytes_at_retained_root(
                b"commit-then-raise",
                "application/json",
                authority.descriptor,
                job.id,
                reverify=authority.reverify,
            )
        with real_connect() as connection:
            assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
            assert connection.execute("SELECT count(*) FROM job_artifacts").fetchone()[0] == 0
        assert not any(
            path.is_file() for path in root.rglob("*") if ".locks" not in path.parts
        )
    finally:
        authority.close()


def test_retained_root_postcommit_failure_preserves_preexisting_verified_content(
    database: Database, tmp_path: Path
) -> None:
    data = b"preexisting-verified-content"
    digest = hashlib.sha256(data).hexdigest()
    root = tmp_path / "artifacts"
    destination = root / digest[:2] / digest[2:4] / digest
    destination.parent.mkdir(parents=True)
    destination.write_bytes(data)
    authority = RetainedDirectoryAuthority.capture(root)
    store = ArtifactStore(database, root)
    calls = 0

    def fail_postcommit_reverify() -> None:
        nonlocal calls
        calls += 1
        authority.reverify()
        if calls == 3:
            raise ValueError("forced authority failure")

    try:
        with pytest.raises(
            ArtifactIntegrityError, match="retained artifact authority verification failed"
        ):
            store.commit_bytes_at_retained_root(
                data,
                "application/json",
                authority.descriptor,
                reverify=fail_postcommit_reverify,
            )
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
        assert destination.read_bytes() == data
        assert tuple(path for path in destination.parent.iterdir() if path.name != digest) == ()
    finally:
        authority.close()


def test_retained_root_authority_close_after_commit_compensates_created_state(
    database: Database, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    authority = RetainedDirectoryAuthority.capture(root)
    store = ArtifactStore(database, root)
    calls = 0

    def close_during_postcommit_reverify() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            authority.close()
        authority.reverify()

    with pytest.raises(ArtifactIntegrityError):
        store.commit_bytes_at_retained_root(
            b"cooperative-close",
            "application/json",
            authority.descriptor,
            reverify=close_during_postcommit_reverify,
        )
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
    assert not any(
        path.is_file() for path in root.rglob("*") if ".locks" not in path.parts
    )


def test_retained_root_postcommit_failure_removes_only_new_association(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ExperimentService.from_paths(database.path, tmp_path / "artifacts")
    jobs = service.create(
        ExperimentDraft(
            baseline=manifest(), variable_path="inference_steps", alternate_value=8
        ),
        capabilities(),
    ).jobs
    root = service.artifacts.root
    detached = tmp_path / "detached-artifacts"
    authority = RetainedDirectoryAuthority.capture(root)
    first = service.artifacts.commit_bytes_at_retained_root(
        b"existing-row",
        "application/json",
        authority.descriptor,
        jobs[0].id,
        reverify=authority.reverify,
    )

    def replace_root_path() -> None:
        root.rename(detached)
        root.mkdir()

    monkeypatch.setattr(
        service.artifacts, "_before_database_commit", replace_root_path
    )
    try:
        with pytest.raises(ArtifactIntegrityError):
            service.artifacts.commit_bytes_at_retained_root(
                b"existing-row",
                "application/json",
                authority.descriptor,
                jobs[1].id,
                reverify=authority.reverify,
            )
        assert service.artifacts.list_for_job(jobs[0].id) == (first,)
        assert service.artifacts.list_for_job(jobs[1].id) == ()
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 1
            assert connection.execute("SELECT count(*) FROM job_artifacts").fetchone()[0] == 1
        assert (detached / first.relative_path).read_bytes() == b"existing-row"
        assert not any(path.is_file() for path in root.rglob("*"))
    finally:
        if root.exists():
            root.rmdir()
        if detached.exists():
            detached.rename(root)
        authority.close()


def test_retained_root_rollback_failure_is_aggregated_after_compensation(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    detached = tmp_path / "detached-artifacts"
    authority = RetainedDirectoryAuthority.capture(root)
    store = ArtifactStore(database, root)
    real_fsync = artifacts_module.os.fsync
    fsync_calls = 0

    def replace_root_path() -> None:
        root.rename(detached)
        root.mkdir()

    def fail_cleanup_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 3:
            raise OSError("forced cleanup fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(store, "_before_database_commit", replace_root_path)
    monkeypatch.setattr(artifacts_module.os, "fsync", fail_cleanup_fsync)
    try:
        with pytest.raises(
            ArtifactIntegrityError, match="^retained artifact rollback failed$"
        ) as raised:
            store.commit_bytes_at_retained_root(
                b"rollback-failure",
                "application/json",
                authority.descriptor,
                reverify=authority.reverify,
            )
        assert str(root) not in str(raised.value)
        assert str(detached) not in str(raised.value)
        assert isinstance(raised.value.__cause__, BaseExceptionGroup)
        primary, *_rollback = raised.value.__cause__.exceptions
        assert str(primary) == "retained artifact authority verification failed"
        assert str(root) not in str(primary)
        assert str(detached) not in str(primary)
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
        assert not any(
            path.is_file() for path in detached.rglob("*") if ".locks" not in path.parts
        )
        assert not any(path.is_file() for path in root.rglob("*"))
    finally:
        if root.exists():
            root.rmdir()
        if detached.exists():
            detached.rename(root)
        authority.close()


def test_verified_open_cancels_before_hashing_artifact(database: Database, tmp_path: Path) -> None:
    store = ArtifactStore(database, tmp_path / "artifacts")
    artifact = store.commit(store.stage_bytes(b"x" * (2 * 1024 * 1024), "video/mp4"), None)
    with database.connect() as connection:
        with pytest.raises(ArtifactIntegrityError, match="cancelled"):
            with store.open_verified_with_connection(
                connection,
                artifact.id,
                sha256=artifact.sha256,
                byte_size=artifact.byte_size,
                media_type=artifact.media_type,
                cancelled=lambda: True,
            ):
                pytest.fail("cancelled artifact must never be yielded")


@pytest.mark.parametrize("consumed_before_mutation", [2048, 4096])
def test_retained_handle_post_stream_reverify_rejects_same_inode_mutation(
    database: Database, tmp_path: Path, consumed_before_mutation: int
) -> None:
    store = ArtifactStore(database, tmp_path / "artifacts")
    artifact = store.commit(store.stage_bytes(b"a" * 4096, "application/octet-stream"), None)
    with database.connect() as connection:
        with store.open_verified_with_connection(
            connection,
            artifact.id,
            sha256=artifact.sha256,
            byte_size=artifact.byte_size,
            media_type=artifact.media_type,
        ) as handle:
            assert handle.read(consumed_before_mutation) == b"a" * consumed_before_mutation
            original = os.fstat(handle.descriptor)
            path = store.root / artifact.relative_path
            with path.open("r+b", buffering=0) as writer:
                os.pwrite(writer.fileno(), b"b", 0)
            os.utime(
                path,
                ns=(original.st_atime_ns, original.st_mtime_ns),
            )
            remaining = artifact.byte_size - consumed_before_mutation
            assert handle.read(remaining) == b"a" * remaining
            with pytest.raises(ArtifactIntegrityError, match="changed during verification"):
                store.reverify_open_handle(connection, handle)


def test_retained_handle_post_read_reverify_honors_cancellation(
    database: Database, tmp_path: Path
) -> None:
    store = ArtifactStore(database, tmp_path / "artifacts")
    artifact = store.commit(store.stage_bytes(b"authentic", "application/octet-stream"), None)
    with database.connect() as connection:
        with store.open_verified_with_connection(
            connection,
            artifact.id,
            sha256=artifact.sha256,
            byte_size=artifact.byte_size,
            media_type=artifact.media_type,
        ) as handle:
            assert handle.read(artifact.byte_size) == b"authentic"
            with pytest.raises(ArtifactIntegrityError, match="cancelled"):
                store.reverify_open_handle(connection, handle, cancelled=lambda: True)


def test_retained_handle_post_read_reverify_accepts_authentic_content(
    database: Database, tmp_path: Path
) -> None:
    store = ArtifactStore(database, tmp_path / "artifacts")
    artifact = store.commit(store.stage_bytes(b"authentic", "application/octet-stream"), None)
    with database.connect() as connection:
        with store.open_verified_with_connection(
            connection,
            artifact.id,
            sha256=artifact.sha256,
            byte_size=artifact.byte_size,
            media_type=artifact.media_type,
        ) as handle:
            assert handle.read(artifact.byte_size) == b"authentic"
            store.reverify_open_handle(connection, handle, cancelled=lambda: False)


def test_connection_resolve_sees_uncommitted_batch_row_and_preserves_transaction(
    database: Database, tmp_path: Path
) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"uncommitted-video")
    store = ArtifactStore(database, tmp_path / "artifacts")
    staged = store.stage(source, "video/mp4")

    def persist(connection: sqlite3.Connection, artifacts: tuple[Artifact, ...]) -> Path:
        artifact = artifacts[0]
        assert connection.in_transaction
        path = store.resolve_with_connection(
            connection,
            artifact.id,
            sha256=artifact.sha256,
            byte_size=artifact.byte_size,
            media_type=artifact.media_type,
        )
        assert connection.in_transaction
        return path

    resolved = store.commit_batch((staged,), persist)

    assert resolved.read_bytes() == b"uncommitted-video"


def test_batch_mutation_after_second_hash_rolls_back_domain_and_artifact(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable-content")
    store = ArtifactStore(database, tmp_path / "artifacts")
    staged = store.stage(source, "application/octet-stream")
    with database.connect() as connection:
        connection.execute("CREATE TABLE batch_marker(value TEXT)")
        connection.commit()
    original = store.reverify_open_handle
    calls = 0

    def mutate_after_second_hash(connection: sqlite3.Connection, handle: object) -> None:
        nonlocal calls
        calls += 1
        original(connection, handle)  # type: ignore[arg-type]
        if calls == 2:
            artifact = handle.artifact  # type: ignore[attr-defined]
            destination = store.root / artifact.relative_path
            replacement = destination.with_name(destination.name + ".replacement")
            replacement.write_bytes(b"changed")
            os.replace(replacement, destination)

    monkeypatch.setattr(store, "reverify_open_handle", mutate_after_second_hash)

    def persist(connection: sqlite3.Connection, artifacts: tuple[Artifact, ...]) -> None:
        connection.execute("INSERT INTO batch_marker VALUES ('published')")

    with pytest.raises(ArtifactIntegrityError, match="changed during verification"):
        store.commit_batch((staged,), persist)
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM batch_marker").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0


@pytest.mark.parametrize("replace_component", ["root", "shard"])
@pytest.mark.parametrize("replace_after_call", [1, 2])
def test_batch_ancestor_replacement_rolls_back_without_fd_leak(
    database: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_component: str,
    replace_after_call: int,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable-content")
    store = ArtifactStore(database, tmp_path / "artifacts")
    staged = store.stage(source, "application/octet-stream")
    with database.connect() as connection:
        connection.execute("CREATE TABLE batch_marker(value TEXT)")
        connection.commit()
    original = store.reverify_open_handle
    calls = 0
    detached = tmp_path / f"detached-{replace_component}"
    descriptors_before = len(tuple(Path("/dev/fd").iterdir()))

    def replace_after_second_hash(connection: sqlite3.Connection, handle: object) -> None:
        nonlocal calls
        calls += 1
        original(connection, handle)  # type: ignore[arg-type]
        if calls != replace_after_call:
            return
        if replace_component == "root":
            store.root.rename(detached)
            store.root.mkdir()
        else:
            shard = store.root / handle.relative_parts[0]  # type: ignore[attr-defined]
            shard.rename(detached)
            shard.mkdir()

    monkeypatch.setattr(store, "reverify_open_handle", replace_after_second_hash)

    def persist(connection: sqlite3.Connection, artifacts: tuple[Artifact, ...]) -> None:
        connection.execute("INSERT INTO batch_marker VALUES ('published')")

    with pytest.raises(ArtifactIntegrityError, match="artifact canonical path changed"):
        store.commit_batch((staged,), persist)
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM batch_marker").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
    with pytest.raises(ArtifactNotFoundError):
        store.resolve("uncommitted")
    assert len(tuple(Path("/dev/fd").iterdir())) == descriptors_before


def test_connection_resolve_requires_exact_stable_descriptor(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"stable-video")
    store = ArtifactStore(database, tmp_path / "artifacts")
    artifact = store.commit(store.stage(source, "video/mp4"), job_id=None)

    with database.connect() as connection:
        for expected_sha256, expected_size, expected_media_type in (
            ("0" * 64, artifact.byte_size, artifact.media_type),
            (artifact.sha256, artifact.byte_size + 1, artifact.media_type),
            (artifact.sha256, artifact.byte_size, "application/octet-stream"),
        ):
            with pytest.raises(ArtifactIntegrityError, match="descriptor mismatch") as raised:
                store.resolve_with_connection(
                    connection,
                    artifact.id,
                    sha256=expected_sha256,
                    byte_size=expected_size,
                    media_type=expected_media_type,
                )
            assert raised.value.__cause__ is None
            assert artifact.id not in str(raised.value)
            assert str(store.root) not in str(raised.value)

        original_verify = store._verify_file
        connection.execute("BEGIN IMMEDIATE")

        def mutate_descriptor(path: Path, registered: Artifact) -> None:
            original_verify(path, registered)
            connection.execute(
                "UPDATE artifacts SET media_type=? WHERE id=?",
                ("application/octet-stream", artifact.id),
            )

        monkeypatch.setattr(store, "_verify_file", mutate_descriptor)
        with pytest.raises(ArtifactIntegrityError, match="descriptor changed"):
            store.resolve_with_connection(
                connection,
                artifact.id,
                sha256=artifact.sha256,
                byte_size=artifact.byte_size,
                media_type=artifact.media_type,
            )
        assert connection.in_transaction
        connection.rollback()


def test_connection_resolve_rejects_foreign_and_closed_connections_path_free(
    database: Database, tmp_path: Path
) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"video")
    store = ArtifactStore(database, tmp_path / "artifacts")
    artifact = store.commit(store.stage(source, "video/mp4"), job_id=None)
    foreign_database = Database(tmp_path / "foreign.sqlite3")
    foreign_database.migrate()

    with foreign_database.connect() as foreign:
        foreign.execute("BEGIN IMMEDIATE")
        with pytest.raises(ArtifactIntegrityError, match="foreign database") as raised:
            store.resolve_with_connection(
                foreign,
                artifact.id,
                sha256=artifact.sha256,
                byte_size=artifact.byte_size,
                media_type=artifact.media_type,
            )
        assert foreign.in_transaction
        assert raised.value.__cause__ is None
        assert artifact.id not in str(raised.value)
        assert str(database.path) not in str(raised.value)
        foreign.rollback()

    closed = database.connect()
    closed.close()
    with pytest.raises(ArtifactIntegrityError, match="connection is closed") as raised:
        store.resolve_with_connection(
            closed,
            artifact.id,
            sha256=artifact.sha256,
            byte_size=artifact.byte_size,
            media_type=artifact.media_type,
        )
    assert raised.value.__cause__ is None
    assert artifact.id not in str(raised.value)
    assert str(database.path) not in str(raised.value)


def test_connection_resolve_missing_is_path_free_and_does_not_end_transaction(
    database: Database, tmp_path: Path
) -> None:
    store = ArtifactStore(database, tmp_path / "artifacts")
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(ArtifactNotFoundError, match="artifact not found") as raised:
            store.resolve_with_connection(
                connection,
                "sensitive-artifact-id",
                sha256="0" * 64,
                byte_size=1,
                media_type="video/mp4",
            )
        assert connection.in_transaction
        assert raised.value.__cause__ is None
        assert "sensitive-artifact-id" not in str(raised.value)
        assert str(store.root) not in str(raised.value)
        connection.rollback()


def test_discard_removes_only_the_exact_staged_handle(database: Database, tmp_path: Path) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"video")
    store = ArtifactStore(database, tmp_path / "artifacts")
    first = store.stage(source, "video/mp4")
    second = store.stage(source, "video/mp4")
    store.discard(first)
    assert not first.path.exists()
    assert second.path.exists()
    outside = StagedArtifact(tmp_path / "outside", first.sha256, first.byte_size, first.media_type)
    with pytest.raises(ArtifactIntegrityError, match="outside staging root"):
        store.discard(outside)


def test_identical_content_preserves_each_job_relationship(
    database: Database, tmp_path: Path
) -> None:
    service = ExperimentService.from_paths(database.path, tmp_path / "artifacts")
    jobs = service.create(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8),
        capabilities(),
    ).jobs
    source = tmp_path / "output.mp4"
    source.write_bytes(b"shared-video")

    first = service.artifacts.commit(service.artifacts.stage(source, "video/mp4"), jobs[0].id)
    second = service.artifacts.commit(service.artifacts.stage(source, "video/mp4"), jobs[1].id)
    repeated = service.artifacts.commit(service.artifacts.stage(source, "video/mp4"), jobs[1].id)

    assert first.id == second.id == repeated.id
    assert first.job_id == jobs[0].id
    assert second.job_id == repeated.job_id == jobs[1].id
    assert service.artifacts.list_for_job(jobs[0].id) == (first,)
    assert service.artifacts.list_for_job(jobs[1].id) == (second,)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1
        assert connection.execute("SELECT job_id FROM artifacts").fetchone()[0] is None
        assert connection.execute("SELECT COUNT(*) FROM job_artifacts").fetchone()[0] == 2


def test_cross_process_identical_content_preserves_each_job_relationship(
    database: Database, tmp_path: Path
) -> None:
    service = ExperimentService.from_paths(database.path, tmp_path / "artifacts")
    jobs = service.create(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8),
        capabilities(),
    ).jobs
    source = tmp_path / "output.mp4"
    source.write_bytes(b"shared-video")
    context = multiprocessing.get_context("fork")
    results: Queue[tuple[str, str]] = context.Queue()
    processes = [
        context.Process(
            target=_commit_in_process,
            args=(
                str(database.path),
                str(service.artifacts.root),
                str(source),
                job.id,
                results,
            ),
        )
        for job in jobs
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0

    artifact_ids = {results.get(timeout=1)[0] for _ in processes}
    assert len(artifact_ids) == 1
    assert [len(service.artifacts.list_for_job(job.id)) for job in jobs] == [1, 1]


def test_commit_rejects_staged_content_changed_after_hashing(
    database: Database, tmp_path: Path
) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"video-bytes")
    store = ArtifactStore(database, tmp_path / "artifacts")
    staged = store.stage(source, "video/mp4")
    staged.path.write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="staged artifact integrity check failed"):
        store.commit(staged, job_id=None)


def test_resolve_rejects_database_path_outside_root(database: Database, tmp_path: Path) -> None:
    store = ArtifactStore(database, tmp_path / "artifacts")
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO artifacts(
                 id, job_id, sha256, byte_size, media_type, relative_path, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("bad", None, "0" * 64, 1, "video/mp4", "../outside", "2026-01-01T00:00:00+00:00"),
        )

    with pytest.raises(ArtifactIntegrityError, match="escapes artifact root"):
        store.resolve("bad")


def test_resolve_rejects_internal_lock_path(database: Database, tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(database, root)
    lock_path = root / ".locks" / f"{'0' * 64}.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(b"")
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO artifacts(
                 id, job_id, sha256, byte_size, media_type, relative_path, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "lock",
                None,
                "0" * 64,
                0,
                "application/octet-stream",
                f".locks/{'0' * 64}.lock",
                "2026-01-01T00:00:00+00:00",
            ),
        )

    with pytest.raises(ArtifactIntegrityError, match="internal artifact path"):
        store.resolve("lock")


def test_commit_rejects_staged_path_outside_staging_root(
    database: Database, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video-bytes")
    store = ArtifactStore(database, tmp_path / "artifacts")
    staged = StagedArtifact(
        path=outside,
        sha256="79fd615a866fe7f9eb4da8d9c41ab57e3bd48056df42fd2c13e4d461a87afbe3",
        byte_size=11,
        media_type="video/mp4",
    )

    with pytest.raises(ArtifactIntegrityError, match="outside staging root"):
        store.commit(staged, job_id=None)

    assert outside.read_bytes() == b"video-bytes"


def test_failed_registration_preserves_staging_and_removes_new_destination(
    database: Database, tmp_path: Path
) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"video-bytes")
    root = tmp_path / "artifacts"
    store = ArtifactStore(database, root)
    staged = store.stage(source, "video/mp4")
    destination = root / staged.sha256[:2] / staged.sha256[2:4] / staged.sha256

    with pytest.raises(sqlite3.IntegrityError):
        store.commit(staged, job_id="job_missing")

    assert staged.path.read_bytes() == b"video-bytes"
    assert not destination.exists()
    assert list(destination.parent.glob(f".{staged.sha256}.*.tmp")) == []

    artifact = store.commit(staged, job_id=None)
    assert store.resolve(artifact.id) == destination
    assert not staged.path.exists()


def test_failed_atomic_batch_registration_removes_new_destination(
    database: Database, tmp_path: Path
) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"batch-video")
    store = ArtifactStore(database, tmp_path / "artifacts")
    staged = store.stage(source, "video/mp4")
    destination = store.root / staged.sha256[:2] / staged.sha256[2:4] / staged.sha256
    with database.connect() as connection:
        connection.execute(
            "CREATE TRIGGER reject_atomic_artifact BEFORE INSERT ON artifacts "
            "BEGIN SELECT RAISE(ABORT, 'forced registration failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced registration failure"):
        store.commit_batch((staged,), lambda _connection, artifacts: artifacts[0])

    assert staged.path.read_bytes() == b"batch-video"
    assert not destination.exists()


def test_failed_registration_preserves_staging_and_preexisting_destination(
    database: Database, tmp_path: Path
) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"video-bytes")
    root = tmp_path / "artifacts"
    store = ArtifactStore(database, root)
    staged = store.stage(source, "video/mp4")
    destination = root / staged.sha256[:2] / staged.sha256[2:4] / staged.sha256
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"video-bytes")

    with pytest.raises(sqlite3.IntegrityError):
        store.commit(staged, job_id="job_missing")

    assert staged.path.read_bytes() == b"video-bytes"
    assert destination.read_bytes() == b"video-bytes"
    assert list(destination.parent.glob(f".{staged.sha256}.*.tmp")) == []

    artifact = store.commit(staged, job_id=None)
    assert store.resolve(artifact.id) == destination
    assert not staged.path.exists()


def test_commit_waits_for_cross_process_sha_lock(database: Database, tmp_path: Path) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"video-bytes")
    root = tmp_path / "artifacts"
    staged = ArtifactStore(database, root).stage(source, "video/mp4")
    staged.path.unlink()
    destination = root / staged.sha256[:2] / staged.sha256[2:4] / staged.sha256
    lock_path = root / ".locks" / f"{staged.sha256}.lock"
    lock_path.parent.mkdir(parents=True)
    context = multiprocessing.get_context("fork")
    results: Queue[tuple[str, str]] = context.Queue()

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        process = context.Process(
            target=_commit_in_process,
            args=(str(database.path), str(root), str(source), None, results),
        )
        process.start()
        process.join(timeout=0.2)

        assert process.is_alive()
        assert not destination.exists()
        with pytest.raises(Empty):
            results.get_nowait()

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    process.join(timeout=5)
    assert process.exitcode == 0
    artifact_id, child_staged_path = results.get(timeout=1)
    assert ArtifactStore(database, root).resolve(artifact_id).read_bytes() == b"video-bytes"
    assert not Path(child_staged_path).exists()
    assert list(destination.parent.glob(f".{staged.sha256}.*.tmp")) == []
