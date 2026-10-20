"""Exercise retained evidence through real owned macOS filesystems."""

import base64
import errno
import importlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from mlx_agent.task_contracts import TaskError, encoded
from mlx_agent.task_store import TaskStore
from tests.unit.test_coding_capture_contracts import context, request

try:
    storage_module = importlib.import_module("mlx_agent.coding_capture_storage")
except ModuleNotFoundError:
    storage_module = None


@unittest.skipUnless(sys.platform == "darwin", "Darwin capture volumes")
class CodingCaptureStorageTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(storage_module, "Owned capture storage is missing")
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name).resolve()
        self.store = TaskStore(self.parent / "tasks")
        self.identifier = str(uuid4())
        self.limits = request()["limits"]
        self.storage = storage_module.CaptureStorage.create(self.store, self.identifier, self.limits)
        self.addCleanup(self.clean_volume)
        self.request = request()
        self.context = context()
        self.request["workbench_context"] = self.context["workbench_context"]
        reference = self.storage.accept_request(self.request)
        self.record = self.store.create_capture(self.identifier, reference, self.context, "b" * 64)

    def clean_volume(self):
        if self.storage.root.exists():
            storage_module.CaptureVolume.remove(self.storage.root)

    def fail_capture(self, unresolved=False):
        return self.store.transition(self.identifier, ("starting", "running"), {
            "state": "failed", "error": {"code": "client_failed", "message": "Client stopped.", "remediation": "Review retained evidence."},
            "client_outcome": "failed", "attribution": "failed", "execution_unresolved": unresolved})

    def manifest(self):
        session = self.storage.write_artifact("session.json", b'{"session":"ses_root","prompt":"secret code"}')
        transport = self.storage.write_artifact("transport.json", b'{"observed_session":"ses_root"}')
        return {"schema": "coding-capture-result/1", "task_id": self.identifier,
                "request_sha256": self.record["request"]["sha256"], "preview_hash": "b" * 64, "context": self.context,
                "session": {"id": "ses_root", "parent_id": None, "children": []},
                "requests": [{"artifact": transport, "session_id": "ses_root", "parent_session_id": None,
                              "project_id": "project", "request_model": "local", "response_model": "local"}],
                "artifacts": [session, transport], "client_exit_code": 0, "elapsed_seconds": 1.0,
                "completed_at": "2026-10-20T12:00:00+00:00", "attribution": "verified"}

    def test_wal_tmp_and_many_files_share_one_budget(self):
        with self.storage.active() as root:
            database = root / "profile/client.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("CREATE TABLE data (value BLOB)")
                connection.execute("INSERT INTO data VALUES (?)", (b"x" * (1024 * 1024),))
                connection.commit()
                self.assertEqual(database.stat().st_dev, (root / "tmp").stat().st_dev)
                self.assertEqual((database.parent / "client.sqlite3-wal").stat().st_dev, root.stat().st_dev)
            connection.close()
            for index in range(100):
                (root / "tmp" / str(index)).write_bytes(b"x")
            descriptor = os.open(root / "tmp/fill", os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                with self.assertRaises(OSError) as caught:
                    for _ in range(256):
                        os.write(descriptor, b"x" * (1024 * 1024))
                self.assertIn(caught.exception.errno, (errno.ENOSPC, errno.EDQUOT))
            finally:
                os.close(descriptor)
            with self.assertRaises(TaskError):
                self.storage.write_artifact("must-not-publish.json", b"new evidence")
        self.assertIsNone(self.store.get(self.identifier)["result"])

    def test_artifact_limit_is_33554432(self):
        with self.assertRaises(TaskError) as caught:
            self.storage.write_artifact("oversized.json", b"x" * (33554432 + 1))
        self.assertEqual(caught.exception.code, "capture_artifact_limit")
        with self.storage.active() as root:
            self.assertFalse((root / "artifacts/oversized.json").exists())

    def test_total_limit_is_134217728(self):
        with self.storage.active() as root:
            info = os.statvfs(root)
            self.assertLessEqual(info.f_blocks * info.f_frsize + 65536, 134217728)
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
        with self.assertRaises(TaskError):
            storage_module.CaptureStorage.create(self.store, str(uuid4()), dict(self.limits, total_bytes=134217729))

    def test_external_metadata_reserve_is_65536(self):
        marker = json.loads((self.storage.root / "volume.json").read_bytes())
        self.assertEqual(marker["capacity_bytes"], 134217728 - 65536)
        outside = [path for path in self.storage.root.iterdir() if path.name not in ("capture.sparsebundle", "mounted")]
        self.assertLessEqual(sum(path.stat().st_size for path in outside) + len(encoded(self.record)), 65536)
        self.assertFalse((self.store.root / "artifacts").exists())

    def test_repeated_transitions_do_not_grow_external_history(self):
        for _ in range(25):
            self.store.request_cancel(self.identifier)
        with sqlite3.connect(self.store.database) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM tasks").fetchone()[0], 1)
            self.assertLessEqual(connection.execute("SELECT length(record) FROM tasks").fetchone()[0], 8192)
        self.assertEqual(set(path.name for path in self.store.root.iterdir()), {"tasks.sqlite3", "captures"})

    def test_enospc_cannot_publish_success(self):
        manifest = self.manifest()
        with self.storage.active(), patch.object(self.storage, "_atomic", side_effect=OSError(errno.ENOSPC, "full")):
            with self.assertRaises(TaskError):
                self.storage.publish(manifest)
        self.assertIsNone(self.store.get(self.identifier)["result"])
        with self.assertRaises(TaskError):
            self.storage.verify()

    def test_unreadable_result_size_cannot_publish_success(self):
        manifest = self.manifest()
        manifest["requests"] *= 100
        with self.assertRaises(TaskError):
            self.storage.publish(manifest)
        with self.storage.active() as root:
            self.assertFalse((root / "artifacts/result.json").exists())
            self.assertTrue((root / "profile").exists())

    def test_manifest_commit_precedes_profile_cleanup(self):
        manifest = self.manifest()
        with self.storage.active() as root:
            (root / "profile/recover.db").write_bytes(b"recovery")
        original = self.storage._cleanup_profile
        observed = []
        def cleanup(root):
            observed.append(json.loads((root / "artifacts/result.json").read_bytes()))
            original(root)
        with patch.object(self.storage, "_cleanup_profile", side_effect=cleanup):
            result = self.storage.publish(manifest)
        self.assertEqual(observed, [manifest])
        reopened = storage_module.CaptureStorage(self.store, self.identifier)
        verified = reopened.verify(result["sha256"])
        self.assertEqual(json.loads(verified["manifest_utf8"]), manifest)
        with reopened.active() as root:
            self.assertFalse((root / "profile").exists())
            self.assertEqual((root / "artifacts/session.json").stat().st_mode & 0o777, 0o600)

    def test_restart_between_each_publication_step_is_recoverable(self):
        manifest = self.manifest()
        with patch.object(self.storage, "_atomic", side_effect=OSError(errno.EIO, "before commit")):
            with self.assertRaises(TaskError):
                self.storage.publish(manifest)
        with patch.object(self.storage, "_cleanup_profile", side_effect=OSError(errno.EIO, "after commit")):
            with self.assertRaises(TaskError):
                self.storage.publish(manifest)
        reopened = storage_module.CaptureStorage(self.store, self.identifier)
        self.assertEqual(json.loads(reopened.verify()["manifest_utf8"]), manifest)
        result = reopened.publish(manifest)
        self.assertEqual(reopened.verify(result["sha256"])["sha256"], result["sha256"])
        self.assertEqual(self.store.get(self.identifier)["state"], "starting")

    def test_bounded_reads_and_export_keep_original_hash(self):
        data = b"hello" * 4000
        ref = self.storage.write_artifact("diagnostics.txt", data)
        page = self.storage.read_artifact("diagnostics.txt", 0)
        self.assertEqual(base64.b64decode(page["data_base64"]), data[:6144])
        self.assertEqual(page["next_offset"], 6144)
        self.assertLess(len(encoded(page)), 12288)
        destination = self.parent / "export.txt"
        self.storage.export_artifact("diagnostics.txt", destination, sanitize=False)
        self.assertEqual(destination.read_bytes(), data)
        self.assertEqual(self.storage.read_artifact("diagnostics.txt", 6144)["sha256"], ref["sha256"])
        with self.assertRaises(TaskError):
            self.storage.export_artifact("diagnostics.txt", destination, sanitize=False)
        for offset, length in ((-1, 6144), (True, 6144), (0, 6145)):
            with self.assertRaises(TaskError):
                self.storage.read_artifact("diagnostics.txt", offset, length)

    def test_symlink_or_changed_artifact_refuses_read_export_remove(self):
        manifest = self.manifest()
        self.storage.publish(manifest)
        self.fail_capture()
        with self.storage.active() as root:
            artifact = root / "artifacts/session.json"
            original = artifact.read_bytes()
            artifact.write_bytes(b"changed")
        for operation in (lambda: self.storage.read_artifact("session.json", 0),
                          lambda: self.storage.export_artifact("session.json", self.parent / "copy.json", False),
                          self.storage.remove_preview):
            with self.assertRaises(TaskError):
                operation()
        with self.storage.active() as root:
            artifact = root / "artifacts/session.json"
            artifact.write_bytes(original)
            artifact.unlink()
            artifact.symlink_to(self.parent / "outside")
        try:
            with self.assertRaises(TaskError):
                self.storage.read_artifact("session.json", 0)
            with self.assertRaises(TaskError):
                self.storage.remove_preview()
        finally:
            with self.storage.active() as root:
                (root / "artifacts/session.json").unlink()

    def test_active_capture_cannot_be_removed(self):
        with self.assertRaises(TaskError):
            self.storage.remove_preview()
        self.assertTrue(self.storage.root.exists())

    def test_terminal_unresolved_process_refuses_removal(self):
        self.fail_capture(unresolved=True)
        with self.assertRaises(TaskError):
            self.storage.remove_preview()
        self.assertTrue(self.storage.root.exists())

    def test_stale_remove_preview_changes_nothing(self):
        self.storage.write_artifact("diagnostics.txt", b"first")
        self.fail_capture()
        preview = self.storage.remove_preview()
        self.storage.write_artifact("more.txt", b"new")
        with self.assertRaises(TaskError):
            self.storage.remove(preview["preview_hash"])
        self.assertTrue(self.storage.root.exists())
        self.assertIsNone(self.store.get(self.identifier)["artifacts_removed_at"])

    def test_removal_lock_covers_preview_through_volume_delete(self):
        self.fail_capture()
        preview = self.storage.remove_preview()
        original = storage_module.CaptureVolume.remove
        competing = storage_module.CaptureStorage(self.store, self.identifier)
        def race(root):
            competing.write_artifact("late.txt", b"concurrent evidence")
            original(root)
        with patch.object(storage_module.CaptureVolume, "remove", side_effect=race):
            with self.assertRaises(TaskError):
                self.storage.remove(preview["preview_hash"])
        self.assertTrue(self.storage.root.exists())

    def test_failed_remove_retains_recovery_metadata(self):
        self.fail_capture()
        preview = self.storage.remove_preview()
        with patch.object(storage_module.CaptureVolume, "remove", side_effect=TaskError("capture_volume_unresolved", "detach failed")):
            with self.assertRaises(TaskError):
                self.storage.remove(preview["preview_hash"])
        self.assertTrue(self.storage.root.exists())
        self.assertLessEqual(self.storage.tombstone.stat().st_size, 2048)
        self.assertIsNone(self.store.get(self.identifier)["artifacts_removed_at"])
        self.storage.remove(preview["preview_hash"])
        self.assertFalse(self.storage.root.exists())
        self.assertEqual(self.store.get(self.identifier)["state"], "failed")
        self.assertIsNotNone(self.store.get(self.identifier)["artifacts_removed_at"])
        self.assertLessEqual(self.storage.tombstone.stat().st_size, 2048)
        with self.assertRaises(TaskError):
            self.storage.read_artifact("request.json", 0)

    def test_sanitized_export_does_not_replace_original(self):
        manifest = self.manifest()
        result = self.storage.publish(manifest)
        destination = self.parent / "share.json"
        self.storage.export_artifact("session.json", destination, sanitize=True)
        shared = json.loads(destination.read_bytes())
        self.assertTrue(shared["content_removed"])
        for private in ("secret code", self.context["project"]["path"], self.request["prompt"], "Authorization"):
            self.assertNotIn(private, destination.read_text())
        self.assertEqual(self.storage.verify(result["sha256"])["sha256"], result["sha256"])
