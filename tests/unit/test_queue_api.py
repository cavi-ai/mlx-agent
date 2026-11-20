import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mlx_agent.queue_cli import execute, _guard
from mlx_agent.convert_queue import QueueStore
from mlx_agent import queue_bridge

class QueueAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "agent" / "queue.json"
        self.receipts = self.root / "receipts"
    def call(self, **request):
        return execute(request, self.target, self.receipts)
    def test_persistent_mutations_and_closed_fields(self):
        result = self.call(operation="enqueue", kind="repo", preview_hash="a"*64, q_bits=4, repo="owner/model")
        item = result["item"]
        self.assertEqual(self.call(operation="snapshot")["conversion_queue"]["items"][0], item)
        self.assertTrue(self.call(operation="cancel", id=item["id"])["changed"])
        self.assertEqual(self.call(operation="snapshot")["conversion_queue"]["items"], [])
        with self.assertRaises(ValueError):
            self.call(operation="snapshot", hidden=True)
    def test_adoption_keeps_source_and_exact_backup(self):
        source = self.root / "legacy.json"
        QueueStore(source).save([])
        original = source.read_bytes()
        preview = self.call(operation="adopt-preview", source=str(source))
        self.assertFalse(self.target.exists())
        adopted = self.call(operation="adopt", source=str(source), preview_hash=preview["preview_hash"])
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(Path(adopted["backup"]).read_bytes(), original)
        with self.assertRaises(ValueError):
            self.call(operation="adopt", source=str(source), preview_hash=preview["preview_hash"])
    def test_stale_preview_refuses_without_state_write(self):
        source = self.root / "legacy.json"
        QueueStore(source).save([])
        preview = self.call(operation="adopt-preview", source=str(source))
        source.write_text(source.read_text()+" ")
        with self.assertRaises(ValueError):
            self.call(operation="adopt", source=str(source), preview_hash=preview["preview_hash"])
        self.assertFalse(self.target.exists())
    def test_failed_publication_can_retry_retained_backup(self):
        source = self.root / "legacy.json"
        QueueStore(source).save([])
        preview = self.call(operation="adopt-preview", source=str(source))
        with patch.object(QueueStore, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.call(operation="adopt", source=str(source), preview_hash=preview["preview_hash"])
        self.assertFalse(self.target.exists())
        self.call(operation="adopt", source=str(source), preview_hash=preview["preview_hash"])
        self.assertTrue(self.target.exists())
    def test_process_lock_refuses_concurrent_writer(self):
        with _guard(self.target):
            with self.assertRaises(BlockingIOError):
                self.call(operation="snapshot")
    def test_receipt_roots_are_isolated_between_operations(self):
        for name in ("one", "two"):
            root = self.root / name / ".mlx-agent-receipts" / "convert"
            root.mkdir(parents=True)
            (root / "receipt.json").write_text(json.dumps({
                "kind": "convert", "pid": 1, "argv": [], "repo": name,
                "q_bits": 4, "out": "/unused", "completed_at": "2026-01-01",
            }))
        for name in ("one", "two", "one"):
            jobs = queue_bridge.jobs(str(self.root), receipts_dir=self.root / name)["jobs"]
            self.assertEqual([job["repo"] for job in jobs], [name])
        execute({"operation": "run"}, self.target, self.root / "one")
        original_cwd = os.getcwd()
        try:
            os.chdir(self.root / "two")
            jobs = queue_bridge.jobs(str(self.root))["jobs"]
        finally:
            os.chdir(original_cwd)
        self.assertEqual([job["repo"] for job in jobs], ["two"])
    def test_unsafe_input_is_preserved(self):
        source = self.root / "legacy.json"
        source.write_text('{"schema_version":"999","items":[]}')
        with self.assertRaises(ValueError):
            self.call(operation="adopt-preview", source=str(source))
        self.assertTrue(source.exists())
        self.assertFalse(self.target.exists())
