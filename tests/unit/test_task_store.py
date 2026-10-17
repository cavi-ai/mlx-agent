import json
import os
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.task_contracts import TaskError
from mlx_agent.task_store import TaskStore
from tests.task_fixtures import identity_evidence


class TaskStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "tasks"
        self.store = TaskStore(self.root)
        self.request = {"schema": "task-request/1", "kind": "classification", "model_path": "/models/local", "parameters": {"state": "refund", "questions": {"route": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": ["billing", "other"]}}}}

    def create(self):
        return self.store.create(self.request, identity_evidence(self.request["model_path"]))

    def start_execution(self, task):
        self.store.transition(task["id"], ("starting",), {"supervisor": {"pid": 42, "identity": "supervisor"}})
        return self.store.transition(task["id"], ("starting",), {
            "state": "running", "execution": {"pid": 43, "identity": "worker"}})

    def test_read_does_not_create_store(self):
        self.assertEqual(self.store.list(20), [])
        with self.assertRaises(TaskError):
            self.store.get("11111111-1111-4111-8111-111111111111")
        self.assertFalse(self.root.exists())

    def test_restart_preserves_terminal_record(self):
        task = self.create()
        task = self.start_execution(task)
        result = self.store.write_result(task["id"], {"schema": "decide/1", "answers": {"route": "billing"}})
        completed = self.store.transition(task["id"], ("running",), {"state": "succeeded", "result": result})
        self.assertEqual(TaskStore(self.root).get(task["id"]), completed)
        self.assertEqual(completed["state"], "succeeded")
        self.assertIsNotNone(completed["completed_at"])
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.root / "tasks.sqlite3").stat().st_mode & 0o777, 0o600)

    def test_terminal_transition_cannot_be_overwritten(self):
        task = self.create()
        failed = self.store.transition(task["id"], ("starting",), {"state": "failed", "error": {"code": "launch_failed", "message": "launch failed", "remediation": "Retry."}})
        with self.assertRaises(TaskError):
            self.store.transition(task["id"], ("failed",), {"state": "running"})
        self.assertEqual(self.store.request_cancel(task["id"]), failed)
        with self.assertRaises(TaskError):
            self.store.transition(task["id"], ("failed",), {"model_path": "/changed"})

    def test_supervisor_cannot_be_replaced(self):
        task = self.create()
        self.store.transition(task["id"], ("starting",), {"supervisor": {"pid": 42, "identity": "original"}})
        with self.assertRaises(TaskError):
            self.store.transition(task["id"], ("starting",), {"supervisor": {"pid": 43, "identity": "replacement"}})
        self.assertEqual(self.store.get(task["id"])["supervisor"]["pid"], 42)

    def test_concurrent_cancel_preserves_result(self):
        task = self.create()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: TaskStore(self.root).request_cancel(task["id"]), range(12)))
        self.assertTrue(self.store.get(task["id"])["cancel_requested"])
        result = self.store.write_result(task["id"], {"answers": {"route": "billing"}})
        self.store.transition(task["id"], ("starting",), {"state": "cancelled"})
        self.assertEqual(json.loads(Path(result["path"]).read_text())["answers"]["route"], "billing")

    def test_cancellation_wins_before_success_publication(self):
        task = self.create()
        result = self.store.write_result(task["id"], {"answers": {"route": "billing"}})
        self.store.request_cancel(task["id"])
        with self.assertRaises(TaskError) as caught:
            self.store.transition(task["id"], ("starting",), {"state": "succeeded", "result": result})
        self.assertEqual(caught.exception.code, "task_cancel_requested")
        self.assertEqual(self.store.get(task["id"])["state"], "starting")

    def test_unknown_schema_and_corrupt_database_are_not_reset(self):
        self.create()
        database = self.root / "tasks.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA user_version = 99")
        before = database.read_bytes()
        with self.assertRaises(TaskError):
            self.store.list(20)
        self.assertEqual(database.read_bytes(), before)
        database.write_bytes(b"corrupt")
        with self.assertRaises(TaskError):
            self.store.list(20)
        self.assertEqual(database.read_bytes(), b"corrupt")

    def test_store_symlinks_and_foreign_permissions_are_rejected(self):
        external = self.root.parent / "external"
        external.mkdir(mode=0o700)
        self.root.symlink_to(external, target_is_directory=True)
        with self.assertRaises(TaskError):
            self.create()
        self.assertEqual(list(external.iterdir()), [])
        self.root.unlink()
        self.root.mkdir(mode=0o755)
        with self.assertRaises(TaskError):
            self.create()

    def test_request_snapshot_survives_source_changes(self):
        task = self.create()
        self.request["parameters"]["state"] = "changed"
        self.assertEqual(self.store.read_request(task["id"])["parameters"]["state"], "refund")
        request_path = self.store.artifact_path(task["id"], "request.json")
        request_path.write_text("{}")
        with self.assertRaises(TaskError):
            self.store.read_request(task["id"])

    def test_disk_failure_does_not_publish_record(self):
        with patch("mlx_agent.task_store.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(TaskError):
                self.create()
        self.assertEqual(self.store.list(20), [])

    def test_artifact_paths_cannot_escape_and_results_are_immutable(self):
        task = self.create()
        for name in ("../outside", "/outside", ".hidden", "a/b"):
            with self.subTest(name=name), self.assertRaises(TaskError):
                self.store.artifact_path(task["id"], name)
        first = self.store.write_result(task["id"], {"answers": {"x": 1}})
        with self.assertRaises(TaskError):
            self.store.write_result(task["id"], {"answers": {"x": 2}})
        self.assertEqual(json.loads(Path(first["path"]).read_text())["answers"], {"x": 1})
