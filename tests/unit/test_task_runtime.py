import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.task_contracts import TaskError
from mlx_agent.task_runtime import cancel_task, start_task, supervise, task_result, task_status
from mlx_agent.task_store import TaskStore
from tests.task_fixtures import identity_evidence


class TaskRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.state = self.root / "tasks"
        self.store = TaskStore(self.state)
        self.request = {"schema": "task-request/1", "kind": "classification", "model_path": str(self.root / "missing-model"), "parameters": {"state": "refund", "questions": {"route": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": ["billing", "other"]}}}}

    def create(self):
        return self.store.create(self.request, identity_evidence(self.request["model_path"]))

    def test_missing_model_does_not_create_a_running_task(self):
        path = self.root / "request.json"
        path.write_text(json.dumps(self.request))
        with self.assertRaises(TaskError):
            start_task(path, self.state)
        self.assertEqual(self.store.list(), [])

    def test_repeated_cancel_is_intent_until_execution_stops(self):
        record = self.create()
        first = cancel_task(record["id"], self.state)
        second = cancel_task(record["id"], self.state)
        self.assertTrue(first["cancel_requested"])
        self.assertTrue(second["cancel_requested"])
        self.assertEqual(second["state"], "starting")

    def test_lost_supervisor_is_failed_without_signaling_any_pid(self):
        record = self.create()
        self.store.transition(record["id"], ("starting",), {"supervisor": {"pid": 4242, "identity": "old"}})
        self.store.transition(record["id"], ("starting",), {
            "state": "running", "execution": {"pid": 4343, "identity": "worker"}})
        with patch("mlx_agent.task_runtime.process_identity", return_value="reused"), patch("mlx_agent.task_runtime.process_alive", return_value=True):
            result = task_status(record["id"], self.state)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"]["code"], "task_interrupted")

    def test_reading_unfinished_result_does_not_run_model(self):
        record = self.create()
        with self.assertRaises(TaskError) as caught:
            task_result(record["id"], self.state)
        self.assertEqual(caught.exception.code, "task_not_succeeded")
        self.assertEqual(self.store.get(record["id"])["state"], "starting")

    def test_lost_claim_does_not_fail_another_supervisors_task(self):
        record = self.create()
        original = TaskStore.transition
        def raced_claim(store, identifier, expected, changes):
            if "supervisor" in changes:
                original(store, identifier, expected, {"supervisor": {"pid": 4242, "identity": "other-supervisor"}})
                raise TaskError("task_state_changed", "Already claimed.")
            return original(store, identifier, expected, changes)
        with patch.object(TaskStore, "transition", raced_claim):
            supervise(record["id"], self.state)
        current = self.store.get(record["id"])
        self.assertEqual(current["state"], "starting")
        self.assertEqual(current["supervisor"]["identity"], "other-supervisor")

    def test_process_probe_failure_is_actionable_and_does_not_rewrite_record(self):
        record = self.create()
        self.store.transition(record["id"], ("starting",), {"supervisor": {"pid": 4242, "identity": "supervisor"}})
        before = self.store.database.read_bytes()
        with patch("mlx_agent.task_runtime.process_identity", side_effect=OSError("process table unavailable")):
            with self.assertRaises(TaskError) as caught:
                task_status(record["id"], self.state)
        self.assertEqual(caught.exception.code, "task_process_inspection_failed")
        self.assertTrue(caught.exception.remediation)
        self.assertEqual(self.store.database.read_bytes(), before)

    def test_unavailable_live_process_identity_preserves_running_record(self):
        record = self.create()
        self.store.transition(record["id"], ("starting",), {"supervisor": {"pid": 4242, "identity": "supervisor"}})
        self.store.transition(record["id"], ("starting",), {
            "state": "running", "execution": {"pid": 4343, "identity": "worker"}})
        before = self.store.database.read_bytes()
        with patch("mlx_agent.task_runtime.process_identity", return_value=None), patch(
                "mlx_agent.task_runtime.process_alive", return_value=True):
            with self.assertRaises(TaskError) as caught:
                task_status(record["id"], self.state)
        self.assertEqual(caught.exception.code, "task_identity_unavailable")
        self.assertEqual(self.store.database.read_bytes(), before)
        self.assertEqual(self.store.get(record["id"])["state"], "running")
