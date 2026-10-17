import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.task_contracts import TaskError
from mlx_agent.task_runtime import cancel_task, start_task, supervise, task_result, task_status
from mlx_agent.task_store import TaskStore


class TaskRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.state = self.root / "tasks"
        self.store = TaskStore(self.state)
        self.request = {"schema": "task-request/1", "kind": "classification", "model_path": str(self.root / "missing-model"), "parameters": {"state": "refund", "questions": {"route": {"type": "choice", "instructions": ["billing", "other"]}}}}

    def test_missing_model_does_not_create_a_running_task(self):
        path = self.root / "request.json"
        path.write_text(json.dumps(self.request))
        with self.assertRaises(TaskError):
            start_task(path, self.state)
        self.assertEqual(self.store.list(), [])

    def test_repeated_cancel_is_intent_until_execution_stops(self):
        record = self.store.create(self.request, {})
        first = cancel_task(record["id"], self.state)
        second = cancel_task(record["id"], self.state)
        self.assertTrue(first["cancel_requested"])
        self.assertTrue(second["cancel_requested"])
        self.assertEqual(second["state"], "starting")

    def test_lost_supervisor_is_failed_without_signaling_any_pid(self):
        record = self.store.create(self.request, {})
        self.store.transition(record["id"], ("starting",), {"state": "running", "supervisor": {"pid": 4242, "identity": "old"}})
        with patch("mlx_agent.task_runtime.process_identity", return_value="reused"), patch("mlx_agent.task_runtime.process_alive", return_value=True):
            result = task_status(record["id"], self.state)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"]["code"], "task_interrupted")

    def test_reading_unfinished_result_does_not_run_model(self):
        record = self.store.create(self.request, {})
        with self.assertRaises(TaskError) as caught:
            task_result(record["id"], self.state)
        self.assertEqual(caught.exception.code, "task_not_succeeded")
        self.assertEqual(self.store.get(record["id"])["state"], "starting")

    def test_lost_claim_does_not_fail_another_supervisors_task(self):
        record = self.store.create(self.request, {})
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
