import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.cli import main
from mlx_agent.task_store import TaskStore, default_root

ROOT = Path(__file__).resolve().parents[2]


class TaskCliTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.state = self.root / "state with spaces"
        self.store = TaskStore(self.state)
        self.request = {"schema": "task-request/1", "kind": "classification", "model_path": "/model", "parameters": {"state": "PRIVATE INPUT", "questions": {"route": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": ["billing", "other"]}}}}

    def run_cli(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["task", *args, "--state-dir", str(self.state), "--json"])
        return code, json.loads(output.getvalue())

    def create(self, result=None):
        record = self.store.create(self.request, {})
        if result is not None:
            reference = self.store.write_result(record["id"], result)
            record = self.store.transition(record["id"], ("starting",), {"state": "succeeded", "result": reference})
        return record

    def test_capabilities_and_empty_history_do_not_create_store(self):
        code, value = self.run_cli("capabilities")
        self.assertEqual(code, 0)
        self.assertEqual(value["operation"], "task.capabilities")
        self.assertEqual(len(value["data"]["tasks"]), 2)
        code, value = self.run_cli("list")
        self.assertEqual(value["data"], {"tasks": [], "has_more": False})
        self.assertFalse(self.state.exists())

    def test_status_and_cancel_use_the_same_persisted_id(self):
        record = self.create()
        code, value = self.run_cli("status", "--id", record["id"])
        self.assertEqual((code, value["data"]["task"]["id"]), (0, record["id"]))
        code, value = self.run_cli("cancel", "--id", record["id"])
        self.assertEqual(code, 0)
        self.assertTrue(value["data"]["task"]["cancel_requested"])
        self.assertNotIn("PRIVATE INPUT", json.dumps(value))
        command = [sys.executable, str(ROOT / "scripts/mlx-agent"), "task", "status", "--id", record["id"], "--state-dir", str(self.state), "--json"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["data"]["task"]["id"], record["id"])

    def test_unknown_id_invalid_limit_and_missing_model_are_actionable(self):
        for args, expected in ((("status", "--id", "bad"), "invalid_task_id"), (("list", "--limit", "0"), "invalid_limit")):
            code, value = self.run_cli(*args)
            self.assertEqual(code, 2)
            self.assertEqual(value["error"]["code"], expected)
            self.assertTrue(value["error"]["remediation"])
        request_path = self.root / "request.json"
        request_path.write_text(json.dumps(self.request))
        code, value = self.run_cli("start", "--request", str(request_path))
        self.assertEqual(code, 2)
        self.assertEqual(value["error"]["code"], "model_unreadable")

    def test_oversized_result_returns_valid_artifact_reference(self):
        record = self.create({"answers": {"large": "x" * 20000}})
        code, value = self.run_cli("result", "--id", record["id"])
        self.assertEqual(code, 0)
        self.assertFalse(value["data"]["inline"])
        self.assertIsNone(value["data"]["result"])
        self.assertEqual(value["data"]["artifact"]["bytes"], 20024)
        self.assertTrue(Path(value["data"]["artifact"]["path"]).is_file())
        self.assertLessEqual(len(json.dumps(value).encode()), 12288)

    def test_history_response_is_bounded_without_slicing_json(self):
        for _ in range(30):
            self.create()
        code, value = self.run_cli("list", "--limit", "100")
        self.assertEqual(code, 0)
        self.assertTrue(value["data"]["has_more"])
        self.assertLess(len(value["data"]["tasks"]), 30)
        self.assertLessEqual(len(json.dumps(value, separators=(",", ":")).encode()), 12288)
        self.assertEqual(value["data"]["tasks"][0]["id"], self.store.list(1)[0]["id"])

    def test_platform_default_follows_engine_policy(self):
        with patch("mlx_agent.task_store.sys.platform", "darwin"), patch("mlx_agent.task_store.Path.home", return_value=Path("/home/person")):
            self.assertEqual(default_root(), Path("/home/person/Library/Application Support/mlx-agent/tasks"))
        with patch("mlx_agent.task_store.sys.platform", "linux"), patch.dict("os.environ", {"XDG_STATE_HOME": "/state"}):
            self.assertEqual(default_root(), Path("/state/mlx-agent/tasks"))
