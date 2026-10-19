import io
import json
import sqlite3
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.cli import main
from mlx_agent.backends import BackendError, load_manifests
from mlx_agent.mcp_server import TOOL_NAME, handle_request
from mlx_agent.task_identity import capture_identity
from mlx_agent.task_store import TaskStore, default_root
from tests.task_fixtures import identity_evidence

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

    def create(self, result=None, identity=None):
        record = self.store.create(self.request, identity or identity_evidence(self.request["model_path"]))
        if result is not None:
            self.store.transition(record["id"], ("starting",), {"supervisor": {"pid": 42, "identity": "supervisor"}})
            record = self.store.transition(record["id"], ("starting",), {
                "state": "running", "execution": {"pid": 43, "identity": "worker"}})
            reference = self.store.write_result(record["id"], result)
            record = self.store.transition(record["id"], ("running",), {"state": "succeeded", "result": reference})
        return record

    def test_capabilities_and_empty_history_do_not_create_store(self):
        code, value = self.run_cli("capabilities")
        self.assertEqual(code, 0)
        self.assertEqual(value["operation"], "task.capabilities")
        tasks = {task["kind"]: task for task in value["data"]["tasks"]}
        self.assertEqual(set(tasks), {"classification", "music_generation", "transcribe", "speak", "image_generation", "image_understanding",
                                      "video_generation", "video_understanding"})
        self.assertEqual((tasks["transcribe"]["request_schema"], tasks["transcribe"]["run_schema"]),
                         ("task-request/2", "task-run/2"))
        self.assertEqual((tasks["speak"]["request_schema"], tasks["speak"]["run_schema"]),
                         ("task-request/2", "task-run/2"))
        self.assertEqual((tasks["image_understanding"]["request_schema"], tasks["image_understanding"]["run_schema"],
                          tasks["image_understanding"]["result_schema"]),
                         ("task-request/2", "task-run/2", "describe/1"))
        self.assertTrue(all(task["model_verification"] for task in value["data"]["tasks"]))
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

    def test_corrupt_nested_record_returns_bounded_cli_errors_without_writing_store(self):
        record = self.create()
        with sqlite3.connect(self.store.database) as connection:
            raw = connection.execute("SELECT record FROM tasks WHERE id = ?", (record["id"],)).fetchone()[0]
            value = json.loads(raw)
            value["supervisor"] = {"pid": "bad"}
            connection.execute("UPDATE tasks SET record = ? WHERE id = ?", (json.dumps(value), record["id"]))
        before = self.store.database.read_bytes()
        command = [sys.executable, str(ROOT / "scripts/mlx-agent"), "task"]
        cases = [("status", "--id", record["id"]), ("list",), ("result", "--id", record["id"]), ("cancel", "--id", record["id"])]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                completed = subprocess.run(command + list(arguments) + ["--state-dir", str(self.state), "--json"],
                                           capture_output=True, text=True, timeout=10, cwd=ROOT)
                self.assertTrue(completed.stdout, completed.stderr)
                payload = json.loads(completed.stdout)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertEqual(payload["status"], "error")
                self.assertEqual(payload["error"]["code"], "invalid_task_record")
                self.assertTrue(payload["error"]["message"])
                self.assertTrue(payload["error"]["remediation"])
                self.assertLessEqual(len(completed.stdout.encode()), 12288)
                self.assertEqual(self.store.database.read_bytes(), before)

    def test_platform_default_follows_engine_policy(self):
        with patch("mlx_agent.task_store.sys.platform", "darwin"), patch("mlx_agent.task_store.Path.home", return_value=Path("/home/person")):
            self.assertEqual(default_root(), Path("/home/person/Library/Application Support/mlx-agent/tasks"))
        with patch("mlx_agent.task_store.sys.platform", "linux"), patch.dict("os.environ", {"XDG_STATE_HOME": "/state"}):
            self.assertEqual(default_root(), Path("/state/mlx-agent/tasks"))

    def test_verify_model_checks_saved_identity_without_mutating_history(self):
        model = self.root / "model"
        model.mkdir()
        config = model / "config.json"
        config.write_text('{"model_type":"laya"}')
        executable = self.root / "python"
        executable.write_bytes(b"runtime")
        self.request["model_path"] = str(model)
        identity = capture_identity(model, load_manifests()["mlx-embeddings"], executable)
        record = self.create({"answers": {"route": {"choice": "billing"}}}, identity)
        before = self.store.database.read_bytes()
        code, value = self.run_cli("verify-model", "--id", record["id"])
        self.assertEqual(code, 0)
        self.assertEqual(value["data"], {"id": record["id"], "model_path": str(model),
                                         "model_fingerprint": identity["model_fingerprint"], "matches": True})
        response = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": TOOL_NAME, "arguments": {"capability": "task", "arguments":
                'verify-model --id {0} --state-dir "{1}"'.format(record["id"], self.state)}}})
        mcp = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(mcp["exit_code"], 0)
        self.assertEqual(json.loads(mcp["stdout"])["data"], value["data"])
        self.assertEqual(self.store.database.read_bytes(), before)
        for mutation in (lambda: config.write_text('{"model_type":"changed"}'),
                         lambda: (config.unlink(), model.rmdir())):
            mutation()
            code, value = self.run_cli("verify-model", "--id", record["id"])
            self.assertEqual(code, 2)
            self.assertIn(value["error"]["code"], ("identity_changed", "identity_unavailable"))
            self.assertEqual(self.store.database.read_bytes(), before)
            code, result = self.run_cli("result", "--id", record["id"])
            self.assertEqual(code, 0)
            self.assertEqual(result["data"]["result"]["answers"]["route"]["choice"], "billing")

    def test_verify_model_missing_record_does_not_create_storage(self):
        code, value = self.run_cli("verify-model", "--id", "11111111-1111-4111-8111-111111111111")
        self.assertEqual(code, 2)
        self.assertEqual(value["error"]["code"], "task_not_found")
        self.assertFalse(self.state.exists())

    def test_verify_model_manifest_failures_preserve_structured_errors_and_store(self):
        record = self.create()
        before = self.store.database.read_bytes()
        errors = (OSError("unreadable manifest"), ValueError("invalid JSON"),
                  BackendError("manifest_invalid", "Invalid backend", "Repair the manifest."))
        for error in errors:
            with self.subTest(error=type(error).__name__), patch("mlx_agent.backends.load_manifests", side_effect=error):
                code, value = self.run_cli("verify-model", "--id", record["id"])
                self.assertEqual(code, 2)
                self.assertEqual(value["error"]["code"], "identity_unavailable")
                self.assertLessEqual(len(json.dumps(value).encode()), 12288)
                self.assertEqual(self.store.database.read_bytes(), before)
