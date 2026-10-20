"""Coding contracts preserve inference compatibility and bounded provenance."""

import importlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from mlx_agent.task_contracts import TaskError, encoded, validate_record, capabilities
from mlx_agent.task_store import TaskStore
from tests.task_fixtures import identity_evidence

try:
    capture = importlib.import_module("mlx_agent.coding_capture_contracts")
except ModuleNotFoundError:
    capture = None


def reference(name="request.json"):
    return {"name": name, "sha256": "a" * 64, "bytes": 100}


def context():
    return {
        "serve": {"receipt": {"path": "/state/serve/8123.json", "sha256": "b" * 64, "bytes": 500},
                  "base_url": "http://127.0.0.1:8123/v1", "model_path": "/models/local",
                  "model_fingerprint": "c" * 64, "runtime": "mlx_lm",
                  "process": {"pid": 42, "identity": "server", "started_at": [100, 200]}},
        "client": {"path": "/usr/local/bin/opencode", "file_identity": [1, 2, 3, 4],
                   "version": "2.0.27", "configuration_sha256": "d" * 64, "policy_sha256": "e" * 64},
        "project": {"path": "/projects/example", "file_identity": [1, 5], "context_sha256": "f" * 64},
        "workbench_context": {"model_signature": "signature", "environment_fingerprint": "environment",
                              "workload_id": str(uuid4())}}


def request():
    identity = context()
    return {"schema": "coding-capture-request/1", "kind": "coding_capture", "model_path": "/models/local",
            "serve_receipt": identity["serve"]["receipt"], "client_path": identity["client"]["path"],
            "project_path": identity["project"]["path"], "prompt": "Implement normalize_tags.",
            "limits": {"artifact_bytes": 33554432, "total_bytes": 134217728},
            "workbench_context": identity["workbench_context"]}


class CodingCaptureContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(capture, "Coding capture contracts are missing")

    def test_request_is_strict_and_preserves_original_context(self):
        value = request()
        accepted = capture.validate_capture_request(value)
        self.assertEqual(accepted, value)
        value["workbench_context"]["model_signature"] = "changed"
        self.assertEqual(accepted["workbench_context"]["model_signature"], "signature")
        for changes in ({"model_path": "relative"}, {"project_path": "/project/../outside"},
                        {"client_path": "/bin/tool\n"}, {"prompt": " "}, {"schema": "coding-capture-request/99"},
                        {"unknown": True}, {"serve_receipt": {"path": "/receipt", "sha256": "bad", "bytes": 1}}):
            with self.subTest(changes=changes), self.assertRaises(TaskError):
                capture.validate_capture_request(dict(request(), **changes))

    def test_limits_reject_booleans_nonfinite_unknown_and_oversized_values(self):
        for limits in ({"artifact_bytes": True, "total_bytes": 134217728},
                       {"artifact_bytes": float("inf"), "total_bytes": 134217728},
                       {"artifact_bytes": 33554433, "total_bytes": 134217728},
                       {"artifact_bytes": 33554432, "total_bytes": 134217729},
                       {"artifact_bytes": 33554432, "total_bytes": 134217728, "extra": 1},
                       {"artifact_bytes": 33554432}, {"artifact_bytes": 1024, "total_bytes": 512}):
            with self.subTest(limits=limits), self.assertRaises(TaskError):
                capture.validate_capture_request(dict(request(), limits=limits))
        value = request()
        value["prompt"] = "x" * (1024 * 1024)
        with self.assertRaises(TaskError):
            capture.validate_capture_request(value)

    def test_capabilities_do_not_change_inference_kinds(self):
        self.assertNotIn("coding_capture", [item["kind"] for item in capabilities()["tasks"]])
        value = capture.capture_capabilities()
        self.assertEqual(value["schema"], "coding-capture-capabilities/1")
        self.assertEqual(value["kind"], "coding_capture")
        self.assertEqual(value["limits"], request()["limits"])

    def test_result_separates_client_exit_from_attribution_and_has_no_quality_verdict(self):
        value = {"schema": "coding-capture-result/1", "task_id": str(uuid4()),
                 "request_sha256": "a" * 64, "preview_hash": "b" * 64, "context": context(),
                 "session": {"id": "ses_root", "parent_id": None, "children": []},
                 "requests": [{"artifact": reference("transport.json"), "session_id": "ses_root",
                               "parent_session_id": None, "project_id": "project", "request_model": "local",
                               "response_model": "local"}],
                 "artifacts": [reference("session.json"), reference("transport.json")],
                 "client_exit_code": 0, "elapsed_seconds": 1.25,
                 "completed_at": "2026-10-20T12:00:00+00:00", "attribution": "verified"}
        self.assertEqual(capture.validate_capture_result(value), value)
        for changes in ({"quality": "passed"}, {"elapsed_seconds": float("nan")}, {"client_exit_code": True},
                        {"artifacts": [reference("../outside")]}, {"attribution": "maybe"},
                        {"requests": []}, {"session": {"id": "ses_other", "parent_id": None, "children": []}}):
            with self.subTest(changes=changes), self.assertRaises(TaskError):
                capture.validate_capture_result(dict(value, **changes))


class CodingCaptureStoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(capture, "Coding capture contracts are missing")
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "tasks"
        self.store = TaskStore(self.root)

    def create(self, identifier=None):
        return self.store.create_capture(identifier or str(uuid4()), reference(), context(), "a" * 64)

    def inference(self, media=False):
        value = {"schema": "task-request/2" if media else "task-request/1",
                 "kind": "speak" if media else "classification", "model_path": "/models/local",
                 "parameters": {"text": "hello"} if media else {"state": "refund", "questions": {
                     "route": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "other"]}}}}
        return self.store.create(value, identity_evidence(value["model_path"]))

    def test_capture_record_round_trip_uses_existing_store(self):
        record = self.create()
        self.assertEqual(TaskStore(self.root).get(record["id"]), record)
        self.assertEqual(validate_record(record), record)
        self.assertEqual(record["schema"], "coding-capture-run/1")
        self.assertEqual(record["client_outcome"], "pending")
        self.assertEqual(record["attribution"], "pending")
        self.assertEqual(record["processes"], {})
        with sqlite3.connect(self.store.database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("tasks",)])
        with self.assertRaises(TaskError):
            self.create(record["id"])
        self.assertEqual(self.store.get(record["id"]), record)

    def test_inference_v1_v2_records_remain_unchanged(self):
        records = [self.inference(), self.inference(media=True)]
        raw = {item["id"]: encoded(item) for item in records}
        self.create()
        for record in records:
            self.assertEqual(encoded(self.store.get(record["id"])), raw[record["id"]])
            self.assertEqual(set(record), {"schema", "id", "kind", "state", "model_path", "created_at", "started_at",
                                          "completed_at", "cancel_requested", "identity", "request", "result", "error",
                                          "supervisor", "execution", "execution_unresolved"} |
                             ({"input_artifact"} if record["schema"] == "task-run/2" else set()))

    def test_capture_paging_filters_before_limit(self):
        older = self.create()
        for _ in range(4):
            self.inference()
        newer = self.create()
        for _ in range(4):
            self.inference(media=True)
        page = self.store.list_page(1, kinds=("coding_capture",))
        self.assertEqual([item["id"] for item in page["tasks"]], [newer["id"]])
        self.assertTrue(page["has_more"])
        self.assertEqual([item["id"] for item in self.store.list_page(2, kinds=("coding_capture",))["tasks"]],
                         [newer["id"], older["id"]])
        self.assertFalse(self.store.list_page(2, kinds=("coding_capture",))["has_more"])

    def test_unknown_schema_does_not_execute(self):
        record = self.create()
        record["schema"] = "coding-capture-run/99"
        with sqlite3.connect(self.store.database) as connection:
            connection.execute("UPDATE tasks SET record=? WHERE id=?", (json.dumps(record), record["id"]))
        before = self.store.database.read_bytes()
        with self.assertRaises(TaskError):
            self.store.request_cancel(record["id"])
        self.assertEqual(self.store.database.read_bytes(), before)

    def test_request_has_no_unbounded_external_copy(self):
        record = self.create()
        self.assertFalse((self.root / "artifacts").exists())
        self.assertEqual(record["request"], reference())
        with self.assertRaises(TaskError):
            self.store.read_request(record["id"])
        with self.assertRaises(TaskError):
            self.store.write_result(record["id"], {"prompt": "must stay inside volume"})
        self.assertFalse((self.root / "artifacts").exists())

    def test_capture_transitions_preserve_identity_and_bound_row(self):
        record = self.create()
        identifier = record["id"]
        processes = {"supervisor": {"pid": 44, "identity": "supervisor", "started_at": [100, 201]},
                     "client": {"pid": 45, "identity": "client", "started_at": [100, 202]}}
        running = self.store.transition(identifier, ("starting",), {"state": "running", "processes": processes})
        self.assertIsNotNone(running["started_at"])
        for changes in ({"context": context()}, {"preview_hash": "b" * 64}, {"request": reference("other.json")},
                        {"processes": dict(processes, borrowed_server=processes["client"])},
                        {"state": "succeeded", "client_outcome": "succeeded", "attribution": "failed",
                         "result": reference("result.json")},
                        {"error": {"code": "failure", "message": "x" * 8192, "remediation": "Retry."}}):
            with self.subTest(changes=changes), self.assertRaises(TaskError):
                self.store.transition(identifier, ("running",), changes)
            self.assertEqual(self.store.get(identifier), running)
        succeeded = self.store.transition(identifier, ("running",), {"state": "succeeded", "client_outcome": "succeeded",
                                           "attribution": "verified", "result": reference("result.json")})
        self.assertLessEqual(len(encoded(succeeded)), 8192)
        self.assertEqual(succeeded["context"], record["context"])

    def test_invalid_creation_does_not_write_a_row(self):
        for ref, identity, digest in ((reference("../escape"), context(), "a" * 64),
                                      (reference(), context(), "bad"),
                                      (reference(), dict(context(), project={"path": "relative"}), "a" * 64)):
            with self.subTest(ref=ref, digest=digest), self.assertRaises(TaskError):
                self.store.create_capture(str(uuid4()), ref, identity, digest)
        self.assertEqual(self.store.list(), [])

    def test_row_size_limit_precedes_sqlite_write(self):
        record = self.create()
        with self.assertRaises(TaskError) as caught:
            self.store.transition(record["id"], ("starting",), {
                "state": "failed", "error": {"code": "failure", "message": "x" * 4000, "remediation": "y" * 4000}})
        self.assertEqual(caught.exception.code, "capture_metadata_limit")
        self.assertEqual(self.store.get(record["id"]), record)

    def test_malformed_session_relationship_is_a_typed_error(self):
        value = {"schema": "coding-capture-result/1", "task_id": str(uuid4()),
                 "request_sha256": "a" * 64, "preview_hash": "b" * 64, "context": context(),
                 "session": {"id": "root", "parent_id": None, "children": [{"id": "child", "parent_id": []}]},
                 "requests": [], "artifacts": [], "client_exit_code": 1, "elapsed_seconds": 0,
                 "completed_at": "2026-10-20T12:00:00+00:00", "attribution": "failed"}
        with self.assertRaises(TaskError):
            capture.validate_capture_result(value)
