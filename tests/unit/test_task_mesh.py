"""Durable 3D task contracts using tiny synthetic artifacts, without model weights."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

from mlx_agent.task_contracts import TaskError, capabilities, validate_request
from mlx_agent.task_kinds import get
from mlx_agent.task_runtime import task_result
from mlx_agent.task_store import TaskStore
from tests.task_fixtures import identity_evidence


ROOT = Path(__file__).resolve().parents[2]


def glb():
    content = json.dumps({"asset": {"version": "2.0"}}, separators=(",", ":")).encode()
    content += b" " * (-len(content) % 4)
    return struct.pack("<III", 0x46546C67, 2, 20 + len(content)) + struct.pack("<II", len(content), 0x4E4F534A) + content


def reference(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class MeshTaskTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.request = {"schema": "task-request/2", "kind": "mesh_generation", "model_path": str(self.root / "model"),
                        "parameters": {"image_path": str(self.root / "input.png"), "seed": 7}}

    def test_mesh_capability_uses_staged_image_and_glb_output(self):
        kind = get("mesh_generation")
        self.assertEqual(kind.input_media, "image")
        self.assertEqual(kind.input_field, "image_path")
        self.assertEqual(kind.output, "output.glb")
        self.assertEqual(kind.result_schema, "mesh/1")
        self.assertEqual(kind.request_schema, "task-request/2")
        self.assertEqual(kind.run_schema, "task-run/2")
        self.assertTrue(next(row for row in capabilities()["tasks"] if row["kind"] == kind.name)["cancellation"])
        staged = str(self.root / "owned-input.png")
        with patch("mlx_agent.task_kinds.plan_mesh", return_value={"backend": "mlx-3d"}) as planner:
            kind.plan(validate_request(self.request), self.root, staged)
        planner.assert_called_once_with(self.request["model_path"], staged, str(self.root / "output.glb"), seed=7)

    def test_request_schema_and_runtime_accept_the_same_mesh_parameters(self):
        schema = Draft202012Validator(json.loads((ROOT / "schemas/task-request.schema.json").read_text()))
        parameters = self.request["parameters"]
        invalid = [{}, dict(parameters, unexpected=True), dict(parameters, image_path="relative.png"),
                   dict(parameters, steps=True), dict(parameters, guidance="high"), dict(parameters, max_faces=1.5)]
        for options, expected in [(parameters, True), *[(value, False) for value in invalid]]:
            request = dict(self.request, parameters=options)
            with self.subTest(parameters=options):
                self.assertEqual(schema.is_valid(request), expected)
                if expected:
                    self.assertEqual(validate_request(request), request)
                else:
                    with self.assertRaises(TaskError):
                        validate_request(request)

    def test_mesh_output_refuses_truncation_and_links(self):
        kind = get("mesh_generation")
        output = self.root / "output.glb"
        output.write_bytes(glb())
        kind.check_output(output, {})
        output.write_bytes(glb()[:-1])
        with self.assertRaises(TaskError):
            kind.check_output(output, {})
        output.unlink()
        original = self.root / "other.glb"
        original.write_bytes(glb())
        output.symlink_to(original)
        with self.assertRaises(TaskError):
            kind.check_output(output, {})

    def test_completed_mesh_retains_verified_input_and_output_artifacts(self):
        image = Path(self.request["parameters"]["image_path"])
        image.write_bytes(b"synthetic image fixture")
        store = TaskStore(self.root / "tasks")
        staged = store.stage_input(str(image), media_kind="image")
        record = store.create(self.request, identity_evidence(self.request["model_path"]),
                              input_artifact=staged["reference"], staged_input=staged)
        identifier = record["id"]
        output = store.artifact_path(identifier, "output.glb")
        output.write_bytes(glb())
        output.chmod(0o600)
        process = {"pid": 999999, "identity": "synthetic worker"}
        store.transition(identifier, ("starting",), {"state": "running", "supervisor": process, "execution": process})
        result = store.write_result(identifier, {"schema": "mesh/1", "path": str(output),
            "input_artifact": record["input_artifact"], "output_artifact": reference(output)})
        store.transition(identifier, ("running",), {"state": "succeeded", "result": result})
        record_schema = Draft202012Validator(json.loads((ROOT / "schemas/task-run.schema.json").read_text()))
        self.assertTrue(record_schema.is_valid(store.get(identifier)))
        self.assertEqual(task_result(identifier, store.root)["result"]["output_artifact"], reference(output))
        output.write_bytes(b"changed")
        with self.assertRaises(TaskError) as failure:
            task_result(identifier, store.root)
        self.assertEqual(failure.exception.code, "output_artifact_changed")
