import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jsonschema import Draft202012Validator, FormatChecker

from mlx_agent.task_contracts import TaskError, validate_record, validate_request
from mlx_agent.task_store import TaskStore

ROOT = Path(__file__).resolve().parents[2]


class TaskSchemaTests(unittest.TestCase):
    def assert_parity(self, schema_name, validator, values):
        schema = json.loads((ROOT / "schemas" / schema_name).read_text())
        schema_validator = Draft202012Validator(schema, format_checker=FormatChecker())
        for value in values:
            with self.subTest(value=value):
                try:
                    validator(value)
                    accepted = True
                except TaskError:
                    accepted = False
                self.assertEqual(schema_validator.is_valid(value), accepted)

    def test_task_request_schema_matches_runtime(self):
        classification = {"schema": "task-request/1", "kind": "classification", "model_path": "/models/local", "parameters": {"state": "refund", "questions": {"route": {"type": "choice", "instructions": ["billing", "other"]}}}}
        music = dict(classification, kind="music_generation", parameters={"caption": "Quiet piano"})
        values = [classification, music, {}, [], dict(music, schema="task-request/2"), dict(music, kind="other"), dict(music, extra=1), dict(music, model_path="relative")]
        for params in ({"caption": ""}, {"caption": "x", "duration": 0}, {"caption": "x", "duration": 360}, {"caption": "x", "steps": True}, {"caption": "x", "seed": 4294967296}, {"caption": "x", "out": "/unowned"}):
            values.append(dict(music, parameters=params))
        values.append(dict(classification, parameters={"state": False, "questions": {"x": {"type": "choice", "instructions": "x"}}}))
        self.assert_parity("task-request.schema.json", validate_request, values)

    def test_task_run_schema_matches_runtime(self):
        with TemporaryDirectory() as directory:
            store = TaskStore(Path(directory).resolve() / "tasks")
            request = {"schema": "task-request/1", "kind": "music_generation", "model_path": "/models/local", "parameters": {"caption": "Quiet piano"}}
            value = store.create(request, {})
            values = [value, {}, dict(value, schema="task-run/2"), dict(value, id="bad"), dict(value, cancel_requested=1), dict(value, state="unknown"), dict(value, created_at="yesterday"), dict(value, state="succeeded"), dict(value, identity=[])]
            completed = copy.deepcopy(value)
            completed.update(state="succeeded", completed_at="2026-10-17T00:00:00+00:00", result={"path": "/result", "sha256": "a" * 64, "bytes": 12})
            values.append(completed)
            values.append(dict(completed, result=None))
            values.append(dict(completed, state="failed", error=None))
            self.assert_parity("task-run.schema.json", validate_record, values)
