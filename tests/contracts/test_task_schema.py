import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jsonschema import Draft202012Validator, FormatChecker

from mlx_agent.task_contracts import TaskError, validate_record, validate_request
from mlx_agent.task_store import TaskStore
from mlx_agent.task_cli import summary
from tests.task_fixtures import identity_evidence

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
        classification = {"schema": "task-request/1", "kind": "classification", "model_path": "/models/local", "parameters": {"state": "refund", "questions": {"route": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": ["billing", "other"]}}}}
        music = dict(classification, kind="music_generation", parameters={"caption": "Quiet piano"})
        values = [classification, music, {}, [], dict(music, schema="task-request/2"), dict(music, kind="other"), dict(music, extra=1), dict(music, model_path="relative")]
        for params in ({"caption": ""}, {"caption": "x", "duration": 0}, {"caption": "x", "duration": 360}, {"caption": "x", "steps": True}, {"caption": "x", "seed": 4294967296}, {"caption": "x", "out": "/unowned"}):
            values.append(dict(music, parameters=params))
        values.append(dict(classification, parameters={"state": False, "questions": {"x": {"type": "choice", "instructions": "x"}}}))
        transcribe = {"schema": "task-request/2", "kind": "transcribe", "model_path": "/models/whisper",
                      "parameters": {"audio_path": "/audio/input.wav"}}
        speak = {"schema": "task-request/2", "kind": "speak", "model_path": "/models/tts",
                 "parameters": {"text": "Hello."}}
        values.extend((transcribe, speak, dict(transcribe, schema="task-request/1"),
                       dict(speak, parameters={"text": "Hello.", "speed": True}),
                       dict(transcribe, parameters={"audio_path": "relative.wav"})))
        values.append(dict(speak, parameters={"text": " " + ("x" * 2000), "voice": "invalid voice",
                                              "speed": 3.0, "lang_code": "not_a_language_tag?"}))
        self.assert_parity("task-request.schema.json", validate_request, values)

    def test_task_run_schema_matches_runtime(self):
        with TemporaryDirectory() as directory:
            store = TaskStore(Path(directory).resolve() / "tasks")
            request = {"schema": "task-request/1", "kind": "music_generation", "model_path": "/models/local", "parameters": {"caption": "Quiet piano"}}
            value = store.create(request, identity_evidence(request["model_path"]))
            store.transition(value["id"], ("starting",), {"supervisor": {"pid": 42, "identity": "supervisor"}})
            value = store.transition(value["id"], ("starting",), {
                "state": "running", "execution": {"pid": 43, "identity": "worker"}})
            values = [value, {}, dict(value, schema="task-run/2"), dict(value, id="bad"), dict(value, cancel_requested=1), dict(value, state="unknown"), dict(value, created_at="yesterday"), dict(value, state="succeeded"), dict(value, identity=[])]
            audio = Path(directory).resolve() / "source.wav"
            audio.write_bytes(b"source audio")
            staged = store.stage_input(audio)
            media_request = {"schema": "task-request/2", "kind": "transcribe", "model_path": "/models/whisper",
                             "parameters": {"audio_path": str(audio)}}
            media_record = store.create(media_request, identity_evidence(media_request["model_path"]),
                                        input_artifact=staged["reference"], staged_input=staged)
            speak_request = {"schema": "task-request/2", "kind": "speak", "model_path": "/models/tts",
                             "parameters": {"text": "Hello."}}
            speak_record = store.create(speak_request, identity_evidence(speak_request["model_path"]))
            values.extend((media_record, speak_record, dict(media_record, input_artifact=None),
                           dict(speak_record, input_artifact=media_record["input_artifact"])))
            malformed = []
            for key, replacement in (("pid", True), ("pid", 0), ("identity", " ")):
                candidate = copy.deepcopy(value)
                candidate["supervisor"][key] = replacement
                malformed.append(candidate)
            for key, replacement in (("path", "relative"), ("sha256", "bad"), ("bytes", -1), ("bytes", True)):
                candidate = copy.deepcopy(value)
                candidate["request"][key] = replacement
                malformed.append(candidate)
            for key, replacement in (("model_fingerprint", "bad"), ("backend_manifest_hash", "bad"),
                                     ("requested_executable", "relative"), ("executable_identity", [1, 2, True, 4])):
                candidate = copy.deepcopy(value)
                candidate["identity"][key] = replacement
                malformed.append(candidate)
            values.extend(malformed)
            result = store.write_result(value["id"], {"sample": "done"})
            completed = store.transition(value["id"], ("running",), {"state": "succeeded", "result": result})
            values.append(completed)
            values.append(dict(completed, result=None))
            values.append(dict(completed, state="failed", error=None))
            failed = copy.deepcopy(completed)
            failed.update(state="failed", result=None, error={"code": "failure", "message": "Failure.", "remediation": "Retry."})
            for key, replacement in (("code", " "), ("message", ""), ("remediation", 1)):
                candidate = copy.deepcopy(failed)
                candidate["error"][key] = replacement
                values.append(candidate)
            self.assert_parity("task-run.schema.json", validate_record, values)
            inconsistent_hash = copy.deepcopy(value)
            inconsistent_hash["identity"]["backend_manifest_hash"] = "0" * 64
            with self.assertRaises(TaskError):
                validate_record(inconsistent_hash)
            mismatched_model = copy.deepcopy(value)
            mismatched_model["identity"]["requested_model_path"] = "/another/model"
            with self.assertRaises(TaskError):
                validate_record(mismatched_model)

    def test_task_run_schema_documents_public_summary_without_private_evidence(self):
        with TemporaryDirectory() as directory:
            store = TaskStore(Path(directory).resolve() / "tasks")
            request = {"schema": "task-request/1", "kind": "classification", "model_path": "/models/local", "parameters": {"state": "refund", "questions": {"route": {"type": "choice", "instructions": "Route this request.", "criteria": ["billing", "other"]}}}}
            stored = store.create(request, identity_evidence(request["model_path"]))
            public = summary(stored)
            schema = json.loads((ROOT / "schemas" / "task-run.schema.json").read_text())
            self.assertTrue(Draft202012Validator(schema, format_checker=FormatChecker()).is_valid(stored))
            self.assertTrue(Draft202012Validator(schema, format_checker=FormatChecker()).is_valid(public))
            self.assertNotIn("backend_manifest", public["identity"])
            self.assertEqual(public["supervisor"], None)
            with self.assertRaises(TaskError):
                validate_record(public)

    def test_public_summaries_validate_for_running_succeeded_and_failed_records(self):
        with TemporaryDirectory() as directory:
            store = TaskStore(Path(directory).resolve() / "tasks")
            request = {"schema": "task-request/1", "kind": "classification", "model_path": "/models/local", "parameters": {"state": "refund", "questions": {"route": {"type": "choice", "instructions": "Route this request.", "criteria": ["billing", "other"]}}}}
            running = store.create(request, identity_evidence(request["model_path"]))
            store.transition(running["id"], ("starting",), {"supervisor": {"pid": 42, "identity": "supervisor"}})
            running = store.transition(running["id"], ("starting",), {
                "state": "running", "execution": {"pid": 43, "identity": "worker"}})
            successful = store.write_result(running["id"], {"answers": {"route": "billing"}})
            successful = store.transition(running["id"], ("running",), {"state": "succeeded", "result": successful})
            failed = store.create(request, identity_evidence(request["model_path"]))
            failed = store.transition(failed["id"], ("starting",), {
                "state": "failed", "error": {"code": "failed", "message": "Failed.", "remediation": "Retry."}})
            schema = json.loads((ROOT / "schemas" / "task-run.schema.json").read_text())
            validator = Draft202012Validator(schema, format_checker=FormatChecker())
            for record in (running, successful, failed):
                with self.subTest(state=record["state"]):
                    self.assertTrue(validator.is_valid(record))
                    self.assertTrue(validator.is_valid(summary(record)))
