import copy
import unittest

from mlx_agent.task_contracts import TaskError, capabilities, validate_request


class TaskContractTests(unittest.TestCase):
    def request(self, kind="classification"):
        parameters = {"state": "refund", "questions": {"route": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": ["billing", "other"]}}}
        if kind == "music_generation":
            parameters = {"caption": "Quiet piano"}
        return {"schema": "task-request/1", "kind": kind, "model_path": "/models/local", "parameters": parameters}

    def test_music_defaults_are_accepted_settings_without_mutating_request(self):
        request = self.request("music_generation")
        original = copy.deepcopy(request)
        result = validate_request(request)
        self.assertEqual(result["parameters"], {"caption": "Quiet piano", "lyrics": "[instrumental]", "duration": 15.0, "steps": 30, "seed": 42})
        self.assertEqual(request, original)

    def test_classification_preserves_structured_input(self):
        request = self.request()
        request["parameters"]["state"] = {"customer": "A", "items": [1, 2]}
        self.assertEqual(validate_request(request), request)

    def test_rejects_unknown_contract_and_unsafe_request_values(self):
        cases = [None, [], {}, dict(self.request(), schema="task-request/2"),
                 dict(self.request(), kind="chat"), dict(self.request(), model_path="relative"),
                 dict(self.request(), extra="silent"), dict(self.request(), parameters={}),
                 dict(self.request(), parameters={"state": True, "questions": {"x": {"type": "choice", "instructions": []}}})]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(TaskError):
                validate_request(value)
        for field, value in (("duration", float("nan")), ("duration", 361), ("steps", True), ("steps", 31), ("seed", -1), ("caption", ""), ("lyrics", "")):
            request = self.request("music_generation")
            request["parameters"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(TaskError):
                validate_request(request)

    def test_capabilities_do_not_claim_selected_model_readiness(self):
        result = capabilities()
        self.assertEqual({item["kind"] for item in result["tasks"]}, {"classification", "music_generation"})
        self.assertTrue(all(item["cancellation"] for item in result["tasks"]))
        self.assertNotIn("ready", result)
