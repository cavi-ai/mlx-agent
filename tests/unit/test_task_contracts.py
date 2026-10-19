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
        self.assertEqual({item["kind"] for item in result["tasks"]},
                         {"classification", "music_generation", "transcribe", "speak", "image_generation", "image_understanding"})
        self.assertTrue(all(item["cancellation"] for item in result["tasks"]))
        self.assertNotIn("ready", result)

    def test_media_capabilities_override_schemas_without_changing_legacy_defaults(self):
        result = capabilities()
        self.assertEqual((result["request_schema"], result["run_schema"]), ("task-request/1", "task-run/1"))
        tasks = {item["kind"]: item for item in result["tasks"]}
        self.assertEqual((tasks["classification"]["result_schema"], tasks["music_generation"]["result_schema"]),
                         ("decide/1", "music/1"))
        self.assertEqual((tasks["transcribe"]["request_schema"], tasks["transcribe"]["run_schema"],
                          tasks["transcribe"]["result_schema"]), ("task-request/2", "task-run/2", "transcribe/1"))
        self.assertEqual((tasks["speak"]["request_schema"], tasks["speak"]["run_schema"],
                          tasks["speak"]["result_schema"]), ("task-request/2", "task-run/2", "speak/1"))

    def test_image_capabilities_and_requests_use_media_v2_without_materializing_defaults(self):
        tasks = {item["kind"]: item for item in capabilities()["tasks"]}
        for kind, result_schema in (("image_generation", "generate/1"), ("image_understanding", "describe/1")):
            with self.subTest(kind=kind):
                self.assertEqual((tasks[kind]["request_schema"], tasks[kind]["run_schema"], tasks[kind]["result_schema"]),
                                 ("task-request/2", "task-run/2", result_schema))
        requests = [
            {"schema": "task-request/2", "kind": "image_generation", "model_path": "/models/image",
             "parameters": {"prompt": "A red kite"}},
            {"schema": "task-request/2", "kind": "image_understanding", "model_path": "/models/vision",
             "parameters": {"image_path": "/images/input.webp", "prompt": "Describe this image."}},
        ]
        for request in requests:
            with self.subTest(request=request):
                self.assertEqual(validate_request(request), request)

    def test_image_requests_reject_wrong_types_unknown_fields_and_wrong_schema(self):
        base = {"schema": "task-request/2", "kind": "image_generation", "model_path": "/models/image",
                "parameters": {"prompt": "A red kite"}}
        invalid = [dict(base, schema="task-request/1"), dict(base, parameters={"prompt": "x", "width": True}),
                   dict(base, parameters={"prompt": "x", "timeout": 2}),
                   {"schema": "task-request/2", "kind": "image_understanding", "model_path": "/models/vision",
                    "parameters": {"image_path": "relative.png", "prompt": "Describe."}},
                   {"schema": "task-request/2", "kind": "image_understanding", "model_path": "/models/vision",
                    "parameters": {"image_path": "/images/a.png", "prompt": "Describe.", "fps": 1.0}}]
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(TaskError):
                validate_request(request)

    def test_speech_request_v2_validates_without_materializing_planner_defaults(self):
        requests = [
            {"schema": "task-request/2", "kind": "transcribe", "model_path": "/models/whisper",
             "parameters": {"audio_path": "/audio/input.flac"}},
            {"schema": "task-request/2", "kind": "speak", "model_path": "/models/tts",
             "parameters": {"text": "Hello.", "voice": "af_heart"}},
        ]
        for request in requests:
            with self.subTest(kind=request["kind"]):
                self.assertEqual(validate_request(request), request)
        domain_values = {"text": " " + ("x" * 2000), "voice": "invalid voice", "speed": 3.0,
                         "lang_code": "not_a_language_tag?"}
        request = {"schema": "task-request/2", "kind": "speak", "model_path": "/models/tts",
                   "parameters": domain_values}
        self.assertEqual(validate_request(request), request)

    def test_speech_request_rejects_unsupported_paths_fields_and_versions(self):
        base = {"schema": "task-request/2", "kind": "transcribe", "model_path": "/models/whisper",
                "parameters": {"audio_path": "/audio/input.wav"}}
        invalid = [dict(base, schema="task-request/1"), dict(base, kind="speak"),
                   dict(base, parameters={"audio_path": "relative.wav"}),
                   dict(base, parameters={"audio_path": "/audio/input.wav", "timeout": 10})]
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(TaskError):
                validate_request(request)
        speak = dict(base, kind="speak", parameters={"text": "Hello.", "speed": True})
        with self.assertRaises(TaskError):
            validate_request(speak)
