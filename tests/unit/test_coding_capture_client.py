"""OpenCode session evidence must retain observed identities without fallback."""

import copy
import importlib
import json
import unittest

from mlx_agent.task_contracts import TaskError, encoded

try:
    client = importlib.import_module("mlx_agent.coding_capture_client")
except ModuleNotFoundError:
    client = None


def transfer(session="ses_root", model="local"):
    return {"info": {"id": session, "projectID": "prj_owned", "model": {"id": model, "providerID": "capture"},
                     "cost": 0, "tokens": {}, "time": {"created": 1792497600000, "updated": 1792497601000},
                     "location": {"directory": "/projects/owned"}},
            "messages": [{"id": "msg_first", "type": "assistant", "agent": "build",
                          "model": {"id": model, "providerID": "capture"},
                          "content": [{"type": "text", "text": "A useful response."}],
                          "time": {"created": 1792497600000, "completed": 1792497601000}}]}


class OpenCodeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(client, "OpenCode capture adapter is missing")

    def test_ndjson_keeps_exact_observed_root_session_and_epoch_milliseconds(self):
        events = []
        reader = client.OpenCodeEvents(events.append, artifact_bytes=33554432)
        raw = encoded({"type": "text", "sessionID": "ses_observed", "timestamp": 1792497600123,
                       "part": {"text": "résumé"}}) + b"\n"
        for byte in raw:
            reader.feed(bytes([byte]))
        reader.finish()
        self.assertEqual(reader.session_id, "ses_observed")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["timestamp"], 1792497600123)
        self.assertEqual(events[0]["part"]["text"], "résumé")

    def test_ndjson_rejects_changed_identity_malformed_partial_and_oversized_lines(self):
        valid = encoded({"type": "step_start", "sessionID": "ses_root", "timestamp": 1792497600000}) + b"\n"
        for suffix in (b"not json\n", b"{}\n", b'\xff\n',
                       encoded({"type": "text", "sessionID": "ses_other", "timestamp": 1792497600000}) + b"\n",
                       encoded({"type": "text", "sessionID": "ses_root", "timestamp": 1792497600}) + b"\n",
                       encoded({"type": "text", "sessionID": "ses_root", "timestamp": True}) + b"\n",
                       b'{"type":"text","sessionID":"ses_root","timestamp":NaN}\n'):
            with self.subTest(suffix=suffix), self.assertRaises(TaskError):
                reader = client.OpenCodeEvents(lambda value: None, artifact_bytes=33554432)
                reader.feed(valid + suffix)
                reader.finish()
        with self.assertRaises(TaskError):
            reader = client.OpenCodeEvents(lambda value: None, artifact_bytes=100)
            reader.feed(b"x" * 101)
        with self.assertRaises(TaskError):
            reader = client.OpenCodeEvents(lambda value: None, artifact_bytes=33554432)
            reader.feed(valid.rstrip(b"\n"))
            reader.finish()

    def test_unknown_event_does_not_become_success_and_error_event_is_retained(self):
        with self.assertRaises(TaskError):
            reader = client.OpenCodeEvents(lambda value: None, artifact_bytes=33554432)
            reader.feed(encoded({"type": "future_event", "sessionID": "ses_root", "timestamp": 1792497600000}) + b"\n")
        events = []
        reader = client.OpenCodeEvents(events.append, artifact_bytes=33554432)
        reader.feed(encoded({"type": "error", "sessionID": "ses_root", "timestamp": 1792497600000,
                             "error": {"message": "Model connection failed."}}) + b"\n")
        reader.finish()
        self.assertTrue(reader.has_error)
        self.assertEqual(events[0]["error"]["message"], "Model connection failed.")

    def test_exact_export_model_session_and_parent_relationship_are_retained(self):
        value = transfer("ses_child")
        value["info"]["parentID"] = "ses_root"
        accepted = client.validate_session_export(encoded(value), "ses_child", "capture/local")
        self.assertEqual(accepted, value)
        value["info"]["id"] = "changed"
        self.assertEqual(accepted["info"]["id"], "ses_child")
        self.assertEqual(accepted["info"]["parentID"], "ses_root")

    def test_wrong_export_session_or_any_assistant_model_is_refused(self):
        for value in (transfer("ses_wrong"), transfer(model="wrong"), {"info": {}, "messages": []},
                      {"session": transfer()}, dict(transfer(), messages=[])):
            with self.subTest(value=value), self.assertRaises(TaskError):
                client.validate_session_export(encoded(value), "ses_root", "capture/local")
        value = transfer()
        other = copy.deepcopy(value["messages"][0])
        other["id"] = "msg_second"
        other["model"] = {"providerID": "cloud", "id": "different"}
        value["messages"].append(other)
        with self.assertRaises(TaskError):
            client.validate_session_export(encoded(value), "ses_root", "capture/local")

    def test_export_accepts_recorded_assistant_model_without_optional_session_model(self):
        value = transfer()
        del value["info"]["model"]
        self.assertEqual(client.validate_session_export(encoded(value), "ses_root", "capture/local"), value)

    def test_global_project_and_schema_valid_session_prefix_are_not_rewritten(self):
        value = transfer("ses123")
        value["info"]["projectID"] = "global"
        self.assertEqual(client.validate_session_export(encoded(value), "ses123", "capture/local"), value)
        events = []
        reader = client.OpenCodeEvents(events.append, artifact_bytes=33554432)
        reader.feed(encoded({"type": "text", "sessionID": "ses123", "timestamp": 1792497600000}) + b"\n")
        reader.finish()
        self.assertEqual(reader.session_id, "ses123")

    def test_export_never_accepts_truncated_or_nonfinite_json(self):
        for raw in (encoded(transfer())[:-1], b'{"info":NaN,"messages":[]}', b"[]", b"\xff"):
            with self.subTest(raw=raw[:30]), self.assertRaises(TaskError):
                client.validate_session_export(raw, "ses_root", "capture/local")
