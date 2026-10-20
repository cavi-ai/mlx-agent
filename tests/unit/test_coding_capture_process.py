"""Owned CLI execution keeps prompts off argv and retains bounded failures."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mlx_agent.task_contracts import TaskError
from tests.unit.test_coding_capture_client import transfer

try:
    process = importlib.import_module("mlx_agent.coding_capture_process")
except ModuleNotFoundError:
    process = None


class ArtifactStore:
    def __init__(self):
        self.content = {}

    def write_artifact(self, name, raw):
        self.content[name] = raw
        return {"name": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


class OwnedCLIExecutionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(process, "Owned CLI execution is missing")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.client = self.root / "client"
        self.client.write_text("#!" + sys.executable + "\n" +
            "import json,os,sys,time\n"
            "prompt=sys.stdin.buffer.read().decode()\n"
            "mode=os.environ.get('CAPTURE_FIXTURE_MODE','valid')\n"
            "sys.stderr.write(json.dumps({'argv':sys.argv[1:],'prompt':prompt,'pwd':os.environ['PWD']}))\n"
            "if sys.argv[1]=='session': print(os.environ['CAPTURE_FIXTURE_EXPORT'])\n"
            "elif mode=='large':\n"
            " sys.stdout.write('x'*8192);sys.stdout.flush();time.sleep(30)\n"
            "elif mode=='partial': sys.stdout.write('{')\n"
            "else:\n"
            " print(json.dumps({'type':'text','sessionID':'ses_owned','timestamp':1792497600000,'part':{'text':'fixture'}}))\n"
            "sys.exit(7 if mode=='failed' else 0)\n")
        self.client.chmod(0o700)
        self.store, self.owned = ArtifactStore(), []
        self.profile = {"client_path": str(self.client), "project_path": str(self.root),
                        "environment": dict(os.environ, PWD=str(self.root)),
                        "server_url": "http://127.0.0.1:12345", "model": "capture/local",
                        "artifact_bytes": 4096, "storage": self.store,
                        "check_context": lambda: None,
                        "on_process": lambda role, identity: self.owned.append((role, identity)),
                        "cancelled": lambda: False}
        self.identity = patch("mlx_agent.coding_capture_process.capture_process_identity",
                              side_effect=lambda pid: {"pid": pid, "identity": "owned fixture", "started_at": [1, 0]})
        self.identity.start()
        self.addCleanup(self.identity.stop)

    def test_prompt_uses_stdin_and_explicit_owned_server(self):
        events = []
        result = process.run_client(self.profile, 'Private prompt with "quotes"\nand a new line.', events.append)
        diagnostic = json.loads(self.store.content["client-stderr.log"])
        self.assertEqual(diagnostic["prompt"], 'Private prompt with "quotes"\nand a new line.')
        self.assertEqual(diagnostic["argv"], ["run", "--server", "http://127.0.0.1:12345", "--format", "json", "--model", "capture/local"])
        self.assertEqual(diagnostic["pwd"], str(self.root))
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["session_id"], "ses_owned")
        self.assertEqual(len(events), 1)
        self.assertEqual(self.owned[0][0], "client")
        self.assertGreater(result["elapsed_seconds"], 0)

    def test_nonzero_exit_retains_diagnostics_and_observed_session(self):
        self.profile["environment"]["CAPTURE_FIXTURE_MODE"] = "failed"
        result = process.run_client(self.profile, "fixture", lambda _: None)
        self.assertEqual(result["exit_code"], 7)
        self.assertEqual(result["session_id"], "ses_owned")
        self.assertIn("client-stderr.log", self.store.content)
        self.assertEqual(result["error"]["code"], "capture_client_failed")

    def test_partial_and_oversized_output_cannot_succeed(self):
        for mode, code in (("partial", "invalid_capture_session"), ("large", "capture_artifact_limit")):
            with self.subTest(mode=mode):
                self.profile["environment"]["CAPTURE_FIXTURE_MODE"] = mode
                result = process.run_client(self.profile, "fixture", lambda _: None)
                self.assertEqual(result["error"]["code"], code)
                self.assertLessEqual(len(self.store.content["client.ndjson"]), 4096)

    def test_changed_context_or_prior_cancel_starts_nothing(self):
        def changed():
            raise TaskError("capture_context_changed", "changed")
        for key, value in (("check_context", changed), ("cancelled", lambda: True)):
            with self.subTest(key=key), patch("subprocess.Popen") as start:
                profile = dict(self.profile, **{key: value})
                with self.assertRaises(TaskError):
                    process.run_client(profile, "must not send", lambda _: None)
                start.assert_not_called()

    def test_cancel_after_spawn_stops_the_owned_command(self):
        self.profile["cancelled"] = lambda: bool(self.owned)
        result = process.run_client(self.profile, "must not send", lambda _: None)
        self.assertEqual(result["error"]["code"], "capture_cancelled")
        self.assertLess(result["exit_code"], 0)
        self.assertEqual(self.store.content["client.ndjson"], b"")

    def test_lost_ownership_retains_diagnostics_without_signalling_reused_pid(self):
        self.profile["environment"]["CAPTURE_FIXTURE_MODE"] = "large"
        self.identity.stop()
        children, start = [], subprocess.Popen
        def create(*args, **kwargs):
            child = start(*args, **kwargs)
            children.append(child)
            return child
        try:
            with patch("mlx_agent.coding_capture_process.subprocess.Popen", side_effect=create), \
                    patch("mlx_agent.coding_capture_process.capture_process_identity",
                          side_effect=lambda pid: {"pid": pid, "identity": "owned" if not self.owned else "changed", "started_at": [1, 0]}), \
                    patch("mlx_agent.coding_capture_process.os.killpg") as signal:
                result = process.run_client(self.profile, "fixture", lambda _: None)
                self.assertTrue(result["execution_unresolved"])
                self.assertEqual(result["error"]["code"], "capture_ownership_lost")
                self.assertIn("client-stderr.log", self.store.content)
                signal.assert_not_called()
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=5)

    def test_export_uses_only_the_observed_session_and_rejects_mismatches(self):
        session = transfer("ses_owned")
        session["info"]["location"]["directory"] = str(self.root)
        self.profile["environment"]["CAPTURE_FIXTURE_EXPORT"] = json.dumps(session)
        result = process.export_session(self.profile, "ses_owned", self.store)
        self.assertIsNone(result["error"])
        self.assertEqual(result["session"]["info"]["id"], "ses_owned")
        self.assertEqual(json.loads(self.store.content["export-stderr.log"])["argv"],
                         ["session", "export", "ses_owned", "--server", "http://127.0.0.1:12345"])
        self.assertEqual(self.owned[-1][0], "exporter")
        result = process.export_session(self.profile, "ses_other", self.store)
        self.assertEqual(result["error"]["code"], "invalid_capture_session")
        self.assertIsNone(result["session"])
        with patch("subprocess.Popen") as start, self.assertRaises(TaskError):
            process.export_session(self.profile, "not-an-observed-session", self.store)
        start.assert_not_called()
