"""Residency uses the real runner process with a tiny in-memory classifier fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from mlx_agent import resident_runner


class ResidencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{}')
        (self.root / "fixture_port.py").write_text('''
import time
class Model:
    count = 0
def load(path):
    return Model(), None
def predict(model, tokenizer, state, questions):
    if state == "wait": time.sleep(30)
    model.count += 1
    return {"answers": {"calls": model.count}}
''')
        self.worker = resident_runner.ResidentRunner(sys.executable, str(self.model),
            environment=dict(os.environ, PYTHONPATH=str(self.root)))
        self.addCleanup(self.worker.close)
        self.request = self.root / "request.json"
        self.request.write_text(json.dumps({"state": "test", "questions": {}}))
        self.argv = [sys.executable, str(Path(resident_runner.__file__).with_name("decide_runner.py")),
                     "--module", "fixture_port", "--model", str(self.model), "--request", str(self.request)]

    def invoke(self, **options):
        response = self.worker.run(self.argv, timeout=5, **options)
        self.assertEqual(response.returncode, 0, response.stderr)
        return json.loads(response.stdout.splitlines()[-1])["answers"]["calls"]

    def test_same_model_object_survives_two_requests(self):
        self.assertEqual(self.invoke(), 1)
        pid = self.worker.process.pid
        self.assertEqual(self.invoke(), 2)
        self.assertEqual(self.worker.process.pid, pid)
        self.assertTrue(self.worker.loaded)

    def test_release_reaps_worker_and_next_job_loads_fresh(self):
        self.assertEqual(self.invoke(), 1)
        process = self.worker.process
        self.worker.close()
        self.assertIsNotNone(process.poll())
        self.assertFalse(self.worker.loaded)
        self.assertEqual(self.invoke(), 1)

    def test_request_cannot_select_a_different_model_or_script(self):
        for argv in (self.argv[:1] + ["/tmp/unknown.py"] + self.argv[2:],
                     ["/different/python"] + self.argv[1:],
                     [str(self.root / "other") if item == str(self.model) else item for item in self.argv]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                self.worker.run(argv, timeout=5)
        self.assertIsNone(self.worker.process)

    def test_cancel_and_timeout_stop_worker_and_clear_residency(self):
        self.request.write_text(json.dumps({"state": "wait", "questions": {}}))
        for timeout, cancel in ((0.2, lambda: False), (5, lambda: True)):
            with self.subTest(timeout=timeout), self.assertRaises((subprocess.TimeoutExpired, InterruptedError)):
                self.worker.run(self.argv, timeout=timeout, cancelled=cancel)
            self.assertIsNone(self.worker.process)
            self.assertFalse(self.worker.loaded)
