"""Exercise real task processes with fixture backends, not model inference."""

import json
import os
import signal
import subprocess
import sys
import time
import unittest
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.backends import INSTALL_MARKER, load_manifests
from mlx_agent.compute import ComputeBusy, ComputeLease, group_members
from mlx_agent.task_contracts import TaskError
from mlx_agent.task_runtime import cancel_task, start_task, supervise, task_result, task_status, _stop
from mlx_agent.task_store import TaskStore


BACKEND = r'''
import json, os, signal, subprocess, sys, time, wave
from pathlib import Path
args = sys.argv[1:]
model = Path(args[args.index("--model") + 1])
config = json.loads((model / "config.json").read_text())
behavior = config.get("fixture_behavior", "success")
if behavior == "descendant":
    child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    (model.parent / "descendant.pid").write_text(str(child.pid))
if behavior == "ignore-term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
(model.parent / "started").write_text(str(os.getpid()))
if behavior in ("sleep", "descendant", "ignore-term"):
    time.sleep(60)
if "--out" in args:
    target = args[args.index("--out") + 1]
    if behavior == "invalid-audio":
        Path(target).write_bytes(b"not audio")
    else:
        with wave.open(target, "wb") as output:
            output.setparams((1, 2, 8000, 2000, "NONE", "not compressed"))
            output.writeframes(b"\x00\x01" * 2000)
    print(json.dumps({"path": target, "sample_rate": 8000, "audio_seconds": .25, "seconds": .1}))
else:
    request = json.loads(Path(args[args.index("--request") + 1]).read_text())
    print(json.dumps({"answers": {"route": "billing"}, "seconds": .1, "usage": {"questions": len(request["questions"])}}))
'''


def launch_inference_supervisor(state, compute, argv, **kwargs):
    identifier = argv[argv.index("--id") + 1]
    code = ("from functools import partial; from pathlib import Path; "
            "from mlx_agent.compute import ComputeLease; from mlx_agent.task_runtime import supervise; "
            "raise SystemExit(supervise({!r}, Path({!r}), lease_factory=partial(ComputeLease, root=Path({!r}))))").format(identifier, str(state), str(compute))
    return subprocess.Popen([sys.executable, "-c", code], **kwargs)


class TaskProcessTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.state = self.root / "tasks"
        self.compute = self.root / "compute"
        self.processes = []
        self.addCleanup(self.stop_processes)
        self.environment = patch.dict(os.environ, {"XDG_DATA_HOME": str(self.root / "data")})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.model = self.root / "model"
        self.model.mkdir()
        for identifier in ("mlx-audio", "mlx-embeddings"):
            manifest = load_manifests()[identifier]
            backend = self.root / "data/mlx-workbench/backends" / identifier
            (backend / "bin").mkdir(parents=True)
            executable = backend / "bin/python"
            executable.write_text("#!" + sys.executable + "\n" + BACKEND)
            executable.chmod(0o700)
            (backend / INSTALL_MARKER).write_text(json.dumps({"id": identifier, "version": manifest["version"]}))
            (backend / "lib/python3.12/site-packages" / manifest["package"]).mkdir(parents=True)

    def stop_processes(self):
        # Only process groups created by this test are eligible for cleanup.
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    if group_members(process.pid):
                        raise
            process.wait(timeout=10)
        if self.state.exists():
            for record in TaskStore(self.state).list(100):
                execution = record.get("execution")
                if execution and group_members(execution["pid"]):
                    try:
                        os.killpg(execution["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def launch(self, argv, **kwargs):
        process = launch_inference_supervisor(self.state, self.compute, argv, **kwargs)
        self.processes.append(process)
        return process

    def prepare_request(self, kind="classification", behavior="success"):
        model_type = "laya" if kind == "classification" else "minimax_music3"
        (self.model / "config.json").write_text(json.dumps({"model_type": model_type, "fixture_behavior": behavior}))
        (self.model / "tokenizer.json").write_text("{}")
        params = {"state": "refund", "questions": {"route": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": ["billing", "other"]}}} if kind == "classification" else {"caption": "Quiet piano"}
        request = self.root / "request.json"
        request.write_text(json.dumps({"schema": "task-request/1", "kind": kind, "model_path": str(self.model), "parameters": params}))
        return request

    def start(self, kind="classification", behavior="success", launcher=None):
        return start_task(self.prepare_request(kind, behavior), self.state, launcher=launcher or self.launch)

    def wait_for(self, condition, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = condition()
            if value:
                return value
            time.sleep(.05)
        self.fail("Task condition was not reached before its deadline")

    def terminal(self, identifier):
        value = task_status(identifier, self.state)
        return value if value["state"] in ("succeeded", "failed", "cancelled") else None

    def test_real_workers_preserve_classifier_and_music_results(self):
        for kind in ("classification", "music_generation"):
            with self.subTest(kind=kind):
                task = self.start(kind)
                result = self.wait_for(lambda: self.terminal(task["id"]))
                self.assertEqual(result["state"], "succeeded", result)
                evidence = task_result(task["id"], self.state)
                self.assertEqual(evidence["result"]["schema"], "decide/1" if kind == "classification" else "music/1")
                self.assertEqual(evidence["result"]["model"], str(self.model))
                self.assertTrue(Path(evidence["artifact"]["path"]).is_file())
                self.processes[-1].wait(timeout=10)

    def test_cancel_waits_for_descendants(self):
        task = self.start(behavior="descendant")
        self.wait_for(lambda: (self.root / "started").exists())
        descendant = int((self.root / "descendant.pid").read_text())
        requested = cancel_task(task["id"], self.state)
        self.assertTrue(requested["cancel_requested"])
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "cancelled", result)
        self.assertFalse(result["execution_unresolved"])
        from mlx_agent.compute import group_members
        self.assertNotIn(descendant, group_members(result["execution"]["pid"]))

    def test_supervisor_loss_keeps_compute_busy(self):
        task = self.start(behavior="sleep")
        self.wait_for(lambda: (self.root / "started").exists())
        supervisor = self.processes[-1]
        supervisor.kill()
        supervisor.wait(timeout=10)
        result = task_status(task["id"], self.state)
        self.assertEqual(result["state"], "failed")
        self.assertTrue(result["execution_unresolved"])
        with self.assertRaises(ComputeBusy):
            with ComputeLease("another model", self.compute):
                pass

    def test_cancel_during_spawn(self):
        task = self.start(behavior="sleep")
        cancel_task(task["id"], self.state)
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "cancelled", result)
        self.assertFalse((self.root / "started").exists())

    def test_caller_exit_does_not_end_task(self):
        request = self.prepare_request()
        code = ("import json; from functools import partial; from pathlib import Path; "
                "from tests.integration.test_task_processes import launch_inference_supervisor; "
                "from mlx_agent.task_runtime import start_task; "
                "print(json.dumps(start_task(Path({!r}), Path({!r}), launcher=partial(launch_inference_supervisor, {!r}, {!r}))))").format(str(request), str(self.state), str(self.state), str(self.compute))
        caller = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        self.assertEqual(caller.returncode, 0, caller.stderr)
        record = json.loads(caller.stdout)
        result = self.wait_for(lambda: self.terminal(record["id"]))
        self.assertEqual(result["state"], "succeeded", result)
        self.assertEqual(task_result(record["id"], self.state)["result"]["answers"], {"route": "billing"})

    def test_term_escalates_to_kill(self):
        record = self.start(behavior="ignore-term")
        self.wait_for(lambda: (self.root / "started").exists())
        cancel_task(record["id"], self.state)
        result = self.wait_for(lambda: self.terminal(record["id"]))
        self.assertEqual(result["state"], "cancelled", result)
        self.assertEqual(group_members(result["execution"]["pid"]), [])

    def test_pid_reuse_never_signals_unrelated_process(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        self.processes.append(process)
        self.assertFalse(_stop(process, "a different process identity", {}, grace=.01))
        self.assertIsNone(process.poll())

    def test_busy_lease_is_a_retained_failure(self):
        with ComputeLease("existing model", self.compute):
            record = self.start()
            result = self.wait_for(lambda: self.terminal(record["id"]))
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"]["code"], "compute_busy")
        self.assertFalse((self.root / "started").exists())

    def test_launch_failure_is_retained(self):
        def unavailable(*args, **kwargs):
            raise OSError("fixture launch failure")
        result = self.start(launcher=unavailable)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"]["code"], "task_launch_failed")
        self.assertEqual(TaskStore(self.state).get(result["id"]), result)

    def test_changed_identity_before_execution_is_refused(self):
        def changed(argv, **kwargs):
            (self.model / "config.json").write_text('{"model_type":"changed"}')
            return self.launch(argv, **kwargs)
        task = self.start(launcher=changed)
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"]["code"], "identity_changed")
        self.assertFalse((self.root / "started").exists())

    def test_invalid_wav_is_not_success(self):
        task = self.start("music_generation", "invalid-audio")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "failed")
        self.assertIsNone(result["result"])

    def test_result_storage_failure_is_not_success(self):
        task = self.start(launcher=lambda *args, **kwargs: None)
        with patch.object(TaskStore, "write_result", side_effect=TaskError("task_storage_failed", "disk full")):
            supervise(task["id"], self.state, lease_factory=partial(ComputeLease, root=self.compute))
        result = task_status(task["id"], self.state)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"]["code"], "task_storage_failed")

    def test_success_cancel_race(self):
        task = self.start(launcher=lambda *args, **kwargs: None)
        original_write = TaskStore.write_result
        def cancel_after_write(store, identifier, value):
            reference = original_write(store, identifier, value)
            store.request_cancel(identifier)
            return reference
        with patch.object(TaskStore, "write_result", cancel_after_write):
            supervise(task["id"], self.state, lease_factory=partial(ComputeLease, root=self.compute))
        result = task_status(task["id"], self.state)
        self.assertEqual(result["state"], "cancelled")
        self.assertIsNone(result["result"])

    def test_changed_music_output_is_not_returned_as_valid_evidence(self):
        task = self.start("music_generation")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "succeeded", result)
        evidence = task_result(task["id"], self.state)
        Path(evidence["result"]["path"]).write_bytes(b"changed")
        with self.assertRaises(TaskError):
            task_result(task["id"], self.state)

    def test_cancel_while_execution_identity_is_being_bound(self):
        from mlx_agent import task_runtime
        task = self.start(launcher=lambda *args, **kwargs: None)
        spawn = task_runtime._spawn
        def cancel_after_spawn(role, identifier, store, **kwargs):
            process = spawn(role, identifier, store, **kwargs)
            store.request_cancel(identifier)
            process.wait(timeout=10)
            return process
        with patch.object(task_runtime, "_spawn", cancel_after_spawn):
            supervise(task["id"], self.state, lease_factory=partial(ComputeLease, root=self.compute))
        result = task_status(task["id"], self.state)
        self.assertEqual(result["state"], "cancelled", result)
        self.assertFalse((self.root / "started").exists())
