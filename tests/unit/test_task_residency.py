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
from unittest.mock import patch

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
        self.assertEqual(self.worker.telemetry['schema'],'worker-memory/1')
        self.assertEqual(self.worker.telemetry['loads'],1)
        self.assertEqual(self.worker.telemetry['cache_hits'],1)
        self.assertIsNone(self.worker.telemetry['active_bytes'])

    def test_resource_tracker_can_stay_idle_across_warm_requests(self):
        from mlx_agent.compute import ComputeLease
        port = self.root / 'fixture_port.py'
        port.write_text(port.read_text().replace('    model.count += 1',
            '    from multiprocessing.shared_memory import SharedMemory\n'
            '    shared = SharedMemory(create=True, size=1)\n'
            '    shared.close()\n    shared.unlink()\n    model.count += 1'))
        self.assertEqual(self.invoke(), 1)
        pid = self.worker.process.pid
        with ComputeLease('test-idle', self.root / 'compute') as lease:
            lease.bind_process_group(pid)
            lease.release_idle_process_group(pid, self.worker.idle_members())
        self.assertEqual(self.invoke(), 2)
        self.assertEqual(self.worker.process.pid, pid)
        self.assertEqual(self.worker.telemetry['loads'], 1)

    def test_unexpected_idle_child_cannot_release_the_compute_lease(self):
        from mlx_agent.compute import ComputeLease
        port = self.root / 'fixture_port.py'
        port.write_text(port.read_text().replace('    model.count += 1',
            '    import subprocess, sys\n'
            '    model.child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])\n'
            '    model.count += 1'))
        self.assertEqual(self.invoke(), 1)
        pid = self.worker.process.pid
        with ComputeLease('test-idle', self.root / 'compute') as lease:
            lease.bind_process_group(pid)
            with self.assertRaisesRegex(ValueError, 'Unexpected child'):
                lease.release_idle_process_group(pid, self.worker.idle_members())
            self.worker.close()

    def test_inherited_warning_filters_allow_tracker_idle_handoff(self):
        self.worker.environment['PYTHONWARNINGS'] = 'default,ignore::FutureWarning,default'
        self.test_resource_tracker_can_stay_idle_across_warm_requests()

    def test_exec_launcher_retains_resource_tracker_idle_handoff(self):
        launcher = self.root / 'python-launcher'
        launcher.write_text('#!' + sys.executable + '\nimport os,sys\nos.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n')
        launcher.chmod(0o700)
        self.worker.executable = str(launcher)
        self.argv[0] = str(launcher)
        self.test_resource_tracker_can_stay_idle_across_warm_requests()

    def test_tracker_from_another_interpreter_is_not_idle(self):
        self.assertEqual(self.invoke(), 1)
        pid = self.worker.process.pid
        identity = ('Mon Jan 1 00:00:00 2024 /different/python'
                    + ' -c from multiprocessing.resource_tracker import main;main(6)')
        with patch('mlx_agent.process_tree._snapshot', return_value={pid + 1: (pid, pid, identity)}):
            with self.assertRaisesRegex(ValueError, 'Unexpected child'):
                self.worker.idle_members()

    def test_tracker_cannot_add_warning_flags_absent_from_launch_environment(self):
        self.worker.environment['PYTHONWARNINGS'] = 'default'
        self.assertEqual(self.invoke(), 1)
        pid = self.worker.process.pid
        identity = ('Mon Jan 1 00:00:00 2024 ' + self.worker.executable
                    + ' -Walways -c from multiprocessing.resource_tracker import main;main(6)')
        with patch('mlx_agent.process_tree._snapshot', return_value={pid + 1: (pid, pid, identity)}):
            with self.assertRaisesRegex(ValueError, 'Unexpected child'):
                self.worker.idle_members()

    def test_launcher_exec_binds_the_ready_worker_identity(self):
        from mlx_agent.compute import process_identity
        launcher = self.root / 'python-launcher'
        launcher.write_text('#!' + sys.executable + '\nimport os,sys,time\ntime.sleep(.5)\nos.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n')
        launcher.chmod(0o700)
        worker = resident_runner.ResidentRunner(str(launcher), str(self.model),
            environment=dict(os.environ, PYTHONPATH=str(self.root)))
        try:
            response = worker.run([str(launcher), *self.argv[1:]], timeout=5)
            self.assertEqual(response.returncode, 0, response.stderr)
            self.assertEqual(worker.identity, process_identity(worker.process.pid))
            worker.close()
            self.assertIsNone(worker.process)
        finally:
            try:
                worker.close()
            except OSError:
                # Test fixture cleanup if the regression leaves the launched child unbound.
                worker.process.terminate()
                worker.process.wait(timeout=5)
                worker.close()

    def test_release_reaps_worker_and_next_job_loads_fresh(self):
        self.assertEqual(self.invoke(), 1)
        process = self.worker.process
        self.worker.close()
        self.assertIsNotNone(process.poll())
        self.assertFalse(self.worker.loaded)
        self.assertEqual(self.invoke(), 1)

    def test_stop_confirmation_reaps_child_before_clearing_residency(self):
        self.assertEqual(self.invoke(), 1)
        process = self.worker.process
        self.addCleanup(process.wait, timeout=5)

        def stopped(*args, **kwargs):
            process.terminate()
            return True

        # Group inspection may confirm termination before Popen has collected
        # the direct child's exit status. close must finish that ownership step.
        with patch.object(resident_runner, '_stop', side_effect=stopped):
            self.worker.close()
        self.assertIsNotNone(process.returncode)
        with self.assertRaises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
        self.assertIsNone(self.worker.process)
        self.assertFalse(self.worker.loaded)

    def test_unconfirmed_child_exit_retains_process_identity(self):
        self.assertEqual(self.invoke(), 1)
        process, identity = self.worker.process, self.worker.identity
        self.addCleanup(process.wait, timeout=5)
        self.addCleanup(process.terminate)
        error = subprocess.TimeoutExpired(process.args, 3)
        with patch.object(resident_runner, '_stop', return_value=True), patch.object(process, 'wait', side_effect=error):
            with self.assertRaises(OSError):
                self.worker.close()
        self.assertIs(self.worker.process, process)
        self.assertEqual(self.worker.identity, identity)
        self.assertTrue(self.worker.loaded)

    def test_unconfirmed_process_group_retains_process_identity(self):
        self.assertEqual(self.invoke(), 1)
        process, identity = self.worker.process, self.worker.identity
        with patch.object(resident_runner, '_stop', return_value=False):
            with self.assertRaises(OSError):
                self.worker.close()
        self.assertIs(self.worker.process, process)
        self.assertEqual(self.worker.identity, identity)
        self.assertTrue(self.worker.loaded)

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
