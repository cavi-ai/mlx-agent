"""Host admission and native bootstrap use isolated, model-free processes."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from mlx_agent.compute import ComputeBusy, ComputeLease, ExecutionAdmission, group_members
from mlx_agent.jit_serve import validate_memory_policy
from mlx_agent.owned_worker import command


class ExecutionAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.compute = self.root / 'compute'

    def test_timeout_cancellation_and_same_backend_reference_count(self):
        admission = ExecutionAdmission('coding', self.compute)
        self.addCleanup(admission.close)
        admission.acquire()
        admission.acquire()
        self.assertEqual(admission.users, 2)
        with self.assertRaises(ComputeBusy):
            ComputeLease('mesh', self.compute).enter_wait(.03)
        admission.release(lambda _: None)
        with self.assertRaises(ComputeBusy):
            ComputeLease('mesh', self.compute).__enter__()
        cancelled = threading.Event()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(ComputeLease('mesh', self.compute).enter_wait, 2, cancelled.is_set)
            cancelled.set()
            with self.assertRaises(InterruptedError):
                future.result(timeout=1)
        admission.release(lambda _: None)
        with ComputeLease('mesh', self.compute):
            pass

    def test_failed_idle_inspection_retains_descriptor_until_confirmed_close(self):
        admission = ExecutionAdmission('coding', self.compute)
        self.addCleanup(admission.close)
        admission.acquire()
        def unresolved(_):
            raise OSError('shutdown unconfirmed')
        with self.assertRaises(OSError):
            admission.release(unresolved)
        self.assertIsNotNone(admission.lease.descriptor)
        with self.assertRaises(ComputeBusy):
            with ComputeLease('mesh', self.compute):
                pass
        admission.close()
        with ComputeLease('mesh', self.compute):
            pass

    def test_unverified_group_is_persisted_before_inspection(self):
        identities = {os.getpid(): 'gateway'}
        with ComputeLease('coding', self.compute, inspector=identities.get, group_probe=lambda _: [4242]) as lease:
            with self.assertRaises(ComputeBusy):
                lease.bind_process_group(4242)
        record = json.loads((self.compute / 'owner.json').read_text())
        self.assertEqual(record['execution_group'], {'pgid': 4242, 'leader_identity': 'unverified'})
        with self.assertRaises(ComputeBusy):
            with ComputeLease('mesh', self.compute, alive=lambda _: False, group_probe=lambda _: [4242]):
                pass

    def test_wait_policy_rejects_invalid_values_without_changing_defaults(self):
        self.assertNotIn('execution_wait_seconds', validate_memory_policy())
        for value in (-1, 3601, True, 1.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_memory_policy({'execution_wait_seconds': value})
        self.assertEqual(validate_memory_policy({'execution_wait_seconds': 3})['execution_wait_seconds'], 3)

    def test_bootstrap_waits_for_persisted_admission_before_running(self):
        marker = self.root / 'started'
        ready_read, ready_write = os.pipe()
        gate_read, gate_write = os.pipe()
        process = None
        try:
            argv = command([sys.executable, '-c', 'import pathlib,time,sys; pathlib.Path(sys.argv[1]).touch(); time.sleep(30)', str(marker)], ready_write, gate_read)
            process = subprocess.Popen(argv, pass_fds=(ready_write, gate_read), start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            os.close(ready_write)
            ready_write = None
            os.close(gate_read)
            gate_read = None
            self.assertEqual(os.read(ready_read, 1), b'R')
            self.assertFalse(marker.exists())
            with ComputeLease('coding', self.compute) as lease:
                lease.bind_process_group(process.pid)
                record = json.loads((self.compute / 'owner.json').read_text())
                self.assertEqual(record['execution_group']['pgid'], process.pid)
                os.write(gate_write, b'G')
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(marker.exists())
                process.terminate()
                process.wait(timeout=3)
        finally:
            for descriptor in (ready_read, ready_write, gate_read, gate_write):
                if descriptor is not None:
                    os.close(descriptor)
            if process is not None:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=3)

    def test_parent_death_stops_private_worker_group(self):
        pidfile = self.root / 'pid'
        program = '''import os,pathlib,subprocess,sys,time
from mlx_agent.owned_worker import command
from mlx_agent.compute import ComputeLease
r,w=os.pipe(); g,h=os.pipe()
p=subprocess.Popen(command([sys.executable,'-c','import time; time.sleep(120)'],w,g),pass_fds=(w,g),start_new_session=True)
os.close(w); os.close(g)
assert os.read(r,1)==b'R'
with ComputeLease('coding',sys.argv[2]) as lease:
    lease.bind_process_group(p.pid)
    os.write(h,b'G'); os.close(h)
    pathlib.Path(sys.argv[1]).write_text(str(p.pid))
    time.sleep(120)
'''
        parent = subprocess.Popen([sys.executable, '-c', program, str(pidfile), str(self.compute)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        worker = None
        try:
            deadline = time.monotonic() + 5
            while not pidfile.exists() and parent.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(pidfile.exists())
            worker = int(pidfile.read_text())
            with self.assertRaises(ComputeBusy):
                with ComputeLease('mesh', self.compute):
                    pass
            parent.kill()
            parent.wait(timeout=3)
            deadline = time.monotonic() + 3
            while group_members(worker) and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertFalse(group_members(worker))
            with ComputeLease('mesh', self.compute):
                pass
        finally:
            if parent.poll() is None:
                parent.kill()
            parent.communicate(timeout=3)
            if worker is not None and group_members(worker):
                os.killpg(worker, signal.SIGKILL)
