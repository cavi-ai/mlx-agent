"""Resource budgets and worker transitions for Strata serving."""
import json
import os
import signal
import subprocess
import sys
import time
import struct
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mlx_agent import strata_runtime as strata
from mlx_agent.jit_serve import GatewayError, ModelWorker, validate_memory_policy
from mlx_agent.serve import ServeError, load_recipes, plan_start
from .test_gguf import _string, write_gguf
from . import test_strata_runtime as fixtures


class StrataMemoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StrataProfileTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.valid_model()
        self.root = self.fixture.root
        first = self.root/'Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'
        write_gguf(first, architecture='qwen4exp', tensor_count=2)
        metadata = first.read_bytes()
        table = (_string('blk.0.ffn_up_exps.weight') + struct.pack('<IQQIQ', 2, 64, 100000000, 42, 0)
                 + _string('output.weight') + struct.pack('<IQIQ', 1, 32, 1, 1800000000))
        prefix = metadata+table
        with first.open('wb') as f:
            f.write(prefix+b'\0'*((-len(prefix))%32))
            f.truncate(f.tell()+1800000064)  # Sparse metadata fixture; tensor data is never loaded.
        self.model = strata.read_model(self.root)

    def test_total_budget_resolves_paging_and_retains_fast_resident_path(self):
        small = strata.memory_allocation(self.model, 5.9, 8192)
        self.assertEqual(small['mode'], 'paged')
        self.assertEqual(small['expert_memory_gb'], .654)
        self.assertLessEqual(small['estimated_resident_bytes'], 5900000000)
        full = strata.memory_allocation(self.model, 7.2, 8192)
        self.assertEqual(full['mode'], 'resident')
        self.assertIsNone(full['expert_memory_gb'])
        self.assertLessEqual(full['estimated_resident_bytes'], 10000000000)

    def test_draft_is_budgeted_and_loaded_only_when_enabled(self):
        model = dict(self.model, mtp='/prepared/mtp', mtp_bytes=2000000000,
                     memory_estimate_bytes=self.model['memory_estimate_bytes']+2000000000)
        self.assertEqual(strata.memory_estimate(model, 1, 8192, speculation=True)
                         - strata.memory_estimate(model, 1, 8192), 2000000000)
        runtime = {'engine': '/runtime/engine', 'root': '/runtime'}
        off = strata.frontend_config(model, runtime, memory_budget_gb=10, context_tokens=8192)
        on = strata.frontend_config(model, runtime, speculation=True, memory_budget_gb=10, context_tokens=8192)
        self.assertNotIn('--mtp', off['args'])
        self.assertIn('--mtp', on['args'])
        self.assertIn('--conversation-cache-mib', off['args'])
        self.assertEqual(off['args'][off['args'].index('--conversation-cache-mib')+1], '0')

    def test_bad_or_too_small_total_budgets_are_rejected(self):
        for budget in (0, 5, True, float('nan'), float('inf'), 1025, '8'):
            with self.subTest(budget=budget), self.assertRaises(strata.StrataError):
                strata.memory_allocation(self.model, budget, 8192)

    def test_preview_binds_total_allocation_and_disallows_conflicting_controls(self):
        self.fixture.installed()
        plan = plan_start(None, 'strata', load_recipes(), path=str(self.root), jit=True,
                          strata_context_tokens=8192, memory_policy={'model_memory_budget_gb': 5.9})
        self.assertEqual(plan['memory_policy']['minimum_headroom_gb'], 2)
        self.assertEqual(plan['strata_memory_allocation']['expert_memory_gb'], .654)
        self.assertIn('--memory-budget-gb', plan['worker_argv'])
        other = plan_start(None, 'strata', load_recipes(), path=str(self.root), jit=True,
                          strata_context_tokens=8192, memory_policy={'model_memory_budget_gb': 10})
        self.assertNotEqual(plan['preview_hash'], other['preview_hash'])
        with self.assertRaises(ServeError):
            plan_start(None, 'strata', load_recipes(), path=str(self.root), jit=True,
                       strata_expert_memory_gb=1, memory_policy={'model_memory_budget_gb': 10})

    def worker(self):
        return ModelWorker({'local_path': str(self.root), 'runtime': 'strata',
            'strata_context_tokens': 8192, 'strata_speculation': False,
            'worker_argv': ['python', 'strata_runtime.py', '--port', '{worker_port}'],
            'memory_policy': {'minimum_headroom_gb': 2}})

    def test_budget_change_preserves_active_requests_then_reloads_when_idle(self):
        worker = self.worker()
        worker.process = Mock()
        worker.process.poll.return_value = None
        worker.model_state = 'loaded'
        worker.active_requests = 1
        status = worker.configure({'model_memory_budget_gb': 5.9})
        self.assertTrue(status['memory_budget_pending'])
        worker.process.terminate.assert_not_called()
        self.assertFalse(worker.unload_if_idle())
        with self.assertRaises(GatewayError) as caught:
            worker.acquire()
        self.assertEqual(caught.exception.code, 'model_reconfiguring')
        worker.release()
        with patch.object(worker, '_stop_worker'):
            self.assertTrue(worker.unload_if_idle())
        self.assertEqual(worker.model_state, 'unloaded')
        self.assertFalse(worker.status()['memory_budget_pending'])
        self.assertIn('--memory-budget-gb', worker.config['worker_argv'])

    def test_invalid_busy_and_failed_persistence_leave_policy_unchanged(self):
        worker = self.worker()
        previous = dict(worker.memory_policy)
        for state, budget in (('loading', 10), ('unloading', 10), ('unloaded', 5)):
            worker.model_state = state
            with self.subTest(state=state), self.assertRaises(GatewayError):
                worker.configure({'model_memory_budget_gb': budget})
            self.assertEqual(worker.memory_policy, previous)
        worker.model_state = 'unloaded'
        worker.config['config_path'] = str(self.root/'gateway.json')
        with patch('mlx_agent.jit_serve._atomic_in_directory', side_effect=OSError('disk full')):
            with self.assertRaises(GatewayError): worker.configure({'model_memory_budget_gb': 10})
        self.assertEqual(worker.memory_policy, previous)
        self.assertNotIn('--memory-budget-gb', worker.config['worker_argv'])

    def test_budget_admission_cannot_be_skipped_and_status_exposes_allocation(self):
        policy = validate_memory_policy({'model_memory_budget_gb': 5.9})
        self.assertEqual(policy['minimum_headroom_gb'], 2)
        worker = self.worker()
        worker.configure({'model_memory_budget_gb': 5.9})
        allocation = worker.status()['memory_allocation']
        required = allocation['estimated_resident_bytes']+2000000000
        worker.memory_probe = lambda: required-1
        with self.assertRaises(GatewayError): worker._check_memory()
        worker.memory_probe = lambda: required
        worker._check_memory()
        self.assertEqual(worker.memory_check['required_available_bytes'], required)

    def test_strata_shutdown_allows_native_engine_to_release_its_children(self):
        worker = self.worker()
        worker.process = Mock()
        worker.process.poll.return_value = None
        process = worker.process
        process.pid = 2000000000
        worker._stop_worker()
        process.wait.assert_called_once_with(timeout=65)

    def test_exited_worker_clears_a_pending_budget_reload(self):
        worker = self.worker()
        worker.process = Mock()
        worker.process.poll.return_value = 0
        worker.model_state = 'loaded'
        worker.memory_budget_pending = True
        self.assertFalse(worker.status()['memory_budget_pending'])
        self.assertEqual(worker.model_state, 'unloaded')

    def test_policy_helper_can_explicitly_clear_but_does_not_implicitly_clear_budget(self):
        from mlx_agent.serve import configure_serve_memory
        with patch('mlx_agent.serve._configure_owned_jit', return_value={}) as send:
            configure_serve_memory(19080, {'model_memory_budget_gb': None})
            self.assertIn('model_memory_budget_gb', send.call_args.args[4])
            self.assertIsNone(send.call_args.args[4]['model_memory_budget_gb'])
            configure_serve_memory(19080, {'minimum_headroom_gb': 4})
            self.assertNotIn('model_memory_budget_gb', send.call_args.args[4])

    @unittest.skipUnless(os.name == 'posix', 'Strata requires a POSIX process tree')
    def test_forced_frontend_exit_does_not_leave_engine_or_kill_other_work(self):
        from mlx_agent.compute import process_identity
        ready = self.root/'engine-ready'
        pidfile = self.root/'engine-pid'
        engine = "import signal,time,pathlib,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(sys.argv[1]).touch(); time.sleep(120)"
        front = ("import signal,subprocess,sys,pathlib,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                 "p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); pathlib.Path(sys.argv[3]).write_text(str(p.pid)); time.sleep(120)")
        process = subprocess.Popen([sys.executable,'-c',front,engine,str(ready),str(pidfile)])
        other = subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
        engine_pid = identity = None
        try:
            deadline = time.monotonic()+10
            while not ready.exists() and time.monotonic()<deadline: time.sleep(.01)
            self.assertTrue(ready.exists())
            engine_pid = int(pidfile.read_text()); identity = process_identity(engine_pid)
            worker = self.worker(); worker.process = process; worker.model_state = 'loaded'
            wait = process.wait
            def timed_wait(timeout=None):
                if timeout == 65: raise subprocess.TimeoutExpired('frontend',65)
                return wait(timeout=timeout)
            with patch.object(process,'wait',side_effect=timed_wait): worker.unload()
            state = subprocess.run(['/bin/ps','-p',str(engine_pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
            self.assertTrue(not state or state.startswith('Z'), state)
            self.assertIsNone(other.poll())
            self.assertEqual(worker.model_state,'unloaded')
        finally:
            if process.poll() is None: process.kill()
            process.wait(timeout=3)
            if engine_pid and identity and process_identity(engine_pid)==identity:
                try: os.kill(engine_pid,signal.SIGKILL)
                except ProcessLookupError: pass
            other.kill(); other.wait(timeout=3)

    def test_failed_child_cleanup_blocks_reload_and_keeps_failed_state(self):
        worker = self.worker()
        worker.strata_children = {2000000000: 'captured engine'}
        worker.model_state = 'failed'
        with patch('mlx_agent.process_tree.remaining', return_value=worker.strata_children), \
                patch('mlx_agent.process_tree.terminate', side_effect=OSError('engine still live')), \
                patch('mlx_agent.jit_serve.subprocess.Popen') as spawn:
            with self.assertRaises(GatewayError) as caught: worker.acquire()
        self.assertEqual(caught.exception.code, 'model_unload_failed')
        self.assertEqual(worker.model_state, 'failed')
        spawn.assert_not_called()

    def test_child_cleanup_never_signals_a_reused_pid(self):
        from mlx_agent import process_tree
        with patch.object(process_tree, '_snapshot', return_value={123: (1, 'replacement')}), \
                patch.object(process_tree.os, 'kill') as kill:
            process_tree.terminate({123: 'captured original'}, grace_seconds=0)
        kill.assert_not_called()
