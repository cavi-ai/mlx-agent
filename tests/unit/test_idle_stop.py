import os
import threading
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.jit_serve import ModelWorker, GatewayError
from mlx_agent.serve import ServeError, stop_serve


class IdleStopTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.worker = ModelWorker({'model': '/model', 'local_path': '/model', 'runtime': 'splash',
                                   'worker_argv': [], 'max_tokens': 100})
        self.addCleanup(self.worker.close)

    def test_idle_stop_seals_admission_and_refuses_busy_states(self):
        for name, value in [('active_requests', 1), ('cache_control_active', True),
                            ('execution_unresolved', True), ('memory_budget_pending', True),
                            ('worker_launch_pending', True), ('model_state', 'loading'), ('model_state', 'unloading')]:
            with self.subTest(name=name, value=value):
                old = getattr(self.worker, name); setattr(self.worker, name, value)
                with self.assertRaises(GatewayError): self.worker.begin_idle_stop()
                self.assertFalse(self.worker.closed)
                setattr(self.worker, name, old)
        self.worker.begin_idle_stop()
        self.assertTrue(self.worker.closed)
        with self.assertRaises(GatewayError) as error: self.worker.acquire()
        self.assertEqual(error.exception.code, 'endpoint_stopping')

    def test_active_admission_and_idle_stop_are_serialized(self):
        entered, finish = threading.Event(), threading.Event()
        errors = []
        def hold(*args):
            entered.set()
            finish.wait(3)
            raise InterruptedError('finish')
        def acquire():
            try: self.worker.acquire()
            except GatewayError as error: errors.append(error.code)
        with mock.patch.object(self.worker.execution, 'acquire', side_effect=hold):
            thread = threading.Thread(target=acquire)
            thread.start()
            self.assertTrue(entered.wait(2))
            with self.assertRaises(GatewayError) as error: self.worker.begin_idle_stop()
            self.assertEqual(error.exception.code, 'model_busy')
            finish.set(); thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.worker.active_requests, 0)
        self.worker.begin_idle_stop()
        with self.assertRaises(GatewayError) as stopped: self.worker.acquire()
        self.assertEqual(stopped.exception.code, 'endpoint_stopping')

    def test_cli_accepts_identity_bound_idle_option(self):
        from mlx_agent.cli import build_parser
        args = build_parser('serve').parse_args(['serve', 'stop', '--port', '8765', '--if-idle',
                                                  '--expected-pid', '1234', '--expected-start', 'birth'])
        self.assertTrue(args.if_idle)
        self.assertEqual((args.expected_pid, args.expected_start), (1234, 'birth'))

    def test_idle_stop_requires_exact_generation_and_never_signals_on_refusal(self):
        receipt = {'pid': 1234, 'port': 8765, 'jit': True, 'runtime': 'splash', 'process_started_at': 'birth'}
        path = self.root / '8765.json'; path.write_text('{}')
        with mock.patch('mlx_agent.serve.receipts_root', return_value=self.root), \
             mock.patch('mlx_agent.serve._read_receipt', return_value=receipt), \
             mock.patch('mlx_agent.serve._pid_start', return_value='birth'), \
             mock.patch('mlx_agent.serve._argv_matches', return_value=True), \
             mock.patch('mlx_agent.serve._jit_control', side_effect=ServeError('model_busy', 'busy', 'wait')) as control:
            terminate = mock.Mock()
            with self.assertRaises(ServeError):
                stop_serve(8765, str(self.root), pid_alive=lambda _: True, pid_command=lambda _: 'command',
                           terminate=terminate, expected_pid=1234, expected_start='birth', if_idle=True)
            control.assert_called_once(); terminate.assert_not_called(); self.assertTrue(path.exists())
        with self.assertRaises(ServeError):
            stop_serve(8765, str(self.root), if_idle=True)

    def test_idle_stop_rejects_reused_pid_with_stale_receipt_birth(self):
        receipt = {'pid': 1234, 'port': 8765, 'jit': True, 'runtime': 'splash', 'process_started_at': 'old'}
        path = self.root / '8765.json'; path.write_text('{}')
        with mock.patch('mlx_agent.serve.receipts_root', return_value=self.root), \
             mock.patch('mlx_agent.serve._read_receipt', return_value=receipt), \
             mock.patch('mlx_agent.serve._pid_start', return_value='new'), \
             mock.patch('mlx_agent.serve._jit_control') as control:
            with self.assertRaises(ServeError):
                stop_serve(8765, str(self.root), expected_pid=1234, expected_start='new', if_idle=True)
            control.assert_not_called(); self.assertTrue(path.exists())

    def test_idle_stop_never_cleans_a_replacement_receipt(self):
        receipt = {'pid': 1234, 'port': 8765, 'jit': True, 'runtime': 'splash', 'process_started_at': 'birth'}
        replacement = dict(receipt, pid=5678)
        path = self.root / '8765.json'; path.write_text('replacement')
        with mock.patch('mlx_agent.serve.receipts_root', return_value=self.root), \
             mock.patch('mlx_agent.serve._read_receipt', side_effect=[receipt, replacement]), \
             mock.patch('mlx_agent.serve._pid_start', return_value='birth'), \
             mock.patch('mlx_agent.serve._argv_matches', return_value=True), \
             mock.patch('mlx_agent.serve._jit_control', return_value={'pid': 1234, 'process_started_at': 'birth', 'accepted': True}), \
             mock.patch('mlx_agent.model_serving.release_stopped_service'):
            terminate = mock.Mock()
            with self.assertRaises(ServeError) as error:
                stop_serve(8765, str(self.root), pid_alive=mock.Mock(side_effect=[True, False]),
                           pid_command=lambda _: 'command', terminate=terminate,
                           expected_pid=1234, expected_start='birth', if_idle=True)
            self.assertEqual(error.exception.code, 'server_changed')
            terminate.assert_not_called(); self.assertEqual(path.read_text(), 'replacement')

    def test_idle_stop_removes_only_matching_receipt(self):
        receipt = {'pid': 1234, 'port': 8765, 'jit': True, 'runtime': 'splash', 'process_started_at': 'birth'}
        path = self.root / '8765.json'; path.write_text('{}')
        with mock.patch('mlx_agent.serve.receipts_root', return_value=self.root), \
             mock.patch('mlx_agent.serve._read_receipt', return_value=receipt), \
             mock.patch('mlx_agent.serve._pid_start', return_value='birth'), \
             mock.patch('mlx_agent.serve._argv_matches', return_value=True), \
             mock.patch('mlx_agent.serve._jit_control', return_value={'pid': 1234, 'process_started_at': 'birth', 'accepted': True}), \
             mock.patch('mlx_agent.model_serving.release_stopped_service') as release:
            terminate = mock.Mock()
            result = stop_serve(8765, str(self.root), pid_alive=mock.Mock(side_effect=[True, False]),
                                pid_command=lambda _: 'command', terminate=terminate,
                                expected_pid=1234, expected_start='birth', if_idle=True)
            self.assertEqual(result['status'], 'stopped')
            release.assert_called_once_with(receipt)
            terminate.assert_not_called(); self.assertFalse(path.exists())

    def test_idle_stop_old_gateway_refusal_preserves_receipt(self):
        receipt = {'pid': 1234, 'port': 8765, 'jit': True, 'runtime': 'splash', 'process_started_at': 'birth'}
        path = self.root / '8765.json'; path.write_text('{}')
        with mock.patch('mlx_agent.serve.receipts_root', return_value=self.root), \
             mock.patch('mlx_agent.serve._read_receipt', return_value=receipt), \
             mock.patch('mlx_agent.serve._pid_start', return_value='birth'), \
             mock.patch('mlx_agent.serve._argv_matches', return_value=True), \
             mock.patch('mlx_agent.serve._jit_control', side_effect=ServeError('control_unavailable', 'unsupported', 'retry')):
            terminate = mock.Mock()
            with self.assertRaises(ServeError):
                stop_serve(8765, str(self.root), pid_alive=lambda _: True, pid_command=lambda _: 'command',
                           terminate=terminate, expected_pid=1234, expected_start='birth', if_idle=True)
            terminate.assert_not_called(); self.assertTrue(path.exists())

    def test_vanished_gateway_never_uses_legacy_orphan_cleanup(self):
        receipt = {'pid': 1234, 'port': 8765, 'jit': True, 'runtime': 'splash', 'process_started_at': 'birth'}
        path = self.root / '8765.json'; path.write_text('{}')
        with mock.patch('mlx_agent.serve.receipts_root', return_value=self.root), \
             mock.patch('mlx_agent.serve._read_receipt', return_value=receipt), \
             mock.patch('mlx_agent.serve._pid_start', return_value='birth'), \
             mock.patch('mlx_agent.serve._owned_jit_worker') as orphan:
            with self.assertRaises(ServeError):
                stop_serve(8765, str(self.root), pid_alive=lambda _: False,
                           expected_pid=1234, expected_start='birth', if_idle=True)
            orphan.assert_not_called(); self.assertTrue(path.exists())

    def test_receipt_target_lock_serializes_publication_and_cleanup(self):
        from mlx_agent.transactions import _target_locks, ConcurrentTransactionError
        target = self.root / '8765.json'
        entered, release, acquired = threading.Event(), threading.Event(), threading.Event()
        def hold():
            with _target_locks([target], blocking=True):
                entered.set(); release.wait(3)
        def wait():
            with _target_locks([target], blocking=True): acquired.set()
        owner = threading.Thread(target=hold); owner.start(); self.assertTrue(entered.wait(2))
        waiter = threading.Thread(target=wait); waiter.start()
        self.assertFalse(acquired.wait(.05))
        with self.assertRaises(ConcurrentTransactionError):
            with _target_locks([target]): pass
        release.set(); owner.join(3); waiter.join(3)
        self.assertTrue(acquired.is_set()); self.assertFalse(waiter.is_alive())

    def test_idle_stop_rejects_wrong_listener_identity(self):
        receipt = {'pid': 1234, 'port': 8765, 'jit': True, 'runtime': 'splash', 'process_started_at': 'birth'}
        path = self.root / '8765.json'; path.write_text('{}')
        with mock.patch('mlx_agent.serve.receipts_root', return_value=self.root), \
             mock.patch('mlx_agent.serve._read_receipt', return_value=receipt), \
             mock.patch('mlx_agent.serve._pid_start', return_value='birth'), \
             mock.patch('mlx_agent.serve._argv_matches', return_value=True), \
             mock.patch('mlx_agent.serve._jit_control', return_value={'pid': 9999, 'process_started_at': 'birth', 'accepted': True}):
            terminate = mock.Mock()
            with self.assertRaises(ServeError):
                stop_serve(8765, str(self.root), pid_alive=lambda _: True, pid_command=lambda _: 'command',
                           terminate=terminate, expected_pid=1234, expected_start='birth', if_idle=True)
            terminate.assert_not_called(); self.assertTrue(path.exists())
