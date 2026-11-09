import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler

from mlx_agent import private_serve


class PrivateRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='mlx-runtime-test-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / 'model'
        self.model.mkdir()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.root / 'worker.sock'))
        self.listener.listen()
        self.addCleanup(self.listener.close)

    def runtime(self):
        folder = self.root / 'venv'
        (folder / 'bin').mkdir(parents=True)
        (folder / 'pyvenv.cfg').write_text('fixture')
        python = folder / 'bin/python'
        python.symlink_to(sys.executable)
        entry = folder / 'bin/mlx_lm.server'
        entry.write_text('#!' + str(python) + '\n')
        entry.chmod(0o755)
        return entry, python

    def test_runtime_interpreter_requires_the_entrypoints_own_environment(self):
        entry, python = self.runtime()
        self.assertEqual(private_serve.runtime_python(str(entry)), str(python))
        command = private_serve.worker_command([str(entry), '--model', 'fixed'], 'fixed', 8)
        self.assertEqual(command[0], str(python))
        self.assertIn('--runtime-child', command)
        entry.write_text('#!/usr/bin/env python3\n')
        with self.assertRaises(ValueError):
            private_serve.runtime_python(str(entry))
        with self.assertRaises(ValueError):
            private_serve.runtime_python('/missing/mlx_lm.server')

    def test_runtime_adopts_unix_and_locks_forwarded_model_and_bind(self):
        calls = []
        def run(server_class=None, handler_class=None):
            server = server_class(('0.0.0.0', 1), handler_class)
            calls.append((server.socket.family, list(sys.argv), os.environ.get('HF_HUB_OFFLINE')))
            server.server_close()
        server = SimpleNamespace(run=run, APIHandler=BaseHTTPRequestHandler)
        server.main = lambda: server.run()
        mlx = ModuleType('mlx_lm')
        mlx.server = server
        previous = sys.argv
        with patch.dict(sys.modules, {'mlx_lm': mlx}), patch.dict(os.environ):
            private_serve.run_runtime(str(self.model), os.dup(self.listener.fileno()), ['--host', '0.0.0.0', '--model', 'other'])
        self.assertEqual(calls[0][0], socket.AF_UNIX)
        self.assertEqual(calls[0][1][-8:], ['--model', str(self.model.resolve()), '--host', '127.0.0.1', '--port', '0', '--log-level', 'ERROR'])
        self.assertEqual(calls[0][2], '1')
        self.assertIs(server.run, run)
        self.assertIs(sys.argv, previous)

    def test_cache_miss_is_offline_and_never_calls_the_server(self):
        mlx = ModuleType('mlx_lm')
        def run(server_class=None, handler_class=None):
            self.fail('a missing cached model reached the server')
        mlx.server = SimpleNamespace(run=run, APIHandler=BaseHTTPRequestHandler)
        hub = ModuleType('huggingface_hub')
        hub.try_to_load_from_cache = lambda *args: None
        with patch.dict(sys.modules, {'mlx_lm': mlx, 'huggingface_hub': hub}), patch.dict(os.environ), \
                patch.object(private_serve, 'private_server_class', return_value=object):
            with self.assertRaisesRegex(ValueError, 'local cache'):
                private_serve.run_runtime('missing/model', 99, [])

    def test_supervisor_reexec_retains_only_the_explicit_descriptor(self):
        entry, python = self.runtime()
        descriptor = os.dup(self.listener.fileno())
        try:
            with patch('os.execv', side_effect=RuntimeError('exec intercepted')) as execute:
                with self.assertRaisesRegex(RuntimeError, 'intercepted'):
                    private_serve.main(['--model', str(self.model), '--socket-fd', str(descriptor), '--runtime-executable', str(entry)])
            self.assertEqual(execute.call_args.args[0], str(python))
            self.assertTrue(os.get_inheritable(descriptor))
            self.assertIn('--runtime-child', execute.call_args.args[1])
        finally:
            os.close(descriptor)

    def test_private_endpoint_cleanup_and_connection_failure_close_socket(self):
        endpoint = private_serve.PrivateEndpoint()
        path = endpoint.path
        connection = endpoint.connection(timeout=1)
        connection.connect()
        connection.close()
        endpoint.close()
        self.assertFalse(Path(path).exists())
        missing = private_serve.UnixHTTPConnection(path, timeout=1)
        with self.assertRaises(OSError):
            missing.connect()
        self.assertIsNone(missing.sock)
