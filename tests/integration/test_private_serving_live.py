"""Opt-in protected gateway proof with an explicitly selected cached model."""
import http.client
import json
import os
from pathlib import Path
import secrets
import subprocess
import socket
from unittest import mock
from tempfile import TemporaryDirectory
import unittest

from mlx_agent.serve import plan_start, start_serve, stop_serve, status_serve, unload_serve, load_recipes
from mlx_agent.private_serve import identity, PRIVATE_PROTOCOL
from mlx_agent.serving_access import read_access_key


class ProtectedServingLocalModelTests(unittest.TestCase):
    def test_cached_model_streams_only_through_protected_gateway(self):
        model = os.environ.get('MLX_AGENT_TEST_LOCAL_MODEL')
        runtime = os.environ.get('MLX_AGENT_TEST_MLX_EXECUTABLE')
        if not model or not runtime:
            self.skipTest('requires an explicitly selected cached model and MLX-LM executable')
        with TemporaryDirectory(prefix='mlx-protected-test-', dir='/tmp') as folder:
            root = Path(folder).resolve()
            key_path = root / 'access-key'
            key_path.write_text(secrets.token_urlsafe(32))
            key_path.chmod(0o600)
            key, digest = read_access_key(str(key_path))
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            control = None
            receipt = None
            environment = mock.patch.dict(os.environ, {'XDG_STATE_HOME': str(root / 'state')})
            environment.start()
            try:
                plan = plan_start(None, 'mlx_lm', load_recipes(), path=model, port=port, jit=True,
                                  api_key_file=str(key_path), receipts_dir=str(root), runtime_executable=runtime)
                result = start_serve(plan, receipts_dir=str(root), confirm=True, preview_hash=plan['preview_hash'])
                receipt = result['receipt']
                self.assertEqual(receipt['protection']['worker_transport'], PRIVATE_PROTOCOL)
                config = json.loads(Path(plan['control_config']).read_text())
                control = config['control_token']
                def request(method, path, body=None, credential=None, origin=None):
                    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
                    try:
                        headers = {'Content-Type': 'application/json'}
                        if credential is not None:
                            headers['Authorization'] = 'Bearer ' + credential
                        if origin is not None:
                            headers['Origin'] = origin
                        connection.request(method, path, json.dumps(body) if body is not None else None, headers)
                        response = connection.getresponse()
                        return response.status, response.read(65536)
                    finally:
                        connection.close()

                self.assertEqual(request('GET', '/v1/models')[0], 401)
                self.assertEqual(request('GET', '/v1/models', credential=control)[0], 401)
                self.assertEqual(request('POST', '/v1/chat/completions', {'messages': []}, key, 'https://attacker.invalid')[0], 403)
                self.assertEqual(status_serve(receipts_dir=str(root))[0]['model_state'], 'unloaded')
                status, body = request('POST', '/v1/chat/completions',
                                       {'messages': [{'role': 'user', 'content': 'Say hello.'}], 'max_tokens': 8,
                                        'stream': True, 'temperature': 0}, key)
                self.assertEqual(status, 200, body[:300])
                self.assertIn(b'data: [DONE]', body)
                self.assertIn(b'"content"', body)
                state = status_serve(receipts_dir=str(root))[0]
                self.assertEqual(state['worker_transport'], PRIVATE_PROTOCOL)
                self.assertIsNone(state['worker_port'])
                worker_pid = state['worker_pid']
                tcp = subprocess.run(['/usr/sbin/lsof', '-nP', '-a', '-p', str(worker_pid), '-iTCP', '-sTCP:LISTEN', '-Fn'], capture_output=True)
                self.assertEqual((tcp.returncode, tcp.stdout, tcp.stderr), (1, b'', b''))
                private_sockets = subprocess.run(['/usr/sbin/lsof', '-a', '-p', str(worker_pid), '-U', '-Fn'], capture_output=True, text=True).stdout
                self.assertIn('/mlx-worker-', private_sockets)
                self.assertEqual(request('POST', '/_mlx/unload', {}, key)[0], 403)
                status, _ = request('POST', '/_mlx/unload', {}, control)
                self.assertEqual(status, 200)
                self.assertEqual(status_serve(receipts_dir=str(root))[0]['model_state'], 'unloaded')
            finally:
                if receipt is not None:
                    stop_serve(port, receipts_dir=str(root), expected_pid=receipt['pid'])
                environment.stop()
