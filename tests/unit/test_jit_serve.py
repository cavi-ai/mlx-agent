import concurrent.futures
import http.client
import json
import sys
import subprocess
import threading
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.jit_serve import make_server, model_fingerprint, available_memory_bytes
from mlx_agent.serve import load_recipes, plan_start, start_serve, stop_serve, unload_serve, status_serve, configure_serve_memory, ServeError


WORKER = '''
import argparse, json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
p = argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--port', type=int); p.add_argument('--trace')
p.add_argument('--host', default='127.0.0.1')
a = p.parse_args()
with open(a.trace, 'a') as f: f.write('start\\n')
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200); self.end_headers()
        if body.get('stream'):
            self.wfile.write(b'data: {"choices":[]}\\n\\n'); self.wfile.flush(); time.sleep(0.3)
            try: self.wfile.write(b'data: [DONE]\\n\\n')
            except BrokenPipeError: pass
        else:
            self.wfile.write(json.dumps({'model': body['model'], 'offline': os.environ.get('HF_HUB_OFFLINE')}).encode())
class Server(ThreadingHTTPServer):
    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
Server(('127.0.0.1', a.port), Handler).serve_forever()
'''


class JITServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'local model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        (self.model / 'model.safetensors').write_bytes(b'fixture')
        self.worker = self.root / 'worker.py'
        self.worker.write_text(WORKER)
        self.trace = self.root / 'starts.txt'
        self.config = {
            'model': str(self.model), 'local_path': str(self.model),
            'fingerprint': model_fingerprint(self.model), 'control_token': 'test-control-token',
            'worker_argv': [sys.executable, str(self.worker), '--model', str(self.model), '--port', '{worker_port}', '--trace', str(self.trace)],
            'max_tokens': 8192,
        }
        self.server = make_server(self.config, port=0)
        self.port = self.server.server_port
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.controller.close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            connection.request(method, path, json.dumps(body) if body is not None else None, headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def unload(self):
        return self.request('POST', '/_mlx/unload', {}, {'Authorization': 'Bearer test-control-token'})

    def configure(self, policy):
        return self.request('POST', '/_mlx/policy', policy, {'Authorization': 'Bearer test-control-token'})

    def test_policy_persistence_is_private_and_save_failure_preserves_live_policy(self):
        path = self.root / 'jit-config.json'
        self.config['config_path'] = str(path)
        self.assertEqual(self.configure({'idle_timeout_seconds': 60})[0], 200)
        saved = json.loads(path.read_text())
        self.assertEqual(saved['memory_policy']['idle_timeout_seconds'], 60)
        self.assertEqual(saved['control_token'], 'test-control-token')
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        target = self.root / 'target.json'
        path.rename(target)
        path.symlink_to(target)
        self.assertEqual(self.configure({'keep_loaded': True})[0], 503)
        self.assertFalse(self.server.controller.status()['memory_policy']['keep_loaded'])
        self.assertEqual(json.loads(target.read_text()), saved)

    def test_unknown_weight_size_blocks_configured_load(self):
        (self.model / 'model.safetensors').unlink()
        self.config['fingerprint'] = model_fingerprint(self.model)
        self.server.controller.memory_probe = lambda: 8_000_000_000
        self.assertEqual(self.configure({'minimum_headroom_gb': 2})[0], 200)
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(status, 503)
        self.assertEqual(body['error']['code'], 'memory_unknown')
        self.assertFalse(self.trace.exists())

    def test_os_headroom_probe_preserves_unknown_and_counts_free_plus_inactive(self):
        output = 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\nPages speculative: 99.\n'
        with mock.patch('mlx_agent.jit_serve.sys.platform', 'darwin'), mock.patch('mlx_agent.jit_serve.subprocess.run') as run:
            run.return_value.stdout = output
            self.assertEqual(available_memory_bytes(), 30 * 16384)
            self.assertEqual(run.call_args.kwargs['timeout'], 2)
            run.side_effect = subprocess.TimeoutExpired('vm_stat', 2)
            self.assertIsNone(available_memory_bytes())
        with mock.patch('mlx_agent.jit_serve.sys.platform', 'linux'), mock.patch('mlx_agent.jit_serve.Path.read_text', return_value='MemAvailable: 123 kB\n'):
            self.assertEqual(available_memory_bytes(), 123 * 1024)

    def test_idle_unload_keeps_endpoint_and_reloads_on_next_request(self):
        now = [100.0]
        self.server.controller.clock = lambda: now[0]
        self.assertEqual(self.configure({'idle_timeout_seconds': 10})[0], 200)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        now[0] = 111.0
        self.assertTrue(self.server.controller.unload_if_idle())
        self.assertEqual(self.request('GET', '/v1/models')[0], 200)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        self.assertEqual(self.trace.read_text().splitlines(), ['start', 'start'])

    def test_keep_loaded_prevents_idle_unload_but_allows_manual_unload(self):
        now = [100.0]
        self.server.controller.clock = lambda: now[0]
        self.assertEqual(self.configure({'idle_timeout_seconds': 1, 'keep_loaded': True})[0], 200)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        now[0] = 200.0
        self.assertFalse(self.server.controller.unload_if_idle())
        self.assertEqual(self.unload()[0], 200)

    def test_stream_holds_idle_lease_and_resets_deadline_when_finished(self):
        now = [100.0]
        self.server.controller.clock = lambda: now[0]
        self.assertEqual(self.configure({'idle_timeout_seconds': 1})[0], 200)
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        self.addCleanup(connection.close)
        connection.request('POST', '/v1/chat/completions', json.dumps({'stream': True, 'messages': []}))
        response = connection.getresponse()
        self.assertEqual(response.readline(), b'data: {"choices":[]}\n')
        now[0] = 200.0
        self.assertFalse(self.server.controller.unload_if_idle())
        self.assertIn(b'[DONE]', response.read())
        self.assertFalse(self.server.controller.unload_if_idle())
        now[0] = 202.0
        self.assertTrue(self.server.controller.unload_if_idle())

    def test_low_or_unknown_memory_blocks_cold_load_and_can_recover(self):
        self.server.controller.memory_probe = lambda: 0
        self.assertEqual(self.configure({'minimum_headroom_gb': 2, 'keep_loaded': True})[0], 200)
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(status, 503)
        self.assertEqual(body['error']['code'], 'insufficient_headroom')
        self.assertFalse(self.trace.exists())
        reading = self.server.controller.status()['memory_check']
        self.assertEqual(reading['available_bytes'], 0)
        self.assertGreater(reading['required_available_bytes'], 2_000_000_000)
        self.server.controller.memory_probe = lambda: None
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(body['error']['code'], 'memory_unknown')
        self.assertFalse(self.trace.exists())
        self.server.controller.memory_probe = lambda: 8_000_000_000
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)

    def test_policy_update_requires_auth_and_rejects_invalid_settings(self):
        self.assertEqual(self.request('POST', '/_mlx/policy', {'keep_loaded': True})[0], 403)
        self.assertEqual(self.configure({'idle_timeout_seconds': 10})[0], 200)
        before = self.server.controller.status()['memory_policy']
        for bad in ({'idle_timeout_seconds': -1}, {'idle_timeout_seconds': True},
                    {'minimum_headroom_gb': float('nan')}, {'keep_loaded': 'yes'}, {'unexpected': 1}):
            self.assertEqual(self.configure(bad)[0], 400)
            self.assertEqual(self.server.controller.status()['memory_policy'], before)

    def test_gateway_does_not_resolve_loopback_hostname(self):
        with mock.patch('socket.getfqdn', side_effect=AssertionError('Loopback startup must not query DNS')):
            server = make_server(self.config, port=0)
        self.addCleanup(server.server_close)
        self.assertEqual(server.server_name, '127.0.0.1')
        self.assertEqual(server.server_port, server.socket.getsockname()[1])

    def test_endpoint_reachable_without_weights_then_reload_same_local_files(self):
        self.assertEqual(self.request('GET', '/v1/models')[0], 200)
        self.assertFalse(self.trace.exists())
        self.assertEqual(self.server.controller.status()['model_state'], 'unloaded')
        status, body = self.request('POST', '/v1/chat/completions', {'model': str(self.model), 'messages': []})
        self.assertEqual(status, 200)
        self.assertEqual(body, {'model': str(self.model), 'offline': '1'})
        self.assertEqual(self.unload()[0], 200)
        self.assertEqual(self.server.controller.status()['model_state'], 'unloaded')
        self.assertEqual(self.request('GET', '/v1/models')[0], 200)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        self.assertEqual(self.trace.read_text().splitlines(), ['start', 'start'])

    def test_concurrent_cold_requests_launch_one_worker(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda _: self.request('POST', '/v1/chat/completions', {'messages': []}), range(4)))
        self.assertTrue(all(status == 200 for status, _ in responses))
        self.assertEqual(self.trace.read_text().splitlines(), ['start'])
        self.assertEqual(self.server.controller.status()['active_requests'], 0)

    def test_live_stream_prevents_unload(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        self.addCleanup(connection.close)
        connection.request('POST', '/v1/chat/completions', json.dumps({'stream': True, 'messages': []}))
        response = connection.getresponse()
        self.assertEqual(response.readline(), b'data: {"choices":[]}\n')
        self.assertEqual(self.unload()[0], 409)
        self.assertIn(b'[DONE]', response.read())
        self.assertEqual(self.unload()[0], 200)

    def test_controls_require_token_and_untrusted_model_requests_never_load(self):
        self.assertEqual(self.request('POST', '/_mlx/unload', {})[0], 403)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'model': 'other/download'})[0], 400)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'draft_model': 'other/download'})[0], 400)
        self.assertEqual(self.request('GET', '/v1/models', headers={'Origin': 'https://evil.example'})[0], 403)
        self.assertFalse(self.trace.exists())

    def test_changed_model_is_refused_before_loading(self):
        (self.model / 'config.json').write_text('{"changed": true}')
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(status, 503)
        self.assertEqual(body['error']['code'], 'model_changed')
        self.assertFalse(self.trace.exists())

    def test_confirmed_engine_gateway_unloads_without_stopping_endpoint(self):
        recipes = load_recipes()
        recipes['mlx_lm'] = dict(recipes['mlx_lm'], executable=sys.executable,
            argv=[sys.executable, str(self.worker), '--model', '{repo}', '--port', '{port}', '--trace', str(self.trace)])
        import socket
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        plan = plan_start(None, 'mlx_lm', recipes, path=str(self.model), port=port, jit=True, receipts_dir=str(self.root))
        processes = []
        def spawn(argv, log_path):
            process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            processes.append(process)
            return process.pid
        outcome = start_serve(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan['preview_hash'], spawn=spawn)
        self.addCleanup(lambda: stop_serve(port, str(self.root), pid_alive=lambda _: processes[0].poll() is None))
        self.assertTrue(outcome['receipt']['jit'])
        self.assertEqual(status_serve(str(self.root))[0]['model_state'], 'unloaded')
        previous = self.port
        self.port = port
        try:
            pid = outcome['receipt']['pid']
            policy = {'idle_timeout_seconds': 60, 'keep_loaded': True, 'minimum_headroom_gb': None}
            with self.assertRaises(ServeError):
                configure_serve_memory(port, policy, str(self.root), expected_pid=pid + 1)
            self.assertEqual(configure_serve_memory(port, policy, str(self.root), expected_pid=pid)['memory_policy'], policy)
            self.assertEqual(status_serve(str(self.root))[0]['memory_policy'], policy)
            self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
            self.assertEqual(unload_serve(port, str(self.root))['status'], 'unloaded')
            self.assertEqual(self.request('GET', '/v1/models')[0], 200)
        finally:
            self.port = previous
