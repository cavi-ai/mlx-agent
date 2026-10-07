import concurrent.futures
import http.client
import json
import sys
import subprocess
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.jit_serve import make_server, model_fingerprint
from mlx_agent.serve import load_recipes, plan_start, start_serve, stop_serve, unload_serve, status_serve


WORKER = '''
import argparse, json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
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
            self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
            self.assertEqual(unload_serve(port, str(self.root))['status'], 'unloaded')
            self.assertEqual(self.request('GET', '/v1/models')[0], 200)
        finally:
            self.port = previous
