import concurrent.futures
import http.client
import json
import sys
import subprocess
import threading
import time
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.jit_serve import make_server, model_fingerprint, available_memory_bytes
from mlx_agent.lora import adapter_identity
from mlx_agent.serve import load_recipes, plan_start, start_serve, stop_serve, unload_serve, status_serve, configure_serve_memory, ServeError


WORKER = '''
import argparse, json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
p = argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--port', type=int); p.add_argument('--trace')
p.add_argument('--host', default='127.0.0.1')
p.add_argument('--adapter-path'); p.add_argument('--mutate-adapter', action='store_true')
p.add_argument('--reply-delay', type=float, default=0); p.add_argument('--keepalive', action='store_true')
a = p.parse_args()
with open(a.trace, 'a') as f: f.write('start\\n')
if a.mutate_adapter:
    from pathlib import Path
    (Path(a.adapter_path) / 'adapter_config.json').write_text('{"changed":true}')
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        # Like MLX-LM, write nothing until the reply starts.
        time.sleep(a.reply_delay)
        self.send_response(200); self.end_headers()
        if body.get('stream') and a.keepalive:
            # MLX-LM reports prompt progress as SSE comments before the first token.
            self.wfile.write(b': keepalive 1/1\\n\\n'); self.wfile.flush()
            self.wfile.write(b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\\n\\n')
            self.wfile.write(b'data: [DONE]\\n\\n')
        elif body.get('stream'):
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

    def stream(self, path, body):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            connection.request('POST', path, json.dumps(body))
            response = connection.getresponse()
            return response.status, response.read().decode()
        finally:
            connection.close()

    def test_client_disconnect_releases_lease_before_worker_replies(self):
        self.config['worker_argv'] += ['--reply-delay', '30']
        control = {'Authorization': 'Bearer test-control-token'}
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        connection.request('POST', '/v1/chat/completions', json.dumps({'messages': []}))
        for _ in range(100):
            status = self.request('GET', '/_mlx/status', headers=control)[1]
            if status['model_state'] == 'loaded' and status['active_requests'] == 1:
                break
            time.sleep(0.1)
        self.assertEqual(status['active_requests'], 1)
        connection.close()
        for _ in range(50):
            if self.request('GET', '/_mlx/status', headers=control)[1]['active_requests'] == 0:
                break
            time.sleep(0.1)
        self.assertEqual(self.request('GET', '/_mlx/status', headers=control)[1]['active_requests'], 0)

    def test_messages_stream_forwards_worker_keepalive_as_ping(self):
        self.config['worker_argv'] += ['--keepalive']
        status, text = self.stream('/v1/messages', {'model': 'default_model', 'max_tokens': 8, 'stream': True,
                                                    'messages': [{'role': 'user', 'content': 'x'}]})
        self.assertEqual(status, 200)
        events = [line[len('event: '):] for line in text.splitlines() if line.startswith('event: ')]
        self.assertEqual(events[:3], ['message_start', 'ping', 'content_block_start'])
        self.assertIn('event: ping\ndata: {"type": "ping"}\n\n', text)

    def test_responses_stream_forwards_worker_keepalive_as_comment(self):
        self.config['worker_argv'] += ['--keepalive']
        status, text = self.stream('/v1/responses', {'model': 'default_model', 'input': 'x', 'stream': True})
        self.assertEqual(status, 200)
        lines = text.splitlines()
        first_delta = lines.index('event: response.output_text.delta')
        self.assertIn(': keepalive', lines[:first_delta])
        numbers = [json.loads(line[len('data: '):])['sequence_number'] for line in lines if line.startswith('data: ')]
        self.assertEqual(numbers, list(range(len(numbers))))

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

    def use_adapter(self):
        adapter = self.root / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text('{}')
        header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
        (adapter / 'adapters.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header + b'1234')
        self.config.update(adapter_path=str(adapter), adapter_identity=adapter_identity(adapter))
        self.config['worker_argv'] += ['--adapter-path', str(adapter)]
        return adapter

    def test_adapter_changes_after_unload_refuse_another_worker(self):
        adapter = self.use_adapter()
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        self.assertEqual(self.unload()[0], 200)
        tensor = adapter / 'adapters.safetensors'
        tensor.write_bytes(tensor.read_bytes()[:-4] + b'5678')
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(status, 503)
        self.assertEqual(body['error']['code'], 'adapter_changed')
        self.assertEqual(self.trace.read_text().splitlines(), ['start'])
        self.assertEqual(self.server.controller.status()['active_requests'], 0)

    def test_adapter_changes_during_load_stop_the_worker(self):
        self.use_adapter()
        self.config['worker_argv'] += ['--mutate-adapter']
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(status, 503)
        self.assertEqual(body['error']['code'], 'adapter_changed')
        self.assertEqual(self.server.controller.status()['model_state'], 'failed')
        self.assertIsNone(self.server.controller.process)
        self.assertEqual(self.server.controller.status()['active_requests'], 0)

    def test_legacy_adapter_without_content_identity_requires_a_fresh_preview(self):
        self.use_adapter()
        del self.config['adapter_identity']
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(status, 503)
        self.assertEqual(body['error']['code'], 'adapter_changed')
        self.assertFalse(self.trace.exists())

    def test_training_receipt_changes_after_unload_refuse_another_worker(self):
        adapter = self.use_adapter()
        receipt = self.root / 'training.json'
        record = {'kind': 'lora', 'exit_status': 'done', 'pid': 9, 'argv': [], 'repo': 'pub/model',
            'run_id': 'run-a', 'out': str(adapter), 'adapter_identity': self.config['adapter_identity'],
            'base_identity': {'path': str(self.model), 'fingerprint': self.config['fingerprint']}}
        receipt.write_text(json.dumps(record))
        plan = plan_start(None, 'mlx_lm', load_recipes(), training_receipt=str(receipt))
        self.config['training_binding'] = plan['training_binding']
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        self.assertEqual(self.unload()[0], 200)
        receipt.write_text(json.dumps(dict(record, run_id='changed')))
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(status, 503)
        self.assertEqual(body['error']['code'], 'training_receipt_changed')
        self.assertEqual(self.trace.read_text().splitlines(), ['start'])

    def test_memory_admission_includes_adapter_tensor_bytes(self):
        import math
        self.use_adapter()
        base_bytes = (self.model / 'model.safetensors').stat().st_size
        adapter_bytes = self.config['adapter_identity']['adapters.safetensors']['bytes']
        required = math.ceil((base_bytes + adapter_bytes) * 1.10 + 1.5e9)
        self.server.controller.memory_probe = lambda: required - 1
        self.assertEqual(self.configure({'minimum_headroom_gb': 0})[0], 200)
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(status, 503)
        self.assertEqual(body['error']['code'], 'insufficient_headroom')
        self.assertEqual(self.server.controller.status()['memory_check']['required_available_bytes'], required)
        self.assertFalse(self.trace.exists())

    def write_architecture(self, nested=False):
        # 2 layers x 2 KV heads x head_dim 8 x key and value x 2 bytes = 128 bytes per token.
        architecture = {'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 8}
        (self.model / 'config.json').write_text(json.dumps({'text_config': architecture} if nested else architecture))
        self.config['fingerprint'] = model_fingerprint(self.model)

    def test_memory_admission_includes_kv_cache_at_the_policy_context(self):
        import math
        self.write_architecture()
        weights = (self.model / 'model.safetensors').stat().st_size
        required = math.ceil(weights * 1.10 + 128 * 32768 + 1.5e9)
        self.server.controller.memory_probe = lambda: required - 1
        self.assertEqual(self.configure({'minimum_headroom_gb': 0, 'context_tokens': 32768})[0], 200)
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(status, 503)
        self.assertEqual(body['error']['code'], 'insufficient_headroom')
        self.assertEqual(self.server.controller.status()['memory_check']['required_available_bytes'], required)
        self.assertFalse(self.trace.exists())
        self.server.controller.memory_probe = lambda: required
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)

    def test_memory_admission_reads_text_config_and_refuses_unknown_kv_size(self):
        import math
        self.server.controller.memory_probe = lambda: 1 << 40
        self.assertEqual(self.configure({'minimum_headroom_gb': 0, 'context_tokens': 4096})[0], 200)
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(status, 503)
        self.assertEqual(body['error']['code'], 'memory_unknown')
        self.assertFalse(self.trace.exists())
        self.write_architecture(nested=True)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        weights = (self.model / 'model.safetensors').stat().st_size
        self.assertEqual(self.server.controller.status()['memory_check']['required_available_bytes'],
                         math.ceil(weights * 1.10 + 128 * 4096 + 1.5e9))

    def test_policy_context_is_optional_validated_and_cleared(self):
        self.assertNotIn('context_tokens', self.server.controller.status()['memory_policy'])
        self.assertEqual(self.configure({'context_tokens': 32768})[0], 400)
        self.assertEqual(self.configure({'minimum_headroom_gb': 2, 'context_tokens': 32768})[0], 200)
        self.assertEqual(self.server.controller.status()['memory_policy']['context_tokens'], 32768)
        for bad in ({'context_tokens': True}, {'context_tokens': 1023}, {'context_tokens': 1048577},
                    {'context_tokens': 4096.0}, {'context_tokens': '4096'}, {'minimum_headroom_gb': None}):
            self.assertEqual(self.configure(bad)[0], 400)
        self.assertEqual(self.server.controller.status()['memory_policy']['context_tokens'], 32768)
        self.assertEqual(self.configure({'context_tokens': None})[0], 200)
        self.assertNotIn('context_tokens', self.server.controller.status()['memory_policy'])

    def test_readiness_timeout_names_the_configured_deadline(self):
        self.config['worker_readiness_seconds'] = 2
        self.config['worker_argv'] = [sys.executable, '-c', 'import time; time.sleep(30)']
        status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
        self.assertEqual(status, 503)
        self.assertEqual(body['error'], {'code': 'model_load_failed',
                                         'message': 'Model worker did not become ready in 2 seconds.'})

    def test_bench_cold_workload_measures_one_load_of_an_unloaded_gateway(self):
        import io
        import socket
        from contextlib import redirect_stdout
        from mlx_agent import cli
        recipes = load_recipes()
        recipes['mlx_lm'] = dict(recipes['mlx_lm'], executable=sys.executable,
            argv=[sys.executable, str(self.worker), '--model', '{repo}', '--port', '{port}', '--trace', str(self.trace), '--keepalive'])
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        plan = plan_start(None, 'mlx_lm', recipes, path=str(self.model), port=port, jit=True, receipts_dir=str(self.root))
        processes = []
        def spawn(argv, log_path):
            process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            processes.append(process)
            return process.pid
        start_serve(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan['preview_hash'], spawn=spawn)
        self.addCleanup(lambda: stop_serve(port, str(self.root), pid_alive=lambda _: processes[0].poll() is None))

        def get(target_port):
            connection = http.client.HTTPConnection('127.0.0.1', target_port, timeout=10)
            try:
                connection.request('GET', '/v1/models')
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()
        model = get(port)[1]['data'][0]['id']

        def bench():
            output = io.StringIO()
            with redirect_stdout(output), mock.patch('mlx_agent.host.HostInventory.inspect', side_effect=RuntimeError('no host')):
                code = cli.main(['bench', 'run', '--repo', model, '--runtime', 'mlx_lm', '--base-url',
                                 'http://127.0.0.1:{}'.format(port), '--workload', 'cold', '--runs', '1',
                                 '--receipts-dir', str(self.root), '--json'])
            return code, json.loads(output.getvalue())

        code, result = bench()
        self.assertEqual(code, 0, result)
        self.assertEqual(self.trace.read_text().splitlines(), ['start'])
        self.assertIsNotNone(result['data']['measurement']['ttft_ms'])
        loaded = status_serve(str(self.root))[0]
        self.assertEqual(loaded['model_state'], 'loaded')
        self.assertEqual(get(loaded['worker_port'])[0], 200)
        code, result = bench()
        self.assertEqual((code, result['error']['code']), (2, 'model_loaded'))
        self.assertEqual(self.trace.read_text().splitlines(), ['start'])
        self.assertEqual(unload_serve(port, str(self.root))['status'], 'unloaded')
        self.assertIsNone(status_serve(str(self.root))[0]['worker_port'])
        self.assertEqual(bench()[0], 0)
        self.assertEqual(self.trace.read_text().splitlines(), ['start', 'start'])

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
            with_context = dict(policy, minimum_headroom_gb=0, context_tokens=8192)
            self.assertEqual(configure_serve_memory(port, with_context, str(self.root), expected_pid=pid)['memory_policy'], with_context)
            self.assertEqual(configure_serve_memory(port, policy, str(self.root), expected_pid=pid)['memory_policy'], policy)
            self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
            self.assertEqual(unload_serve(port, str(self.root))['status'], 'unloaded')
            self.assertEqual(self.request('GET', '/v1/models')[0], 200)
        finally:
            self.port = previous
