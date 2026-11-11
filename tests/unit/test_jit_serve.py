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
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
p = argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--port', type=int); p.add_argument('--trace')
p.add_argument('--host', default='127.0.0.1')
p.add_argument('--adapter-path'); p.add_argument('--mutate-adapter', action='store_true')
p.add_argument('--reply-delay', type=float, default=0); p.add_argument('--keepalive', action='store_true')
p.add_argument('--stream-release')
p.add_argument('--usage', action='store_true')
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
            if a.usage:
                self.wfile.write(b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\\n\\n')
            else:
                self.wfile.write(b'data: {"choices":[]}\\n\\n')
            self.wfile.flush()
            if a.stream_release:
                deadline = time.monotonic() + 10
                while not Path(a.stream_release).exists() and time.monotonic() < deadline: time.sleep(.01)
            else: time.sleep(0.3)
            try:
                if a.usage:
                    self.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}\\n\\n')
                self.wfile.write(b'data: [DONE]\\n\\n')
            except BrokenPipeError: pass
        else:
            reply = {'model': body['model'], 'offline': os.environ.get('HF_HUB_OFFLINE')}
            if a.usage:
                reply.update(usage={'prompt_tokens':7,'completion_tokens':2}, request=body,
                             choices=[{'index':0,'message':{'role':'assistant','content':'hi'},'finish_reason':'stop'}])
            self.wfile.write(json.dumps(reply).encode())
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
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / 'local model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        (self.model / 'model.safetensors').write_bytes(b'fixture')
        self.worker = self.root / 'worker.py'
        self.worker.write_text(WORKER)
        self.trace = self.root / 'starts.txt'
        self.config = {
            'memory_state_dir': str(self.root / 'memory'),
            'compute_state_dir': str(self.root / 'compute'),
            'model': str(self.model), 'local_path': str(self.model),
            'fingerprint': model_fingerprint(self.model), 'control_token': 'test-control-token',
            'worker_argv': [sys.executable, str(self.worker), '--model', str(self.model), '--port', '{worker_port}', '--trace', str(self.trace)],
            'max_tokens': 8192,
        }
        self.server = self.start_gateway(self.config)
        self.port = self.server.server_port

    def start_gateway(self, config):
        server = make_server(config, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.controller.close)
        self.addCleanup(server.shutdown)
        return server

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            connection.request(method, path, json.dumps(body) if body is not None else None, headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_request_defaults_reach_worker_and_status_keeps_prompt_private(self):
        config = dict(self.config, request_defaults={'system_prompt': 'Be concise.', 'temperature': 0.6, 'top_p': 0.9})
        config['worker_argv'] = self.config['worker_argv'] + ['--usage']
        self.server = self.start_gateway(config)
        self.port = self.server.server_port
        status, reply = self.request('POST', '/v1/chat/completions',
            {'model': str(self.model), 'messages': [{'role': 'user', 'content': 'Hello'}], 'temperature': 0})
        self.assertEqual(status, 200)
        self.assertEqual(reply['request']['messages'][0], {'role': 'system', 'content': 'Be concise.'})
        self.assertEqual(reply['request']['temperature'], 0)
        self.assertEqual(reply['request']['top_p'], 0.9)
        status, snapshot = self.request('GET', '/_mlx/status', headers={'Authorization': 'Bearer test-control-token'})
        self.assertEqual(status, 200)
        self.assertNotIn('Be concise.', json.dumps(snapshot))
        self.assertEqual(len(snapshot['request_defaults']['system_prompt_sha256']), 64)
        self.assertEqual(snapshot['inference_telemetry']['input_tokens'], 7)
        self.assertEqual(snapshot['inference_telemetry']['output_tokens'], 2)
        self.assertEqual(self.request('GET', '/_mlx/status')[0], 403)

    def test_usage_is_counted_once_for_each_streaming_api(self):
        self.config['worker_argv'] += ['--usage']
        for path in ('/v1/chat/completions', '/v1/completions', '/v1/messages', '/v1/responses'):
            body = {'model': str(self.model), 'max_tokens': 8, 'stream': True}
            if path == '/v1/completions': body['prompt'] = 'Hello'
            elif path == '/v1/responses':
                body['input'] = 'Hello'
                body['max_output_tokens'] = body.pop('max_tokens')
            else: body['messages'] = [{'role': 'user', 'content': 'Hello'}]
            connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
            try:
                connection.request('POST', path, json.dumps(body))
                response = connection.getresponse()
                payload = response.read()
                self.assertEqual(response.status, 200, path + ': ' + payload.decode(errors='replace'))
                self.assertTrue(payload, path)
            finally:
                connection.close()
        snapshot = self.request('GET', '/_mlx/status', headers={'Authorization': 'Bearer test-control-token'})[1]['inference_telemetry']
        self.assertEqual(snapshot['completed_requests'], 4)
        self.assertEqual(snapshot['measured_requests'], 4)
        self.assertEqual((snapshot['input_tokens'], snapshot['output_tokens']), (28, 8))

    def test_request_defaults_are_reviewed_in_gateway_plan(self):
        arguments = dict(repo=None, runtime='mlx_lm', recipes=load_recipes(), path=str(self.model),
                         jit=True, receipts_dir=str(self.root), system_prompt='Be concise.', default_temperature=0.4)
        plan = plan_start(**arguments)
        self.assertEqual(plan['request_defaults'], {'system_prompt': 'Be concise.', 'temperature': 0.4})
        with self.assertRaises(ServeError): plan_start(**dict(arguments, jit=False))
        with self.assertRaises(ServeError): plan_start(**dict(arguments, runtime='tasks'))
        with self.assertRaises(ServeError): plan_start(**dict(arguments, default_top_p=2))

    def test_busy_memory_registry_does_not_terminate_the_loaded_worker(self):
        import fcntl
        from mlx_agent.jit_serve import ModelWorker
        controller = ModelWorker(self.config)
        self.addCleanup(controller.close)
        controller.memory_registry.configure_budget(64_000_000_000, 8_000_000_000)
        controller.memory_registry.lock_timeout = .02
        process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'],
                                   start_new_session=True)
        controller.process = process
        controller.model_state = 'loaded'
        with (controller.memory_registry.root / 'memory.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            controller.guard_agent_budget()
        self.assertIsNone(process.poll())
        self.assertFalse(controller.closed)
        self.assertEqual(controller.model_state, 'loaded')

    def test_memory_guard_cannot_close_a_worker_that_finished_unloading(self):
        from mlx_agent.jit_serve import ModelWorker
        controller = ModelWorker(self.config)
        self.addCleanup(controller.close)
        process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'],
                                   start_new_session=True)
        controller.process = process
        controller.model_state = 'loaded'

        def snapshot_after_unload():
            controller.unload()
            return {'agent_budget': {'status': 'unavailable'}}

        with mock.patch.object(controller.memory_registry, 'snapshot', side_effect=snapshot_after_unload):
            controller.guard_agent_budget()
        self.assertIsNotNone(process.poll())
        self.assertEqual(controller.unload()['model_state'], 'unloaded')
        self.assertFalse(controller.closed)

    def test_chat_runtime_advertises_allocator_controls_as_unavailable(self):
        control = {'Authorization': 'Bearer test-control-token'}
        code, value = self.request('POST', '/_mlx/policy', {'allocator_cache_limit_mib': 4}, headers=control)
        self.assertEqual(code, 400, value)
        self.assertEqual(value['error']['code'], 'cache_control_unavailable')
        self.assertFalse(self.server.controller.status()['allocator_cache']['supported'])
        self.assertFalse(self.trace.exists())

    def test_inference_key_is_checked_before_loading_and_control_stays_separate(self):
        import hashlib
        token = 'client-' + 'a' * 40
        self.config['api_key_sha256'] = hashlib.sha256(token.encode()).hexdigest()
        for method, path in [('GET', '/v1/models'), ('POST', '/v1/chat/completions'),
                             ('POST', '/v1/messages'), ('POST', '/v1/responses'), ('GET', '/v1/responses/unknown')]:
            with self.subTest(path=path):
                self.assertEqual(self.request(method, path, {} if method == 'POST' else None)[0], 401)
        self.assertFalse(self.trace.exists())
        self.assertEqual(self.request('GET', '/v1/models', headers={'Authorization': 'Bearer wrong'})[0], 401)
        authorized = {'Authorization': 'Bearer ' + token}
        self.assertEqual(self.request('GET', '/v1/models', headers=authorized)[0], 200)
        self.assertEqual(self.request('GET', '/_mlx/status', headers=authorized)[0], 403)
        self.assertEqual(self.request('GET', '/_mlx/status', headers={'Authorization': 'Bearer test-control-token'})[0], 200)

    def test_protected_gateway_worker_cannot_be_used_directly(self):
        import hashlib
        from mlx_agent.inference_auth import runtime_fingerprint
        token = 'client-' + 'b' * 40
        self.config['api_key_sha256'] = hashlib.sha256(token.encode()).hexdigest()
        self.config['auth_runtime_sha256'] = runtime_fingerprint()
        headers = {'Authorization': 'Bearer ' + token}
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []}, headers)[0], 200)
        worker_port = self.server.controller.worker_port
        for supplied in (None, 'wrong', token):
            client = http.client.HTTPConnection('127.0.0.1', worker_port, timeout=5)
            try:
                client.request('GET', '/v1/models', headers={} if supplied is None else {'Authorization': 'Bearer ' + supplied})
                reply = client.getresponse()
                self.assertEqual(reply.status, 401)
                reply.read()
            finally:
                client.close()

    def test_missing_authentication_runtime_identity_fails_before_worker_start(self):
        import hashlib
        token = 'client-' + 'e' * 40
        self.config['api_key_sha256'] = hashlib.sha256(token.encode()).hexdigest()
        result = self.request('POST', '/v1/chat/completions', {'messages': []}, {'Authorization': 'Bearer ' + token})
        self.assertEqual(result[0], 503)
        self.assertFalse(self.trace.exists())

    def test_protected_launch_readiness_and_receipt_do_not_require_plaintext_key(self):
        import hashlib
        import socket
        token = 'client-' + 'd' * 40
        digest = hashlib.sha256(token.encode()).hexdigest()
        recipes = load_recipes()
        recipes['mlx_lm'] = dict(recipes['mlx_lm'], executable=sys.executable,
            argv=[sys.executable, str(self.worker), '--model', '{repo}', '--port', '{port}', '--trace', str(self.trace)])
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0)); port = reservation.getsockname()[1]
        plan = plan_start(None, 'mlx_lm', recipes, path=str(self.model), port=port, jit=True,
                          receipts_dir=str(self.root), api_key_sha256=digest)
        processes = []
        def spawn(argv, log_path):
            config_path = Path(argv[argv.index('--config') + 1])
            fixture_config = json.loads(config_path.read_text())
            fixture_config.update(compute_state_dir=str(self.root / 'compute'), memory_state_dir=str(self.root / 'memory'))
            config_path.write_text(json.dumps(fixture_config))
            with open(log_path, 'ab') as output:
                process = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            return process.pid
        result = start_serve(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan['preview_hash'], spawn=spawn)
        self.addCleanup(processes[0].wait, 5)
        self.addCleanup(lambda: stop_serve(port, str(self.root), pid_alive=lambda _: processes[0].poll() is None))
        self.assertEqual(result['receipt']['api_key_sha256'], digest)
        self.assertNotIn(token, json.dumps(result))
        self.assertEqual(status_serve(str(self.root))[0]['api_key_sha256'], digest)
        previous, self.port = self.port, port
        try:
            self.assertEqual(self.request('GET', '/v1/models')[0], 401)
            self.assertEqual(self.request('GET', '/v1/models', headers={'Authorization': 'Bearer ' + token})[0], 200)
        finally:
            self.port = previous

    def unload(self):
        return self.request('POST', '/_mlx/unload', {}, {'Authorization': 'Bearer test-control-token'})

    def test_protected_gateway_rejects_unauthenticated_calls_without_loading(self):
        from mlx_agent.serving_access import read_access_key
        key = self.root / 'access-key'
        key.write_text('private-inference-key-0123456789abcdef')
        key.chmod(0o600)
        _, digest = read_access_key(str(key))
        from mlx_agent.private_serve import identity
        self.config.update(access_key_file=str(key), access_key_sha256=digest,
                           private_worker=identity())
        for method, path in [('GET', '/v1/models'), ('GET', '/v1/responses/unknown'), ('POST', '/v1/chat/completions')]:
            self.assertEqual(self.request(method, path, {'messages': []} if method == 'POST' else None)[0], 401)
        self.assertEqual(self.request('GET', '/v1/models', headers={'Authorization': 'Bearer test-control-token'})[0], 401)
        self.assertEqual(self.request('GET', '/v1/models', headers={'Authorization': 'Bearer ' + key.read_text()})[0], 200)
        self.assertEqual(self.request('GET', '/health')[0], 200)
        self.assertFalse(self.trace.exists())
        key.write_text('rotated-inference-key-0123456789abcdef')
        self.assertEqual(self.request('GET', '/v1/models')[0], 503)
        self.assertFalse(self.trace.exists())

    def configure(self, policy):
        return self.request('POST', '/_mlx/policy', policy, {'Authorization': 'Bearer test-control-token'})

    def paused_reaper(self):
        check = self.server.controller.unload_if_idle
        patcher = mock.patch.object(self.server.controller, 'unload_if_idle', return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        return check

    def load_pressure_guarded(self):
        check = self.paused_reaper()
        self.server.controller.memory_probe = lambda: 8_000_000_000
        self.assertEqual(self.configure({'minimum_headroom_gb': 2, 'unload_under_pressure': True})[0], 200)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        return check, self.server.controller.status()['worker_pid']

    def test_pressure_policy_requires_reserve_and_preserves_legacy_shape(self):
        before = self.server.controller.status()['memory_policy']
        for changes in ({'unload_under_pressure': True}, {'unload_under_pressure': 'yes'},
                        {'minimum_headroom_gb': 2, 'unload_under_pressure': True, 'keep_loaded': True}):
            self.assertEqual(self.configure(changes)[0], 400)
            self.assertEqual(self.server.controller.status()['memory_policy'], before)
        self.assertEqual(self.configure({'unload_under_pressure': False})[0], 200)
        self.assertEqual(self.server.controller.status()['memory_policy'], before)

    def test_loaded_guard_refuses_low_and_unknown_headroom_and_recovers(self):
        _, pid = self.load_pressure_guarded()
        for reading, code in ((0, 'insufficient_headroom'), (None, 'memory_unknown')):
            self.server.controller.memory_probe = lambda: reading
            status, body = self.request('POST', '/v1/chat/completions', {'messages': []})
            self.assertEqual((status, body['error']['code']), (503, code))
            state = self.server.controller.status()
            self.assertEqual((state['worker_pid'], state['active_requests']), (pid, 0))
            self.assertIsNotNone(state['request_blocked_reason'])
        # The loaded worker is not charged for its resident weights again.
        self.server.controller.memory_probe = lambda: 2_000_000_000
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        state = self.server.controller.status()
        self.assertIsNone(state['request_blocked_reason'])
        self.assertEqual(state['resident_memory_check']['minimum_available_bytes'], 2_000_000_000)

    def test_pressure_eviction_keeps_endpoint_and_reloads_after_recovery(self):
        check, first_pid = self.load_pressure_guarded()
        self.server.controller.memory_probe = lambda: 0
        self.assertTrue(check())
        state = self.server.controller.status()
        self.assertEqual((state['model_state'], state['last_unload_reason']), ('unloaded', 'memory_pressure'))
        self.assertEqual(self.request('GET', '/v1/models')[0], 200)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 503)
        self.assertEqual(self.trace.read_text().splitlines(), ['start'])
        self.server.controller.memory_probe = lambda: 8_000_000_000
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        self.assertNotEqual(self.server.controller.status()['worker_pid'], first_pid)
        self.assertEqual(self.trace.read_text().splitlines(), ['start', 'start'])

    def test_unknown_pressure_never_authorizes_worker_eviction(self):
        check, pid = self.load_pressure_guarded()
        self.server.controller.memory_probe = lambda: None
        self.assertFalse(check())
        self.assertEqual(self.server.controller.status()['worker_pid'], pid)

    def test_pressure_opt_out_does_not_probe_loaded_requests_or_reaper(self):
        check = self.paused_reaper()
        self.server.controller.memory_probe = lambda: 8_000_000_000
        self.assertEqual(self.configure({'minimum_headroom_gb': 2})[0], 200)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        self.server.controller.memory_probe = mock.Mock(side_effect=AssertionError('unexpected warm probe'))
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        self.assertFalse(check())
        self.server.controller.memory_probe.assert_not_called()

    def blocked_pressure_probe(self, check):
        entered, release = threading.Event(), threading.Event()
        identity = [None]
        def probe():
            if threading.get_ident() == identity[0]:
                entered.set()
                release.wait(5)
                return 0
            return 8_000_000_000
        self.server.controller.memory_probe = probe
        def sweep():
            identity[0] = threading.get_ident()
            return check()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.addCleanup(pool.shutdown)
        self.addCleanup(release.set)
        future = pool.submit(sweep)
        self.assertTrue(entered.wait(2))
        return pool, release, future

    def test_pressure_probe_does_not_block_policy_and_discards_changed_policy(self):
        check, pid = self.load_pressure_guarded()
        pool, release, future = self.blocked_pressure_probe(check)
        try:
            changed = pool.submit(self.configure, {'unload_under_pressure': False, 'keep_loaded': True})
            self.assertEqual(changed.result(timeout=1)[0], 200)
        finally:
            release.set()
        self.assertFalse(future.result(timeout=2))
        self.assertEqual(self.server.controller.status()['worker_pid'], pid)

    def test_pressure_probe_cannot_evict_a_replacement_worker(self):
        check, old_pid = self.load_pressure_guarded()
        _, release, future = self.blocked_pressure_probe(check)
        try:
            self.assertEqual(self.unload()[0], 200)
            self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
            new_pid = self.server.controller.status()['worker_pid']
            self.assertNotEqual(new_pid, old_pid)
        finally:
            release.set()
        self.assertFalse(future.result(timeout=2))
        self.assertEqual(self.server.controller.status()['worker_pid'], new_pid)

    def test_completed_request_invalidates_a_pending_pressure_observation(self):
        check, pid = self.load_pressure_guarded()
        _, release, future = self.blocked_pressure_probe(check)
        try:
            self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
            with self.server.controller.condition:
                self.assertTrue(self.server.controller.condition.wait_for(lambda: self.server.controller.active_requests == 0, timeout=2))
        finally:
            release.set()
        self.assertFalse(future.result(timeout=2))
        self.assertEqual(self.server.controller.status()['worker_pid'], pid)

    def test_policy_changed_back_invalidates_a_pending_pressure_observation(self):
        check, pid = self.load_pressure_guarded()
        _, release, future = self.blocked_pressure_probe(check)
        try:
            self.assertEqual(self.configure({'unload_under_pressure': False})[0], 200)
            self.assertEqual(self.configure({'unload_under_pressure': True})[0], 200)
        finally:
            release.set()
        self.assertFalse(future.result(timeout=2))
        self.assertEqual(self.server.controller.status()['worker_pid'], pid)

    def test_pressure_probe_and_new_rejections_preserve_an_active_stream(self):
        signal = self.root / 'release-stream'
        self.config['worker_argv'] += ['--stream-release', str(signal)]
        self.addCleanup(signal.touch)
        check, pid = self.load_pressure_guarded()
        _, release, future = self.blocked_pressure_probe(check)
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        self.addCleanup(connection.close)
        connection.request('POST', '/v1/chat/completions', json.dumps({'stream': True, 'messages': []}))
        response = connection.getresponse()
        self.assertEqual(response.readline(), b'data: {"choices":[]}\n')
        release.set()
        self.assertFalse(future.result(timeout=2))
        self.server.controller.memory_probe = lambda: 0
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 503)
        self.assertEqual(self.server.controller.status()['active_requests'], 1)
        self.assertEqual(self.server.controller.status()['worker_pid'], pid)
        signal.touch()
        self.assertIn(b'[DONE]', response.read())
        with self.server.controller.condition:
            self.assertTrue(self.server.controller.condition.wait_for(lambda: self.server.controller.active_requests == 0, timeout=2))
        self.assertTrue(check())

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

    def test_distinct_endpoints_share_pending_memory_admission(self):
        from mlx_agent.jit_serve import ModelWorker, GatewayError
        first=self.server.controller
        second=ModelWorker(self.config)
        self.addCleanup(second.close)
        first.memory_probe=second.memory_probe=lambda:3_000_000_000
        first._check_memory()
        with self.assertRaises(GatewayError) as caught:second._check_memory()
        self.assertEqual(caught.exception.code,'insufficient_headroom')
        self.assertGreater(second.memory_check['other_reserved_bytes'],0)
        first._release_memory()
        second._check_memory()

    def test_headroom_changed_during_cold_probe_is_registered(self):
        controller=self.server.controller
        def probe():
            controller.configure({'minimum_headroom_gb':2})
            return 8_000_000_000
        controller.memory_probe=probe
        controller._check_memory()
        self.assertEqual(controller.memory_registry.snapshot()['headroom_bytes'],2_000_000_000)

    def test_coding_reservation_waits_for_output_and_is_released_on_unload(self):
        controller=self.server.controller
        controller.acquire()
        controller.release()
        pending=controller.memory_registry.snapshot()['reserved_bytes']
        self.assertGreater(pending,controller.workspace_bytes)
        self.assertEqual(self.request('POST','/v1/chat/completions',{'messages':[]})[0],200)
        self.assertEqual(controller.memory_registry.snapshot()['reserved_bytes'],controller.workspace_bytes)
        controller.unload()
        self.assertEqual(controller.memory_registry.snapshot()['entries'],0)

    def test_configured_kv_growth_stays_reserved_after_first_output(self):
        controller=self.server.controller
        controller.configure({'minimum_headroom_gb':0,'context_tokens':2048})
        controller.memory_probe=lambda:8_000_000_000
        with mock.patch('mlx_agent.jit_serve._kv_cache_bytes',return_value=500_000_000):
            self.assertEqual(self.request('POST','/v1/chat/completions',{'messages':[]})[0],200)
        self.assertEqual(controller.memory_registry.snapshot()['reserved_bytes'],2_000_000_000)

    def test_settlement_contention_retains_coding_reservation_until_retry(self):
        from mlx_agent.residency import MemoryAdmissionError
        controller=self.server.controller
        with mock.patch.object(controller.memory_registry,'settle',side_effect=MemoryAdmissionError('memory_registry_busy','busy')):
            self.assertEqual(self.request('POST','/v1/chat/completions',{'messages':[]})[0],200)
        process=controller.process
        self.assertGreater(controller.memory_registry.snapshot()['reserved_bytes'],controller.workspace_bytes)
        self.assertIn('reservation',controller.last_error)
        self.assertEqual(self.request('POST','/v1/chat/completions',{'messages':[]})[0],200)
        self.assertIs(controller.process,process)
        self.assertEqual(controller.memory_registry.snapshot()['reserved_bytes'],controller.workspace_bytes)
        self.assertIsNone(controller.last_error)

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

    def test_client_disconnect_retains_execution_until_worker_replies(self):
        self.config['worker_argv'] += ['--reply-delay', '1.5']
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
        time.sleep(.2)
        self.assertEqual(self.server.controller.active_requests, 1)
        from mlx_agent.compute import ComputeLease, ComputeBusy
        with self.assertRaises(ComputeBusy):
            with ComputeLease('other-model', self.root / 'compute'):
                pass
        for _ in range(50):
            if self.request('GET', '/_mlx/status', headers=control)[1]['active_requests'] == 0:
                break
            time.sleep(0.1)
        self.assertEqual(self.request('GET', '/_mlx/status', headers=control)[1]['active_requests'], 0)

    def test_busy_host_does_not_load_worker_and_retry_reuses_it(self):
        from mlx_agent.compute import ComputeLease
        body = {'messages': []}
        with ComputeLease('other-model', self.root / 'compute'):
            status, result = self.request('POST', '/v1/chat/completions', body)
            self.assertEqual(status, 503)
            self.assertEqual(result['error']['code'], 'compute_busy')
            self.assertFalse(self.trace.exists())
        self.assertEqual(self.request('POST', '/v1/chat/completions', body)[0], 200)
        pid = self.server.controller.process.pid
        with ComputeLease('other-model', self.root / 'compute'):
            self.assertEqual(self.request('POST', '/v1/chat/completions', body)[0], 503)
        self.assertEqual(self.request('POST', '/v1/chat/completions', body)[0], 200)
        self.assertEqual(self.server.controller.process.pid, pid)
        self.assertEqual(self.trace.read_text().splitlines(), ['start'])

    def test_waiting_model_starts_after_execution_and_same_backend_batches(self):
        from mlx_agent.jit_serve import ModelWorker
        from mlx_agent.compute import ComputeLease, ComputeBusy
        self.config['worker_argv'] += ['--reply-delay', '.6']
        second_trace = self.root / 'second.txt'
        second_config = dict(self.config, worker_argv=[sys.executable, str(self.worker), '--model', str(self.model),
            '--port', '{worker_port}', '--trace', str(second_trace)], memory_policy={'execution_wait_seconds': 3})
        second = ModelWorker(second_config)
        self.addCleanup(second.close)
        # This case exercises execution ordering, independently of host headroom.
        self.server.controller.memory_probe = second.memory_probe = lambda: 8_000_000_000
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(self.request, 'POST', '/v1/chat/completions', {'messages': []})
            deadline = time.monotonic() + 5
            while self.server.controller.execution.users != 1 and time.monotonic() < deadline:
                time.sleep(.01)
            batch = pool.submit(self.request, 'POST', '/v1/chat/completions', {'messages': []})
            waiting = pool.submit(second.acquire)
            deadline = time.monotonic() + .4
            while self.server.controller.execution.users != 2 and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(self.server.controller.execution.users, 2)
            self.assertFalse(second_trace.exists())
            with self.assertRaises(ComputeBusy):
                with ComputeLease('third-model', self.root / 'compute'):
                    pass
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(batch.result()[0], 200)
            self.assertIsInstance(waiting.result(), int)
        second.release()
        self.assertEqual(self.trace.read_text().splitlines(), ['start'])
        self.assertEqual(second_trace.read_text().splitlines(), ['start'])
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
        self.assertEqual(self.trace.read_text().splitlines(), ['start'])

    def test_disconnected_waiter_starts_no_worker(self):
        from mlx_agent.compute import ComputeLease
        self.server.controller.configure({'execution_wait_seconds': 2})
        with ComputeLease('other-model', self.root / 'compute'):
            connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
            connection.request('POST', '/v1/chat/completions', json.dumps({'messages': []}))
            deadline = time.monotonic() + 1
            while not self.server.controller.execution.waiting and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(self.server.controller.execution.waiting, 1)
            connection.close()
            deadline = time.monotonic() + 1
            while self.server.controller.active_requests and time.monotonic() < deadline:
                time.sleep(.01)
        self.assertEqual(self.server.controller.active_requests, 0)
        self.assertFalse(self.trace.exists())

    def test_failed_disconnect_drain_stops_backend_before_compute_release(self):
        from mlx_agent.compute import ComputeLease
        self.config['worker_argv'] += ['--reply-delay', '10']
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        upstreams = []
        connect = self.server.controller.connection
        def tracked_connection(*args, **kwargs):
            upstream = connect(*args, **kwargs)
            upstream.getresponse = mock.Mock(wraps=upstream.getresponse)
            upstreams.append(upstream)
            return upstream
        with mock.patch('mlx_agent.jit_serve.DRAIN_TIMEOUT_SECONDS', .1), \
                mock.patch.object(self.server.controller, 'connection', side_effect=tracked_connection):
            connection.request('POST', '/v1/chat/completions', json.dumps({'messages': []}))
            deadline = time.monotonic() + 3
            while self.server.controller.model_state != 'loaded' and time.monotonic() < deadline:
                time.sleep(.01)
            process = self.server.controller.process
            self.assertIsNotNone(process)
            connection.close()
            deadline = time.monotonic() + 3
            while self.server.controller.active_requests and time.monotonic() < deadline:
                time.sleep(.01)
        self.assertEqual(self.server.controller.active_requests, 0)
        self.assertIsNotNone(process.poll())
        self.assertIsNone(self.server.controller.process)
        self.assertEqual(len(upstreams), 1)
        self.assertEqual(upstreams[0].getresponse.call_count, 1)
        with ComputeLease('other-model', self.root / 'compute'):
            pass

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
            config_path = Path(argv[argv.index('--config') + 1])
            fixture_config = json.loads(config_path.read_text())
            fixture_config.update(compute_state_dir=str(self.root / 'compute'), memory_state_dir=str(self.root / 'memory'))
            config_path.write_text(json.dumps(fixture_config))
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
        deadline = time.monotonic() + 2
        while status_serve(str(self.root))[0]['active_requests'] and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(status_serve(str(self.root))[0]['active_requests'], 0)
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
            config_path = Path(argv[argv.index('--config') + 1])
            fixture_config = json.loads(config_path.read_text())
            fixture_config.update(compute_state_dir=str(self.root / 'compute'), memory_state_dir=str(self.root / 'memory'))
            config_path.write_text(json.dumps(fixture_config))
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
            pressure = dict(policy, keep_loaded=False, minimum_headroom_gb=0, unload_under_pressure=True)
            self.assertEqual(configure_serve_memory(port, pressure, str(self.root), expected_pid=pid)['memory_policy'], pressure)
            self.assertEqual(configure_serve_memory(port, policy, str(self.root), expected_pid=pid)['memory_policy'], policy)
            with_context = dict(policy, minimum_headroom_gb=0, context_tokens=8192)
            self.assertEqual(configure_serve_memory(port, with_context, str(self.root), expected_pid=pid)['memory_policy'], with_context)
            self.assertEqual(configure_serve_memory(port, policy, str(self.root), expected_pid=pid)['memory_policy'], policy)
            self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 200)
            from mlx_agent.serve import _owned_jit_worker, receipts_root
            worker_pid = status_serve(str(self.root))[0]['worker_pid']
            self.assertEqual(_owned_jit_worker(outcome['receipt'], receipts_root(str(self.root))), worker_pid)
            with mock.patch('mlx_agent.serve._pid_start', return_value='reused-process'):
                self.assertIsNone(_owned_jit_worker(outcome['receipt'], receipts_root(str(self.root))))
            self.assertEqual(unload_serve(port, str(self.root))['status'], 'unloaded')
            self.assertEqual(self.request('GET', '/v1/models')[0], 200)
        finally:
            self.port = previous
