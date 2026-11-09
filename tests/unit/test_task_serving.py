"""Managed task endpoints bind local capabilities, jobs and residency."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlx_agent import task_serving
from mlx_agent.backends import load_manifests, with_ports
from mlx_agent.serve import load_recipes, plan_start, resolve_serve_runtime, ServeError
from mlx_agent.jit_serve import validate_memory_policy
from .backend_fixtures import synthetic_manifests, synthetic_registries


class TaskServingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text(json.dumps({'model_type': 'laya'}))
        (self.model / 'model.safetensors').write_bytes(b'fixture')
        self.manifests = synthetic_manifests()
        self.manifests["mlx-embeddings"] = load_manifests()["mlx-embeddings"]
        self.registries = synthetic_registries(self.manifests, tuple(self.manifests))
        self.registries["mlx-embeddings"]["registry"] = with_ports(self.manifests["mlx-embeddings"]["registry"], self.manifests["mlx-embeddings"])

    def test_local_classifier_resolves_to_task_endpoint(self):
        runtime, python = resolve_serve_runtime(None, str(self.model), 'auto', load_recipes(),
            manifests=self.manifests, registries=self.registries, backend_root=self.root / 'backends')
        self.assertEqual(runtime, 'tasks')
        self.assertTrue(python.endswith('/mlx-embeddings/bin/python'))

    def test_plan_binds_type_identity_and_both_residency_modes(self):
        identity = {'backend': 'mlx-embeddings', 'model_fingerprint': 'a' * 64}
        with patch('mlx_agent.task_serving.resolve_model', return_value=(['classification'], sys.executable, identity)):
            plans = [plan_start(None, 'tasks', load_recipes(), path=str(self.model), receipts_dir=self.root,
                memory_policy={'keep_loaded': keep, 'release_after_request': not keep}) for keep in (False, True)]
        self.assertNotEqual(plans[0]['preview_hash'], plans[1]['preview_hash'])
        self.assertEqual(plans[0]['task_kinds'], ['classification'])
        self.assertTrue(plans[0]['jit'])
        self.assertIn('task_serve.py', ' '.join(plans[0]['argv']))
        self.assertEqual(plans[0]['task_identity'], identity)
        self.assertEqual(plans[0]['memory_policy']['release_after_request'], True)

    def test_keep_loaded_and_release_after_request_cannot_conflict(self):
        with self.assertRaises(ValueError):
            validate_memory_policy({'keep_loaded': True, 'release_after_request': True})
        self.assertTrue(validate_memory_policy({'release_after_request': True})['release_after_request'])

    def test_unknown_and_draft_models_have_no_task_capabilities(self):
        for kind in ('unknown', 'draft'):
            (self.model / 'config.json').write_text(json.dumps({'model_type': kind}))
            with self.subTest(kind=kind), self.assertRaises(ServeError):
                task_serving.resolve_model(self.model, self.manifests, self.registries)


class TaskHTTPTests(unittest.TestCase):
    def setUp(self):
        import os
        import threading
        from mlx_agent.compute import ComputeLease
        from mlx_agent.task_identity import capture_identity
        from mlx_agent.task_serve import TaskServer
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{"model_type":"laya"}')
        (self.root / 'fixture_port.py').write_text('''
import time
class Model:
    count = 0
def load(path): return Model(), None
def predict(model, tokenizer, state, questions):
    if state == "wait": time.sleep(30)
    if state == "fail": raise RuntimeError("synthetic failure")
    model.count += 1
    return {"answers": {"calls": model.count}}
''')
        manifest = load_manifests()['mlx-embeddings']
        identity = capture_identity(self.model, manifest, Path(sys.executable))
        config = {'local_path': str(self.model), 'model': str(self.model), 'runtime': 'tasks', 'fingerprint': 'unused',
            'port': 0, 'task_identity': identity, 'task_kinds': ['classification'], 'task_state_dir': str(self.root / 'tasks'),
            'max_tokens': 0, 'worker_argv': [sys.executable], 'control_token': 'a' * 32,
            'memory_policy': {'keep_loaded': True}}
        self.server = TaskServer(('127.0.0.1', 0), config)
        self.server.controller.config['port'] = self.server.server_port
        self.server.controller.resident.environment = dict(os.environ, PYTHONPATH=str(self.root))
        self.port = self.server.server_port
        def planner(model, request):
            return {'backend': 'mlx-embeddings', 'model': model, 'request': request,
                'argv': [sys.executable, str(Path(task_serving.__file__).with_name('decide_runner.py')),
                         '--module', 'fixture_port', '--model', model, '--request', request]}
        for context in (patch('mlx_agent.task_kinds.plan_decide', side_effect=planner),
                        patch('mlx_agent.inference_runner.sync_ports'),
                        patch('mlx_agent.task_serving.ComputeLease', side_effect=lambda operation: ComputeLease(operation, self.root / 'compute'))):
            context.start()
            self.addCleanup(context.stop)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.controller.close()
        self.server.server_close()

    def request(self, method, path, value=None, headers=None, parse=True):
        import http.client
        client = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            client.request(method, path, json.dumps(value) if value is not None else None, headers or {})
            response = client.getresponse()
            raw = response.read()
            return response.status, json.loads(raw) if parse else raw
        finally:
            client.close()

    def start(self, state='test'):
        status, value = self.request('POST', '/v1/tasks', {'kind': 'classification', 'parameters': {
            'state': state, 'questions': {'route': {'type': 'choice', 'instructions': 'Choose one.'}}}})
        self.assertEqual(status, 202, value)
        return value['id']

    def wait(self, identifier):
        import time
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            status, value = self.request('GET', '/v1/tasks/' + identifier)
            self.assertEqual(status, 200, value)
            if (value['state'] in ('succeeded', 'failed', 'cancelled') and self.server.controller.active_requests == 0
                    and self.server.controller.model_state not in ('loading', 'unloading')):
                return value
            time.sleep(.02)
        self.fail('The synthetic task did not finish before its deadline.')

    def test_http_reuses_weights_then_applies_release_mode(self):
        status, inventory = self.request('GET', '/v1/models')
        self.assertEqual(status, 200)
        self.assertEqual(inventory['data'][0]['task_kinds'], ['classification'])
        for expected in (1, 2):
            identifier = self.start()
            record = self.wait(identifier)
            self.assertEqual(record['state'], 'succeeded', record.get('error'))
            status, result = self.request('GET', '/v1/tasks/' + identifier + '/result')
            self.assertEqual(status, 200, result)
            self.assertEqual(result['result']['answers']['calls'], expected)
            self.assertEqual(self.server.controller.status()['model_state'], 'loaded')
        headers = {'Authorization': 'Bearer ' + 'a' * 32}
        status, policy = self.request('POST', '/_mlx/policy', {'keep_loaded': False, 'release_after_request': True}, headers)
        self.assertEqual(status, 200, policy)
        identifier = self.start()
        self.assertEqual(self.wait(identifier)['state'], 'succeeded')
        self.assertEqual(self.server.controller.status()['model_state'], 'unloaded')
        self.assertIsNone(self.server.controller.resident.process)
        self.assertEqual(self.request('GET', '/v1/models')[0], 200)

    def test_completed_result_can_be_downloaded_and_changed_bytes_are_refused(self):
        identifier = self.start()
        self.assertEqual(self.wait(identifier)['state'], 'succeeded')
        status, result = self.request('GET', '/v1/tasks/' + identifier + '/artifacts/result')
        self.assertEqual(status, 200, result)
        self.assertEqual(result['answers']['calls'], 1)
        path = self.server.controller.store.artifact_path(identifier, 'result.json')
        path.write_text('{}')
        self.assertEqual(self.request('GET', '/v1/tasks/' + identifier + '/artifacts/result')[0], 404)

    def test_cancel_reaps_worker_and_allows_next_job(self):
        import time
        identifier = self.start('wait')
        deadline = time.monotonic() + 5
        while self.server.controller.resident.process is None and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertEqual(self.request('POST', '/v1/tasks/' + identifier + '/cancel', {})[0], 200)
        self.assertEqual(self.wait(identifier)['state'], 'cancelled')
        self.assertIsNone(self.server.controller.resident.process)
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')

    def test_host_fence_and_type_binding_refuse_other_operations(self):
        self.assertEqual(self.request('GET', '/v1/models', headers={'Host': 'evil.invalid'})[0], 403)
        self.assertEqual(self.request('POST', '/v1/tasks', {'kind': 'music_generation', 'parameters': {'caption': 'hi'}})[0], 400)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []})[0], 404)
        self.assertEqual(self.request('POST', '/_mlx/unload', {})[0], 403)

    def test_failed_job_releases_compute_for_retry(self):
        self.assertEqual(self.wait(self.start('fail'))['state'], 'failed')
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')

    def test_changed_model_refuses_reuse_and_releases_stale_weights(self):
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        (self.model / 'config.json').write_text('{"model_type":"laya","changed":true}')
        status, value = self.request('POST', '/v1/tasks', {'kind': 'classification', 'parameters': {
            'state': 'test', 'questions': {'route': {'type': 'choice', 'instructions': 'Choose.'}}}})
        self.assertEqual(status, 400, value)
        self.assertIsNone(self.server.controller.resident.process)
