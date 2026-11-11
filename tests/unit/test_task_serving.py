"""Managed task endpoints bind local capabilities, jobs and residency."""
import json
from pathlib import Path
import sys
import tempfile
import time
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

    def test_task_memory_budget_is_part_of_the_reviewed_plan(self):
        identity = {'backend': 'mlx-embeddings', 'model_fingerprint': 'a' * 64}
        with patch('mlx_agent.task_serving.resolve_model', return_value=(['classification'], sys.executable, identity)):
            plan = plan_start(None, 'tasks', load_recipes(), path=str(self.model), receipts_dir=self.root,
                memory_policy={'model_memory_budget_gb': 12})
        self.assertEqual(plan['memory_policy']['model_memory_budget_gb'], 12)

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
    def test_authentication_covers_jobs_results_artifacts_and_cancellation(self):
        import hashlib
        token = 'client-' + 'a' * 40
        self.server.config['api_key_sha256'] = hashlib.sha256(token.encode()).hexdigest()
        for method, path in [('GET', '/v1/models'), ('POST', '/v1/tasks'), ('GET', '/v1/tasks/unknown'),
                             ('GET', '/v1/tasks/unknown/result'), ('GET', '/v1/tasks/unknown/artifacts/output'),
                             ('POST', '/v1/tasks/unknown/cancel')]:
            with self.subTest(path=path):
                self.assertEqual(self.request(method, path, {} if method == 'POST' else None)[0], 401)
        self.assertEqual(self.request('GET', '/v1/models', headers={'Authorization': 'Bearer ' + token})[0], 200)
        self.assertIsNone(self.server.controller.resident.process)

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
        (self.model / 'model.safetensors').write_bytes(b'fixture')
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
            'memory_state_dir': str(self.root / 'memory'),
            'compute_state_dir': str(self.root / 'compute'),
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
                        patch('mlx_agent.task_serving.ComputeLease', side_effect=lambda operation, root=None, **options: ComputeLease(operation, self.root / 'compute', **options))):
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

    def test_measured_job_breach_reports_cap_error_and_releases_only_its_worker(self):
        from mlx_agent.memory_budget import process_snapshot
        controller = self.server.controller
        controller.memory_registry.configure_budget(8_000_000_000, 1_000_000_000)
        controller.memory_probe = lambda: 100_000_000_000
        controller.configure({'model_memory_budget_gb': 2})
        mlx = self.root / 'mlx'
        mlx.mkdir()
        (mlx / '__init__.py').write_text('')
        (mlx / 'core.py').write_text('def set_memory_limit(limit): pass\n')
        identifier = self.start('wait')
        deadline = time.monotonic() + 5
        while controller.store.get(identifier)['state'] == 'starting' and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(controller.store.get(identifier)['state'], 'running')
        worker = controller.resident.process
        def probe():
            rows = process_snapshot()
            if worker.pid in rows:
                rows[worker.pid]['rss'] = 2_000_000_001
            return rows
        controller.memory_registry.process_probe = probe
        controller.guard_agent_budget()
        record = self.wait(identifier)
        self.assertEqual(record['state'], 'failed')
        self.assertEqual(record['error']['code'], 'job_memory_cap_exceeded')
        self.assertIsNotNone(worker.poll())
        self.assertEqual(controller.memory_registry.snapshot()['entries'], 0)
        self.assertEqual(list((self.root / 'compute').glob('execution-*.json')), [])

    def test_guard_error_does_not_replace_a_later_execution_failure(self):
        from mlx_agent.memory_budget import process_snapshot
        controller = self.server.controller
        controller.memory_registry.configure_budget(8_000_000_000, 1_000_000_000)
        controller.memory_probe = lambda: 100_000_000_000
        controller.configure({'model_memory_budget_gb': 2})
        mlx = self.root / 'mlx'
        mlx.mkdir()
        (mlx / '__init__.py').write_text('')
        (mlx / 'core.py').write_text('def set_memory_limit(limit): pass\n')
        identifier = self.start('wait')
        deadline = time.monotonic() + 5
        while controller.store.get(identifier)['state'] == 'starting' and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(controller.store.get(identifier)['state'], 'running')
        worker = controller.resident.process
        def probe():
            rows = process_snapshot()
            if worker.pid in rows:
                rows[worker.pid]['rss'] = 2_000_000_001
            return rows
        controller.memory_registry.process_probe = probe
        controller.guard_agent_budget()
        stopped = self.wait(identifier)
        self.assertEqual(stopped['error']['code'], 'job_memory_cap_exceeded')
        # A later persisted job encounters the stopped endpoint, not the old breach.
        request = controller.store.read_request(identifier)
        later = controller.store.create(request, controller.config['task_identity'])
        controller._execute(later['id'])
        failed = controller.store.get(later['id'])
        self.assertEqual(failed['state'], 'failed')
        self.assertEqual(failed['error']['code'], 'endpoint_stopped')
        self.assertEqual(controller.memory_registry.snapshot()['entries'], 0)
        self.assertEqual(list((self.root / 'compute').glob('execution-*.json')), [])

    def test_denied_cap_releases_cold_reservation_before_retry(self):
        controller = self.server.controller
        controller.memory_registry.configure_budget(8_000_000_000, 1_000_000_000)
        controller.memory_probe = lambda: 100_000_000_000
        controller.configure({'model_memory_budget_gb': 1})
        record = self.wait(self.start())
        self.assertEqual(record['state'], 'failed')
        self.assertEqual(record['error']['code'], 'job_memory_cap_exceeded')
        self.assertEqual(controller.memory_registry.snapshot()['entries'], 0)
        self.assertIsNone(controller.resident.process)
        controller.configure({'model_memory_budget_gb': 8})
        record = self.wait(self.start())
        self.assertEqual(record['state'], 'failed')
        self.assertEqual(record['error']['code'], 'agent_memory_budget_exceeded')
        self.assertEqual(controller.memory_registry.snapshot()['entries'], 0)

    def test_capped_routes_execute_together_under_one_real_budget(self):
        from mlx_agent.compute import ComputeBusy, ComputeLease
        from mlx_agent.task_serve import TaskServer
        controller = self.server.controller
        controller.memory_registry.configure_budget(8_000_000_000, 1_000_000_000)
        controller.memory_probe = lambda: 100_000_000_000
        controller.configure({'model_memory_budget_gb': 2})
        # The model-free fixture exposes only the MLX ceiling entrypoint.
        mlx = self.root / 'mlx'
        mlx.mkdir()
        (mlx / '__init__.py').write_text('')
        (mlx / 'core.py').write_text('def set_memory_limit(limit): pass\n')
        config = dict(controller.config, task_state_dir=str(self.root / 'second-tasks'), port=0)
        second = TaskServer(('127.0.0.1', 0), config)
        second.controller.resident.environment = dict(controller.resident.environment)
        second.controller.memory_probe = controller.memory_probe
        self.addCleanup(second.server_close)
        self.addCleanup(second.controller.close)
        first_id = self.start('wait')
        second_record = second.controller.submit({'kind': 'classification', 'parameters': {
            'state': 'wait', 'questions': {'route': {'type': 'choice', 'instructions': 'Choose one.'}}}})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            records = [controller.store.get(first_id), second.controller.store.get(second_record['id'])]
            if all(record['state'] == 'running' for record in records):
                break
            if any(record['state'] == 'failed' for record in records):
                self.fail(str(records))
            time.sleep(.01)
        self.assertTrue(all(record['state'] == 'running' for record in records), records)
        budget = controller.memory_registry.snapshot()['agent_budget']
        self.assertTrue(budget['within_budget'], budget)
        self.assertEqual(sum(row['execution_bytes'] for row in budget['allocations'].values()), 4_000_000_000)
        owners = [json.loads(path.read_text()) for path in (self.root / 'compute').glob('execution-*.json')]
        self.assertEqual(len(owners), 2)
        self.assertEqual({r['execution_group']['pgid'] for r in owners},
                         {record['execution']['pid'] for record in records})
        with self.assertRaises(ComputeBusy):
            with ComputeLease('cache', self.root / 'compute'):
                pass
        controller.store.request_cancel(first_id)
        second.controller.store.request_cancel(second_record['id'])
        self.wait(first_id)
        second.controller.job_thread.join(timeout=5)
        self.assertFalse(second.controller.job_thread.is_alive())
        self.assertEqual(list((self.root / 'compute').glob('execution-*.json')), [])
        controller.configure({'model_memory_budget_gb': 3})
        self.assertEqual(controller.memory_policy['model_memory_budget_gb'], 3)
        controller.configure({'model_memory_budget_gb': None})
        self.assertNotIn('model_memory_budget_gb', controller.memory_policy)

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
        with self.server.controller.condition:
            self.assertTrue(self.server.controller.condition.wait_for(
                lambda: self.server.controller.model_state == 'unloaded', timeout=5),
                'The idle worker did not finish applying release mode.')
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

    def test_endpoint_stop_records_failure_without_inventing_user_cancellation(self):
        identifier = self.start('wait')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.request('GET', '/v1/tasks/' + identifier)[1]['state'] == 'running':
                break
            time.sleep(.02)
        self.server.controller.close()
        record = self.request('GET', '/v1/tasks/' + identifier)[1]
        self.assertEqual(record['state'], 'failed', record)
        self.assertEqual(record['error']['code'], 'endpoint_stopped')
        self.assertFalse(record['cancel_requested'])
        self.assertFalse(record['execution_unresolved'])
        self.assertIsNone(self.server.controller.resident.process)

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

    def fake_allocator(self):
        port = self.root / 'fixture_port.py'
        with port.open('a') as stream:
            stream.write("""
import sys, types
cache_state = {'bytes': 12 * 1024**2, 'limit': 128 * 1024**2}
def set_limit(value):
    old = cache_state['limit']; cache_state['limit'] = value; return old
sys.modules['mlx.core'] = types.SimpleNamespace(
    set_cache_limit=set_limit, get_cache_memory=lambda: cache_state['bytes'],
    get_active_memory=lambda: 2 * 1024**2, get_peak_memory=lambda: 4 * 1024**2,
    synchronize=lambda: None, clear_cache=lambda: cache_state.update(bytes=0))
original_predict = predict
def predict(*args):
    result = original_predict(*args)
    cache_state['bytes'] = 12 * 1024**2
    return result
""")

    def test_allocator_limits_manual_and_idle_reclaim_keep_model_loaded(self):
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        process = controller.resident.process
        headers = {'Authorization': 'Bearer ' + 'a' * 32}
        self.assertEqual(self.request('POST', '/_mlx/cache', {})[0], 403)
        code, value = self.request('POST', '/_mlx/cache', {}, headers)
        self.assertEqual(code, 200, value)
        self.assertEqual(value['worker_memory']['cache_bytes'], 0)
        self.assertEqual(value['worker_memory']['loads'], 1)
        code, value = self.request('POST', '/_mlx/policy', {'allocator_cache_limit_mib': 4, 'cache_reclaim': 'idle'}, headers)
        self.assertEqual(code, 200, value)
        self.assertFalse(value['allocator_cache']['pending'])
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        value = controller.status()
        self.assertEqual(value['worker_memory']['cache_bytes'], 0)
        self.assertEqual(value['worker_memory']['loads'], 1)
        self.assertEqual(value['worker_memory']['cache_hits'], 1)
        self.assertIs(controller.resident.process, process)
        self.assertEqual(value['allocator_cache']['last_reclaim']['reason'], 'idle')

    def test_pressure_reclaim_retains_pinned_worker_and_busy_host_defers(self):
        from mlx_agent.compute import ComputeLease
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'minimum_headroom_gb': 1, 'cache_reclaim': 'pressure'})
        controller.memory_probe = lambda: 0
        with ComputeLease('other-workload', self.root / 'compute'):
            self.assertFalse(controller.unload_if_idle())
            self.assertEqual(controller.worker_memory['cache_bytes'], 12 * 1024**2)
        controller.unload_if_idle()
        with controller.condition:
            self.assertTrue(controller.condition.wait_for(lambda: controller.worker_memory['cache_bytes'] == 0, timeout=2))
        self.assertEqual(controller.model_state, 'loaded')
        self.assertEqual(controller.worker_memory['cache_bytes'], 0)
        self.assertEqual(controller.worker_memory['loads'], 1)

    def test_cache_policy_defers_during_active_job_and_manual_reclaim_refuses(self):
        self.fake_allocator()
        port = self.root / 'fixture_port.py'
        with port.open('a') as stream:
            stream.write("""
from pathlib import Path
previous_predict = predict
def predict(*args):
    if args[2] == 'hold':
        marker = Path(__file__).with_name('release-cache-job')
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline: time.sleep(.01)
    return previous_predict(*args)
""")
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        process = controller.resident.process
        identifier = self.start('hold')
        headers = {'Authorization': 'Bearer ' + 'a' * 32}
        self.assertEqual(self.request('POST', '/_mlx/cache', {}, headers)[0], 409)
        code, value = self.request('POST', '/_mlx/policy', {'allocator_cache_limit_mib': 4, 'cache_reclaim': 'idle'}, headers)
        self.assertEqual(code, 200, value)
        self.assertTrue(value['allocator_cache']['pending'])
        self.assertEqual(value['worker_memory']['cache_bytes'], 12 * 1024**2)
        (self.root / 'release-cache-job').touch()
        self.assertEqual(self.wait(identifier)['state'], 'succeeded')
        state = controller.status()
        self.assertFalse(state['allocator_cache']['pending'])
        self.assertEqual(state['worker_memory']['cache_bytes'], 0)
        self.assertIs(controller.resident.process, process)

    def test_cache_fence_failure_stops_worker_and_releases_only_confirmed_group(self):
        from mlx_agent.compute import ComputeBusy, ComputeLease
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        with patch.object(ComputeLease, 'release_idle_process_group', side_effect=ComputeBusy('group changed')):
            code, value = self.request('POST', '/_mlx/cache', {}, {'Authorization': 'Bearer ' + 'a' * 32})
        self.assertEqual(code, 503, value)
        self.assertIsNone(self.server.controller.resident.process)
        with ComputeLease('retry', self.root / 'compute'):
            pass

    def test_unavailable_cache_controls_preserve_the_loaded_worker(self):
        self.fake_allocator()
        with (self.root / 'fixture_port.py').open('a') as stream:
            stream.write("\nsys.modules['mlx.core'].clear_cache = None\n")
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        process = controller.resident.process
        code, value = self.request('POST', '/_mlx/cache', {}, {'Authorization': 'Bearer ' + 'a' * 32})
        self.assertEqual(code, 503, value)
        self.assertIs(controller.resident.process, process)
        self.assertEqual(controller.model_state, 'loaded')
        with self.assertRaises(task_serving.GatewayError):
            controller.configure({'allocator_cache_limit_mib': 4})
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')

    def test_invalid_allocator_policy_returns_an_error_without_loading(self):
        code, value = self.request('POST', '/_mlx/policy', {'allocator_cache_limit_mib': True},
                                   {'Authorization': 'Bearer ' + 'a' * 32})
        self.assertEqual(code, 400, value)
        self.assertEqual(value['error']['code'], 'invalid_policy')
        self.assertIsNone(self.server.controller.resident.process)

    def test_waiting_job_and_cancelled_waiter_preserve_residency(self):
        from mlx_agent.compute import ComputeLease
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'execution_wait_seconds': 2})
        process = controller.resident.process
        with ComputeLease('other-workload', self.root / 'compute'):
            identifier = self.start()
            time.sleep(.1)
            self.assertEqual(self.request('POST', '/v1/tasks/' + identifier + '/cancel', {})[0], 200)
            self.assertEqual(self.wait(identifier)['state'], 'cancelled')
            self.assertIs(controller.resident.process, process)
            identifier = self.start()
            time.sleep(.1)
            self.assertEqual(self.request('GET', '/v1/tasks/' + identifier)[1]['state'], 'starting')
        self.assertEqual(self.wait(identifier)['state'], 'succeeded')
        self.assertIs(controller.resident.process, process)

    def test_compute_contention_keeps_warm_model_for_retry(self):
        from mlx_agent.compute import ComputeLease
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        process = self.server.controller.resident.process
        with ComputeLease('other-workload', self.root / 'compute'):
            denied = self.wait(self.start())
        self.assertEqual(denied['state'], 'failed')
        self.assertEqual(denied['error']['code'], 'compute_busy')
        self.assertIs(self.server.controller.resident.process, process)
        self.assertEqual(self.server.controller.status()['model_state'], 'loaded')
        self.assertIsNone(process.poll())
        identifier = self.start()
        self.assertEqual(self.wait(identifier)['state'], 'succeeded')
        status, result = self.request('GET', '/v1/tasks/' + identifier + '/result')
        self.assertEqual(status, 200, result)
        self.assertEqual(result['result']['answers']['calls'], 2)
        self.assertIs(self.server.controller.resident.process, process)

    def test_compute_contention_does_not_start_a_cold_worker(self):
        from mlx_agent.compute import ComputeLease
        with ComputeLease('other-workload', self.root / 'compute'):
            denied = self.wait(self.start())
        self.assertEqual(denied['error']['code'], 'compute_busy')
        self.assertIsNone(self.server.controller.resident.process)
        self.assertEqual(self.server.controller.status()['model_state'], 'unloaded')
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')

    def test_workspace_admission_keeps_warm_worker_and_reports_load_reuse(self):
        self.assertEqual(self.wait(self.start())['state'],'succeeded')
        controller=self.server.controller
        process=controller.resident.process
        controller.memory_probe=lambda:0
        denied=self.wait(self.start())
        self.assertEqual(denied['error']['code'],'insufficient_headroom')
        self.assertIs(controller.resident.process,process)
        self.assertEqual(controller.memory_registry.snapshot()['reserved_bytes'],0)
        controller.memory_probe=lambda:1<<40
        self.assertEqual(self.wait(self.start())['state'],'succeeded')
        self.assertIs(controller.resident.process,process)
        state=controller.status()
        self.assertEqual(state['worker_memory']['loads'],1)
        self.assertEqual(state['worker_memory']['cache_hits'],1)
        self.assertEqual(state['memory_reservation']['allocation']['phase'],'resident')
        controller.unload()
        self.assertEqual(controller.memory_registry.snapshot()['entries'],0)

    def test_lost_compute_ownership_after_execution_stops_worker(self):
        from mlx_agent.compute import ComputeBusy, ComputeLease
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        process = self.server.controller.resident.process
        with patch.object(ComputeLease, 'release_idle_process_group', side_effect=ComputeBusy('ownership changed')):
            failed = self.wait(self.start())
        self.assertEqual(failed['state'], 'failed')
        self.assertEqual(failed['error']['code'], 'compute_busy')
        self.assertIsNone(self.server.controller.resident.process)
        self.assertEqual(self.server.controller.status()['model_state'], 'unloaded')
        identifier = self.start()
        self.assertEqual(self.wait(identifier)['state'], 'succeeded')
        status, result = self.request('GET', '/v1/tasks/' + identifier + '/result')
        self.assertEqual(status, 200, result)
        self.assertEqual(result['result']['answers']['calls'], 1)
        self.assertIsNot(self.server.controller.resident.process, process)

    def test_settlement_contention_keeps_completed_result_and_warm_model(self):
        from mlx_agent.residency import MemoryAdmissionError
        controller=self.server.controller
        with patch.object(controller.memory_registry,'settle',side_effect=MemoryAdmissionError('memory_registry_busy','busy')):
            self.assertEqual(self.wait(self.start())['state'],'succeeded')
        process=controller.resident.process
        self.assertIn('reservation',controller.last_error)
        self.assertGreater(controller.memory_registry.snapshot()['reserved_bytes'],0)
        self.assertEqual(self.wait(self.start())['state'],'succeeded')
        self.assertIs(controller.resident.process,process)
        self.assertEqual(controller.worker_memory['cache_hits'],1)
        self.assertIsNone(controller.last_error)
        self.assertEqual(controller.memory_registry.snapshot()['reserved_bytes'],0)

    def test_changed_model_refuses_reuse_and_releases_stale_weights(self):
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        (self.model / 'config.json').write_text('{"model_type":"laya","changed":true}')
        status, value = self.request('POST', '/v1/tasks', {'kind': 'classification', 'parameters': {
            'state': 'test', 'questions': {'route': {'type': 'choice', 'instructions': 'Choose.'}}}})
        self.assertEqual(status, 400, value)
        self.assertIsNone(self.server.controller.resident.process)
