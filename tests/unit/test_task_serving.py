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
from mlx_agent.inference_auth import runtime_fingerprint
from mlx_agent.jit_serve import model_fingerprint, validate_memory_policy
from .backend_fixtures import synthetic_manifests, synthetic_registries


def _claim_test_gateway(test, server=None):
    import os
    from mlx_agent.compute import process_birth_identity
    from mlx_agent.model_serving import acquire, state_root
    environment = patch.dict(os.environ, {'XDG_STATE_HOME': str(test.root / 'state')})
    environment.start()
    test.addCleanup(environment.stop)
    discovery = patch('mlx_agent.service_registry.discover_services', return_value={'services': [], 'errors': []})
    discovery.start()
    test.addCleanup(discovery.stop)
    server = test.server if server is None else server
    service = {'port': server.server_port, 'runtime': 'tasks', 'auth_required': False}
    claim = acquire(str(test.model), kind='service', service=service, root=state_root())
    claim.update(state='active', pid=os.getpid(), birth=process_birth_identity(os.getpid()),
                 pgid=os.getpgrp(), jit=False, ready=True)
    server.controller.resident.model_serving_claim = claim


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
    def test_execution_admits_selected_cold_and_warm_growth_before_dispatch(self):
        from mlx_agent.task_serving import _trellis_component_budget
        controller = self.server.controller
        controller.memory_probe = lambda: 100_000_000_000
        path = self.root / 'trellis'
        path.mkdir()
        names = ('image_encoder', 'ss_flow', 'ss_decoder', 'shape_flow_512', 'shape_flow_1024',
                 'shape_decoder', 'tex_flow_512', 'tex_flow_1024', 'tex_decoder')
        (path / 'config.json').write_text(json.dumps({'model_type': 'trellis2', 'components': {n: {} for n in names}}))
        for i, name in enumerate(names, 1):
            (path / (name + '.safetensors')).write_bytes(b'x' * i)
        resident = ['image_encoder', 'ss_flow', 'ss_decoder', 'shape_flow_512', 'shape_decoder', 'tex_flow_512', 'tex_decoder']
        telemetry = {'schema':'worker-memory/1', 'active_bytes':1, 'cache_bytes':0, 'peak_bytes':1,
                     'loads':1, 'cache_hits':0, 'load_seconds':0., 'captured_at':1.,
                     'components': {'schema':'resident-components/1', 'resident':resident,
                       'loads':{n:1 for n in resident}, 'load_seconds':{n:0. for n in resident},
                       'checkpoint_bytes':{n:1 for n in resident}}}
        observed = []
        cold, warm = controller._check_memory, controller._reserve_workspace
        def check(*, weight_bytes=None):
            observed.append(('cold', weight_bytes))
            return cold(weight_bytes=weight_bytes)
        def reserve(extra_bytes=0):
            observed.append(('warm', extra_bytes))
            return warm(extra_bytes)
        def component_budget(_path, _kind, _plan, _telemetry, *, warm):
            return _trellis_component_budget(path, 'mesh_generation',
                {'argv': ['runner', '--resolution', '1024' if warm else '512']},
                telemetry if warm else None, warm=warm)
        with patch('mlx_agent.task_serving._trellis_component_budget', side_effect=component_budget), \
             patch.object(controller, '_check_memory', side_effect=check), \
             patch.object(controller, '_reserve_workspace', side_effect=reserve):
            self.assertEqual(self.wait(self.start())['state'], 'succeeded')
            pid = controller.resident.process.pid
            self.assertEqual(self.wait(self.start())['state'], 'succeeded')
            self.assertEqual(controller.resident.process.pid, pid)
            controller.memory_policy['model_memory_budget_gb'] = 1.5
            denied = self.wait(self.start())
            self.assertEqual(denied['error']['code'], 'job_memory_cap_exceeded')
            self.assertEqual(controller.resident.process.pid, pid)
            failure = controller.status()['last_memory_failure']
            self.assertEqual(failure['reason'], 'job_memory_cap_exceeded')
            self.assertEqual(failure['route_limit_bytes'], 1_500_000_000)
            self.assertGreater(failure['required_bytes'], failure['route_limit_bytes'])
            import hashlib
            bound = controller.memory_reservation['allocation']['worker']
            self.assertEqual(failure['worker_pid'], pid)
            self.assertEqual(failure['worker_pgid'], bound['pgid'])
            self.assertEqual(failure['worker_identity_sha256'], hashlib.sha256(bound['identity'].encode()).hexdigest())
        self.assertEqual(observed, [('cold', 32), ('warm', 15), ('warm', 15)])

    def test_resident_component_statistics_reach_worker_status(self):
        port = self.root / 'fixture_port.py'
        source = port.read_text().replace('def load(path): return Model(), None', """class Loaded(tuple):
    def component_statistics(self):
        return {'schema':'resident-components/1', 'resident':['image_encoder'],
                'loads':{'image_encoder':1}, 'load_seconds':{'image_encoder':0.25},
                'checkpoint_bytes':{'image_encoder':24}}
def load(path): return Loaded((Model(), None))""", 1)
        port.write_text(source)
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        memory = self.server.controller.status()['worker_memory']
        self.assertEqual(memory['components']['resident'], ['image_encoder'])
        self.assertEqual(memory['components']['loads'], {'image_encoder':1})
        self.assertEqual(memory['components']['checkpoint_bytes'], {'image_encoder':24})
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        self.assertEqual(self.server.controller.status()['worker_memory']['cache_hits'], 1)

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
        config = {'local_path': str(self.model), 'model': str(self.model), 'runtime': 'tasks',
            'fingerprint': model_fingerprint(self.model), 'jit': True,
            'task_runtime_fingerprint': task_serving.runtime_fingerprint(identity['backend']),
            'port': 0, 'task_identity': identity, 'task_kinds': ['classification'], 'task_state_dir': str(self.root / 'tasks'),
            'max_tokens': 0, 'worker_argv': [sys.executable], 'control_token': 'a' * 32,
            'memory_state_dir': str(self.root / 'memory'),
            'compute_state_dir': str(self.root / 'compute'),
            'memory_policy': {'keep_loaded': True}}
        self.server = TaskServer(('127.0.0.1', 0), config)
        _claim_test_gateway(self)
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
        # A later job reports its own failure, not the old breach.
        failed = self.wait(self.start('fail'))
        self.assertEqual(failed['state'], 'failed')
        self.assertEqual(failed['error']['code'], 'decide_failed')
        self.assertEqual(controller.memory_registry.snapshot()['entries'], 0)
        self.assertEqual(list((self.root / 'compute').glob('execution-*.json')), [])

    def test_guard_stop_keeps_the_endpoint_admitting_jobs(self):
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
        self.assertEqual(self.wait(identifier)['error']['code'], 'job_memory_cap_exceeded')
        self.assertIsNotNone(worker.poll())
        # The guard released only its worker; the same endpoint admits and runs the next job.
        record = self.wait(self.start())
        self.assertEqual(record['state'], 'succeeded', record)
        self.assertEqual(self.request('GET', '/v1/tasks/' + record['id'] + '/result')[0], 200)

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
        from mlx_agent.model_serving import write_policy
        write_policy([str(self.model)])
        config = dict(controller.config, task_state_dir=str(self.root / 'second-tasks'), port=0)
        second = TaskServer(('127.0.0.1', 0), config)
        self.addCleanup(second.server_close)
        self.addCleanup(second.controller.close)
        _claim_test_gateway(self, second)
        second.controller.resident.environment = dict(controller.resident.environment)
        second.controller.memory_probe = controller.memory_probe
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
    set_cache_limit=set_limit, set_memory_limit=lambda value: value, get_cache_memory=lambda: cache_state['bytes'],
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

    def test_admission_pressure_reclaims_once_and_rechecks_before_dispatch(self):
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim': 'pressure', 'minimum_headroom_gb': 1})
        process = controller.resident.process
        original = controller._reserve_workspace
        calls = []
        def admission(extra_bytes=0):
            calls.append(controller.resident.telemetry['cache_bytes'])
            if len(calls) == 1:
                raise task_serving.GatewayError(503, 'insufficient_headroom', 'first admission')
            return original(extra_bytes)
        with patch.object(controller, 'unload_if_idle', return_value=False), \
             patch.object(controller, '_reserve_workspace', side_effect=admission):
            result = self.wait(self.start())
        self.assertEqual(result['state'], 'succeeded', result)
        self.assertEqual(calls, [12 * 1024**2, 0])
        self.assertIs(controller.resident.process, process)
        self.assertEqual(controller.worker_memory['loads'], 1)
        self.assertEqual(controller.worker_memory['cache_hits'], 1)
        self.assertEqual(controller.resident.cache_state['last_reclaim']['reason'], 'pressure')

    def test_admission_pressure_second_failure_is_final(self):
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim': 'pressure', 'minimum_headroom_gb': 1})
        process = controller.resident.process
        calls = []
        def admission(extra_bytes=0):
            calls.append(controller.resident.telemetry['cache_bytes'])
            raise task_serving.GatewayError(503, 'job_memory_cap_exceeded', 'fresh admission')
        with patch.object(controller, 'unload_if_idle', return_value=False), \
             patch.object(controller, '_reserve_workspace', side_effect=admission):
            result = self.wait(self.start())
        self.assertEqual(result['error']['code'], 'job_memory_cap_exceeded')
        self.assertEqual(calls, [12 * 1024**2, 0])
        self.assertIs(controller.resident.process, process)
        self.assertEqual(controller.worker_memory['loads'], 1)

    def test_admission_off_does_not_reclaim_or_retry(self):
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        process = controller.resident.process
        with patch.object(controller, '_reserve_workspace', side_effect=task_serving.GatewayError(503, 'insufficient_headroom', 'first')) as admission:
            result = self.wait(self.start())
        self.assertEqual(result['error']['code'], 'insufficient_headroom')
        self.assertEqual(admission.call_count, 1)
        self.assertIs(controller.resident.process, process)
        self.assertEqual(controller.worker_memory['cache_bytes'], 12 * 1024**2)

    def test_pressure_reclaims_to_fit_shared_budget_admission(self):
        import os
        from mlx_agent.memory_budget import process_snapshot
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1})
        controller.memory_probe = lambda: 100_000_000_000
        process = controller.resident.process
        def measured_processes():
            rows = process_snapshot()
            for pid, size in ((os.getpid(), 100_000_000),
                              (process.pid, 2 * 1024**2 + controller.resident.telemetry['cache_bytes'])):
                if pid in rows:
                    rows[pid]['rss'] = rows[pid]['physical_bytes'] = size
            return rows
        controller.memory_registry.process_probe = measured_processes
        controller.memory_registry.configure_budget(100_000_000_000, 0)
        used = controller.memory_registry.snapshot()['agent_budget']['accounted_bytes']
        controller.memory_registry.configure_budget(used + controller.workspace_bytes - 6 * 1024**2, 0)
        with patch.object(controller, 'unload_if_idle', return_value=False):
            result = self.wait(self.start())
        self.assertEqual(result['state'], 'succeeded', result)
        self.assertIs(controller.resident.process, process)
        self.assertEqual(controller.resident.cache_state['last_reclaim']['reason'], 'pressure')
        self.assertTrue(controller.memory_registry.snapshot()['agent_budget']['within_budget'])

    def test_admission_shared_budget_refusal_rechecks_after_reclaim(self):
        from mlx_agent.residency import MemoryAdmissionError
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1})
        process = controller.resident.process
        original = controller.memory_registry.reserve_execution
        calls = []
        def admit(*args, **kwargs):
            calls.append(controller.resident.telemetry['cache_bytes'])
            if len(calls)==1:
                raise MemoryAdmissionError('agent_memory_budget_exceeded', 'budget')
            return original(*args, **kwargs)
        with patch.object(controller, 'unload_if_idle', return_value=False), \
             patch.object(controller.memory_registry, 'reserve_execution', side_effect=admit):
            result = self.wait(self.start())
        self.assertEqual(result['state'], 'succeeded', result)
        self.assertEqual(calls, [12 * 1024**2, 0])
        self.assertIs(controller.resident.process, process)

    def test_admission_settlement_contention_finishes_and_keeps_worker(self):
        from mlx_agent.residency import MemoryAdmissionError
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1})
        process = controller.resident.process
        before = controller.memory_registry.inspect()['entries'][controller.memory_token]
        with patch.object(controller, 'unload_if_idle', return_value=False), \
             patch.object(controller, '_reserve_workspace', side_effect=task_serving.GatewayError(503, 'insufficient_headroom', 'first')), \
             patch.object(controller, '_settle_task_memory', return_value=False), \
             patch.object(controller.memory_registry, 'settle', side_effect=MemoryAdmissionError('memory_registry_busy', 'busy')):
            result = self.wait(self.start())
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['error']['code'], 'memory_registry_busy')
        self.assertIs(controller.resident.process, process)
        self.assertEqual(controller.active_requests, 0)
        self.assertIn('Memory reservation remains pending', controller.status()['last_error'])
        after = controller.memory_registry.inspect()['entries'][controller.memory_token]
        self.assertEqual(after, before)

    def test_cancel_after_admission_reclaim_preserves_worker(self):
        import threading
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1})
        process = controller.resident.process
        entered, resume = threading.Event(), threading.Event()
        control = controller.resident.control_cache
        def reclaim(*args, **kwargs):
            result = control(*args, **kwargs)
            entered.set()
            if not resume.wait(5):raise TimeoutError('cancel handshake')
            return result
        with patch.object(controller, 'unload_if_idle', return_value=False), \
             patch.object(controller, '_reserve_workspace', side_effect=task_serving.GatewayError(503, 'insufficient_headroom', 'first')), \
             patch.object(controller.resident, 'control_cache', side_effect=reclaim):
            identifier = self.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.request('POST', '/v1/tasks/'+identifier+'/cancel', {})[0], 200)
            finally:
                resume.set()
            result = self.wait(identifier)
        self.assertEqual(result['state'], 'cancelled')
        self.assertIs(controller.resident.process, process)
        self.assertEqual(controller.worker_memory['loads'], 1)

    def test_pressure_retries_job_cap_using_same_compute_lease(self):
        from mlx_agent.compute import ComputeLease
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1, 'model_memory_budget_gb':1.51})
        controller.memory_probe = lambda: 100_000_000_000
        process = controller.resident.process
        original = ComputeLease.__enter__
        enters = []
        def enter(lease):
            enters.append(lease.operation)
            return original(lease)
        with patch.object(controller, 'unload_if_idle', return_value=False), patch.object(ComputeLease, '__enter__', enter):
            result = self.wait(self.start())
        self.assertEqual(result['state'], 'succeeded', result)
        self.assertEqual(len(enters), 1)
        self.assertIs(controller.resident.process, process)
        self.assertEqual(controller.resident.cache_state['last_reclaim']['reason'], 'pressure')
        self.assertEqual(controller.worker_memory['loads'], 1)

    def test_pressure_does_not_retry_when_cache_evidence_is_unusable(self):
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1})
        process = controller.resident.process
        for mode in ('idle', 'unavailable', 'zero', 'unknown'):
            with self.subTest(mode=mode):
                policy = controller.memory_policy['cache_reclaim']
                cache = controller.resident.cache_state['available']
                value = controller.resident.telemetry['cache_bytes']
                if mode == 'idle':controller.memory_policy['cache_reclaim'] = 'idle'
                if mode == 'unavailable':controller.resident.cache_state['available'] = False
                if mode == 'zero':controller.resident.telemetry['cache_bytes'] = 0
                if mode == 'unknown':controller.resident.telemetry['cache_bytes'] = None
                try:
                    with patch.object(controller, 'unload_if_idle', return_value=False), \
                         patch.object(controller, '_reserve_workspace', side_effect=task_serving.GatewayError(503,'insufficient_headroom','first')) as admit, \
                         patch.object(controller.resident, 'control_cache', wraps=controller.resident.control_cache) as control:
                        result = self.wait(self.start())
                    self.assertEqual(result['error']['code'], 'insufficient_headroom')
                    self.assertEqual(admit.call_count, 1)
                    control.assert_not_called()
                    self.assertIs(controller.resident.process, process)
                finally:
                    controller.memory_policy['cache_reclaim'] = policy
                    controller.resident.cache_state['available'] = cache
                    controller.resident.telemetry['cache_bytes'] = value

    def test_cancel_before_admission_reclaim_preserves_worker(self):
        import threading
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1})
        process = controller.resident.process
        entered, resume = threading.Event(), threading.Event()
        def deny(*args, **kwargs):
            entered.set()
            if not resume.wait(5):raise TimeoutError('cancel handshake')
            raise task_serving.GatewayError(503, 'insufficient_headroom', 'first')
        with patch.object(controller, 'unload_if_idle', return_value=False), \
             patch.object(controller, '_reserve_workspace', side_effect=deny), \
             patch.object(controller.resident, 'control_cache', wraps=controller.resident.control_cache) as control:
            identifier = self.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.request('POST', '/v1/tasks/'+identifier+'/cancel', {})[0], 200)
            finally:resume.set()
            result = self.wait(identifier)
        self.assertEqual(result['state'], 'cancelled')
        control.assert_not_called()
        self.assertIs(controller.resident.process, process)

    def test_pressure_rechecks_fresh_os_headroom_after_reclaim(self):
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1})
        probes = []
        def probe():
            cache = controller.resident.telemetry['cache_bytes']
            value = 1_000_000_001 if cache else 100_000_000_000
            probes.append(value)
            return value
        controller.memory_probe = probe
        with patch.object(controller, 'unload_if_idle', return_value=False):
            result = self.wait(self.start())
        self.assertEqual(result['state'], 'succeeded', result)
        self.assertEqual(probes[:2], [1_000_000_001, 100_000_000_000])
        self.assertEqual(controller.resident.cache_state['last_reclaim']['reason'], 'pressure')

    def test_unavailable_admission_reclaim_keeps_original_refusal_and_worker(self):
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1})
        process = controller.resident.process
        with patch.object(controller, 'unload_if_idle', return_value=False), \
             patch.object(controller, '_reserve_workspace', side_effect=task_serving.GatewayError(503,'insufficient_headroom','first')) as admission, \
             patch.object(controller.resident, 'control_cache', return_value=False) as control:
            result = self.wait(self.start())
        self.assertEqual(result['error']['code'], 'insufficient_headroom')
        self.assertEqual(admission.call_count, 1)
        self.assertEqual(control.call_count, 1)
        self.assertIs(controller.resident.process, process)

    def test_admission_reclaim_fence_failure_stops_owned_worker(self):
        from mlx_agent.compute import ComputeBusy, ComputeLease
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1})
        with patch.object(controller, 'unload_if_idle', return_value=False), \
             patch.object(controller, '_reserve_workspace', side_effect=task_serving.GatewayError(503,'insufficient_headroom','first')), \
             patch.object(ComputeLease, 'release_idle_process_group', side_effect=ComputeBusy('group changed')):
            result = self.wait(self.start())
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['error']['code'], 'compute_busy')
        self.assertIsNone(controller.resident.process)
        with ComputeLease('after-reclaim-fence', self.root / 'compute'):
            pass

    def test_cold_admission_never_reclaims(self):
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1})
        with patch.object(controller, '_check_memory', side_effect=task_serving.GatewayError(503,'insufficient_headroom','first')) as admission, \
             patch.object(controller.resident, 'control_cache', wraps=controller.resident.control_cache) as control:
            result = self.wait(self.start())
        self.assertEqual(result['error']['code'], 'insufficient_headroom')
        self.assertEqual(admission.call_count, 1)
        control.assert_not_called()
        self.assertIsNone(controller.resident.process)

    def test_pressure_does_not_retry_unknown_or_busy_admission(self):
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1})
        for code in ('memory_unknown', 'memory_registry_busy'):
            with self.subTest(code=code), patch.object(controller, 'unload_if_idle', return_value=False), \
                 patch.object(controller, '_reserve_workspace', side_effect=task_serving.GatewayError(503,code,'unavailable')) as admission, \
                 patch.object(controller.resident, 'control_cache', wraps=controller.resident.control_cache) as control:
                result = self.wait(self.start())
                self.assertEqual(result['error']['code'], code)
                self.assertEqual(admission.call_count, 1)
                control.assert_not_called()

    def test_policy_cap_change_during_reclaim_applies_only_to_next_job(self):
        from mlx_agent.compute import ComputeLease
        self.fake_allocator()
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        controller.configure({'cache_reclaim':'pressure', 'minimum_headroom_gb':1, 'model_memory_budget_gb':2})
        controller.memory_probe = lambda: 100_000_000_000
        original_admit, original_control = controller._reserve_workspace, controller.resident.control_cache
        original_execution, original_enter = controller.memory_registry.reserve_execution, ComputeLease.__enter__
        attempts, caps, modes = [], [], []
        def admit(extra_bytes=0):
            attempts.append(extra_bytes)
            if len(attempts)==1:raise task_serving.GatewayError(503, 'insufficient_headroom', 'first')
            return original_admit(extra_bytes)
        def reclaim(*args, **kwargs):
            controller.configure({'model_memory_budget_gb':None})
            return original_control(*args, **kwargs)
        def reserve(key, limit=None, probe=None):
            caps.append(limit)
            return original_execution(key, limit, probe)
        def enter(lease):
            modes.append(lease.shared)
            return original_enter(lease)
        with patch.object(controller, 'unload_if_idle', return_value=False), \
             patch.object(controller.memory_registry, 'shared_execution_enabled', return_value=True), \
             patch.object(controller, '_reserve_workspace', side_effect=admit), \
             patch.object(controller.resident, 'control_cache', side_effect=reclaim), \
             patch.object(controller.memory_registry, 'reserve_execution', side_effect=reserve), \
             patch.object(ComputeLease, '__enter__', enter):
            self.assertEqual(self.wait(self.start())['state'], 'succeeded')
            self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        self.assertEqual(caps, [2_000_000_000, None])
        self.assertEqual(modes, [True, False])

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

    def test_cancel_after_lease_before_dispatch_preserves_warm_worker(self):
        from mlx_agent.compute import ComputeLease
        self.assertEqual(self.wait(self.start())['state'], 'succeeded')
        controller = self.server.controller
        process = controller.resident.process
        before = controller.memory_registry.snapshot()
        original_enter = ComputeLease.enter_wait

        def cancel_after_enter(lease, *args, **kwargs):
            original_enter(lease, *args, **kwargs)
            controller.store.request_cancel(lease.operation.removeprefix('task:'))

        with patch.object(ComputeLease, 'enter_wait', cancel_after_enter):
            cancelled = self.wait(self.start())
        self.assertEqual(cancelled['state'], 'cancelled')
        self.assertIs(controller.resident.process, process)
        self.assertIsNone(process.poll())
        self.assertEqual(controller.memory_registry.snapshot(), before)
        self.assertEqual(controller.active_requests, 0)
        self.assertEqual(list((self.root / 'compute').glob('execution-*.json')), [])
        self.assertEqual(controller.status()['worker_memory']['loads'], 1)
        identifier = self.start()
        self.assertEqual(self.wait(identifier)['state'], 'succeeded')
        status, result = self.request('GET', '/v1/tasks/' + identifier + '/result')
        self.assertEqual(status, 200)
        self.assertEqual(result['result']['answers']['calls'], 2)
        self.assertIs(controller.resident.process, process)
        self.assertEqual(controller.status()['worker_memory']['loads'], 1)

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


class EmbeddingHTTPTests(unittest.TestCase):
    request = TaskHTTPTests.request
    close_server = TaskHTTPTests.close_server

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
        (self.model / 'config.json').write_text('{"model_type":"qwen3"}')
        (self.model / 'model.safetensors').write_bytes(b'fixture')
        (self.root / 'mlx_embeddings.py').write_text('''
class Rows:
    def __init__(self, rows): self.rows = rows
    def tolist(self): return self.rows
class Result:
    def __init__(self, rows): self.text_embeds = Rows(rows)
def load(path, **options): return object(), object()
def generate(model, tokenizer, texts, max_length):
    return Result([[float(len(text)), float(max_length), 0.5] for text in texts])
''')
        identity = capture_identity(self.model, load_manifests()['mlx-embeddings'], Path(sys.executable))
        config = {'local_path': str(self.model), 'model': str(self.model), 'runtime': 'tasks',
            'fingerprint': model_fingerprint(self.model), 'jit': True,
            'task_runtime_fingerprint': task_serving.runtime_fingerprint(identity['backend']),
            'port': 0, 'task_identity': identity, 'task_kinds': ['embedding'], 'task_state_dir': str(self.root / 'tasks'),
            'max_tokens': 0, 'worker_argv': [sys.executable], 'control_token': 'a' * 32,
            'memory_state_dir': str(self.root / 'memory'), 'compute_state_dir': str(self.root / 'compute'),
            'memory_policy': {'keep_loaded': True}}
        self.server = TaskServer(('127.0.0.1', 0), config)
        _claim_test_gateway(self)
        self.server.controller.config['port'] = self.server.server_port
        self.server.controller.resident.environment = dict(os.environ, PYTHONPATH=str(self.root))
        self.port = self.server.server_port
        def planner(model, request):
            return {'backend': 'mlx-embeddings', 'model': model, 'request': request,
                'count': len(json.loads(Path(request).read_text())['input']),
                'argv': [sys.executable, str(Path(task_serving.__file__).with_name('embed_runner.py')),
                         '--model', model, '--request', request]}
        for context in (patch('mlx_agent.task_kinds.plan_embed', side_effect=planner),
                        patch('mlx_agent.inference_runner.sync_ports'),
                        patch('mlx_agent.task_serving.ComputeLease', side_effect=lambda operation, root=None, **options: ComputeLease(operation, self.root / 'compute', **options))):
            context.start()
            self.addCleanup(context.stop)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def embed(self, value, headers=None):
        return self.request('POST', '/v1/embeddings', dict({'model': str(self.model)}, **value), headers)

    def test_openai_embeddings_return_ordered_vectors_from_the_resident_model(self):
        status, value = self.embed({'input': 'hello'})
        self.assertEqual(status, 200, value)
        self.assertEqual(value, {'object': 'list', 'model': str(self.model),
            'data': [{'object': 'embedding', 'index': 0, 'embedding': [5.0, 8192.0, 0.5]}]})
        texts = ['x' * (index % 40 + 1) for index in range(130)]
        status, value = self.embed({'input': texts, 'dimensions': 3})
        self.assertEqual(status, 200, value)
        self.assertEqual([row['index'] for row in value['data']], list(range(130)))
        self.assertEqual([row['embedding'][0] for row in value['data']], [float(len(text)) for text in texts])
        self.assertEqual(len(self.server.controller.store.list()), 3)
        self.assertEqual(self.server.controller.active_requests, 0)

    def test_accepts_the_gateway_model_aliases_and_echoes_the_requested_name(self):
        status, value = self.request('POST', '/v1/embeddings', {'model': self.model.name, 'input': 'hello'})
        self.assertEqual(status, 200, value)
        self.assertEqual(value['model'], self.model.name)

    def test_base64_encoding_packs_little_endian_float32(self):
        import base64
        import struct
        status, value = self.embed({'input': ['abc', 'abcdefg'], 'encoding_format': 'base64'})
        self.assertEqual(status, 200, value)
        self.assertEqual([list(struct.unpack('<3f', base64.b64decode(row['embedding']))) for row in value['data']],
                         [[3.0, 8192.0, 0.5], [7.0, 8192.0, 0.5]])

    def test_refuses_other_models_token_input_and_unmatched_dimensions(self):
        self.assertEqual(self.request('POST', '/v1/embeddings', {'model': 'other', 'input': 'a'})[0], 404)
        for body in ({'input': [[1, 2, 3]]}, {'input': []}, {'input': 'a', 'encoding_format': 'int8'},
                     {'input': 'a', 'input_type': 'query'}):
            with self.subTest(body=body):
                self.assertEqual(self.embed(body)[0], 400)
        status, value = self.embed({'input': 'a', 'dimensions': 1024})
        self.assertEqual((status, value['error']['code']), (400, 'unsupported_dimensions'))

    def test_requires_the_api_key(self):
        import hashlib
        token = 'client-' + 'b' * 40
        self.server.config['api_key_sha256'] = hashlib.sha256(token.encode()).hexdigest()
        self.server.config['auth_runtime_sha256'] = runtime_fingerprint()
        self.server.config['jit'] = True
        self.assertEqual(self.embed({'input': 'a'})[0], 401)
        self.assertEqual(self.embed({'input': 'a'}, {'Authorization': 'Bearer ' + token})[0], 200)


class SystemOneHTTPTests(unittest.TestCase):
    request = TaskHTTPTests.request
    close_server = TaskHTTPTests.close_server

    def setUp(self):
        import os
        import threading
        from mlx_agent.compute import ComputeLease
        from mlx_agent.task_identity import capture_identity
        from mlx_agent.task_serve import TaskServer
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / 'kev-model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{"model_type":"kev"}')
        (self.model / 'model.safetensors').write_bytes(b'fixture')
        (self.root / 'fake_kev.py').write_text('''
def load(path, **options): return object(), object()
def predict(model, tokenizer, state, questions):
    return {"answers": {name: {"type": "noul", "noul": round(len(state) / 100, 2)} for name in questions},
            "usage": {"input_tokens": len(state), "output_tokens": 3}}
''')
        identity = capture_identity(self.model, load_manifests()['mlx-embeddings'], Path(sys.executable))
        config = {'local_path': str(self.model), 'model': str(self.model), 'runtime': 'tasks',
            'fingerprint': model_fingerprint(self.model), 'jit': True,
            'task_runtime_fingerprint': task_serving.runtime_fingerprint(identity['backend']),
            'port': 0, 'task_identity': identity, 'task_kinds': ['classification'], 'task_state_dir': str(self.root / 'tasks'),
            'max_tokens': 0, 'worker_argv': [sys.executable], 'control_token': 'a' * 32,
            'memory_state_dir': str(self.root / 'memory'), 'compute_state_dir': str(self.root / 'compute'),
            'memory_policy': {'keep_loaded': True}}
        self.server = TaskServer(('127.0.0.1', 0), config)
        _claim_test_gateway(self)
        self.server.controller.config['port'] = self.server.server_port
        self.server.controller.resident.environment = dict(os.environ, PYTHONPATH=str(self.root))
        self.port = self.server.server_port
        def planner(model, request):
            return {'backend': 'mlx-embeddings', 'model': model, 'request': request,
                'argv': [sys.executable, str(Path(task_serving.__file__).with_name('decide_runner.py')),
                         '--module', 'fake_kev', '--model', model, '--request', request]}
        for context in (patch('mlx_agent.task_kinds.plan_decide', side_effect=planner),
                        patch('mlx_agent.inference_runner.sync_ports'),
                        patch('mlx_agent.task_serving.ComputeLease', side_effect=lambda operation, root=None, **options: ComputeLease(operation, self.root / 'compute', **options))):
            context.start()
            self.addCleanup(context.stop)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def decide(self, value, headers=None):
        body = dict({'model': self.model.name, 'state': 'charged twice',
                     'questions': {'refund': {'type': 'noul', 'instructions': 'refund?'}}}, **value)
        return self.request('POST', '/v1/systemone', body, headers)

    def test_systemone_answers_from_the_resident_model_with_latency(self):
        status, value = self.decide({})
        self.assertEqual(status, 200, value)
        latency = value.pop('latency_ms')
        self.assertTrue(isinstance(latency, float) and latency >= 0)
        self.assertEqual(value, {'model': self.model.name, 'answers': {'refund': {'type': 'noul', 'noul': 0.13}},
                                 'usage': {'input_tokens': 13, 'output_tokens': 3}})
        status, value = self.decide({'state': 'x' * 50, 'questions': {'a': {'type': 'noul', 'instructions': 'a'},
                                                                      'b': {'type': 'noul', 'instructions': 'b'}}})
        self.assertEqual((status, value['answers']), (200, {'a': {'type': 'noul', 'noul': 0.5}, 'b': {'type': 'noul', 'noul': 0.5}}))
        self.assertEqual(len(self.server.controller.store.list()), 2)
        self.assertEqual(self.server.controller.active_requests, 0)

    def test_warm_decisions_never_list_every_host_process(self):
        self.assertEqual(self.decide({})[0], 200)
        launched = []
        recording = [True]

        def audit(event, arguments):
            if event == 'subprocess.Popen' and recording[0]:
                launched.append([str(item) for item in arguments[1]])
        sys.addaudithook(audit)
        try:
            for _ in range(3):
                self.assertEqual(self.decide({})[0], 200)
        finally:
            recording[0] = False
        # Listing every process reads each command line: hundreds of milliseconds on a busy host.
        self.assertEqual([argv for argv in launched if argv[:1] == ['/bin/ps'] and any(
            option.startswith('-') and 'a' in option for option in argv[1:])], [])
        self.assertTrue(launched)

    def test_refuses_other_models_unknown_fields_and_invalid_questions(self):
        self.assertEqual(self.decide({'model': 'other'})[0], 404)
        for body in ({'assets': {}}, {'questions': {}}, {'questions': {'q': {'type': 'pick', 'instructions': 'x'}}},
                     {'state': 3}):
            with self.subTest(body=body):
                self.assertEqual(self.decide(body)[0], 400)
        self.assertEqual(self.server.controller.store.list(), [])

    def test_embedding_routes_do_not_serve_decisions(self):
        self.server.config['task_kinds'] = ['embedding']
        self.assertEqual(self.decide({})[0], 404)
