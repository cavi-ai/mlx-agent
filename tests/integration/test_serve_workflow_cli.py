"""Role resolution using isolated model-free typed and chat gateways."""
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from mlx_agent import task_serving, workflow_activation, workflow_sets
from mlx_agent.backends import load_manifests
from mlx_agent.cli import main
from mlx_agent.serve import _preview_hash, stop_serve
from mlx_agent.serve_profiles import save_profile
from mlx_agent.task_identity import capture_identity


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class WorkflowRoleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = Path(__file__).resolve().parents[2] / 'src'
        (self.root / 'fixture_port.py').write_text('''
import time
class Model:
    def __init__(self, path): self.path = path
def load(path): return Model(path), None
def predict(model, tokenizer, state, questions):
    if state == "wait": time.sleep(30)
    return {"answers": {"model": model.path}}
''')
        self.launcher = self.root / 'task_fixture_launcher.py'
        self.launcher.write_text('''
import os, runpy, sys
from pathlib import Path
from mlx_agent import task_kinds, task_serving, inference_runner, task_serve
root = str(Path(__file__).resolve().parent)
task_serving.backend_environment = lambda: {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": root}
inference_runner.sync_ports = lambda *args, **kwargs: None
def planner(model, request):
    return {"backend": "mlx-embeddings", "model": model, "request": request,
            "argv": [sys.executable, str(Path(task_serving.__file__).with_name("decide_runner.py")),
                     "--module", "fixture_port", "--model", model, "--request", request]}
task_kinds.plan_decide = planner
original_init = task_serve.TaskServer.__init__
def isolated_init(self, address, config):
    config = dict(config, compute_state_dir=str(Path(root) / "compute"), memory_state_dir=str(Path(root) / "compute"))
    original_init(self, address, config)
task_serve.TaskServer.__init__ = isolated_init
task_serve.main()
''')
        self.launcher_hash = hashlib.sha256(self.launcher.read_bytes()).hexdigest()
        self.models = {}
        for name in ('classifier', 'media', 'coder'):
            model = self.root / name
            model.mkdir()
            (model / 'config.json').write_text('{"model_type":"laya"}')
            (model / 'model.safetensors').write_bytes(b'fixture')
            self.models[name] = model
        save_profile('small', {'min_headroom_gb': 2, 'idle_timeout': 0,
                               'keep_loaded': True, 'unload_under_pressure': False, 'execution_wait_seconds': 5}, self.root)
        ports = set()
        while len(ports) < 3: ports.add(free_port())
        self.ports = dict(zip(('classifier', 'media', 'coder'), sorted(ports)))
        self.definition = {'schema': 'serving-workflow/1', 'name': 'studio', 'members': {
            name: {'path': str(self.models[name]), 'runtime': 'mlx_lm' if name == 'coder' else 'tasks',
                   'port': self.ports[name], 'profile': 'small'} for name in self.models}}
        workflow_sets.save_workflow('studio', self.definition, self.root)
        original_member_plan = workflow_sets._member_plan
        def member_plan(member, root):
            plan = original_member_plan(member, root)
            if plan['runtime'] == 'tasks':
                plan['argv'][1] = str(self.launcher)
                plan['fixture_launcher_sha256'] = self.launcher_hash
                plan['preview_hash'] = _preview_hash(plan)
            return plan
        def runtime(repo, path, requested, recipes):
            return requested, sys.executable
        def resolve_model(path, *args, **kwargs):
            identity = capture_identity(Path(path), load_manifests()['mlx-embeddings'], Path(sys.executable))
            return ['classification'], sys.executable, identity
        for context in (patch('mlx_agent.workflow_sets._member_plan', side_effect=member_plan),
                        patch('mlx_agent.workflow_sets.resolve_serve_runtime', side_effect=runtime),
                        patch('mlx_agent.task_serving.resolve_model', side_effect=resolve_model),
                        patch('mlx_agent.service_registry.register_receipts'),
                        patch.dict(os.environ, {'PYTHONPATH': str(self.source)})):
            context.start()
            self.addCleanup(context.stop)
        self.addCleanup(self.cleanup_services)

    def cleanup_services(self):
        for port in self.ports.values():
            try: stop_serve(port, self.root)
            except Exception: pass

    def request(self, port, method, path, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            connection.request(method, path, None if body is None else json.dumps(body))
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def wait(self, port, identifier):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status, value = self.request(port, 'GET', '/v1/tasks/' + identifier)
            if status == 200 and value['state'] in ('succeeded', 'failed', 'cancelled'):
                return value
            time.sleep(.05)
        self.fail('Synthetic task did not settle.')

    def test_resolved_roles_route_typed_jobs_to_their_own_gateway(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        started = workflow_activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        self.assertEqual(started['state'], 'active', started)
        descriptors = {}
        for role in self.definition['members']:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['serve', 'workflow', 'resolve', '--name', 'studio', '--role', role,
                             '--receipts-dir', str(self.root), '--json'])
            self.assertEqual(code, 0, output.getvalue())
            descriptors[role] = json.loads(output.getvalue())['data']
            self.assertEqual(descriptors[role]['origin'], 'http://127.0.0.1:' + str(self.ports[role]))
            self.assertEqual(descriptors[role]['model_state'], 'unloaded')
            self.assertNotIn('control_token', output.getvalue())
        self.assertEqual(descriptors['classifier']['task_kinds'], ['classification'])
        self.assertEqual(descriptors['media']['task_kinds'], ['classification'])
        self.assertEqual(descriptors['coder']['runtime'], 'mlx_lm')
        payload = {'kind': 'classification', 'parameters': {'state': 'test', 'questions': {
            'route': {'type': 'choice', 'instructions': 'Choose one.'}}}}
        status, first = self.request(self.ports['classifier'], 'POST', '/v1/tasks', payload)
        self.assertEqual(status, 202, first)
        status, second = self.request(self.ports['media'], 'POST', '/v1/tasks', payload)
        self.assertEqual(status, 202, second)
        self.assertNotEqual(first['id'], second['id'])
        results = {}
        for role, job in [('classifier', first), ('media', second)]:
            port = self.ports[role]
            outcome = self.wait(port, job['id'])
            self.assertEqual(outcome['state'], 'succeeded', outcome)
            status, result = self.request(port, 'GET', '/v1/tasks/' + job['id'] + '/result')
            self.assertEqual(status, 200, result)
            self.assertEqual(result['result']['answers']['model'], str(self.models[role]))
            results[role] = result['result']
        self.assertEqual(self.request(self.ports['classifier'], 'GET', '/v1/tasks/' + second['id'])[0], 404)
        wait_payload = json.loads(json.dumps(payload))
        wait_payload['parameters']['state'] = 'wait'
        status, job = self.request(self.ports['media'], 'POST', '/v1/tasks', wait_payload)
        self.assertEqual(status, 202, job)
        self.assertEqual(self.request(self.ports['classifier'], 'POST', '/v1/tasks/' + job['id'] + '/cancel', {})[0], 400)
        self.assertEqual(self.request(self.ports['media'], 'POST', '/v1/tasks/' + job['id'] + '/cancel', {})[0], 200)
        cancelled = self.wait(self.ports['media'], job['id'])
        self.assertEqual(cancelled['state'], 'cancelled')
        print(json.dumps({'fixture': 'model-free typed API', 'launcher_sha256': self.launcher_hash,
                          'launcher_source': self.launcher.read_text(), 'request': payload,
                          'job_ids': {'classifier': first['id'], 'media': second['id'], 'cancelled': job['id']},
                          'results': results, 'cancelled_state': cancelled['state']}, sort_keys=True))
        self.assertEqual(workflow_activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')
