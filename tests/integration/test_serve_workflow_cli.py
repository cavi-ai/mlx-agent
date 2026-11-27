"""Role resolution using isolated model-free typed and chat gateways."""
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import socket
import stat
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from mlx_agent import task_serving, workflow_activation, workflow_sets
from mlx_agent.backends import load_manifests
from mlx_agent.cli import main
from mlx_agent.serve import _preview_hash, load_recipes, plan_start, receipts_root, start_serve, stop_serve
from mlx_agent.serve_profiles import save_profile
from mlx_agent.task_identity import capture_identity


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def resolve_fixture_task_model(path):
    identity = capture_identity(Path(path), load_manifests()['mlx-embeddings'], Path(sys.executable))
    return ['classification'], sys.executable, identity


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
from mlx_agent import task_kinds, task_serving, inference_runner, task_serve, jit_serve
jit_serve.available_memory_bytes = lambda: 0
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
    self.controller.memory_probe = lambda: 32_000_000_000
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
        for context in (patch('mlx_agent.workflow_sets._member_plan', side_effect=member_plan),
                        patch('mlx_agent.workflow_sets.resolve_serve_runtime', side_effect=runtime),
                        patch('mlx_agent.task_serving.resolve_model', side_effect=resolve_fixture_task_model),
                        patch('mlx_agent.service_registry.register_receipts'),
                        patch.dict(os.environ, {'PYTHONPATH': str(self.source), 'XDG_STATE_HOME': str(self.root / 'state')})):
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

    def test_retained_task_workflow_keeps_its_pre_upgrade_plan_hash(self):
        save_profile('retained', {'min_headroom_gb': 2, 'idle_timeout': 0,
                                  'keep_loaded': True, 'unload_under_pressure': False,
                                  'execution_wait_seconds': 0, 'release_after_request': False}, self.root)
        member = dict(self.definition['members']['classifier'], profile='retained')
        plan = workflow_sets._member_plan(member, self.root)
        legacy_policy = {'minimum_headroom_gb': 2, 'idle_timeout_seconds': 0, 'keep_loaded': True,
                         'execution_wait_seconds': 0, 'release_after_request': False}
        self.assertEqual(plan['memory_policy'], legacy_policy)
        self.assertEqual(plan['preview_hash'], _preview_hash(dict(plan, memory_policy=legacy_policy)))

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


class ServePolicyResetIntegrationTests(unittest.TestCase):
    """CLI policy updates against the live text and task gateways with unloaded models."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        os.chmod(self.root, 0o700)
        state_env = patch.dict(os.environ, {'XDG_STATE_HOME': str(self.root / 'state')})
        state_env.start()
        self.addCleanup(state_env.stop)
        self.model = self.root / 'fixture-model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{"model_type": "llama"}')
        (self.model / 'model.safetensors').write_bytes(b'fixture')

    def _start_live_gateway(self, runtime):
        port = free_port()
        recipes = load_recipes()
        recipes['mlx_lm'] = dict(
            recipes['mlx_lm'],
            executable=sys.executable,
            argv=[sys.executable, '-c', 'import time; time.sleep(60)', '--model', '{repo}', '--port', '{port}'],
        )
        # Substitute local backend discovery for the synthetic model; use the real task plan and gateway.
        with patch.object(task_serving, 'resolve_model', side_effect=resolve_fixture_task_model):
            plan = plan_start(None, runtime, recipes, path=str(self.model), port=port, jit=True, receipts_dir=self.root)

        outcome = start_serve(
            plan,
            receipts_dir=str(self.root),
            confirm=True,
            preview_hash=plan['preview_hash'],
        )
        receipt = outcome['receipt']
        self.addCleanup(stop_serve, port, self.root, expected_pid=receipt['pid'])
        self.assertTrue(receipt['jit'])
        self.assertEqual(receipt['runtime'], runtime)
        return port, receipt['pid']

    def _run_cli(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(list(args))
        raw = output.getvalue()
        self.assertNotIn('control_token', raw)
        return code, json.loads(raw)

    def _status_entry(self, port):
        code, payload = self._run_cli('serve', 'status', '--receipts-dir', str(self.root), '--json')
        self.assertEqual(code, 0, payload)
        matches = [item for item in payload['data']['servers'] if item['port'] == port]
        self.assertEqual(len(matches), 1, payload)
        return matches[0]

    def _config_policy(self, port):
        path = receipts_root(str(self.root)) / '{}.jit-config.json'.format(port)
        self.assertFalse(path.is_symlink())
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        return json.loads(path.read_text())['memory_policy']

    def test_live_cli_serve_policy_resets_omitted_wait_and_release_defaults(self):
        port, pid = self._start_live_gateway('mlx_lm')
        base_cmd = ['serve', 'policy', '--port', str(port), '--expected-pid', str(pid),
                    '--receipts-dir', str(self.root), '--json']

        code, res = self._run_cli(*base_cmd, '--release-after-request', '--execution-wait', '30')
        self.assertEqual(code, 0, res)
        expected_release = {
            'idle_timeout_seconds': 0,
            'keep_loaded': False,
            'minimum_headroom_gb': None,
            'execution_wait_seconds': 30,
            'release_after_request': True,
        }
        self.assertEqual(res['data']['memory_policy'], expected_release)
        self.assertEqual(self._status_entry(port)['memory_policy'], expected_release)
        self.assertEqual(self._config_policy(port), expected_release)

        # Direct transition to keep_loaded omits stale release_after_request and execution_wait_seconds.
        code, res = self._run_cli(*base_cmd, '--keep-loaded')
        self.assertEqual(code, 0, res)
        expected_keep = {'idle_timeout_seconds': 0, 'keep_loaded': True, 'minimum_headroom_gb': None}
        self.assertEqual(res['data']['memory_policy'], expected_keep)
        self.assertEqual(self._status_entry(port)['memory_policy'], expected_keep)
        self.assertEqual(self._config_policy(port), expected_keep)

        # Re-enable both, then transition to unload-after-idle without passing wait/release flags.
        code, res = self._run_cli(*base_cmd, '--release-after-request', '--execution-wait', '15')
        self.assertEqual(code, 0, res)
        code, res = self._run_cli(*base_cmd, '--idle-timeout', '120')
        self.assertEqual(code, 0, res)
        expected_idle = {'idle_timeout_seconds': 120, 'keep_loaded': False, 'minimum_headroom_gb': None}
        self.assertEqual(res['data']['memory_policy'], expected_idle)
        self.assertEqual(self._status_entry(port)['memory_policy'], expected_idle)
        self.assertEqual(self._config_policy(port), expected_idle)

    def test_live_cli_serve_policy_preserves_budget_until_explicitly_cleared(self):
        port, pid = self._start_live_gateway('tasks')
        self.assertTrue(self._status_entry(port)['allocator_cache']['supported'])
        base_cmd = ['serve', 'policy', '--port', str(port), '--expected-pid', str(pid),
                    '--receipts-dir', str(self.root), '--json']

        code, res = self._run_cli(
            *base_cmd, '--memory-budget-gb', '8', '--min-headroom-gb', '2',
            '--release-after-request', '--execution-wait', '25',
        )
        self.assertEqual(code, 0, res)
        expected_initial = {
            'idle_timeout_seconds': 0,
            'keep_loaded': False,
            'minimum_headroom_gb': 2.0,
            'model_memory_budget_gb': 8.0,
            'execution_wait_seconds': 25,
            'release_after_request': True,
        }
        self.assertEqual(self._status_entry(port)['memory_policy'], expected_initial)
        self.assertEqual(self._config_policy(port), expected_initial)

        # Replacing with keep_loaded clears wait/release defaults while keeping model_memory_budget_gb.
        code, res = self._run_cli(*base_cmd, '--keep-loaded', '--min-headroom-gb', '2')
        self.assertEqual(code, 0, res)
        expected_kept_budget = {
            'idle_timeout_seconds': 0,
            'keep_loaded': True,
            'minimum_headroom_gb': 2.0,
            'model_memory_budget_gb': 8.0,
        }
        self.assertEqual(res['data']['memory_policy'], expected_kept_budget)
        self.assertEqual(self._status_entry(port)['memory_policy'], expected_kept_budget)
        self.assertEqual(self._config_policy(port), expected_kept_budget)

        # Explicit --clear-memory-budget removes model_memory_budget_gb cleanly.
        code, res = self._run_cli(*base_cmd, '--idle-timeout', '90', '--clear-memory-budget')
        self.assertEqual(code, 0, res)
        expected_cleared = {'idle_timeout_seconds': 90, 'keep_loaded': False, 'minimum_headroom_gb': None}
        self.assertEqual(res['data']['memory_policy'], expected_cleared)
        self.assertEqual(self._status_entry(port)['memory_policy'], expected_cleared)
        self.assertEqual(self._config_policy(port), expected_cleared)
