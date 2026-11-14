"""Workflow definitions are private, bounded and previewed without activation."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from mlx_agent import workflow_sets as workflows
from mlx_agent.residency import MemoryRegistry


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.definition = {'schema': 'serving-workflow/1', 'name': 'studio', 'members': {
            'classifier': {'path': '/model/a', 'runtime': 'tasks', 'port': 8085, 'profile': 'small'},
            'coder': {'path': '/model/b', 'runtime': 'mlx_lm', 'port': 8086, 'profile': 'small'}}}

    def test_round_trip_and_strict_validation(self):
        workflows.save_workflow('studio', self.definition, self.root)
        self.assertEqual(workflows.load_workflow('studio', self.root), self.definition)
        path = self.root / '.mlx-agent-receipts/serve/workflows/studio.json'
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        for mutation in ('extra', 'port', 'both', 'name'):
            value = copy.deepcopy(self.definition)
            if mutation == 'extra': value['members']['coder']['secret'] = 'secret'
            elif mutation == 'port': value['members']['coder']['port'] = 8085
            elif mutation == 'both': value['members']['coder']['repo'] = 'a/b'
            else: value['name'] = '../bad'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                workflows.validate_workflow(value)

    def test_inspection_missing_and_corrupt_does_not_change_files(self):
        root = self.root / 'memory'
        registry = MemoryRegistry(root)
        self.assertEqual(registry.inspect()['state'], 'uninitialized')
        self.assertFalse(root.exists())
        root.mkdir(mode=0o700)
        path = root / 'memory.json'
        path.write_text('invalid')
        path.chmod(0o600)
        before = {p.name: p.read_bytes() for p in root.iterdir()}
        self.assertEqual(registry.inspect()['state'], 'unknown')
        self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)

    def test_preview_reports_joint_estimate_and_hash_ignores_free_memory(self):
        plans = [{'preview_hash': 'a'*64, 'runtime': 'tasks'}, {'preview_hash': 'b'*64, 'runtime': 'mlx_lm'}]
        estimates = [{'size': 40, 'reserve_bytes': 20, 'workspace_bytes': 15, 'growth_bytes': 5},
                     {'size': 50, 'reserve_bytes': 25, 'workspace_bytes': 20, 'growth_bytes': 5}]
        registry = {'state': 'ok', 'entries': {}, 'agent_budget': {'admission_bytes': 100, 'accounted_bytes': 0}, 'errors': []}
        with patch('mlx_agent.workflow_sets._member_plan', side_effect=plans * 2), \
             patch('mlx_agent.workflow_sets.reservation', side_effect=estimates * 2), \
             patch('mlx_agent.workflow_sets.MemoryRegistry.inspect', return_value=registry), \
             patch('mlx_agent.workflow_sets.available_memory_bytes', side_effect=[1000, 80]):
            first = workflows.preview_workflow(self.definition, receipts_dir=self.root)
            second = workflows.preview_workflow(self.definition, receipts_dir=self.root)
        self.assertEqual(first['resource_forecast']['status'], 'fits_estimate')
        self.assertEqual(second['resource_forecast']['status'], 'does_not_fit')
        self.assertEqual(first['preview_hash'], second['preview_hash'])
        self.assertEqual(first['resource_forecast']['estimates']['incremental_bytes'], 90)

    def test_inspection_rejects_symlinked_root_without_creating_target(self):
        outside = self.root / 'outside'
        outside.mkdir(mode=0o700)
        (self.root / 'linked').symlink_to(outside, target_is_directory=True)
        result = MemoryRegistry(self.root / 'linked' / 'missing').inspect()
        self.assertEqual(result['state'], 'unknown')
        self.assertEqual(list(outside.iterdir()), [])

    def test_existing_allocation_keeps_joint_forecast_unknown(self):
        plan = {'preview_hash': 'a' * 64, 'runtime': 'tasks'}
        estimate = {'size': 40, 'reserve_bytes': 20, 'workspace_bytes': 15, 'growth_bytes': 5}
        inspection = {'state': 'ok', 'entries': {'owned': {}},
                      'agent_budget': {'admission_bytes': 1000, 'accounted_bytes': 40}, 'errors': []}
        with patch('mlx_agent.workflow_sets._member_plan', return_value=plan), \
             patch('mlx_agent.workflow_sets.reservation', return_value=estimate), \
             patch('mlx_agent.workflow_sets.MemoryRegistry.inspect', return_value=inspection), \
             patch('mlx_agent.workflow_sets.available_memory_bytes', return_value=1000):
            result = workflows.preview_workflow(self.definition, receipts_dir=self.root)
        self.assertEqual(result['resource_forecast']['status'], 'unknown')

    def test_real_plan_hash_tracks_profile_and_model_changes(self):
        from mlx_agent.serve_profiles import save_profile
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{"model_type":"qwen2"}')
        weights = model / 'model.safetensors'
        weights.write_bytes(b'weights')
        profile = {'min_headroom_gb': 2, 'idle_timeout': 300, 'keep_loaded': False,
                   'unload_under_pressure': False}
        save_profile('small', profile, self.root)
        member = {'path': str(model), 'runtime': 'mlx_lm', 'port': 8086, 'profile': 'small'}
        first = workflows._member_plan(member, self.root)['preview_hash']
        save_profile('small', dict(profile, idle_timeout=301), self.root, replace=True)
        second = workflows._member_plan(member, self.root)['preview_hash']
        self.assertNotEqual(first, second)
        weights.write_bytes(b'changed-weights')
        third = workflows._member_plan(member, self.root)['preview_hash']
        self.assertNotEqual(second, third)

    def test_reuse_binds_reviewed_identity_without_exposing_control_token(self):
        root = self.root / 'receipts'
        root.mkdir(mode=0o700)
        plan = {'preview_hash': 'a' * 64, 'runtime': 'tasks', 'port': 8085,
                'local_path': '/model/a', 'fingerprint': 'f' * 64,
                'worker_argv': ['worker'], 'memory_policy': {'keep_loaded': True}}
        receipt = {'kind': 'serve', 'schema_version': '1.0', 'pid': 123, 'port': 8085, 'runtime': 'tasks', 'jit': True,
                   'preview_hash': 'a' * 64}
        config = dict(plan, control_token='private-token')
        state = {'gateway_pid': 123, 'worker_pid': 456, 'model_state': 'loaded'}
        for name, value in [('8085.json', receipt), ('8085.jit-config.json', config), ('8085.jit-state.json', state)]:
            path = root / name
            path.write_text(json.dumps(value))
            path.chmod(0o600)
        with patch('mlx_agent.workflow_sets._pid_alive', return_value=True), \
             patch('mlx_agent.workflow_sets._pid_start', return_value='start'), \
             patch('mlx_agent.workflow_sets._pid_command', return_value='command'), \
             patch('mlx_agent.workflow_sets._argv_matches', return_value=True):
            candidate = workflows._reuse_candidate(root, plan)
            self.assertEqual(candidate['pid'], 123)
            self.assertNotIn('private-token', json.dumps(candidate))
            config['memory_policy'] = {'keep_loaded': False}
            (root / '8085.jit-config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                workflows._reuse_candidate(root, plan)

    def test_cli_save_show_and_list(self):
        import io
        from contextlib import redirect_stdout
        from mlx_agent.cli import main
        source = self.root / 'definition.json'
        source.write_text(json.dumps(self.definition))
        for args in (['save', '--name', 'studio', '--file', str(source)],
                     ['show', '--name', 'studio'], ['list']):
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['serve', 'workflow', *args, '--receipts-dir', str(self.root), '--json'])
            self.assertEqual(code, 0, output.getvalue())

    def test_missing_repo_is_blocked_offline(self):
        value = copy.deepcopy(self.definition)
        value['members']['coder'] = {'repo': 'absent/model', 'runtime': 'mlx_lm', 'port': 8086, 'profile': 'small'}
        with patch('mlx_agent.workflow_sets.cached_snapshot', return_value=None):
            with self.assertRaises(ValueError):
                workflows._member_plan(value['members']['coder'], self.root)
