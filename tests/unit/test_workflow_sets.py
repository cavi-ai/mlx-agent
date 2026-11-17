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
    def test_forecast_accounts_existing_pending_and_reused_resident(self):
        plan = {'preview_hash':'a'*64, 'runtime':'tasks', 'port':8085, 'local_path':'/model/a',
                'fingerprint':'f'*64, 'worker_argv':['worker'], 'memory_policy':{}}
        candidate = {'pid':123, 'process_started_at':'start'}
        row = {'owner':[123,'gateway identity'], 'worker':{'pid':456,'identity':'worker identity','pgid':456},
               'phase':'resident','model_key':workflows._model_key(plan), 'resident_bytes':40,
               'workspace_bytes':15,'growth_bytes':5,'reserved_bytes':20,'headroom_bytes':10,'telemetry':None}
        inspection = {'state':'ok','errors':[],'entries':{'owned':row},'agent_budget':None}
        state = {'gateway_pid':123,'worker_pid':456,'worker_started_at':'worker-start','model_state':'loaded'}
        with patch('mlx_agent.workflow_sets._private_json', return_value=state), \
             patch('mlx_agent.workflow_sets.process_identity', side_effect=lambda pid: {123:'gateway identity',456:'worker identity'}[pid]), \
             patch('mlx_agent.workflow_sets._pid_start', side_effect=lambda pid: {123:'start',456:'worker-start'}[pid]), \
             patch('mlx_agent.workflow_sets.os.getpgid', return_value=456):
            result = workflows._resource_forecast({'classifier':{'action':'reuse','candidate':candidate,
                'estimate':{'size':60},'plan':plan}}, inspection, 1000, self.root)
        self.assertEqual(result['status'], 'fits_estimate')
        self.assertEqual(result['estimates']['incremental_bytes'], 0)
        self.assertEqual(result['estimates']['required_available_bytes'], 30)
        members = {'classifier':{'action':'reuse','candidate':candidate,'estimate':{'size':60},'plan':plan}}
        with patch('mlx_agent.workflow_sets._private_json', return_value=state), \
             patch('mlx_agent.workflow_sets.process_identity', side_effect=lambda pid: {123:'gateway identity',456:'worker identity'}[pid]), \
             patch('mlx_agent.workflow_sets._pid_start', side_effect=lambda pid: {123:'start',456:'worker-start'}[pid]), \
             patch('mlx_agent.workflow_sets.os.getpgid', return_value=456):
            for field, value in [('workspace_bytes', True), ('growth_bytes', -1), ('phase', 'launching'),
                                 ('model_key', 'changed'), ('execution_bytes', 80)]:
                changed = copy.deepcopy(inspection)
                changed['entries']['owned'][field] = value
                with self.subTest(field=field):
                    self.assertEqual(workflows._resource_forecast(members, changed, 1000, self.root)['status'], 'unknown')
            changed = copy.deepcopy(inspection)
            changed['entries']['second'] = copy.deepcopy(row)
            self.assertEqual(workflows._resource_forecast(members, changed, 1000, self.root)['status'], 'unknown')

    def test_strata_materialized_growth_is_not_reserved_twice(self):
        plan = {'runtime':'strata','port':8085,'local_path':'/model/a','worker_argv':['worker'],
                'fingerprint':'f'*64,'memory_policy':{}}
        row = {'owner':[123,'gateway identity'],'worker':{'pid':456,'identity':'worker identity','pgid':456},
               'phase':'resident','model_key':workflows._model_key(plan),'resident_bytes':40,
               'workspace_bytes':15,'growth_bytes':25,'reserved_bytes':40,'headroom_bytes':0,'telemetry':None}
        budget = {'accounted_bytes':100,'admission_bytes':100,'pending_bytes':10,'resident_bytes':90,
                  'external_bytes':0,'overhead_bytes':0,'observed_bytes':90,'limit_bytes':100,'reserve_bytes':0,
                  'allocations':{'owned':{'resident_bytes':90,'observed_bytes':90,'pending_bytes':10}}}
        inspection = {'state':'ok','errors':[],'entries':{'owned':row},'agent_budget':budget}
        state = {'gateway_pid':123,'worker_pid':456,'worker_started_at':'worker-start','model_state':'loaded'}
        with patch('mlx_agent.workflow_sets._private_json', return_value=state), \
             patch('mlx_agent.workflow_sets.process_identity', side_effect=lambda pid: {123:'gateway identity',456:'worker identity'}[pid]), \
             patch('mlx_agent.workflow_sets._pid_start', side_effect=lambda pid: {123:'start',456:'worker-start'}[pid]), \
             patch('mlx_agent.workflow_sets.os.getpgid', return_value=456):
            result = workflows._resource_forecast({'coder':{'action':'reuse','candidate':{'pid':123,
                'process_started_at':'start'},'estimate':{'size':100},'plan':plan}}, inspection, 1000, self.root)
        self.assertEqual(result['status'], 'fits_estimate')
        self.assertEqual(result['estimates']['incremental_bytes'], 0)

    def test_forecast_unrelated_pending_and_budget_shortfall(self):
        plan = {'runtime':'tasks','port':8085,'memory_policy':{}}
        members = {'classifier':{'action':'start','plan':plan,'estimate':{'size':60}}}
        row = {'owner':[222,'other identity'],'worker':None,'phase':'resident','model_key':'f'*64,
               'resident_bytes':20,'workspace_bytes':20,'growth_bytes':0,'reserved_bytes':20,
               'headroom_bytes':10,'telemetry':None}
        inspection = {'state':'ok','errors':[],'entries':{'other':row},'agent_budget':None}
        fit = workflows._resource_forecast(members, inspection, 100, self.root)
        self.assertEqual(fit['status'], 'fits_estimate')
        self.assertEqual(fit['estimates']['required_available_bytes'], 90)
        self.assertEqual(workflows._resource_forecast(members, inspection, 89, self.root)['status'], 'does_not_fit')
        inspection['agent_budget'] = {'accounted_bytes':50,'admission_bytes':100,'pending_bytes':20,
            'resident_bytes':20,'external_bytes':0,'overhead_bytes':10,'observed_bytes':30,'limit_bytes':100,
            'reserve_bytes':0,'allocations':{'other':{'resident_bytes':20,'pending_bytes':20,'observed_bytes':20}}}
        denied = workflows._resource_forecast(members, inspection, 1000, self.root)
        self.assertEqual(denied['status'], 'does_not_fit')
        self.assertIn('shared budget', denied['reason'])
        denied = workflows._resource_forecast(members, inspection, None, self.root)
        self.assertEqual(denied['status'], 'does_not_fit')
        self.assertIn('shared budget', denied['reason'])
        malformed = copy.deepcopy(inspection)
        malformed['agent_budget'] = None
        malformed['entries']['other'] = {'reserved_bytes':20,'headroom_bytes':10}
        self.assertEqual(workflows._resource_forecast(members, malformed, 1000, self.root)['status'], 'unknown')
        row['reserved_bytes'] = True
        self.assertEqual(workflows._resource_forecast(members, inspection, 1000, self.root)['status'], 'unknown')

    def test_forecast_reuse_missing_state_is_unknown(self):
        plan = {'runtime':'tasks','port':8085,'memory_policy':{}}
        members = {'classifier':{'action':'reuse','candidate':{'pid':123},'plan':plan,'estimate':{'size':60}}}
        inspection = {'state':'ok','errors':[],'entries':{},'agent_budget':None}
        with patch('mlx_agent.workflow_sets._private_json', side_effect=FileNotFoundError('missing')):
            result = workflows._resource_forecast(members, inspection, 1000, self.root)
        self.assertEqual(result['status'], 'unknown')
        self.assertIn('state is unavailable', result['reason'])

    def test_forecast_unloaded_reuse_and_resident_task_cap(self):
        plan = {'runtime':'tasks','port':8085,'local_path':'/model/a','worker_argv':['worker'],
                'fingerprint':'f'*64,'memory_policy':{'model_memory_budget_gb':0.00000005}}
        candidate = {'pid':123,'process_started_at':'start'}
        member = {'classifier':{'action':'reuse','candidate':candidate,'plan':plan,'estimate':{'size':40}}}
        inspection = {'state':'ok','errors':[],'entries':{},'agent_budget':None}
        with patch('mlx_agent.workflow_sets._private_json', return_value={'gateway_pid':123,'worker_pid':None,'model_state':'unloaded'}), \
             patch('mlx_agent.workflow_sets.process_identity', return_value='gateway identity'), \
             patch('mlx_agent.workflow_sets._pid_start', return_value='start'):
            result = workflows._resource_forecast(member, inspection, 1000, self.root)
        self.assertEqual(result['estimates']['incremental_bytes'], 50)
        row = {'owner':[123,'gateway identity'],'worker':{'pid':456,'identity':'worker identity','pgid':456},
               'phase':'resident','model_key':workflows._model_key(plan),'resident_bytes':40,
               'workspace_bytes':15,'growth_bytes':5,'reserved_bytes':20,'headroom_bytes':0,'telemetry':None}
        inspection['entries']={'owned':row}
        state={'gateway_pid':123,'worker_pid':456,'worker_started_at':'worker-start','model_state':'loaded'}
        with patch('mlx_agent.workflow_sets._private_json', return_value=state), \
             patch('mlx_agent.workflow_sets.process_identity', side_effect=lambda pid: {123:'gateway identity',456:'worker identity'}[pid]), \
             patch('mlx_agent.workflow_sets._pid_start', side_effect=lambda pid: {123:'start',456:'worker-start'}[pid]), \
             patch('mlx_agent.workflow_sets.os.getpgid', return_value=456):
            result = workflows._resource_forecast(member, inspection, 1000, self.root)
        self.assertEqual(result['status'], 'does_not_fit')
        self.assertIn('route cap', result['reason'])
        self.assertEqual(result['members']['classifier']['configured_limit_bytes'], 50)
        self.assertEqual(result['members']['classifier']['required_limit_bytes'], 60)
        self.assertIsNotNone(result['estimates']['incremental_bytes'])

    def test_preview_hash_excludes_reused_allocation_changes(self):
        plan = {'preview_hash':'a'*64,'runtime':'tasks','port':8085,'local_path':'/model/a',
                'worker_argv':['worker'],'fingerprint':'f'*64,'memory_policy':{}}
        definition = copy.deepcopy(self.definition)
        definition['members'] = {'classifier':definition['members']['classifier']}
        candidate = {'pid':123,'process_started_at':'start'}
        row = {'owner':[123,'gateway identity'],'worker':{'pid':456,'identity':'worker identity','pgid':456},
               'phase':'resident','model_key':workflows._model_key(plan),'resident_bytes':40,
               'workspace_bytes':15,'growth_bytes':5,'reserved_bytes':20,'headroom_bytes':10,'telemetry':None}
        first = {'state':'ok','errors':[],'entries':{'owned':row},'agent_budget':None}
        second = copy.deepcopy(first)
        second['entries']['owned']['reserved_bytes'] = 40
        state={'gateway_pid':123,'worker_pid':456,'worker_started_at':'worker-start','model_state':'loaded'}
        with patch('mlx_agent.workflow_sets._member_plan', return_value=plan), \
             patch('mlx_agent.workflow_sets.reservation', return_value={'size':60}), \
             patch('mlx_agent.workflow_sets.os.path.lexists', return_value=True), \
             patch('mlx_agent.workflow_sets._reuse_candidate', return_value=candidate), \
             patch('mlx_agent.workflow_sets.MemoryRegistry.inspect', side_effect=[first,second]), \
             patch('mlx_agent.workflow_sets.available_memory_bytes', side_effect=[1000,80]), \
             patch('mlx_agent.workflow_sets._private_json', return_value=state), \
             patch('mlx_agent.workflow_sets.process_identity', side_effect=lambda pid: {123:'gateway identity',456:'worker identity'}[pid]), \
             patch('mlx_agent.workflow_sets._pid_start', side_effect=lambda pid: {123:'start',456:'worker-start'}[pid]), \
             patch('mlx_agent.workflow_sets.os.getpgid', return_value=456):
            left = workflows.preview_workflow(definition, receipts_dir=self.root)
            right = workflows.preview_workflow(definition, receipts_dir=self.root)
        self.assertEqual(left['preview_hash'], right['preview_hash'])
        self.assertNotEqual(left['resource_forecast']['estimates']['required_available_bytes'],
                            right['resource_forecast']['estimates']['required_available_bytes'])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for context in (patch('mlx_agent.model_serving.resolve_model', side_effect=lambda value: value),
                        patch('mlx_agent.model_serving.read_policy', return_value={'duplicate_model_allowlist': []}),
                        patch('mlx_agent.workflow_sets._existing_claim_candidate', return_value=None)):
            context.start()
            self.addCleanup(context.stop)
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

    def test_empty_uncapped_registry_uses_host_headroom_conservatively(self):
        plan = {'preview_hash': 'a' * 64, 'runtime': 'tasks'}
        estimate = {'size': 40, 'reserve_bytes': 20, 'workspace_bytes': 15, 'growth_bytes': 5}
        results = []
        for state, available in [('ok', 100), ('uninitialized', 70), ('unknown', 100)]:
            inspection = {'state': state, 'entries': {}, 'agent_budget': None,
                          'errors': ['unavailable'] if state == 'unknown' else []}
            with patch('mlx_agent.workflow_sets._member_plan', return_value=plan), \
                 patch('mlx_agent.workflow_sets.reservation', return_value=estimate), \
                 patch('mlx_agent.workflow_sets.MemoryRegistry.inspect', return_value=inspection), \
                 patch('mlx_agent.workflow_sets.available_memory_bytes', return_value=available):
                results.append(workflows.preview_workflow(self.definition, receipts_dir=self.root))
        self.assertEqual([r['resource_forecast']['status'] for r in results],
                         ['fits_estimate', 'does_not_fit', 'unknown'])
        self.assertEqual(len({r['preview_hash'] for r in results}), 1)

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
                   'preview_hash': 'a' * 64, 'process_started_at': 'start'}
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

    def test_cold_reuse_requires_recorded_start_and_checks_optional_state(self):
        root = self.root / 'receipts'
        root.mkdir(mode=0o700)
        plan = {'preview_hash': 'a' * 64, 'runtime': 'tasks', 'port': 8085,
                'local_path': '/model/a', 'fingerprint': 'f' * 64,
                'worker_argv': ['worker'], 'memory_policy': {'keep_loaded': True}}
        receipt = {'kind': 'serve', 'schema_version': '1.0', 'pid': 123, 'port': 8085,
                   'runtime': 'tasks', 'jit': True, 'preview_hash': 'a' * 64,
                   'process_started_at': 'start'}
        for name, value in [('8085.json', receipt), ('8085.jit-config.json', plan)]:
            path = root / name
            path.write_text(json.dumps(value))
            path.chmod(0o600)
        with patch('mlx_agent.workflow_sets._pid_alive', return_value=True), \
             patch('mlx_agent.workflow_sets._pid_start', return_value='start'), \
             patch('mlx_agent.workflow_sets._pid_command', return_value='command'), \
             patch('mlx_agent.workflow_sets._argv_matches', return_value=True):
            self.assertEqual(workflows._reuse_candidate(root, plan)['pid'], 123)
            receipt['process_started_at'] = 'different'
            (root / '8085.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                workflows._reuse_candidate(root, plan)
            receipt['process_started_at'] = 'start'
            (root / '8085.json').write_text(json.dumps(receipt))
            state = root / '8085.jit-state.json'
            state.write_text(json.dumps({'gateway_pid': 999}))
            state.chmod(0o600)
            with self.assertRaises(ValueError):
                workflows._reuse_candidate(root, plan)
            state.unlink()
            state.symlink_to(root / '8085.json')
            with self.assertRaises(OSError):
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
