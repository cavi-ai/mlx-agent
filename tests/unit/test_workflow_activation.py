"""Workflow lifecycle owns only its identity-bound gateway processes."""
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from mlx_agent import workflow_activation as activation
from mlx_agent import workflow_sets
from mlx_agent.serve import start_serve, stop_serve
from mlx_agent.serve_profiles import save_profile


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class WorkflowActivationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{"model_type":"qwen2"}')
        (model / 'model.safetensors').write_bytes(b'fixture')
        save_profile('small', {'min_headroom_gb': 2, 'idle_timeout': 0,
                               'keep_loaded': True, 'unload_under_pressure': False}, self.root)
        self.port = free_port()
        self.definition = {'schema': 'serving-workflow/1', 'name': 'studio', 'members': {
            'coding': {'path': str(model), 'runtime': 'mlx_lm', 'port': self.port, 'profile': 'small'}}}
        workflow_sets.save_workflow('studio', self.definition, self.root)
        for context in (patch('mlx_agent.workflow_sets.resolve_serve_runtime', return_value=('mlx_lm', sys.executable)),
                        patch('mlx_agent.service_registry.register_receipts'),
                        patch('mlx_agent.service_registry.discover_services', return_value={'services': [], 'errors': []}),
                        patch.dict(os.environ, {'PYTHONPATH': str(Path(__file__).resolve().parents[2] / 'src'),
                                                'XDG_STATE_HOME': str(self.root / 'state')})):
            context.start()
            self.addCleanup(context.stop)
        self.addCleanup(self.cleanup_gateway)

    def cleanup_gateway(self):
        self.cleanup_port(self.port)

    def cleanup_port(self, port):
        try:
            stop_serve(port, self.root)
        except Exception:
            pass

    def test_resolve_chat_role_uses_live_owned_endpoint(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        running = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        resolved = activation.resolve_workflow_role('studio', 'coding', receipts_dir=self.root)
        self.assertEqual(resolved['schema'], 'serving-workflow-role/1')
        self.assertEqual(resolved['activation_id'], running['activation_id'])
        self.assertEqual(resolved['origin'], 'http://127.0.0.1:' + str(self.port))
        self.assertEqual(resolved['base_url'], resolved['origin'] + '/v1')
        self.assertEqual(resolved['runtime'], 'mlx_lm')
        from mlx_agent.model_serving import resolve_receipt_endpoint
        service = workflow_sets._private_json(self.root / '.mlx-agent-receipts/serve' / (str(self.port) + '.json'))
        endpoint = resolve_receipt_endpoint({'receipt': str(self.root / '.mlx-agent-receipts/serve' / (str(self.port) + '.json')),
            'receipts_dir': str(self.root), 'alive': True, 'argv_match': True, 'pid': service['pid'],
            'port': self.port, 'runtime': 'mlx_lm', 'process_started_at': service['process_started_at']})
        self.assertEqual(resolved['endpoint_identity']['service_id'], endpoint['service_id'])
        self.assertEqual(resolved['model_id'], str(self.root / 'model'))
        self.assertIn({'method': 'POST', 'path': '/v1/chat/completions'}, resolved['operations'])
        self.assertIsNone(resolved['credential_ref'])
        self.assertNotIn('control_token', str(resolved))
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')
        with self.assertRaises(ValueError):
            activation.resolve_workflow_role('studio', 'coding', receipts_dir=self.root)

    def test_resolve_retains_normalized_credential_reference_across_cwd(self):
        import json
        key = self.root / 'key.txt'
        key.write_text('x' * 48)
        key.chmod(0o600)
        self.definition['members']['coding']['api_key_file'] = 'key.txt'
        workflow_sets._path('studio', self.root).write_text(json.dumps(self.definition))
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
            started = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
            self.assertEqual(started['state'], 'active', started)
            os.chdir('/')
            descriptor = activation.resolve_workflow_role('studio', 'coding', receipts_dir=self.root)
            self.assertEqual(descriptor['credential_ref'], str(key))
            self.assertNotIn('x' * 48, json.dumps(descriptor))
            self.assertNotIn('control_token', json.dumps(descriptor))
            retained = activation._path('studio', self.root).read_text()
            self.assertNotIn('x' * 48, retained)
            self.assertNotIn('control_token', retained)
        finally:
            os.chdir(previous)
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_resolve_active_member_of_partial_activation(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        path = activation._path('studio', self.root)
        record = activation._read(path)
        record['state'] = 'partial'
        activation._write(path, record)
        result = activation.resolve_workflow_role('studio', 'coding', receipts_dir=self.root)
        self.assertEqual(result['workflow_state'], 'partial')
        with patch('mlx_agent.workflow_activation._jit_control', side_effect=RuntimeError('unavailable')):
            with self.assertRaises(RuntimeError):
                activation.resolve_workflow_role('studio', 'coding', receipts_dir=self.root)
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_resolve_rejects_changed_identity_and_missing_role(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        with self.assertRaises(ValueError):
            activation.resolve_workflow_role('studio', 'other', receipts_dir=self.root)
        path = activation._path('studio', self.root)
        record = activation._read(path)
        record['members']['coding']['process_started_at'] = 'wrong'
        activation._write(path, record)
        with self.assertRaises(ValueError):
            activation.resolve_workflow_role('studio', 'coding', receipts_dir=self.root)

    def test_lazy_activation_status_and_owned_stop(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        result = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        self.assertEqual(result['state'], 'active')
        self.assertEqual(result['members']['coding']['ownership'], 'created')
        status = activation.status_workflow('studio', receipts_dir=self.root)
        self.assertEqual(status['members']['coding']['endpoint_state'], 'available')
        self.assertEqual(status['members']['coding']['model_state'], 'unloaded')
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_repeat_start_keeps_created_ownership_and_activation_identity(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        first = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        again = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        self.assertEqual(again['activation_id'], first['activation_id'])
        self.assertEqual(again['members']['coding']['ownership'], 'created')
        self.assertEqual(again['members']['coding']['pid'], first['members']['coding']['pid'])
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_repeat_start_accepts_fresh_preview_without_changing_ownership(self):
        original_preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        first = activation.start_workflow('studio', original_preview['preview_hash'], receipts_dir=self.root)
        fresh = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        self.assertNotEqual(original_preview['preview_hash'], fresh['preview_hash'])
        with self.assertRaises(ValueError):
            activation.start_workflow('studio', '0' * 64, receipts_dir=self.root)
        again = activation.start_workflow('studio', fresh['preview_hash'], receipts_dir=self.root)
        self.assertEqual(again['activation_id'], first['activation_id'])
        self.assertEqual(again['members']['coding']['pid'], first['members']['coding']['pid'])
        self.assertEqual(again['members']['coding']['ownership'], 'created')
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_stop_owned_gateway_when_control_transport_unavailable(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        result = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        pid = result['members']['coding']['pid']
        with patch('mlx_agent.workflow_activation._jit_control', side_effect=RuntimeError('control unavailable')):
            self.assertEqual(activation.status_workflow('studio', receipts_dir=self.root)['state'], 'partial')
            stopped = activation.stop_workflow('studio', receipts_dir=self.root)
        self.assertEqual(stopped['state'], 'stopped')
        self.assertFalse(activation._pid_alive(pid))

    def test_status_and_stop_survive_deleted_definition_and_profile(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        result = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        self.assertEqual(result['state'], 'active')
        workflow_sets._path('studio', self.root).unlink()
        (self.root / '.mlx-agent-receipts' / 'serve' / 'profiles' / 'small.json').unlink()
        self.assertEqual(activation.status_workflow('studio', receipts_dir=self.root)['members']['coding']['endpoint_state'], 'available')
        self.assertEqual(activation.resolve_workflow_role('studio', 'coding', receipts_dir=self.root)['runtime'], 'mlx_lm')
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_reused_cold_gateway_survives_workflow_stop(self):
        member = self.definition['members']['coding']
        plan = workflow_sets._member_plan(member, self.root)
        original = start_serve(plan, receipts_dir=self.root, confirm=True, preview_hash=plan['preview_hash'])['receipt']
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        self.assertEqual(preview['members']['coding']['action'], 'reuse')
        result = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        self.assertEqual(result['members']['coding']['ownership'], 'reused')
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')
        self.assertTrue(activation._pid_alive(original['pid']))

    def test_owner_defers_until_reusing_workflow_stops(self):
        first = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        owner = activation.start_workflow('studio', first['preview_hash'], receipts_dir=self.root)
        other = {'schema': 'serving-workflow/1', 'name': 'other', 'members': self.definition['members']}
        workflow_sets.save_workflow('other', other, self.root)
        preview = workflow_sets.preview_workflow(other, receipts_dir=self.root)
        self.assertEqual(preview['members']['coding']['action'], 'reuse')
        borrower = activation.start_workflow('other', preview['preview_hash'], receipts_dir=self.root)
        self.assertEqual(borrower['state'], 'active')
        deferred = activation.stop_workflow('studio', receipts_dir=self.root)
        self.assertEqual(deferred['state'], 'partial')
        self.assertEqual(deferred['members']['coding']['error_code'], 'service_referenced')
        self.assertTrue(activation._pid_alive(owner['members']['coding']['pid']))
        self.assertEqual(activation.stop_workflow('other', receipts_dir=self.root)['state'], 'stopped')
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_cross_root_binding_requires_opt_in(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        owner = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        other_root = self.root / 'other-root'
        other_root.mkdir()
        save_profile('small', {'min_headroom_gb': 2, 'idle_timeout': 0,
                               'keep_loaded': True, 'unload_under_pressure': False}, other_root)
        member = dict(self.definition['members']['coding'], port=free_port())
        other = {'schema': 'serving-workflow/1', 'name': 'other', 'members': {'coding': member}}
        blocked = workflow_sets.preview_workflow(other, receipts_dir=other_root)
        self.assertEqual(blocked['members']['coding']['action'], 'blocked')
        member['reuse_existing'] = True
        workflow_sets.save_workflow('other', other, other_root)
        reviewed = workflow_sets.preview_workflow(other, receipts_dir=other_root)
        self.assertEqual(reviewed['members']['coding']['action'], 'reuse', reviewed)
        borrower = activation.start_workflow('other', reviewed['preview_hash'], receipts_dir=other_root)
        self.assertEqual(borrower['state'], 'active', borrower)
        self.assertEqual(borrower['members']['coding']['pid'], owner['members']['coding']['pid'])
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'partial')
        self.assertEqual(activation.stop_workflow('other', receipts_dir=other_root)['state'], 'stopped')
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_failed_owner_stop_remains_retryable(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        with patch('mlx_agent.workflow_activation.stop_serve', side_effect=RuntimeError('stop failed')):
            failed = activation.stop_workflow('studio', receipts_dir=self.root)
        self.assertEqual(failed['state'], 'partial')
        self.assertIn('stop failed', failed['members']['coding']['error'])
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_interrupted_stop_settles_after_process_has_exited(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        write = activation._write
        def interrupt(path, value):
            if value['members']['coding']['state'] == 'stopped':
                raise OSError('interrupted after stop')
            return write(path, value)
        with patch('mlx_agent.workflow_activation._write', side_effect=interrupt), self.assertRaises(OSError):
            activation.stop_workflow('studio', receipts_dir=self.root)
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_old_created_activation_recovers_current_claim(self):
        from mlx_agent.model_serving import Claim, state_root, _lock, _read_claims, _write_claims
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        path = activation._path('studio', self.root)
        record = activation._read(path)
        token = record['members']['coding'].pop('claim')
        record['members']['coding'].pop('config_identity')
        activation._write(path, record)
        claim = Claim(state_root(), token['key'], token['generation'])
        with _lock(claim.root, claim.key):
            rows = _read_claims(claim.root, claim.key)
            rows[0].pop('workflow_refs', None)
            _write_claims(claim.root, claim.key, rows)
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_allowlisted_endpoint_still_honors_opt_in_reuse(self):
        from mlx_agent.model_serving import write_policy
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        owner = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        write_policy([self.definition['members']['coding']['path']])
        member = dict(self.definition['members']['coding'], port=free_port(), reuse_existing=True)
        other = {'schema': 'serving-workflow/1', 'name': 'other', 'members': {'coding': member}}
        workflow_sets.save_workflow('other', other, self.root)
        reviewed = workflow_sets.preview_workflow(other, receipts_dir=self.root)
        self.assertEqual(reviewed['members']['coding']['action'], 'reuse', reviewed)
        borrower = activation.start_workflow('other', reviewed['preview_hash'], receipts_dir=self.root)
        self.assertEqual(borrower['members']['coding']['pid'], owner['members']['coding']['pid'])
        self.assertEqual(activation.stop_workflow('other', receipts_dir=self.root)['state'], 'stopped')
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_same_root_different_port_reuses_only_with_opt_in(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        owner = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        member = dict(self.definition['members']['coding'], port=free_port(), reuse_existing=True)
        other = {'schema': 'serving-workflow/1', 'name': 'other', 'members': {'coding': member}}
        workflow_sets.save_workflow('other', other, self.root)
        reviewed = workflow_sets.preview_workflow(other, receipts_dir=self.root)
        self.assertEqual(reviewed['members']['coding']['action'], 'reuse', reviewed)
        borrower = activation.start_workflow('other', reviewed['preview_hash'], receipts_dir=self.root)
        self.assertEqual(borrower['state'], 'active', borrower)
        self.assertEqual(borrower['members']['coding']['pid'], owner['members']['coding']['pid'])
        again = activation.start_workflow('other', reviewed['preview_hash'], receipts_dir=self.root)
        self.assertEqual(again['activation_id'], borrower['activation_id'])
        self.assertEqual(activation.stop_workflow('other', receipts_dir=self.root)['state'], 'stopped')
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_operator_stop_allows_borrower_to_detach(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        other = {'schema': 'serving-workflow/1', 'name': 'other', 'members': self.definition['members']}
        workflow_sets.save_workflow('other', other, self.root)
        reviewed = workflow_sets.preview_workflow(other, receipts_dir=self.root)
        activation.start_workflow('other', reviewed['preview_hash'], receipts_dir=self.root)
        self.assertEqual(stop_serve(self.port, self.root)['status'], 'stopped')
        self.assertEqual(activation.stop_workflow('other', receipts_dir=self.root)['state'], 'stopped')

    def test_existing_endpoint_rejects_different_policy_and_credential(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        save_profile('other', {'min_headroom_gb': 3, 'idle_timeout': 0,
                               'keep_loaded': True, 'unload_under_pressure': False}, self.root)
        key = self.root / 'service.key'
        key.write_text('x' * 48)
        key.chmod(0o600)
        base = dict(self.definition['members']['coding'], port=free_port(), reuse_existing=True)
        for override in ({'profile': 'other'}, {'api_key_file': str(key)}, {'runtime': 'mlx-vlm'}):
            member = dict(base, **override)
            other = {'schema': 'serving-workflow/1', 'name': 'other', 'members': {'coding': member}}
            with self.subTest(override=override), patch('mlx_agent.workflow_sets.resolve_serve_runtime',
                    return_value=(member['runtime'], sys.executable)):
                reviewed = workflow_sets.preview_workflow(other, receipts_dir=self.root)
                self.assertEqual(reviewed['members']['coding']['action'], 'blocked')
                self.assertFalse((self.root / '.mlx-agent-receipts/serve' / (str(member['port']) + '.json')).exists())
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_role_rejects_changed_service_configuration(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        path = self.root / '.mlx-agent-receipts/serve' / (str(self.port) + '.jit-config.json')
        original = path.read_bytes()
        try:
            import json
            config = json.loads(original)
            config['max_tokens'] += 1
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                activation.resolve_workflow_role('studio', 'coding', receipts_dir=self.root)
        finally:
            path.write_bytes(original)
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_changed_start_identity_never_signals_gateway(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        result = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        pid = result['members']['coding']['pid']
        path = activation._path('studio', self.root)
        record = activation._read(path)
        record['members']['coding']['process_started_at'] = 'wrong start'
        activation._write(path, record)
        stopped = activation.stop_workflow('studio', receipts_dir=self.root)
        self.assertEqual(stopped['state'], 'partial')
        self.assertIn('identity changed', stopped['members']['coding']['error'])
        self.assertTrue(activation._pid_alive(pid))

    def test_duplicate_member_reports_existing_endpoint_without_launching_second(self):
        import json
        second_port = free_port()
        while second_port == self.port:
            second_port = free_port()
        self.definition['members']['z_media'] = dict(self.definition['members']['coding'], port=second_port)
        workflow_sets._path('studio', self.root).write_text(json.dumps(self.definition))
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        self.assertEqual(preview['members']['z_media']['action'], 'blocked')
        with self.assertRaises(ValueError):
            activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        self.assertFalse((self.root / '.mlx-agent-receipts/serve' / (str(second_port) + '.json')).exists())
        self.assertFalse((self.root / '.mlx-agent-receipts/serve' / (str(self.port) + '.json')).exists())

    def test_racing_port_after_first_launch_retains_owned_endpoint(self):
        import json
        from mlx_agent import serve
        second_port = free_port()
        while second_port == self.port:
            second_port = free_port()
        self.definition['members']['z_media'] = dict(self.definition['members']['coding'], port=second_port)
        from mlx_agent.model_serving import write_policy
        write_policy([self.definition['members']['coding']['path']])
        path = workflow_sets._path('studio', self.root)
        path.write_text(json.dumps(self.definition))
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        blocker = socket.socket()
        self.addCleanup(blocker.close)
        real_start = serve.start_serve
        def start_then_occupy(plan, **kwargs):
            outcome = real_start(plan, **kwargs)
            blocker.bind(('127.0.0.1', second_port))
            blocker.listen()
            return outcome
        with patch('mlx_agent.workflow_activation.start_serve', side_effect=start_then_occupy):
            result = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        self.assertEqual(result['state'], 'partial')
        self.assertIn('cleanup_required', result['members']['coding'])
        self.assertEqual(result['members']['z_media']['state'], 'launch_failed')
        status = activation.status_workflow('studio', receipts_dir=self.root)
        self.assertEqual(status['members']['coding']['endpoint_state'], 'available')
        stopped = activation.stop_workflow('studio', receipts_dir=self.root)
        self.assertEqual(stopped['state'], 'stopped')

    def test_partial_failure_keeps_reused_gateway_and_cleans_only_created(self):
        import json
        from mlx_agent import serve
        owned_port, race_port = free_port(), free_port()
        while len({self.port, owned_port, race_port}) != 3:
            owned_port, race_port = free_port(), free_port()
        self.addCleanup(self.cleanup_port, owned_port)
        original_member = self.definition['members']['coding']
        from mlx_agent.model_serving import write_policy
        write_policy([original_member['path']])
        plan = workflow_sets._member_plan(original_member, self.root)
        borrowed = start_serve(plan, receipts_dir=self.root, confirm=True, preview_hash=plan['preview_hash'])['receipt']
        self.definition['members'] = {'a_borrowed': original_member,
            'b_owned': dict(original_member, port=owned_port), 'z_race': dict(original_member, port=race_port)}
        workflow_sets._path('studio', self.root).write_text(json.dumps(self.definition))
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        self.assertEqual(preview['members']['a_borrowed']['action'], 'reuse')
        blocker = socket.socket()
        self.addCleanup(blocker.close)
        def launch_and_block(plan, **kwargs):
            result = serve.start_serve(plan, **kwargs)
            blocker.bind(('127.0.0.1', race_port))
            blocker.listen()
            return result
        with patch('mlx_agent.workflow_activation.start_serve', side_effect=launch_and_block):
            result = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        self.assertEqual(result['state'], 'partial')
        self.assertEqual(result['members']['a_borrowed']['ownership'], 'reused')
        self.assertEqual(result['members']['b_owned']['ownership'], 'created')
        owned_pid = result['members']['b_owned']['pid']
        self.assertEqual(activation.status_workflow('studio', receipts_dir=self.root)['members']['b_owned']['endpoint_state'], 'available')
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')
        self.assertTrue(activation._pid_alive(borrowed['pid']))
        self.assertFalse(activation._pid_alive(owned_pid))

    def test_repeat_start_rejects_changed_model(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        original = activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        (self.root / 'model' / 'model.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        self.assertEqual(activation._read(activation._path('studio', self.root))['activation_id'], original['activation_id'])
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_competing_invocations_do_not_double_launch(self):
        import threading
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        gate = threading.Barrier(2)
        results = []
        def run():
            gate.wait()
            try:
                results.append(activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)['state'])
            except (ValueError, OSError) as error:
                results.append(type(error).__name__)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(20)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(results.count('active'), 1, results)
        self.assertEqual(len(results), 2)
        self.assertEqual(activation.stop_workflow('studio', receipts_dir=self.root)['state'], 'stopped')

    def test_cli_start_status_and_stop(self):
        import io
        from contextlib import redirect_stdout
        from mlx_agent.cli import main
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        for args in (['start', '--name', 'studio', '--confirm', '--preview-hash', preview['preview_hash']],
                     ['status', '--name', 'studio'], ['stop', '--name', 'studio']):
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['serve', 'workflow', *args, '--receipts-dir', str(self.root), '--json'])
            self.assertEqual(code, 0, output.getvalue())

    def test_occupied_port_is_rejected_before_any_launch(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        with socket.socket() as blocker:
            blocker.bind(('127.0.0.1', self.port))
            blocker.listen()
            with self.assertRaises(ValueError):
                activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        self.assertFalse(activation._path('studio', self.root).exists())

    def test_reviewed_plan_stale_does_not_launch(self):
        preview = workflow_sets.preview_workflow(self.definition, receipts_dir=self.root)
        (self.root / 'model' / 'model.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            activation.start_workflow('studio', preview['preview_hash'], receipts_dir=self.root)
        self.assertFalse((self.root / '.mlx-agent-receipts' / 'serve' / (str(self.port) + '.json')).exists())
