import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from mlx_agent import served, workflow_operations as operations, workflow_sets
from mlx_agent.inference_runner import InferenceError
from mlx_agent.task_contracts import TaskError


class WorkflowOperationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identifier = '8f171a53-d135-4fcb-8a67-c52e3f222c20'
        self.identity = {'model_path': str(self.root), 'model_fingerprint': 'f' * 64,
                         'backend': 'fixture', 'backend_manifest_hash': 'b' * 64,
                         'executable_identity': {'size': 1}}
        self.role = {'activation_id': 'activation', 'workflow_state': 'active', 'role': 'one', 'runtime': 'tasks',
                     'origin': 'http://127.0.0.1:8085', 'model_id': str(self.root), 'task_kinds': ['classification'],
                     'endpoint_identity': {'service_id': 'a' * 64, 'pid': 123,
                                           'process_started_at': 'start', 'member_plan_hash': 'p' * 64},
                     'credential_ref': '/private/key'}
        self.member = {'pid': 123, 'port': 8085, 'process_started_at': 'start',
                       'member_plan_hash': 'p' * 64, 'state': 'active'}
        self.definition = {'schema': 'serving-workflow/1', 'name': 'studio', 'members': {'one': {'profile': 'small'}}}
        self.activation = {'state': 'active', 'activation_id': 'activation', 'members': {'one': self.member},
                           'definition_hash': workflow_sets._hash(self.definition)}
        self.plan = {'preview_hash': 'p' * 64, 'runtime': 'tasks', 'port': 8085,
                     'local_path': str(self.root), 'fingerprint': 'f' * 64,
                     'worker_argv': ['worker'], 'memory_policy': {}}
        self.inspection = {'state': 'uninitialized', 'entries': {}, 'errors': [], 'agent_budget': None}

    def test_stage_subset_reuses_forecast_with_full_pending_registry(self):
        row = {'owner': [222, 'other identity'], 'worker': None, 'phase': 'resident', 'model_key': 'f' * 64,
               'resident_bytes': 20, 'workspace_bytes': 20, 'growth_bytes': 0, 'reserved_bytes': 20,
               'headroom_bytes': 0, 'telemetry': None}
        self.inspection = {'state': 'ok', 'entries': {'other': row}, 'errors': [], 'agent_budget': None}
        state = {'gateway_pid': 123, 'worker_pid': None, 'model_state': 'unloaded'}
        with patch.object(operations.activation, '_read', return_value=self.activation), \
             patch.object(workflow_sets, 'load_workflow', return_value=self.definition), \
             patch.object(workflow_sets, '_member_plan', return_value=self.plan), \
             patch.object(operations, 'load_profile', return_value={'profile': 'stable'}), \
             patch.object(operations.activation, 'resolve_workflow_role', return_value=self.role), \
             patch.object(operations, 'reservation', return_value={'size': 60}), \
             patch.object(operations.MemoryRegistry, 'inspect', return_value=self.inspection), \
             patch.object(operations, 'available_memory_bytes', return_value=100), \
             patch.object(workflow_sets, '_private_json', return_value=state), \
             patch.object(workflow_sets, '_pid_start', return_value='start'), \
             patch.object(workflow_sets, 'process_identity', return_value='gateway identity'):
            result = operations.stage('studio', ['one'], receipts_dir=self.root)
            self.assertEqual(result['resource_forecast']['status'], 'fits_estimate')
            self.assertEqual(result['resource_forecast']['estimates']['baseline_pending_bytes'], 20)
            self.assertEqual(result['resource_forecast']['estimates']['required_available_bytes'], 80)
            whole = {'one': {'action': 'reuse', 'candidate': {'pid': 123, 'process_started_at': 'start'},
                             'plan': self.plan, 'estimate': {'size': 60}},
                     'two': {'action': 'start', 'plan': self.plan, 'estimate': {'size': 60}}}
            self.assertEqual(workflow_sets._resource_forecast(whole, self.inspection, 100, self.root)['status'], 'does_not_fit')
            with self.assertRaises(ValueError): operations.stage('studio', ['one', 'one'], receipts_dir=self.root)
            with self.assertRaises(ValueError): operations.stage('studio', ['missing'], receipts_dir=self.root)

    def test_stage_rejects_changed_generation_or_plan(self):
        changed = dict(self.role, endpoint_identity=dict(self.role['endpoint_identity'], service_id='b' * 64))
        with patch.object(operations.activation, '_read', return_value=self.activation), \
             patch.object(workflow_sets, 'load_workflow', return_value=self.definition), \
             patch.object(workflow_sets, '_member_plan', return_value=self.plan), \
             patch.object(operations, 'load_profile', return_value={'profile': 'stable'}), \
             patch.object(operations, 'reservation', return_value={'size': 60}), \
             patch.object(operations.MemoryRegistry, 'inspect', return_value=self.inspection), \
             patch.object(operations, 'available_memory_bytes', return_value=100), \
             patch.object(workflow_sets, '_resource_forecast', return_value={'status': 'unknown'}), \
             patch.object(operations.activation, 'resolve_workflow_role', side_effect=[self.role, changed]):
            with self.assertRaisesRegex(ValueError, 'changed during'): operations.stage('studio', ['one'], receipts_dir=self.root)
        self.member['member_plan_hash'] = 'changed'
        with patch.object(operations.activation, '_read', return_value=self.activation), \
             patch.object(workflow_sets, 'load_workflow', return_value=self.definition), \
             patch.object(workflow_sets, '_member_plan', return_value=self.plan), \
             patch.object(operations, 'load_profile', return_value={'profile': 'stable'}):
            with self.assertRaisesRegex(ValueError, 'plan changed'): operations.stage('studio', ['one'], receipts_dir=self.root)

    def _handoff_case(self, record_changes=None, result_changes=None, changed_role=False, out=None, expired=False,
                      changed_download=False, initial_role=None, corrupt_result=False):
        result = {'answers': {'refund': True}}
        result.update(result_changes or {})
        raw = json.dumps(result).encode()
        record = {'id': self.identifier, 'state': 'succeeded', 'kind': 'classification',
                  'model_path': str(self.root), 'identity': self.identity, 'completed_at': 'now'}
        record.update(record_changes or {})
        outcome = {'task': record, 'artifact': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}
        if expired:
            response = InferenceError('artifacts_expired', 'Expired.', 'Inspect task.')
        else:
            response = lambda port, key, method, path, **kwargs: io.BytesIO(
                b'data' if path.endswith('/artifacts/output') else
                (raw + b' ' if corrupt_result else raw) if path.endswith('/artifacts/result') else json.dumps(outcome).encode())
        role_after = dict(self.role, endpoint_identity=dict(self.role['endpoint_identity'], service_id='c' * 64))
        roles = [initial_role or self.role, role_after if changed_role else self.role]
        if out is not None:
            roles.append(role_after if changed_download else self.role)
        with patch.object(operations.activation, '_read', return_value=self.activation), \
             patch.object(operations.activation, 'resolve_workflow_role', side_effect=roles), \
             patch.object(workflow_sets, '_private_json', return_value={'local_path': str(self.root), 'task_identity': self.identity}), \
             patch.object(served, '_key', return_value='private-token'), \
             patch.object(served, '_open', side_effect=response):
            return operations.handoff('studio', 'one', self.identifier, receipts_dir=self.root, out=out)

    def test_handoff_returns_verified_structured_result_without_credentials(self):
        result = self._handoff_case()
        self.assertEqual(result['result']['answers'], {'refund': True})
        self.assertEqual(result['task_provenance']['model_fingerprint'], 'f' * 64)
        self.assertEqual(result['endpoint_identity']['service_id'], 'a' * 64)
        self.assertNotIn('private-token', json.dumps(result))
        self.assertNotIn('credential_ref', json.dumps(result))

    def test_handoff_rejects_failed_expired_mismatched_and_changed_tasks(self):
        for changes in ({'state': 'failed'}, {'id': 'other'}, {'kind': 'mesh_generation'},
                        {'identity': dict(self.identity, model_fingerprint='z' * 64)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self._handoff_case(record_changes=changes)
        with self.assertRaises(InferenceError): self._handoff_case(expired=True)
        with self.assertRaisesRegex(ValueError, 'changed during'): self._handoff_case(changed_role=True)
        with self.assertRaisesRegex(ValueError, 'recorded digest'): self._handoff_case(corrupt_result=True)
        new = dict(self.role, activation_id='replacement')
        with self.assertRaisesRegex(ValueError, 'changed before'): self._handoff_case(initial_role=new)
        with self.assertRaises(TaskError): operations.handoff('studio', 'one', '../invalid', receipts_dir=self.root)

    def test_handoff_publishes_only_after_identity_recheck(self):
        output = self.root / 'result.glb'
        reference = {'bytes': 4, 'sha256': hashlib.sha256(b'data').hexdigest()}
        with self.assertRaisesRegex(ValueError, 'changed during artifact'):
            self._handoff_case(result_changes={'output_artifact': reference}, out=str(output), changed_download=True)
        self.assertFalse(output.exists())
        result = self._handoff_case(result_changes={'output_artifact': reference}, out=str(output))
        self.assertEqual(output.read_bytes(), b'data')
        self.assertEqual(result['output_artifact'], reference)
        with self.assertRaises(FileExistsError):
            self._handoff_case(result_changes={'output_artifact': reference}, out=str(output))
        self.assertEqual(output.read_bytes(), b'data')

    def test_download_checks_digest_size_and_exclusive_path(self):
        output = self.root / 'artifact.glb'
        reference = {'bytes': 4, 'sha256': hashlib.sha256(b'data').hexdigest()}
        with patch.object(served, '_open', return_value=io.BytesIO(b'data')):
            served._download(8085, None, self.identifier, output, reference)
        self.assertEqual(output.read_bytes(), b'data')
        with patch.object(served, '_open', return_value=io.BytesIO(b'data')), self.assertRaises(FileExistsError):
            served._download(8085, None, self.identifier, output, reference)
        output.unlink()
        with patch.object(served, '_open', return_value=io.BytesIO(b'excess')), self.assertRaises(InferenceError):
            served._download(8085, None, self.identifier, output, reference)
        self.assertFalse(output.exists())

    def test_cli_routes_stage_and_handoff_arguments(self):
        from mlx_agent.cli import main
        output = io.StringIO()
        with patch.object(operations, 'stage', return_value={'schema': 'serving-workflow-stage/1'}) as stage, \
             redirect_stdout(output):
            code = main(['serve', 'workflow', 'stage', '--name', 'studio', '--role', 'one', '--role', 'two',
                         '--receipts-dir', str(self.root), '--json'])
        self.assertEqual(code, 0, output.getvalue())
        stage.assert_called_once_with('studio', ['one', 'two'], receipts_dir=str(self.root))
        output = io.StringIO()
        with patch.object(operations, 'handoff', return_value={'schema': 'serving-workflow-handoff/1'}) as handoff, \
             redirect_stdout(output):
            code = main(['serve', 'workflow', 'handoff', '--name', 'studio', '--role', 'one', '--task-id',
                         self.identifier, '--out', str(self.root / 'output'), '--receipts-dir', str(self.root), '--json'])
        self.assertEqual(code, 0, output.getvalue())
        handoff.assert_called_once_with('studio', 'one', self.identifier, out=str(self.root / 'output'),
                                        receipts_dir=str(self.root))


if __name__ == '__main__':
    unittest.main()
