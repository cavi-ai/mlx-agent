import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from mlx_agent import endpoint_contract, model_serving


class EndpointContractTests(unittest.TestCase):
    def test_claim_identity_matches_duplicate_error_and_changes_with_generation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            receipt_path = root / '9000.json'
            receipt = {'pid': 42, 'port': 9000, 'runtime': 'tasks', 'process_started_at': 'birth',
                       'model_artifact': str(root), 'jit': False}
            item = {'receipt': str(receipt_path), 'receipts_dir': str(root), 'alive': True,
                    'argv_match': True, 'process_started_at': 'birth', 'pid': 42,
                    'port': 9000, 'runtime': 'tasks'}
            row = {'generation': 'a' * 64, 'pid': 42, 'birth': 'birth', 'kind': 'service',
                   'ready': True, 'receipt_path': str(receipt_path),
                   'service': {'port': 9000, 'runtime': 'tasks', 'auth_required': True}}
            with patch('mlx_agent.serve._read_receipt', return_value=receipt), \
                 patch('mlx_agent.model_serving.resolve_model', return_value=str(root)), \
                 patch('mlx_agent.model_serving.state_root', return_value=root / 'state'), \
                 patch('mlx_agent.model_serving._read_claims', return_value=[row]), \
                 patch('mlx_agent.model_serving._verify_ready_service'):
                endpoint = model_serving.resolve_receipt_endpoint(item)
                self.assertEqual(endpoint, model_serving._service_info(row))
                identity = endpoint['service_id']
                self.assertEqual(identity, hashlib.sha256(('a' * 64).encode()).hexdigest())
                row['retiring'] = True
                self.assertEqual(model_serving.resolve_receipt_endpoint(item)['state'], 'unknown')
                row.pop('retiring')
                row['generation'] = 'b' * 64
                self.assertNotEqual(identity, model_serving.resolve_receipt_endpoint(item)['service_id'])
                item['process_started_at'] = 'other'
                self.assertIsNone(model_serving.resolve_receipt_endpoint(item))

    def test_task_capabilities_include_classification_only_when_supported(self):
        operation = {'method': 'POST', 'path': '/v1/systemone'}
        self.assertIn(operation, endpoint_contract.operations('tasks', ['classification']))
        self.assertNotIn(operation, endpoint_contract.operations('tasks', ['embedding']))

    def test_attach_and_retire_are_serialized(self):
        from concurrent.futures import ThreadPoolExecutor
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'claims'
            model_serving._state_dir(root)
            key, generation = 'a' * 64, 'b' * 64
            owner = '1' * 36 + ':owner'
            borrower = '2' * 36 + ':borrower'
            row = {'generation': generation, 'pid': os.getpid(), 'birth': 'birth', 'state': 'active',
                   'kind': 'service', 'ready': True, 'service': {'port': 9000, 'runtime': 'tasks',
                   'auth_required': False}, 'workflow_refs': [owner]}
            model_serving._write_claims(root, key, [row])
            claim = model_serving.Claim(root, key, generation)
            barrier = threading.Barrier(2)
            def act(reference, action):
                barrier.wait()
                try:
                    return claim.workflow_reference(reference, action)
                except model_serving.ModelServingError:
                    return False
            with patch('mlx_agent.model_serving._live', return_value=True), \
                 patch('mlx_agent.model_serving._verify_ready_service'), ThreadPoolExecutor(max_workers=2) as pool:
                attach = pool.submit(act, borrower, 'attach')
                retire = pool.submit(act, owner, 'retire')
                outcomes = (attach.result(), retire.result())
            self.assertIn(outcomes, ((True, False), (False, True)))
            result = model_serving._read_claims(root, key)[0]
            self.assertFalse(result.get('retiring', False) and borrower in result['workflow_refs'])

    def test_reference_limit_keeps_claim_readable(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'claims'
            model_serving._state_dir(root)
            key, generation = 'a' * 64, 'b' * 64
            refs = [f'{index:036x}:role' for index in range(256)]
            row = {'generation': generation, 'pid': os.getpid(), 'birth': 'birth', 'state': 'active',
                   'kind': 'service', 'ready': True, 'service': {'port': 9000, 'runtime': 'tasks',
                   'auth_required': False}, 'workflow_refs': refs}
            model_serving._write_claims(root, key, [row])
            claim = model_serving.Claim(root, key, generation)
            with patch('mlx_agent.model_serving._live', return_value=True), \
                 patch('mlx_agent.model_serving._verify_ready_service'), self.assertRaises(ValueError):
                claim.workflow_reference(f'{256:036x}:role', 'attach')
            self.assertEqual(model_serving._read_claims(root, key)[0]['workflow_refs'], refs)

    def test_absent_generation_detach_requires_dead_process_and_group(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'claims'
            claim = model_serving.Claim(root, 'a' * 64, 'b' * 64)
            with patch('mlx_agent.compute.process_alive', return_value=False), \
                 patch('mlx_agent.compute.process_birth_identity', return_value=None), \
                 patch('mlx_agent.compute.group_members', return_value=[]):
                self.assertTrue(claim.workflow_reference('1' * 36 + ':role', 'detach', (42, 'birth')))
            with patch('mlx_agent.compute.process_alive', return_value=True), self.assertRaises(model_serving.ModelServingError):
                claim.workflow_reference('1' * 36 + ':role', 'detach', (42, 'birth'))


if __name__ == '__main__':
    unittest.main()
