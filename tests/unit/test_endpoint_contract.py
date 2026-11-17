import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
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
                identity = model_serving.receipt_service_id(item)
                self.assertEqual(identity, model_serving._service_info(row)['service_id'])
                self.assertEqual(identity, hashlib.sha256(('a' * 64).encode()).hexdigest())
                row['generation'] = 'b' * 64
                self.assertNotEqual(identity, model_serving.receipt_service_id(item))
                item['process_started_at'] = 'other'
                self.assertIsNone(model_serving.receipt_service_id(item))

    def test_task_capabilities_include_classification_only_when_supported(self):
        operation = {'method': 'POST', 'path': '/v1/systemone'}
        self.assertIn(operation, endpoint_contract.operations('tasks', ['classification']))
        self.assertNotIn(operation, endpoint_contract.operations('tasks', ['embedding']))


if __name__ == '__main__':
    unittest.main()
