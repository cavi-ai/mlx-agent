import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlx_agent import embed
from mlx_agent.task_contracts import validate_request, TaskError


class EmbedTests(unittest.TestCase):
    def test_typed_request_requires_bounded_nonempty_text_batch(self):
        request = {'schema': 'task-request/2', 'kind': 'embedding', 'model_path': '/models/local',
                   'parameters': {'input': ['hello', 'world']}}
        self.assertEqual(validate_request(request), request)
        for value in ([], [''], 'hello', [True], ['a'] * 129):
            with self.subTest(value=repr(value)[:60]), self.assertRaises(TaskError):
                validate_request(dict(request, parameters={'input': value}))

    def test_embedding_result_rejects_nonfinite_and_wrong_batch(self):
        for vectors in ([[float('nan')]], [[1], [2]], [[]], [[1, 2], [3]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                embed.validate_vectors(vectors, 1)
        self.assertEqual(embed.validate_vectors([[1, 2.0]], 1), [[1.0, 2.0]])

    def test_hyphenated_embedding_architecture_matches_the_backend(self):
        from mlx_agent.backends import load_manifests, with_ports
        from mlx_agent.task_serving import resolve_model
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory).resolve()
            (model / 'config.json').write_text('{"model_type":"xlm-roberta"}')
            request = model / 'request.json'
            request.write_text('{"input":["hello"]}')
            manifest = load_manifests()['mlx-embeddings']
            manifests = {'mlx-embeddings': manifest}
            registries = {'mlx-embeddings': {'installed': True, 'registry': with_ports(manifest['registry'], manifest)}}
            with patch('mlx_agent.embed.is_installed', return_value=True):
                plan = embed.plan_embed(str(model), str(request), manifests=manifests, registries=registries)
            self.assertEqual(plan['backend'], 'mlx-embeddings')
            self.assertEqual(plan['count'], 1)
            self.assertEqual(resolve_model(model, manifests, registries, capture=False)[0], ['embedding'])
