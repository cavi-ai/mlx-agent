"""Paged draft storage, cold-load admission and resident-mode boundaries."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

from mlx_agent import strata_runtime as strata
from mlx_agent.gguf import read_gguf_header
from .test_gguf import write_gguf, _kv, _UINT32
from . import test_strata_runtime as fixtures


class StrataDraftMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tensors = json.loads((Path(__file__).parents[1] / 'fixtures/strata-mtp-layout.json').read_text())
        self.metadata = {'qwen4exp.embedding_length': 2560, 'qwen4exp.expert_count': 512,
                         'qwen4exp.expert_feed_forward_length': 640, 'qwen4exp.hyper_connection.count': 4}
        self.model = {'mtp_bytes': 5214301696, 'mtp_parameter_bytes': 1596714496}

    def test_supported_layout_accounts_for_packed_weights_and_half_precision_scales(self):
        self.assertEqual(strata._mtp_parameter_bytes(self.tensors, self.metadata), 1596714496)
        self.assertEqual(strata._mtp_parameter_bytes(list(reversed(self.tensors)), self.metadata), 1596714496)

    def test_gguf_reader_retains_native_draft_dimensions(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'model.gguf'
            write_gguf(path, architecture='qwen4exp', extra=[_kv(k, _UINT32, v) for k, v in self.metadata.items()])
            header = read_gguf_header(path)
        self.assertEqual(strata._mtp_parameter_bytes(self.tensors, header['metadata']), 1596714496)

    def test_prepared_unknown_head_keeps_its_raw_byte_charge(self):
        fixture = fixtures.StrataProfileTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.valid_model()
        folder = fixture.root / 'mtp'
        folder.mkdir()
        tensors = []
        for i in range(31):
            filename = str(i) + '.bin'
            (folder / filename).write_bytes(b'\0\0')
            tensors.append({'file': filename, 'name': 'mtp.unknown_' + str(i), 'shape': [1],
                            'dtype': 'BF16', 'bytes': 2, 'sha256': '0' * 64})
        (folder / 'mtp-manifest.json').write_text(json.dumps(tensors))
        marker = fixture.root / strata.MODEL_MARKER
        profile = json.loads(marker.read_text())
        profile['mtp'] = 'mtp'
        marker.write_text(json.dumps(profile))
        model = strata.read_model(fixture.root)
        self.assertEqual(model['mtp_bytes'], 62)
        self.assertEqual(model['mtp_parameter_bytes'], 62)

    def test_unrecognized_layout_and_architecture_keep_raw_accounting(self):
        cases = []
        for key, value in (('dtype', 'F16'), ('bytes', 1), ('shape', [512, 1280, 2559]), ('name', 'unknown')):
            tensors = copy.deepcopy(self.tensors)
            tensors[0][key] = value
            cases.append((tensors, self.metadata))
        cases.extend(((self.tensors[:-1], self.metadata), (self.tensors + [self.tensors[0]], self.metadata),
                      ([self.tensors[0]] * 31, self.metadata), (self.tensors, {}), (self.tensors, None)))
        for key in self.metadata:
            metadata = dict(self.metadata)
            metadata[key] += 1
            cases.append((self.tensors, metadata))
        for tensors, metadata in cases:
            with self.subTest(tensors=tensors[0].get('name'), metadata=metadata):
                self.assertIsNone(strata._mtp_parameter_bytes(tensors, metadata))

    def test_paged_capacity_uses_packed_storage_but_preserves_the_startup_floor(self):
        with patch.object(strata, '_memory_parts', return_value=(18189310976, 33973862400)):
            allocation = strata.memory_allocation(self.model, 24, 16384, True)
            self.assertEqual(allocation['expert_memory_gb'], 9.428)
            self.assertEqual(allocation['mode'], 'paged')
            self.assertLessEqual(allocation['estimated_resident_bytes'], 24000000000)
            with self.assertRaises(strata.StrataError):
                strata.memory_allocation(self.model, 18.689, 16384, True)
            self.assertEqual(strata.memory_allocation(self.model, 18.690, 16384, True)['mode'], 'paged')
            at_floor = strata.memory_allocation(self.model, 18.689310976, 16384, True)
            self.assertEqual(at_floor['estimated_resident_bytes'], 18689310976)
            self.assertLessEqual(at_floor['estimated_resident_bytes'], at_floor['budget_bytes'])

    def test_packed_whole_model_does_not_select_resident_loading(self):
        with patch.object(strata, '_memory_parts', return_value=(18189310976, 33973862400)):
            self.assertEqual(strata.memory_allocation(self.model, 50, 16384, True)['mode'], 'paged')
            self.assertEqual(strata.memory_allocation(self.model, 53, 16384, True)['mode'], 'resident')

    def test_explicit_expert_estimate_covers_both_steady_state_and_startup(self):
        with patch.object(strata, '_memory_parts', return_value=(18189310976, 33973862400)):
            self.assertEqual(strata.memory_estimate(self.model, .5, 16384, True), 18689310976)
            self.assertEqual(strata.memory_estimate(self.model, 9.428, 16384, True), 23999723776)
            for packed in (None, True, -1, 6000000000):
                model = dict(self.model, mtp_parameter_bytes=packed)
                self.assertEqual(strata.memory_estimate(model, 1, 16384, True), 19189310976)

    def test_plain_lookup_and_unrecognized_drafts_retain_their_allocation(self):
        with patch.object(strata, '_memory_parts', return_value=(12975009280, 33973862400)):
            self.assertEqual(strata.memory_allocation(self.model, 24, 16384, False)['expert_memory_gb'], 11.024)
        with patch.object(strata, '_memory_parts', return_value=(18189310976, 33973862400)):
            self.assertEqual(strata.memory_allocation({'mtp_bytes': 5214301696}, 24, 16384, True)['expert_memory_gb'], 5.810)

    def test_frontend_binds_the_half_precision_draft_accounting(self):
        model = dict(self.model, root='/model', gguf='/model/weights.gguf', tokenizer='/model/tokenizer', mtp='/model/mtp')
        with patch.object(strata, '_memory_parts', return_value=(18189310976, 33973862400)):
            config = strata.frontend_config(model, {'engine': '/engine', 'root': '/runtime'},
                                           decoding='mtp', memory_budget_gb=24, context_tokens=16384)
        self.assertEqual(config['args'][config['args'].index('--dtype') + 1], 'float16')
        self.assertEqual(config['args'][config['args'].index('--expert-memory') + 1], '9.428')
