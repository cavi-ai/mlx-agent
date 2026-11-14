"""Request-aware TRELLIS selection and telemetry contracts."""
import importlib
import unittest
import json
import tempfile
from pathlib import Path

from mlx_agent.worker_telemetry import validate

components = importlib.import_module('mlx_agent.resources.ports.mlx-3d.trellis2.components')


class ComponentContractTests(unittest.TestCase):
    def test_selection_matches_resolution_and_texture_completeness(self):
        base = {'image_encoder', 'ss_flow', 'ss_decoder', 'shape_flow_512', 'shape_decoder'}
        full = {'components': {name: {} for name in base | {'shape_flow_1024', 'tex_flow_512', 'tex_flow_1024', 'tex_decoder'}}}
        shape = {'components': {name: {} for name in base | {'shape_flow_1024'}}}
        for resolution in (512, 1024, 1536):
            expected = base | ({'shape_flow_1024'} if resolution > 512 else set())
            self.assertEqual(set(components.required_components(shape, resolution)), expected)
            self.assertEqual(set(components.required_components(full, resolution)), expected | {'tex_decoder', 'tex_flow_512' if resolution == 512 else 'tex_flow_1024'})
        with self.assertRaises(ValueError):
            components.required_components(full, 256)

    def test_telemetry_accepts_old_and_component_records_but_rejects_bad_values(self):
        old = {'schema': 'worker-memory/1', 'active_bytes': 1, 'cache_bytes': 2, 'peak_bytes': 3,
               'loads': 1, 'cache_hits': 0, 'load_seconds': 0.2, 'captured_at': 1.0}
        self.assertEqual(validate(old), old)
        value = dict(old, components={'schema': 'resident-components/1', 'resident': ['image_encoder'],
                     'loads': {'image_encoder': 1}, 'load_seconds': {'image_encoder': 0.1},
                     'checkpoint_bytes': {'image_encoder': 24}})
        self.assertEqual(validate(value), value)
        value['components']['load_seconds']['image_encoder'] = float('nan')
        with self.assertRaises(ValueError):
            validate(value)
        value['components']['load_seconds']['image_encoder'] = 0.1
        value['components']['resident'].append('unknown')
        with self.assertRaises(ValueError):
            validate(value)
        value['components']['resident'].pop()
        value['components']['checkpoint_bytes'].clear()
        with self.assertRaises(ValueError):
            validate(value)


class AdmissionContractTests(unittest.TestCase):
    def test_selected_cold_and_missing_warm_bytes(self):
        from mlx_agent.task_serving import _trellis_component_budget
        names = ('image_encoder', 'ss_flow', 'ss_decoder', 'shape_flow_512', 'shape_flow_1024',
                 'shape_decoder', 'tex_flow_512', 'tex_flow_1024', 'tex_decoder')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text(json.dumps({'model_type': 'trellis2', 'components': {n: {} for n in names}}))
            for i, name in enumerate(names, 1):
                (path / (name + '.safetensors')).write_bytes(b'x' * i)
            plan = {'argv': ['python', 'runner', '--resolution', '512']}
            cold, extra = _trellis_component_budget(path, 'mesh_generation', plan, None, warm=False)
            self.assertEqual((cold, extra), (32, 0))
            selected, conservative = _trellis_component_budget(path, 'mesh_generation', plan, None, warm=True)
            self.assertEqual(selected, 1 + 2 + 3 + 4 + 6 + 7 + 9)
            self.assertEqual(conservative, 50)
            resident = ['image_encoder', 'ss_flow', 'ss_decoder', 'shape_flow_512', 'shape_decoder', 'tex_flow_512', 'tex_decoder']
            telemetry = {'schema':'worker-memory/1', 'active_bytes': 1, 'cache_bytes': 0, 'peak_bytes': 1,
                         'loads':1, 'cache_hits':0, 'load_seconds':0., 'captured_at':1.,
                         'components': {'schema':'resident-components/1', 'resident':resident,
                           'loads':{n:1 for n in resident}, 'load_seconds':{n:0. for n in resident},
                           'checkpoint_bytes':{n:1 for n in resident}}}
            plan['argv'][-1] = '1024'
            selected, missing = _trellis_component_budget(path, 'mesh_generation', plan, telemetry, warm=True)
            self.assertEqual(selected, 1 + 2 + 3 + 4 + 5 + 6 + 8 + 9)
            self.assertEqual(missing, 15)
