"""TRELLIS.2 port: conversion layout and refusals, and the native sparse, dense and mesh primitives."""
import importlib
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.backends import PORTS_DIR

HAS_MLX = importlib.util.find_spec('mlx') is not None
SHELL = 'mlx3d_ports_under_test'
DINOV3 = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
STRUCTURE = 'microsoft/TRELLIS-image-large'
FLOW_ARGS = {'pe_mode': 'rope', 'share_mod': True, 'qk_rms_norm': True, 'qk_rms_norm_cross': True}
CHECKPOINTS = {
    'sparse_structure_flow_model': ('ckpts/ss_flow', 'SparseStructureFlowModel'),
    'sparse_structure_decoder': (STRUCTURE + '/ckpts/ss_dec', 'SparseStructureDecoder'),
    'shape_slat_flow_model_512': ('ckpts/slat_512', 'SLatFlowModel'),
    'shape_slat_flow_model_1024': ('ckpts/slat_1024', 'SLatFlowModel'),
    'shape_slat_decoder': ('ckpts/shape_dec', 'FlexiDualGridVaeDecoder'),
}


def port(name):
    """A trellis2 module imported through a package over the backend's ports folder, as the backend installs it."""
    if SHELL not in sys.modules:
        shell = types.ModuleType(SHELL)
        shell.__path__ = [str(PORTS_DIR / 'mlx-3d')]
        sys.modules[SHELL] = shell
    return importlib.import_module(SHELL + '.trellis2.' + name)


def pipeline_json(name='Trellis2ImageTo3DPipeline', flow_args=FLOW_ARGS):
    sampler = {'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.5,
               'guidance_interval': [0.6, 1.0], 'rescale_t': 3.0}
    return {'name': name, 'args': {
        'models': {key: reference for key, (reference, _) in CHECKPOINTS.items()},
        'image_cond_model': {'name': 'DinoV3FeatureExtractor', 'args': {'model_name': DINOV3}},
        'sparse_structure_sampler': {'params': dict(sampler, guidance_rescale=0.7, rescale_t=5.0)},
        'shape_slat_sampler': {'params': dict(sampler, steps=10, guidance_strength=6.0)},
        'shape_slat_normalization': {'mean': [0.0] * 4, 'std': [1.0] * 4},
    }}


def write_source(root, flow_args=FLOW_ARGS, tensors=None):
    """A TRELLIS.2 repository, its structure-decoder repository and a DINOv3 snapshot under ``root``."""
    repos = {'': root / 'trellis2', STRUCTURE: root / 'structure', DINOV3: root / 'dinov3'}
    for path in repos.values():
        (path / 'ckpts').mkdir(parents=True)
    (repos[''] / 'pipeline.json').write_text(json.dumps(pipeline_json()))
    for reference, kind in CHECKPOINTS.values():
        repo = STRUCTURE if reference.startswith(STRUCTURE) else ''
        stem = repos[repo] / reference[len(repo):].lstrip('/')
        args = dict(flow_args) if kind.endswith('FlowModel') else {}
        Path(str(stem) + '.json').write_text(json.dumps({'name': kind, 'args': args}))
        if tensors is not None:
            tensors(Path(str(stem) + '.safetensors'))
    (repos[DINOV3] / 'config.json').write_text(json.dumps({'model_type': 'dinov3_vit'}))
    if tensors is not None:
        tensors(repos[DINOV3] / 'model.safetensors')
    return repos[''], {STRUCTURE: repos[STRUCTURE], DINOV3: repos[DINOV3]}


class Trellis2ConversionRefusalTests(unittest.TestCase):
    def test_refuses_another_pipeline_without_creating_output(self):
        convert = port('convert').convert
        with TemporaryDirectory() as folder:
            source, sources = write_source(Path(folder))
            (source / 'pipeline.json').write_text(json.dumps(pipeline_json(name='Trellis2TexturingPipeline')))
            with self.assertRaisesRegex(ValueError, 'Trellis2ImageTo3DPipeline'):
                convert(source, Path(folder) / 'out', sources)
            self.assertFalse((Path(folder) / 'out').exists())

    def test_refuses_without_the_required_repositories(self):
        convert = port('convert').convert
        with TemporaryDirectory() as folder:
            source, sources = write_source(Path(folder))
            with self.assertRaisesRegex(ValueError, '--requires ' + STRUCTURE):
                convert(source, Path(folder) / 'out', {DINOV3: sources[DINOV3]})
            with self.assertRaisesRegex(ValueError, '--requires ' + DINOV3):
                convert(source, Path(folder) / 'out', {STRUCTURE: sources[STRUCTURE]})
            self.assertFalse((Path(folder) / 'out').exists())

    def test_refuses_a_flow_variant_the_port_does_not_implement(self):
        convert = port('convert').convert
        with TemporaryDirectory() as folder:
            source, sources = write_source(Path(folder), flow_args=dict(FLOW_ARGS, pe_mode='ape'))
            with self.assertRaisesRegex(ValueError, 'pe_mode'):
                convert(source, Path(folder) / 'out', sources)

    def test_refuses_projection_conditioned_flows(self):
        convert = port('convert').convert
        with TemporaryDirectory() as folder:
            source, sources = write_source(Path(folder), flow_args=dict(FLOW_ARGS, image_attn_mode='proj'))
            with self.assertRaisesRegex(ValueError, 'image_attn_mode'):
                convert(source, Path(folder) / 'out', sources)
            self.assertFalse((Path(folder) / 'out').exists())

    def test_refuses_missing_weights(self):
        convert = port('convert').convert
        with TemporaryDirectory() as folder:
            source, sources = write_source(Path(folder))
            with self.assertRaisesRegex(ValueError, 'missing'):
                convert(source, Path(folder) / 'out', sources)
            self.assertFalse((Path(folder) / 'out').exists())


@unittest.skipUnless(HAS_MLX, 'requires MLX on Apple Silicon')
class Trellis2NativeTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        self.addCleanup(mx.set_default_device, previous)

    def test_conversion_writes_components_mesh_defaults_and_provenance(self):
        import mlx.core as mx
        from mlx_agent.gguf import scan_mlx_outputs
        convert = port('convert').convert

        def tensors(path):
            mx.save_safetensors(str(path), {'proj.weight': mx.ones((4, 64)), 'conv.weight': mx.ones((2, 3, 3, 3, 2))})

        with TemporaryDirectory() as folder:
            source, sources = write_source(Path(folder), tensors=tensors)
            models = Path(folder) / 'models'
            convert(source, models / 'TRELLIS.2-MLX-32bit', sources)
            convert(source, models / 'TRELLIS.2-MLX-8bit', sources, quantize=True)
            full = json.loads((models / 'TRELLIS.2-MLX-32bit' / 'config.json').read_text())
            quantized = models / 'TRELLIS.2-MLX-8bit'
            flow = mx.load(str(quantized / 'ss_flow.safetensors'))
            decoder = mx.load(str(quantized / 'ss_decoder.safetensors'))
            listed = {item['name']: item for item in scan_mlx_outputs([models])}
            leftovers = [path.name for path in models.iterdir() if path.name.startswith('.')]
        self.assertEqual(full['model_type'], 'trellis2')
        self.assertEqual(full['mesh'], {'resolutions': [512, 1024, 1536], 'resolution': 1024, 'steps': 10, 'guidance': 6.0})
        self.assertEqual(full['pipeline']['sparse_structure_sampler']['rescale_t'], 5.0)
        self.assertNotIn('quantization', full)
        self.assertEqual(sorted(listed), ['TRELLIS.2-MLX-32bit', 'TRELLIS.2-MLX-8bit'])
        self.assertEqual(listed['TRELLIS.2-MLX-32bit']['provenance']['q_bits'], 32)
        self.assertEqual(listed['TRELLIS.2-MLX-8bit']['provenance']['q_bits'], 8)
        self.assertEqual(listed['TRELLIS.2-MLX-32bit']['provenance']['source']['requires'],
                         {repo: str(path) for repo, path in sorted(sources.items())})
        self.assertIn('proj.scales', flow)
        self.assertEqual(flow['conv.weight'].shape, (2, 3, 3, 3, 2))
        self.assertNotIn('proj.scales', decoder)
        self.assertEqual(leftovers, [])

    def test_sparse_convolution_matches_a_dense_convolution_at_occupied_voxels(self):
        import mlx.core as mx
        import numpy as np
        sparse = port('sparse')
        Weights = importlib.import_module(SHELL + '.triposg.weights').Weights
        rng = np.random.default_rng(3)
        grid = rng.random((6, 6, 6)) < 0.4
        coords = np.argwhere(grid).astype(np.int32)
        feats = rng.standard_normal((len(coords), 5)).astype(np.float32)
        weight = rng.standard_normal((7, 3, 3, 3, 5)).astype(np.float32)
        bias = rng.standard_normal(7).astype(np.float32)
        dense = np.zeros((1, 6, 6, 6, 5), dtype=np.float32)
        dense[0, coords[:, 0], coords[:, 1], coords[:, 2]] = feats
        reference = np.array(mx.conv3d(mx.array(dense), mx.array(weight), padding=1) + mx.array(bias))
        expected = reference[0, coords[:, 0], coords[:, 1], coords[:, 2]]
        values = {'conv.weight': mx.array(weight), 'conv.bias': mx.array(bias)}
        for label, w in (('checkpoint layout', values),
                         ('precomputed kernel', {'conv.kernel': sparse.conv_kernel(values['conv.weight']),
                                                 'conv.bias': values['conv.bias']})):
            with self.subTest(label):
                out = sparse.sparse_conv(Weights(w), 'conv', sparse.Voxels(mx.array(feats), coords))
                np.testing.assert_allclose(np.array(out.feats), expected, rtol=1e-4, atol=1e-4)

    def test_subdivision_places_each_kept_octant_by_its_axis_bits(self):
        import mlx.core as mx
        import numpy as np
        sparse = port('sparse')
        voxels = sparse.Voxels(mx.zeros((1, 8)), np.array([[1, 2, 3]]))
        feats = mx.arange(16, dtype=mx.float32).reshape(1, 16)
        keep = np.zeros((1, 8), dtype=bool)
        keep[0, [0, 5, 6]] = True
        coords, (split,) = sparse.subdivide(voxels, (feats,), keep)
        self.assertEqual(coords.tolist(), [[2, 4, 6], [3, 4, 7], [2, 5, 7]])
        self.assertEqual(np.array(split).tolist(), [[0, 1], [10, 11], [12, 13]])

    def test_pixel_shuffle_moves_channel_blocks_to_voxel_offsets(self):
        import mlx.core as mx
        import numpy as np
        structure = port('structure')
        out = np.array(structure.pixel_shuffle(mx.arange(16, dtype=mx.float32).reshape(1, 1, 1, 1, 16)))
        self.assertEqual(out.shape, (1, 2, 2, 2, 2))
        for i, j, k, c in np.ndindex(2, 2, 2, 2):
            self.assertEqual(out[0, i, j, k, c], c * 8 + i * 4 + j * 2 + k)

    def test_occupancy_pools_the_decoder_grid_to_the_latent_resolution(self):
        import mlx.core as mx
        import numpy as np
        structure = port('structure')
        logits = np.full((1, 8, 8, 8), -1.0, dtype=np.float32)
        logits[0, 5, 0, 7] = 2.0
        self.assertEqual(structure.occupied(mx.array(logits), 4).tolist(), [[2, 0, 3]])

    def test_dual_grid_splits_each_quad_along_its_heavier_diagonal(self):
        import numpy as np
        dual_grid = port('dual_grid')
        coords = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]])
        vertices = np.full((4, 3), 0.5, dtype=np.float32)
        crossed = np.zeros((4, 3), dtype=bool)
        crossed[0, 0] = True
        cases = (([2, 1, 2, 1], [[0, 1, 2], [0, 2, 3]]), ([1, 2, 1, 2], [[0, 1, 3], [3, 1, 2]]))
        for weights, triangles in cases:
            with self.subTest(weights=weights):
                points, faces = dual_grid.to_mesh(coords, vertices, crossed, np.array(weights)[:, None], 4)
                self.assertEqual(faces.tolist(), triangles)
        np.testing.assert_allclose(points[0], [-0.375, -0.375, -0.375])
        np.testing.assert_allclose(points[2], [-0.375, -0.125, -0.125])
        _, faces = dual_grid.to_mesh(coords[:3], vertices[:3], crossed[:3], np.ones((3, 1)), 4)
        self.assertEqual(faces.shape, (0, 3))

    def test_cascade_lowers_the_resolution_until_the_tokens_fit(self):
        import numpy as np
        pipeline = port('pipeline')
        line = np.stack([np.arange(512), np.zeros(512), np.zeros(512)], axis=1)
        coords, resolution = pipeline.cascade_coords(line, 512, 1536, max_tokens=81)
        self.assertEqual((len(coords), resolution), (80, 1280))
        self.assertEqual(coords[:, 0].tolist(), list(range(80)))
        coords, resolution = pipeline.cascade_coords(line, 512, 1536, max_tokens=10)
        self.assertEqual((len(coords), resolution), (64, 1024))

    def test_schedule_rescales_time_toward_the_noisy_end(self):
        flow = port('flow')
        self.assertEqual(flow.schedule(2, 3.0), [(1.0, 0.75), (0.75, 0.0)])
        self.assertEqual(flow.schedule(1, 1.0), [(1.0, 0.0)])

    def test_load_and_generate_refuse_what_they_cannot_run(self):
        pipeline = port('pipeline')
        with TemporaryDirectory() as folder:
            (Path(folder) / 'config.json').write_text(json.dumps({'model_type': 'triposg'}))
            with self.assertRaisesRegex(ValueError, 'TRELLIS.2'):
                pipeline.load(folder)
            with self.assertRaisesRegex(ValueError, '512, 1024 or 1536'):
                pipeline.generate(({}, {}), Path(folder) / 'in.png', Path(folder) / 'out.glb', resolution=256)
            self.assertFalse((Path(folder) / 'out.glb').exists())

    def test_crop_centers_the_opaque_object_and_premultiplies_alpha(self):
        import numpy as np
        from PIL import Image
        dinov3 = port('dinov3')
        rgba = np.zeros((40, 60, 4), dtype=np.uint8)
        rgba[10:30, 20:30] = (200, 100, 50, 255)
        rgba[0, 0] = (255, 255, 255, 100)
        rgba[19, 18] = (255, 255, 255, 51)
        crop = np.asarray(dinov3.crop_object(Image.fromarray(rgba))).astype(int)
        self.assertEqual(crop.shape, (18, 18, 3))
        self.assertEqual(crop[9, 9].tolist(), [200, 100, 50])
        self.assertEqual(crop[9, 0].tolist(), [0, 0, 0])
        self.assertLessEqual(abs(crop[9, 2] - 51).max(), 1)
        self.assertEqual(dinov3.crop_object(Image.fromarray(rgba), margin=1.1).size, (20, 20))
        with self.assertRaisesRegex(ValueError, 'no opaque object'):
            dinov3.crop_object(Image.fromarray(np.zeros((8, 8, 4), dtype=np.uint8)))


if __name__ == '__main__':
    unittest.main()
