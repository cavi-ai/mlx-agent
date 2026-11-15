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
HAS_SURFACE = all(importlib.util.find_spec(name) for name in ('scipy', 'skimage'))
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
    'tex_slat_flow_model_512': ('ckpts/tex_512', 'SLatFlowModel'),
    'tex_slat_flow_model_1024': ('ckpts/tex_1024', 'SLatFlowModel'),
    'tex_slat_decoder': ('ckpts/tex_dec', 'SparseUnetVaeDecoder'),
}
DECODER_ARGS = {'SparseUnetVaeDecoder': {'pred_subdiv': False}}


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
        'tex_slat_sampler': {'params': dict(sampler, guidance_strength=1.0, guidance_rescale=0.0)},
        'tex_slat_normalization': {'mean': [0.5] * 4, 'std': [2.0] * 4},
    }}


def write_source(root, flow_args=FLOW_ARGS, tensors=None, decoder_args=DECODER_ARGS):
    """A TRELLIS.2 repository, its structure-decoder repository and a DINOv3 snapshot under ``root``."""
    repos = {'': root / 'trellis2', STRUCTURE: root / 'structure', DINOV3: root / 'dinov3'}
    for path in repos.values():
        (path / 'ckpts').mkdir(parents=True)
    (repos[''] / 'pipeline.json').write_text(json.dumps(pipeline_json()))
    for reference, kind in CHECKPOINTS.values():
        repo = STRUCTURE if reference.startswith(STRUCTURE) else ''
        stem = repos[repo] / reference[len(repo):].lstrip('/')
        args = dict(flow_args) if kind.endswith('FlowModel') else dict(decoder_args.get(kind, {}))
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

    def test_refuses_a_texture_decoder_that_predicts_its_own_subdivision(self):
        convert = port('convert').convert
        with TemporaryDirectory() as folder:
            source, sources = write_source(Path(folder), decoder_args={})
            with self.assertRaisesRegex(ValueError, 'tex_slat_decoder'):
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
    def test_components_load_lazily_and_reuse_union_atomically(self):
        import mlx.core as mx
        from unittest.mock import patch
        pipeline = port('pipeline')
        names = pipeline.COMPONENTS + pipeline.TEXTURE_COMPONENTS
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps({'model_type': 'trellis2', 'components': {name: {} for name in names}}))
            for name in names:
                mx.save_safetensors(str(root / (name + '.safetensors')), {'value': mx.array([1.0])})
            real_load = mx.load
            with patch.object(pipeline.mx, 'load', wraps=real_load) as load:
                model = pipeline.load(root)
                self.assertEqual(load.call_count, 0)
                with self.assertRaises(ValueError):
                    pipeline.prepare(model, 256)
                self.assertEqual(load.call_count, 0)
                pipeline.prepare(model, 512)
                self.assertEqual(load.call_count, 7)
                self.assertNotIn('shape_flow_1024', model.weights)
                pipeline.prepare(model, 512)
                self.assertEqual(load.call_count, 7)
                def fail_second(path):
                    if path.endswith('tex_flow_1024.safetensors'):
                        raise OSError('component read failed')
                    return real_load(path)
                load.side_effect = fail_second
                before = model.component_statistics()
                with self.assertRaises(OSError):
                    pipeline.prepare(model, 1024)
                self.assertEqual(model.component_statistics(), before)
                load.side_effect = real_load
                pipeline.prepare(model, 1024)
                pipeline.prepare(model, 512)
                self.assertEqual(set(model.weights), set(names))
                self.assertTrue(all(count == 1 for count in model.component_statistics()['loads'].values()))

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
            texture_flow = mx.load(str(quantized / 'tex_flow_1024.safetensors'))
            texture_decoder = mx.load(str(quantized / 'tex_decoder.safetensors'))
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
        self.assertIn('proj.scales', texture_flow)
        self.assertNotIn('proj.scales', texture_decoder)
        self.assertEqual(full['components']['tex_decoder'], {'pred_subdiv': False})
        self.assertEqual(full['pipeline']['tex_slat_sampler']['guidance_strength'], 1.0)
        self.assertEqual(full['pipeline']['tex_slat_normalization'], {'mean': [0.5] * 4, 'std': [2.0] * 4})
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

    def test_attribute_decoder_replays_the_shape_subdivision_in_voxel_order(self):
        import mlx.core as mx
        import numpy as np
        sparse = port('sparse')
        Weights = importlib.import_module(SHELL + '.triposg.weights').Weights
        rng = np.random.default_rng(5)

        def tensor(*shape):
            return mx.array(rng.standard_normal(shape).astype(np.float32))

        def trunk(out_channels, predicts):
            values = {'from_latent.weight': tensor(16, 4), 'from_latent.bias': tensor(16),
                      'blocks.0.0.norm1.weight': tensor(16), 'blocks.0.0.norm1.bias': tensor(16),
                      'blocks.0.0.conv1.weight': tensor(64, 3, 3, 3, 16), 'blocks.0.0.conv1.bias': tensor(64),
                      'blocks.0.0.conv2.weight': tensor(8, 3, 3, 3, 8), 'blocks.0.0.conv2.bias': tensor(8),
                      'output_layer.weight': tensor(out_channels, 8), 'output_layer.bias': tensor(out_channels)}
            if predicts:
                values.update({'blocks.0.0.to_subdiv.weight': tensor(8, 16), 'blocks.0.0.to_subdiv.bias': tensor(8)})
            return Weights(values)

        config = {'model_channels': [16, 8], 'num_blocks': [0, 0]}
        coords = np.array([[0, 0, 0], [1, 2, 3], [3, 3, 3]], dtype=np.int32)
        latent = tensor(3, 4)
        kept = []
        voxels, _, _, _ = sparse.decode(trunk(7, True), config, latent, coords, kept)
        texture = trunk(6, False)
        painted, attributes = sparse.decode_attributes(texture, config, tensor(3, 4), coords, kept)
        self.assertEqual(len(kept), 1)
        self.assertGreater(len(voxels), 0)
        self.assertEqual(painted.tolist(), voxels.tolist())
        self.assertEqual(attributes.shape, (len(voxels), 6))
        everything = [np.ones((3, 8), dtype=bool)]
        full, _ = sparse.decode_attributes(texture, config, latent, coords, everything)
        self.assertEqual(len(full), 24)

    def test_vertex_colors_come_from_the_holding_voxel_then_the_closest_one(self):
        import numpy as np
        pipeline = port('pipeline')
        coords = np.array([[0, 0, 0], [3, 3, 3]])
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        points = np.array([[-0.4, -0.4, -0.4], [0.4, 0.4, 0.4], [-0.4, -0.4, -0.1], [0.1, 0.4, 0.4]])
        sampled = pipeline.voxel_colors(points, coords, colors, 4)
        self.assertEqual(sampled.tolist(), [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        far = pipeline.voxel_colors(np.array([[0.0, 0.0, 0.0]]), np.array([[0, 0, 0], [15, 15, 15]]), colors, 16)
        self.assertEqual(far.tolist(), [[0.5, 0.0, 0.5]])
        np.testing.assert_allclose(pipeline.srgb_to_linear([0.0, 0.04045, 0.5, 1.0, 2.0]),
                                   [0.0, 0.04045 / 12.92, 0.21404114, 1.0, 1.0], rtol=1e-6)

    def test_glb_carries_vertex_colors_as_a_float_color_attribute(self):
        import struct
        import numpy as np
        write_glb = importlib.import_module(SHELL + '.triposg.geometry').write_glb
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.25]], dtype=np.float32)
        with TemporaryDirectory() as folder:
            write_glb(Path(folder) / 'painted.glb', vertices, faces, colors)
            write_glb(Path(folder) / 'plain.glb', vertices, faces)
            painted = (Path(folder) / 'painted.glb').read_bytes()
            plain = (Path(folder) / 'plain.glb').read_bytes()
            with self.assertRaisesRegex(ValueError, 'Vertex colors'):
                write_glb(Path(folder) / 'bad.glb', vertices, faces, colors[:2])
            self.assertFalse((Path(folder) / 'bad.glb').exists())

        def parts(data):
            size = struct.unpack('<I', data[12:16])[0]
            return json.loads(data[20:20 + size]), data[28 + size:]

        document, binary = parts(painted)
        self.assertEqual(struct.unpack('<I', painted[8:12])[0], len(painted))
        primitive = document['meshes'][0]['primitives'][0]
        accessor = document['accessors'][primitive['attributes']['COLOR_0']]
        view = document['bufferViews'][accessor['bufferView']]
        self.assertEqual((accessor['componentType'], accessor['type'], accessor['count']), (5126, 'VEC3', 3))
        stored = np.frombuffer(binary[view['byteOffset']:view['byteOffset'] + view['byteLength']], dtype='<f4')
        self.assertEqual(stored.reshape(3, 3).tolist(), colors.tolist())
        self.assertEqual(document['buffers'][0]['byteLength'], 36 + 12 + 36)
        self.assertNotIn('COLOR_0', parts(plain)[0]['meshes'][0]['primitives'][0]['attributes'])

    def test_guide_voxels_are_the_surface_shell_with_the_largest_extent_spanning_the_cube(self):
        import itertools
        import numpy as np
        guide = port('guide')
        write_glb = importlib.import_module(SHELL + '.triposg.geometry').write_glb
        # glTF y-up box, 4 tall; TRELLIS z-up maps glTF y to z and glTF -z to y.
        corners = np.array(list(itertools.product((-0.5625, 0.5625), (0.0, 4.0), (-1.0625, 1.0625))), dtype=np.float32)
        faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                          [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.uint32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'box.glb'
            write_glb(path, corners, faces)
            cells = {tuple(cell) for cell in guide.surface_coords(path, 32).tolist()}
        shell = {(x, y, z) for x in range(11, 21) for y in range(7, 25) for z in range(32)
                 if x in (11, 20) or y in (7, 24) or z in (0, 31)}
        self.assertEqual(cells, shell)

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

    @unittest.skipUnless(HAS_SURFACE, 'requires scipy and scikit-image')
    def test_closed_surface_wraps_an_open_branching_mesh_in_one_outward_shell(self):
        import numpy as np
        closed = port('closed')
        grid, band = 32, 0.75
        # Two unit-quad sheets crossing in a line: open borders, four faces on each crossing edge, mixed winding.
        steps = np.arange(8, 25)
        u, w = (axis.ravel() for axis in np.meshgrid(steps, steps, indexing='ij'))
        level = np.full(len(u), 16)
        vertices = (np.concatenate([np.stack([u, w, level], axis=1), np.stack([u, level, w], axis=1)]) / grid - 0.5)
        cell = np.array([[i * 17 + j, (i + 1) * 17 + j, (i + 1) * 17 + j + 1, i * 17 + j + 1]
                         for i in range(16) for j in range(16)])
        quads = np.concatenate([cell, cell + 17 * 17])
        faces = np.concatenate([quads[:, [0, 1, 2]], quads[::2, [0, 2, 3]], quads[1::2, [0, 3, 2]]])
        points, triangles = closed.closed_surface(vertices.astype(np.float32), faces, grid, band)
        directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        # Each directed edge occurs once and its reverse once: every edge joins two consistently wound triangles.
        self.assertEqual(len(np.unique(directed, axis=0)), len(directed))
        self.assertEqual(sorted(map(tuple, directed)), sorted(map(tuple, directed[:, ::-1])))
        a, b, c = (points[triangles[:, i]].astype(np.float64) for i in range(3))
        self.assertGreater(np.einsum('ij,ij->', a, np.cross(b, c)), 0)
        # Every vertex sits about one band from the nearer sheet, in cells.
        cells = (points.astype(np.float64) + 0.5) * grid

        def reach(axis, first, second):
            inside = np.stack([np.clip(cells[:, first], 8, 24), np.clip(cells[:, second], 8, 24)], axis=1)
            flat = np.linalg.norm(cells[:, [first, second]] - inside, axis=1)
            return np.hypot(flat, cells[:, axis] - 16)

        distance = np.minimum(reach(2, 0, 1), reach(1, 0, 2))
        self.assertLess(np.abs(distance - band).max(), 0.25)

    @unittest.skipUnless(HAS_SURFACE, 'requires scipy and scikit-image')
    def test_closed_surface_fills_a_hollow_shell_except_what_an_opening_shows(self):
        import numpy as np
        closed = port('closed')
        grid, low, high = 32, 8, 24
        steps = np.arange(low, high + 1)
        size = len(steps)
        u, w = (axis.ravel() for axis in np.meshgrid(steps, steps, indexing='ij'))
        cell = np.array([[i * size + j, (i + 1) * size + j, (i + 1) * size + j + 1, i * size + j + 1]
                         for i in range(size - 1) for j in range(size - 1)])
        # A window four cells wide in the middle of the top sheet only.
        window = np.array([6 <= i < 10 and 6 <= j < 10 for i in range(size - 1) for j in range(size - 1)])
        sheets, quads = [], []
        for axis in range(3):
            for side in (low, high):
                sheet = np.insert(np.stack([u, w], axis=1), axis, side, axis=1)
                kept = cell[~window] if (axis, side) == (2, high) else cell
                quads.append(kept + sum(len(s) for s in sheets))
                sheets.append(sheet)
        quads = np.concatenate(quads)
        vertices = (np.concatenate(sheets) / grid - 0.5).astype(np.float32)
        faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        points, triangles = closed.closed_surface(vertices, faces, grid)
        a, b, c = (points[triangles[:, i]].astype(np.float64) for i in range(3))
        volume = np.einsum('ij,ij->', a, np.cross(b, c)) / 6 * grid ** 3
        # The box holds 16^3 cells; the window shows a shaft about 2.5 cells wide down to the floor.
        self.assertGreater(volume, 16 ** 3)
        self.assertLess(volume, 17.5 ** 3 - 2 * 2 * 14)
        directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        self.assertEqual(sorted(map(tuple, directed)), sorted(map(tuple, directed[:, ::-1])))

    def test_fins_and_their_vertices_leave_the_mesh(self):
        import numpy as np
        closed = port('closed')
        vertices = np.arange(18, dtype=np.float32).reshape(6, 3)
        tetrahedron = [[0, 2, 3], [0, 3, 5], [0, 5, 2], [2, 5, 3]]
        fin = [[0, 1, 2], [2, 1, 0], [3, 4, 5], [4, 3, 5]]
        points, faces = closed.without_fins(vertices, np.array(tetrahedron[:2] + fin + tetrahedron[2:], dtype=np.int32))
        self.assertEqual(points.tolist(), vertices[[0, 2, 3, 5]].tolist())
        self.assertEqual(faces.tolist(), [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
        self.assertEqual(faces.dtype, np.int32)

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
