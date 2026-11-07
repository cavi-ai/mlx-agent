"""Pixal3D port: checkpoint reader, image operations, NAF, MoGe geometry, projection, cascade and conversion."""
import importlib
import importlib.util
import json
import os
import pickle
import sys
import types
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.backends import PORTS_DIR

HAS_MLX = importlib.util.find_spec('mlx') is not None
HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HAS_SCIPY = importlib.util.find_spec('scipy') is not None
SHELL = 'mlx3d_ports_under_test'
FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'pixal3d'
DINOV3 = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
MOGE = 'Ruicheng/moge-2-vitl'
NAF = 'naf_release.pth'
FLOW_ARGS = {'pe_mode': 'rope', 'share_mod': True, 'qk_rms_norm': True, 'qk_rms_norm_cross': True,
             'image_attn_mode': 'proj'}
CHECKPOINTS = {
    'sparse_structure_flow_model': ('ckpts/ss_flow', 'SparseStructureFlowModel', {}),
    'sparse_structure_decoder': ('ckpts/ss_dec', 'SparseStructureDecoder', None),
    'shape_slat_flow_model_512': ('ckpts/slat_512', 'ElasticSLatFlowModel', {'proj_in_channels': 2048}),
    'shape_slat_flow_model_1024': ('ckpts/slat_1024', 'ElasticSLatFlowModel', {'proj_in_channels': 2048}),
    'shape_slat_decoder': ('ckpts/shape_dec', 'FlexiDualGridVaeDecoder', None),
}


def port(name):
    """A port module imported through a package over the backend's ports folder, as the backend installs it."""
    if SHELL not in sys.modules:
        shell = types.ModuleType(SHELL)
        shell.__path__ = [str(PORTS_DIR / 'mlx-3d')]
        sys.modules[SHELL] = shell
    return importlib.import_module(SHELL + '.' + name)


def pipeline_json(name='Trellis2ImageTo3DPipeline'):
    sampler = {'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.5,
               'guidance_interval': [0.6, 1.0], 'rescale_t': 3.0}
    return {'name': name, 'args': {
        'models': {key: reference for key, (reference, _, _) in CHECKPOINTS.items()},
        'image_cond_model': {'name': 'DinoV3FeatureExtractor', 'args': {'model_name': DINOV3}},
        'sparse_structure_sampler': {'params': dict(sampler, guidance_rescale=0.7, rescale_t=5.0)},
        'shape_slat_sampler': {'params': dict(sampler, steps=10, guidance_strength=6.0)},
        'shape_slat_normalization': {'mean': [0.0] * 4, 'std': [1.0] * 4},
        'default_pipeline_type': '1536_cascade',
    }}


def write_source(root, flow_args=FLOW_ARGS, tensors=None):
    """A Pixal3D repository, a DINOv3 snapshot, a MoGe snapshot and the NAF weights under ``root``."""
    source, dinov3, moge = root / 'pixal3d', root / 'dinov3', root / 'moge'
    for path in (source / 'ckpts', dinov3, moge):
        path.mkdir(parents=True)
    (source / 'pipeline.json').write_text(json.dumps(pipeline_json()))
    for reference, kind, extra in CHECKPOINTS.values():
        args = dict(flow_args, **extra) if extra is not None else {}
        (source / (reference + '.json')).write_text(json.dumps({'name': kind, 'args': args}))
        if tensors is not None:
            tensors(source / (reference + '.safetensors'))
    (dinov3 / 'config.json').write_text(json.dumps({'model_type': 'dinov3_vit', 'hidden_size': 1024}))
    if tensors is not None:
        tensors(dinov3 / 'model.safetensors')
        (moge / 'model.pt').write_bytes((FIXTURES / 'moge.pt').read_bytes())
    return source, {DINOV3: dinov3, MOGE: moge}, {NAF: FIXTURES / 'naf.pth'}


class Pixal3DConversionRefusalTests(unittest.TestCase):
    def convert(self, *args, **kwargs):
        return port('pixal3d.convert').convert(*args, **kwargs)

    def test_refuses_another_pipeline_without_creating_output(self):
        with TemporaryDirectory() as folder:
            source, sources, downloads = write_source(Path(folder))
            (source / 'pipeline.json').write_text(json.dumps(pipeline_json(name='Trellis2TexturingPipeline')))
            with self.assertRaisesRegex(ValueError, 'Trellis2ImageTo3DPipeline'):
                self.convert(source, Path(folder) / 'out', sources, downloads)
            self.assertFalse((Path(folder) / 'out').exists())

    def test_refuses_without_the_files_it_reads_elsewhere(self):
        with TemporaryDirectory() as folder:
            source, sources, downloads = write_source(Path(folder))
            with self.assertRaisesRegex(ValueError, '--requires ' + DINOV3):
                self.convert(source, Path(folder) / 'out', {MOGE: sources[MOGE]}, downloads)
            with self.assertRaisesRegex(ValueError, '--requires ' + MOGE):
                self.convert(source, Path(folder) / 'out', {DINOV3: sources[DINOV3]}, downloads)
            with self.assertRaisesRegex(ValueError, '--download ' + NAF):
                self.convert(source, Path(folder) / 'out', sources, {})
            self.assertFalse((Path(folder) / 'out').exists())

    def test_refuses_flows_without_projection_conditioning(self):
        with TemporaryDirectory() as folder:
            source, sources, downloads = write_source(Path(folder), flow_args=dict(FLOW_ARGS, image_attn_mode='cross'))
            with self.assertRaisesRegex(ValueError, 'image_attn_mode'):
                self.convert(source, Path(folder) / 'out', sources, downloads)

    def test_refuses_projected_features_of_another_width(self):
        with TemporaryDirectory() as folder:
            source, sources, downloads = write_source(Path(folder))
            path = source / 'ckpts' / 'slat_1024.json'
            described = json.loads(path.read_text())
            described['args']['proj_in_channels'] = 1024
            path.write_text(json.dumps(described))
            with self.assertRaisesRegex(ValueError, 'proj_in_channels'):
                self.convert(source, Path(folder) / 'out', sources, downloads)

    def test_refuses_missing_weights(self):
        with TemporaryDirectory() as folder:
            source, sources, downloads = write_source(Path(folder))
            with self.assertRaisesRegex(ValueError, 'missing'):
                self.convert(source, Path(folder) / 'out', sources, downloads)
            self.assertFalse((Path(folder) / 'out').exists())


@unittest.skipUnless(HAS_NUMPY, 'requires numpy')
class CheckpointReaderTests(unittest.TestCase):
    def test_reads_every_tensor_torch_saved_including_views_and_bfloat16(self):
        import numpy as np
        state = port('pixal3d.checkpoint').load(FIXTURES / 'tensors.pt')
        self.assertEqual(list(state), ['f32', 'f16', 'bf16', 'i64', 'view', 'shared'])
        np.testing.assert_array_equal(state['f32'], np.arange(6, dtype=np.float32).reshape(2, 3))
        self.assertEqual(state['f16'].dtype, np.float16)
        self.assertEqual(state['f16'].tolist(), [0.5, -2.0])
        self.assertEqual(state['bf16'].dtype, np.float32)
        self.assertEqual(state['bf16'].tolist(), [1.5, -3.25])
        self.assertEqual(state['i64'].tolist(), [7, -1])
        np.testing.assert_array_equal(state['view'], np.arange(12, dtype=np.float32).reshape(3, 4).T[1:])
        self.assertEqual(state['shared'].tolist(), [3.0, 4.0, 5.0])

    def test_reads_plain_configuration_beside_the_tensors(self):
        loaded = port('pixal3d.checkpoint').load(FIXTURES / 'config.pt')
        self.assertEqual(loaded['model_config'], {'remap_output': 'exp', 'dims': [1, 2]})
        self.assertEqual(loaded['model']['w'].tolist(), [[1.0, 2.0]])

    def test_refuses_a_pickle_that_calls_anything_but_tensor_rebuilding(self):
        checkpoint = port('pixal3d.checkpoint')
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'hostile.pt'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('hostile/data.pkl', pickle.dumps(_Hostile(), protocol=2))
            with self.assertRaisesRegex(pickle.UnpicklingError, 'not a tensor'):
                checkpoint.load(path)
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('other.txt', b'')
            with self.assertRaisesRegex(ValueError, 'Not a PyTorch zip checkpoint'):
                checkpoint.load(path)


class _Hostile:
    def __reduce__(self):
        return os.getcwd, ()


@unittest.skipUnless(HAS_MLX, 'requires MLX on Apple Silicon')
class Pixal3DNativeTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        self.addCleanup(mx.set_default_device, previous)

    def test_image_operations_follow_pytorch(self):
        """Expected values are what PyTorch's F.interpolate, F.pad, F.adaptive_avg_pool2d and F.group_norm return."""
        import mlx.core as mx
        import numpy as np
        ops = port('pixal3d.ops')
        Weights = port('triposg.weights').Weights
        row = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8, 1)
        rows = mx.broadcast_to(mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1), (1, 3, 4, 1))
        cases = {
            'antialiased shrink': (ops.resize(row, (1, 3), antialias=True), [1.1052632, 3.5, 5.8947372]),
            'bilinear growth': (ops.resize(row[:, :, :3], (1, 7)),
                                [0.0, 0.14285715, 0.5714286, 1.0, 1.4285715, 1.857143, 2.0]),
            'adaptive pool': (ops.adaptive_avg_pool(row[:, :, :5], (1, 3)), [0.5, 2.0, 3.5]),
            'reflect pad': (ops.pad(rows, 2, 'reflect')[:, 2], [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0]),
            'replicate pad': (ops.pad(rows, 2, 'replicate')[:, 2], [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]),
        }
        for label, (value, expected) in cases.items():
            with self.subTest(label):
                np.testing.assert_allclose(np.array(value).ravel(), expected, rtol=1e-5, atol=1e-6)
        bicubic = ops.bicubic_scale(np.array([[0., 1., 4., 9., 16.]], dtype=np.float32)[:, :, None], (1.0, 7.1 / 5))
        np.testing.assert_allclose(bicubic.ravel(), [-0.0805353, 0.2543238, 1.6310846, 3.7960398, 6.8257957,
                                                     12.058713, 16.346123], rtol=1e-5, atol=1e-5)
        values = mx.array(np.array([[1., 2., 3., 4.], [5., 6., 7., 9.]], dtype=np.float32).reshape(1, 2, 1, 4))
        w = Weights({'gn.weight': mx.array([1., 2., 1., 1.]), 'gn.bias': mx.array([0., 0., 1., 0.])})
        normed = np.array(ops.group_norm(w, 'gn', values, 2))[0, :, 0].T.ravel()
        np.testing.assert_allclose(normed, [-1.2126766, 0.72760606, -1.455212, 2.4253535, -0.15311232, 1.524142,
                                            -0.73379886, 1.362769], rtol=1e-5, atol=1e-5)

    def test_naf_attention_matches_dilated_neighborhood_attention_over_upsampled_keys(self):
        """Reference: keys and values nearest-upsampled to the output, then each pixel attends to the clamped
        kernel x kernel window of its dilation group, as NATTEN's na2d with dilation output // patches does."""
        import mlx.core as mx
        import numpy as np
        naf = port('pixal3d.naf')
        rng = np.random.default_rng(5)
        cells, step, kernel, heads = (4, 5), 2, 3, 2
        height, width = cells[0] * step, cells[1] * step
        q = rng.standard_normal((height, width, 8)).astype(np.float32)
        k = rng.standard_normal(cells + (8,)).astype(np.float32)
        v = rng.standard_normal(cells + (6,)).astype(np.float32)
        out = np.array(naf.attend(mx.array(q[None]), mx.array(k[None]), mx.array(v[None]), kernel, heads))[0]
        rows = np.floor((np.arange(height) + 0.5) * cells[0] / height).astype(int)
        cols = np.floor((np.arange(width) + 0.5) * cells[1] / width).astype(int)
        keys, values = k[rows][:, cols], v[rows][:, cols]
        expected = np.zeros((height, width, 6), dtype=np.float32)
        for y, x in np.ndindex(height, width):
            ys = [(min(max(y // step - kernel // 2, 0), cells[0] - kernel) + i) * step + y % step for i in range(kernel)]
            xs = [(min(max(x // step - kernel // 2, 0), cells[1] - kernel) + i) * step + x % step for i in range(kernel)]
            for head in range(heads):
                window_k = keys[ys][:, xs, head * 4:(head + 1) * 4].reshape(-1, 4)
                window_v = values[ys][:, xs, head * 3:(head + 1) * 3].reshape(-1, 3)
                scores = window_k @ q[y, x, head * 4:(head + 1) * 4] / 2.0
                weights = np.exp(scores - scores.max())
                expected[y, x, head * 3:(head + 1) * 3] = weights @ window_v / weights.sum()
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
        with self.assertRaisesRegex(ValueError, 'whole multiple'):
            naf.attend(mx.array(q[None, :7]), mx.array(k[None]), mx.array(v[None]), kernel, heads)

    def test_naf_rotary_encoding_keeps_norms_and_depends_only_on_offsets(self):
        import mlx.core as mx
        import numpy as np
        naf = port('pixal3d.naf')
        rng = np.random.default_rng(2)
        q, k = (np.broadcast_to(rng.standard_normal(16).astype(np.float32), (1, 6, 6, 16)) for _ in range(2))
        periods = np.array([0.7, 3.1], dtype=np.float32)
        rq, rk = (np.array(naf.rope(mx.array(np.ascontiguousarray(x)), periods, heads=2))[0] for x in (q, k))
        np.testing.assert_allclose(np.linalg.norm(rq.reshape(6, 6, 2, 8), axis=-1),
                                   np.linalg.norm(q[0].reshape(6, 6, 2, 8), axis=-1), rtol=1e-5)
        dot = lambda a, b: (rq[a].reshape(2, 8) * rk[b].reshape(2, 8)).sum(-1)
        np.testing.assert_allclose(dot((0, 0), (1, 2)), dot((3, 2), (4, 4)), rtol=1e-4, atol=1e-5)
        self.assertFalse(np.allclose(dot((0, 0), (1, 2)), dot((0, 0), (2, 1))))

    def test_moge_upsamples_by_transposed_convolution_like_pytorch(self):
        """Expected values are F.conv_transpose2d(kernel_size=2, stride=2) of the same input."""
        import mlx.core as mx
        import numpy as np
        moge = port('pixal3d.moge')
        Weights = port('triposg.weights').Weights
        w = Weights({'up.weight': mx.arange(8, dtype=mx.float32).reshape(1, 2, 2, 2), 'up.bias': mx.array([0.5, -1.0])})
        out = np.array(moge.conv_transpose2x(w, 'up', mx.array([[[[1.0], [2.0]]]])))[0]
        np.testing.assert_allclose(out[..., 0], [[0.5, 1.5, 0.5, 2.5], [2.5, 3.5, 4.5, 6.5]])
        np.testing.assert_allclose(out[..., 1], [[3.0, 4.0, 7.0, 9.0], [5.0, 6.0, 11.0, 13.0]])

    def test_moge_fits_on_the_tokens_and_pixels_pytorch_picks(self):
        import numpy as np
        moge = port('pixal3d.moge')
        self.assertEqual(moge.token_grid(1038, 1038, 3600), (60, 60))
        self.assertEqual(moge.token_grid(400, 800, 3600), (42, 85))
        self.assertEqual(moge.fit_indices(100)[:12].tolist(), [0, 1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17])
        uv = moge.view_plane_uv(4, 2, 2.0)
        self.assertEqual(uv.shape, (2, 4, 2))
        np.testing.assert_allclose(uv[0, 0], [-2 / 5 ** 0.5 * 3 / 4, -1 / 5 ** 0.5 / 2], rtol=1e-6)

    @unittest.skipUnless(HAS_SCIPY, 'requires scipy')
    def test_moge_recovers_the_focal_length_of_a_shifted_point_map(self):
        import numpy as np
        moge = port('pixal3d.moge')
        rng = np.random.default_rng(4)
        uv = rng.uniform(-0.6, 0.6, (500, 2))
        depth = rng.uniform(2.0, 4.0, 500)
        points = np.concatenate([uv * depth[:, None] / 1.3, (depth - 0.7)[:, None]], axis=1)
        self.assertAlmostEqual(moge.solve_focal(uv, points), 1.3, places=3)

    def test_projection_maps_the_cube_onto_the_image_like_the_reference_camera(self):
        import numpy as np
        projection = port('pixal3d.projection')
        angle = 0.6
        distance = projection.distance(angle)
        coords = np.array([[8, 8, 8], [0, 8, 8], [16, 8, 8], [8, 16, 8], [0, 8, 16]])
        pixels = projection.pixels(projection.grid_points(coords, 17), angle, distance, 512)
        np.testing.assert_allclose(pixels[:4], [[256, 256], [0, 256], [512, 256], [256, 0]], atol=1e-3)
        self.assertLess(pixels[4, 0], 0)

    def test_the_side_the_input_camera_sees_is_the_glb_front_and_up_stays_up(self):
        import numpy as np
        projection, pipeline = port('pixal3d.projection'), port('pixal3d.pipeline')
        angle, resolution = 0.6, 17
        coords = np.array([[0, 8, 0], [0, 8, 16], [8, 16, 8], [8, 0, 8]])
        pixels = projection.pixels(projection.grid_points(coords, resolution), angle, projection.distance(angle), 512)
        nearer = int(np.argmax(np.abs(pixels[:2, 0] - 256)))  # perspective spreads the side nearer the camera
        vertices = pipeline.gltf_vertices(np.linspace(-0.5, 0.5, resolution)[coords])
        self.assertEqual(int(np.argmax(vertices[:2, 2])), nearer)
        self.assertLess(pixels[2, 1], pixels[3, 1])  # higher in the image
        self.assertGreater(vertices[2, 1], vertices[3, 1])

    def test_projection_samples_like_grid_sample_with_border_padding(self):
        """Expected values are F.grid_sample(bilinear, align_corners=False, padding_mode='border')."""
        import mlx.core as mx
        import numpy as np
        projection = port('pixal3d.projection')
        feature_map = mx.broadcast_to(mx.arange(4, dtype=mx.float32).reshape(1, 4, 1), (3, 4, 1))
        positions = np.array([[-0.5, 31.5], [10.0, 31.5], [31.5, 31.5], [70.0, 31.5]], dtype=np.float32)
        np.testing.assert_allclose(np.array(projection.sample(feature_map, positions, 64)).ravel(),
                                   [0.0, 0.15625, 1.5, 3.0], atol=1e-6)

    def test_cascade_rounds_onto_the_projection_grid_and_lowers_until_the_tokens_fit(self):
        import numpy as np
        pipeline = port('pixal3d.pipeline')
        line = np.stack([np.arange(512), np.zeros(512), np.zeros(512)], axis=1)
        coords, resolution = pipeline.cascade_coords(line, 512, 1536)
        self.assertEqual((len(coords), resolution, int(coords[:, 0].max())), (96, 1536, 95))
        coords, resolution = pipeline.cascade_coords(line, 512, 1536, max_tokens=90)
        self.assertEqual((len(coords), resolution), (88, 1408))
        coords, resolution = pipeline.cascade_coords(line, 512, 1536, max_tokens=10)
        self.assertEqual((len(coords), resolution), (64, 1024))

    def test_load_and_generate_refuse_what_they_cannot_run(self):
        pipeline = port('pixal3d.pipeline')
        with TemporaryDirectory() as folder:
            (Path(folder) / 'config.json').write_text(json.dumps({'model_type': 'trellis2'}))
            with self.assertRaisesRegex(ValueError, 'Pixal3D'):
                pipeline.load(folder)
            with self.assertRaisesRegex(ValueError, '1024 or 1536'):
                pipeline.generate(({}, {}), Path(folder) / 'in.png', Path(folder) / 'out.glb', resolution=512)
            self.assertFalse((Path(folder) / 'out.glb').exists())

    def test_projected_features_add_to_the_cross_attention_output(self):
        """A projection block whose proj_linear is only a bias b equals the plain block with b in to_out."""
        import mlx.core as mx
        import numpy as np
        flow = port('trellis2.flow')
        Weights = port('triposg.weights').Weights
        rng = np.random.default_rng(1)
        shapes = {
            'input_layer': (8, 3), 't_embedder.mlp.0': (8, 256), 't_embedder.mlp.2': (8, 8),
            'adaLN_modulation.1': (48, 8), 'blocks.0.self_attn.to_qkv': (24, 8), 'blocks.0.self_attn.to_out': (8, 8),
            'blocks.0.cross_attn.to_q': (8, 8), 'blocks.0.cross_attn.to_kv': (16, 5),
            'blocks.0.cross_attn.to_out': (8, 8), 'blocks.0.mlp.mlp.0': (16, 8), 'blocks.0.mlp.mlp.2': (8, 16),
            'out_layer': (3, 8),
        }
        cross = {}
        for name, shape in shapes.items():
            cross[name + '.weight'] = mx.array(rng.standard_normal(shape).astype(np.float32) * 0.3)
            cross[name + '.bias'] = mx.array(rng.standard_normal(shape[0]).astype(np.float32) * 0.1)
        for name in ('blocks.0.self_attn.q_rms_norm', 'blocks.0.self_attn.k_rms_norm',
                     'blocks.0.cross_attn.q_rms_norm', 'blocks.0.cross_attn.k_rms_norm'):
            cross[name + '.gamma'] = mx.array(rng.uniform(0.5, 1.5, (2, 4)).astype(np.float32))
        cross['blocks.0.modulation'] = mx.array(rng.standard_normal(48).astype(np.float32) * 0.1)
        cross['blocks.0.norm2.weight'] = mx.ones(8)
        cross['blocks.0.norm2.bias'] = mx.zeros(8)
        offset = mx.array(rng.standard_normal(8).astype(np.float32))
        projected = {name.replace('.cross_attn.', '.cross_attn.cross_attn_block.'): value for name, value in cross.items()}
        projected['blocks.0.cross_attn.proj_linear.weight'] = mx.zeros((8, 6))
        projected['blocks.0.cross_attn.proj_linear.bias'] = offset
        shifted = dict(cross, **{'blocks.0.cross_attn.to_out.bias': cross['blocks.0.cross_attn.to_out.bias'] + offset})
        config = {'num_blocks': 1, 'num_heads': 2}
        x = mx.array(rng.standard_normal((1, 5, 3)).astype(np.float32))
        t = mx.array([500.0])
        context = mx.array(rng.standard_normal((1, 4, 5)).astype(np.float32))
        rope = flow.rope_tables(np.argwhere(np.ones((5, 1, 1))), 4)
        proj = mx.array(rng.standard_normal((1, 5, 6)).astype(np.float32))
        with_proj = flow.velocity(Weights(projected), config, x, t, context, rope, proj)
        plain = flow.velocity(Weights(shifted), config, x, t, context, rope)
        np.testing.assert_allclose(np.array(with_proj), np.array(plain), rtol=1e-4, atol=1e-5)
        projected['blocks.0.cross_attn.proj_linear.weight'] = mx.array(rng.standard_normal((8, 6)).astype(np.float32))
        changed = flow.velocity(Weights(projected), config, x, t, context, rope, proj)
        self.assertGreater(float(mx.abs(changed - with_proj).max()), 1e-3)

    def test_conversion_writes_every_component_its_defaults_and_provenance(self):
        import mlx.core as mx
        from mlx_agent.gguf import scan_mlx_outputs
        convert = port('pixal3d.convert').convert

        def tensors(path):
            mx.save_safetensors(str(path), {'proj.weight': mx.ones((4, 64)), 'conv.weight': mx.ones((2, 3, 3, 3, 2))})

        with TemporaryDirectory() as folder:
            source, sources, downloads = write_source(Path(folder), tensors=tensors)
            models = Path(folder) / 'models'
            convert(source, models / 'Pixal3D-MLX-32bit', sources, downloads)
            convert(source, models / 'Pixal3D-MLX-8bit', sources, downloads, quantize=True)
            full = json.loads((models / 'Pixal3D-MLX-32bit' / 'config.json').read_text())
            quantized = models / 'Pixal3D-MLX-8bit'
            flow = mx.load(str(quantized / 'shape_flow_1024.safetensors'))
            camera = mx.load(str(quantized / 'moge.safetensors'))
            upsampler = mx.load(str(quantized / 'naf.safetensors'))
            names = sorted(path.name for path in quantized.iterdir())
            listed = {item['name']: item for item in scan_mlx_outputs([models])}
            leftovers = [path.name for path in models.iterdir() if path.name.startswith('.')]
        self.assertEqual(full['model_type'], 'pixal3d')
        self.assertEqual(full['mesh'], {'resolutions': [1024, 1536], 'resolution': 1536, 'steps': 10, 'guidance': 6.0})
        self.assertEqual(full['components']['moge']['remap_output'], 'exp')
        self.assertEqual(names, ['config.json', 'image_encoder.safetensors', 'mlx-converter.json', 'moge.safetensors',
                                 'naf.safetensors', 'shape_decoder.safetensors', 'shape_flow_1024.safetensors',
                                 'shape_flow_512.safetensors', 'ss_decoder.safetensors', 'ss_flow.safetensors'])
        self.assertIn('proj.scales', flow)
        self.assertEqual(sorted(camera), ['encoder.backbone.cls_token', 'neck.input_blocks.0.weight'])
        self.assertEqual(camera['encoder.backbone.cls_token'].tolist(), [[[0.25, 0.25]]])
        self.assertEqual(sorted(upsampler), ['image_encoder.encoder.0.weight', 'image_encoder.rope.periods',
                                             'image_encoder.sem_encoder.0.weight'])
        self.assertEqual(listed['Pixal3D-MLX-8bit']['provenance']['q_bits'], 8)
        self.assertEqual(listed['Pixal3D-MLX-32bit']['provenance']['source']['downloads'], {NAF: str(downloads[NAF])})
        self.assertEqual(leftovers, [])


if __name__ == '__main__':
    unittest.main()
