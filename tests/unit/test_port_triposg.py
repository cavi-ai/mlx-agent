"""Portable conversion refusals and native MLX TripoSG regression coverage."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.backends import PORTS_DIR

PORT = PORTS_DIR / 'mlx-3d'
sys.path.insert(0, str(PORT))
HAS_MLX = importlib.util.find_spec('mlx') is not None


class TripoSGConversionTests(unittest.TestCase):
    def test_refuses_incomplete_source_without_creating_output(self):
        from triposg.convert import convert
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'model_index.json').write_text(json.dumps({'_class_name': 'TripoSGPipeline'}))
            with self.assertRaises(ValueError):
                convert(root, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_refuses_unsupported_pipeline(self):
        from triposg.convert import convert
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'model_index.json').write_text(json.dumps({'_class_name': 'OtherPipeline'}))
            with self.assertRaisesRegex(ValueError, 'TripoSGPipeline'):
                convert(root, root / 'output')


@unittest.skipUnless(HAS_MLX, 'requires MLX on Apple Silicon')
class TripoSGNativeTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        self.addCleanup(mx.set_default_device, previous)

    def test_checkpoint_scheduler_preserves_positive_rectified_flow(self):
        from triposg.pipeline import flow_schedule
        self.assertEqual(flow_schedule({'_class_name':'RectifiedFlowScheduler','num_train_timesteps':1000},4),
                         [(1000, .25),(750,.25),(500,.25),(250,.25)])
        schedule = flow_schedule({'_class_name':'FlowMatchEulerDiscreteScheduler','num_train_timesteps':1000},2)
        self.assertAlmostEqual(schedule[0][1], -.999)
        self.assertAlmostEqual(schedule[1][1], -.001)

    def test_native_components_match_reference_outputs(self):
        import mlx.core as mx
        import numpy as np
        from triposg.weights import Weights
        from triposg.transformer import denoise, Decoder
        from triposg.dinov2 import encode
        root = Path(__file__).parents[1] / 'fixtures' / 'triposg'
        configs = json.loads((root / 'parity-config.json').read_text())
        with np.load(root / 'parity.npz') as fixture:
            for device in (mx.cpu, mx.gpu):
                mx.set_default_device(device)
                weights = {}
                for name in configs:
                    with np.load(root / (name + '-weights.npz')) as values:
                        weights[name] = Weights({key: mx.array(value) for key, value in values.items()})
                outputs = {
                    'transformer': denoise(weights['transformer'], configs['transformer'], mx.array(fixture['latents']),
                                           fixture['times'].tolist(), mx.array(fixture['features'])),
                    'vae': Decoder(weights['vae'], configs['vae'], mx.array(fixture['latents']))(mx.array(fixture['queries'])),
                    'dino': encode(weights['dino'], configs['dino'], mx.array(fixture['pixels'].transpose(0, 2, 3, 1))),
                }
                for name, output in outputs.items():
                    with self.subTest(component=name, device=str(device)):
                        np.testing.assert_allclose(np.array(output), fixture[name], atol=1e-5 if device == mx.cpu else 2e-3,
                                                   rtol=2e-3)

    def test_seeded_noise_reproduces_the_reference_generator(self):
        import numpy as np
        from triposg.pipeline import initial_latents
        noise = np.array(initial_latents(42, 64))
        self.assertEqual((noise.shape, noise.dtype), ((1, 2048, 64), np.float32))
        np.testing.assert_allclose(noise[0, 0, :5], [0.30471708, -1.03998411, 0.7504512, 0.94056472, -1.95103519], atol=1e-7)

    def test_field_query_keeps_point_order_across_chunks_and_accepts_no_points(self):
        import mlx.core as mx
        import numpy as np
        from triposg.pipeline import FieldQuery
        points = np.random.default_rng(3).uniform(-1, 1, (10, 3)).astype('float16')
        query = FieldQuery(lambda batch: mx.sum(batch.astype(mx.float32), axis=-1), chunk=3)
        np.testing.assert_array_equal(query(points), points.astype('float32').sum(axis=-1))
        self.assertEqual(query(points[:0]).shape, (0,))

    def test_attention_preserves_the_reference_head_interleaving(self):
        import mlx.core as mx
        import numpy as np
        from triposg.weights import Weights
        from triposg.transformer import attention
        rng = np.random.default_rng(8)
        values = {f'a.{name}.weight': rng.normal(size=(8, 8)).astype('float32') * .1
                  for name in ('to_q', 'to_k', 'to_v', 'to_out.0')}
        values['a.to_out.0.bias'] = np.zeros(8, dtype='float32')
        x = rng.normal(size=(1, 3, 8)).astype('float32')
        qkv = np.concatenate([x @ values[f'a.to_{name}.weight'].T for name in ('q','k','v')], axis=-1)
        q, k, v = np.split(qkv.reshape(1, 3, 2, 12), 3, axis=-1)
        q,k,v = [a.transpose(0,2,1,3) for a in (q,k,v)]
        logits = (q @ k.transpose(0,1,3,2)) / 2
        probs = np.exp(logits-logits.max(axis=-1, keepdims=True)); probs /= probs.sum(axis=-1,keepdims=True)
        expected = (probs @ v).transpose(0,2,1,3).reshape(1,3,8) @ values['a.to_out.0.weight'].T
        actual = attention(Weights({k: mx.array(v) for k,v in values.items()}), 'a', mx.array(x), 2)
        np.testing.assert_allclose(np.array(actual), expected, atol=1e-6)

    def test_quantized_linear_matches_dequantized_weights(self):
        import mlx.core as mx
        import numpy as np
        from triposg.weights import Weights
        weight = mx.random.normal((16,64))
        q,s,b = mx.quantize(weight, group_size=64, bits=8)
        x = mx.ones((1,2,64))
        store = Weights({'a.weight': q, 'a.scales':s, 'a.biases':b}, bits=8)
        expected = x @ mx.dequantize(q,s,b,group_size=64,bits=8).T
        np.testing.assert_allclose(np.array(store.linear('a',x)),np.array(expected),atol=1e-4)

    def test_glb_writer_never_replaces_an_existing_file(self):
        import numpy as np
        from triposg.geometry import write_glb
        with TemporaryDirectory() as folder:
            out = Path(folder)/'mesh.glb'
            vertices = np.array([[0,0,0],[1,0,0],[0,1,0]],dtype='float32')
            faces = np.array([[0,1,2]],dtype='uint32')
            write_glb(out,vertices,faces)
            original = out.read_bytes()
            self.assertEqual(original[:4],b'glTF')
            with self.assertRaises(FileExistsError): write_glb(out,vertices,faces)
            self.assertEqual(out.read_bytes(),original)

HAS_GEOMETRY = all(importlib.util.find_spec(name) for name in ('numpy', 'scipy', 'skimage', 'fast_simplification'))


def sphere(points):
    import numpy as np
    return 200 * (.5 - np.linalg.norm(points.astype('float32'), axis=1))


def torus(points):
    import numpy as np
    x, y, z = points.astype('float32').T
    return -200 * (np.sqrt((np.hypot(x, y) - .5) ** 2 + z ** 2) - .2)


def euler_characteristic(faces):
    import numpy as np
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    return len(np.unique(faces)) - len(edges) + len(faces)


@unittest.skipUnless(HAS_GEOMETRY, 'requires numpy, scipy, scikit-image and fast-simplification')
class TripoSGGeometryTests(unittest.TestCase):
    def test_octree_queries_a_dense_grid_then_only_cells_near_the_surface(self):
        import numpy as np
        from triposg.geometry import extract_mesh
        for depth in (5, 6, 7):
            with self.subTest(depth=depth):
                seconds = {}
                vertices, faces, queries = extract_mesh(sphere, depth, seconds=seconds)
                self.assertEqual(len(queries), 2)
                self.assertEqual(queries[0], (2 ** (depth - 1)) ** 3)
                self.assertEqual(queries[1] % 8, 0)
                self.assertLess(queries[1], (2 ** depth) ** 3 // (2 if depth < 6 else 4))
                self.assertGreater(queries[1], 8 * 4 * np.pi * .25 / (2.01 / 2 ** (depth - 1)) ** 2)
                self.assertEqual(sorted(seconds), ['decode_dense', 'decode_refine', 'marching_cubes'])
                self.assertTrue(all(value >= 0 for value in seconds.values()))
                self.assertLess(abs(np.linalg.norm(vertices, axis=1) - .5).max(), 2.01 / 2 ** depth)
                self.assertEqual(euler_characteristic(faces), 2)

    def test_octree_meshes_a_torus_and_reports_its_topology(self):
        import numpy as np
        from triposg.geometry import extract_mesh
        vertices, faces, _ = extract_mesh(torus, 7)
        self.assertEqual(euler_characteristic(faces), 0)
        ring, tube = np.hypot(vertices[:, 0], vertices[:, 1]), vertices[:, 2]
        self.assertLess(abs(np.hypot(ring - .5, tube) - .2).max(), 2.01 / 2 ** 7)

    def test_a_field_without_a_zero_crossing_is_refused(self):
        import numpy as np
        from triposg.geometry import extract_mesh
        with self.assertRaisesRegex(ValueError, 'zero crossing'):
            extract_mesh(lambda points: np.full(len(points), 3.0), 5)

    def test_decimation_caps_the_face_count_and_keeps_the_surface(self):
        import numpy as np
        from triposg.geometry import decimate, extract_mesh
        vertices, faces, _ = extract_mesh(sphere, 7)
        self.assertGreater(len(faces), 5000)
        kept = decimate(vertices, faces, len(faces))
        self.assertIs(kept[0], vertices)
        self.assertIs(kept[1], faces)
        reduced_vertices, reduced_faces = decimate(vertices, faces, 2000)
        self.assertLessEqual(len(reduced_faces), 2000)
        self.assertGreater(len(reduced_faces), 1500)
        self.assertLess(abs(np.linalg.norm(reduced_vertices, axis=1) - .5).max(), .01)
        self.assertEqual(euler_characteristic(reduced_faces), 2)


@unittest.skipUnless(HAS_MLX and HAS_GEOMETRY, 'requires MLX, numpy, scipy, scikit-image and fast-simplification')
class TripoSGGenerationTests(unittest.TestCase):
    """Generation through the real pipeline on the small random fixture checkpoint."""

    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        from PIL import Image
        cls.previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        cls.folder = TemporaryDirectory()
        root = Path(cls.folder.name)
        fixtures = Path(__file__).parents[1] / 'fixtures' / 'triposg'
        configs = json.loads((fixtures / 'parity-config.json').read_text())
        for name, source in (('transformer', 'transformer'), ('vae', 'vae'), ('image_encoder_dinov2', 'dino')):
            with np.load(fixtures / (source + '-weights.npz')) as values:
                mx.save_safetensors(str(root / (name + '.safetensors')), {key: mx.array(values[key]) for key in values.files})
        (root / 'config.json').write_text(json.dumps({
            'model_type': 'triposg',
            'components': {'transformer': configs['transformer'], 'vae': configs['vae'], 'image_encoder_dinov2': configs['dino']},
            'scheduler': {'_class_name': 'RectifiedFlowScheduler', 'num_train_timesteps': 1000, 'shift': 1},
            'preprocessor': {'do_resize': True, 'size': {'shortest_edge': 32}, 'crop_size': {'height': 28, 'width': 28},
                             'do_center_crop': True, 'resample': 3, 'do_rescale': True, 'rescale_factor': 1 / 255,
                             'do_normalize': True, 'image_mean': [.485, .456, .406], 'image_std': [.229, .224, .225]}}))
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (40, 48, 3), dtype=np.uint8)).save(root / 'input.png')
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        import mlx.core as mx
        mx.set_default_device(cls.previous)
        cls.folder.cleanup()

    def generate(self, name, **settings):
        from triposg.pipeline import generate, load
        return generate(load(self.root), str(self.root / 'input.png'), str(self.root / name), steps=2, seed=3, **settings)

    def test_result_reports_stage_seconds_queries_and_a_capped_mesh(self):
        result = self.generate('capped.glb', resolution=32, max_faces=500)
        self.assertEqual(list(result['stage_seconds']), ['conditioning', 'sampling', 'decode_dense', 'decode_refine',
                                                         'marching_cubes', 'decimation', 'export'])
        self.assertEqual(result['queries'], [16 ** 3, 32 ** 3])
        self.assertEqual(result['resolution'], 32)
        self.assertLessEqual(result['faces'], 500)
        self.assertGreater(result['faces'], 400)
        self.assertEqual((self.root / 'capped.glb').read_bytes()[:4], b'glTF')

    def test_resolution_sets_the_final_grid_and_the_cap_only_bites_above_it(self):
        coarse = self.generate('coarse.glb', resolution=32, max_faces=10_000_000)
        fine = self.generate('fine.glb', resolution=64, max_faces=10_000_000)
        self.assertEqual(fine['queries'][0], 32 ** 3)
        self.assertEqual(fine['queries'][1], 64 ** 3)
        self.assertGreater(fine['faces'], 2 * coarse['faces'])

    def test_runner_adds_the_load_stage_to_the_port_result(self):
        import contextlib
        import io
        from mlx_agent.mesh_runner import main
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main(['--module', 'triposg', '--model', str(self.root), '--image', str(self.root / 'input.png'),
                  '--out', str(self.root / 'runner.glb'), '--steps', '2', '--seed', '3', '--resolution', '32',
                  '--guidance', '7', '--max-faces', '500'])
        result = json.loads(buffer.getvalue())
        self.assertEqual(list(result['stage_seconds'])[:2], ['load', 'conditioning'])
        self.assertEqual(result['stage_seconds']['load'], result['load_seconds'])
        self.assertLessEqual(result['faces'], 500)


if __name__ == '__main__': unittest.main()
