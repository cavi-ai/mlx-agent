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

    def test_adaptive_extraction_recovers_a_sphere(self):
        import mlx.core as mx
        import numpy as np
        from triposg.geometry import extract_mesh
        vertices, faces = extract_mesh(lambda points: mx.sum(points * points, axis=-1) - .25, resolution=64)
        self.assertTrue(len(faces) > 100)
        np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), .5, atol=.004)

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

if __name__ == '__main__': unittest.main()
