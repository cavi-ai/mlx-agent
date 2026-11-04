"""Parity of the TripoSG port against the PyTorch reference, in float32 on the CPU unless noted.

Optional suite, outside the default run: it needs torch, transformers, diffusers, mlx and a diffusers TripoSG checkpoint.
    TRIPOSG_WEIGHTS   diffusers checkpoint directory (image_encoder_dinov2, transformer, vae, scheduler, ...)
    TRIPOSG_UPSTREAM  checkout of the upstream TripoSG repository (its `triposg` package)
    TRIPOSG_CONVERTED converted MLX directory (octree test) and TRIPOSG_LATENTS a saved (1, 2048, 64) latent .npy
Run: PYTHONPATH=src python -m unittest discover -s port_tests -t . -v
"""

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from mlx_agent.backends import PORTS_DIR

WEIGHTS = os.environ.get("TRIPOSG_WEIGHTS")
UPSTREAM = os.environ.get("TRIPOSG_UPSTREAM")
CONVERTED = os.environ.get("TRIPOSG_CONVERTED")
LATENTS = os.environ.get("TRIPOSG_LATENTS")
BOUND = 1e-4


def have(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


def load_port():
    """The port package under its own name so it never shadows the upstream `triposg` package."""
    name = "mlx_triposg_port"
    if name in sys.modules:
        return sys.modules[name]
    root = PORTS_DIR / "mlx-3d" / "triposg"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def use_upstream():
    sys.modules.setdefault("diso", types.SimpleNamespace(DiffDMC=None))
    if UPSTREAM and UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)


def sub(name):
    return importlib.import_module(load_port().__name__ + "." + name)


def relative_error(ours, reference):
    import numpy as np

    a, b = np.asarray(ours, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b)), float(np.abs(a - b).max() / np.abs(b).max())


def checkpoint(component, filename="diffusion_pytorch_model.safetensors"):
    """The component's float32 tensors in the layout the port reads, as the converter writes them."""
    import mlx.core as mx

    values = mx.load(str(Path(WEIGHTS) / component / filename))
    return sub("weights").Weights({name: value.astype(mx.float32) for name, value in values.items()})


@unittest.skipUnless(WEIGHTS and have("torch", "transformers", "mlx", "numpy", "PIL"), "TRIPOSG_WEIGHTS, torch, transformers and mlx are required")
class EncoderParityTests(unittest.TestCase):
    def test_preprocessing_and_dinov2_match_transformers_in_float32_on_cpu(self):
        import mlx.core as mx
        import numpy as np
        import torch
        from PIL import Image
        from transformers import BitImageProcessor, Dinov2Model

        mx.set_default_device(mx.cpu)
        root = Path(WEIGHTS)
        reference = Dinov2Model.from_pretrained(str(root / "image_encoder_dinov2"), dtype=torch.float32).eval()
        processor = BitImageProcessor.from_pretrained(str(root / "feature_extractor_dinov2"))
        config = json.loads((root / "image_encoder_dinov2" / "config.json").read_text(encoding="utf-8"))
        processor_config = json.loads((root / "feature_extractor_dinov2" / "preprocessor_config.json").read_text(encoding="utf-8"))
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as folder:
            for shape in ((300, 420, 3), (420, 300, 3), (512, 512, 3)):
                image = Image.fromarray(rng.integers(0, 255, shape, dtype=np.uint8))
                path = Path(folder) / ("image%dx%d.png" % shape[:2])
                image.save(path)
                ours_pixels = np.array(sub("dinov2").preprocess(str(path), processor_config)).transpose(0, 3, 1, 2)
                reference_pixels = processor(image, return_tensors="pt").pixel_values.numpy()
                difference = float(np.abs(ours_pixels - reference_pixels).max())
                print("preprocess %s max abs pixel difference %.3g" % (shape[:2], difference))
                self.assertEqual(ours_pixels.shape, reference_pixels.shape)
                self.assertLess(difference, 1e-6)
        with torch.no_grad():
            expected = reference(torch.from_numpy(reference_pixels)).last_hidden_state.numpy()
        actual = np.array(sub("dinov2").encode(checkpoint("image_encoder_dinov2", "model.safetensors"), config,
                                               mx.array(reference_pixels.transpose(0, 2, 3, 1))))
        rel_l2, rel_peak = relative_error(actual, expected)
        print("dinov2 relative L2 %.3g peak %.3g" % (rel_l2, rel_peak))
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(rel_l2, BOUND)
        self.assertLess(rel_peak, BOUND)


@unittest.skipUnless(WEIGHTS and UPSTREAM and have("torch", "diffusers", "mlx", "numpy"), "TRIPOSG_WEIGHTS, TRIPOSG_UPSTREAM, torch, diffusers and mlx are required")
class SamplerParityTests(unittest.TestCase):
    def test_flow_schedule_and_euler_update_equal_the_reference_scheduler(self):
        import numpy as np
        import torch

        use_upstream()
        from triposg.schedulers.scheduling_rectified_flow import RectifiedFlowScheduler

        scheduler = json.loads((Path(WEIGHTS) / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
        rng = np.random.default_rng(5)
        for steps in (1, 4, 50):
            with self.subTest(steps=steps):
                reference = RectifiedFlowScheduler(num_train_timesteps=1000, shift=1.0)
                reference.set_timesteps(steps)
                schedule = sub("pipeline").flow_schedule(scheduler, steps)
                timesteps = np.array([timestep for timestep, _ in schedule], dtype=np.float32)
                self.assertTrue(np.array_equal(timesteps, reference.timesteps.numpy()))
                sample = rng.standard_normal((1, 8, 4)).astype(np.float32)
                worst = 0.0
                for index, (_, delta) in enumerate(schedule):
                    velocity = rng.standard_normal(sample.shape).astype(np.float32)
                    expected = reference.step(torch.from_numpy(velocity), reference.timesteps[index],
                                              torch.from_numpy(sample), return_dict=False)[0].numpy()
                    actual = sample + np.float32(delta) * velocity
                    worst = max(worst, float(np.abs(actual - expected).max()))
                    sample = expected
                print("schedule steps %d timesteps equal, largest Euler update difference %.3g" % (steps, worst))
                self.assertLess(worst, 1e-6)

    def test_transformer_and_vae_match_in_float32_on_cpu(self):
        import mlx.core as mx
        import numpy as np
        import torch

        use_upstream()
        from triposg.models.autoencoders import TripoSGVAEModel
        from triposg.models.transformers import TripoSGDiTModel

        mx.set_default_device(mx.cpu)
        root = Path(WEIGHTS)
        transformer = sub("transformer")
        dit_config = json.loads((root / "transformer" / "config.json").read_text(encoding="utf-8"))
        vae_config = json.loads((root / "vae" / "config.json").read_text(encoding="utf-8"))
        rng = np.random.default_rng(1234)
        x = rng.standard_normal((2, 2048, 64)).astype(np.float32)
        cond = rng.standard_normal((1, 257, 1024)).astype(np.float32)
        enc = np.concatenate([np.zeros_like(cond), cond])
        z = rng.standard_normal((1, 2048, 64)).astype(np.float32)
        points = rng.uniform(-1.005, 1.005, (1, 4096, 3)).astype(np.float32)

        reference = TripoSGDiTModel.from_pretrained(str(root / "transformer"), torch_dtype=torch.float32).eval()
        with torch.no_grad():
            expected = reference(torch.from_numpy(x), torch.full((2,), 360.0), encoder_hidden_states=torch.from_numpy(enc),
                                 return_dict=False)[0].numpy()
        del reference
        actual = np.array(transformer.denoise(checkpoint("transformer"), dit_config, mx.array(x), [360.0, 360.0], mx.array(enc)))
        rel_l2, rel_peak = relative_error(actual, expected)
        print("transformer relative L2 %.3g peak %.3g" % (rel_l2, rel_peak))
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(rel_l2, BOUND)
        self.assertLess(rel_peak, BOUND)

        reference = TripoSGVAEModel.from_pretrained(str(root / "vae"), torch_dtype=torch.float32).eval()
        with torch.no_grad():
            expected = reference.decode(torch.from_numpy(z), sampled_points=torch.from_numpy(points)).sample.numpy()
        del reference
        actual = np.array(transformer.Decoder(checkpoint("vae"), vae_config, mx.array(z))(mx.array(points)))
        rel_l2, rel_peak = relative_error(actual, expected)
        print("vae decoder relative L2 %.3g peak %.3g, sign agreement %.6f" % (rel_l2, rel_peak, float((np.sign(actual) == np.sign(expected)).mean())))
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(rel_l2, BOUND)
        self.assertLess(rel_peak, BOUND)


@unittest.skipUnless(UPSTREAM and CONVERTED and LATENTS and have("torch", "mlx", "numpy", "skimage", "scipy", "einops"),
                     "TRIPOSG_UPSTREAM, TRIPOSG_CONVERTED, TRIPOSG_LATENTS, torch, mlx, scikit-image and einops are required")
class OctreeParityTests(unittest.TestCase):
    def test_extraction_queries_the_same_points_and_builds_the_same_grid(self):
        import mlx.core as mx
        import numpy as np
        import torch

        use_upstream()
        from triposg import inference_utils as reference

        mx.set_default_device(mx.gpu)
        pipeline, octree, geometry, transformer, weights = (sub(name) for name in ("pipeline", "octree", "geometry", "transformer", "weights"))
        config = json.loads((Path(CONVERTED) / "config.json").read_text(encoding="utf-8"))
        vae = weights.Weights(mx.load(str(Path(CONVERTED) / "vae.safetensors")), **config.get("quantization", {}))
        query = pipeline.FieldQuery(transformer.Decoder(vae, config["components"]["vae"], mx.array(np.load(LATENTS))))
        reference_counts, captured = [], {}

        def reference_query(x):
            points = x.reshape(-1, 3).numpy()
            reference_counts.append(len(points))
            return torch.from_numpy(query(points)).reshape(1, -1, 1)

        original = reference.expand_edge_region_fast

        def expand_on_cpu(edge_coords, grid_size):
            zeros = torch.zeros
            reference.torch.zeros = lambda *a, **k: zeros(*a, **{**k, "device": "cpu", "dtype": torch.float32})
            try:
                return original(edge_coords, grid_size)
            finally:
                reference.torch.zeros = zeros

        class Capture:
            @staticmethod
            def marching_cubes(volume, level, method="lewiner"):
                from skimage import measure

                captured["grid"] = volume.copy()
                return measure.marching_cubes(volume, level, method=method)

        reference.expand_edge_region_fast, reference.measure = expand_on_cpu, Capture
        try:
            with torch.no_grad():
                expected = reference.hierarchical_extract_geometry(
                    reference_query, torch.device("cpu"), bounds=octree.DEFAULT_BOUNDS, dense_octree_depth=8, hierarchical_octree_depth=9)
        finally:
            reference.expand_edge_region_fast = original
        grid, counts = octree.extract_grid(query, octree.DEFAULT_BOUNDS, 8, 9)
        mismatches = int((grid.astype(np.float32) != captured["grid"].astype(np.float32)).sum())
        vertices, faces = geometry.marching_cubes(grid, octree.DEFAULT_BOUNDS, 9)
        print("octree queries %s reference %s, mismatching cells %d of %d, faces %d vs %d, largest vertex difference %.3g" % (
            counts, reference_counts, mismatches, grid.size, len(faces), len(expected[0][1]),
            float(np.abs(vertices - expected[0][0]).max()) if vertices.shape == expected[0][0].shape else float("nan")))
        self.assertEqual(counts, reference_counts)
        self.assertEqual(mismatches, 0)
        self.assertTrue(np.array_equal(faces, expected[0][1]))
        self.assertTrue(np.allclose(vertices, expected[0][0], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
