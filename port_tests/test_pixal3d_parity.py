"""Parity of the Pixal3D port against the PyTorch references, in float32 on the CPU.

Optional suite, outside the default run: it needs torch, torchvision, transformers, scipy, mlx and checkouts of
    PIXAL3D_UPSTREAM  the Pixal3D repository (its `pixal3d` package)
    NAF_UPSTREAM      valeoai/NAF
    MOGE_UPSTREAM     microsoft/MoGe
The extensions the references import are replaced by plain PyTorch definitions, as in the TRELLIS.2 suite;
NATTEN's neighborhood attention becomes dense attention over each query's window in its dilation subgrid.
Run: PYTHONPATH=src python -m unittest discover -s port_tests -t . -v
"""

import importlib
import importlib.util
import math
import os
import sys
import types
import unittest
from pathlib import Path

from port_tests.test_trellis2_parity import (BOUND, FLOW, _dense_attention_backend, _dense_conv_backend, _shell, have,
                                             randomize, relative_error, sub, weights)

PIXAL3D = os.environ.get("PIXAL3D_UPSTREAM")
NAF = os.environ.get("NAF_UPSTREAM")
MOGE = os.environ.get("MOGE_UPSTREAM")


def _absent(name):
    """A module whose every attribute raises when called: imported at load time, never used by the tested code."""
    def unavailable(*args, **kwargs):
        raise NotImplementedError("reference operation outside the parity suite")

    def attribute(key):
        if key.startswith("__"):
            raise AttributeError(key)
        return unavailable

    module = types.ModuleType(name)
    module.__getattr__ = attribute
    return module


def use_pixal3d():
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    root = Path(PIXAL3D) / "pixal3d"
    # Package shells skip the renderers, pipelines, trainers and CUDA extensions the tested modules never call.
    _shell("pixal3d", root)
    for package in ("trainers", "trainers/flow_matching", "trainers/flow_matching/mixins"):
        _shell("pixal3d." + package.replace("/", "."), root / package)
    _dense_conv_backend()
    _dense_attention_backend()

    class EasyDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    sys.modules.setdefault("easydict", types.SimpleNamespace(EasyDict=EasyDict))


def _neighborhood_attention():
    """NATTEN's 2D neighborhood attention, legacy API: the queries of residue r along an axis attend to the kernel
    positions of residue r centered on them, the window shifted inside the subgrid at its borders."""
    import torch

    def axis(length, kernel, dilation):
        windows = []
        for index in range(length):
            residue, position = index % dilation, index // dilation
            count = len(range(residue, length, dilation))
            start = min(max(position - kernel // 2, 0), count - kernel)
            windows.append([(start + offset) * dilation + residue for offset in range(kernel)])
        return torch.tensor(windows)

    def neighbors(x, kernel_size, dilation):
        batch, heads, height, width, dim = x.shape
        rows, cols = axis(height, kernel_size[0], dilation[0]), axis(width, kernel_size[1], dilation[1])
        gathered = x[:, :, rows][:, :, :, :, cols]  # b n h k w k d
        return gathered.permute(0, 1, 2, 4, 3, 5, 6).reshape(batch, heads, height, width, -1, dim)

    def na2d_qk(q, k, kernel_size, dilation):
        return (q.unsqueeze(-2) * neighbors(k, kernel_size, dilation)).sum(-1)

    def na2d_av(attn, v, kernel_size, dilation):
        return (attn.unsqueeze(-1) * neighbors(v, kernel_size, dilation)).sum(-2)

    return types.SimpleNamespace(na2d_qk=na2d_qk, na2d_av=na2d_av)


def naf_class():
    root = Path(NAF) / "src"
    # NAF's top-level package is `src`; the shell keeps it apart from this repository's `src` folder.
    if getattr(sys.modules.get("src"), "__path__", None) != [str(root)]:
        for name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
            del sys.modules[name]
    _shell("src", root)
    _shell("src.model", root / "model")
    # The stand-in leaves with the import, so other packages still see NATTEN as missing.
    borrowed = not _installed("natten")
    if borrowed:
        functional = _neighborhood_attention()
        sys.modules["natten"] = types.SimpleNamespace(functional=functional)
        sys.modules["natten.functional"] = functional
    try:
        from src.model.naf import NAF as Reference
    finally:
        if borrowed:
            del sys.modules["natten"], sys.modules["natten.functional"]
    return Reference


def _installed(name):
    return name in sys.modules or importlib.util.find_spec(name) is not None


def moge_model(config):
    """The MoGe-2 reference model; ``utils3d`` keeps only the intrinsics matrix inference returns."""
    import torch

    def intrinsics_from_focal_center(fx, fy, cx, cy):
        zero, one = torch.zeros_like(fx), torch.ones_like(fx)
        rows = [fx, zero, cx.expand_as(fx), zero, fy, cy.expand_as(fx), zero, zero, one]
        return torch.stack(rows, dim=-1).unflatten(-1, (3, 3))

    _shell("moge", Path(MOGE) / "moge")
    if not _installed("utils3d") and not _installed("utils3d_moge"):
        utils3d = _absent("utils3d")
        utils3d.pt = types.SimpleNamespace(intrinsics_from_focal_center=intrinsics_from_focal_center)
        sys.modules["utils3d"] = utils3d
    # The geometry module imports OpenCV for helpers inference never calls; the stand-in leaves with the import,
    # so other packages still see OpenCV as missing.
    borrowed = not _installed("cv2")
    if borrowed:
        sys.modules["cv2"] = _absent("cv2")
    try:
        from moge.model.v2 import MoGeModel
    finally:
        if borrowed:
            del sys.modules["cv2"]
    return MoGeModel(**config)


PROJ_FLOW = dict(FLOW, image_attn_mode="proj", proj_in_channels=24)


@unittest.skipUnless(PIXAL3D and have("torch", "torchvision", "transformers", "mlx", "numpy"),
                     "PIXAL3D_UPSTREAM, torch and mlx are required")
class ProjectionParityTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx

        use_pixal3d()
        mx.set_default_device(mx.cpu)

    def test_projected_features_match_the_reference_grid_for_patch_and_upsampled_maps(self):
        import mlx.core as mx
        import numpy as np
        import torch
        from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGrid

        projection, flow = sub("pixal3d.projection"), sub("trellis2.flow")
        resolution, image_size, angle = 6, 64, 0.62
        distance = projection.distance(angle)
        grid = ProjGrid(grid_resolution=resolution, image_resolution=image_size)
        torch.manual_seed(15)
        for side in (4, 32):
            features = torch.randn(1, side, side, 8)
            with torch.no_grad():
                reference = grid(features, torch.tensor([angle]), torch.tensor([distance]), torch.tensor([1.0]))[0]
            coords = flow.grid_positions(resolution)
            positions = projection.pixels(projection.grid_points(coords, resolution), angle, distance, image_size)
            ours = projection.sample(mx.array(features[0].numpy()), positions, image_size)
            self.assertLess(relative_error(np.array(ours), reference.numpy()), BOUND, side)


@unittest.skipUnless(PIXAL3D and have("torch", "torchvision", "transformers", "mlx", "numpy"),
                     "PIXAL3D_UPSTREAM, torch and mlx are required")
class ProjectedFlowParityTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx

        use_pixal3d()
        mx.set_default_device(mx.cpu)

    def _structure_flow(self):
        from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel

        return randomize(SparseStructureFlowModel(**PROJ_FLOW), 16)

    def test_structure_flow_velocity_with_projected_features_matches_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch

        model = self._structure_flow()
        torch.manual_seed(17)
        x, cond, proj = torch.randn(2, 8, 4, 4, 4), torch.randn(2, 5, 32), torch.randn(2, 64, 24)
        t = torch.tensor([973.0, 120.0])
        with torch.no_grad():
            reference = model(x, t, {"global": cond, "proj": proj}).permute(0, 2, 3, 4, 1).reshape(2, 64, 8).numpy()
        flow = sub("trellis2.flow")
        tokens = mx.array(x.permute(0, 2, 3, 4, 1).reshape(2, 64, 8).numpy())
        rope = flow.rope_tables(flow.grid_positions(4), FLOW["model_channels"] // FLOW["num_heads"])
        ours = flow.velocity(weights(model), PROJ_FLOW, tokens, mx.array(t.numpy()), mx.array(cond.numpy()), rope,
                             proj=mx.array(proj.numpy()))
        self.assertLess(relative_error(np.array(ours), reference), BOUND)

    def test_guided_sampling_with_zero_projection_for_the_negative_branch_matches_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch
        from pixal3d.pipelines.samplers import FlowEulerGuidanceIntervalSampler

        model = self._structure_flow()
        torch.manual_seed(18)
        noise, cond, proj = torch.randn(1, 8, 4, 4, 4), torch.randn(1, 5, 32), torch.randn(1, 64, 24)
        params = dict(steps=6, guidance_strength=7.5, guidance_rescale=0.7, guidance_interval=[0.6, 1.0], rescale_t=5.0)
        positive = {"global": cond, "proj": proj}
        negative = {"global": torch.zeros_like(cond), "proj": torch.zeros_like(proj)}
        with torch.no_grad():
            reference = FlowEulerGuidanceIntervalSampler(1e-5).sample(
                model, noise, positive, negative, verbose=False, **params).samples
        flow = sub("trellis2.flow")
        rope = flow.rope_tables(flow.grid_positions(4), FLOW["model_channels"] // FLOW["num_heads"])
        tokens = mx.array(noise.permute(0, 2, 3, 4, 1).reshape(1, 64, 8).numpy())
        cond_mx = mx.array(cond.numpy())
        ours = flow.sample(weights(model), PROJ_FLOW, tokens, cond_mx, mx.zeros_like(cond_mx), rope, **params,
                           proj=mx.array(proj.numpy()))
        self.assertLess(relative_error(np.array(ours), reference.permute(0, 2, 3, 4, 1).reshape(1, 64, 8).numpy()), BOUND)

    def test_elastic_latent_flow_with_projected_voxel_features_matches_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch
        from pixal3d.models.structured_latent_flow import ElasticSLatFlowModel
        from pixal3d.modules.sparse import SparseTensor

        config = dict(PROJ_FLOW, resolution=8, in_channels=16, out_channels=16)
        model = randomize(ElasticSLatFlowModel(**config), 19)
        generator = np.random.default_rng(20)
        coords = np.unique(generator.integers(0, 8, size=(40, 3)), axis=0).astype(np.int32)
        index = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32), torch.from_numpy(coords)], dim=1)
        torch.manual_seed(21)
        feats, cond, proj = torch.randn(len(coords), 16), torch.randn(1, 5, 32), torch.randn(len(coords), 24)
        with torch.no_grad():
            reference = model(SparseTensor(feats=feats, coords=index), torch.tensor([640.0]),
                              {"global": cond, "proj": SparseTensor(feats=proj, coords=index)}).feats.numpy()
        flow = sub("trellis2.flow")
        rope = flow.rope_tables(coords, config["model_channels"] // config["num_heads"])
        ours = flow.velocity(weights(model), config, mx.array(feats.numpy())[None], mx.array([640.0]),
                             mx.array(cond.numpy()), rope, proj=mx.array(proj.numpy())[None])
        self.assertLess(relative_error(np.array(ours[0]), reference), BOUND)


@unittest.skipUnless(NAF and have("torch", "einops", "mlx", "numpy"), "NAF_UPSTREAM, torch, einops and mlx are required")
class NafParityTests(unittest.TestCase):
    def test_upsampled_features_match_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch

        mx.set_default_device(mx.cpu)
        model = randomize(naf_class()(), 22, scale=0.1)
        torch.manual_seed(23)
        image, features = torch.rand(1, 3, 96, 96), torch.randn(1, 16, 10, 10)
        with torch.no_grad():
            reference = model(image, features, (20, 20)).permute(0, 2, 3, 1).numpy()
        ours = sub("pixal3d.naf").upsample(weights(model), mx.array(image.permute(0, 2, 3, 1).numpy()),
                                           mx.array(features.permute(0, 2, 3, 1).numpy()), (20, 20))
        self.assertLess(relative_error(np.array(ours), reference), BOUND)


MOGE_HEAD = dict(dim_in=[32, 16, 16, 8, 8], dim_res_blocks=[32, 16, 16, 8, 8], num_res_blocks=[0, 1, 1, 1, 0],
                 res_block_in_norm="none", res_block_hidden_norm="none",
                 resamplers=["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"])
MOGE_CONFIG = dict(encoder=dict(backbone="dinov2_vitl14", intermediate_layers=[1, 3], dim_out=32),
                   neck=dict(MOGE_HEAD, dim_in=[34, 2, 2, 2, 2], dim_out=None),
                   points_head=dict(MOGE_HEAD, dim_out=[None, None, None, None, 3]),
                   mask_head=dict(MOGE_HEAD, dim_out=[None, None, None, None, 1]),
                   remap_output="exp", num_tokens_range=[16, 36])


@unittest.skipUnless(MOGE and have("torch", "scipy", "mlx", "numpy"), "MOGE_UPSTREAM, torch, scipy and mlx are required")
class MogeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import torch

        mx.set_default_device(mx.cpu)
        model = moge_model(MOGE_CONFIG)
        # Blocks after the last intermediate layer never reach the features, the point map or the mask.
        model.encoder.backbone.blocks = model.encoder.backbone.blocks[:4]
        randomize(model, 24, scale=0.05)
        with torch.no_grad():
            model.mask_head.output_blocks[-1].bias.fill_(4.0)
        cls.model = model
        torch.manual_seed(25)
        cls.image = torch.rand(1, 3, 50, 70)

    def test_encoder_features_match_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch

        with torch.no_grad():
            reference = self.model.encoder(self.image, 5, 7).permute(0, 2, 3, 1).numpy()
        ours = sub("pixal3d.moge").encode(weights(self.model), MOGE_CONFIG,
                                          mx.array(self.image.permute(0, 2, 3, 1).numpy()), 5, 7)
        self.assertLess(relative_error(np.array(ours), reference), BOUND)

    def test_neck_and_heads_match_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch

        torch.manual_seed(26)
        inputs = [torch.randn(1, 34, 5, 7)] + [torch.randn(1, 2, 5 * 2 ** level, 7 * 2 ** level) for level in range(1, 5)]
        with torch.no_grad():
            neck = self.model.neck(inputs)
            reference = [self.model.points_head(neck)[-1], self.model.mask_head(neck)[-1]]
        moge, w = sub("pixal3d.moge"), weights(self.model)
        ours_neck = moge.conv_stack(w, "neck", MOGE_CONFIG["neck"], [mx.array(x.permute(0, 2, 3, 1).numpy()) for x in inputs])
        for head, expected in zip(("points_head", "mask_head"), reference):
            ours = moge.conv_stack(w, head, MOGE_CONFIG[head], ours_neck)[-1]
            self.assertLess(relative_error(np.array(ours), expected.permute(0, 2, 3, 1).numpy()), BOUND, head)

    def test_camera_field_of_view_matches_reference_inference(self):
        import mlx.core as mx

        reference = self.model.infer(self.image, use_fp16=False, force_projection=False)["intrinsics"][0]
        expected = 2 * math.atan(1 / (2 * float(reference[0, 0])))
        ours = sub("pixal3d.moge").camera_angle_x(weights(self.model), MOGE_CONFIG,
                                                  mx.array(self.image.permute(0, 2, 3, 1).numpy()))
        self.assertTrue(0.05 < expected < 3.0, expected)
        self.assertAlmostEqual(ours, expected, delta=BOUND * expected)


if __name__ == "__main__":
    unittest.main()
