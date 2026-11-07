"""Parity of the TRELLIS.2 port against the PyTorch reference, in float32 on the CPU.

Optional suite, outside the default run: it needs torch, transformers, mlx and a TRELLIS.2 checkout.
    TRELLIS2_UPSTREAM  checkout of microsoft/TRELLIS.2 (its `trellis2` package and `o-voxel`)
The CUDA extensions the reference imports are replaced by plain PyTorch definitions: a dense conv3d for the
submanifold convolution, dense softmax attention per sequence, and a dictionary for the voxel hash map.
Run: PYTHONPATH=src python -m unittest discover -s port_tests -t . -v
"""

import importlib
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

from mlx_agent.backends import PORTS_DIR

UPSTREAM = os.environ.get("TRELLIS2_UPSTREAM")
BOUND = 1e-4
PACKAGE = "mlx_trellis2_ports"


def have(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


def sub(name):
    """A port module, imported from a package shell over the backend's ports folder."""
    if PACKAGE not in sys.modules:
        shell = types.ModuleType(PACKAGE)
        shell.__path__ = [str(PORTS_DIR / "mlx-3d")]
        sys.modules[PACKAGE] = shell
    return importlib.import_module(PACKAGE + "." + name)


def _shell(name, path):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _dense_conv_backend():
    import torch
    import torch.nn.functional as F

    def sparse_submanifold_conv3d(feats, coords, shape, weight, bias, neighbor_cache=None, dilation=(1, 1, 1)):
        co, k1, _, _, ci = weight.shape
        index = coords.long()
        dense = feats.new_zeros((int(shape[0]), ci) + tuple(int(size) for size in shape[2:]))
        dense[index[:, 0], :, index[:, 1], index[:, 2], index[:, 3]] = feats
        out = F.conv3d(dense, weight.permute(0, 4, 1, 2, 3), bias, padding=k1 // 2)
        return out[index[:, 0], :, index[:, 1], index[:, 2], index[:, 3]], None

    spconv = types.SimpleNamespace(sparse_submanifold_conv3d=sparse_submanifold_conv3d,
                                   set_algorithm=lambda *args: None, set_hashmap_ratio=lambda *args: None)
    package = types.ModuleType("flex_gemm")
    package.ops = types.SimpleNamespace(spconv=spconv)
    sys.modules["flex_gemm"] = package
    sys.modules["flex_gemm.ops"] = package.ops
    sys.modules["flex_gemm.ops.spconv"] = spconv


def _dense_attention_backend():
    import torch
    import torch.nn.functional as F

    class BlockDiagonalMask:
        @staticmethod
        def from_seqlens(q_seqlen, kv_seqlen):
            return list(q_seqlen), list(kv_seqlen)

    def memory_efficient_attention(q, k, v, mask):
        outputs, q_start, kv_start = [], 0, 0
        for q_len, kv_len in zip(*mask):
            outputs.append(F.scaled_dot_product_attention(
                q[:, q_start:q_start + q_len].transpose(1, 2), k[:, kv_start:kv_start + kv_len].transpose(1, 2),
                v[:, kv_start:kv_start + kv_len].transpose(1, 2)).transpose(1, 2))
            q_start, kv_start = q_start + q_len, kv_start + kv_len
        return torch.cat(outputs, dim=1)

    ops = types.SimpleNamespace(memory_efficient_attention=memory_efficient_attention,
                                fmha=types.SimpleNamespace(BlockDiagonalMask=BlockDiagonalMask))
    sys.modules["xformers"] = types.SimpleNamespace(ops=ops)
    sys.modules["xformers.ops"] = ops


def _dictionary_hashmap():
    import torch

    def insert(keys, values, coords, *size):
        keys.table = {tuple(row[1:]): index for index, row in enumerate(coords.tolist())}

    def lookup(keys, values, query, *size):
        return torch.tensor([keys.table.get(tuple(row[1:]), 0xFFFFFFFF) for row in query.tolist()], dtype=torch.uint32)

    sys.modules["o_voxel._C"] = types.SimpleNamespace(hashmap_insert_3d_idx_as_val_cuda=insert,
                                                      hashmap_lookup_3d_cuda=lookup)


def _uncalled_mesh_backends():
    """The reference mesh and texture modules import these at load time; decoding never calls them."""
    def unavailable(*args, **kwargs):
        raise NotImplementedError("reference operation outside the parity suite")

    def absent(name):
        def attribute(key):
            if key.startswith("__"):
                raise AttributeError(key)
            return unavailable

        module = types.ModuleType(name)
        module.__getattr__ = attribute
        return module

    for name in ("cumesh", "trimesh"):
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            sys.modules[name] = absent(name)
            if name == "trimesh":
                sys.modules["trimesh.visual"] = sys.modules[name].visual = absent("trimesh.visual")
    grid_sample = types.SimpleNamespace(grid_sample_3d=unavailable)
    sys.modules["flex_gemm"].ops.grid_sample = grid_sample
    sys.modules["flex_gemm.ops.grid_sample"] = grid_sample


def use_upstream():
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
    root = Path(UPSTREAM)
    # Package shells skip the renderers, pipelines and CUDA extensions the tested modules never call.
    _shell("trellis2", root / "trellis2")
    _shell("o_voxel", root / "o-voxel" / "o_voxel")
    _dense_conv_backend()
    _dense_attention_backend()
    _dictionary_hashmap()
    _uncalled_mesh_backends()

    class EasyDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    sys.modules.setdefault("easydict", types.SimpleNamespace(EasyDict=EasyDict))


def randomize(model, seed, scale=0.2):
    """Every parameter random, so zero-initialized layers contribute to the comparison."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * scale)
    return model.eval()


def weights(model):
    import mlx.core as mx

    values = {name: mx.array(value.detach().float().numpy()) for name, value in model.state_dict().items()}
    return sub("triposg.weights").Weights(values)


def relative_error(ours, reference):
    import numpy as np

    a, b = np.asarray(ours, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


FLOW = dict(resolution=4, in_channels=8, out_channels=8, model_channels=96, cond_channels=32, num_blocks=2,
            num_heads=2, mlp_ratio=2.0, pe_mode="rope", share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
            dtype="float32")


@unittest.skipUnless(UPSTREAM and have("torch", "mlx", "numpy"), "TRELLIS2_UPSTREAM, torch and mlx are required")
class FlowParityTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx

        use_upstream()
        mx.set_default_device(mx.cpu)

    def _structure_flow(self):
        from trellis2.models.sparse_structure_flow import SparseStructureFlowModel

        return randomize(SparseStructureFlowModel(**FLOW), 1)

    def test_structure_flow_velocity_matches_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch

        model = self._structure_flow()
        torch.manual_seed(2)
        x, cond, t = torch.randn(2, 8, 4, 4, 4), torch.randn(2, 5, 32), torch.tensor([973.0, 120.0])
        with torch.no_grad():
            reference = model(x, t, cond).permute(0, 2, 3, 4, 1).reshape(2, 64, 8).numpy()
        flow = sub("trellis2.flow")
        tokens = mx.array(x.permute(0, 2, 3, 4, 1).reshape(2, 64, 8).numpy())
        rope = flow.rope_tables(flow.grid_positions(4), FLOW["model_channels"] // FLOW["num_heads"])
        ours = flow.velocity(weights(model), FLOW, tokens, mx.array(t.numpy()), mx.array(cond.numpy()), rope)
        self.assertLess(relative_error(np.array(ours), reference), BOUND)

    def test_guidance_interval_sampler_matches_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch
        from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler

        model = self._structure_flow()
        torch.manual_seed(3)
        noise, cond = torch.randn(1, 8, 4, 4, 4), torch.randn(1, 5, 32)
        params = dict(steps=6, guidance_strength=7.5, guidance_rescale=0.7, guidance_interval=[0.6, 1.0], rescale_t=5.0)
        with torch.no_grad():
            reference = FlowEulerGuidanceIntervalSampler(1e-5).sample(
                model, noise, cond, torch.zeros_like(cond), verbose=False, **params).samples
        flow = sub("trellis2.flow")
        rope = flow.rope_tables(flow.grid_positions(4), FLOW["model_channels"] // FLOW["num_heads"])
        tokens = mx.array(noise.permute(0, 2, 3, 4, 1).reshape(1, 64, 8).numpy())
        cond_mx = mx.array(cond.numpy())
        ours = flow.sample(weights(model), FLOW, tokens, cond_mx, mx.zeros_like(cond_mx), rope, **params)
        self.assertLess(relative_error(np.array(ours), reference.permute(0, 2, 3, 4, 1).reshape(1, 64, 8).numpy()), BOUND)

    def test_latent_flow_over_sparse_voxels_matches_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch
        from trellis2.models.structured_latent_flow import SLatFlowModel
        from trellis2.modules.sparse import SparseTensor

        config = dict(FLOW, resolution=8, in_channels=16, out_channels=16)
        model = randomize(SLatFlowModel(**config), 4)
        generator = np.random.default_rng(5)
        coords = np.unique(generator.integers(0, 8, size=(40, 3)), axis=0).astype(np.int32)
        torch.manual_seed(6)
        feats, cond = torch.randn(len(coords), 16), torch.randn(1, 5, 32)
        x = SparseTensor(feats=feats, coords=torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32),
                                                         torch.from_numpy(coords)], dim=1))
        with torch.no_grad():
            reference = model(x, torch.tensor([640.0]), cond).feats.numpy()
        flow = sub("trellis2.flow")
        rope = flow.rope_tables(coords, config["model_channels"] // config["num_heads"])
        ours = flow.velocity(weights(model), config, mx.array(feats.numpy())[None], mx.array([640.0]),
                             mx.array(cond.numpy()), rope)
        self.assertLess(relative_error(np.array(ours[0]), reference), BOUND)


@unittest.skipUnless(UPSTREAM and have("torch", "mlx", "numpy"), "TRELLIS2_UPSTREAM, torch and mlx are required")
class DecoderParityTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx

        use_upstream()
        mx.set_default_device(mx.cpu)

    def test_structure_decoder_matches_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch
        from trellis2.models.sparse_structure_vae import SparseStructureDecoder

        config = dict(out_channels=1, latent_channels=8, num_res_blocks=1, channels=[32, 16, 8], num_res_blocks_middle=1)
        model = randomize(SparseStructureDecoder(**config), 7, scale=0.1)
        torch.manual_seed(8)
        latent = torch.randn(1, 8, 4, 4, 4)
        with torch.no_grad():
            reference = model(latent)[:, 0].numpy()
        ours = sub("trellis2.structure").decode(weights(model), config, mx.array(latent.permute(0, 2, 3, 4, 1).numpy()))
        self.assertLess(relative_error(np.array(ours), reference), BOUND)

    def _shape_decoder(self):
        from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder

        config = dict(resolution=16, model_channels=[64, 32, 16], latent_channels=8, num_blocks=[1, 1, 0],
                      block_type=["SparseConvNeXtBlock3d"] * 3, up_block_type=["SparseResBlockC2S3d"] * 2,
                      block_args=[{}, {}, {}])
        model = FlexiDualGridVaeDecoder(**config)
        randomize(model, 9, scale=0.15)
        return model, config

    def _latent(self):
        import numpy as np
        import torch
        from trellis2.modules.sparse import SparseTensor

        generator = np.random.default_rng(10)
        coords = np.unique(generator.integers(0, 4, size=(30, 3)), axis=0).astype(np.int32)
        torch.manual_seed(11)
        feats = torch.randn(len(coords), 8)
        tensor = SparseTensor(feats=feats, coords=torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32),
                                                              torch.from_numpy(coords)], dim=1))
        return tensor, feats, coords

    def test_shape_decoder_outputs_match_reference_voxel_for_voxel(self):
        import mlx.core as mx
        import numpy as np
        import torch
        from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder

        model, config = self._shape_decoder()
        tensor, feats, coords = self._latent()
        with torch.no_grad():
            reference = SparseUnetVaeDecoder.forward(model, tensor)
        sparse = sub("trellis2.sparse")
        w = weights(model)
        voxels = sparse.run(w, config, mx.array(feats.numpy()), coords)
        ours = np.array(w.linear("output_layer", mx.fast.layer_norm(voxels.feats, None, None, 1e-5)))
        reference_coords = reference.coords[:, 1:].numpy()
        self.assertEqual(sorted(map(tuple, voxels.coords.tolist())), sorted(map(tuple, reference_coords.tolist())))
        order = np.lexsort(voxels.coords.T[::-1])
        reference_order = np.lexsort(reference_coords.T[::-1])
        self.assertLess(relative_error(ours[order], reference.feats.numpy()[reference_order]), BOUND)

    def test_cascade_upsampling_reaches_the_reference_coordinates(self):
        import mlx.core as mx
        import torch

        model, config = self._shape_decoder()
        tensor, feats, coords = self._latent()
        with torch.no_grad():
            reference = model.upsample(tensor, upsample_times=2)[:, 1:].numpy()
        ours = sub("trellis2.sparse").run(weights(model), config, mx.array(feats.numpy()), coords, stop_level=2)
        self.assertEqual(sorted(map(tuple, ours.tolist())), sorted(map(tuple, reference.tolist())))

    def test_dual_grid_mesh_matches_reference(self):
        import numpy as np
        import torch
        from o_voxel.convert import flexible_dual_grid_to_mesh

        generator = np.random.default_rng(12)
        coords = np.unique(generator.integers(0, 6, size=(150, 3)), axis=0)
        vertices = generator.uniform(-0.5, 1.5, size=(len(coords), 3)).astype(np.float32)
        crossed = generator.random((len(coords), 3)) > 0.4
        split = generator.uniform(0.1, 2.0, size=(len(coords), 1)).astype(np.float32)
        ref_vertices, ref_faces = flexible_dual_grid_to_mesh(
            torch.from_numpy(coords).int(), torch.from_numpy(vertices), torch.from_numpy(crossed),
            torch.from_numpy(split), aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], grid_size=6, train=False)
        ours_vertices, ours_faces = sub("trellis2.dual_grid").to_mesh(coords, vertices, crossed, split, 6)
        self.assertGreater(len(ours_faces), 0)
        np.testing.assert_allclose(ours_vertices, ref_vertices.numpy(), atol=1e-6)
        np.testing.assert_array_equal(ours_faces, ref_faces.numpy())


@unittest.skipUnless(have("torch", "transformers", "mlx", "numpy", "PIL"), "torch, transformers and mlx are required")
class EncoderParityTests(unittest.TestCase):
    def test_dinov3_features_match_transformers(self):
        import mlx.core as mx
        import numpy as np
        import torch
        import torch.nn.functional as F
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        mx.set_default_device(mx.cpu)
        config = DINOv3ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
                                 image_size=64, patch_size=16, num_register_tokens=4)
        model = randomize(DINOv3ViTModel(config), 13, scale=0.1)
        torch.manual_seed(14)
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            hidden = model.embeddings(image, bool_masked_pos=None)
            position = model.rope_embeddings(image)
            for layer in model.layer:
                hidden = layer(hidden, position_embeddings=position)
            reference = F.layer_norm(hidden, hidden.shape[-1:]).numpy()
        ours = sub("trellis2.dinov3").encode(weights(model), config.to_dict(), mx.array(image.permute(0, 2, 3, 1).numpy()))
        self.assertLess(relative_error(np.array(ours), reference), BOUND)


if __name__ == "__main__":
    unittest.main()
