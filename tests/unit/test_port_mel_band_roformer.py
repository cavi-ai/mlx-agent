"""The bundled Mel-Band-RoFormer port's chunked separation; runs where numpy is importable."""

import importlib
import importlib.util
import sys
import types
import unittest
from unittest import mock

from mlx_agent.backends import PORTS_DIR

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def load_port():
    sys.path.insert(0, str(PORTS_DIR / "mlx-audio"))
    try:
        return importlib.import_module("mel_band_roformer")
    finally:
        sys.path.pop(0)


class EchoModel:
    """Returns its input as the vocal stem and records each chunk shape it sees."""

    def __init__(self, chunk_size, num_overlap):
        self.config = types.SimpleNamespace(chunk_size=chunk_size, num_overlap=num_overlap, sample_rate=44100)
        self.shapes = []

    def __call__(self, batch):
        self.shapes.append(batch.shape)
        return batch


@unittest.skipUnless(HAS_NUMPY, "numpy is not installed")
class ChunkedSeparationTests(unittest.TestCase):
    def separate(self, model, mixture):
        import numpy as np

        core = types.ModuleType("mlx.core")
        core.array, core.float32 = np.asarray, np.float32
        package = types.ModuleType("mlx")
        package.core = core
        with mock.patch.dict(sys.modules, {"mlx": package, "mlx.core": core}):
            return load_port().separate(model, mixture)

    def test_overlapped_chunks_rebuild_every_sample_of_the_mix(self):
        import numpy as np

        mixture = np.random.default_rng(7).uniform(-1, 1, size=(2, 37)).astype(np.float32)
        for overlap in (1, 2, 4):
            with self.subTest(num_overlap=overlap):
                model = EchoModel(8, overlap)
                vocals = self.separate(model, mixture)
                self.assertEqual(vocals.shape, (2, 37))
                np.testing.assert_allclose(vocals, mixture, rtol=1e-5, atol=1e-6)
                self.assertEqual(set(model.shapes), {(1, 2, 8)})

    def test_a_clip_shorter_than_one_chunk_is_padded_and_trimmed(self):
        import numpy as np

        mixture = np.linspace(-0.5, 0.5, 6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_allclose(self.separate(EchoModel(8, 2), mixture), mixture, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
