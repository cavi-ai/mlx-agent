"""The bundled Mel-Band-RoFormer port's chunked separation; runs where numpy is importable."""

import errno
import importlib
import importlib.util
import sys
import types
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
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

    def generate(self, out, rest, samples, vocals):
        import numpy as np

        audio_io = types.ModuleType("mlx_audio.audio_io")
        audio_io.read = lambda *args, **kwargs: (samples, 44100)

        def write(path, data, rate):
            with wave.open(str(path), "wb") as wav:
                wav.setparams((2, 2, rate, len(data), "NONE", "not compressed"))
                wav.writeframes((np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())

        audio_io.write = write
        port = load_port()
        with mock.patch.dict(sys.modules, {"mlx_audio.audio_io": audio_io}), mock.patch.object(port, "separate", return_value=vocals):
            return port.generate(EchoModel(8, 2), "mix.wav", str(out), str(rest))

    def test_stems_are_written_in_bounded_pcm_blocks(self):
        import numpy as np

        samples = np.tile(np.array([[0.5, -0.5]], dtype=np.float32), (150_000, 1))
        vocals = samples.T * np.array([[3], [0.5]], dtype=np.float32)
        chunks = []
        write = wave.Wave_write.writeframesraw

        def record(wav, data):
            chunks.append(len(data))
            return write(wav, data)

        with TemporaryDirectory() as directory:
            out = Path(directory) / "vocals.wav"
            rest = Path(directory) / "remainder.wav"
            with mock.patch.object(wave.Wave_write, "writeframesraw", record):
                self.generate(out, rest, samples, vocals)
            self.assertGreater(len(chunks), 2)
            self.assertLess(max(chunks), len(samples) * 4)
            for path, pair in ((out, (32767, -8191)), (rest, (-32767, -8191))):
                with self.subTest(path=path), wave.open(str(path), "rb") as wav:
                    self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()),
                                     (2, 2, 44100, len(samples)))
                    pcm = np.frombuffer(wav.readframes(len(samples)), dtype="<i2").reshape(-1, 2)
                    np.testing.assert_array_equal(pcm, np.tile(pair, (len(samples), 1)))

    def test_second_output_conflict_preserves_file_and_allows_retry(self):
        import numpy as np

        samples = np.zeros((8, 2), dtype=np.float32)
        with TemporaryDirectory() as directory:
            out = Path(directory) / "vocals.wav"
            rest = Path(directory) / "remainder.wav"
            rest.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                self.generate(out, rest, samples, samples.T)
            self.assertFalse(out.exists())
            self.assertEqual(rest.read_bytes(), b"existing")
            self.assertEqual(list(Path(directory).iterdir()), [rest])
            rest.unlink()
            self.generate(out, rest, samples, samples.T)
            self.assertEqual(set(Path(directory).iterdir()), {out, rest})

    def test_second_output_staging_failure_leaves_no_published_stem(self):
        import numpy as np

        samples = np.zeros((8, 2), dtype=np.float32)
        with TemporaryDirectory() as directory:
            out = Path(directory) / "vocals.wav"
            rest = Path(directory) / "missing" / "remainder.wav"
            with self.assertRaises(FileNotFoundError):
                self.generate(out, rest, samples, samples.T)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_outputs_still_publish_exclusively_without_hard_link_support(self):
        import numpy as np

        samples = np.zeros((8, 2), dtype=np.float32)
        with TemporaryDirectory() as directory, mock.patch(
                "os.link", side_effect=OSError(errno.ENOTSUP, "hard links unsupported")):
            out = Path(directory) / "vocals.wav"
            rest = Path(directory) / "remainder.wav"
            rest.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                self.generate(out, rest, samples, samples.T)
            self.assertFalse(out.exists())
            self.assertEqual(rest.read_bytes(), b"existing")
            rest.unlink()
            self.generate(out, rest, samples, samples.T)
            for path in (out, rest):
                with wave.open(str(path), "rb") as wav:
                    self.assertEqual(wav.getnframes(), len(samples))
            self.assertEqual(set(Path(directory).iterdir()), {out, rest})


if __name__ == "__main__":
    unittest.main()
