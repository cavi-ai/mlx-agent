"""The bundled SAM-Audio port's stem files and text encoder lookup; runs where numpy is importable."""

import importlib
import importlib.util
import json
import sys
import types
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mlx_agent.backends import PORTS_DIR

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "numpy is not installed")
class SamAudioPortTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(PORTS_DIR / "mlx-audio"))
        self.addCleanup(sys.path.remove, str(PORTS_DIR / "mlx-audio"))
        self.port = importlib.import_module("sam_audio")
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_stems_are_written_at_the_mix_length_and_model_rate(self):
        import numpy as np

        length = 9
        samples = np.full((length, 1), 0.25, dtype=np.float32)
        target = np.full(length + 5, 0.5, dtype=np.float32)
        residual = np.full(length - 3, -0.5, dtype=np.float32)
        seen = []

        def separate_long(audios, descriptions):
            seen.append((audios.shape, descriptions))
            return types.SimpleNamespace(target=[target], residual=[residual])

        model = types.SimpleNamespace(sample_rate=48000, separate_long=separate_long)
        core = types.ModuleType("mlx.core")
        core.array, core.float32 = np.asarray, np.float32
        package = types.ModuleType("mlx")
        package.core = core
        audio_io = types.ModuleType("mlx_audio.audio_io")
        audio_io.read = lambda *args, **kwargs: (samples, 48000)
        out, rest = self.root / "drums.wav", self.root / "rest.wav"
        with mock.patch.dict(sys.modules, {"mlx": package, "mlx.core": core, "mlx_audio.audio_io": audio_io}):
            result = self.port.generate(model, "mix.wav", str(out), str(rest), prompt="drums")
        self.assertEqual(seen, [((1, 1, length), ["drums"])])
        self.assertEqual((result["prompt"], result["sample_rate"], result["rest"]), ("drums", 48000, str(rest)))
        for path, expected in ((out, [16383] * length), (rest, [-16383] * (length - 3) + [0] * 3)):
            with self.subTest(path=path.name), wave.open(str(path), "rb") as wav:
                self.assertEqual((wav.getnchannels(), wav.getframerate(), wav.getnframes()), (1, 48000, length))
                self.assertEqual(np.frombuffer(wav.readframes(length), dtype="<i2").tolist(), expected)

    def test_a_checkpoint_without_its_text_encoder_beside_it_is_refused(self):
        checkpoint = self.root / "sam-audio-base"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(json.dumps({"text_encoder": {"name": "t5-base"}}), encoding="utf-8")
        with self.assertRaisesRegex(FileNotFoundError, "t5-base"):
            self.port.load(str(checkpoint))
        (self.root / "t5-base").mkdir()
        for name in ("config.json", "model.safetensors"):
            (self.root / "t5-base" / name).write_text("{}", encoding="utf-8")
        self.assertEqual(self.port.text_encoder_folder(checkpoint), self.root / "t5-base")


if __name__ == "__main__":
    unittest.main()
