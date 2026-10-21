import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlx_agent.kokoro_runtime import kokoro_data_directory


class KokoroDataDirectoryTests(unittest.TestCase):
    def test_long_packaged_path_uses_a_private_short_alias_and_restores_environment(self):
        with TemporaryDirectory() as root:
            data = Path(root, "x" * 170, "espeak-ng-data")
            data.mkdir(parents=True)
            loader = SimpleNamespace(get_data_path=lambda: str(data))
            with patch.dict("sys.modules", {"espeakng_loader": loader}), patch.dict(os.environ, {"ESPEAK_DATA_PATH": "previous"}):
                with kokoro_data_directory():
                    alias = Path(os.environ["ESPEAK_DATA_PATH"])
                    self.assertLess(len(os.fsencode(alias / "espeak-ng-data")), 160)
                    self.assertEqual((alias / "espeak-ng-data").resolve(), data.resolve())
                    self.assertEqual(alias.stat().st_mode & 0o777, 0o700)
                self.assertFalse(alias.exists())
                self.assertEqual(os.environ["ESPEAK_DATA_PATH"], "previous")

    def test_failure_removes_only_the_alias_and_unsets_a_previously_absent_variable(self):
        with TemporaryDirectory() as root:
            loader = SimpleNamespace(get_data_path=lambda: root)
            with patch.dict("sys.modules", {"espeakng_loader": loader}), patch.dict(os.environ):
                os.environ.pop("ESPEAK_DATA_PATH", None)
                with self.assertRaisesRegex(RuntimeError, "generation failed"):
                    with kokoro_data_directory():
                        alias = Path(os.environ["ESPEAK_DATA_PATH"])
                        raise RuntimeError("generation failed")
                self.assertNotIn("ESPEAK_DATA_PATH", os.environ)
                self.assertFalse(alias.exists())
                self.assertTrue(Path(root).is_dir())
