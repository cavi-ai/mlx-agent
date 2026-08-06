import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.gguf_runner import _paths_without_script_dir


class PathsWithoutScriptDirTests(unittest.TestCase):
    def test_removes_script_directory(self):
        with TemporaryDirectory() as raw:
            script_dir = Path(raw)
            other = script_dir.parent / "site-packages"
            entries = [str(script_dir), str(other), "/usr/lib/python3"]
            cleaned = _paths_without_script_dir(entries, script_dir, script_dir.parent)
            self.assertEqual(cleaned, [str(other), "/usr/lib/python3"])

    def test_removes_cwd_placeholder_when_cwd_is_script_dir(self):
        with TemporaryDirectory() as raw:
            script_dir = Path(raw)
            cleaned = _paths_without_script_dir(
                ["", ".", str(script_dir), "/safe"],
                script_dir,
                script_dir,
            )
            self.assertEqual(cleaned, ["/safe"])

    def test_keeps_cwd_placeholder_when_cwd_differs(self):
        with TemporaryDirectory() as raw:
            script_dir = Path(raw) / "runner"
            script_dir.mkdir()
            cwd = Path(raw) / "elsewhere"
            cwd.mkdir()
            cleaned = _paths_without_script_dir(["", "."], script_dir, cwd)
            self.assertEqual(cleaned, ["", "."])
