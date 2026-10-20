import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

ROOT = Path(__file__).resolve().parents[2]


def _installed_copy(directory):
    """Lay the package out as a wheel install does, beside an unrelated package."""
    site = Path(directory) / "lib" / "site-packages"
    shutil.copytree(ROOT / "src/mlx_agent", site / "mlx_agent",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (site / "unrelated_package.py").write_text("", encoding="utf-8")
    return site


class InstalledPackageTests(unittest.TestCase):
    def test_provider_commands_report_missing_plugin_files(self):
        with tempfile.TemporaryDirectory() as directory:
            site = _installed_copy(directory)
            environment = dict(os.environ, PYTHONPATH=str(site), PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([sys.executable, "-m", "mlx_agent", "providers", "--json"],
                env=environment, cwd=directory, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(json.loads(result.stdout)["error"]["code"], "plugin_files_unavailable")

    def test_strata_runtime_exposes_only_its_package(self):
        with tempfile.TemporaryDirectory() as directory:
            site = _installed_copy(directory)
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            environment.pop("PYTHONPATH", None)
            probe = (
                "import importlib, importlib.util, json, runpy, sys\n"
                "sys.argv = ['strata_runtime', '--help']\n"
                "try:\n"
                "    runpy.run_path(" + repr(str(site / "mlx_agent/strata_runtime.py")) + ", run_name='__main__')\n"
                "except SystemExit:\n"
                "    pass\n"
                "print(json.dumps({'unrelated': importlib.util.find_spec('unrelated_package') is not None,\n"
                "                  'gguf': importlib.import_module('mlx_agent.gguf').__file__}))\n"
            )
            result = subprocess.run([sys.executable, "-c", probe], env=environment, cwd=directory,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            observed = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertFalse(observed["unrelated"])
            self.assertEqual(Path(observed["gguf"]).resolve(), (site / "mlx_agent/gguf.py").resolve())

    @unittest.skipIf(tomllib is None, "tomllib requires Python 3.11")
    def test_console_scripts_resolve_to_callables(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(set(project["scripts"]), {"mlx-agent", "mlx-agent-mcp"})
        for name, target in project["scripts"].items():
            module, _, attribute = target.partition(":")
            self.assertTrue(callable(getattr(importlib.import_module(module), attribute)), name)


if __name__ == "__main__":
    unittest.main()
