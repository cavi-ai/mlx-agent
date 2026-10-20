"""Extension admission must precede any client or server execution."""

import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mlx_agent.task_contracts import TaskError

try:
    plugins = importlib.import_module("mlx_agent.coding_capture_plugins")
except ModuleNotFoundError:
    plugins = None


class ExtensionAdmissionTests(unittest.TestCase):
    def test_native_definitions_require_a_supported_bootstrap_before_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "native.mjs"
            source.write_text("export default {id:'native',setup(){}}")
            with patch.object(plugins, "verify_extensions"), patch("subprocess.Popen") as start:
                with self.assertRaises(TaskError) as failure:
                    plugins.prepare_extension_bootstrap({"bundles": [{"legacy": False, "path": str(source), "id": "native"}]},
                                                        root / "profile", {}, root)
                self.assertEqual(failure.exception.code, "capture_client_unavailable")
                start.assert_not_called()
                self.assertFalse((root / "profile").exists())

    def test_lost_process_ownership_releases_local_handles_without_signalling(self):
        host = plugins.ExtensionHost.__new__(plugins.ExtensionHost)
        host.process = Mock(pid=12345)
        host.process.poll.return_value = None
        host.identity = {"pid": 12345, "identity": "original", "started_at": [1, 0]}
        host.channel, stream = Mock(), Mock()
        host.logs = [stream]
        with patch("mlx_agent.capture_platform.capture_process_identity", return_value={"pid": 12345}), \
                patch("os.killpg") as signal:
            with self.assertRaises(TaskError) as failure:
                host.close()
            self.assertEqual(failure.exception.code, "capture_ownership_lost")
            signal.assert_not_called()
        host.process.stdin.close.assert_called_once()
        host.channel.close.assert_called_once()
        stream.close.assert_called_once()

    def test_package_targets_including_cached_packages_never_start_a_process(self):
        self.assertIsNotNone(plugins, "Extension admission is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached = root / "node_modules" / "cached-plugin"
            cached.mkdir(parents=True)
            (cached / "index.js").write_text("export default {id:'cached',setup(){}}")
            for target in ("cached-plugin", "@example/plugin@1.0.0", "file:./plugin", "./plugin.ts"):
                with self.subTest(target=target), patch("subprocess.Popen") as start:
                    with self.assertRaises(TaskError) as failure:
                        plugins.freeze_extensions(Path("/unexecuted-client"), [target], root / "output")
                    self.assertEqual(failure.exception.code, "capture_client_unavailable")
                    start.assert_not_called()
                    self.assertFalse((root / "output").exists())

    def test_missing_and_linked_entries_are_refused_before_start(self):
        self.assertIsNotNone(plugins, "Extension admission is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real.ts"
            real.write_text("export default {id:'local',setup(){}}")
            link = root / "link.ts"
            link.symlink_to(real)
            for source in (root / "missing.ts", link):
                with self.subTest(source=source), patch("subprocess.Popen") as start:
                    with self.assertRaises(TaskError):
                        plugins.freeze_extensions(Path("/unexecuted-client"), [str(source)], root / "output")
                    start.assert_not_called()
