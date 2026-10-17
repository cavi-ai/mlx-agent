import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.task_contracts import TaskError
from mlx_agent.task_identity import capture_identity, check_identity
from mlx_agent.backends import load_manifests


class TaskIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"laya"}')
        self.executable = self.root / "python"
        self.executable.write_bytes(b"runtime")
        self.manifest = load_manifests()["mlx-embeddings"]

    def test_hf_snapshot_links_and_model_drift(self):
        blob = self.root / "blob"
        blob.write_bytes(b"weights")
        (self.model / "model.safetensors").symlink_to(blob)
        identity = capture_identity(self.model, self.manifest, self.executable)
        check_identity(identity)
        blob.write_bytes(b"new weights")
        with self.assertRaises(TaskError):
            check_identity(identity)

    def test_config_content_is_rechecked(self):
        identity = capture_identity(self.model, self.manifest, self.executable)
        (self.model / "config.json").write_text('{"model_type":"other"}')
        with self.assertRaises(TaskError):
            check_identity(identity)

    def test_executable_identity_is_rechecked(self):
        identity = capture_identity(self.model, self.manifest, self.executable)
        self.executable.unlink()
        self.executable.write_bytes(b"replacement")
        with self.assertRaises(TaskError):
            check_identity(identity)

    def test_changed_backend_manifest_is_rechecked(self):
        identity = capture_identity(self.model, self.manifest, self.executable)
        manifests = load_manifests()
        manifests["mlx-embeddings"]["version"] = "changed"
        with patch("mlx_agent.backends.load_manifests", return_value=manifests):
            with self.assertRaises(TaskError):
                check_identity(identity)

    def test_retargeted_model_root_link_is_rejected(self):
        alias = self.root / "selected"
        alias.symlink_to(self.model, target_is_directory=True)
        identity = capture_identity(alias, self.manifest, self.executable)
        replacement = self.root / "other"
        replacement.mkdir()
        (replacement / "config.json").write_text('{"model_type":"laya"}')
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        with self.assertRaises(TaskError):
            check_identity(identity)
