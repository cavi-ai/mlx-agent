"""Volume creation failures must not strand unowned or unrelated storage."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from mlx_agent import capture_volume
from mlx_agent.task_contracts import TaskError


class PartialVolumeCreationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name).resolve()
        self.identifier = str(uuid4())
        self.root = self.parent / self.identifier

    def partial_failure(self, arguments):
        self.assertEqual(arguments[1], "create")
        image = self.root / "capture.sparsebundle"
        image.mkdir(mode=0o700)
        data = image / "partial"
        data.write_bytes(b"partial image")
        data.chmod(0o600)
        raise TaskError("capture_volume_unavailable", "creation failed")

    def create(self):
        with patch.object(capture_volume.sys, "platform", "darwin"):
            capture_volume.CaptureVolume.create(self.root, self.identifier, 128 * 1024 * 1024 - 65536)

    def test_partial_image_is_removed_when_create_fails_before_marker(self):
        with patch.object(capture_volume, "_run", side_effect=self.partial_failure), self.assertRaises(TaskError):
            self.create()
        self.assertFalse(self.root.exists())

    def test_replaced_root_is_never_removed_after_creation_failure(self):
        def replaced(arguments):
            self.root.rename(self.parent / "original")
            self.root.mkdir(mode=0o700)
            (self.root / "unrelated").write_bytes(b"keep")
            raise TaskError("capture_volume_unavailable", "creation failed")
        with patch.object(capture_volume, "_run", side_effect=replaced), self.assertRaises(TaskError):
            self.create()
        self.assertEqual((self.root / "unrelated").read_bytes(), b"keep")
        self.assertTrue((self.parent / "original").is_dir())

    def test_failed_cleanup_reports_the_partial_recovery_path(self):
        with patch.object(capture_volume, "_run", side_effect=self.partial_failure), \
                patch.object(capture_volume.shutil, "rmtree", side_effect=OSError("cannot remove")), \
                self.assertRaises(TaskError) as failure:
            self.create()
        self.assertEqual(failure.exception.code, "capture_volume_unresolved")
        self.assertIn(str(self.root), failure.exception.remediation)
        self.assertTrue((self.root / "capture.sparsebundle" / "partial").is_file())
