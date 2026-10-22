"""Volume creation failures must not strand unowned or unrelated storage."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from mlx_agent import capture_volume
from mlx_agent.task_contracts import TaskError


class MountListingTests(unittest.TestCase):
    """The hdiutil listing can lag an attach or half-list an image while other images change."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / str(uuid4())
        self.root.mkdir(mode=0o700)
        self.calls, self.volume_uuid = [], str(uuid4()).upper()
        sleep = patch.object(capture_volume.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def listing(self, mounted=True):
        entity = {"dev-entry": "/dev/disk99"}
        if mounted:
            entity["mount-point"] = str(self.root / "mounted")
        return {"images": [{"image-path": str(self.root / "capture.sparsebundle"), "system-entities": [entity]}]}

    def tools(self, listings):
        """Each hdiutil listing returns the next given state; the last state repeats."""
        def run(arguments):
            self.calls.append(arguments[1])
            if arguments[0] == capture_volume.DISKUTIL:
                return {"MountPoint": str(self.root / "mounted"), "VolumeUUID": self.volume_uuid, "WritableVolume": True}
            if arguments[1] == "info":
                return listings.pop(0) if len(listings) > 1 else listings[0]
            return {}
        return run

    def mount(self, listings):
        record = {"volume_uuid": None, "capacity_bytes": 2 ** 62}
        with patch.object(capture_volume, "_run", side_effect=self.tools(listings)):
            with capture_volume._mounted(self.root, record, readonly=False) as (mounted, _):
                self.assertEqual(self.root / "mounted", mounted)

    def test_mount_reads_again_when_the_listing_lags_the_attach(self):
        self.mount([{"images": []}, {"images": []}, self.listing(), self.listing()])
        self.assertEqual("detach", self.calls[-1])

    def test_half_listed_image_is_read_again_before_mounting(self):
        self.mount([self.listing(mounted=False), {"images": []}, self.listing(), self.listing()])
        self.assertEqual("detach", self.calls[-1])

    def test_a_listing_that_stays_inconsistent_is_still_refused(self):
        with self.assertRaises(TaskError) as failure:
            self.mount([self.listing(mounted=False)])
        self.assertEqual("unsafe_capture_volume", failure.exception.code)
        self.assertEqual(["info"] * len(capture_volume.LISTING_DELAYS), self.calls)


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
