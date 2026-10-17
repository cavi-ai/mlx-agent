import json
import os
from pathlib import Path
import tempfile
import unittest
from mlx_agent.compute import ComputeLease, ComputeBusy
from unittest.mock import patch

class ComputeLeaseTests(unittest.TestCase):
    def test_lock_excludes_other_entrypoints_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "compute"
            with ComputeLease("generation", root):
                with self.assertRaises(ComputeBusy):
                    with ComputeLease("training", root):
                        pass
            with ComputeLease("training", root):
                pass

    def test_detached_owner_survives_launcher_exit_and_stale_pid_is_reclaimed(self):
        identities = {os.getpid(): "launcher", 4242: "trainer"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "compute"
            factory = lambda name: ComputeLease(name, root, inspector=identities.get,
                                                alive=lambda pid: pid in identities)
            with factory("training") as owner:
                owner.handoff(4242)
            with self.assertRaises(ComputeBusy):
                with factory("video"):
                    pass
            identities[4242] = "reused-pid"
            with factory("video"):
                pass

    def test_unverifiable_detached_process_retains_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "compute"
            def inspector(pid):
                return "launcher" if pid == os.getpid() else None
            with self.assertRaises(ComputeBusy):
                with ComputeLease("training", root, inspector=inspector, alive=lambda pid: True) as owner:
                    owner.handoff(4242)
            self.assertEqual(json.loads((root / "owner.json").read_text())["pid"], 4242)
            with self.assertRaises(ComputeBusy):
                with ComputeLease("video", root, inspector=inspector, alive=lambda pid: True):
                    pass

    def test_unverifiable_live_owner_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "compute"
            root.mkdir(mode=0o700)
            record = root / "owner.json"
            record.write_text(json.dumps({"schema": 1, "pid": 4242, "identity": "original", "operation": "training"}))
            record.chmod(0o600)
            with self.assertRaises(ComputeBusy):
                with ComputeLease("video", root, inspector=lambda pid: None, alive=lambda pid: True):
                    pass

    def test_execution_group_keeps_ownership_after_supervisor_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "compute"
            identities = {os.getpid(): "supervisor", 4242: "execution"}
            members = [4242, 4343]
            factory = lambda: ComputeLease("task", root, inspector=identities.get,
                alive=lambda pid: pid in identities, group_probe=lambda _: list(members))
            with factory() as lease:
                with patch("mlx_agent.compute.os.getpgid", return_value=4242):
                    lease.bind_process_group(4242)
            del identities[os.getpid()]
            del identities[4242]
            members[:] = [4343]
            with self.assertRaises(ComputeBusy):
                with factory():
                    pass
            self.assertEqual(json.loads((root / "owner.json").read_text())["execution_group"]["pgid"], 4242)
            members.clear()
            identities[os.getpid()] = "new-supervisor"
            with factory():
                pass

    def test_uninspectable_execution_group_never_releases_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "compute"
            def unavailable(_):
                raise OSError("process table unavailable")
            with ComputeLease("task", root, inspector=lambda _: "identity", group_probe=unavailable) as lease:
                with patch("mlx_agent.compute.os.getpgid", return_value=4242):
                    lease.bind_process_group(4242)
            with self.assertRaises(ComputeBusy):
                with ComputeLease("new", root, alive=lambda _: False, group_probe=unavailable):
                    pass
