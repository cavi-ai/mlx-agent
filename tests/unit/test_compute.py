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

    def test_unknown_launched_group_identity_retains_lease_before_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "compute"
            def identity(pid):
                return "launcher" if pid == os.getpid() else None
            factory = lambda: ComputeLease("convert", root, inspector=identity, alive=lambda _: True,
                group_probe=lambda _: [4242, 4343])
            with self.assertRaises(ComputeBusy):
                with factory() as lease:
                    lease.handoff(4242, process_group=True)
            record = json.loads((root / "owner.json").read_text())
            self.assertEqual(record["schema"], 2)
            self.assertEqual(record["execution_group"]["pgid"], 4242)
            with self.assertRaises(ComputeBusy):
                with factory():
                    pass

    def test_dead_launched_leader_cannot_release_surviving_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "compute"
            def identity(pid):
                return "launcher" if pid == os.getpid() else None
            factory = lambda: ComputeLease("convert", root, inspector=identity,
                alive=lambda pid: pid == os.getpid(), group_probe=lambda _: [4343])
            with factory() as lease:
                lease.handoff(4242, process_group=True)
            with self.assertRaises(ComputeBusy):
                with factory():
                    pass

    def test_unknown_handoff_preserves_an_already_bound_execution_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "compute"
            identities = {os.getpid(): "launcher", 4242: "monitor"}
            with self.assertRaises(ComputeBusy):
                with ComputeLease("convert", root, inspector=identities.get, alive=lambda _: True,
                                  group_probe=lambda _: [4242]) as lease:
                    with patch("mlx_agent.compute.os.getpgid", return_value=4242):
                        lease.bind_process_group(4242)
                    del identities[4242]
                    lease.handoff(4242)
            self.assertEqual(json.loads((root / "owner.json").read_text())["execution_group"]["pgid"], 4242)

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

class SharedComputeLeaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'compute'

    def test_shared_executions_persist_independent_records_and_exclude_exclusive_work(self):
        with ComputeLease('serve:first', self.root, shared=True) as first:
            with ComputeLease('task:second', self.root, shared=True) as second:
                records = [json.loads(path.read_text()) for path in self.root.glob('execution-*.json')]
                self.assertEqual({r['operation'] for r in records}, {'serve:first', 'task:second'})
                self.assertTrue(all(r['pid'] == os.getpid() and r['identity'] for r in records))
                self.assertNotEqual(first.record_path, second.record_path)
                with self.assertRaises(ComputeBusy):
                    with ComputeLease('cache', self.root):
                        pass
            self.assertEqual(len(list(self.root.glob('execution-*.json'))), 1)
        self.assertEqual(list(self.root.glob('execution-*.json')), [])
        with ComputeLease('training', self.root):
            with self.assertRaises(ComputeBusy):
                with ComputeLease('serve:first', self.root, shared=True):
                    pass

    def test_shared_orphan_group_blocks_both_modes_until_confirmed_stopped(self):
        identities = {os.getpid(): 'gateway', 4242: 'worker'}
        members = [4242, 4343]
        def lease(shared):
            return ComputeLease('task', self.root, shared=shared, inspector=identities.get,
                alive=lambda pid: pid in identities, group_probe=lambda _: list(members))
        with lease(True) as owner:
            with patch('mlx_agent.compute.os.getpgid', return_value=4242):
                owner.bind_process_group(4242)
            path = owner.record_path
        del identities[4242]
        members[:] = [4343]
        for shared in (False, True):
            with self.subTest(shared=shared), self.assertRaises(ComputeBusy):
                with lease(shared):
                    pass
        self.assertEqual(json.loads(path.read_text())['execution_group']['pgid'], 4242)
        # The stale gateway identity must also be gone before reclaiming ownership.
        identities[os.getpid()] = 'new-gateway'
        members.clear()
        with lease(True):
            self.assertFalse(path.exists())

    def test_shared_admission_respects_legacy_detached_owner(self):
        identities = {os.getpid(): 'launcher', 4242: 'trainer'}
        with ComputeLease('training', self.root, inspector=identities.get) as owner:
            owner.handoff(4242)
        with self.assertRaises(ComputeBusy):
            with ComputeLease('serve', self.root, shared=True, inspector=identities.get,
                              alive=lambda pid: pid in identities):
                pass
        identities[4242] = 'replacement'
        with ComputeLease('serve', self.root, shared=True, inspector=identities.get,
                          alive=lambda pid: pid in identities):
            self.assertEqual(len(list(self.root.glob('execution-*.json'))), 1)

    def test_shared_uninspectable_group_keeps_its_record(self):
        def unavailable(_):
            raise OSError('unavailable')
        with ComputeLease('task', self.root, shared=True, inspector=lambda _: 'identity',
                          group_probe=unavailable) as owner:
            with patch('mlx_agent.compute.os.getpgid', return_value=4242):
                owner.bind_process_group(4242)
            path = owner.record_path
        for shared in (False, True):
            with self.subTest(shared=shared), self.assertRaises(ComputeBusy):
                with ComputeLease('next', self.root, shared=shared, alive=lambda _: False,
                                  group_probe=unavailable):
                    pass
        self.assertTrue(path.exists())

    def test_shared_record_publication_survives_competing_admission(self):
        import fcntl
        identities = {os.getpid(): 'gateway', 4242: 'worker'}
        members = [4242]
        real_open = os.open
        interleaved = False
        def lease(name, root=None):
            return ComputeLease(name, root or self.root, shared=True, inspector=identities.get,
                alive=lambda pid: pid in identities, group_probe=lambda _: list(members))
        def admit_between_creation_and_lock(path, flags, *args, **kwargs):
            nonlocal interleaved
            descriptor = real_open(path, flags, *args, **kwargs)
            if not interleaved and flags & os.O_EXCL and 'execution-' in Path(path).name:
                interleaved = True
                with lease('serve:second'):
                    pass
            return descriptor
        with patch('mlx_agent.compute.os.open', side_effect=admit_between_creation_and_lock):
            with lease('serve:first') as first:
                first.handoff(4242, process_group=True)
                path = first.record_path
                self.assertTrue(path.exists(), 'Competing admission removed the new ownership record.')
                record = json.loads(path.read_text())
                self.assertEqual(record['pid'], 4242)
                self.assertEqual(record['execution_group']['pgid'], 4242)
        self.assertTrue(interleaved)
        with self.assertRaises(ComputeBusy):
            with lease('serve:third'):
                pass
        members.clear()
        del identities[4242]
        with lease('serve:third'):
            self.assertFalse(path.exists())

        blocker = None
        def contend_before_owner_lock(path, flags, *args, **kwargs):
            nonlocal blocker
            descriptor = real_open(path, flags, *args, **kwargs)
            if flags & os.O_EXCL and 'execution-' in Path(path).name:
                blocker = real_open(path, os.O_RDWR)
                fcntl.flock(blocker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return descriptor
        contended = self.root / 'contended'
        try:
            with patch('mlx_agent.compute.os.open', side_effect=contend_before_owner_lock):
                with self.assertRaises(ComputeBusy):
                    with lease('serve:contended', contended):
                        pass
        finally:
            if blocker is not None:
                os.close(blocker)
        self.assertEqual(list(contended.glob('execution-*.json')), [])
        self.assertEqual(list(contended.glob('.*.pending')), [])

    def test_shared_execution_is_visible_across_processes(self):
        import subprocess
        import sys
        program = """import sys
from mlx_agent.compute import ComputeLease
with ComputeLease('serve:child', sys.argv[1], shared=True):
    print('ready', flush=True)
    sys.stdin.readline()
"""
        child = subprocess.Popen([sys.executable, '-c', program, str(self.root)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            with ComputeLease('task:parent', self.root, shared=True):
                records = [json.loads(path.read_text()) for path in self.root.glob('execution-*.json')]
                self.assertEqual({record['pid'] for record in records}, {os.getpid(), child.pid})
                with self.assertRaises(ComputeBusy):
                    ComputeLease('conversion', self.root).__enter__()
            child.communicate('finish\n', timeout=3)
            self.assertEqual(child.returncode, 0)
            self.assertEqual(list(self.root.glob('execution-*.json')), [])
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=3)
