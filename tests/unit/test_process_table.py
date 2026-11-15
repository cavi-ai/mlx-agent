"""The in-process table matches ps for live processes and leaves zombies out."""
import ctypes
import errno
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import subprocess
import sys
import time
import unittest

from mlx_agent import process_table
from mlx_agent.compute import ComputeBusy, ComputeLease, process_identity


class ProcessTableTests(unittest.TestCase):
    def leader_with_zombie(self):
        """A new process group whose leader leaves one exited child unreaped."""
        leader = subprocess.Popen([sys.executable, '-c',
            'import subprocess,time; child = subprocess.Popen(["/bin/sleep", "0"]); print(child.pid, flush=True); time.sleep(30)'],
            stdout=subprocess.PIPE, text=True, start_new_session=True)
        self.addCleanup(leader.stdout.close)
        self.addCleanup(leader.wait, 5)
        self.addCleanup(leader.kill)
        zombie = int(leader.stdout.readline())
        deadline = time.monotonic() + 3
        state = ''
        while time.monotonic() < deadline and not state.startswith('Z'):
            state = subprocess.check_output(['/bin/ps', '-o', 'stat=', '-p', str(zombie)], text=True).strip()
            time.sleep(.01)
        self.assertTrue(state.startswith('Z'), state)
        return leader, zombie

    def test_group_and_rows_list_live_members_without_zombies(self):
        leader, zombie = self.leader_with_zombie()
        self.assertEqual(process_table.group(leader.pid), [leader.pid])
        self.assertEqual(process_table.rows([leader.pid, zombie]), {leader.pid: (os.getpid(), leader.pid)})
        self.assertEqual(process_table.rows()[leader.pid], (os.getpid(), leader.pid))
        self.assertNotIn(zombie, process_table.rows())

    def test_identities_match_process_identity_and_skip_exited_processes(self):
        leader, _ = self.leader_with_zombie()
        exited = subprocess.Popen(['/bin/sleep', '0'])
        exited.wait()
        self.assertEqual(process_table.identities([leader.pid, exited.pid]), {leader.pid: process_identity(leader.pid)})
        self.assertEqual(process_table.identities([exited.pid]), {})

    @unittest.skipUnless(sys.platform == 'darwin', 'The native memory snapshot reads libproc.')
    def test_native_memory_snapshot_matches_ps_for_the_processes_it_identifies(self):
        from mlx_agent.memory_budget import process_snapshot
        leader, zombie = self.leader_with_zombie()
        native, legacy = process_snapshot({leader.pid}), process_snapshot()
        self.assertNotIn(zombie, native)
        for pid in (os.getpid(), leader.pid):
            self.assertEqual((native[pid]['parent'], native[pid]['group']), (legacy[pid]['parent'], legacy[pid]['group']))
            self.assertIsInstance(native[pid]['physical_bytes'], int)
        self.assertEqual(native[leader.pid]['identity'].split(), legacy[leader.pid]['identity'].split())
        self.assertNotIn('identity', native[os.getpid()])
        self.assertLess(abs(native[leader.pid]['rss'] - legacy[leader.pid]['rss']), 4 * 1024 * 1024)


class DarwinEnumerationTests(unittest.TestCase):
    class Libproc:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.calls = []

        def _call(self, group, buffer, size):
            self.calls.append((group, size, ctypes.get_errno()))
            found, error = next(self.responses)
            ctypes.set_errno(error)
            for index in range(min(found, size // ctypes.sizeof(ctypes.c_int))):
                buffer[index] = index + 1
            return found

        def proc_listallpids(self, buffer, size):
            return self._call(None, buffer, size)

        def proc_listpgrppids(self, group, buffer, size):
            return self._call(group, buffer, size)

    def test_zero_with_errno_fails_and_stale_errno_does_not(self):
        for group, error in ((None, errno.EPERM), (123, errno.EIO)):
            with self.subTest(group=group):
                lib = self.Libproc([(0, error), (0, 0)])
                with patch.object(process_table, '_libproc', return_value=lib):
                    with self.assertRaises(OSError) as caught:
                        process_table._darwin_pids(group)
                    self.assertEqual(caught.exception.errno, error)
                    ctypes.set_errno(errno.EPERM)
                    self.assertEqual(process_table._darwin_pids(group), [])
                self.assertEqual([call[2] for call in lib.calls], [0, 0])

    def test_full_buffer_grows_and_returns_nonempty_results(self):
        for group in (None, 123):
            with self.subTest(group=group):
                lib = self.Libproc([(4096, 0), (2, 0)])
                with patch.object(process_table, '_libproc', return_value=lib):
                    self.assertEqual(process_table._darwin_pids(group), [1, 2])
                self.assertEqual(lib.calls[1][1], lib.calls[0][1] * 2)

    @unittest.skipUnless(sys.platform == 'darwin', 'libproc is available on macOS')
    def test_enumeration_error_preserves_execution_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / 'compute'
            root.mkdir(mode=0o700)
            record = root / 'execution-test.json'
            payload = json.dumps({'schema': 2, 'pid': 123, 'identity': 'old', 'operation': 'task',
                                  'execution_group': {'pgid': 123, 'leader_identity': 'old'}}).encode()
            record.write_bytes(payload)
            record.chmod(0o600)
            lib = self.Libproc([(0, errno.EPERM)])
            with patch.object(process_table, '_libproc', return_value=lib):
                with self.assertRaises(ComputeBusy):
                    with ComputeLease('music', root):
                        pass
            self.assertEqual(record.read_bytes(), payload)


if __name__ == '__main__':
    unittest.main()
