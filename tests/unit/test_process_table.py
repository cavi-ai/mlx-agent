"""The in-process table matches ps for live processes and leaves zombies out."""
import os
import subprocess
import sys
import time
import unittest

from mlx_agent import process_table
from mlx_agent.compute import process_identity


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


if __name__ == '__main__':
    unittest.main()
