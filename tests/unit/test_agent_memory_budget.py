"""A shared budget counts pending loads, warm models and external agent families."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from mlx_agent.residency import MemoryAdmissionError, MemoryRegistry


class AgentBudgetTests(unittest.TestCase):
    def test_reused_group_expires_before_unreadable_memory_is_accounted(self):
        self.rows[302]['identity'] = 'Mon Jan  1 00:00:00 2001 agent'
        self.budget(external=[302])
        path = self.root / 'budget.json'
        os.utime(path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
        del self.rows[301], self.rows[302]
        self.rows[303] = {'parent': 1, 'group': 301, 'rss': None}
        self.rows[301] = {'parent': 1, 'group': 301, 'rss': None,
                          'identity': 'Fri Jan  1 00:00:00 2099 other'}
        # The original root is gone; this replacement is the new group leader.
        result = self.registry.snapshot()['agent_budget']
        self.assertEqual(result.get('external_bytes'), 0, result)
        self.assertEqual(json.loads(path.read_text())['external'], [])

    def test_expired_binding_inspection_is_read_only_then_snapshot_persists(self):
        self.budget(external=[301])
        path = self.root / 'budget.json'
        before = path.read_bytes()
        del self.rows[301], self.rows[302]
        self.assertEqual(self.registry.inspect()['agent_budget']['external_bytes'], 0)
        self.assertEqual(path.read_bytes(), before)
        self.registry.snapshot()
        self.assertEqual(json.loads(path.read_text())['external'], [])

    def test_reparented_group_and_ambiguous_identity_remain_bound(self):
        self.rows[302]['identity'] = 'Mon Jan  1 00:00:00 2001 agent'
        self.budget(external=[302])
        path = self.root / 'budget.json'
        os.utime(path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
        del self.rows[302]
        self.rows[303] = {'parent': 1, 'group': 301, 'rss': 5}
        self.rows[301] = {'parent': 1, 'group': 301, 'rss': 3,
                          'identity': 'Fri Jan  1 00:00:00 2010 leader'}
        before = path.read_bytes()
        self.assertEqual(self.registry.snapshot()['agent_budget']['external_bytes'], 8)
        self.assertEqual(path.read_bytes(), before)
        self.rows[301]['identity'] = 'unparseable'
        self.assertEqual(self.registry.snapshot()['agent_budget']['external_bytes'], 8)
        self.assertEqual(path.read_bytes(), before)

    def test_missing_root_identity_retains_binding_despite_newer_leader(self):
        self.budget(external=[302])
        path = self.root / 'budget.json'
        os.utime(path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
        self.rows[302].pop('identity')
        self.rows[302]['group'] = 400
        self.rows[301]['identity'] = 'Fri Jan  1 00:00:00 2099 leader'
        before = path.read_bytes()
        self.assertEqual(self.registry.snapshot()['agent_budget']['status'], 'unavailable')
        self.assertEqual(path.read_bytes(), before)

    def test_unchanged_budget_is_not_rewritten_and_reconfigure_expires(self):
        self.budget(external=[301])
        path = self.root / 'budget.json'
        old = path.stat().st_mtime_ns
        self.registry.configure_budget(200, 20)
        self.assertEqual(path.stat().st_mtime_ns, old)
        del self.rows[301], self.rows[302]
        self.registry.configure_budget(200, 20)
        self.assertEqual(json.loads(path.read_text())['external'], [])
        self.assertEqual(json.loads(path.read_text())['limit_bytes'], 200)
        self.assertEqual(json.loads(path.read_text())['reserve_bytes'], 20)

    def test_live_child_exit_expires_external_binding(self):
        import subprocess
        import sys
        registry = MemoryRegistry(self.root / 'child')
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            registry.configure_budget(64_000_000_000, 8_000_000_000, [child.pid])
            child.terminate()
            child.wait(timeout=5)
            registry.snapshot()
            policy = json.loads((self.root / 'child' / 'budget.json').read_text())
            self.assertEqual(policy['external'], [])
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def test_ambiguous_local_start_uses_earliest_plausible_epoch(self):
        import time
        from unittest.mock import patch
        from mlx_agent.memory_budget import _earliest_start_ns
        previous = os.environ.get('TZ')
        try:
            with patch.dict(os.environ, {'TZ': 'America/New_York'}):
                time.tzset()
                self.assertEqual(_earliest_start_ns('Sun Nov  3 01:30:00 2024 process'),
                                 1_730_611_800_000_000_000)
                self.assertIsNone(_earliest_start_ns('not a valid start'))
        finally:
            if previous is None:
                os.environ.pop('TZ', None)
            else:
                os.environ['TZ'] = previous
            time.tzset()

    def test_unknown_process_accounting_retains_observed_headroom(self):
        self.budget()
        def unavailable():
            raise OSError('process table unavailable')
        self.registry.process_probe = unavailable
        with self.assertRaises(MemoryAdmissionError) as failure:
            self.cold('first')
        self.assertEqual(failure.exception.code, 'memory_unknown')
        self.assertEqual(failure.exception.details['available_bytes'], 10000)
        self.assertEqual(failure.exception.details['required_available_bytes'], 70)

    def test_execution_unknown_process_retains_own_reservation(self):
        self.budget()
        self.cold('first')
        def unavailable():
            raise OSError('process table unavailable')
        self.registry.process_probe = unavailable
        with self.assertRaises(MemoryAdmissionError) as failure:
            self.registry.reserve_execution('first', limit_bytes=100)
        self.assertEqual(failure.exception.code, 'memory_unknown')
        self.assertEqual(failure.exception.details['workspace_bytes'], 10)
        self.assertEqual(failure.exception.details['route_limit_bytes'], 100)

    def test_execution_refusals_retain_numeric_accounting(self):
        self.budget(limit=150, reserve=20)
        self.cold('first')
        with self.assertRaises(MemoryAdmissionError) as failure:
            self.registry.reserve_execution('first', limit_bytes=69, probe=lambda: 10000)
        self.assertEqual(failure.exception.code, 'job_memory_cap_exceeded')
        self.assertEqual(failure.exception.details['required_bytes'], 70)
        self.assertEqual(failure.exception.details['agent_budget']['admission_bytes'], 130)
        with self.assertRaises(MemoryAdmissionError) as failure:
            self.registry.reserve_execution('first', limit_bytes=140, probe=lambda: 10000)
        self.assertEqual(failure.exception.code, 'agent_memory_budget_exceeded')
        self.assertEqual(failure.exception.details['agent_budget']['limit_bytes'], 150)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'compute'
        self.rows = {101: {'parent': 1, 'group': 101, 'rss': 5, 'identity': 'owner'},
                     201: {'parent': 101, 'group': 201, 'rss': 70, 'identity': 'worker'},
                     301: {'parent': 1, 'group': 301, 'rss': 10, 'identity': 'browser'},
                     302: {'parent': 301, 'group': 301, 'rss': 5, 'identity': 'child'}}
        self.registry = MemoryRegistry(self.root, owner=(101, 'owner'),
            inspector=lambda pid: self.rows.get(pid, {}).get('identity'),
            alive=lambda pid: pid in self.rows, group_probe=lambda pid: [],
            process_probe=lambda: self.rows)

    def budget(self, limit=200, reserve=20, external=None):
        return self.registry.configure_budget(limit, reserve, external or [])

    def cold(self, key, resident=60, workspace=10):
        return self.registry.reserve(key, 'a' * 64, resident, workspace, 0, lambda: 10000)

    def test_pending_and_resident_allocations_share_one_aggregate_limit(self):
        self.budget(limit=150, reserve=20)
        self.cold('first')  # 70 pending plus 5 owner overhead
        self.registry.launching('first')
        self.registry.bind_worker('first', 201, 'worker', 201)
        self.registry.settle('first', resident_bytes=60)
        # 70 actual worker RSS + 5 owner + 56 next allocation exceeds 130.
        with self.assertRaises(MemoryAdmissionError) as caught:
            self.cold('second', resident=46)
        self.assertEqual(caught.exception.code, 'agent_memory_budget_exceeded')
        self.assertEqual(self.registry.snapshot()['entries'], 1)
        self.cold('second', resident=45)

    def test_external_process_tree_is_counted_once_even_when_roots_overlap(self):
        self.budget(limit=100, reserve=10, external=[301, 302])
        self.cold('first', resident=60)
        budget = self.registry.snapshot()['agent_budget']
        self.assertEqual(budget['external_bytes'], 15)
        self.assertEqual(budget['accounted_bytes'], 90)  # 70 + 15 + 5
        with self.assertRaises(MemoryAdmissionError):
            self.cold('second', resident=1, workspace=0)

    def test_replaced_external_pid_is_not_charged_as_the_original_agent(self):
        self.budget(external=[301])
        self.rows[301]['identity'] = 'replacement'
        self.rows[301]['group'] = 400
        del self.rows[302]
        self.assertEqual(self.registry.snapshot()['agent_budget']['external_bytes'], 0)

    def test_unknown_process_accounting_blocks_new_load_without_losing_warm_worker(self):
        self.budget()
        self.cold('first')
        self.registry.launching('first')
        self.registry.bind_worker('first', 201, 'worker', 201)
        self.registry.settle('first', resident_bytes=60)
        self.rows[201]['rss'] = None
        with self.assertRaises(MemoryAdmissionError) as caught:
            self.cold('second', resident=1)
        self.assertEqual(caught.exception.code, 'memory_unknown')
        self.assertEqual(self.registry.snapshot()['entries'], 1)

    def test_warm_workspace_is_charged_once_and_denial_preserves_previous_reservation(self):
        self.budget(limit=120, reserve=20)
        self.cold('first')
        self.registry.launching('first')
        self.registry.bind_worker('first', 201, 'worker', 201)
        self.registry.settle('first', resident_bytes=60)
        self.registry.reserve_workspace('first', 25, lambda: 10000)
        with self.assertRaises(MemoryAdmissionError):
            self.registry.reserve_workspace('first', 26, lambda: 10000)
        self.assertEqual(self.registry.snapshot()['reserved_bytes'], 25)

    def test_active_execution_reserves_remaining_capacity_against_competing_loads(self):
        self.budget(limit=150, reserve=20)
        self.cold('first')
        self.registry.launching('first')
        self.registry.bind_worker('first', 201, 'worker', 201)
        self.registry.settle('first', resident_bytes=60)
        self.assertEqual(self.registry.reserve_execution('first'), 125)
        with self.assertRaises(MemoryAdmissionError):
            self.cold('second', resident=1, workspace=0)
        self.registry.settle('first', resident_bytes=70)
        self.cold('second', resident=45, workspace=10)

    def test_job_cap_includes_warm_pending_component_reservation(self):
        self.budget(limit=300, reserve=20)
        self.cold('first')
        self.registry.settle('first', resident_bytes=60)
        self.registry.reserve_workspace('first', 40, lambda: 10000)
        with self.assertRaises(MemoryAdmissionError) as caught:
            self.registry.reserve_execution('first', 99, lambda: 10000)
        self.assertEqual(caught.exception.code, 'job_memory_cap_exceeded')
        self.assertEqual(self.registry.reserve_execution('first', 100, lambda: 10000), 100)

    def test_capped_jobs_share_admission_and_growth_consumes_the_reservation(self):
        self.budget(limit=220, reserve=20)
        self.cold('first')
        self.assertEqual(self.registry.reserve_execution('first', 100), 100)
        self.cold('second', resident=50)
        self.assertEqual(self.registry.reserve_execution('second', 90), 90)
        budget = self.registry.snapshot()['agent_budget']
        self.assertEqual(budget['accounted_bytes'], 195)
        self.assertTrue(budget['within_budget'])
        self.registry.launching('first')
        self.registry.bind_worker('first', 201, 'worker', 201)
        # Materializing the worker cannot add its usage on top of its ceiling.
        self.assertEqual(self.registry.snapshot()['agent_budget']['accounted_bytes'], 195)
        self.registry.settle('first', resident_bytes=60)
        self.registry.reserve_workspace('first', 10, lambda: 10000)
        self.registry.reserve_execution('first', 100)
        self.rows[201]['rss'] = 95
        budget = self.registry.snapshot()['agent_budget']
        self.assertEqual(budget['accounted_bytes'], 195)
        self.assertEqual(budget['allocations']['first']['pending_bytes'], 5)
        self.assertTrue(budget['within_budget'])
        self.registry.settle('first', resident_bytes=95)
        self.assertEqual(self.registry.snapshot()['agent_budget']['accounted_bytes'], 190)

    def test_cap_admission_failure_preserves_rows_and_does_not_take_partial_capacity(self):
        self.budget(limit=180, reserve=20)
        self.cold('first')
        self.registry.reserve_execution('first', 100)
        self.cold('second', resident=40, workspace=10)
        before = (self.root / 'memory.json').read_bytes()
        with self.assertRaises(MemoryAdmissionError) as caught:
            self.registry.reserve_execution('second', 60)
        self.assertEqual(caught.exception.code, 'agent_memory_budget_exceeded')
        self.assertEqual((self.root / 'memory.json').read_bytes(), before)
        self.assertEqual(self.registry.snapshot()['agent_budget']['accounted_bytes'], 155)
        self.registry.release('first')
        self.assertEqual(self.registry.reserve_execution('second', 60), 60)

    def test_cap_smaller_than_model_or_workspace_fails_before_dispatch(self):
        self.budget()
        self.cold('first')
        before = (self.root / 'memory.json').read_bytes()
        with self.assertRaises(MemoryAdmissionError) as caught:
            self.registry.reserve_execution('first', 69)
        self.assertEqual(caught.exception.code, 'job_memory_cap_exceeded')
        self.assertEqual(caught.exception.details['required_bytes'], 70)
        self.assertEqual((self.root / 'memory.json').read_bytes(), before)

    def test_cap_without_shared_policy_is_enforced_and_does_not_enable_sharing(self):
        self.cold('first')
        self.assertFalse(self.registry.shared_execution_enabled())
        self.assertEqual(self.registry.reserve_execution('first', 80), 80)
        import json
        row = json.loads((self.root / 'memory.json').read_text())['entries']['first']
        self.assertEqual(row['reserved_bytes'], 80)
        self.assertEqual(row['execution_bytes'], 80)
        self.budget()
        self.assertTrue(self.registry.shared_execution_enabled())

    def test_full_job_ceiling_must_also_fit_system_headroom(self):
        self.budget()
        self.cold('first')
        with self.assertRaises(MemoryAdmissionError) as caught:
            self.registry.reserve_execution('first', 100, probe=lambda: 99)
        self.assertEqual(caught.exception.code, 'insufficient_headroom')
        self.assertEqual(self.registry.snapshot()['reserved_bytes'], 70)

    def test_corrupt_or_symlink_policy_cannot_disable_an_existing_budget(self):
        self.budget()
        path = self.root / 'budget.json'
        path.write_text('{}')
        with self.assertRaises(MemoryAdmissionError):
            self.cold('first')
        path.unlink()
        target = self.root / 'outside'
        target.write_text('unique')
        path.symlink_to(target)
        with self.assertRaises(MemoryAdmissionError):
            self.cold('first')
        self.assertEqual(target.read_text(), 'unique')

    def test_cli_configures_and_reads_the_same_persisted_budget(self):
        import io
        import json
        from contextlib import redirect_stdout
        from unittest.mock import patch

        from mlx_agent.cli import main
        with patch('mlx_agent.residency.MemoryRegistry', return_value=self.registry):
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['serve', 'budget', '--limit-gb', '64', '--reserve-gb', '8', '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())['data']['admission_bytes'], 56_000_000_000)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(['serve', 'budget', '--json']), 0)
            self.assertEqual(json.loads(output.getvalue())['data']['agent_budget']['limit_bytes'], 64_000_000_000)

    def test_defunct_worker_is_not_mistaken_for_a_replacement_process(self):
        import subprocess
        import sys
        import time

        from mlx_agent.compute import process_identity
        registry = MemoryRegistry(self.root / 'live')
        registry.configure_budget(64_000_000_000, 8_000_000_000)
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'],
                                 start_new_session=True)
        try:
            identity = process_identity(child.pid)
            registry.reserve('worker', 'a' * 64, 1024, 0, 0, lambda: 10**12)
            registry.launching('worker')
            registry.bind_worker('worker', child.pid, identity, child.pid)
            registry.settle('worker', resident_bytes=1024)
            child.terminate()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                state = subprocess.check_output(['/bin/ps', '-p', str(child.pid), '-o', 'stat='], text=True).strip()
                if state.startswith('Z'):
                    break
                time.sleep(.01)
            self.assertTrue(state.startswith('Z'))
            if sys.platform == 'darwin':
                from mlx_agent.memory_budget import _physical_bytes
                # ps may have read the state before the command became defunct.
                with self.assertRaises(ProcessLookupError):
                    _physical_bytes(child.pid)
            budget = registry.snapshot()['agent_budget']
            self.assertNotEqual(budget.get('status'), 'unavailable', budget)
            self.assertTrue(budget['within_budget'])
        finally:
            child.wait(timeout=5)
            registry.release('worker')

    def test_live_worker_identity_replacement_still_blocks_admission(self):
        self.budget()
        self.cold('first')
        self.registry.launching('first')
        self.registry.bind_worker('first', 201, 'worker', 201)
        self.registry.settle('first', resident_bytes=60)
        self.rows[201]['identity'] = 'replacement'
        self.assertEqual(self.registry.snapshot()['agent_budget']['status'], 'unavailable')
        with self.assertRaises(MemoryAdmissionError):
            self.cold('second')

    def test_metal_physical_footprint_is_charged_when_rss_is_smaller(self):
        self.rows[201]['rss'] = 7
        self.rows[201]['physical_bytes'] = 70
        self.budget(limit=100, reserve=20, external=[201])
        self.assertEqual(self.registry.snapshot()['agent_budget']['observed_bytes'], 75)
        with self.assertRaises(MemoryAdmissionError):
            self.cold('first', resident=6, workspace=0)

    def test_unavailable_physical_footprint_cannot_fall_back_to_smaller_rss(self):
        self.rows[201]['physical_bytes'] = None
        with self.assertRaises(MemoryAdmissionError):
            self.budget(external=[201])

    def test_changing_budget_limits_keeps_included_external_processes(self):
        self.budget(external=[301])
        result = self.registry.configure_budget(300, 20)
        self.assertEqual(result['external_bytes'], 15)


if __name__ == '__main__':
    unittest.main()
