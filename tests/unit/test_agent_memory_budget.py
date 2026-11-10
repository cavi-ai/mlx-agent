"""A shared budget counts pending loads, warm models and external agent families."""
import tempfile
import unittest
from pathlib import Path

from mlx_agent.residency import MemoryAdmissionError, MemoryRegistry


class AgentBudgetTests(unittest.TestCase):
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

    def test_changing_budget_limits_keeps_included_external_processes(self):
        self.budget(external=[301])
        result = self.registry.configure_budget(300, 20)
        self.assertEqual(result['external_bytes'], 15)


if __name__ == '__main__':
    unittest.main()
