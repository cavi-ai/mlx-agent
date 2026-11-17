"""Shared reservations use real cross-process locks, without model runtimes."""
import json
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest


class RegistryTests(unittest.TestCase):
    maxDiff = None
    def setUp(self):
        from mlx_agent.residency import MemoryRegistry, MemoryAdmissionError
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'memory'
        self.Registry, self.Error = MemoryRegistry, MemoryAdmissionError
        self.registry = MemoryRegistry(self.root)

    def reserve(self, key, resident=80, workspace=10, available=150, headroom=10):
        return self.registry.reserve(key, 'a'*64, resident, workspace, headroom, lambda: available)

    def test_pending_loads_share_headroom_without_counting_resident_bytes_twice(self):
        self.reserve('a')
        with self.assertRaises(self.Error) as caught:
            self.reserve('b', resident=50)
        self.assertEqual(caught.exception.code, 'insufficient_headroom')
        self.registry.settle('a', resident_bytes=80, workspace_bytes=0)
        # The OS now reports70 free after80 resident bytes. Deducting those80
        # again would incorrectly reject a50+10 allocation with10 headroom.
        self.reserve('b', resident=50, available=70)
        rows = self.registry.snapshot()
        self.assertEqual(rows['resident_bytes'],130)
        self.assertEqual(rows['reserved_bytes'],60)
        self.registry.release('b')
        self.assertEqual(self.registry.snapshot()['reserved_bytes'],0)

    def test_workspace_denial_preserves_existing_resident_entry(self):
        self.reserve('a')
        self.registry.settle('a', resident_bytes=80, workspace_bytes=0)
        with self.assertRaises(self.Error):
            self.registry.reserve_workspace('a',31,lambda:40)
        self.assertEqual(self.registry.snapshot()['reserved_bytes'],0)
        self.assertEqual(self.registry.snapshot()['resident_bytes'],80)
        self.registry.reserve_workspace('a',30,lambda:40)
        self.assertEqual(self.registry.snapshot()['reserved_bytes'],30)

    def test_job_cap_includes_warm_growth_without_shared_policy(self):
        self.reserve('a', resident=80, workspace=10, available=500)
        self.registry.settle('a', resident_bytes=80)
        self.registry.reserve_workspace('a', 30, lambda: 500)
        with self.assertRaises(self.Error) as caught:
            self.registry.reserve_execution('a', 100, lambda: 500)
        self.assertEqual(caught.exception.code, 'job_memory_cap_exceeded')
        self.assertEqual(self.registry.reserve_execution('a', 110, lambda: 500), 110)

    def test_auto_execution_uses_available_headroom_for_cold_and_warm(self):
        from unittest.mock import patch
        self.reserve('a', resident=80, workspace=10, available=500)
        budget = {'allocations': {'a': {'resident_bytes': 0, 'pending_bytes': 90}},
                  'admission_bytes': 1000, 'accounted_bytes': 90}
        with patch.object(self.registry, '_agent_budget', return_value=budget):
            diagnostics = {}
            self.assertEqual(self.registry.reserve_execution('a', None, lambda: 160, diagnostics), 150)
            self.assertEqual(diagnostics['agent_capacity_bytes'], 1000)
            self.assertEqual(diagnostics['physical_ceiling_bytes'], 150)
            self.assertEqual(diagnostics['effective_grant_bytes'], 150)
        self.registry.settle('a', resident_bytes=80, workspace_bytes=0)
        self.registry.reserve_workspace('a', 10, lambda: 500)
        budget['allocations']['a'] = {'resident_bytes': 80, 'pending_bytes': 10}
        with patch.object(self.registry, '_agent_budget', return_value=budget):
            self.assertEqual(self.registry.reserve_execution('a', None, lambda: 70, overhead_bytes=20), 120)

    def test_execution_refusals_preserve_reservation(self):
        from unittest.mock import patch
        self.reserve('a', resident=80, workspace=10, available=500)
        budget = {'allocations': {'a': {'resident_bytes': 0, 'pending_bytes': 90}},
                  'admission_bytes': 1000, 'accounted_bytes': 90}
        with patch.object(self.registry, '_agent_budget', return_value=budget):
            before = self.registry.snapshot()
            for limit, probe, expected in ((None, lambda: 80, 'insufficient_headroom'),
                                           (200, lambda: 150, 'insufficient_headroom'),
                                           (None, lambda: None, 'memory_unknown')):
                with self.subTest(expected=expected), self.assertRaises(self.Error) as caught:
                    self.registry.reserve_execution('a', limit, probe)
                self.assertEqual(caught.exception.code, expected)
                self.assertEqual(self.registry.snapshot(), before)

    def test_auto_grant_leaves_existing_controller_margin(self):
        from unittest.mock import patch

        from mlx_agent.memory_budget import account
        self.reserve('a', resident=80, workspace=10, available=1000, headroom=0)
        budget = {'allocations': {'a': {'resident_bytes': 0, 'pending_bytes': 90}},
                  'admission_bytes': 80_000_000_000, 'accounted_bytes': 50_000_090}
        with patch.object(self.registry, '_agent_budget', return_value=budget):
            diagnostics = {}
            grant = self.registry.reserve_execution('a', None, lambda: 90_000_000_000,
                                                    diagnostics, overhead_bytes=512_000_000)
        self.assertEqual(grant, 79_438_000_000)
        self.assertEqual(diagnostics['controller_headroom_bytes'], 512_000_000)
        policy = {'limit_bytes': 88_000_000_000, 'reserve_bytes': 8_000_000_000, 'external': []}
        entry = {'owner': [1, 'owner'], 'phase': 'reserved', 'reserved_bytes': grant, 'worker': None}
        def measured(overhead):
            rows = {1: {'parent': 0, 'group': 1, 'identity': 'owner', 'rss': overhead, 'physical_bytes': overhead}}
            return account(policy, {'a': entry}, rows, (1, 'owner'))['within_budget']
        self.assertTrue(measured(50_032_768))
        self.assertFalse(measured(562_000_001))

    def test_margin_is_auto_only_and_preserves_minimum(self):
        from unittest.mock import patch
        self.reserve('a', resident=80, workspace=10, available=500)
        budget = {'allocations': {'a': {'resident_bytes': 0, 'pending_bytes': 90}},
                  'admission_bytes': 1000, 'accounted_bytes': 90}
        with patch.object(self.registry, '_agent_budget', return_value=budget):
            self.assertEqual(self.registry.reserve_execution('a', 100, lambda: 500, overhead_bytes=50), 100)
        self.registry.release('a')
        self.reserve('a', resident=80, workspace=10, available=500)
        with patch.object(self.registry, '_agent_budget', return_value=budget), self.assertRaises(self.Error):
            self.registry.reserve_execution('a', None, lambda: 120, overhead_bytes=30)

    def test_live_headroom_changes_apply_to_other_endpoints(self):
        self.reserve('a')
        self.registry.settle('a',workspace_bytes=0)
        self.registry.set_headroom('a',30)
        with self.assertRaises(self.Error):self.reserve('b',resident=10,workspace=0,headroom=0,available=35)
        self.registry.set_headroom('a',20)
        self.reserve('b',resident=10,workspace=0,headroom=0,available=35)

    def test_dead_owner_recovery_retains_live_or_unknown_workers_and_ambiguous_launches(self):
        alive={11:True,12:True,41:True}
        identities={11:'old',12:'new',41:'worker'}
        options=dict(inspector=lambda p:identities.get(p), alive=lambda p:alive.get(p,False), group_probe=lambda p:[])
        old=self.Registry(self.root,owner=(11,'old'),**options)
        for key in ('reserved','launched','bound'):
            old.reserve(key,'b'*64,20,0,0,lambda:200)
        old.launching('launched')
        old.launching('bound')
        old.bind_worker('bound',41,'worker',41)
        alive[11]=False
        current=self.Registry(self.root,owner=(12,'new'),**options)
        self.assertEqual(current.snapshot()['entries'],2)
        identities[41]=None
        self.assertEqual(current.snapshot()['entries'],2)
        alive[41]=False
        self.assertEqual(current.snapshot()['entries'],1)
        # An unbound launch cannot be assumed harmless after its owner dies.
        self.assertEqual(current.snapshot()['reserved_bytes'],20)

    def test_symlink_and_corrupt_registry_are_refused(self):
        self.reserve('a')
        path=self.root/'memory.json'
        path.write_text('{"schema":"bad"}')
        with self.assertRaises(self.Error):self.registry.snapshot()
        path.unlink()
        target=self.root/'untouched'
        target.write_text('unique')
        path.symlink_to(target)
        with self.assertRaises(self.Error):self.registry.snapshot()
        self.assertEqual(target.read_text(),'unique')

    def test_two_processes_cannot_spend_the_same_available_memory(self):
        program='''
import json,sys
from mlx_agent.residency import MemoryRegistry,MemoryAdmissionError
r=MemoryRegistry(sys.argv[1])
sys.stdin.readline()
try:
    r.reserve(sys.argv[2],'c'*64,80,10,10,lambda:150)
    outcome='reserved'
except MemoryAdmissionError as error:
    import traceback
    outcome=error.code if error.code=='insufficient_headroom' else traceback.format_exc()
print(json.dumps(outcome),flush=True)
sys.stdin.readline()
'''
        processes=[subprocess.Popen([sys.executable,'-c',program,str(self.root),key],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for key in ('a','b')]
        try:
            for process in processes:
                process.stdin.write('go\n');process.stdin.flush()
            outcomes=[]
            for process in processes:
                ready,_,_=select.select([process.stdout],[],[],8)
                self.assertTrue(ready,'reservation child did not respond')
                line=process.stdout.readline()
                self.assertTrue(line,process.stderr.read() if process.poll() is not None else '')
                outcomes.append(json.loads(line))
            self.assertCountEqual(outcomes,['reserved','insufficient_headroom'])
        finally:
            for process in processes:
                process.communicate('stop\n',timeout=8)


class TelemetryTests(unittest.TestCase):
    def test_torch_allocation_does_not_report_mlx_counters(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from mlx_agent.worker_telemetry import capture
        torch = SimpleNamespace(mps=SimpleNamespace(current_allocated_memory=lambda: 123))
        core = SimpleNamespace(get_active_memory=lambda: 999, get_cache_memory=lambda: 888,
                               get_peak_memory=lambda: 777)
        with patch.dict(sys.modules, {'torch': torch, 'mlx.core': core}):
            value = capture({'loads': 0, 'cache_hits': 0, 'load_seconds': 0.0}, 'torch')
        self.assertEqual((value['active_bytes'], value['cache_bytes'], value['peak_bytes']), (123, None, None))

    def test_unavailable_allocator_methods_remain_unknown(self):
        from unittest.mock import patch,Mock
        from types import SimpleNamespace
        from mlx_agent.worker_telemetry import capture
        core=SimpleNamespace(get_active_memory=Mock(side_effect=RuntimeError('unavailable')))
        with patch.dict(sys.modules,{'mlx.core':core}):
            self.assertIsNone(capture({'loads':0,'cache_hits':0,'load_seconds':0.0})['active_bytes'])

    def test_missing_runtime_is_unknown_and_counters_are_versioned(self):
        from unittest.mock import patch
        from mlx_agent.worker_telemetry import capture,validate
        with patch.dict(sys.modules,{'mlx.core':None}):
            value=capture({'loads':1,'cache_hits':2,'load_seconds':.25})
        self.assertEqual(value['schema'],'worker-memory/1')
        self.assertIsNone(value['active_bytes'])
        self.assertIsNone(value['cache_bytes'])
        self.assertEqual(validate(value)['cache_hits'],2)

    def test_invalid_metrics_do_not_become_trusted_allocations(self):
        from mlx_agent.worker_telemetry import capture,validate
        value=capture({'loads':0,'cache_hits':0,'load_seconds':0.0})
        for field,bad in (('active_bytes',-1),('cache_bytes',True),('loads',1.5),('load_seconds',float('nan'))):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):validate(dict(value,**{field:bad}))
