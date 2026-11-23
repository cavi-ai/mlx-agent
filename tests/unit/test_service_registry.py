import json
import os
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.service_registry import MAX_INDEX_FILES, MAX_ROOTS, register_receipts, registered_roots, discover_services, live_receipt_roots


class ServiceRegistryTests(unittest.TestCase):
    def test_enumeration_stops_after_bounded_entries_even_for_non_json(self):
        from mlx_agent import service_registry as registry_module

        with TemporaryDirectory() as folder:
            index = Path(folder) / 'index'
            index.mkdir()
            for i in range(MAX_INDEX_FILES + 5):
                (index / f'{i:04d}.other').touch()
            real = os.scandir
            counts = []

            class Limited:
                def __init__(self, path):
                    self.iterator = real(path)
                    self.count = 0

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.iterator.close()
                    counts.append(self.count)

                def __iter__(self):
                    return self

                def __next__(self):
                    self.count += 1
                    if self.count > MAX_INDEX_FILES + 1:
                        raise AssertionError('unbounded enumeration')
                    return next(self.iterator)

            with patch.object(registry_module.os, 'scandir', side_effect=Limited):
                result = discover_services(registry=index, process_roots=[], inspect=lambda _: [])
            self.assertTrue(result['errors'])
            self.assertLessEqual(max(counts), MAX_INDEX_FILES + 1)

    def test_registration_survives_concurrent_stale_prune(self):
        import threading
        from mlx_agent import service_registry as registry_module
        from mlx_agent.transactions import ConcurrentTransactionError, _target_locks

        with TemporaryDirectory() as folder:
            root = Path(folder)
            index = root / 'index'
            target = root / 'target'
            target.mkdir()
            register_receipts(target, index)
            target.rmdir()
            hit, proceed, started = threading.Event(), threading.Event(), threading.Event()
            real_unlink = os.unlink

            def delayed(path, *args, **kwargs):
                if str(path).endswith('.json') and not hit.is_set():
                    hit.set()
                    if not proceed.wait(5):
                        raise TimeoutError('stale prune was not released')
                return real_unlink(path, *args, **kwargs)

            def register_again():
                started.set()
                register_receipts(target, index)

            other = root / 'other'
            other.mkdir()
            with patch.object(registry_module.os, 'unlink', delayed):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(register_receipts, other, index)
                    try:
                        self.assertTrue(hit.wait(5))
                        with self.assertRaises(ConcurrentTransactionError):
                            with _target_locks([index]):
                                pass
                        target.mkdir()
                        second = pool.submit(register_again)
                        self.assertTrue(started.wait(5))
                    finally:
                        proceed.set()
                    first.result()
                    second.result()
            self.assertIn(str(target.resolve()), registered_roots(index))

    def test_non_object_records_are_errors_without_aborting_discovery(self):
        for value in ([], None, 17, 'record'):
            with self.subTest(value=value), TemporaryDirectory() as folder:
                registry = Path(folder) / 'registry'; registry.mkdir()
                (registry / 'bad.json').write_text(json.dumps(value))
                result = discover_services(registry=registry, process_roots=[], inspect=lambda _: [])
                self.assertEqual(result['services'], [])
                self.assertEqual(len(result['errors']), 1)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires named pipes')
    def test_registry_fifo_is_rejected_without_blocking(self):
        with TemporaryDirectory() as folder:
            registry = Path(folder) / 'registry'; registry.mkdir()
            os.mkfifo(registry / 'fifo.json')
            code = "from mlx_agent.service_registry import discover_services; import sys; r=discover_services(registry=sys.argv[1], process_roots=[]); assert len(r['errors']) == 1"
            subprocess.run([sys.executable, '-c', code, str(registry)], timeout=3, check=True)

    def test_live_gateway_arguments_preserve_spaces(self):
        with TemporaryDirectory(prefix='service space ') as folder:
            config = str(Path(folder) / '.mlx-agent-receipts/serve/19081.jit-config.json')
            process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)',
                                        '/fixture/src/mlx_agent/jit_serve.py', '--config', config])
            try:
                self.assertIn(folder, live_receipt_roots())
            finally:
                process.terminate(); process.wait(timeout=5)


    def test_independent_roots_survive_concurrent_registration(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            registry = root / 'registry'
            roots = [root / str(index) for index in range(12)]
            for item in roots:
                item.mkdir()
            with ThreadPoolExecutor(max_workers=4) as workers:
                list(workers.map(lambda item: register_receipts(item, registry), roots))
            self.assertEqual(set(registered_roots(registry)), {str(item.resolve()) for item in roots})
            register_receipts(roots[0], registry)
            self.assertEqual(len(registered_roots(registry)), len(roots))

    def test_corrupt_and_symlinked_records_are_reported_without_becoming_roots(self):
        with TemporaryDirectory() as folder:
            registry = Path(folder) / 'registry'
            register_receipts(folder, registry)
            (registry / 'bad.json').write_text('{bad')
            (registry / 'link.json').symlink_to(next(registry.glob('*.json')))
            result = discover_services(registry=registry, process_roots=[], inspect=lambda root: [])
            self.assertEqual(len(result['errors']), 2)
            self.assertEqual(result['schema_version'], 1)
            self.assertEqual(result['services'], [])
            self.assertTrue(result['capabilities']['scoped_watch'])

    def test_symlinked_directories_are_refused(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'real').mkdir()
            (root / 'link').symlink_to(root / 'real', target_is_directory=True)
            with self.assertRaises(ValueError):
                register_receipts(folder, root / 'link')

    def test_missing_registered_root_does_not_hide_healthy_services_but_active_roots_fail_closed(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); registry = root / 'registry'; old = root / 'old'; good = root / 'good'
            old.mkdir(); good.mkdir(); register_receipts(old, registry); register_receipts(good, registry)
            old.rmdir()
            def inspect(path):
                return [{'receipt': str(Path(path) / '.mlx-agent-receipts/serve/1.json'), 'alive': False, 'argv_match': False}]
            result = discover_services(registry=registry, process_roots=[], inspect=inspect)
            self.assertEqual(len(result['services']), 1)
            self.assertEqual(result['errors'], [])
            self.assertEqual(len(discover_services([str(old)], registry=registry, process_roots=[], inspect=inspect)['errors']), 1)
            self.assertEqual(len(discover_services(registry=registry, process_roots=[str(old)], inspect=inspect)['errors']), 1)

    def test_duplicate_roots_are_inspected_once_and_ownership_is_preserved(self):
        with TemporaryDirectory() as folder:
            registry = Path(folder) / 'registry'
            register_receipts(folder, registry)
            calls = []
            def inspect(root):
                calls.append(root)
                return [{'receipt': str(Path(root) / '.mlx-agent-receipts/serve/9000.json'),
                         'port': 9000, 'pid': 42, 'alive': True, 'argv_match': True, 'jit': True, 'model_state': 'unloaded'}]
            result = discover_services([folder], registry=registry, process_roots=[folder], inspect=inspect)
            self.assertEqual(len(calls), 1)
            service = result['services'][0]
            self.assertEqual(service['receipts_dir'], str(Path(folder).resolve()))
            self.assertEqual(service['model_state'], 'unloaded')
            self.assertTrue(service['argv_match'])
            self.assertIn('service_id', service)

    def test_a_failed_source_does_not_erase_healthy_services(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            good, bad = root / 'good', root / 'bad'
            good.mkdir(); bad.mkdir()
            def inspect(path):
                if Path(path).name == 'bad':
                    raise PermissionError('unavailable')
                return [{'receipt': str(good / '.mlx-agent-receipts/serve/1.json'), 'alive': False, 'argv_match': False}]
            result = discover_services([str(bad), str(good)], registry=root / 'missing', process_roots=[], inspect=inspect)
            self.assertEqual(len(result['services']), 1)
            self.assertFalse(result['services'][0]['alive'])
            self.assertEqual(len(result['errors']), 1)

    def test_process_discovery_uses_only_current_users_gateway_config_path(self):
        path = '/models/service/.mlx-agent-receipts/serve/19080.jit-config.json'
        rows = f'42 {os.getuid()} python /engine/src/mlx_agent/jit_serve.py --config {path}\n'
        rows += f'43 {os.getuid()+1} python /engine/src/mlx_agent/jit_serve.py --config /other/.mlx-agent-receipts/serve/1.jit-config.json\n'
        rows += f'44 {os.getuid()} python random.py --config /ignored/.mlx-agent-receipts/serve/1.jit-config.json\n'
        self.assertEqual(live_receipt_roots(rows), ['/models/service'])

    def test_stale_registrations_above_max_roots_do_not_trigger_limit_error(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            registry = root / 'registry'
            registry.mkdir()
            live = []
            for index in range(MAX_ROOTS):
                item = root / f'live-{index:02d}'
                item.mkdir()
                live.append(str(item.resolve()))
                (registry / f'z-live-{index:02d}.json').write_text(json.dumps({
                    'schema_version': 1, 'receipts_dir': str(item),
                }))
            for index in range(15):
                stale = root / f'stale-{index:02d}'
                (registry / f'a-stale-{index:02d}.json').write_text(json.dumps({
                    'schema_version': 1, 'receipts_dir': str(stale),
                }))
            inspected = []
            result = discover_services(
                registry=registry,
                process_roots=[],
                inspect=lambda path: inspected.append(path) or [],
            )
            self.assertEqual(result['errors'], [])
            self.assertEqual(set(inspected), set(live))
            self.assertEqual(len(inspected), MAX_ROOTS)

    def test_more_than_max_roots_existing_directories_emits_limit_error_and_caps(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            registry = root / 'registry'
            for index in range(MAX_ROOTS + 3):
                item = root / f'live-{index:02d}'
                item.mkdir()
                register_receipts(item, registry)
            inspected = []
            result = discover_services(
                registry=registry,
                process_roots=[],
                inspect=lambda path: inspected.append(path) or [],
            )
            self.assertEqual(len(inspected), MAX_ROOTS)
            self.assertEqual(
                result['errors'],
                [{'source': str(registry), 'message': 'Service location limit reached.'}],
            )

    def test_exceeding_max_index_files_scan_bound_emits_limit_error(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            registry = root / 'registry'
            registry.mkdir()
            for index in range(MAX_INDEX_FILES + 2):
                stale = root / f'stale-{index:03d}'
                (registry / f'{index:04d}.json').write_text(json.dumps({
                    'schema_version': 1, 'receipts_dir': str(stale),
                }))
            result = discover_services(registry=registry, process_roots=[], inspect=lambda _: [])
            self.assertEqual(result['services'], [])
            self.assertEqual(
                result['errors'],
                [{'source': str(registry), 'message': 'Service location limit reached.'}],
            )

    def test_register_receipts_prunes_stale_registrations_and_preserves_live_and_unsafe(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            registry = root / 'registry'
            live_a = root / 'live-a'
            live_b = root / 'live-b'
            stale = root / 'stale'
            live_a.mkdir()
            live_b.mkdir()
            stale.mkdir()

            register_receipts(live_a, registry)
            register_receipts(stale, registry)
            stale_files = set(registry.glob('*.json'))
            stale.rmdir()

            corrupt = registry / 'corrupt.json'
            corrupt.write_text('{bad')
            link = registry / 'link.json'
            link.symlink_to(next(iter(stale_files)))

            register_receipts(live_b, registry)

            self.assertEqual(set(registered_roots(registry)), {str(live_a.resolve()), str(live_b.resolve())})
            # Stale regular registration was pruned; corrupt file and symlink were left untouched.
            self.assertTrue(corrupt.exists())
            self.assertTrue(link.is_symlink())
            remaining_records = [
                json.loads(item.read_text())['receipts_dir']
                for item in registry.glob('*.json')
                if not item.is_symlink() and item.name != 'corrupt.json'
            ]
            self.assertEqual(set(remaining_records), {str(live_a.resolve()), str(live_b.resolve())})

    def test_live_receipt_roots_discovers_task_serve_processes(self):
        uid = os.getuid()
        rows = (
            f"101 {uid} python3 /opt/mlx_agent/jit_serve.py --config /a/.mlx-agent-receipts/serve/x.jit-config.json\n"
            f"102 {uid} python3 /opt/mlx_agent/task_serve.py --config /b/.mlx-agent-receipts/serve/y.jit-config.json\n"
        )
        self.assertEqual(live_receipt_roots(process_rows=rows), ['/a', '/b'])

    def test_prune_stale_registrations_fsyncs_only_on_unlink_and_prunes_enotdir(self):
        import tempfile
        from unittest.mock import patch
        from mlx_agent.service_registry import _prune_stale_registrations, discover_services

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / 'registry'
            live = root / 'live'
            replaced = root / 'replaced' / 'nested'
            live.mkdir()
            replaced.mkdir(parents=True)
            register_receipts(live, registry)
            register_receipts(replaced, registry)

            # No-op prune should not call fsync.
            with patch('mlx_agent.service_registry.os.fsync', wraps=os.fsync) as fsync_spy:
                _prune_stale_registrations(registry)
                self.assertEqual(fsync_spy.call_count, 0)

            # Replace ancestor directory with a regular file (ENOTDIR) -> pruned with fsync.
            replaced.rmdir()
            (root / 'replaced').rmdir()
            (root / 'replaced').write_text('now-a-file', encoding='utf-8')

            with patch('mlx_agent.service_registry.os.fsync', wraps=os.fsync) as fsync_spy:
                _prune_stale_registrations(registry)
                self.assertEqual(fsync_spy.call_count, 1)

            self.assertEqual(registered_roots(registry), [str(live.resolve())])

            # Symlinked registration root must still surface an error in discover_services.
            symlink_root = root / 'sym'
            target_dir = root / 'target'
            target_dir.mkdir()
            symlink_root.mkdir()
            register_receipts(symlink_root, registry)
            symlink_root.rmdir()
            symlink_root.symlink_to(target_dir, target_is_directory=True)
            result = discover_services(registry=registry, process_roots=[], inspect=lambda _r: [])
            self.assertTrue(any('refusing unsafe directory component' in err['message'] for err in result['errors']))
