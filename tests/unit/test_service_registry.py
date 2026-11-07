import json
import os
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.service_registry import register_receipts, registered_roots, discover_services, live_receipt_roots


class ServiceRegistryTests(unittest.TestCase):
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

    def test_symlinked_directories_are_refused(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'real').mkdir()
            (root / 'link').symlink_to(root / 'real', target_is_directory=True)
            with self.assertRaises(ValueError):
                register_receipts(folder, root / 'link')

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
