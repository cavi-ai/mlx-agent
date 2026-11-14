import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent import cli
from mlx_agent.gguf import file_signature, inventory


class StandaloneAssetsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def file(self, name, data=b'weights'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_normal_scan_inventories_assets_without_hashing(self):
        path = self.file('checkpoints/model.safetensors')
        with patch('mlx_agent.standalone_assets.hash_file', side_effect=AssertionError('unexpected hash')):
            result = inventory([str(self.root)])
        assets = [item for item in result['storage']['items'] if item['kind'] == 'asset']
        self.assertEqual([item['path'] for item in assets], [str(path)])
        self.assertEqual(assets[0]['asset']['sha256'], None)
        self.assertEqual(assets[0]['asset']['mtime_ns'], path.stat().st_mtime_ns)
        self.assertEqual(assets[0]['asset']['ctime_ns'], path.stat().st_ctime_ns)
        self.assertEqual(assets[0]['asset']['nlink'], 1)
        self.assertFalse(result['asset_scan']['compared'])
        self.assertEqual(result['duplicates'], [])

    def test_explicit_comparison_uses_full_bytes_and_size_candidates(self):
        first = self.file('a.safetensors', b'a' * 16 + b'x' + b'z' * 16)
        second = self.file('nested/b.ckpt', b'a' * 16 + b'x' + b'z' * 16)
        different = self.file('c.ckpt', b'a' * 16 + b'y' + b'z' * 16)
        self.assertEqual(file_signature(first, chunk_bytes=16), file_signature(different, chunk_bytes=16))
        self.file('unique.ckpt', b'unique')
        result = inventory([str(self.root)], compare_assets=True)
        groups = [item for item in result['duplicates'] if item.get('verification') == 'full_sha256']
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['keep'], str(first))
        self.assertEqual(groups[0]['model_key'], first.name)
        self.assertEqual(groups[0]['redundant'], [str(second)])
        self.assertEqual(result['asset_scan']['candidates'], 3)
        self.assertEqual(result['asset_scan']['checked'], 3)
        self.assertEqual(result['asset_scan']['errors'], [])

    def test_hardlinks_and_symlinks_never_become_removable_duplicates(self):
        original = self.file('original.ckpt')
        os.link(original, self.root / 'hardlink.ckpt')
        (self.root / 'link.ckpt').symlink_to(original)
        outside = self.root.parent / (self.root.name + '-outside')
        outside.mkdir()
        self.addCleanup(lambda: outside.rmdir())
        (self.root / 'outside').symlink_to(outside, target_is_directory=True)
        result = inventory([str(self.root)], compare_assets=True)
        self.assertFalse([group for group in result['duplicates'] if group.get('verification') == 'full_sha256'])
        self.assertEqual(result['asset_scan']['candidates'], 0)
        self.assertEqual(result['storage']['disk_bytes'], len(b'weights'))
        self.assertNotIn(str(self.root / 'link.ckpt'), [item['path'] for item in result['storage']['items']])

    def test_model_directories_provider_internals_and_shards_excluded(self):
        for marker in ('config.json', 'model_index.json', 'model.safetensors.index.json'):
            self.file(marker.replace('.', '-') + '/' + marker, b'{}')
            self.file(marker.replace('.', '-') + '/weights.safetensors')
        self.file('models--org--repo/blobs/sha.safetensors')
        self.file('ollama/blobs/weight.ckpt')
        self.file('ollama/manifests/manifest', b'{}')
        self.file('model-00001-of-00002.safetensors')
        allowed = self.file('checkpoints/standalone.ckpt')
        result = inventory([str(self.root)])
        self.assertEqual([item['path'] for item in result['storage']['items'] if item['kind'] == 'asset'], [str(allowed)])

    def test_failed_hash_reports_partial_coverage(self):
        first = self.file('a.ckpt')
        self.file('b.ckpt')
        with patch('mlx_agent.standalone_assets.hash_file', side_effect=OSError('changed or unreadable')):
            result = inventory([str(self.root)], compare_assets=True)
        self.assertEqual(result['asset_scan']['candidates'], 2)
        self.assertEqual(result['asset_scan']['checked'], 0)
        self.assertEqual(len(result['asset_scan']['errors']), 2)
        self.assertTrue(all(item['path'] for item in result['asset_scan']['errors']))
        self.assertFalse([group for group in result['duplicates'] if group.get('verification') == 'full_sha256'])

    def test_real_mid_read_change_reports_error(self):
        self.file('a.ckpt', b'a' * 32)
        self.file('b.ckpt', b'a' * 32)
        from mlx_agent import standalone_assets
        actual_read = os.read
        changed = False
        def mutate(fd, size):
            nonlocal changed
            data = actual_read(fd, size)
            if not changed:
                changed = True
                (self.root / 'a.ckpt').write_bytes(b'b' * 32)
            return data
        with patch.object(standalone_assets.os, 'read', side_effect=mutate):
            result = inventory([str(self.root)], compare_assets=True)
        self.assertTrue(result['asset_scan']['errors'])
        self.assertEqual(result['asset_scan']['checked'], 1)
        self.assertFalse([group for group in result['duplicates'] if group.get('verification') == 'full_sha256'])

    def test_registered_nested_root_inside_model_directory_is_excluded(self):
        self.file('model/config.json', b'{}')
        checkpoint = self.file('model/subfolder/checkpoint.ckpt')
        result = inventory([str(checkpoint.parent)])
        self.assertFalse([item for item in result['storage']['items'] if item['kind'] == 'asset'])

    def test_regular_assets_do_not_become_inference_models(self):
        checkpoint = self.file('checkpoints/a.ckpt')
        result = inventory([str(self.root)])
        self.assertIn(str(checkpoint), [item['path'] for item in result['storage']['items']])
        self.assertEqual(result['models'], [])
        self.assertEqual(result['outputs'], [])

    def test_swapped_directory_symlink_cannot_escape_during_comparison(self):
        source = self.file('inner/a.ckpt')
        self.file('b.ckpt')
        with TemporaryDirectory() as outside:
            (Path(outside) / 'a.ckpt').write_bytes(b'weights')
            from mlx_agent import standalone_assets
            original = standalone_assets.hash_file
            swapped = False
            def replace_directory(path, *args):
                nonlocal swapped
                if path == str(source) and not swapped:
                    swapped = True
                    (self.root / 'inner').rename(self.root / 'moved')
                    (self.root / 'inner').symlink_to(outside, target_is_directory=True)
                return original(path, *args)
            with patch.object(standalone_assets, 'hash_file', side_effect=replace_directory):
                result = inventory([str(self.root)], compare_assets=True)
        self.assertTrue(any(item['path'] == str(source) for item in result['asset_scan']['errors']))
        self.assertFalse([group for group in result['duplicates'] if group.get('verification') == 'full_sha256'])

    def test_two_registered_roots_are_compared(self):
        first = self.file('one/first.ckpt')
        second = self.file('two/other.safetensors')
        result = inventory([str(first.parent)], [str(second.parent)], compare_assets=True)
        group = next(group for group in result['duplicates'] if group.get('verification') == 'full_sha256')
        self.assertEqual({group['keep'], *group['redundant']}, {str(first), str(second)})

    def test_new_model_marker_during_comparison_invalidates_evidence(self):
        checkpoint = self.file('one/a.ckpt')
        self.file('two/b.ckpt')
        from mlx_agent import standalone_assets
        original = standalone_assets.hash_file
        def mark_owner(path, *args):
            result = original(path, *args)
            if path == str(checkpoint):
                (checkpoint.parent / 'config.json').write_text('{}')
            return result
        with patch.object(standalone_assets, 'hash_file', side_effect=mark_owner):
            result = inventory([str(self.root)], compare_assets=True)
        self.assertTrue(any(item['path'] == str(checkpoint) for item in result['asset_scan']['errors']))
        self.assertFalse([group for group in result['duplicates'] if group.get('verification') == 'full_sha256'])

    def test_asset_limit_reports_incomplete_coverage(self):
        self.file('a.ckpt'); self.file('b.ckpt')
        result = inventory([str(self.root)], limit=1)
        self.assertTrue(result['asset_scan']['errors'])
        self.assertIn('limit', result['asset_scan']['errors'][0]['message'])

    def test_relative_and_overlapping_roots_count_files_once(self):
        path = self.file('nested/asset.ckpt')
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        result = inventory(['.', 'nested'])
        self.assertEqual(len([item for item in result['storage']['items'] if item['kind'] == 'asset']), 1)
        self.assertEqual(result['storage']['disk_bytes'], path.stat().st_size)
        self.assertEqual(sum(row['total_bytes'] for row in result['storage']['roots']), path.stat().st_size)

    def test_excluded_quarantine_is_not_inventoried_or_hashed(self):
        first = self.file('models/a.ckpt')
        second = self.file('models/b.safetensors')
        quarantined = self.file('models/holding-area/old.ckpt')
        from mlx_agent import standalone_assets
        original = standalone_assets.hash_file
        visited = []
        def record(path, *args):
            visited.append(path)
            return original(path, *args)
        with patch.object(standalone_assets, 'hash_file', side_effect=record):
            result = inventory([str(first.parent)], compare_assets=True,
                               asset_exclude_roots=[str(quarantined.parent)])
        items = [item['path'] for item in result['storage']['items'] if item['kind'] == 'asset']
        self.assertEqual(set(items), {str(first), str(second)})
        self.assertEqual(set(visited), {str(first), str(second)})
        self.assertEqual(result['asset_scan']['excluded_roots'], [str(quarantined.parent.resolve())])
        self.assertEqual(result['asset_scan']['errors'], [])
        self.assertEqual(len([group for group in result['duplicates'] if group.get('verification') == 'full_sha256']), 1)

    def test_registered_root_inside_exclusion_is_skipped(self):
        quarantined = self.file('holding-area/nested/old.ckpt')
        result = inventory([str(quarantined.parent)], compare_assets=True,
                           asset_exclude_roots=[str(self.root / 'holding-area')])
        self.assertFalse([item for item in result['storage']['items'] if item['kind'] == 'asset'])
        self.assertEqual(result['asset_scan']['candidates'], 0)
        self.assertEqual(result['asset_scan']['excluded_roots'], [str((self.root / 'holding-area').resolve())])

    def test_real_cli_reports_full_digest_and_readable_human_output(self):
        first = self.file('a.ckpt')
        self.file('b.safetensors')
        quarantined = self.file('holding-area/old.ckpt')
        script = Path(__file__).resolve().parents[2] / 'scripts/mlx-agent'
        arguments = [sys.executable, str(script), 'convert', 'scan', '--gguf-root', str(self.root),
                     '--receipts-dir', str(self.root / 'receipts'), '--compare-assets', '--asset-exclude-root', str(quarantined.parent)]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        json_run = subprocess.run(arguments + ['--json'], capture_output=True, text=True, env=environment, timeout=20)
        self.assertEqual(json_run.returncode, 0, json_run.stderr)
        result = json.loads(json_run.stdout)['data']
        group = next(group for group in result['duplicates'] if group.get('verification') == 'full_sha256')
        self.assertEqual(group['keep'], str(first))
        self.assertEqual(group['model_key'], first.name)
        self.assertNotIn(str(quarantined), [item['path'] for item in result['storage']['items']])
        self.assertEqual(result['asset_scan']['excluded_roots'], [str(quarantined.parent.resolve())])
        human = subprocess.run(arguments, capture_output=True, text=True, env=environment, timeout=20)
        self.assertEqual(human.returncode, 0, human.stderr)
        self.assertIn('identical model files, keep ' + str(first), human.stdout)
        self.assertNotIn('asset:sha256:', human.stdout)

    def test_cli_comparison_is_opt_in(self):
        self.file('a.ckpt'); self.file('b.ckpt')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main(['convert', 'scan', '--gguf-root', str(self.root), '--compare-assets', '--json'])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())['data']['asset_scan']['compared'])


if __name__ == '__main__':
    unittest.main()
