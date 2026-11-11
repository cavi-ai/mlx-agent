import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mlx_agent import serve_profiles as profiles


class ServeProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = dict(memory_budget_gb=24, min_headroom_gb=24, strata_context_tokens=16384,
                             strata_decoding="lookup", idle_timeout=300, unload_under_pressure=True,
                             keep_loaded=False)

    def test_common_profile_has_no_strata_requirement(self):
        settings = dict(min_headroom_gb=24, idle_timeout=600, keep_loaded=False,
                        unload_under_pressure=True, execution_wait_seconds=2)
        profiles.save_profile('working', settings, self.root)
        self.assertEqual(profiles.load_profile('working', self.root), settings)

    def test_cli_saves_task_budget_without_adding_strata_settings(self):
        import io
        from contextlib import redirect_stdout
        from mlx_agent.cli import main
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(['serve', 'profile', 'save', 'tasks', '--receipts-dir', str(self.root),
                         '--min-headroom-gb', '2', '--memory-budget-gb', '12', '--json'])
        self.assertEqual(code, 0, output.getvalue())
        settings = profiles.load_profile('tasks', self.root)
        self.assertEqual(settings['memory_budget_gb'], 12)
        self.assertNotIn('strata_context_tokens', settings)
        self.assertNotIn('strata_decoding', settings)

    def test_task_budget_profile_needs_no_strata_context_or_decoding(self):
        settings = dict(min_headroom_gb=2, idle_timeout=0, keep_loaded=False,
                        unload_under_pressure=False, memory_budget_gb=12)
        profiles.save_profile('tasks', settings, self.root)
        self.assertEqual(profiles.load_profile('tasks', self.root), settings)

    def test_round_trip_replace_delete_and_private_storage(self):
        profiles.save_profile("shared", self.settings, self.root)
        self.assertEqual(profiles.load_profile("shared", self.root), self.settings)
        self.assertEqual(profiles.list_profiles(self.root), {"shared": self.settings})
        with self.assertRaises(ValueError):
            profiles.save_profile("shared", self.settings, self.root)
        changed = dict(self.settings, memory_budget_gb=20)
        profiles.save_profile("shared", changed, self.root, replace=True)
        self.assertEqual(profiles.load_profile("shared", self.root), changed)
        files = list(self.root.rglob("shared.json"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].stat().st_mode & 0o777, 0o600)
        profiles.delete_profile("shared", self.root)
        self.assertEqual(profiles.list_profiles(self.root), {})
        with self.assertRaises(ValueError):
            profiles.load_profile("shared", self.root)

    def test_rejects_unsafe_names_fields_and_values_without_writing(self):
        for name in ("../escape", "", "UPPER", "/tmp/x", "a" * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                profiles.save_profile(name, self.settings, self.root)
        for delta in ({"exe": "evil"}, {"memory_budget_gb": float("nan")},
                      {"min_headroom_gb": True}, {"strata_context_tokens": 12},
                      {"strata_decoding": "auto"}, {"keep_loaded": True}, {"idle_timeout": -1}):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                profiles.save_profile("bad", dict(self.settings, **delta), self.root)
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_refuses_symlinked_store_and_modified_profile(self):
        profiles.save_profile("shared", self.settings, self.root)
        file = next(self.root.rglob("shared.json"))
        original = file.read_bytes()
        file.write_text(json.dumps(dict(self.settings, exe="evil")))
        with self.assertRaises(ValueError):
            profiles.load_profile("shared", self.root)
        file.write_bytes(original)
        external = self.root / "outside.json"
        external.write_bytes(original)
        file.unlink()
        file.symlink_to(external)
        for operation in (lambda: profiles.load_profile("shared", self.root),
                          lambda: profiles.save_profile("shared", self.settings, self.root, replace=True),
                          lambda: profiles.delete_profile("shared", self.root)):
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual(external.read_bytes(), original)

    def test_missing_store_listing_has_no_side_effect(self):
        self.assertEqual(profiles.list_profiles(self.root), {})
        self.assertEqual(list(self.root.iterdir()), [])

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO required")
    def test_replace_refuses_a_nonregular_target_without_removing_it(self):
        import stat
        profiles.save_profile("shared", self.settings, self.root)
        path = next(self.root.rglob("shared.json"))
        path.unlink()
        os.mkfifo(path, 0o600)
        before = path.lstat().st_ino
        with self.assertRaises(ValueError):
            profiles.save_profile("shared", self.settings, self.root, replace=True)
        self.assertTrue(stat.S_ISFIFO(path.lstat().st_mode))
        self.assertEqual(path.lstat().st_ino, before)
