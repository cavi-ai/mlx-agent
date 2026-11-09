import hashlib
import os
import sys
import time
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mlx_agent.serving_access import read_access_key
from mlx_agent.serve import plan_start, load_recipes, ServeError


class ServingAccessTests(unittest.TestCase):
    def test_spawn_reaps_an_exited_owned_child_and_closes_parent_log(self):
        from mlx_agent.serve import _default_spawn, _pid_alive
        with TemporaryDirectory() as folder, warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always', ResourceWarning)
            pid = _default_spawn([sys.executable, '-c', 'pass'], str(Path(folder) / 'worker.log'))
            try:
                deadline = time.monotonic() + 3
                while _pid_alive(pid) and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertFalse(_pid_alive(pid), 'an exited owned child still appears alive')
                self.assertFalse([w for w in observed if issubclass(w.category, ResourceWarning)])
            finally:
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass

    def test_key_requires_owned_private_file_and_reviewed_digest(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'access-key'
            path.write_text('a' * 64 + '\n')
            path.chmod(0o600)
            key, digest = read_access_key(str(path))
            self.assertEqual(key, 'a' * 64)
            self.assertEqual(digest, hashlib.sha256(key.encode()).hexdigest())
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                read_access_key(str(path))
            path.chmod(0o600)
            link = Path(folder) / 'link'
            link.symlink_to(path)
            with self.assertRaises((ValueError, OSError)):
                read_access_key(str(link))
            path.write_text('b' * 64)
            with self.assertRaises(ValueError):
                read_access_key(str(path), digest)

    def test_protected_plan_binds_key_and_worker_without_retaining_credentials(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            (model / 'model.safetensors').write_bytes(b'fixture')
            key = root / 'access-key'
            key.write_text('access-key-for-local-inference-0123456789')
            key.chmod(0o600)
            args = dict(repo=None, runtime='mlx_lm', recipes=load_recipes(), path=str(model),
                        jit=True, api_key_file=str(key), receipts_dir=str(root))
            plan = plan_start(**args)
            self.assertIn('private_worker', plan)
            self.assertEqual(plan['readiness'], 'http://127.0.0.1:8080/_mlx/status')
            self.assertNotIn(key.read_text(), str(plan))
            key.write_text('different-key-for-local-inference-0123456789')
            self.assertNotEqual(plan_start(**args)['preview_hash'], plan['preview_hash'])
            with self.assertRaises(ServeError):
                plan_start(**dict(args, jit=False))
            accelerated = plan_start(**dict(args, prompt_lookup=True))
            self.assertNotIn('private_worker', accelerated)
            self.assertEqual(accelerated['api_key_sha256'], plan_start(**args)['api_key_sha256'])
