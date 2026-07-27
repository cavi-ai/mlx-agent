import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main
from mlx_agent.model_doctor import (
    PruneError,
    execute_prune,
    plan_prune,
)
from tests.unit.test_model_doctor import _hf_repo


class PlanPruneTests(unittest.TestCase):
    def test_only_incomplete_snapshots_are_candidates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _hf_repo(root, "pub/good")
            _hf_repo(root, "pub/broken", dangling=True)
            plan = plan_prune(root)
            self.assertEqual(len(plan["candidates"]), 1)
            self.assertEqual(plan["candidates"][0]["repo"], "pub/broken")
            self.assertEqual(len(plan["preview_hash"]), 64)

    def test_healthy_cache_has_no_candidates(self):
        with TemporaryDirectory() as directory:
            _hf_repo(Path(directory), "pub/good")
            plan = plan_prune(Path(directory))
            self.assertEqual(plan["candidates"], [])
            self.assertEqual(plan["total_bytes"], 0)


class ExecutePruneTests(unittest.TestCase):
    def test_preview_without_confirm(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _hf_repo(root, "pub/broken", dangling=True)
            plan = plan_prune(root)
            outcome = execute_prune(plan)
            self.assertEqual(outcome["status"], "preview")
            self.assertTrue((root / "models--pub--broken").is_dir())

    def test_confirm_requires_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _hf_repo(root, "pub/broken", dangling=True)
            plan = plan_prune(root)
            with self.assertRaises(PruneError) as caught:
                execute_prune(plan, confirm=True)
            self.assertEqual(caught.exception.code, "preview_hash_required")

    def test_stale_hash_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _hf_repo(root, "pub/broken", dangling=True)
            plan = plan_prune(root)
            with self.assertRaises(PruneError) as caught:
                execute_prune(plan, confirm=True, preview_hash="0" * 64)
            self.assertEqual(caught.exception.code, "preview_stale")

    def test_confirmed_prune_removes_exact_candidates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _hf_repo(root, "pub/good")
            _hf_repo(root, "pub/broken", dangling=True)
            plan = plan_prune(root)
            outcome = execute_prune(
                plan, confirm=True, preview_hash=plan["preview_hash"]
            )
            self.assertEqual(outcome["status"], "pruned")
            self.assertFalse((root / "models--pub--broken").exists())
            self.assertTrue((root / "models--pub--good").is_dir())

    def test_outside_cache_target_is_refused(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {
                "cache": str(root),
                "candidates": [{"repo": "pub/x", "path": str(root.parent), "bytes": 0}],
                "preview_hash": "abc",
            }
            with self.assertRaises(PruneError) as caught:
                execute_prune(plan, confirm=True, preview_hash="abc")
            self.assertEqual(caught.exception.code, "unsafe_target")


class PruneCliTests(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_prune_preview_requires_confirmation(self):
        with TemporaryDirectory() as directory:
            _hf_repo(Path(directory), "pub/broken", dangling=True)
            code, output = self._run([
                "doctor", "models", "--prune", "--hf-cache", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertTrue(payload["data"]["requires_confirmation"])
            self.assertEqual(len(payload["data"]["plan"]["candidates"]), 1)

    def test_prune_confirm_applies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _hf_repo(root, "pub/broken", dangling=True)
            code, output = self._run([
                "doctor", "models", "--prune", "--hf-cache", directory, "--json",
            ])
            plan = json.loads(output)["data"]["plan"]
            code, output = self._run([
                "doctor", "models", "--prune", "--hf-cache", directory,
                "--confirm", "--preview-hash", plan["preview_hash"], "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["data"]["status"], "pruned")
            self.assertFalse((root / "models--pub--broken").exists())

    def test_prune_nothing_is_clean_exit(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "doctor", "models", "--prune", "--hf-cache", directory, "--json",
            ])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
