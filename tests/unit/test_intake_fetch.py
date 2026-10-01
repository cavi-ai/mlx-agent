import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent import fetch_runner
from mlx_agent.intake_fetch import FETCH_IGNORE_PATTERNS, FETCH_RUNNER, FetchError, plan_fetch, start_fetch, status_fetch


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.spawned = []

    def spawn(self, argv, log_path, env):
        self.spawned.append((argv, env))
        return 777

    def test_plan_shapes(self):
        plan = plan_fetch("https://huggingface.co/unsloth/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q4_K_M.gguf",
                          local_dir="/models/gguf/Qwen3-8B-GGUF", python="/venv/bin/python")
        self.assertEqual(plan["argv"], ["/venv/bin/python", str(FETCH_RUNNER), "--repo", "unsloth/Qwen3-8B-GGUF",
                                        "--revision", "main", "--file", "Qwen3-8B-Q4_K_M.gguf",
                                        "--local-dir", "/models/gguf/Qwen3-8B-GGUF"])
        snapshot = plan_fetch("org/name", hf_cache="/hf", python="/venv/bin/python")
        self.assertEqual(snapshot["argv"][-2:], ["--cache-dir", "/hf"])
        self.assertIsNone(snapshot["file"])
        with self.assertRaises(FetchError):
            plan_fetch("org/name", local_dir="relative/dir")

    def test_snapshot_plan_skips_formats_mlx_never_reads(self):
        snapshot = plan_fetch("org/name", hf_cache="/hf", python="/venv/bin/python")
        argv = snapshot["argv"]
        for pattern in ("*.h5", "*.msgpack"):
            self.assertIn(pattern, argv)
            self.assertEqual(argv[argv.index(pattern) - 1], "--ignore")
        self.assertEqual(argv[-2:], ["--cache-dir", "/hf"])
        self.assertEqual(snapshot["ignore_patterns"], list(FETCH_IGNORE_PATTERNS))
        self.assertEqual(argv[:6], ["/venv/bin/python", str(FETCH_RUNNER), "--repo", "org/name", "--revision", "main"])
        self.assertEqual(argv[6:8], ["--ignore", FETCH_IGNORE_PATTERNS[0]])
        single = plan_fetch("org/name", file="model.gguf", python="/venv/bin/python")
        self.assertNotIn("--ignore", single["argv"])
        self.assertEqual(single["ignore_patterns"], [])

    def test_start_and_status_lifecycle(self):
        plan = plan_fetch("org/name", python="/venv/bin/python")
        self.assertEqual(start_fetch(plan)["status"], "preview")
        with self.assertRaises(FetchError) as caught:
            start_fetch(plan, receipts_dir=str(self.base), confirm=True, preview_hash=plan["preview_hash"],
                        spawn=self.spawn, module_present=lambda names: list(names))
        self.assertEqual(caught.exception.code, "runtime_not_installed")
        outcome = start_fetch(plan, receipts_dir=str(self.base), confirm=True, preview_hash=plan["preview_hash"],
                              spawn=self.spawn, module_present=lambda names: [], env={"PATH": "/bin", "AWS_SECRET_ACCESS_KEY": "x"})
        argv, env = self.spawned[0]
        self.assertEqual(argv[-2], "--marker")
        self.assertEqual(env, {"PATH": "/bin"})
        marker = Path(outcome["receipt"]["marker"])
        running = status_fetch(str(self.base), pid_alive=lambda pid: True)
        self.assertEqual([job["state"] for job in running], ["running"])
        marker.write_text(json.dumps({"exit_status": "done", "path": "/hf/snap", "finished_at": "t"}), encoding="utf-8")
        done = status_fetch(str(self.base), pid_alive=lambda pid: False)
        self.assertEqual((done[0]["state"], done[0]["path"]), ("done", "/hf/snap"))
        marker.unlink()
        failed = status_fetch(str(self.base), pid_alive=lambda pid: False)
        self.assertEqual(failed[0]["state"], "failed")

    def test_runner_writes_marker_on_failure(self):
        marker = self.base / "m.done.json"
        code = fetch_runner.main(["--repo", "org/name", "--revision", "main", "--marker", str(marker)],
                                 download=lambda **kwargs: (_ for _ in ()).throw(OSError("offline")))
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["exit_status"], "failed")

    def test_runner_passes_patterns_and_local_dir(self):
        marker = self.base / "m.done.json"
        seen = {}

        def download(**kwargs):
            seen.update(kwargs)
            return "/models/x"

        code = fetch_runner.main(["--repo", "org/name", "--revision", "main", "--file", "a.gguf",
                                  "--local-dir", "/models/x", "--marker", str(marker)], download=download)
        self.assertEqual(code, 0)
        self.assertEqual(seen, {"repo_id": "org/name", "revision": "main", "allow_patterns": ["a.gguf"], "local_dir": "/models/x"})
        self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["path"], "/models/x")

    def test_runner_passes_ignore_patterns(self):
        marker = self.base / "m.done.json"
        seen = {}

        def download(**kwargs):
            seen.update(kwargs)
            return "/models/x"

        code = fetch_runner.main(["--repo", "org/name", "--revision", "main", "--ignore", "*.h5",
                                  "--ignore", "*.msgpack", "--marker", str(marker)], download=download)
        self.assertEqual(code, 0)
        self.assertEqual(seen, {"repo_id": "org/name", "revision": "main", "ignore_patterns": ["*.h5", "*.msgpack"]})


if __name__ == "__main__":
    unittest.main()
