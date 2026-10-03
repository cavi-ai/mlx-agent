import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent import fetch_runner
from mlx_agent.backends import load_manifests
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

    def test_a_ported_snapshot_downloads_only_the_port_files(self):
        plan = plan_fetch("convaiinnovations/laya", python="/venv/bin/python", model_type="laya")
        files = load_manifests()["mlx-embeddings"]["port_signatures"]["laya"]
        self.assertEqual((plan["allow_patterns"], plan["ignore_patterns"]), (files, []))
        argv = plan["argv"]
        self.assertEqual([argv[index + 1] for index, value in enumerate(argv) if value == "--allow"], files)
        self.assertNotIn("--ignore", argv)
        other = plan_fetch("org/name", python="/venv/bin/python", model_type="qwen2")
        self.assertEqual((other["allow_patterns"], other["ignore_patterns"]), ([], list(FETCH_IGNORE_PATTERNS)))
        self.assertEqual(other["preview_hash"], plan_fetch("org/name", python="/venv/bin/python")["preview_hash"])
        single = plan_fetch("org/name", file="a.gguf", python="/venv/bin/python", model_type="laya")
        self.assertEqual((single["allow_patterns"], single["file"]), ([], "a.gguf"))

    def test_a_subfolder_snapshot_downloads_only_that_folder(self):
        ported = plan_fetch("convaiinnovations/laya/multilingual", python="/venv/bin/python", model_type="laya")
        files = load_manifests()["mlx-embeddings"]["port_signatures"]["laya"]
        self.assertEqual(ported["allow_patterns"], ["multilingual/" + name for name in files])
        self.assertEqual((ported["subfolder"], ported["ignore_patterns"]), ("multilingual", []))
        self.assertIn("multilingual", ported["slug"])
        generic = plan_fetch("https://huggingface.co/org/name/tree/main/chat", python="/venv/bin/python")
        self.assertEqual((generic["allow_patterns"], generic["ignore_patterns"]), (["chat/*"], list(FETCH_IGNORE_PATTERNS)))
        argv = generic["argv"]
        self.assertEqual(argv[argv.index("--allow") + 1], "chat/*")
        self.assertNotEqual(generic["preview_hash"], plan_fetch("org/name", python="/venv/bin/python")["preview_hash"])

    def test_a_recipe_downloads_its_base_snapshot_and_its_lora(self):
        plan = plan_fetch("abenzerps/Qwen-Image-2.1-Uncensored-GGUF", python="/venv/bin/python", model_type="qwen_image_21")
        argv = plan["argv"]
        self.assertEqual(plan["allow_patterns"], ["qwen-image-2.1-uncensored-lora.safetensors"])
        self.assertEqual(plan["base"], {"repo": "Qwen/Qwen-Image-2.1", "revision": "d26bb61231c349cf6b7896fa83353113880e1ba3"})
        self.assertEqual(argv[argv.index("--base-repo") + 1], "Qwen/Qwen-Image-2.1")
        self.assertEqual([argv[i + 1] for i, value in enumerate(argv) if value == "--base-ignore"], list(FETCH_IGNORE_PATTERNS))
        self.assertNotIn("--ignore", argv)
        self.assertIsNone(plan_fetch("abenzerps/Qwen-Image-2.1-Uncensored-GGUF", python="/venv/bin/python")["base"])

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
        self.assertEqual((outcome["receipt"]["subfolder"], running[0]["subfolder"]), (plan["subfolder"], plan["subfolder"]))
        folder = plan_fetch("org/name/chat", python="/venv/bin/python")
        start_fetch(folder, receipts_dir=str(self.base), confirm=True, preview_hash=folder["preview_hash"],
                    spawn=self.spawn, module_present=lambda names: [])
        jobs = status_fetch(str(self.base), pid_alive=lambda pid: True)
        self.assertEqual({(job["repo"], job["subfolder"]) for job in jobs},
                         {(plan["repo"], plan["subfolder"]), ("org/name", "chat")})
        (self.base / ".mlx-agent-receipts" / "fetch" / "{0}.json".format(folder["slug"])).unlink()
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
        seen.clear()
        fetch_runner.main(["--repo", "org/name", "--revision", "main", "--allow", "model.safetensors",
                           "--allow", "encoder/config.json", "--marker", str(marker)], download=download)
        self.assertEqual(seen["allow_patterns"], ["model.safetensors", "encoder/config.json"])
        calls = []
        fetch_runner.main(["--repo", "org/lora", "--revision", "main", "--allow", "x.safetensors", "--base-repo", "org/base",
                           "--base-revision", "abc", "--base-ignore", "*.h5", "--marker", str(marker)],
                          download=lambda **kwargs: calls.append(kwargs) or "/hf/" + kwargs["repo_id"])
        self.assertEqual(calls, [{"repo_id": "org/base", "revision": "abc", "ignore_patterns": ["*.h5"]},
                                 {"repo_id": "org/lora", "revision": "main", "allow_patterns": ["x.safetensors"]}])
        self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["path"], "/hf/org/lora")


if __name__ == "__main__":
    unittest.main()
