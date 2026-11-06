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

    def test_a_port_that_reads_other_repositories_downloads_their_files_too(self):
        plan = plan_fetch("microsoft/TRELLIS.2-4B", hf_cache="/hf", python="/venv/bin/python", model_type="trellis2")
        requires = load_manifests()["mlx-3d"]["port_requires"]["trellis2"]
        self.assertEqual(plan["requires"], {repo: requires[repo] for repo in sorted(requires)})
        argv = plan["argv"]
        pulled = [argv[index + 1] for index, value in enumerate(argv) if value == "--require"]
        self.assertEqual(pulled, ["facebook/dinov3-vitl16-pretrain-lvd1689m=config.json",
                                  "facebook/dinov3-vitl16-pretrain-lvd1689m=model.safetensors",
                                  "microsoft/TRELLIS-image-large=ckpts/ss_dec_conv3d_16l8_fp16.json",
                                  "microsoft/TRELLIS-image-large=ckpts/ss_dec_conv3d_16l8_fp16.safetensors"])
        self.assertEqual(argv[-2:], ["--cache-dir", "/hf"])
        self.assertNotIn("requires", plan_fetch("VAST-AI/TripoSG", python="/venv/bin/python", model_type="triposg"))
        single = plan_fetch("microsoft/TRELLIS.2-4B", file="a.gguf", python="/venv/bin/python", model_type="trellis2")
        self.assertNotIn("--require", single["argv"])

    def test_a_port_that_reads_a_file_outside_hugging_face_downloads_it_by_its_digest(self):
        plan = plan_fetch("TencentARC/Pixal3D", hf_cache="/hf", python="/venv/bin/python", model_type="pixal3d",
                          downloads_dir="/dl")
        entry = {"url": "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth",
                 "sha256": "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f", "size": 2664431}
        argv = plan["argv"]
        self.assertEqual([argv[index + 1] for index, value in enumerate(argv) if value == "--download"],
                         ["{0}={1}={2}".format(entry["sha256"], entry["size"], entry["url"])])
        self.assertEqual(argv[argv.index("--downloads-dir") + 1], "/dl")
        self.assertEqual(plan["downloads"], [entry])
        self.assertEqual(plan["network"], ["huggingface.co", "github.com"])
        self.assertEqual(sorted(plan["requires"]), ["Ruicheng/moge-2-vitl", "facebook/dinov3-vitl16-pretrain-lvd1689m"])
        trellis = plan_fetch("microsoft/TRELLIS.2-4B", python="/venv/bin/python", model_type="trellis2", downloads_dir="/dl")
        self.assertNotIn("--download", trellis["argv"])
        self.assertNotIn("downloads", trellis)
        self.assertEqual(trellis["network"], ["huggingface.co"])

    def test_runner_keeps_a_download_only_when_its_size_and_digest_match(self):
        import hashlib
        import io

        payload = b"naf weights"
        digest = hashlib.sha256(payload).hexdigest()
        url = "https://example.org/files/naf.pth"
        opened = []

        def opener(address):
            opened.append(address)
            return io.BytesIO(payload)

        downloads = self.base / "downloads"
        marker = self.base / "m.done.json"
        argv = ["--repo", "org/model", "--revision", "main", "--downloads-dir", str(downloads), "--marker", str(marker)]
        self.assertEqual(fetch_runner.main(argv + ["--download", "{0}={1}={2}".format(digest, len(payload), url)],
                                           download=lambda **kwargs: "/hf/snapshot", opener=opener), 0)
        stored = downloads / digest / "naf.pth"
        self.assertEqual(stored.read_bytes(), payload)
        self.assertEqual(opened, [url])
        self.assertEqual(fetch_runner.main(argv + ["--download", "{0}={1}={2}".format(digest, len(payload), url)],
                                           download=lambda **kwargs: "/hf/snapshot", opener=opener), 0)
        self.assertEqual(opened, [url])
        wrong = "0" * 64
        snapshots = []
        code = fetch_runner.main(argv + ["--download", "{0}={1}={2}".format(wrong, len(payload), url)],
                                 download=lambda **kwargs: snapshots.append(kwargs), opener=opener)
        self.assertEqual(code, 1)
        self.assertEqual(list((downloads / wrong).iterdir()), [])
        self.assertEqual(snapshots, [])
        result = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual((result["exit_status"], result["error"]),
                         ("failed", "{0} did not match its recorded size and sha256.".format(url)))

    def test_runner_requests_required_files_before_the_snapshot(self):
        marker = self.base / "m.done.json"
        requests = []

        def hub(**kwargs):
            requests.append(kwargs)
            if kwargs["repo_id"] == "gated/encoder":
                raise PermissionError("403 gated")
            return "/hf/snapshot"

        argv = ["--repo", "org/model", "--revision", "main", "--allow", "pipeline.json",
                "--require", "other/decoder=ckpts/dec.json", "--require", "other/decoder=ckpts/dec.safetensors",
                "--cache-dir", "/hf", "--marker", str(marker)]
        self.assertEqual(fetch_runner.main(argv, download=hub), 0)
        self.assertEqual(requests, [
            {"repo_id": "other/decoder", "allow_patterns": ["ckpts/dec.json", "ckpts/dec.safetensors"], "cache_dir": "/hf"},
            {"repo_id": "org/model", "revision": "main", "allow_patterns": ["pipeline.json"], "cache_dir": "/hf"},
        ])
        requests.clear()
        code = fetch_runner.main(argv[:-2] + ["--require", "gated/encoder=model.safetensors", "--marker", str(marker)],
                                 download=hub)
        self.assertEqual(code, 1)
        self.assertEqual([request["repo_id"] for request in requests], ["gated/encoder"])
        result = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual((result["exit_status"], result["error"]), ("failed", "403 gated"))

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
