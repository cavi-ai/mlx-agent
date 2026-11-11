import json
import os
import signal
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from jsonschema import Draft202012Validator, FormatChecker

from mlx_agent.serve import (ServeError, load_recipes, plan_start, resolve_serve_runtime, start_serve,
                             status_serve, stop_serve)
from mlx_agent.splash_runtime import SplashError, read_target
from .test_gguf import write_gguf

SCHEMA = Path(__file__).resolve().parents[2] / "schemas/serve-receipt.schema.json"
SHARD = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{0}-of-00002.gguf"


class SplashServeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.receipts = str(self.root / "receipts")
        self.launcher = self.install("Splash 1.6.0")

    def install(self, version_line):
        launcher = self.root / "bin/splash"
        launcher.parent.mkdir(exist_ok=True)
        launcher.write_text("#!/bin/sh\nprintf '%s\\n' '{0}'\n".format(version_line))
        launcher.chmod(0o755)
        environment = patch.dict(os.environ, {"MLX_AGENT_SPLASH": str(launcher)})
        environment.start()
        self.addCleanup(environment.stop)
        return str(launcher)

    def strata_profile(self):
        profile = self.root / "Qwen3.8-Flash-Next-Q2_0"
        (profile / "Q2_0").mkdir(parents=True)
        (profile / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "serving_runtime": "strata"}))
        (profile / "strata-model.json").write_text(json.dumps(
            {"schema_version": 1, "gguf": "Q2_0/" + SHARD.format(1), "mtp": None}))
        for part in (1, 2):
            write_gguf(profile / "Q2_0" / SHARD.format(part), architecture="qwen4exp")
        (profile / "tokenizer").mkdir()
        for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja"):
            (profile / "tokenizer" / name).write_text("{}")
        return profile

    def draft(self):
        draft = self.root / "draft"
        draft.mkdir()
        (draft / "config.json").write_text(json.dumps({"dflash_config": {"block_size": 16}}))
        (draft / "model.safetensors").write_bytes(b"draft weights")
        return draft

    def plan(self, path, **options):
        return plan_start(None, "splash", load_recipes(), path=str(path), receipts_dir=self.receipts, **options)

    def start(self, plan, spawn=None):
        return start_serve(plan, receipts_dir=self.receipts, confirm=True, preview_hash=plan["preview_hash"],
                           which=lambda executable: executable, port_free=lambda port: True,
                           spawn=spawn or (lambda argv, log_path: 4242), readiness=lambda url, deadline: True)

    def code(self, call):
        with self.assertRaises(ServeError) as error:
            call()
        return error.exception.code

    def test_strata_profile_serves_its_gguf_through_the_reviewed_launcher(self):
        profile = self.strata_profile()
        self.assertEqual(resolve_serve_runtime(None, str(profile), "splash", load_recipes()), ("splash", self.launcher))
        plan = self.plan(profile, splash_max_memory_gb=64, splash_max_context_tokens=32768, thinking="off")
        self.assertEqual(plan["argv"], [
            self.launcher, "serve", "--model", "local/Qwen3.8-Flash-Next-Q2_0",
            "--model-dir", str(profile / "Q2_0" / SHARD.format(1)), "--offline", "--host", "127.0.0.1",
            "--port", "8086", "--max-memory", "64000000000", "--max-context", "32768",
            "--default-reasoning-effort", "none"])
        self.assertEqual((plan["splash_source"], plan["splash_version"], plan["weight_format"], plan["thinking"]),
                         ("strata_profile", "1.6.0", "gguf", "off"))
        self.assertEqual(plan["readiness"], "http://127.0.0.1:8086/ready")
        other = self.plan(profile, splash_max_memory_gb=48, splash_max_context_tokens=32768, thinking="off")
        self.assertNotEqual(plan["preview_hash"], other["preview_hash"])

    def test_split_gguf_requires_every_part_and_names_the_model(self):
        first = write_gguf(self.root / "gguf" / SHARD.format(1), architecture="qwen4exp")
        self.assertEqual(self.code(lambda: self.plan(first)), "model_not_local")
        write_gguf(self.root / "gguf" / SHARD.format(2), architecture="qwen4exp")
        plan = self.plan(first)
        self.assertEqual(plan["splash_target"], str(first))
        self.assertEqual(plan["splash_served_model"], "local/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0")
        self.assertIsNone(plan["splash_source"])

    def test_served_name_follows_the_splash_model_rule(self):
        target = write_gguf(self.root / "my model--v2..final.gguf")
        self.assertEqual(self.plan(target)["splash_served_model"], "local/my-model-v2.final")

    def test_quantized_mlx_directory_is_served_and_unquantized_is_refused(self):
        model = self.root / "mlx"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"weights")
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        self.assertEqual(self.code(lambda: self.plan(model)), "splash_model_invalid")
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "quantization": {"bits": 4}}))
        plan = self.plan(model)
        self.assertEqual((plan["weight_format"], plan["splash_target"]), ("mlx", str(model)))

    def test_cached_mlx_repository_is_served_under_its_repository_id(self):
        repo, hf = "mlx-community/Qwen3.8-27B-4bit", str(self.root / "hf")
        self.assertEqual(self.code(lambda: plan_start(repo, "splash", load_recipes(), hf_cache=hf)), "model_not_local")
        entry = self.root / "hf/models--mlx-community--Qwen3.8-27B-4bit"
        snapshot = entry / "snapshots/abc123"
        snapshot.mkdir(parents=True)
        (entry / "refs").mkdir()
        (entry / "refs/main").write_text("abc123")
        (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64}}))
        (snapshot / "model.safetensors").write_bytes(b"weights")
        self.assertEqual(resolve_serve_runtime(repo, None, "splash", load_recipes(), hf_cache=hf), ("splash", self.launcher))
        plan = plan_start(repo, "splash", load_recipes(), hf_cache=hf, receipts_dir=self.receipts)
        self.assertEqual(plan["argv"][2:6], ["--model", repo, "--model-dir", str(snapshot)])
        receipt = self.start(plan)["receipt"]
        validator = Draft202012Validator(json.loads(SCHEMA.read_text()), format_checker=FormatChecker())
        self.assertEqual(list(validator.iter_errors(receipt)), [])
        self.assertEqual((receipt["repo"], receipt["splash_served_model"]), (repo, repo))
        command = "python -u -P -m server.server prepared --model {0} --port 8086".format(repo)
        self.assertTrue(status_serve(self.receipts, pid_alive=lambda pid: True, pid_command=lambda pid: command)[0]["argv_match"])
        (snapshot / "model.safetensors").write_bytes(b"other weights")
        self.assertEqual(self.code(lambda: self.start(plan)), "preview_stale")
        with self.assertRaises(SplashError) as error:
            read_target(snapshot, served_model="pub/name--copy")
        self.assertEqual(error.exception.code, "invalid_repo")

    def test_launcher_must_report_a_release_with_split_gguf_support(self):
        target = write_gguf(self.root / "model.gguf")
        self.install("Splash 1.3.9")
        self.assertEqual(self.code(lambda: self.plan(target)), "splash_version_unsupported")
        self.install("Splash (source checkout)")
        self.assertEqual(self.code(lambda: self.plan(target)), "splash_not_installed")
        with patch.dict(os.environ, {"MLX_AGENT_SPLASH": str(self.root / "missing")}):
            self.assertEqual(self.code(lambda: self.plan(target)), "splash_not_installed")

    def test_unsupported_selections_are_refused(self):
        target = write_gguf(self.root / "model.gguf")
        adapter = self.root / "adapter"
        adapter.mkdir()
        self.assertEqual(self.code(lambda: self.plan(target, jit=True)), "invalid_arguments")
        self.assertEqual(self.code(lambda: self.plan(target, adapter_path=str(adapter))), "unsupported_runtime")
        self.assertEqual(self.code(lambda: plan_start("pub/model", "mlx_lm", load_recipes(),
                                                      splash_max_context_tokens=8192)), "invalid_arguments")
        for options in ({"splash_max_memory_gb": 0.5}, {"splash_max_memory_gb": True},
                        {"splash_max_context_tokens": 1_010_001}, {"splash_max_context_tokens": 511}):
            with self.subTest(options=options):
                self.assertEqual(self.code(lambda: self.plan(target, **options)), "invalid_arguments")

    def test_draft_is_validated_and_bound_to_the_preview(self):
        target = write_gguf(self.root / "model.gguf")
        draft = self.draft()
        plan = self.plan(target, splash_draft_model=str(draft))
        self.assertEqual(plan["argv"][plan["argv"].index("--draft-model") + 1], str(draft))
        (draft / "model.safetensors").write_bytes(b"other draft weights")
        spawn = Mock()
        self.assertEqual(self.code(lambda: self.start(plan, spawn)), "preview_stale")
        spawn.assert_not_called()
        (draft / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        self.assertEqual(self.code(lambda: self.plan(target, splash_draft_model=str(draft))), "splash_draft_invalid")

    def test_confirm_refuses_target_or_launcher_changed_after_preview(self):
        profile = self.strata_profile()
        plan = self.plan(profile)
        (profile / "Q2_0" / SHARD.format(2)).write_bytes(b"changed")
        spawn = Mock()
        self.assertEqual(self.code(lambda: self.start(plan, spawn)), "preview_stale")
        target = write_gguf(self.root / "model.gguf")
        plan = self.plan(target)
        Path(self.launcher).write_text("#!/bin/sh\nprintf '%s\\n' 'Splash 1.6.1'\n")
        self.assertEqual(self.code(lambda: self.start(plan, spawn)), "preview_stale")
        spawn.assert_not_called()

    def test_receipt_owns_the_server_the_launcher_execs(self):
        profile = self.strata_profile()
        plan = self.plan(profile)
        receipt = self.start(plan)["receipt"]
        validator = Draft202012Validator(json.loads(SCHEMA.read_text()), format_checker=FormatChecker())
        self.assertEqual(list(validator.iter_errors(receipt)), [])
        self.assertEqual((receipt["runtime"], receipt["splash_served_model"], receipt["splash_version"]),
                         ("splash", "local/Qwen3.8-Flash-Next-Q2_0", "1.6.0"))
        server = ("/opt/homebrew/Cellar/splash/1.6.0/libexec/runtime/bin/python -u -P -m server.server "
                  "/Users/me/Library/Application Support/Splash/models/prepared --model {0} "
                  "--binary /opt/homebrew/Cellar/splash/1.6.0/libexec/engine/splash --port {1} --host 127.0.0.1")
        own = server.format("local/Qwen3.8-Flash-Next-Q2_0", 8086)
        foreign = server.format("local/other", 8086)
        status = lambda command: status_serve(self.receipts, pid_alive=lambda pid: True,
                                              pid_command=lambda pid: command)[0]["argv_match"]
        self.assertEqual((status(own), status(foreign)), (True, False))
        signals = []
        self.assertEqual(self.code(lambda: stop_serve(8086, self.receipts, pid_alive=lambda pid: True,
                                                      pid_command=lambda pid: foreign,
                                                      terminate=lambda pid, sig: signals.append(sig))),
                         "pid_argv_mismatch")
        self.assertEqual(signals, [])

    def test_stop_waits_for_the_engine_grace_period_before_killing(self):
        plan = self.plan(write_gguf(self.root / "model.gguf"))
        self.start(plan)
        now, signals = [0.0], []
        command = "python -u -P -m server.server prepared --model local/model --port 8086"

        def sleep(seconds):
            now[0] += seconds

        result = stop_serve(8086, self.receipts, pid_alive=lambda pid: now[0] < 12, pid_command=lambda pid: command,
                            terminate=lambda pid, sig: signals.append(sig), clock=lambda: now[0], sleep=sleep)
        self.assertEqual((result["status"], result.get("forced"), signals), ("stopped", None, [signal.SIGTERM]))
        self.assertFalse((Path(self.receipts) / "8086.json").exists())


if __name__ == "__main__":
    unittest.main()
