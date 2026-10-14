import json
import os
import sys
import struct
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent import strata_runtime as strata
from .test_gguf import write_gguf
from mlx_agent.serve import ServeError, load_recipes, plan_start, resolve_serve_runtime
from mlx_agent.gguf import scan_gguf, scan_mlx_outputs
from mlx_agent import serve


class StrataProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "serving_runtime": "strata"}))

    def profile(self, **changes):
        value = {"schema_version": 1, "gguf": "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", "mtp": None}
        value.update(changes)
        (self.root / "strata-model.json").write_text(json.dumps(value))

    def test_existing_vq_pack_is_refused_without_changing_it(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"model_type": "qwen4_exp", "quantization": {"bits": 8}}))
        before = config.read_bytes()
        with self.assertRaises(strata.StrataError) as error:
            strata.read_model(self.root)
        self.assertEqual(error.exception.code, "strata_model_required")
        self.assertEqual(config.read_bytes(), before)

    def test_profile_cannot_supply_executable_or_escape_its_directory(self):
        for changes in ({"exe": "evil"}, {"gguf": "../model.gguf"}, {"gguf": "/model.gguf"}, {"schema_version": True}):
            with self.subTest(changes=changes):
                self.profile(**changes)
                with self.assertRaises(strata.StrataError):
                    strata.read_model(self.root)

    def test_incomplete_pair_is_refused(self):
        self.profile()
        first = self.root / "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
        first.parent.mkdir()
        first.write_bytes(b"GGUF")
        with self.assertRaises(strata.StrataError) as error:
            strata.read_model(self.root)
        self.assertEqual(error.exception.code, "strata_model_incomplete")

    def test_profile_refuses_symbolic_link_to_external_weights(self):
        self.profile()
        first = self.root / "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
        first.parent.mkdir()
        external = self.root / "external.gguf"
        external.write_bytes(b"GGUF")
        first.symlink_to(external)
        with self.assertRaises(strata.StrataError):
            strata.read_model(self.root)

    def test_uninstalled_runtime_has_an_actionable_error(self):
        with self.assertRaises(strata.StrataError) as error:
            strata.installed_runtime(self.root)
        self.assertEqual(error.exception.code, "strata_not_installed")

    def test_source_change_changes_runtime_fingerprint(self):
        for folder in ("engine/strata_mlx", "http/serve", "http/tools"):
            (self.root / folder).mkdir(parents=True)
        source = self.root / "engine/strata_mlx/serve.py"
        source.write_text("original")
        first = strata.source_digest(self.root)
        source.write_text("modified")
        self.assertNotEqual(first, strata.source_digest(self.root))

    def valid_model(self):
        self.profile()
        weights = self.root / "Q2_0"
        weights.mkdir()
        for part in (1, 2):
            write_gguf(weights / f"Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{part}-of-00002.gguf", architecture="qwen4exp")
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja"):
            (tokenizer / name).write_text("{}")

    def test_valid_local_pair_has_a_fingerprint_that_changes_with_assets(self):
        self.valid_model()
        model = strata.read_model(self.root)
        self.assertEqual(model["gguf"], str(self.root / "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"))
        (self.root / "tokenizer/vocab.json").write_text('{"a": 0}')
        self.assertNotEqual(model["fingerprint"], strata.read_model(self.root)["fingerprint"])

    def test_ngram_companion_may_omit_architecture_metadata(self):
        self.valid_model()
        companion = self.root / "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
        companion.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
        self.assertEqual(strata.read_model(self.root)["root"], str(self.root))

    def test_disk_ngram_table_does_not_inflate_resident_memory_admission(self):
        self.valid_model()
        before = strata.read_model(self.root)
        companion = self.root / "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
        with companion.open("ab") as handle:
            handle.write(b"x" * 65536)
        after = strata.read_model(self.root)
        self.assertGreater(after["weight_bytes"], before["weight_bytes"])
        self.assertEqual(after["memory_estimate_bytes"], before["memory_estimate_bytes"])
        self.assertNotEqual(after["fingerprint"], before["fingerprint"])

    def test_standalone_jit_memory_admission_retains_headroom_and_unknown_memory_guards(self):
        self.valid_model()
        from mlx_agent import jit_serve
        program = """
import runpy, sys
module = runpy.run_path(sys.argv[1])
worker = module['ModelWorker']({'local_path': sys.argv[2], 'runtime': 'strata',
                              'memory_policy': {'minimum_headroom_gb': 2}})
worker.memory_probe = lambda: 1 << 40
worker._check_memory()
assert worker.memory_check['required_available_bytes'] > 10_000_000_000
for available, code in ((100, 'insufficient_headroom'), (None, 'memory_unknown')):
    worker.memory_probe = lambda: available
    try:
        worker._check_memory()
    except module['GatewayError'] as error:
        assert error.code == code, error.code
    else:
        raise AssertionError('memory admission was bypassed')
"""
        result = subprocess.run([sys.executable, "-c", program, str(Path(jit_serve.__file__).resolve()), str(self.root)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frontend_receives_only_controlled_offline_engine_arguments(self):
        self.valid_model()
        model = strata.read_model(self.root)
        runtime = {"python": "/installed/python", "engine": "/installed/strata-mlx", "root": "/installed"}
        config = strata.frontend_config(model, runtime, speculation=False)
        self.assertEqual(config["exe"], runtime["engine"])
        self.assertEqual(config["args"][0], "--native")
        self.assertEqual(config["host"], "127.0.0.1")
        self.assertEqual(config["model_name"], model["root"])
        self.assertEqual(config["args"][config["args"].index("--spec") + 1], "1")
        self.assertEqual(config["args"][config["args"].index("--suffix-draft") + 1], "0")

    def installed(self):
        runtime = self.root / "runtime"
        for folder in ("engine/strata_mlx", "engine/.venv/bin", "http/serve", "http/tools"):
            (runtime / folder).mkdir(parents=True, exist_ok=True)
        for source in ("engine/strata_mlx/serve.py", "http/serve/server.py", "http/tools/strata_tokenizer.py"):
            (runtime / source).write_text("# pinned source")
        (runtime / "engine/.venv/bin/python").symlink_to(sys.executable)
        (runtime / "strata-mlx").write_text(strata.engine_launcher(runtime))
        (runtime / "strata-mlx").chmod(0o700)
        marker = {"engine_revision": strata.ENGINE_REVISION, "http_revision": strata.HTTP_REVISION,
                  "source_sha256": strata.source_digest(runtime)}
        (runtime / strata.INSTALL_MARKER).write_text(json.dumps(marker))
        self.environment = patch.dict(os.environ, {"MLX_AGENT_STRATA_ROOT": str(runtime)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        return runtime

    def test_auto_selects_strata_only_for_an_explicit_valid_local_profile(self):
        self.valid_model()
        runtime = self.installed()
        resolved, executable = resolve_serve_runtime(None, str(self.root), "auto", load_recipes())
        self.assertEqual(resolved, "strata")
        self.assertEqual(executable, str(runtime / "engine/.venv/bin/python"))

    def test_strata_plan_binds_model_runtime_and_speculation_to_preview(self):
        self.valid_model()
        self.installed()
        plan = plan_start(None, "strata", load_recipes(), path=str(self.root), strata_speculation=True)
        off = plan_start(None, "strata", load_recipes(), path=str(self.root), strata_speculation=False)
        default = plan_start(None, "strata", load_recipes(), path=str(self.root))
        self.assertFalse(default["strata_speculation"])
        self.assertEqual(default["preview_hash"], off["preview_hash"])
        self.assertNotEqual(plan["preview_hash"], off["preview_hash"])
        self.assertEqual(plan["runtime"], "strata")
        self.assertEqual(plan["weight_format"], "gguf")
        self.assertEqual(plan["weight_quantization"], "GSQ-RCO Q2_0")
        self.assertTrue(plan["weights_preserved"])
        self.assertIn("--expected-model-fingerprint", plan["argv"])
        self.assertIn("--expected-runtime-fingerprint", plan["argv"])
        self.assertEqual(plan["bind"], "127.0.0.1")
        self.assertEqual(plan["readiness_deadline"], 900)
        (self.root / "tokenizer/vocab.json").write_text('{"b": 1}')
        changed = plan_start(None, "strata", load_recipes(), path=str(self.root))
        self.assertNotEqual(plan["preview_hash"], changed["preview_hash"])

    def test_jit_preserves_strata_adapter_and_reviewed_assets(self):
        self.valid_model()
        self.installed()
        plan = plan_start(None, "strata", load_recipes(), path=str(self.root), jit=True)
        self.assertIn("strata_runtime.py", " ".join(plan["worker_argv"]))
        self.assertIn("--expected-model-fingerprint", plan["worker_argv"])
        self.assertEqual(plan["worker_argv"][plan["worker_argv"].index("--host") + 1], "127.0.0.1")

    def test_strata_refuses_repo_ids_and_lora_without_downloading(self):
        self.valid_model()
        self.installed()
        for arguments in ({"repo": "Qwen/Qwen3.8-Flash-Next"}, {"repo": None, "path": str(self.root), "adapter_path": "/adapter"}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ServeError):
                    plan_start(runtime="strata", recipes=load_recipes(), **arguments)

    def test_inventory_includes_a_valid_profile_and_excludes_a_changed_one(self):
        self.valid_model()
        self.assertEqual([item["runtime"] for item in scan_mlx_outputs([self.root])], ["strata"])
        (self.root / "tokenizer/vocab.json").unlink()
        self.assertEqual(scan_mlx_outputs([self.root]), [])

    def test_prepared_profile_shards_are_not_separate_conversion_candidates(self):
        self.valid_model()
        self.assertEqual(scan_gguf([self.root]), [])

    def test_exited_child_ends_readiness_wait_without_waiting_for_full_deadline(self):
        with patch.object(serve.os, "waitpid", return_value=(42, 256)), patch.object(serve.urllib.request, "build_opener") as opener:
            self.assertFalse(serve._default_readiness("http://127.0.0.1:8084/v1/models", 900, child_pid=42))
            opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
