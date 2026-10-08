import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.serve import ServeError, load_recipes, plan_start, resolve_serve_runtime
from .backend_fixtures import synthetic_manifests, synthetic_registries


class ServeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.manifests = synthetic_manifests()
        # A specialized loader is declared by its backend, never remote code.
        self.manifests["mlx-vlm"]["registry"]["vision_language"]["model_types"].append("prism_hadamard_qwen35")
        self.registries = synthetic_registries(self.manifests, ("mlx-lm", "mlx-vlm"))

    def resolve(self, runtime="auto"):
        return resolve_serve_runtime(None, str(self.model), runtime, load_recipes(),
                                     manifests=self.manifests, registries=self.registries,
                                     backend_root=self.root / "backends")

    def config(self, value):
        (self.model / "config.json").write_text(json.dumps(value))

    def test_prism_uses_declared_isolated_backend_without_executing_repo_code(self):
        self.config({"model_type": "prism_hadamard_qwen35", "requires_runtime": "runtime/artifact.py"})
        runtime, executable = self.resolve()
        self.assertEqual(runtime, "mlx-vlm")
        self.assertEqual(executable, str(self.root / "backends/mlx-vlm/bin/mlx_vlm.server"))

    def test_text_model_keeps_builtin_and_vision_uses_vlm(self):
        self.config({"model_type": "llama"})
        self.assertEqual(self.resolve()[0], "mlx_lm")
        self.config({"model_type": "llama", "vision_config": {}})
        self.assertEqual(self.resolve()[0], "mlx-vlm")

    def test_auto_refuses_missing_and_unknown_configs(self):
        for value in (None, {"model_type": "unimplemented"}, {"model_type": "whisper"}):
            if value is not None:
                self.config(value)
            with self.assertRaises(ServeError):
                self.resolve()

    def test_explicit_runtime_remains_available_but_rejects_known_mismatch(self):
        self.assertEqual(self.resolve("mlx_lm")[0], "mlx_lm")
        self.config({"model_type": "prism_hadamard_qwen35"})
        with self.assertRaises(ServeError) as caught:
            self.resolve("mlx_lm")
        self.assertEqual(caught.exception.code, "unsupported_runtime")

    def test_explicit_runtime_handles_non_object_config_without_a_traceback(self):
        self.config(["invalid config document"])
        self.assertEqual(self.resolve("mlx_lm")[0], "mlx_lm")
        with self.assertRaises(ServeError) as caught:
            self.resolve()
        self.assertEqual(caught.exception.code, "model_config_unavailable")

    def test_missing_backend_never_falls_back_to_an_unmanaged_executable(self):
        self.config({"model_type": "prism_hadamard_qwen35"})
        self.registries["mlx-vlm"]["installed"] = False
        with self.assertRaises(ServeError) as caught:
            self.resolve()
        self.assertEqual(caught.exception.code, "runtime_not_installed")

    def test_resolved_executable_is_bound_in_eager_and_jit_preview(self):
        self.config({"model_type": "prism_hadamard_qwen35"})
        (self.model / "model.safetensors").write_bytes(b"weights")
        runtime, executable = self.resolve()
        for jit in (False, True):
            plan = plan_start(None, runtime, load_recipes(), path=str(self.model), jit=jit,
                              runtime_executable=executable, receipts_dir=self.root)
            self.assertEqual(plan["worker_argv" if jit else "argv"][0], executable)
            argv = plan["worker_argv" if jit else "argv"]
            self.assertEqual(argv.count("--host"), 1)
            self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
            other = plan_start(None, runtime, load_recipes(), path=str(self.model), jit=jit,
                               runtime_executable=executable + ".changed", receipts_dir=self.root)
            self.assertNotEqual(plan["preview_hash"], other["preview_hash"])
