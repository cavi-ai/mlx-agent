import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from mlx_agent.prompt_lookup_runtime import identity, main
from mlx_agent.serve import ServeError, load_recipes, plan_start, start_serve, status_serve


class PromptLookupRuntimeTests(unittest.TestCase):
    @patch("mlx_agent.prompt_lookup_runtime.can_wrap_runtime", return_value=True)
    @patch("mlx_agent.prefill_runtime.can_wrap_runtime", return_value=True)
    def test_preview_binds_both_options_and_rejects_changed_source(self, *_):
        native = plan_start("pub/model", "mlx_lm", load_recipes())
        plan = plan_start("pub/model", "mlx_lm", load_recipes(), prompt_lookup=True, m5_prefill=True)
        self.assertNotIn("decode_profile", native)
        self.assertEqual(plan["argv"].count("--expected-decode-fingerprint"), 1)
        self.assertIn(plan["prefill_kernel"]["fingerprint"], plan["argv"])
        self.assertIn(identity()["fingerprint"], plan["argv"])
        self.assertNotEqual(native["preview_hash"], plan["preview_hash"])
        with patch("mlx_agent.prompt_lookup_runtime.identity", return_value={}):
            with self.assertRaises(ServeError) as caught:
                start_serve(plan, confirm=True, preview_hash=plan["preview_hash"])
        self.assertEqual(caught.exception.code, "decode_profile_changed")

    def test_foreign_runtime_and_other_backends(self):
        with patch("mlx_agent.prompt_lookup_runtime.can_wrap_runtime", return_value=False):
            native = plan_start("pub/model", "mlx_lm", load_recipes(), runtime_executable="/other/mlx_lm.server")
            plan = plan_start("pub/model", "mlx_lm", load_recipes(), runtime_executable="/other/mlx_lm.server", prompt_lookup=True)
        self.assertEqual(native["argv"], plan["argv"])
        self.assertFalse(plan["decode_profile"]["adapter_enabled"])
        with self.assertRaises(ServeError):
            plan_start("pub/model", "mlx-vlm", load_recipes(), prompt_lookup=True)

    @patch("mlx_agent.prompt_lookup_runtime.can_wrap_runtime", return_value=True)
    def test_jit_preserves_worker_binding_receipt_and_status(self, _):
        with TemporaryDirectory() as folder:
            model = Path(folder) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"local")
            plan = plan_start(None, "mlx_lm", load_recipes(), path=str(model), jit=True,
                              receipts_dir=folder, prompt_lookup=True)
            self.assertIn(plan["decode_profile"]["fingerprint"], plan["worker_argv"])
            result = start_serve(plan, receipts_dir=folder, confirm=True, preview_hash=plan["preview_hash"],
                                 which=lambda _: "/python", port_free=lambda _: True,
                                 spawn=lambda *_: 4242, readiness=lambda *_: True)
            status = status_serve(folder, pid_alive=lambda _: False)[0]
            self.assertEqual(result["receipt"]["decode_profile"], plan["decode_profile"])
            self.assertEqual(status["decode_profile"], plan["decode_profile"])

    def test_dependency_fallback_preserves_arguments_and_server(self):
        observed = []
        generator = object()
        server = SimpleNamespace(ResponseGenerator=generator, main=lambda: observed.append(sys.argv[1:]))
        with patch.dict(sys.modules, {"mlx_lm": SimpleNamespace(server=server)}), \
                patch("mlx_agent.prompt_lookup_runtime.sys.platform", "linux"):
            main(["--expected-decode-fingerprint", identity()["fingerprint"], "--model", "local"])
        self.assertEqual(observed, [["--model", "local"]])
        self.assertIs(server.ResponseGenerator, generator)
