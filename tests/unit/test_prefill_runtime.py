import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from mlx_agent.prefill_runtime import can_wrap_runtime, identity, main, runtime_supported
from mlx_agent.serve import ServeError, load_recipes, plan_start, start_serve, status_serve


class PrefillRuntimeTests(unittest.TestCase):
    def test_compatibility_requires_tested_runtime_and_m5(self):
        versions = {"mlx": "0.31.1", "mlx-metal": "0.31.1", "mlx-lm": "0.29.1"}
        self.assertTrue(runtime_supported("darwin", "26.6.2", versions, "applegpu_g17s"))
        for system, os_version, deps, architecture in (
            ("linux", "26.6.2", versions, "applegpu_g17s"),
            ("darwin", "25.0", versions, "applegpu_g17s"),
            ("darwin", "26.6.2", dict(versions, mlx="0.32.4"), "applegpu_g17s"),
            ("darwin", "26.6.2", versions, "applegpu_g16s"),
        ):
            with self.subTest(system=system, os_version=os_version, deps=deps, architecture=architecture):
                self.assertFalse(runtime_supported(system, os_version, deps, architecture))

    @patch("mlx_agent.prefill_runtime.can_wrap_runtime", return_value=True)
    def test_opt_in_is_bound_to_preview_and_runtime_source(self, _same_runtime):
        recipes = load_recipes()
        stock = plan_start("pub/model", "mlx_lm", recipes)
        accelerated = plan_start("pub/model", "mlx_lm", recipes, m5_prefill=True)
        self.assertNotIn("prefill_kernel", stock)
        self.assertEqual(stock["argv"][0], "mlx_lm.server")
        self.assertEqual(accelerated["argv"][0], sys.executable)
        self.assertEqual(accelerated["prefill_kernel"]["fingerprint"], identity()["fingerprint"])
        self.assertTrue(accelerated["prefill_kernel"]["adapter_enabled"])
        self.assertIn(identity()["fingerprint"], accelerated["argv"])
        self.assertNotEqual(stock["preview_hash"], accelerated["preview_hash"])
        with patch("mlx_agent.prefill_runtime.identity", return_value={"fingerprint": "changed"}):
            with self.assertRaises(ServeError) as caught:
                start_serve(accelerated, confirm=True, preview_hash=accelerated["preview_hash"])
        self.assertEqual(caught.exception.code, "prefill_kernel_changed")

    def test_other_runtime_rejects_opt_in(self):
        with self.assertRaises(ServeError) as caught:
            plan_start("pub/model", "mlx-vlm", load_recipes(), m5_prefill=True)
        self.assertEqual(caught.exception.code, "invalid_arguments")

    @patch("mlx_agent.prefill_runtime.can_wrap_runtime", return_value=True)
    def test_jit_keeps_kernel_binding_in_worker_receipt_and_status(self, _same_runtime):
        with TemporaryDirectory() as folder:
            model = Path(folder) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"local")
            plan = plan_start(None, "mlx_lm", load_recipes(), path=str(model),
                              jit=True, receipts_dir=folder, m5_prefill=True)
            self.assertIn(plan["prefill_kernel"]["fingerprint"], plan["worker_argv"])
            result = start_serve(plan, receipts_dir=folder, confirm=True, preview_hash=plan["preview_hash"],
                                 which=lambda _: "/python", port_free=lambda _: True,
                                 spawn=lambda *_: 4242, readiness=lambda *_: True)
            status = status_serve(folder, pid_alive=lambda _: False)[0]
            for record in (result["receipt"], status):
                self.assertEqual(record["prefill_kernel"], plan["prefill_kernel"])

    def test_foreign_and_missing_executables_keep_the_native_environment(self):
        plans = []
        with patch("mlx_agent.prefill_runtime.can_wrap_runtime", return_value=False):
            for executable in ("/runtime-a/bin/mlx_lm.server", "/runtime-b/bin/mlx_lm.server", "/missing/mlx_lm.server"):
                native = plan_start("pub/model", "mlx_lm", load_recipes(), runtime_executable=executable)
                requested = plan_start("pub/model", "mlx_lm", load_recipes(), runtime_executable=executable, m5_prefill=True)
                self.assertEqual(requested["argv"], native["argv"])
                self.assertFalse(requested["prefill_kernel"]["adapter_enabled"])
                plans.append(requested)
        self.assertEqual(len({p["preview_hash"] for p in plans}), 3)

    def test_interpreter_check_does_not_adopt_another_environment(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            python = root / "bin/python"
            python.parent.mkdir()
            entry = python.with_name("mlx_lm.server")
            entry.write_text("#!" + str(python) + "\n")
            foreign = root / "other/mlx_lm.server"
            foreign.parent.mkdir()
            foreign.write_text("#!" + str(python) + "\n")
            with patch("mlx_agent.prefill_runtime.sys.executable", str(python)):
                with patch("mlx_agent.prefill_runtime.shutil.which", return_value=str(entry)):
                    self.assertTrue(can_wrap_runtime("mlx_lm.server"))
                    entry.write_text("#!/other/python\n")
                    self.assertFalse(can_wrap_runtime("mlx_lm.server"))
                for path in (str(foreign), None):
                    with patch("mlx_agent.prefill_runtime.shutil.which", return_value=path):
                        self.assertFalse(can_wrap_runtime("mlx_lm.server"))

    def test_unsupported_runtime_delegates_unchanged_and_restores_server(self):
        observed = []
        original_load = object()
        server = SimpleNamespace(load=original_load, main=lambda: observed.append(list(sys.argv[1:])))
        core = SimpleNamespace(device_info=lambda: {"architecture": "unsupported"})
        modules = {"mlx": SimpleNamespace(core=core), "mlx.core": core,
                   "mlx_lm": SimpleNamespace(server=server)}
        with patch.dict(sys.modules, modules):
            main(["--expected-kernel-fingerprint", identity()["fingerprint"], "--model", "local", "--port", "8080"])
        self.assertEqual(observed, [["--model", "local", "--port", "8080"]])
        self.assertIs(server.load, original_load)


if __name__ == "__main__":
    unittest.main()
