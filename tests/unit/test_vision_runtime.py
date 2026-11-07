import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator
from mlx_agent import vision_runtime
from mlx_agent.serve import ServeError, load_recipes, plan_start, start_serve, status_serve


class VisionRuntimeTests(unittest.TestCase):
    def environment(self, root):
        prefix = Path(root)/"backend"
        (prefix/"bin").mkdir(parents=True)
        python = prefix/"bin/python"
        python.write_text("")
        python.chmod(0o755)
        (prefix/"pyvenv.cfg").write_text("include-system-site-packages = false\n")
        entry = prefix/"bin/mlx_vlm.server"
        entry.write_text("#!"+str(python)+"\n")
        entry.chmod(0o755)
        site = prefix/"lib/python3.12/site-packages"
        site.mkdir(parents=True)
        for name, version in vision_runtime.VERSIONS.items():
            folder = site/(name.replace("-", "_")+"-"+version+".dist-info")
            folder.mkdir()
            (folder/"METADATA").write_text("Metadata-Version: 2.1\nName: "+name+"\nVersion: "+version+"\n")
        sources = {}
        for name in vision_runtime.UPSTREAM_SOURCES:
            path = site/"mlx_vlm"/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture "+name)
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return entry, python, sources

    def test_preview_reads_isolated_metadata_and_binds_source_without_running_python(self):
        with TemporaryDirectory() as root:
            entry, python, sources = self.environment(root)
            with patch.object(vision_runtime, "UPSTREAM_SOURCES", sources):
                profile = vision_runtime.identity(str(entry))
                self.assertEqual(profile["versions"], vision_runtime.VERSIONS)
                self.assertEqual(vision_runtime.interpreter(str(entry)), str(python))
                self.assertTrue(vision_runtime.available(profile))
                plan = plan_start("pub/model", "mlx-vlm", load_recipes(), runtime_executable=str(entry), vision_cache=True)
                self.assertEqual(plan["vision_profile"]["fingerprint"], profile["fingerprint"])
                self.assertEqual(plan["argv"][:3], [str(python), "-I", "-B"])
                (entry.parent.parent/"lib/python3.12/site-packages/mlx_vlm/server/generation.py").write_text("# changed")
                with self.assertRaises(ServeError) as caught:
                    start_serve(plan, confirm=True, preview_hash=plan["preview_hash"])
                self.assertEqual(caught.exception.code, "vision_profile_changed")

    def test_foreign_entrypoint_and_unknown_runtime_keep_native_arguments(self):
        with TemporaryDirectory() as root:
            entry, _, sources = self.environment(root)
            entry.write_text("#!/other/python\n")
            self.assertIsNone(vision_runtime.interpreter(str(entry)))
            plan = plan_start("pub/model", "mlx-vlm", load_recipes(), runtime_executable=str(entry), vision_cache=True)
            self.assertFalse(plan["vision_profile"]["adapter_enabled"])
            self.assertEqual(plan["argv"][0], str(entry))
            with self.assertRaises(ServeError):
                plan_start("pub/model", "mlx_lm", load_recipes(), vision_cache=True)

    def test_jit_receipt_and_status_retain_the_reviewed_profile(self):
        with TemporaryDirectory() as root:
            entry, python, sources = self.environment(root)
            model = Path(root)/"model"
            model.mkdir()
            (model/"config.json").write_text("{}")
            (model/"model.safetensors").write_bytes(b"fixture")
            with patch.object(vision_runtime, "UPSTREAM_SOURCES", sources):
                plan = plan_start(None, "mlx-vlm", load_recipes(), path=str(model), jit=True,
                    receipts_dir=root, runtime_executable=str(entry), vision_cache=True)
                self.assertEqual(plan["worker_argv"][:3], [str(python), "-I", "-B"])
                result = start_serve(plan, receipts_dir=root, confirm=True, preview_hash=plan["preview_hash"],
                    which=lambda _: "/python", port_free=lambda _: True, spawn=lambda *_: 4242, readiness=lambda *_: True)
                receipt = result["receipt"]
                self.assertEqual(receipt["vision_profile"], plan["vision_profile"])
                schema = json.loads((Path(__file__).resolve().parents[2]/"schemas/serve-receipt.schema.json").read_text())
                Draft202012Validator(schema).validate(receipt)
                rows = status_serve(root, pid_alive=lambda _: True)
                self.assertEqual(rows[0]["vision_profile"], plan["vision_profile"])

    def test_worker_rejects_stale_source_before_importing_gpu_dependencies(self):
        with self.assertRaises(SystemExit), patch.dict(sys.modules, {"mlx": None, "mlx_vlm": None}):
            vision_runtime.main(["--runtime-executable", "/missing/mlx_vlm.server", "--expected-vision-fingerprint", "stale"])

    def test_dependency_skew_leaves_the_native_entrypoint_unchanged(self):
        with TemporaryDirectory() as root:
            entry, _, sources = self.environment(root)
            metadata = next(entry.parent.parent.glob("lib/*/site-packages/mlx_vlm-*.dist-info/METADATA"))
            metadata.write_text("Name: mlx-vlm\nVersion: 0.7.5\n")
            with patch.object(vision_runtime, "UPSTREAM_SOURCES", sources):
                plan = plan_start("pub/model", "mlx-vlm", load_recipes(), runtime_executable=str(entry), vision_cache=True)
            self.assertFalse(plan["vision_profile"]["adapter_enabled"])
            self.assertEqual(plan["argv"][0], str(entry))

    def test_worker_restores_native_loader_and_preserves_arguments_on_error(self):
        with TemporaryDirectory() as root:
            entry, _, sources = self.environment(root)
            model, processor = object(), object()
            original = lambda *args, **kwargs: (model, processor)
            generation = SimpleNamespace(load=original)
            observed = []
            def run():
                observed.append(sys.argv[1:])
                self.assertEqual(generation.load("local", None), (model, processor))
                raise RuntimeError("server stopped")
            server = SimpleNamespace(main=run)
            modules = {"mlx_vlm": SimpleNamespace(server=server), "mlx_vlm.server": SimpleNamespace(generation=generation)}
            previous = sys.argv
            with patch.object(vision_runtime, "UPSTREAM_SOURCES", sources), patch.dict(sys.modules, modules), \
                    patch("mlx_agent._vision_cache.apply", return_value=True) as adapted, patch.object(sys, "platform", "darwin"):
                fingerprint = vision_runtime.identity(str(entry))["fingerprint"]
                with self.assertRaises(RuntimeError):
                    vision_runtime.main(["--runtime-executable", str(entry), "--expected-vision-fingerprint", fingerprint,
                                         "--model", "local", "--port", "8083"])
                adapted.assert_called_once_with(model)
            self.assertIs(generation.load, original)
            self.assertIs(sys.argv, previous)
            self.assertEqual(observed, [["--model", "local", "--port", "8083"]])
