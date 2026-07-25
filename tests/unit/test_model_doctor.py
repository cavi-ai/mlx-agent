import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.model_doctor import (
    adapter_references,
    check_endpoint,
    collect_runtime_inventory,
    inspect_hf_cache,
    run_model_doctor,
    scan_receipt_after_hashes,
    scan_wired_configs,
)
from mlx_agent.wiring import ConfigAdapter


def _hf_repo(root, name, blob_bytes=b"weights", dangling=False):
    repo = root / "models--{0}".format(name.replace("/", "--"))
    blob = repo / "blobs" / "ab" / "cd1234"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(blob_bytes)
    snapshot = repo / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    link = snapshot / "model.safetensors"
    if dangling:
        link.symlink_to(repo / "blobs" / "ab" / "missing")
    else:
        link.symlink_to(blob)
    refs = repo / "refs"
    refs.mkdir()
    (refs / "main").write_text("rev1")
    return repo


class FakeRuntime:
    def __init__(self, name, response=None, error=None):
        self.name = name
        self.response = response or {}
        self.error = error

    def list_models(self):
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.response


class HfCacheTests(unittest.TestCase):
    def test_inventory_summarizes_repos(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _hf_repo(root, "pub/model-a", blob_bytes=b"x" * 100)
            (root / "datasets--pub--data").mkdir()
            inventory = inspect_hf_cache(root)
            self.assertEqual(len(inventory), 1)
            item = inventory[0]
            self.assertEqual(item["id"], "pub/model-a")
            self.assertEqual(item["bytes"], 100)
            self.assertTrue(item["complete"])
            self.assertEqual(item["revisions"], ["rev1"])

    def test_dangling_blob_marks_incomplete(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _hf_repo(root, "pub/model-b", dangling=True)
            inventory = inspect_hf_cache(root)
            self.assertFalse(inventory[0]["complete"])

    def test_missing_cache_returns_empty(self):
        self.assertEqual(inspect_hf_cache("/tmp/definitely-not-a-cache"), [])


class RuntimeInventoryTests(unittest.TestCase):
    def test_ollama_sizes_are_collected(self):
        runtime = FakeRuntime("ollama", {
            "models": [{"name": "qwen3:32b", "size": 1234}]
        })
        inventory, errors = collect_runtime_inventory([runtime])
        self.assertEqual(errors, [])
        self.assertEqual(inventory[0]["id"], "qwen3:32b")
        self.assertEqual(inventory[0]["bytes"], 1234)

    def test_unreachable_runtime_is_classified(self):
        runtime = FakeRuntime("ollama", error="connection refused")
        inventory, errors = collect_runtime_inventory([runtime])
        self.assertEqual(inventory, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("connection refused", errors[0])


class WiredConfigScanTests(unittest.TestCase):
    def test_json_marker_config_is_found(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "opencode.json"
            config.write_text(json.dumps({
                "mlx_agent_wire": {
                    "marker": "MLX_AGENT_WIRE",
                    "version": "2.0",
                    "runtime": "mlx_lm",
                    "model": "pub/model",
                    "provider": {"base_url": "http://127.0.0.1:8080/v1"},
                }
            }))
            (root / "unrelated.json").write_text("{}")
            wired = scan_wired_configs([root])
            self.assertEqual(len(wired), 1)
            self.assertEqual(wired[0]["model"], "pub/model")
            self.assertEqual(wired[0]["endpoint"], "http://127.0.0.1:8080/v1")
            self.assertEqual(wired[0]["runtime"], "mlx_lm")

    def test_ollama_modelfile_is_found(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Modelfile").write_text("# MLX_AGENT_WIRE v1\nFROM pub/model\n")
            wired = scan_wired_configs([root])
            self.assertEqual(len(wired), 1)
            self.assertEqual(wired[0]["model"], "pub/model")
            self.assertIsNone(wired[0]["endpoint"])

    def test_litellm_yaml_is_found(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = ConfigAdapter("litellm")
            content = adapter.render("pub/model")
            (root / "router.yaml").write_text(content)
            wired = scan_wired_configs([root])
            self.assertEqual(len(wired), 1)
            self.assertEqual(wired[0]["model"], "pub/model")
            self.assertEqual(wired[0]["endpoint"], "http://127.0.0.1:8080/v1")

    def test_skipped_directories_are_not_walked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            hidden = root / "node_modules"
            hidden.mkdir()
            (hidden / "config.json").write_text(json.dumps({
                "mlx_agent_wire": {"model": "pub/hidden"}
            }))
            self.assertEqual(scan_wired_configs([root]), [])


class AdapterReferencesTests(unittest.TestCase):
    def test_ollama_reference(self):
        adapter = ConfigAdapter("ollama")
        reference = adapter_references(adapter, "# MLX_AGENT_WIRE v1\nFROM pub/model\n")
        self.assertEqual(reference, {"model": "pub/model", "endpoint": None})

    def test_unmarked_json_returns_none(self):
        adapter = ConfigAdapter("mlx_lm")
        self.assertIsNone(adapter_references(adapter, "{}"))


class ReceiptScanTests(unittest.TestCase):
    def test_after_hashes_are_read(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_dir = root / ".mlx-agent-receipts" / "abc"
            receipt_dir.mkdir(parents=True)
            (receipt_dir / "receipt.json").write_text(json.dumps({
                "transaction_id": "abc",
                "status": "applied",
                "after_hashes": {"/tmp/x.json": "deadbeef"},
            }))
            records = scan_receipt_after_hashes([root])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["after_hashes"], {"/tmp/x.json": "deadbeef"})


class EndpointCheckTests(unittest.TestCase):
    def test_closed_loopback_port_is_down(self):
        self.assertEqual(check_endpoint("http://127.0.0.1:1/v1"), "down")

    def test_invalid_endpoint_is_down(self):
        self.assertEqual(check_endpoint("http://example.com/v1"), "down")

    def test_empty_endpoint_is_none(self):
        self.assertIsNone(check_endpoint(None))


class RunModelDoctorTests(unittest.TestCase):
    def test_missing_model_finding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "hf"
            cache.mkdir()
            (root / "opencode.json").write_text(json.dumps({
                "mlx_agent_wire": {
                    "marker": "MLX_AGENT_WIRE",
                    "runtime": "mlx_lm",
                    "model": "pub/gone",
                    "provider": {"base_url": "http://127.0.0.1:8080/v1"},
                }
            }))
            report = run_model_doctor([root], cache, [])
            codes = [finding["code"] for finding in report["findings"]]
            self.assertIn("drift_missing_model", codes)
            self.assertEqual(report["summary"]["wired_configs"], 1)

    def test_hash_mismatch_finding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "wired.json"
            target.write_text("{}")
            receipt_dir = root / ".mlx-agent-receipts" / "abc"
            receipt_dir.mkdir(parents=True)
            (receipt_dir / "receipt.json").write_text(json.dumps({
                "transaction_id": "abc",
                "status": "applied",
                "after_hashes": {str(target): "0" * 64},
            }))
            report = run_model_doctor([root], root / "hf", [])
            codes = [finding["code"] for finding in report["findings"]]
            self.assertIn("drift_hash_mismatch", codes)

    def test_hash_match_is_quiet(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "wired.json"
            content = b"{}\n"
            target.write_bytes(content)
            receipt_dir = root / ".mlx-agent-receipts" / "abc"
            receipt_dir.mkdir(parents=True)
            (receipt_dir / "receipt.json").write_text(json.dumps({
                "transaction_id": "abc",
                "status": "applied",
                "after_hashes": {str(target): hashlib.sha256(content).hexdigest()},
            }))
            report = run_model_doctor([root], root / "hf", [])
            self.assertEqual(report["findings"], [])

    def test_endpoint_conflict_between_runtimes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _hf_repo((root / "hf"), "pub/model")
            for name, runtime in (("a.json", "mlx_lm"), ("b.json", "lmstudio")):
                (root / name).write_text(json.dumps({
                    "mlx_agent_wire": {
                        "marker": "MLX_AGENT_WIRE",
                        "runtime": runtime,
                        "model": "pub/model",
                        "provider": {"base_url": "http://127.0.0.1:8080/v1"},
                    }
                }))
            report = run_model_doctor([root], root / "hf", [])
            codes = [finding["code"] for finding in report["findings"]]
            self.assertIn("drift_endpoint_conflict", codes)

    def test_endpoints_are_checked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wired.json").write_text(json.dumps({
                "mlx_agent_wire": {
                    "marker": "MLX_AGENT_WIRE",
                    "runtime": "mlx_lm",
                    "model": "pub/model",
                    "provider": {"base_url": "http://127.0.0.1:1/v1"},
                }
            }))
            report = run_model_doctor([root], root / "hf", [])
            self.assertEqual(report["endpoints"][0]["status"], "down")


if __name__ == "__main__":
    unittest.main()
