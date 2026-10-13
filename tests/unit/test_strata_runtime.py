import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent import strata_runtime as strata
from .test_gguf import write_gguf


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

    def test_frontend_receives_only_controlled_offline_engine_arguments(self):
        self.valid_model()
        model = strata.read_model(self.root)
        runtime = {"python": "/installed/python", "root": "/installed"}
        config = strata.frontend_config(model, runtime, speculation=False)
        self.assertEqual(config["exe"], runtime["python"])
        self.assertEqual(config["host"], "127.0.0.1")
        self.assertEqual(config["model_name"], model["root"])
        self.assertEqual(config["args"][config["args"].index("--spec") + 1], "1")
        self.assertEqual(config["args"][config["args"].index("--suffix-draft") + 1], "off")


if __name__ == "__main__":
    unittest.main()
