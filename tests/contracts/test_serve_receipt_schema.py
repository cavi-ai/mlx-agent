import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator, FormatChecker

from mlx_agent.serve import load_recipes, plan_start, start_serve


ROOT = Path(__file__).resolve().parents[2]


class ServeReceiptSchemaTests(unittest.TestCase):
    def receipt(self):
        with TemporaryDirectory() as folder, \
                patch("mlx_agent.prefill_runtime.can_wrap_runtime", return_value=True), \
                patch("mlx_agent.prompt_lookup_runtime.can_wrap_runtime", return_value=True):
            model = Path(folder) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"local")
            plan = plan_start(None, "mlx_lm", load_recipes(), path=str(model),
                              jit=True, receipts_dir=folder, m5_prefill=True, prompt_lookup=True)
            result = start_serve(plan, receipts_dir=folder, confirm=True,
                                 preview_hash=plan["preview_hash"], which=lambda _: "/python",
                                 port_free=lambda _: True, spawn=lambda *_: 4242,
                                 readiness=lambda *_: True)
            return result["receipt"]

    def validator(self):
        schema = json.loads((ROOT / "schemas/serve-receipt.schema.json").read_text())
        return Draft202012Validator(schema, format_checker=FormatChecker())

    def test_accelerated_start_receipt_matches_the_published_contract(self):
        receipt = self.receipt()
        validator = self.validator()
        self.assertTrue(validator.is_valid(receipt),
                        [error.message for error in validator.iter_errors(receipt)])

    def test_known_profile_names_remain_supported_and_unknown_names_are_rejected(self):
        receipt, validator = self.receipt(), self.validator()
        for decode in ("prompt-lookup-v1", "prompt-lookup-v2", "prompt-lookup-v3"):
            for prefill in ("m5-q4-prefill-v1", "m5-q4-prefill-v2"):
                value = copy.deepcopy(receipt)
                value["decode_profile"]["name"] = decode
                value["prefill_kernel"]["name"] = prefill
                with self.subTest(decode=decode, prefill=prefill):
                    self.assertTrue(validator.is_valid(value))
        for field, unknown in (("decode_profile", "prompt-lookup-v4"),
                               ("prefill_kernel", "m5-q4-prefill-v3")):
            value = copy.deepcopy(receipt)
            value[field]["name"] = unknown
            self.assertFalse(validator.is_valid(value))
