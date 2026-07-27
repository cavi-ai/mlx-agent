import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mlx_agent.cli import main


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "scout_responses.json"
REPO = "mlx-community/Qwen3-8B-Instruct-4bit"


class ContextFitCliTests(unittest.TestCase):
    def setUp(self):
        os.environ["MLX_AGENT_FIXTURE"] = str(FIXTURE)
        self.addCleanup(lambda: os.environ.pop("MLX_AGENT_FIXTURE", None))

    def _discover(self, extra):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["discover", "--json"] + extra)
        return code, json.loads(buffer.getvalue())

    def _candidate(self, payload):
        for records in payload["data"]["roles"].values():
            for record in records:
                if record["repo"] == REPO:
                    return record
        return None

    def test_kv_block_present_with_architecture(self):
        code, payload = self._discover([])
        self.assertEqual(code, 0)
        candidate = self._candidate(payload)
        self.assertIsNotNone(candidate)
        kv = candidate["estimates"].get("kv")
        self.assertIsNotNone(kv)
        self.assertEqual(kv["src"], "huggingface_model_metadata")
        self.assertGreater(kv["max_context_tokens"], 0)
        self.assertNotIn("kv_gb", kv)

    def test_context_flag_adds_kv_gb(self):
        code, payload = self._discover(["--context", "32768"])
        self.assertEqual(code, 0)
        candidate = self._candidate(payload)
        kv = candidate["estimates"]["kv"]
        self.assertEqual(kv["context_tokens"], 32768)
        self.assertAlmostEqual(kv["kv_gb"], 4.8, places=1)

    def test_huge_context_flips_fit(self):
        code, payload = self._discover(["--context", "1048576"])
        self.assertEqual(code, 0)
        candidate = self._candidate(payload)
        self.assertFalse(candidate["fits"])

    def test_default_fit_unchanged(self):
        code, payload = self._discover([])
        self.assertEqual(code, 0)
        candidate = self._candidate(payload)
        self.assertTrue(candidate["fits"])

    def test_out_of_range_context_is_rejected(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["discover", "--context", "8", "--json"])
        self.assertEqual(code, 2)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["error"]["code"], "invalid_arguments")


if __name__ == "__main__":
    unittest.main()
