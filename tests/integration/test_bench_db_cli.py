import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import mlx_agent.bench
from mlx_agent.cli import main


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "scout_responses.json"


class CommunityBenchDiscoveryTests(unittest.TestCase):
    def setUp(self):
        os.environ["MLX_AGENT_FIXTURE"] = str(FIXTURE)
        self.addCleanup(lambda: os.environ.pop("MLX_AGENT_FIXTURE", None))
        self._previous = mlx_agent.bench._community_bench_cache
        mlx_agent.bench._community_bench_cache = None
        self.addCleanup(self._restore)

    def _restore(self):
        mlx_agent.bench._community_bench_cache = self._previous

    def _discover(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["discover", "--json"])
        return code, json.loads(buffer.getvalue())

    def test_matching_chip_annotates_candidate(self):
        mlx_agent.bench._community_bench_cache = {
            ("mlx-community/Qwen3-8B-Instruct-4bit", "Apple M4 Max"): {
                "decode_toks": 55.0, "prefill_toks": 1200.0,
                "ttft_ms": 300.0, "samples": 4,
            }
        }
        code, payload = self._discover()
        self.assertEqual(code, 0)
        candidate = None
        for records in payload["data"]["roles"].values():
            for record in records:
                if record["repo"] == "mlx-community/Qwen3-8B-Instruct-4bit":
                    candidate = record
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["community_bench"]["decode_toks"], 55.0)
        self.assertEqual(candidate["community_bench"]["chip"], "Apple M4 Max")

    def test_no_match_leaves_candidate_unannotated(self):
        mlx_agent.bench._community_bench_cache = {}
        code, payload = self._discover()
        self.assertEqual(code, 0)
        for records in payload["data"]["roles"].values():
            for record in records:
                self.assertNotIn("community_bench", record)


if __name__ == "__main__":
    unittest.main()
