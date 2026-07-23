"""Integration tests for adopt start --from-research."""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main
from mlx_agent.interview import DomainIntent
from mlx_agent.research import ResearchCandidate, ResearchPack, write_pack


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "scout_responses.json"


def _pack_with_candidates(order):
    candidates = []
    for index, repo in enumerate(order):
        record = {
            "repo": repo,
            "role": "coding",
            "roles": ["coding"],
            "downloads": 100 - index,
            "likes": 1,
            "license": "mit",
            "est_ram_gb": 4.0,
            "fits": True,
            "trusted": True,
            "wiring": "mlx_lm.server --model {0}".format(repo),
        }
        candidates.append(ResearchCandidate(
            repo=repo,
            role="coding",
            score=90.0 - index,
            wiring=record["wiring"],
            signals=(),
            provenance=(),
            card_present=False,
            card_excerpt="",
            record=record,
        ))
    return ResearchPack(
        intent=DomainIntent(
            domain="coding assistant",
            roles=("coding",),
            keywords=("python",),
            license_allow=("mit",),
            memory_gb=32.0,
            notes="from research",
        ),
        candidates=tuple(candidates),
        generated_at="2026-07-23T00:00:00+00:00",
    )


class AdoptFromResearchTests(unittest.TestCase):
    def setUp(self):
        os.environ["MLX_AGENT_FIXTURE"] = str(FIXTURE)
        self.addCleanup(lambda: os.environ.pop("MLX_AGENT_FIXTURE", None))

    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_from_research_seeds_shortlist_in_pack_order(self):
        order = ["pack/first", "pack/second", "pack/third"]
        pack = _pack_with_candidates(order)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            md_path = write_pack(
                "# pack\n",
                pack.intent,
                root=root,
                pack=pack,
            )
            json_path = md_path.with_suffix(".json")
            state_path = root / "adoption.json"
            code, output = self._run([
                "adopt", "start",
                "--state", str(state_path),
                "--from-research", str(json_path),
                "--shortlist-limit", "2",
                "--no-network",
                "--json",
            ])
            self.assertEqual(code, 0, output)
            payload = json.loads(output)
            self.assertEqual(payload["status"], "ok")
            state = payload["data"]["state"]
            shortlist_repos = [item["repo"] for item in state["shortlist"]]
            self.assertEqual(shortlist_repos, ["pack/first", "pack/second"])
            self.assertEqual(state["request"]["source"]["type"], "research_pack")
            self.assertEqual(state["request"]["domain"], "coding assistant")
            self.assertEqual(state["request"]["keywords"], ["python"])
            self.assertTrue(state["discovery"].get("seeded_from_research"))
            self.assertIn("scoring", state["comparisons"][0])

    def test_from_research_rejects_markdown(self):
        pack = _pack_with_candidates(["pack/only"])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            md_path = write_pack("# pack\n", pack.intent, root=root, pack=pack)
            state_path = root / "adoption.json"
            code, output = self._run([
                "adopt", "start",
                "--state", str(state_path),
                "--from-research", str(md_path),
                "--json",
            ])
            self.assertEqual(code, 2)
            self.assertIn("invalid_research_pack", output)


if __name__ == "__main__":
    unittest.main()
