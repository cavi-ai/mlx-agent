"""Soft-blend scoring in adoption compare phase."""

import tempfile
import unittest
from pathlib import Path

from mlx_agent.adoption import (
    EVIDENCE_SCORES,
    AdoptionRequest,
    AdoptionWorkflow,
)
from mlx_agent.contracts import ResultEnvelope
from mlx_agent.host import HostInventory
from mlx_agent.verification import (
    EvidenceStrength,
    VerificationEvidence,
    VerificationStatus,
)


class FakeHF:
    def __init__(self, cards=None):
        self.cards = cards or {}
        self.calls = []

    def fetch_model_card(self, repo, timeout=8):
        self.calls.append(repo)
        return self.cards.get(repo)


class FakeDiscovery:
    def __init__(self, candidates, cards=None):
        self.host = HostInventory(ram_gb=32, chip="Test Apple", ollama=True)
        self.candidates = candidates
        self._huggingface = FakeHF(cards)

    def discover(self, request):
        del request
        roles = {}
        for item in self.candidates:
            roles.setdefault(item["role"], []).append(dict(item))
        return ResultEnvelope.ok(
            "discover",
            {"host": self.host.to_dict(), "fast": False, "roles": roles},
        )


class StubVerifier:
    def __init__(self, strength=EvidenceStrength.RUNTIME_TESTED):
        self.strength = strength

    def clear_inventory_cache(self):
        return None

    def verify(self, item, host, allow_network=True):
        del host, allow_network
        return VerificationEvidence(
            repo=item["repo"],
            role=item["role"],
            strength=self.strength,
            status=VerificationStatus.VERIFIED
            if self.strength == EvidenceStrength.RUNTIME_TESTED
            else VerificationStatus.METADATA_ONLY,
            available_locally=True,
            loads=True,
            reasoning_confirmed=False,
            runtime="ollama",
            note="ok",
            details={},
        )


def _candidate(repo, role="coding", downloads=10, license_name="mit", trusted=True, rank_score=99):
    return {
        "repo": repo,
        "role": role,
        "roles": [role],
        "downloads": downloads,
        "likes": 1,
        "license": license_name,
        "est_ram_gb": 4.0,
        "fits": True,
        "trusted": trusted,
        "rank_score": rank_score,
        "wiring": "mlx_lm.server --model {0}".format(repo),
        "tags": [],
    }


class AdoptionScoringBlendTests(unittest.TestCase):
    def _run_to_compare(self, candidates, cards=None, request_kwargs=None):
        request_kwargs = request_kwargs or {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscovery(candidates, cards=cards),
                verifier=StubVerifier(),
                state_path=path,
            )
            roles = tuple(dict.fromkeys(item["role"] for item in candidates))
            state = workflow.start(AdoptionRequest(
                roles=roles,
                state_path=path,
                shortlist_limit=4,
                allow_network=False,
                **request_kwargs,
            ))
            while "compare" not in state.completed_phases:
                state = workflow.advance(state)
            return state

    def test_compare_uses_scoring_core_not_rank_score_or_trusted(self):
        # Same evidence strength; high rank_score/trusted on weak card candidate
        # must not beat a keyword-matching candidate with low rank_score.
        weak = _candidate("local/weak", downloads=1, rank_score=999, trusted=True)
        strong = _candidate("local/strong", downloads=1, rank_score=1, trusted=False)
        state = self._run_to_compare(
            [weak, strong],
            cards={
                "local/weak": "A generic model.",
                "local/strong": "Excellent OCR model with usage examples.",
            },
            request_kwargs={"keywords": ("ocr",), "domain": "legal"},
        )
        by_repo = {item["repo"]: item for item in state.comparisons}
        self.assertIn("scoring", by_repo["local/strong"])
        self.assertIn("signals", by_repo["local/strong"]["scoring"])
        self.assertIn("provenance", by_repo["local/strong"]["scoring"])
        self.assertGreater(by_repo["local/strong"]["score"], by_repo["local/weak"]["score"])
        # Soft blend: evidence base still present.
        base = EVIDENCE_SCORES[EvidenceStrength.RUNTIME_TESTED.value]
        self.assertGreaterEqual(by_repo["local/strong"]["score"], base)
        self.assertLessEqual(by_repo["local/strong"]["score"], base + 100)

    def test_evidence_strength_outranks_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            high_meta = _candidate("local/meta-high", downloads=1000000)
            low_runtime = _candidate("local/runtime-low", downloads=1)

            class SplitVerifier:
                def clear_inventory_cache(self):
                    return None

                def verify(self, item, host, allow_network=True):
                    del host, allow_network
                    if item["repo"] == "local/meta-high":
                        strength = EvidenceStrength.METADATA_ONLY
                        status = VerificationStatus.METADATA_ONLY
                    else:
                        strength = EvidenceStrength.RUNTIME_TESTED
                        status = VerificationStatus.VERIFIED
                    return VerificationEvidence(
                        repo=item["repo"],
                        role=item["role"],
                        strength=strength,
                        status=status,
                        available_locally=True,
                        loads=True,
                        reasoning_confirmed=False,
                        runtime="ollama",
                        note="ok",
                        details={},
                    )

            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscovery([high_meta, low_runtime]),
                verifier=SplitVerifier(),
                state_path=path,
            )
            state = workflow.start(AdoptionRequest(
                roles=("coding",),
                state_path=path,
                allow_network=False,
            ))
            while "compare" not in state.completed_phases:
                state = workflow.advance(state)
            by_repo = {item["repo"]: item for item in state.comparisons}
            self.assertGreater(
                by_repo["local/runtime-low"]["score"],
                by_repo["local/meta-high"]["score"],
            )


if __name__ == "__main__":
    unittest.main()
