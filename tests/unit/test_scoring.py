import unittest

from mlx_agent.interview import DomainIntent
from mlx_agent.scoring import (
    SIGNAL_WEIGHTS,
    ScoreResult,
    Signal,
    rank_scored,
    score_candidate,
)


def _intent(**overrides):
    base = {"domain": "legal", "roles": ("vision",), "keywords": ("ocr",)}
    base.update(overrides)
    return DomainIntent(**base)


class ScoreCandidateTests(unittest.TestCase):
    def test_clean_weighted_score(self):
        intent = _intent()
        metadata = {
            "roles": ["vision"],
            "license": "apache-2.0",
            "downloads": 0,
            "likes": 0,
            "est_ram_gb": None,
            "tags": [],
        }
        result = score_candidate(intent, metadata, "This model does OCR.")
        self.assertIsInstance(result, ScoreResult)
        self.assertEqual(result.score, 75.0)

    def test_signals_carry_sources(self):
        intent = _intent()
        result = score_candidate(intent, {"roles": ["vision"], "downloads": 0}, "ocr")
        by_id = {signal.id: signal for signal in result.signals}
        self.assertIn("role_match", by_id)
        self.assertEqual(by_id["role_match"].source, "local_role_derivation")
        self.assertEqual(by_id["popularity"].source, "huggingface_model_list")
        self.assertEqual(by_id["keyword_match"].source, "card_text_and_tags")

    def test_inapplicable_license_signal_is_excluded(self):
        intent = _intent(license_allow=())
        result = score_candidate(intent, {"roles": ["vision"], "downloads": 0}, "ocr")
        ids = {signal.id for signal in result.signals if signal.applicable}
        self.assertNotIn("license_ok", ids)

    def test_license_restriction_penalizes_mismatch(self):
        intent = _intent(license_allow=("mit",))
        allowed = score_candidate(
            intent, {"roles": ["vision"], "license": "mit", "downloads": 0}, "ocr"
        )
        denied = score_candidate(
            intent, {"roles": ["vision"], "license": "gpl-3.0", "downloads": 0}, "ocr"
        )
        self.assertGreater(allowed.score, denied.score)

    def test_popularity_is_monotonic(self):
        intent = _intent(keywords=())
        low = score_candidate(intent, {"roles": ["vision"], "downloads": 10}, None)
        high = score_candidate(intent, {"roles": ["vision"], "downloads": 1000000}, None)
        self.assertGreater(high.score, low.score)

    def test_memory_fit_applies_only_with_budget_and_estimate(self):
        intent = _intent(memory_gb=16.0)
        fits = score_candidate(
            intent, {"roles": ["vision"], "downloads": 0, "est_ram_gb": 8.0}, "ocr"
        )
        too_big = score_candidate(
            intent, {"roles": ["vision"], "downloads": 0, "est_ram_gb": 40.0}, "ocr"
        )
        self.assertGreater(fits.score, too_big.score)
        no_estimate = score_candidate(
            intent, {"roles": ["vision"], "downloads": 0, "est_ram_gb": None}, "ocr"
        )
        ids = {signal.id for signal in no_estimate.signals if signal.applicable}
        self.assertNotIn("memory_fit", ids)

    def test_provenance_only_covers_applicable_signals(self):
        intent = _intent(license_allow=())
        result = score_candidate(intent, {"roles": ["vision"], "downloads": 0}, "ocr")
        provenance_fields = {
            field for record in result.provenance for field in record["fields"]
        }
        self.assertNotIn("license_ok", provenance_fields)
        self.assertIn("role_match", provenance_fields)

    def test_weights_are_fixed_and_documented(self):
        self.assertEqual(
            set(SIGNAL_WEIGHTS),
            {
                "role_match",
                "keyword_match",
                "popularity",
                "license_ok",
                "memory_fit",
                "card_quality",
            },
        )


class RankScoredTests(unittest.TestCase):
    def test_sorts_by_score_then_repo(self):
        intent = _intent(keywords=())
        a = ("b/model", score_candidate(intent, {"roles": ["vision"], "downloads": 5}, None))
        b = ("a/model", score_candidate(intent, {"roles": ["vision"], "downloads": 5}, None))
        c = ("c/model", score_candidate(intent, {"roles": ["vision"], "downloads": 999999}, None))
        ranked = rank_scored([a, b, c])
        self.assertEqual([repo for repo, _ in ranked], ["c/model", "a/model", "b/model"])


if __name__ == "__main__":
    unittest.main()
