import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.contracts import ResultEnvelope
from mlx_agent.interview import DomainIntent
from mlx_agent.research import (
    _CARD_EXCERPT_CHARS,
    ResearchPack,
    candidate_metadata,
    generate_pack,
    render_pack,
    slugify,
    write_pack,
)


def _candidate(repo, role, downloads=100, license="mit", est_ram=4.0):
    return {
        "repo": repo,
        "role": role,
        "roles": [role],
        "downloads": downloads,
        "likes": 3,
        "est_ram_gb": est_ram,
        "license": license,
        "wiring": "`mlx_lm.server --model {0}`".format(repo),
        "facts": {"repository": repo, "downloads": downloads, "likes": 3},
    }


class FakeDiscovery:
    def __init__(self, buckets, error_roles=()):
        self._buckets = buckets
        self._error_roles = set(error_roles)
        self.requests = []

    def discover(self, request):
        self.requests.append(request)
        if request.role in self._error_roles:
            return ResultEnvelope.fail(
                "discover", "network_unavailable", "boom", "retry", retryable=True
            )
        data = {
            "host": {"chip": "M3", "ram_gb": 32, "ollama": True, "lmstudio": True},
            "fast": False,
            "roles": {request.role: self._buckets.get(request.role, [])},
        }
        return ResultEnvelope.ok("discover", data)


class FakeHF:
    def __init__(self, cards):
        self._cards = cards

    def fetch_model_card(self, repo, timeout=8):
        return self._cards.get(repo)


class SlugifyTests(unittest.TestCase):
    def test_slugify_normalizes(self):
        self.assertEqual(slugify("Legal Contract Review!"), "legal-contract-review")
        self.assertEqual(slugify("   "), "domain")


class CandidateMetadataTests(unittest.TestCase):
    def test_maps_discovery_candidate(self):
        metadata = candidate_metadata(_candidate("a/x", "vision"))
        self.assertEqual(metadata["roles"], ["vision"])
        self.assertEqual(metadata["license"], "mit")
        self.assertEqual(metadata["downloads"], 100)
        self.assertEqual(metadata["est_ram_gb"], 4.0)


class GeneratePackTests(unittest.TestCase):
    def _intent(self):
        return DomainIntent(domain="legal", roles=("vision",), keywords=("ocr",))

    def test_ranks_and_limits(self):
        buckets = {
            "vision": [
                _candidate("low/model", "vision", downloads=1),
                _candidate("high/model", "vision", downloads=1000000),
            ]
        }
        cards = {
            "low/model": "A model.",
            "high/model": "This model does OCR. Usage examples included.",
        }
        pack = generate_pack(
            self._intent(),
            FakeDiscovery(buckets),
            FakeHF(cards),
            limit=1,
            now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        )
        self.assertIsInstance(pack, ResearchPack)
        self.assertEqual(len(pack.candidates), 1)
        self.assertEqual(pack.candidates[0].repo, "high/model")
        self.assertGreater(pack.candidates[0].score, 0.0)

    def test_discovery_error_becomes_warning(self):
        pack = generate_pack(
            self._intent(),
            FakeDiscovery({}, error_roles=("vision",)),
            FakeHF({}),
            now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        )
        self.assertEqual(pack.candidates, ())
        self.assertTrue(any(w["code"] == "network_unavailable" for w in pack.warnings))

    def test_one_request_per_role(self):
        intent = DomainIntent(domain="x", roles=("vision", "general"))
        discovery = FakeDiscovery({"vision": [], "general": []})
        generate_pack(intent, discovery, FakeHF({}),
                      now=datetime(2026, 7, 22, tzinfo=timezone.utc))
        self.assertEqual([r.role for r in discovery.requests], ["vision", "general"])


class RenderPackTests(unittest.TestCase):
    def test_stable_headings(self):
        buckets = {"vision": [_candidate("acme/ocr", "vision", downloads=500)]}
        pack = generate_pack(
            DomainIntent(domain="Legal", roles=("vision",), keywords=("ocr",)),
            FakeDiscovery(buckets),
            FakeHF({"acme/ocr": "OCR model. Usage."}),
            now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        )
        markdown = render_pack(pack)
        self.assertIn("# MLX Research Pack: Legal", markdown)
        self.assertIn("## Candidates", markdown)
        self.assertIn("## Next steps", markdown)
        self.assertIn("`acme/ocr`", markdown)
        self.assertIn("/100", markdown)

    def test_card_excerpt_is_bounded_and_secret_free(self):
        long_card = "OCR usage. " + ("secret-token-xyz " * 500)
        buckets = {"vision": [_candidate("acme/ocr", "vision", downloads=10)]}
        pack = generate_pack(
            DomainIntent(domain="Legal", roles=("vision",), keywords=("ocr",)),
            FakeDiscovery(buckets),
            FakeHF({"acme/ocr": long_card}),
            now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        )
        excerpt = pack.candidates[0].card_excerpt
        self.assertLessEqual(len(excerpt), _CARD_EXCERPT_CHARS)
        markdown = render_pack(pack)
        self.assertNotIn("apiKey", markdown)
        self.assertNotIn("api_key", markdown)


class WritePackTests(unittest.TestCase):
    def test_writes_inside_project_folder(self):
        with TemporaryDirectory() as root:
            path = write_pack(
                "# pack\n",
                DomainIntent(domain="Legal Review", roles=("vision",)),
                root=root,
                now=datetime(2026, 7, 22, 13, 30, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(str(path).startswith(str(Path(root).resolve())))
            self.assertEqual(path.parent.name, "mlx-research")
            self.assertTrue(path.name.startswith("legal-review-"))
            self.assertEqual(path.read_text(), "# pack\n")


if __name__ == "__main__":
    unittest.main()
