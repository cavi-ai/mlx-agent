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
    def __init__(self, cards, adapters=None, datasets=None, dataset_cards=None):
        self._cards = cards
        self._adapters = adapters if adapters is not None else []
        self._datasets = datasets if datasets is not None else []
        self._dataset_cards = dataset_cards or {}

    def fetch_model_card(self, repo, timeout=8):
        return self._cards.get(repo)

    def fetch_dataset_card(self, repo, timeout=8):
        return self._dataset_cards.get(repo)

    def list_adapters(self, search="", limit_fetch=20, timeout=10):
        return list(self._adapters)[:limit_fetch]

    def list_datasets(self, search="", limit_fetch=20, timeout=10):
        return list(self._datasets)[:limit_fetch]


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

    def test_role_fallback_when_roles_absent(self):
        metadata = candidate_metadata({"repo": "a/x", "role": "coding"})
        self.assertEqual(metadata["roles"], ["coding"])


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

    def test_repo_in_multiple_roles_is_deduped(self):
        shared = _candidate("shared/model", "vision")
        buckets = {"vision": [shared], "general": [dict(shared, role="general")]}
        intent = DomainIntent(domain="x", roles=("vision", "general"), keywords=())
        pack = generate_pack(intent, FakeDiscovery(buckets), FakeHF({}),
                             now=datetime(2026, 7, 22, tzinfo=timezone.utc))
        repos = [c.repo for c in pack.candidates]
        self.assertEqual(repos.count("shared/model"), 1)


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
        self.assertIn("## Modality foundations", markdown)
        self.assertIn("## Next steps", markdown)
        self.assertIn("`acme/ocr`", markdown)
        self.assertIn("/100", markdown)

    def test_card_excerpt_is_bounded(self):
        sentinel = "BEYOND-CAP-SENTINEL"
        long_card = "OCR usage. " + ("filler " * 500) + sentinel
        buckets = {"vision": [_candidate("acme/ocr", "vision", downloads=10)]}
        pack = generate_pack(
            DomainIntent(domain="Legal", roles=("vision",), keywords=("ocr",)),
            FakeDiscovery(buckets),
            FakeHF({"acme/ocr": long_card}),
            now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        )
        excerpt = pack.candidates[0].card_excerpt
        self.assertLessEqual(len(excerpt), _CARD_EXCERPT_CHARS)
        # Content beyond the cap is not copied into the artifact.
        self.assertNotIn(sentinel, excerpt)

    def test_empty_pack_renders_no_candidates_message(self):
        pack = generate_pack(
            DomainIntent(domain="Empty", roles=("vision",)),
            FakeDiscovery({"vision": []}),
            FakeHF({}),
            now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        )
        markdown = render_pack(pack)
        self.assertIn("No candidates were found", markdown)


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

    def test_hostile_domain_stays_contained(self):
        with TemporaryDirectory() as root:
            path = write_pack(
                "# pack\n",
                DomainIntent(domain="../../etc/passwd", roles=("vision",)),
                root=root,
                now=datetime(2026, 7, 22, 13, 30, 0, tzinfo=timezone.utc),
            )
            self.assertEqual(path.parent, Path(root).resolve() / "mlx-research")
            self.assertNotIn("..", path.name)


if __name__ == "__main__":
    unittest.main()
