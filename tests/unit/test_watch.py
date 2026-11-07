import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.watch import (
    WatchError,
    build_snapshot,
    collect_owned,
    diff_snapshots,
    read_baseline,
    snapshot_candidates,
    write_snapshot,
    validate_owned_repos,
)


DISCOVERY_DATA = {
    "roles": {
        "coding": [
            {
                "repo": "pub/model-a-4bit",
                "base": "pub/model-a",
                "license": "apache-2.0",
                "facts": {"weight_bytes": 1000, "gated": "public"},
            },
            {
                "repo": "pub/unrelated-4bit",
                "base": "pub/unrelated",
                "license": "mit",
                "facts": {"weight_bytes": 500, "gated": "public"},
            },
        ]
    }
}

OWNED = [{"id": "pub/model-a-8bit", "source": "hf-cache"}]


class OwnershipScopeTests(unittest.TestCase):
    def test_excluded_cache_is_never_located_or_scanned(self):
        evidence = {}
        with patch("mlx_agent.watch.default_hf_cache") as default, \
             patch("mlx_agent.watch.inspect_hf_cache") as inspect, \
             patch("mlx_agent.watch.collect_runtime_inventory", return_value=([], [])), \
             patch("mlx_agent.watch.scan_wired_configs", return_value=[]) as wired:
            owned = collect_owned([], include_hf_cache=False, wired_roots=[],
                                  owned_repos=["managed/model"], evidence=evidence)
        default.assert_not_called()
        inspect.assert_not_called()
        wired.assert_called_once_with([])
        self.assertEqual(owned, [{"id": "managed/model", "source": "managed"}])
        self.assertIsNone(evidence["scope"]["hf_cache"])
        self.assertEqual(evidence["scope"]["wired_roots"], [])

    def test_exact_roots_and_explicit_managed_ids_do_not_expand_search(self):
        with TemporaryDirectory() as directory:
            roots = [Path(directory) / "client.json", Path(directory) / "receipts"]
            evidence = {}
            with patch("mlx_agent.watch.collect_runtime_inventory", return_value=([], [])), \
                 patch("mlx_agent.watch.scan_wired_configs", return_value=[{"model": "wired/model"}]) as scan:
                owned = collect_owned([], include_hf_cache=False, wired_roots=roots,
                                      owned_repos=["managed/model", "MANAGED/MODEL"], evidence=evidence)
            scan.assert_called_once_with([Path(value) for value in sorted(str(root.resolve()) for root in roots)])
            self.assertEqual(owned, [{"id": "managed/model", "source": "managed"}, {"id": "wired/model", "source": "wired"}])
            self.assertEqual(evidence["scope"]["owned_repos"], ["managed/model"])
            self.assertIn("wired", evidence["ownership_checks"]["unchecked_sources"])

    def test_only_selected_hf_cache_is_scanned(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = {}
            with patch("mlx_agent.watch.default_hf_cache") as default, \
                 patch("mlx_agent.watch.inspect_hf_cache", return_value=[{"id": "pub/model"}]) as inspect, \
                 patch("mlx_agent.watch.collect_runtime_inventory", return_value=([], [])), \
                 patch("mlx_agent.watch.scan_wired_configs", return_value=[]):
                owned = collect_owned([], hf_cache=root, wired_roots=[], evidence=evidence)
            default.assert_not_called()
            inspect.assert_called_once_with(root)
            self.assertEqual(owned, [{"id": "pub/model", "source": "hf-cache"}])
            self.assertEqual(evidence["scope"]["hf_cache"], str(root.resolve()))
            self.assertIn("hf-cache", evidence["ownership_checks"]["checked_sources"])

    def test_cli_compatible_default_is_current_project_but_explicit_empty_stays_empty(self):
        with TemporaryDirectory() as directory, \
             patch("mlx_agent.watch.Path.cwd", return_value=Path(directory)), \
             patch("mlx_agent.watch.collect_runtime_inventory", return_value=([], [])), \
             patch("mlx_agent.watch.scan_wired_configs", return_value=[]) as scan:
            collect_owned([], include_hf_cache=False)
            scan.assert_called_once_with([Path(directory).resolve()])

    def test_malformed_explicit_repositories_are_rejected_before_scanning(self):
        for repo in ("", "org", "org/model/checkpoint", "https://huggingface.co/org/model", "/org/model", "org/..", " org/model", "org/model\n"):
            with self.subTest(repo=repo), patch("mlx_agent.watch.inspect_hf_cache") as inspect:
                with self.assertRaises(WatchError) as caught:
                    collect_owned([], owned_repos=[repo])
                self.assertEqual(caught.exception.code, "invalid_owned_repo")
                inspect.assert_not_called()
        self.assertEqual(validate_owned_repos(["org/model", "ORG/MODEL"]), ["org/model"])

    def test_malformed_runtime_inventory_cannot_prove_model_absence(self):
        def snapshot(response):
            evidence = {}
            runtime = SimpleNamespace(name="test-runtime", list_models=lambda: response)
            with patch("mlx_agent.watch.scan_wired_configs", return_value=[]):
                owned = collect_owned([runtime], include_hf_cache=False, wired_roots=[], evidence=evidence)
            return build_snapshot(owned, {}, **evidence)
        baseline = snapshot({"data": [{"id": "owner/model"}]})
        for response in ({}, {"error": "warming"}, {"data": None}, {"data": [{}]}, {"models": "bad"}):
            with self.subTest(response=response):
                current = snapshot(response)
                self.assertIn("test-runtime", current["ownership_checks"]["unchecked_sources"])
                self.assertEqual(diff_snapshots(baseline, current), [])
        empty = snapshot({"data": []})
        self.assertIn("test-runtime", empty["ownership_checks"]["checked_sources"])
        self.assertEqual([item["code"] for item in diff_snapshots(baseline, empty)], ["owned_missing"])

    def test_failed_runtime_is_not_absence_evidence(self):
        evidence = {}
        with patch("mlx_agent.watch.collect_runtime_inventory", return_value=([], ["ollama: offline"])), \
             patch("mlx_agent.watch.scan_wired_configs", return_value=[]):
            collect_owned([SimpleNamespace(name="ollama"), SimpleNamespace(name="mlx_lm")],
                          include_hf_cache=False, wired_roots=[], evidence=evidence)
        self.assertIn("ollama", evidence["ownership_checks"]["unchecked_sources"])
        self.assertNotIn("ollama", evidence["ownership_checks"]["checked_sources"])
        self.assertIn("mlx_lm", evidence["ownership_checks"]["checked_sources"])

    def test_unavailable_cache_is_not_absence_evidence(self):
        evidence = {}
        with TemporaryDirectory() as directory, \
             patch("mlx_agent.watch.inspect_hf_cache", side_effect=PermissionError("denied")), \
             patch("mlx_agent.watch.collect_runtime_inventory", return_value=([], [])), \
             patch("mlx_agent.watch.scan_wired_configs", return_value=[]):
            collect_owned([], hf_cache=directory, wired_roots=[], evidence=evidence)
        self.assertIn("hf-cache", evidence["ownership_checks"]["unchecked_sources"])
        self.assertNotIn("hf-cache", evidence["ownership_checks"]["checked_sources"])


class SnapshotTests(unittest.TestCase):
    def test_candidates_are_flattened(self):
        candidates = snapshot_candidates(DISCOVERY_DATA)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates["pub/model-a-4bit"]["weight_bytes"], 1000)
        self.assertEqual(candidates["pub/model-a-4bit"]["base"], "pub/model-a")

    def test_write_rotates_previous(self):
        with TemporaryDirectory() as directory:
            first = build_snapshot(OWNED, snapshot_candidates(DISCOVERY_DATA), now="t1")
            write_snapshot(directory, first)
            second = build_snapshot(OWNED, snapshot_candidates(DISCOVERY_DATA), now="t2")
            write_snapshot(directory, second)
            baseline = read_baseline(directory)
            self.assertEqual(baseline["created_at"], "t2")
            self.assertEqual(baseline["previous"]["created_at"], "t1")

    def test_scope_and_check_metadata_rotate_with_previous_snapshot(self):
        scope = {"version": 1, "wired_roots": ["/selected/client.json"], "hf_cache": None,
                 "owned_repos": ["managed/model"], "runtime_sources": []}
        checks = {"checked_sources": ["managed"], "unchecked_sources": ["wired"]}
        with TemporaryDirectory() as directory:
            first = build_snapshot(OWNED, {}, now="t1", scope=scope, ownership_checks=checks)
            write_snapshot(directory, first)
            write_snapshot(directory, build_snapshot([], {}, now="t2", scope=scope, ownership_checks=checks))
            saved = read_baseline(directory)
        self.assertEqual(saved["scope"], scope)
        self.assertEqual(saved["previous"]["scope"], scope)
        self.assertEqual(saved["previous"]["ownership_checks"], checks)

    def test_missing_baseline_is_classified(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(WatchError) as caught:
                read_baseline(directory)
            self.assertEqual(caught.exception.code, "missing_baseline")


class DiffTests(unittest.TestCase):
    def _baseline(self):
        return build_snapshot(
            OWNED, snapshot_candidates(DISCOVERY_DATA), now="t1"
        )

    def test_no_changes_is_quiet(self):
        current = build_snapshot(OWNED, snapshot_candidates(DISCOVERY_DATA), now="t2")
        self.assertEqual(diff_snapshots(self._baseline(), current), [])

    def test_unrelated_repos_are_ignored(self):
        data = {
            "roles": {
                "coding": [
                    {
                        "repo": "pub/brand-new-4bit",
                        "base": "pub/brand-new",
                        "license": "mit",
                        "facts": {"weight_bytes": 5, "gated": "public"},
                    }
                ]
            }
        }
        current = build_snapshot(OWNED, snapshot_candidates(data), now="t2")
        self.assertEqual(diff_snapshots(self._baseline(), current), [])

    def test_new_quant_of_owned_is_reported(self):
        data = {
            "roles": {
                "coding": [
                    {
                        "repo": "pub/model-a-6bit",
                        "base": "pub/model-a",
                        "license": "apache-2.0",
                        "facts": {"weight_bytes": 750, "gated": "public"},
                    }
                ]
            }
        }
        current = build_snapshot(OWNED, snapshot_candidates(data), now="t2")
        findings = diff_snapshots(self._baseline(), current)
        self.assertEqual([finding["code"] for finding in findings], ["new_quant_of_owned"])
        self.assertEqual(findings[0]["repo"], "pub/model-a-6bit")

    def test_weight_change_is_reported(self):
        data = {
            "roles": {
                "coding": [
                    {
                        "repo": "pub/model-a-4bit",
                        "base": "pub/model-a",
                        "license": "apache-2.0",
                        "facts": {"weight_bytes": 1200, "gated": "public"},
                    }
                ]
            }
        }
        current = build_snapshot(OWNED, snapshot_candidates(data), now="t2")
        codes = [finding["code"] for finding in diff_snapshots(self._baseline(), current)]
        self.assertIn("updated_tracked_repo", codes)

    def test_gated_flip_is_reported(self):
        data = {
            "roles": {
                "coding": [
                    {
                        "repo": "pub/model-a-4bit",
                        "base": "pub/model-a",
                        "license": "apache-2.0",
                        "facts": {"weight_bytes": 1000, "gated": "auto"},
                    }
                ]
            }
        }
        current = build_snapshot(OWNED, snapshot_candidates(data), now="t2")
        codes = [finding["code"] for finding in diff_snapshots(self._baseline(), current)]
        self.assertIn("gated_changed", codes)

    def test_unreported_access_is_not_a_gated_change(self):
        for gated in ("unknown", None):
            data = {
                "roles": {
                    "coding": [
                        {
                            "repo": "pub/model-a-4bit",
                            "base": "pub/model-a",
                            "license": "apache-2.0",
                            "facts": {"weight_bytes": 1000, "gated": gated},
                        }
                    ]
                }
            }
            current = build_snapshot(OWNED, snapshot_candidates(data), now="t2")
            codes = [finding["code"] for finding in diff_snapshots(self._baseline(), current)]
            self.assertNotIn("gated_changed", codes, gated)

    def test_scope_changes_suppress_only_absence_findings(self):
        scope = {"version": 1, "wired_roots": ["/selected/client.json"], "hf_cache": "/selected/cache",
                 "owned_repos": [], "runtime_sources": []}
        checks = {"checked_sources": ["hf-cache"], "unchecked_sources": []}
        baseline = build_snapshot(OWNED, snapshot_candidates(DISCOVERY_DATA), scope=scope, ownership_checks=checks)
        for changed in (dict(scope, hf_cache=None), dict(scope, wired_roots=["/other/client.json"]),
                        dict(scope, owned_repos=["managed/new-model"])):
            with self.subTest(scope=changed):
                current = build_snapshot([], {}, scope=changed, ownership_checks=checks)
                self.assertEqual(diff_snapshots(baseline, current), [])
        candidates = snapshot_candidates(DISCOVERY_DATA)
        candidates["pub/model-a-4bit"]["weight_bytes"] = 1200
        candidates["pub/model-a-6bit"] = dict(candidates["pub/model-a-4bit"])
        current = build_snapshot(OWNED, candidates, scope=dict(scope, hf_cache=None), ownership_checks=checks)
        codes = [item["code"] for item in diff_snapshots(baseline, current)]
        self.assertIn("updated_tracked_repo", codes)
        self.assertIn("new_quant_of_owned", codes)
        self.assertNotIn("owned_missing", codes)

    def test_missing_requires_matching_scope_and_checked_source(self):
        scope = {"version": 1, "wired_roots": [], "hf_cache": "/selected/cache", "owned_repos": [], "runtime_sources": []}
        checks = {"checked_sources": ["hf-cache"], "unchecked_sources": []}
        baseline = build_snapshot(OWNED, {}, scope=scope, ownership_checks=checks)
        current = build_snapshot([], {}, scope=scope, ownership_checks=checks)
        self.assertEqual([item["code"] for item in diff_snapshots(baseline, current)], ["owned_missing"])
        for unchecked in ({"checked_sources": [], "unchecked_sources": ["hf-cache"]},
                          {"checked_sources": ["hf-cache"], "unchecked_sources": ["hf-cache"]}):
            current = build_snapshot([], {}, scope=scope, ownership_checks=unchecked)
            self.assertEqual(diff_snapshots(baseline, current), [])
        self.assertEqual(diff_snapshots(self._baseline(), build_snapshot([], {}, scope=scope, ownership_checks=checks)), [])

    def test_legacy_baselines_without_scope_cannot_assert_absence(self):
        current = build_snapshot([], snapshot_candidates(DISCOVERY_DATA), now="t2")
        codes = [finding["code"] for finding in diff_snapshots(self._baseline(), current)]
        self.assertNotIn("owned_missing", codes)


if __name__ == "__main__":
    unittest.main()
