import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.watch import (
    WatchError,
    build_snapshot,
    diff_snapshots,
    read_baseline,
    snapshot_candidates,
    write_snapshot,
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

    def test_owned_missing_is_reported(self):
        current = build_snapshot([], snapshot_candidates(DISCOVERY_DATA), now="t2")
        codes = [finding["code"] for finding in diff_snapshots(self._baseline(), current)]
        self.assertIn("owned_missing", codes)


if __name__ == "__main__":
    unittest.main()
