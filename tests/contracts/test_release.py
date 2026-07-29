"""The release pipeline's own contracts: version lockstep and consumer envelopes."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import mlx_agent

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import release_envelope  # noqa: E402
import release_notes  # noqa: E402
import validate_release_version  # noqa: E402


COMMIT = "0" * 40
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


class VersionLockstepTests(unittest.TestCase):
    def test_every_declared_version_agrees_with_the_runtime(self):
        declared = validate_release_version.collect()
        self.assertGreaterEqual(len(declared), 8)
        for name, value in declared.items():
            with self.subTest(source=name):
                self.assertEqual(mlx_agent.__version__, value)

    def test_validation_passes_for_the_matching_tag(self):
        version, errors = validate_release_version.validate("v{0}".format(mlx_agent.__version__))
        self.assertEqual([], errors)
        self.assertEqual(mlx_agent.__version__, version)

    def test_validation_rejects_a_mismatched_tag(self):
        _version, errors = validate_release_version.validate("v99.0.0")
        self.assertTrue(any("does not match" in item for item in errors))

    def test_the_shipping_documentation_artifact_exists(self):
        artifact = ROOT / "docs" / "mlx-agent" / "v{0}".format(mlx_agent.__version__)
        self.assertTrue((artifact / "manifest.json").is_file())


class ReleaseEnvelopeTests(unittest.TestCase):
    def test_manifest_matches_the_consumer_identity_contract(self):
        manifest = release_envelope.release_manifest("0.5.0", "v0.5.0", COMMIT)
        self.assertEqual(
            {"schemaVersion", "slug", "kind", "version", "tag", "repository", "commit"},
            set(manifest),
        )
        self.assertEqual(1, manifest["schemaVersion"])
        self.assertEqual("mlx-agent", manifest["slug"])
        self.assertEqual("product-docs", manifest["kind"])
        self.assertEqual("cavi-ai/mlx-agent", manifest["repository"])

    def test_dispatch_body_matches_the_consumer_envelope_contract(self):
        body = release_envelope.dispatch_payload(
            "0.5.0", "v0.5.0", COMMIT,
            "https://github.com/cavi-ai/mlx-agent/releases/download/v0.5.0/a.tar.gz",
            "a" * 64,
        )
        self.assertEqual({"event_type", "client_payload"}, set(body))
        self.assertEqual("cavi-oss-release", body["event_type"])
        envelope = body["client_payload"]
        self.assertEqual(
            {"schemaVersion", "slug", "kind", "version", "tag", "repository", "commit", "artifact"},
            set(envelope),
        )
        self.assertEqual({"url", "sha256", "format"}, set(envelope["artifact"]))
        self.assertEqual("tar.gz", envelope["artifact"]["format"])

    def test_manifest_and_dispatch_describe_the_same_release(self):
        manifest = release_envelope.release_manifest("0.5.0", "v0.5.0", COMMIT)
        envelope = release_envelope.dispatch_payload(
            "0.5.0", "v0.5.0", COMMIT,
            "https://github.com/cavi-ai/mlx-agent/releases/download/v0.5.0/a.tar.gz",
            "a" * 64,
        )["client_payload"]
        for key in manifest:
            self.assertEqual(manifest[key], envelope[key])

    def test_tag_must_match_version(self):
        with self.assertRaises(release_envelope.EnvelopeError):
            release_envelope.release_manifest("0.5.0", "v0.4.0", COMMIT)

    def test_commit_must_be_a_full_sha(self):
        with self.assertRaises(release_envelope.EnvelopeError):
            release_envelope.release_manifest("0.5.0", "v0.5.0", "abc123")

    def test_artifact_url_must_be_plain_https(self):
        for url in (
            "http://github.com/a/b/releases/download/v0.5.0/a.tar.gz",
            "https://github.com/a/b/releases/download/v0.5.0/a.tar.gz?token=x",
            "https://user@github.com/a/b/releases/download/v0.5.0/a.tar.gz",
        ):
            with self.subTest(url=url):
                with self.assertRaises(release_envelope.EnvelopeError):
                    release_envelope.dispatch_payload(
                        "0.5.0", "v0.5.0", COMMIT, url, "a" * 64
                    )

    def test_artifact_digest_must_be_lowercase_hex(self):
        with self.assertRaises(release_envelope.EnvelopeError):
            release_envelope.dispatch_payload(
                "0.5.0", "v0.5.0", COMMIT,
                "https://github.com/a/b/releases/download/v0.5.0/a.tar.gz",
                "A" * 64,
            )

    def test_cli_emits_valid_json_for_both_documents(self):
        for argv in (
            ["manifest", "--version", "0.5.0", "--tag", "v0.5.0", "--commit", COMMIT],
            [
                "dispatch", "--version", "0.5.0", "--tag", "v0.5.0", "--commit", COMMIT,
                "--url", "https://github.com/a/b/releases/download/v0.5.0/a.tar.gz",
                "--sha256", "a" * 64,
            ],
        ):
            with self.subTest(action=argv[0]):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "release_envelope.py")] + argv,
                    capture_output=True, text=True, cwd=str(ROOT),
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIsInstance(json.loads(result.stdout), dict)


class ReleaseNotesTests(unittest.TestCase):
    def test_the_shipping_version_has_a_changelog_section(self):
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        body = release_notes.notes_for(mlx_agent.__version__, text)
        self.assertIsNotNone(body)
        self.assertTrue(body.strip())

    def test_an_unreleased_version_has_no_section(self):
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIsNone(release_notes.notes_for("99.0.0", text))

    def test_a_dated_heading_matches_its_version(self):
        text = "# Changelog\n\n## 1.2.3 - 2026-01-01\n\n- did a thing\n"
        self.assertEqual("- did a thing", release_notes.notes_for("1.2.3", text))


class ReleaseWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_release_runs_on_tags_and_locks_the_version_first(self):
        self.assertIn('tags: ["v*"]', self.workflow)
        self.assertIn("scripts/validate_release_version.py --tag", self.workflow)

    def test_release_runs_every_committed_gate(self):
        for gate in (
            "scripts/validate_contracts.py",
            "scripts/validate_json_schemas.py",
            "scripts/generate_adapters.py --check",
            "scripts/render_compatibility.py --check",
            "scripts/build_docs.py --check",
            "unittest discover -s tests",
        ):
            with self.subTest(gate=gate):
                self.assertIn(gate, self.workflow)

    def test_release_publishes_the_artifact_and_notifies_consumers(self):
        self.assertIn("gh release create", self.workflow)
        self.assertIn("release_envelope.py manifest", self.workflow)
        self.assertIn("release_envelope.py dispatch", self.workflow)
        self.assertIn("CONSUMER_DISPATCH_TOKEN", self.workflow)
        self.assertIn("cavi-ai/cavi-home", self.workflow)

    def test_release_rebuilds_docs_with_the_exact_tag_commit_before_packaging(self):
        self.assertIn(
            'scripts/build_docs.py --commit "$COMMIT" --destination "$staging/docs/mlx-agent/v${VERSION}"',
            self.workflow,
        )

    def test_tagging_never_queues_an_unclaimable_job(self):
        apple = (ROOT / ".github" / "workflows" / "apple-silicon.yml").read_text(encoding="utf-8")
        self.assertIn("MLX_AGENT_LIVE_RUNTIME_HEALTH", apple)


if __name__ == "__main__":
    unittest.main()
