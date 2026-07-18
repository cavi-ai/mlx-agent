"""Documentation and compatibility claims must match the shipped provider contract."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "compatibility" / "providers.json"
RELEASE_EVIDENCE_PATH = ROOT / "compatibility" / "release-evidence.json"
README_PATH = ROOT / "README.md"
RENDERER_PATH = ROOT / "scripts" / "render_compatibility.py"
APPLE_SILICON_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "apple-silicon.yml"
PROVIDERS = ("claude", "codex", "gemini", "opencode", "agentskills")
NATIVE_PROVIDERS = ("claude", "codex", "gemini", "opencode")
COMMANDS = ("mlx-scout", "mlx-adopt", "mlx-wire")
EVIDENCE_FIELDS = ("schema", "install_round_trip", "native_discovery", "bundle_execution", "model_backed_invocation")


class DocumentationContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MATRIX_PATH.is_file(), "compatibility/providers.json must exist")
        self.matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
        self.readme = README_PATH.read_text(encoding="utf-8")

    def test_matrix_records_complete_evidence_for_every_provider(self):
        self.assertEqual("0.2.0", self.matrix["plugin_version"])
        self.assertEqual(set(PROVIDERS), set(self.matrix["providers"]))
        self.assertEqual(
            {"supported", "not-run", "blocked", "static", "fixture"},
            set(self.matrix["allowed_evidence_statuses"]),
        )
        for provider_id, entry in self.matrix["providers"].items():
            with self.subTest(provider=provider_id):
                self.assertEqual(provider_id, entry["id"])
                self.assertIsInstance(entry["minimum_version"], str)
                self.assertIsInstance(entry["last_tested_version"], str)
                self.assertEqual(["user", "project"], entry["scopes"])
                self.assertTrue(entry["config_paths"])
                self.assertEqual(set(COMMANDS), {item["command"] for item in entry["capabilities"].values()})
                self.assertEqual({"scout", "adopt", "wire"}, set(entry["capabilities"]))
                self.assertEqual({"status", "date", "summary"}, set(entry["last_smoke_test"]))
                self.assertEqual(
                    {"id", "status", "date", "environment", "cli_version", "scopes_tested", "native_discovery", "fixture_bundle", "uninstall"},
                    set(entry["release_evidence"]),
                )
                self.assertTrue(entry["release_evidence"]["id"])
                self.assertIn(entry["release_evidence"]["status"], self.matrix["allowed_evidence_statuses"])
                date.fromisoformat(entry["release_evidence"]["date"])
                self.assertTrue(entry["release_evidence"]["environment"])
                self.assertIsInstance(entry["release_evidence"]["scopes_tested"], list)
                self.assertIn(entry["last_smoke_test"]["status"], self.matrix["allowed_evidence_statuses"])
                date.fromisoformat(entry["last_smoke_test"]["date"])
                self.assertTrue(entry["last_smoke_test"]["summary"])
                self.assertEqual(set(EVIDENCE_FIELDS), set(entry["evidence"]))
                for evidence in EVIDENCE_FIELDS:
                    self.assertIn(entry["evidence"][evidence]["status"], self.matrix["allowed_evidence_statuses"])

    def test_release_evidence_is_redacted_and_covers_every_matrix_provider(self):
        self.assertTrue(RELEASE_EVIDENCE_PATH.is_file())
        evidence = json.loads(RELEASE_EVIDENCE_PATH.read_text(encoding="utf-8"))
        self.assertEqual("1.0", evidence["schema_version"])
        self.assertEqual(self.matrix["plugin_version"], evidence["plugin_version"])
        self.assertEqual(self.matrix["allowed_evidence_statuses"], evidence["allowed_evidence_statuses"])
        self.assertEqual(set(PROVIDERS), set(evidence["providers"]))
        for provider_id, matrix_entry in self.matrix["providers"].items():
            with self.subTest(provider=provider_id):
                self.assertEqual(matrix_entry["release_evidence"], evidence["providers"][provider_id])
        self.assertTrue(evidence["commands"])
        commands = {item["id"]: item for item in evidence["commands"]}
        self.assertEqual(0, commands["diff-check"]["exit_status"])
        self.assertEqual(0, commands["git-status"]["exit_status"])
        self.assertEqual("", commands["git-status"]["output"])
        self.assertTrue(commands["git-status"]["clean"])
        self.assertIn("before_sha256", evidence["wire_transaction"])
        self.assertIn("after_sha256", evidence["wire_transaction"])
        self.assertTrue(evidence["wire_transaction"]["receipt_secret_scan"]["passed"])
        serialized = json.dumps(evidence).lower()
        self.assertNotIn("auth.json", serialized)
        self.assertNotIn("secret-value", serialized)

    def test_native_provider_docs_cover_install_lifecycle_and_exact_invocation(self):
        for provider_id in NATIVE_PROVIDERS:
            with self.subTest(provider=provider_id):
                entry = self.matrix["providers"][provider_id]
                document = ROOT / entry["documentation"]
                self.assertTrue(document.is_file(), "native provider documentation is required")
                text = document.read_text(encoding="utf-8")
                for command in ("install", "update", "uninstall", "doctor"):
                    self.assertIn("mlx-agent {0}".format(command), text)
                for capability in entry["capabilities"].values():
                    self.assertIn(capability["invocation"], text)
                self.assertIn("--dry-run", text)
                self.assertIn("--preview-hash", text)

    def test_universal_docs_cover_workflows_and_safety_boundaries(self):
        for path in (
            "docs/install/index.md",
            "docs/guides/scout.md",
            "docs/guides/adopt.md",
            "docs/guides/wire.md",
            "docs/security.md",
            "docs/adding-a-provider.md",
            "docs/migrating-from-v0.1.md",
        ):
            self.assertTrue((ROOT / path).is_file(), path)
        for name in COMMANDS:
            self.assertIn(name, self.readme)
        self.assertIn("compatibility/providers.json", self.readme)
        self.assertIn("docs/install/index.md", self.readme)

    def test_public_tree_excludes_internal_agent_work_artifacts(self):
        artifact_root = ROOT / "docs" / "superpowers"
        self.assertFalse(any(path.is_file() for path in artifact_root.rglob("*")))

    def test_pull_requests_use_the_hosted_apple_silicon_runner(self):
        workflow = APPLE_SILICON_WORKFLOW_PATH.read_text(encoding="utf-8")
        fixture_job, release_job = workflow.split("  release-live-runtime-health:", 1)
        self.assertIn("runs-on: macos-15", fixture_job)
        self.assertIn('test "$(uname -m)" = "arm64"', fixture_job)
        self.assertNotIn("self-hosted", fixture_job)
        self.assertIn("runs-on: [self-hosted, macOS, ARM64, apple-silicon]", release_job)

    def test_readme_claims_only_providers_in_compatibility_matrix(self):
        provider_docs = {
            path.stem for path in (ROOT / "docs" / "install").glob("*.md")
            if path.name != "index.md"
        }
        self.assertEqual(set(NATIVE_PROVIDERS), provider_docs)
        for entry in self.matrix["providers"].values():
            if entry["id"] != "agentskills":
                self.assertIn(entry["display_name"], self.readme)
        documented_versions = set(re.findall(r"\b\d+\.\d+\.\d+\b", self.readme))
        documented_versions.update(
            version
            for path in (ROOT / "docs" / "install").glob("*.md")
            for version in re.findall(r"\b\d+\.\d+\.\d+\b", path.read_text(encoding="utf-8"))
        )
        matrix_versions = {
            entry["last_tested_version"] for entry in self.matrix["providers"].values()
            if re.fullmatch(r"\d+\.\d+\.\d+", entry["last_tested_version"])
        }
        matrix_versions.add(self.matrix["plugin_version"])
        self.assertTrue(documented_versions.issubset(matrix_versions))

    def test_compatibility_block_is_rendered_from_the_matrix_and_current(self):
        self.assertTrue(RENDERER_PATH.is_file(), "compatibility renderer must exist")
        result = subprocess.run(
            ["python3", str(RENDERER_PATH), "--check"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("<!-- compatibility:begin -->", self.readme)
        self.assertIn("<!-- compatibility:end -->", self.readme)
        for entry in self.matrix["providers"].values():
            self.assertIn(entry["display_name"], self.readme)
            self.assertIn(", ".join(entry["scopes"]), self.readme)
            for path in entry["config_paths"]:
                self.assertIn(path, self.readme)
            for capability in entry["capabilities"].values():
                self.assertIn(capability["invocation"], self.readme)
            self.assertIn(entry["last_smoke_test"]["status"], self.readme)
            self.assertIn(entry["last_smoke_test"]["date"], self.readme)
            self.assertIn(entry["last_smoke_test"]["summary"], self.readme)
            for field in EVIDENCE_FIELDS:
                status = entry["evidence"][field]["status"]
                self.assertIn(status, self.readme)
                if status in {"blocked", "not-run"}:
                    self.assertIn("{0} — not supported".format(status), self.readme)

    def test_compatibility_renderer_detects_readme_drift(self):
        self.assertTrue(RENDERER_PATH.is_file(), "compatibility renderer must exist")
        with tempfile.TemporaryDirectory() as directory:
            readme = Path(directory) / "README.md"
            shutil.copyfile(README_PATH, readme)
            readme.write_text(readme.read_text(encoding="utf-8").replace("<!-- compatibility:end -->", "<!-- compatibility:drift -->"), encoding="utf-8")
            result = subprocess.run(
                ["python3", str(RENDERER_PATH), "--check", "--readme", str(readme)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("compatibility block", result.stderr)

    def test_compatibility_renderer_rejects_release_evidence_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            release_evidence = Path(directory) / "release-evidence.json"
            evidence = json.loads(RELEASE_EVIDENCE_PATH.read_text(encoding="utf-8"))
            current = evidence["providers"]["gemini"]["status"]
            evidence["providers"]["gemini"]["status"] = "verified" if current != "verified" else "fixture"
            release_evidence.write_text(json.dumps(evidence), encoding="utf-8")
            result = subprocess.run(
                ["python3", str(RENDERER_PATH), "--check", "--release-evidence", str(release_evidence)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("release evidence", result.stderr)

    def test_wire_docs_do_not_claim_model_download_or_pull_behavior(self):
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (README_PATH, ROOT / "docs" / "guides" / "wire.md", ROOT / "docs" / "security.md")
        )
        self.assertNotIn("Pull a chosen model", text)
        self.assertIn("does not pull, install, or download model weights", text)

    def test_release_checklist_records_the_authorized_runtime_hardening_scope(self):
        checklist = (ROOT / "docs" / "release-checklist.md").read_text(encoding="utf-8")
        security = (ROOT / "docs" / "security.md").read_text(encoding="utf-8")
        self.assertIn("retargeted by user authorization", checklist)
        self.assertIn("runtime/installer hardening", checklist)
        self.assertIn("legacy-lock migration", checklist)
        self.assertIn("physical parent identity", checklist)
        self.assertIn("migrated parent", checklist)
        self.assertIn("all older mlx-agent processes stopped", security)
        self.assertIn("older binary is unsupported", security)
        self.assertIn("legacy_lock_recreated", security)
        self.assertIn("OpenCode/Bun native smoke unavailable", checklist)


if __name__ == "__main__":
    unittest.main()
