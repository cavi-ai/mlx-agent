import json
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlx_agent.installer import Installer, InstallerConflictError
from mlx_agent.providers import ProviderRegistry
from mlx_agent.cli import main
from mlx_agent.transactions import Transaction, rollback


ROOT = Path(__file__).resolve().parents[2]


class InstallerRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.config = self.root / "config"
        self.project = self.root / "project"
        self.project.mkdir()
        self.registry = ProviderRegistry(
            ROOT / "plugin.json", home=self.home, config_root=self.config
        )
        self.installer = Installer(self.registry, project_root=self.project)

    def tearDown(self):
        self.temporary.cleanup()

    def test_multi_provider_install_is_confirmed_and_doctor_reports_clean_install(self):
        plan = self.installer.plan("install", ["claude", "codex"], "user", self.project)
        self.assertTrue(plan.preview["requires_confirmation"])
        with self.assertRaises(PermissionError):
            self.installer.execute(plan, confirmed=False)

        receipt = self.installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertEqual("applied", receipt.status)
        self.assertTrue(receipt.receipt_path)
        doctor = self.installer.plan("doctor", ["claude", "codex"], "user", self.project)
        result = self.installer.execute(doctor, confirmed=False)
        self.assertTrue(result["healthy"])
        self.assertEqual([], result["problems"])

    def test_project_scope_uses_only_project_root_and_reinstall_is_a_noop(self):
        definition = self.registry.definitions()["gemini"]
        plan = self.installer.plan("install", ["gemini"], "project", self.project)
        receipt = self.installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertEqual("applied", receipt.status)
        self.assertTrue(all(str(target).startswith(str(self.project.resolve())) for target in receipt.targets))
        self.assertFalse(definition.user_root.exists())

        again = self.installer.plan("install", ["gemini"], "project", self.project)
        self.assertTrue(again.noop)
        self.assertEqual("noop", self.installer.execute(again, confirmed=False).status)

    def test_update_uses_new_declared_artifact_version_and_uninstall_restores_only_receipt_owned_files(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        source = self.root / "adapter.md"
        source.write_text("version one\n")
        manifest["providers"] = {"fixture": {
            "native": False,
            "capabilities": ["scout"],
            "commands": [],
            "detect_commands": [],
            "user_root": "{config_root}/fixture",
            "project_root": "{project}/.fixture",
            "artifacts": [{"source": "adapter.md", "destination": "skills/adapter.md"}],
            "config_paths": [],
        }}
        manifest_path = self.root / "plugin.json"
        manifest_path.write_text(json.dumps(manifest))
        registry = ProviderRegistry(manifest_path, home=self.home, config_root=self.config)
        installer = Installer(registry, project_root=self.project)
        target = self.config / "fixture" / "skills" / "adapter.md"

        first = installer.plan("install", ["fixture"], "user", self.project)
        installer.execute(first, confirmed=first.preview["preview_hash"])
        self.assertEqual("version one\n", target.read_text())
        source.write_text("version two\n")
        update = installer.plan("update", ["fixture"], "user", self.project)
        installer.execute(update, confirmed=update.preview["preview_hash"])
        self.assertEqual("version two\n", target.read_text())

        fixture = self.config / "pre-existing.txt"
        fixture.write_text("preserve me\n")
        removal = installer.plan("uninstall", ["fixture"], "user", self.project)
        receipt = installer.execute(removal, confirmed=removal.preview["preview_hash"])
        self.assertEqual("rolled_back", receipt.status)
        self.assertFalse(target.exists())
        self.assertEqual("preserve me\n", fixture.read_text())

    def test_uninstall_refuses_user_modified_artifacts(self):
        plan = self.installer.plan("install", ["opencode"], "user", self.project)
        self.installer.execute(plan, confirmed=plan.preview["preview_hash"])
        target = self.registry.definitions()["opencode"].destination("user", self.project) / self.registry.definitions()["opencode"].artifacts[0].destination
        target.write_text("user modified\n")

        with self.assertRaises(InstallerConflictError):
            self.installer.plan("uninstall", ["opencode"], "user", self.project)

    def test_install_refuses_an_identical_but_unreceipted_artifact(self):
        definition = self.registry.definitions()["claude"]
        artifact = definition.artifacts[0]
        target = definition.destination("user", self.project) / artifact.destination
        target.parent.mkdir(parents=True)
        target.write_text(artifact.source.read_text())

        with self.assertRaises(InstallerConflictError):
            self.installer.plan("install", ["claude"], "user", self.project)

    def test_cli_without_a_provider_lists_detected_choices_instead_of_installing_all(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["install", "--scope", "project", "--project", str(self.project), "--dry-run", "--json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(0, code)
        self.assertEqual("install", payload["operation"])
        self.assertTrue(payload["data"]["selection_required"])
        self.assertFalse((self.project / ".claude").exists())

    def test_cli_applies_only_the_separately_reviewed_preview(self):
        preview_output = io.StringIO()
        with patch.dict(os.environ, {"MLX_AGENT_CONFIG_ROOT": str(self.config)}), contextlib.redirect_stdout(preview_output):
            self.assertEqual(0, main(["install", "claude", "--scope", "project", "--project", str(self.project), "--dry-run", "--json"]))
        preview = json.loads(preview_output.getvalue())["data"]["preview"]

        apply_output = io.StringIO()
        with patch.dict(os.environ, {"MLX_AGENT_CONFIG_ROOT": str(self.config)}), contextlib.redirect_stdout(apply_output):
            self.assertEqual(0, main([
                "install", "claude", "--scope", "project", "--project", str(self.project),
                "--confirm", "--preview-hash", preview["preview_hash"], "--json",
            ]))
        receipt = json.loads(apply_output.getvalue())["data"]["result"]
        self.assertEqual("applied", receipt["status"])
        self.assertTrue((self.project / ".claude" / "skills" / "mlx-scout" / "SKILL.md").is_file())

    def _two_artifact_installer(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        (self.root / "one.md").write_text("one-v1\n")
        (self.root / "two.md").write_text("two-v1\n")
        manifest["providers"] = {"fixture": {
            "native": False, "capabilities": ["scout"], "commands": [], "detect_commands": [],
            "user_root": "{config_root}/fixture", "project_root": "{project}/.fixture",
            "artifacts": [
                {"source": "one.md", "destination": "skills/one.md"},
                {"source": "two.md", "destination": "skills/two.md"},
            ], "config_paths": [],
        }}
        manifest_path = self.root / "two-artifact-plugin.json"
        manifest_path.write_text(json.dumps(manifest))
        registry = ProviderRegistry(manifest_path, home=self.home, config_root=self.config)
        return Installer(registry, project_root=self.project)

    def test_execute_rejects_an_outer_confirmation_when_an_inner_preview_changed(self):
        plan = self.installer.plan("install", ["claude"], "user", self.project)
        plan.transactions[0].changes[0]["content"] = "different reviewed content\n"
        with self.assertRaisesRegex(PermissionError, "installer preview is stale"):
            self.installer.execute(plan, confirmed=plan.preview["preview_hash"])

    def test_second_provider_failure_compensates_first_and_records_batch_rollback(self):
        calls = []

        def transaction_factory(**kwargs):
            calls.append(kwargs)
            if len(calls) == 4:
                kwargs["fault_injector"] = lambda point: (_ for _ in ()).throw(RuntimeError("second provider interrupted")) if point == "before_replace:0" else None
            return Transaction(**kwargs)

        installer = Installer(self.registry, project_root=self.project, transaction_factory=transaction_factory)
        plan = installer.plan("install", ["claude", "codex"], "user", self.project)
        receipt = installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertEqual("rolled_back", receipt.status)
        self.assertFalse((self.config / ".claude" / "skills" / "mlx-scout" / "SKILL.md").exists())
        self.assertFalse((self.config / ".codex" / "skills" / "mlx-scout" / "SKILL.md").exists())
        self.assertTrue(Path(receipt.batch_path).is_file())
        self.assertEqual("rolled_back", json.loads(Path(receipt.batch_path).read_text())["status"])

    def test_interrupted_batch_keeps_an_explicit_recovery_journal_when_compensation_fails(self):
        calls = []

        def transaction_factory(**kwargs):
            calls.append(kwargs)
            if len(calls) == 4:
                kwargs["fault_injector"] = lambda point: (_ for _ in ()).throw(RuntimeError("interrupted")) if point == "before_replace:0" else None
            return Transaction(**kwargs)

        installer = Installer(
            self.registry, project_root=self.project, transaction_factory=transaction_factory,
            rollback_func=lambda *args, **kwargs: SimpleNamespace(status="rollback_failed"),
        )
        plan = installer.plan("install", ["claude", "codex"], "user", self.project)
        receipt = installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertEqual("recovery_required", receipt.status)
        self.assertTrue((self.config / ".claude" / "skills" / "mlx-scout" / "SKILL.md").is_file())
        self.assertEqual("recovery_required", json.loads(Path(receipt.batch_path).read_text())["status"])

    def test_partial_two_artifact_update_keeps_per_artifact_history_for_doctor_and_uninstall(self):
        installer = self._two_artifact_installer()
        install = installer.plan("install", ["fixture"], "user", self.project)
        installer.execute(install, confirmed=install.preview["preview_hash"])
        (self.root / "one.md").write_text("one-v2\n")
        update = installer.plan("update", ["fixture"], "user", self.project)
        self.assertEqual(1, update.preview["changes"])
        installer.execute(update, confirmed=update.preview["preview_hash"])
        doctor = installer.execute(installer.plan("doctor", ["fixture"], "user", self.project))
        self.assertTrue(doctor["healthy"])
        removal = installer.plan("uninstall", ["fixture"], "user", self.project)
        installer.execute(removal, confirmed=removal.preview["preview_hash"])
        self.assertFalse((self.config / "fixture" / "skills" / "one.md").exists())
        self.assertFalse((self.config / "fixture" / "skills" / "two.md").exists())

    def test_uninstall_rechecks_expected_hash_under_rollback_lock(self):
        install = self.installer.plan("install", ["claude"], "user", self.project)
        self.installer.execute(install, confirmed=install.preview["preview_hash"])
        removal = self.installer.plan("uninstall", ["claude"], "user", self.project)
        target = self.config / ".claude" / "skills" / "mlx-scout" / "SKILL.md"

        def mutate_then_rollback(receipt_path, **kwargs):
            target.write_text("changed after uninstall preflight\n")
            return rollback(receipt_path, **kwargs)

        installer = Installer(self.registry, project_root=self.project, rollback_func=mutate_then_rollback)
        with self.assertRaises(InstallerConflictError):
            installer.execute(removal, confirmed=removal.preview["preview_hash"])
        self.assertEqual("changed after uninstall preflight\n", target.read_text())

    def test_uninstall_failure_restores_completed_provider_removals(self):
        install = self.installer.plan("install", ["claude", "codex"], "user", self.project)
        self.installer.execute(install, confirmed=install.preview["preview_hash"])
        removal = self.installer.plan("uninstall", ["claude", "codex"], "user", self.project)
        calls = []

        def fail_second_rollback(receipt_path, **kwargs):
            calls.append(receipt_path)
            if len(calls) == 2:
                return SimpleNamespace(status="rollback_failed")
            return rollback(receipt_path, **kwargs)

        installer = Installer(self.registry, project_root=self.project, rollback_func=fail_second_rollback)
        receipt = installer.execute(removal, confirmed=removal.preview["preview_hash"])
        self.assertEqual("rolled_back", receipt.status)
        self.assertTrue((self.config / ".claude" / "skills" / "mlx-scout" / "SKILL.md").is_file())
        self.assertTrue((self.config / ".codex" / "skills" / "mlx-scout" / "SKILL.md").is_file())

    def test_between_receipt_user_change_is_not_clobbered_by_uninstall_compensation(self):
        installer = self._two_artifact_installer()
        install = installer.plan("install", ["fixture"], "user", self.project)
        installer.execute(install, confirmed=install.preview["preview_hash"])
        (self.root / "one.md").write_text("one-v2\n")
        update = installer.plan("update", ["fixture"], "user", self.project)
        installer.execute(update, confirmed=update.preview["preview_hash"])
        removal = installer.plan("uninstall", ["fixture"], "user", self.project)
        changed = self.config / "fixture" / "skills" / "two.md"
        calls = []

        def mutate_between_receipts(receipt_path, **kwargs):
            calls.append(receipt_path)
            if len(calls) == 2:
                changed.write_text("user changed between receipts\n")
            return rollback(receipt_path, **kwargs)

        installer = Installer(self._two_artifact_installer().registry, project_root=self.project, rollback_func=mutate_between_receipts)
        receipt = installer.execute(removal, confirmed=removal.preview["preview_hash"])
        self.assertEqual("rolled_back", receipt.status)
        self.assertEqual("user changed between receipts\n", changed.read_text())

    def test_doctor_rejects_leaf_and_ancestor_symlink_artifacts(self):
        install = self.installer.plan("install", ["claude"], "user", self.project)
        self.installer.execute(install, confirmed=install.preview["preview_hash"])
        target = self.config / ".claude" / "skills" / "mlx-scout" / "SKILL.md"
        external = self.root / "external.md"
        external.write_text(target.read_text())
        target.unlink()
        target.symlink_to(external)
        result = self.installer.execute(self.installer.plan("doctor", ["claude"], "user", self.project))
        self.assertFalse(result["healthy"])

        target.unlink()
        target.parent.parent.rename(target.parent.parent.with_name("moved-skills"))
        target.parent.parent.symlink_to(target.parent.parent.with_name("moved-skills"), target_is_directory=True)
        result = self.installer.execute(self.installer.plan("doctor", ["claude"], "user", self.project))
        self.assertFalse(result["healthy"])
