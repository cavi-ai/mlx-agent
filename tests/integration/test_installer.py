import json
import contextlib
import io
import os
import fcntl
import tempfile
import unittest
import subprocess
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from mlx_agent.installer import Installer, InstallerConflictError, InstallerPathError
from mlx_agent.providers import ProviderRegistry
from mlx_agent.cli import _installer_registry, main
from mlx_agent.transactions import Transaction, _target_lock_name, rollback


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
        # Tests that drive the CLI build their registry from the environment.
        # Pin the host roots so the developer's own dotfiles never decide the
        # result; individual tests still layer their own overrides on top.
        environment = patch.dict(os.environ, {
            "MLX_AGENT_HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.root / "xdg-config"),
            "XDG_STATE_HOME": str(self.config),
        })
        environment.start()
        self.addCleanup(environment.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def test_cli_registry_routes_opencode_and_receipts_to_xdg_roots(self):
        native_home = self.root / "native-home"
        xdg_config = self.root / "mirza-config"
        xdg_state = self.root / "mirza-state"
        with patch.dict(os.environ, {
            "MLX_AGENT_HOME": str(native_home),
            "XDG_CONFIG_HOME": str(xdg_config),
            "XDG_STATE_HOME": str(xdg_state),
        }, clear=False):
            registry = _installer_registry()
            definition = registry.definitions()["opencode"]

        self.assertEqual((xdg_config / "opencode").resolve(), definition.user_root)
        self.assertEqual(xdg_state.resolve(), registry.config_root)

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
        self.assertFalse(result["healthy"])
        self.assertEqual(
            {"claude": "staged", "codex": "staged"},
            {item["provider"]: item["integration_state"] for item in result["providers"]},
        )
        self.assertEqual(
            ["provider_staged", "provider_staged"],
            [item["code"] for item in result["problems"]],
        )

    def test_claude_install_stages_complete_bundle_and_all_capabilities_execute(self):
        plan = self.installer.plan("install", ["claude"], "user", self.project)
        receipt = self.installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertEqual("applied", receipt.status)
        package = self.home / ".claude" / "plugins" / "mlx-agent"
        for capability in ("scout", "adopt", "wire"):
            self.assertTrue((package / "commands" / "mlx-{0}.md".format(capability)).is_file())
        self.assertTrue((package / ".mcp.json").is_file())
        self.assertTrue((package / "scripts" / "mlx-agent-mcp").is_file())
        self.assertTrue((package / "src" / "mlx_agent" / "mcp_server.py").is_file())

        fixture = ROOT / "tests" / "fixtures" / "scout_responses.json"
        environment = dict(os.environ, MLX_AGENT_FIXTURE=str(fixture))
        launcher = package / "scripts" / "mlx-agent"
        executions = (
            (["discover", "--limit", "1", "--json"], "discover"),
            ([
                "adopt", "start", "--state", str(self.root / "claude-adoption.json"),
                "--shortlist-limit", "1", "--fast", "--no-network", "--json",
            ], "adopt-start"),
            ([
                "wire", "render", "mlx-community/Test-4bit", "--target", "mlx_lm",
                "--path", str(self.root / "claude-mlx-lm.json"), "--json",
            ], "wire-render"),
        )
        for arguments, operation in executions:
            result = subprocess.run(
                ["python3", str(launcher)] + arguments,
                cwd=self.root, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(operation, json.loads(result.stdout)["operation"])

        doctor = self.installer.execute(
            self.installer.plan("doctor", ["claude"], "user", self.project)
        )
        self.assertFalse(doctor["healthy"])
        self.assertEqual("staged", doctor["providers"][0]["integration_state"])

    def test_project_scope_uses_only_project_root_and_reinstall_is_a_noop(self):
        definition = self.registry.definitions()["agy"]
        plan = self.installer.plan("install", ["agy"], "project", self.project)
        receipt = self.installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertEqual("applied", receipt.status)
        self.assertTrue(all(str(target).startswith(str(self.project.resolve())) for target in receipt.targets))
        self.assertFalse(definition.user_root.exists())

        again = self.installer.plan("install", ["agy"], "project", self.project)
        self.assertTrue(again.noop)
        self.assertEqual("noop", self.installer.execute(again, confirmed=False).status)

    def test_plan_and_doctor_report_absent_native_provider_as_staged_not_healthy(self):
        installer = Installer(
            self.registry,
            project_root=self.project,
            executable_lookup=lambda command, path=None: None,
        )
        plan = installer.plan("install", ["agy"], "project", self.project)
        self.assertEqual("absent", plan.compatibility[0]["state"])
        self.assertEqual(
            "applied", installer.execute(
                plan, confirmed=plan.preview["preview_hash"]
            ).status,
        )
        result = installer.execute(
            installer.plan("doctor", ["agy"], "project", self.project)
        )
        self.assertFalse(result["healthy"])
        self.assertEqual("staged", result["providers"][0]["integration_state"])
        self.assertIn("provider_absent", [item["code"] for item in result["problems"]])

    def test_doctor_keeps_native_visibility_separate_from_artifact_validity(self):
        installer = Installer(
            self.registry,
            project_root=self.project,
            executable_lookup=lambda command, path=None: "/fake/bin/agy"
            if command == "agy" else None,
            probe_runner=lambda argv, **kwargs: subprocess.CompletedProcess(
                argv, 0, b"agy 1.1.14\n"
            ),
        )
        plan = installer.plan("install", ["agy"], "user", self.project)
        installer.execute(plan, confirmed=plan.preview["preview_hash"])
        definition = self.registry.definitions()["agy"]
        target = definition.artifact_destination(
            "user", self.project, definition.artifacts[0]
        )
        target.unlink()

        result = installer.execute(
            installer.plan("doctor", ["agy"], "user", self.project)
        )
        self.assertFalse(result["healthy"])
        self.assertEqual("invalid", result["providers"][0]["artifact_state"])
        self.assertEqual("native-visible", result["providers"][0]["integration_state"])
        codes = [item["code"] for item in result["problems"]]
        self.assertIn("artifact_invalid", codes)
        self.assertNotIn("provider_absent", codes)

    def test_agentskills_uninstall_removes_empty_owned_skill_directories_in_both_scopes(self):
        for scope in ("user", "project"):
            with self.subTest(scope=scope):
                plan = self.installer.plan("install", ["agentskills"], scope, self.project)
                self.assertEqual("applied", self.installer.execute(plan, confirmed=plan.preview["preview_hash"]).status)
                root = self.home / ".agents" if scope == "user" else self.project / ".agents"
                launcher = root / "skills" / "mlx-scout" / "scripts" / "mlx-agent"
                result = subprocess.run(
                    ["python3", str(launcher), "discover", "--limit", "1", "--json"],
                    env=dict(os.environ, MLX_AGENT_FIXTURE=str(ROOT / "tests" / "fixtures" / "scout_responses.json")),
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                removal = self.installer.plan("uninstall", ["agentskills"], scope, self.project)
                self.assertEqual("rolled_back", self.installer.execute(removal, confirmed=removal.preview["preview_hash"]).status)
                for capability in ("mlx-scout", "mlx-adopt", "mlx-wire"):
                    self.assertFalse((root / "skills" / capability).exists())

    def test_legacy_lock_migration_keeps_agentskills_uninstall_clean_and_doctor_rejects_recreation(self):
        definition = self.registry.definitions()["agentskills"]
        artifact = definition.artifacts[0]
        target = definition.artifact_destination("user", self.project, artifact)
        target.parent.mkdir(parents=True)
        legacy = target.parent / _target_lock_name(target)
        legacy.write_text("stale legacy lock\n")

        install = self.installer.plan("install", ["agentskills"], "user", self.project)
        applied = self.installer.execute(install, confirmed=install.preview["preview_hash"])
        self.assertEqual("applied", applied.status)
        self.assertFalse(legacy.exists())
        receipt = json.loads(Path(applied.receipt_path).read_text())
        self.assertEqual("legacy-target-locks-v1", receipt["lock_migration"])

        legacy.write_text("recreated legacy lock\n")
        doctor = self.installer.execute(self.installer.plan("doctor", ["agentskills"], "user", self.project))
        self.assertFalse(doctor["healthy"])
        problem = next(item for item in doctor["problems"] if item["code"] == "legacy_lock_recreated")
        self.assertIn("stop older mlx-agent processes", problem["remediation"])

        removal = self.installer.plan("uninstall", ["agentskills"], "user", self.project)
        with self.assertRaisesRegex(InstallerConflictError, "legacy_lock_recreated"):
            self.installer.execute(removal, confirmed=removal.preview["preview_hash"])

    def test_target_specific_legacy_state_is_isolated_between_user_and_project_scopes(self):
        installer = self._two_artifact_installer()
        targets = {
            "user": self.config / "fixture" / "skills" / "one.md",
            "project": self.project / ".fixture" / "skills" / "one.md",
        }
        for scope, target in targets.items():
            target.parent.mkdir(parents=True)
            (target.parent / _target_lock_name(target)).write_text("stale\n")
            plan = installer.plan("install", ["fixture"], scope, self.project)
            self.assertEqual("applied", installer.execute(plan, confirmed=plan.preview["preview_hash"]).status)

        user_state = json.loads((self.config / "mlx-agent" / "installer-receipts" / "locks" / "legacy-lock-migration-v1.json").read_text())
        project_state = json.loads((self.project / ".mlx-agent" / "installer-receipts" / "locks" / "legacy-lock-migration-v1.json").read_text())
        self.assertIn(_target_lock_name(targets["user"])[16:-5], user_state["targets"])
        self.assertIn(_target_lock_name(targets["project"])[16:-5], project_state["targets"])
        self.assertNotEqual(user_state["targets"], project_state["targets"])

    def test_busy_legacy_lock_refuses_installer_install_update_and_rollback(self):
        installer = self._two_artifact_installer()
        target = self.config / "fixture" / "skills" / "one.md"
        target.parent.mkdir(parents=True)
        legacy = target.parent / _target_lock_name(target)
        legacy.write_text("legacy\n")

        descriptor = os.open(legacy, os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            install = installer.plan("install", ["fixture"], "user", self.project)
            with self.assertRaisesRegex(InstallerConflictError, "legacy_lock_busy"):
                installer.execute(install, confirmed=install.preview["preview_hash"])
            self.assertFalse(target.exists())
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

        install = installer.plan("install", ["fixture"], "user", self.project)
        self.assertEqual("applied", installer.execute(install, confirmed=install.preview["preview_hash"]).status)
        self.assertEqual("one-v1\n", target.read_text())
        (self.root / "one.md").write_text("one-v2\n")
        legacy.write_text("recreated\n")
        descriptor = os.open(legacy, os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            update = installer.plan("update", ["fixture"], "user", self.project)
            with self.assertRaisesRegex(InstallerConflictError, "legacy_lock_busy"):
                installer.execute(update, confirmed=update.preview["preview_hash"])
            self.assertEqual("one-v1\n", target.read_text())

            removal = installer.plan("uninstall", ["fixture"], "user", self.project)
            with self.assertRaisesRegex(InstallerConflictError, "legacy_lock_busy"):
                installer.execute(removal, confirmed=removal.preview["preview_hash"])
            self.assertEqual("one-v1\n", target.read_text())
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def test_cli_classifies_busy_legacy_lock(self):
        target = self.home / ".agents" / "skills" / "mlx-scout" / "SKILL.md"
        target.parent.mkdir(parents=True)
        legacy = target.parent / _target_lock_name(target)
        legacy.write_text("legacy\n")
        with patch.dict(os.environ, {
            "MLX_AGENT_CONFIG_ROOT": str(self.config),
            "MLX_AGENT_HOME": str(self.home),
        }):
            preview = io.StringIO()
            with contextlib.redirect_stdout(preview):
                self.assertEqual(0, main(["install", "agentskills", "--scope", "user", "--project", str(self.project), "--dry-run", "--json"]))
            preview_hash = json.loads(preview.getvalue())["data"]["preview"]["preview_hash"]
            descriptor = os.open(legacy, os.O_RDWR | os.O_NOFOLLOW)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(2, main(["install", "agentskills", "--scope", "user", "--project", str(self.project), "--confirm", "--preview-hash", preview_hash, "--json"]))
                self.assertEqual("legacy_lock_busy", json.loads(output.getvalue())["error"]["code"])
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def test_doctor_reports_a_symlinked_legacy_lock_without_following_it(self):
        definition = self.registry.definitions()["agentskills"]
        target = definition.artifact_destination("user", self.project, definition.artifacts[0])
        target.parent.mkdir(parents=True)
        outside = self.root / "outside.lock"
        outside.write_text("outside\n")
        (target.parent / _target_lock_name(target)).symlink_to(outside)
        result = self.installer.execute(self.installer.plan("doctor", ["agentskills"], "user", self.project))
        self.assertFalse(result["healthy"])
        self.assertIn("legacy_lock_invalid", [item["code"] for item in result["problems"]])

    def test_update_uses_new_declared_artifact_version_and_uninstall_restores_only_receipt_owned_files(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        source = self.root / "adapter.md"
        source.write_text("version one\n")
        manifest["providers"] = {"fixture": {
            "native": False,
            "capabilities": ["scout"],
            "commands": [],
            "detect_commands": [],
            "invocation": {"kind": "skill", "prefix": ""},
            "minimum_version": None,
            "last_tested_version": None,
            "version_probe": None,
            "install_mode": "portable",
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

    def test_claude_install_is_the_repository_root_and_carries_no_provider_copy(self):
        plan = self.installer.plan("install", ["claude"], "user", self.project)
        self.installer.execute(plan, confirmed=plan.preview["preview_hash"])
        package = self.home / ".claude" / "plugins" / "mlx-agent"
        installed = self._files(package)
        self.assertNotIn("providers", {Path(path).parts[0] for path in installed})
        self.assertEqual(
            sorted({Path(path).parts[0] for path in installed}),
            [".claude-plugin", ".mcp.json", "agents", "commands", "scripts", "src"],
        )
        for path in installed:
            self.assertEqual((ROOT / path).read_bytes(), (package / path).read_bytes(), path)
        self.assertEqual(
            ["scripts/mlx-adopt.workflow.mjs", "scripts/mlx-agent", "scripts/mlx-agent-mcp"],
            [path for path in installed if path.startswith("scripts/")],
        )

    def test_agentskills_install_copies_the_root_runtime_into_every_skill_and_each_runs_alone(self):
        plan = self.installer.plan("install", ["agentskills"], "user", self.project)
        self.installer.execute(plan, confirmed=plan.preview["preview_hash"])
        skills = self.home / ".agents" / "skills"
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        runtime = sorted(
            str(path.relative_to(ROOT / "src" / "mlx_agent"))
            for path in (ROOT / "src" / "mlx_agent").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
        for capability in ("scout", "adopt", "wire", "bench", "doctor", "watch", "fleet"):
            skill = skills / "mlx-{0}".format(capability)
            with self.subTest(skill=skill.name):
                self.assertEqual(
                    sorted(["SKILL.md", "scripts/mlx-agent", "scripts/mlx-agent-mcp"] + ["src/mlx_agent/" + path for path in runtime]),
                    self._files(skill),
                )
                skill_text = (skill / "SKILL.md").read_text()
                self.assertEqual(
                    (ROOT / "providers" / "agentskills" / skill.name / "SKILL.md").read_text(), skill_text
                )
                self.assertIn("<skill-dir>/scripts/mlx-agent", skill_text)
                result = subprocess.run(
                    ["python3", str(skill / "scripts" / "mlx-agent"), "discover", "--help"],
                    cwd=unrelated, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_codex_and_agy_install_one_package_root_runtime_shared_by_every_skill(self):
        plan = self.installer.plan("install", ["codex", "agy"], "user", self.project)
        self.installer.execute(plan, confirmed=plan.preview["preview_hash"])
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        for package in (self.home / "plugins" / "mlx-agent", self.home / ".gemini" / "config" / "plugins" / "mlx-agent"):
            with self.subTest(package=package.name):
                installed = self._files(package)
                self.assertEqual([], [path for path in installed if path.startswith("skills/") and not path.endswith("/SKILL.md")])
                for capability in ("scout", "adopt", "wire", "bench", "doctor", "watch", "fleet"):
                    skill = package / "skills" / "mlx-{0}".format(capability)
                    self.assertEqual(["SKILL.md"], self._files(skill))
                    self.assertTrue((skill.parent.parent / "scripts" / "mlx-agent").is_file())
                result = subprocess.run(
                    ["python3", str(package / "scripts" / "mlx-agent"), "discover", "--help"],
                    cwd=unrelated, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(0, result.returncode, result.stderr)

    def _layout_installer(self, layout):
        """Point one fixture provider (and an enclosing sibling) at a per-skill or package-root runtime layout."""
        package = self.root / "layout-package"
        for name, text in {
            "skill-a/SKILL.md": "skill a {0}\n".format(layout),
            "skill-b/SKILL.md": "skill b {0}\n".format(layout),
            "runtime/module.py": "VALUE = 1\n",
            "runtime/launcher": "#!/usr/bin/env python3\n",
            "outer/SKILL.md": "outer\n",
        }.items():
            (package / name).parent.mkdir(parents=True, exist_ok=True)
            (package / name).write_text(text)
        if layout == "per-skill":
            artifacts = [
                ("skill-a/SKILL.md", "skills/a/SKILL.md"),
                ("skill-b/SKILL.md", "skills/b/SKILL.md"),
                ("runtime/module.py", "skills/a/src/mlx_agent/module.py"),
                ("runtime/module.py", "skills/b/src/mlx_agent/module.py"),
                ("runtime/launcher", "skills/a/scripts/mlx-agent"),
                ("runtime/launcher", "skills/b/scripts/mlx-agent"),
            ]
        else:
            artifacts = [
                ("skill-a/SKILL.md", "skills/a/SKILL.md"),
                ("skill-b/SKILL.md", "skills/b/SKILL.md"),
                ("runtime/module.py", "src/mlx_agent/module.py"),
                ("runtime/launcher", "scripts/mlx-agent"),
            ]

        def provider(user_root, project_root, artifacts):
            return {
                "native": False, "capabilities": ["scout"], "commands": [], "detect_commands": [],
                "invocation": {"kind": "skill", "prefix": ""}, "minimum_version": None,
                "last_tested_version": None, "version_probe": None, "install_mode": "portable",
                "user_root": user_root, "project_root": project_root,
                "artifacts": [
                    {"source": "layout-package/" + source, "destination": destination}
                    for source, destination in artifacts
                ],
                "config_paths": [],
            }

        manifest = json.loads((ROOT / "plugin.json").read_text())
        manifest["providers"] = {
            "inner": provider("{config_root}/outer/inner", "{project}/.outer/inner", artifacts),
            "outer": provider("{config_root}/outer", "{project}/.outer", [("outer/SKILL.md", "skills/outer/SKILL.md")]),
        }
        (self.root / "plugin.json").write_text(json.dumps(manifest))
        return Installer(
            ProviderRegistry(self.root / "plugin.json", home=self.home, config_root=self.config),
            project_root=self.project,
        )

    def _apply(self, installer, action, providers):
        plan = installer.plan(action, providers, "user", self.project)
        return plan, installer.execute(plan, confirmed=plan.preview["preview_hash"])

    def _files(self, directory):
        return sorted(str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file())

    def test_update_to_a_new_layout_prunes_receipt_owned_files_the_manifest_no_longer_declares(self):
        old_installer = self._layout_installer("per-skill")
        self._apply(old_installer, "install", ["inner", "outer"])
        inner = self.config / "outer" / "inner"
        self.assertIn("skills/a/src/mlx_agent/module.py", self._files(inner))
        outer_before = (self.config / "outer" / "skills" / "outer" / "SKILL.md").read_bytes()

        installer = self._layout_installer("package-root")
        plan, receipt = self._apply(installer, "update", ["inner"])
        self.assertEqual("applied", receipt.status)
        self.assertEqual(
            ["scripts/mlx-agent", "skills/a/SKILL.md", "skills/b/SKILL.md", "src/mlx_agent/module.py"],
            self._files(inner),
        )
        self.assertEqual("skill a package-root\n", (inner / "skills" / "a" / "SKILL.md").read_text())
        self.assertEqual(outer_before, (self.config / "outer" / "skills" / "outer" / "SKILL.md").read_bytes())
        for removed in ("skills/a/src/mlx_agent/module.py", "skills/b/scripts/mlx-agent"):
            self.assertIn("-VALUE = 1" if removed.endswith(".py") else "-#!/usr/bin/env python3", plan.preview["diff"])
        self.assertTrue(all(item.status == "applied" for item in receipt.receipts))

        doctor = installer.execute(installer.plan("doctor", ["inner"], "user", self.project))
        self.assertEqual([], [item for item in doctor["problems"] if item["code"] == "artifact_invalid"])
        again = installer.plan("update", ["inner"], "user", self.project)
        self.assertTrue(again.noop)

    def test_uninstall_after_a_pruning_update_leaves_nothing_and_spares_the_enclosing_provider(self):
        self._apply(self._layout_installer("per-skill"), "install", ["inner", "outer"])
        installer = self._layout_installer("package-root")
        self._apply(installer, "update", ["inner"])
        _plan, receipt = self._apply(installer, "uninstall", ["inner"])
        self.assertEqual("rolled_back", receipt.status)
        inner = self.config / "outer" / "inner"
        self.assertEqual([], self._files(inner))
        self.assertEqual(["scripts", "skills", "src"], sorted(path.name for path in inner.rglob("*")))
        self.assertEqual(["skills/outer/SKILL.md"], self._files(self.config / "outer"))

    def test_uninstall_compensation_after_every_rollback_skips_pruned_paths_already_absent(self):
        self._apply(self._layout_installer("per-skill"), "install", ["inner"])
        installer = self._layout_installer("package-root")
        self._apply(installer, "update", ["inner"])
        inner = self.config / "outer" / "inner"
        before = {path: (inner / path).read_bytes() for path in self._files(inner)}
        plan = installer.plan("uninstall", ["inner"], "user", self.project)
        record = installer._batch_update

        def fail_to_record_completion(path, status, receipts, **kwargs):
            if status == "complete":
                raise OSError("injected journal failure after every rollback completed")
            return record(path, status, receipts, **kwargs)

        installer._batch_update = fail_to_record_completion
        receipt = installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertEqual("rolled_back", receipt.status)
        self.assertEqual(before, {path: (inner / path).read_bytes() for path in self._files(inner)})

    def test_uninstall_refuses_a_file_recreated_at_a_pruned_path(self):
        self._apply(self._layout_installer("per-skill"), "install", ["inner"])
        installer = self._layout_installer("package-root")
        self._apply(installer, "update", ["inner"])
        recreated = self.config / "outer" / "inner" / "skills" / "b" / "src" / "mlx_agent" / "module.py"
        recreated.parent.mkdir(parents=True, exist_ok=True)
        recreated.write_text("VALUE = 'user file'\n")
        with self.assertRaisesRegex(InstallerConflictError, "recreated at a pruned path"):
            installer.plan("uninstall", ["inner"], "user", self.project)
        self.assertEqual("VALUE = 'user file'\n", recreated.read_text())

    def test_uninstall_removes_a_layout_the_manifest_no_longer_declares_without_an_update(self):
        self._apply(self._layout_installer("per-skill"), "install", ["inner", "outer"])
        installer = self._layout_installer("package-root")
        _plan, receipt = self._apply(installer, "uninstall", ["inner"])
        self.assertEqual("rolled_back", receipt.status)
        self.assertEqual([], self._files(self.config / "outer" / "inner"))
        self.assertEqual(["skills"], sorted(path.name for path in (self.config / "outer" / "inner").rglob("*")))
        self.assertEqual(["skills/outer/SKILL.md"], self._files(self.config / "outer"))

    def test_update_refuses_to_prune_a_user_modified_stale_file_and_changes_nothing(self):
        self._apply(self._layout_installer("per-skill"), "install", ["inner"])
        inner = self.config / "outer" / "inner"
        stale = inner / "skills" / "b" / "src" / "mlx_agent" / "module.py"
        stale.write_text("VALUE = 'user edit'\n")
        before = {path: (inner / path).read_bytes() for path in self._files(inner)}
        installer = self._layout_installer("package-root")
        with self.assertRaisesRegex(InstallerConflictError, "user-modified"):
            installer.plan("update", ["inner"], "user", self.project)
        self.assertEqual(before, {path: (inner / path).read_bytes() for path in self._files(inner)})

    def test_failed_pruning_update_restores_every_file_it_removed(self):
        self._apply(self._layout_installer("per-skill"), "install", ["inner"])
        inner = self.config / "outer" / "inner"
        before = {path: (inner / path).read_bytes() for path in self._files(inner)}
        installer = self._layout_installer("package-root")
        plan = installer.plan("update", ["inner"], "user", self.project)
        calls = []

        # Changes 0-3 write the new layout; 4-7 remove the old one. Fail after two removals.
        self.assertEqual([False] * 4 + [True] * 4, [bool(change.get("remove")) for change in plan.transactions[0].changes])

        def fail_after_second_removal(point):
            calls.append(point)
            if point == "after_replace:5":
                raise OSError("injected failure")

        installer.transaction_factory = lambda **kwargs: Transaction(fault_injector=fail_after_second_removal, **kwargs)
        receipt = installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertNotEqual("applied", receipt.status)
        self.assertIn("after_replace:5", calls)
        self.assertEqual(before, {path: (inner / path).read_bytes() for path in self._files(inner)})

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

    def test_update_refuses_a_symlink_even_when_it_resolves_to_owned_identical_bytes(self):
        definition = self.registry.definitions()["claude"]
        artifact = definition.artifacts[0]
        target = definition.artifact_destination("user", self.project, artifact)
        plan = self.installer.plan("install", ["claude"], "user", self.project)
        self.assertEqual(
            "applied", self.installer.execute(
                plan, confirmed=plan.preview["preview_hash"]
            ).status,
        )
        outside = self.root / "outside-identical.md"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside)

        with self.assertRaisesRegex(ValueError, "unsafe target leaf"):
            self.installer.plan("update", ["claude"], "user", self.project)
        self.assertEqual(artifact.source.read_bytes(), outside.read_bytes())

    def _symlinked_claude_root(self):
        root = self.home / ".claude" / "plugins"
        root.mkdir(parents=True)
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (root / "mlx-agent").symlink_to(elsewhere, target_is_directory=True)
        return elsewhere.resolve()

    def test_symlinked_destination_root_installs_into_its_target(self):
        elsewhere = self._symlinked_claude_root()
        registry = ProviderRegistry(
            ROOT / "plugin.json", home=self.home, config_root=self.config
        )
        installer = Installer(registry, project_root=self.project)
        self.assertEqual(elsewhere, registry.definitions()["claude"].user_root)

        plan = installer.plan("install", ["claude"], "user", self.project)
        receipt = installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertEqual("applied", receipt.status)
        self.assertTrue((elsewhere / "commands" / "mlx-scout.md").is_file())

    def test_destination_root_symlinked_after_resolution_is_refused_by_name(self):
        """The guard covers a root that becomes a symlink after it was resolved."""
        elsewhere = self._symlinked_claude_root()
        unresolved = self.home / ".claude" / "plugins" / "mlx-agent"
        definitions = self.registry.definitions()
        definitions["claude"] = replace(definitions["claude"], user_root=unresolved)
        with patch.object(self.registry, "definitions", return_value=definitions):
            with self.assertRaises(InstallerPathError) as caught:
                self.installer.plan("install", ["claude"], "user", self.project)
        self.assertIn("symlink", str(caught.exception))
        self.assertIn(str(unresolved.name), str(caught.exception))
        self.assertIn(elsewhere.name, str(caught.exception))
        self.assertIn("XDG_CONFIG_HOME", caught.exception.remediation)

    def test_receipts_default_to_the_xdg_state_directory(self):
        registry = ProviderRegistry(ROOT / "plugin.json", home=self.home)
        self.assertEqual((self.home / ".local" / "state").resolve(), registry.config_root)

    def test_cli_without_a_provider_lists_detected_choices_instead_of_installing_all(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["install", "--scope", "project", "--project", str(self.project), "--dry-run", "--json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(0, code)
        self.assertEqual("install", payload["operation"])
        self.assertTrue(payload["data"]["selection_required"])
        self.assertFalse((self.project / ".claude").exists())

    def test_human_preview_surfaces_explicit_provider_compatibility_diagnostics(self):
        output = io.StringIO()
        with patch("mlx_agent.providers.shutil.which", return_value=None), patch.dict(
            os.environ,
            {"MLX_AGENT_CONFIG_ROOT": str(self.config), "MLX_AGENT_HOME": str(self.home)},
        ), contextlib.redirect_stdout(output):
            self.assertEqual(0, main([
                "install", "agy", "--scope", "project", "--project", str(self.project),
                "--dry-run",
            ]))
        text = output.getvalue()
        self.assertIn("agy: absent", text)
        self.assertIn("provider executable was not found", text)
        self.assertFalse((self.project / ".agents").exists())

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
        self.assertTrue((self.project / ".claude" / "plugins" / "mlx-agent" / "commands" / "mlx-scout.md").is_file())

    def _two_artifact_installer(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        (self.root / "one.md").write_text("one-v1\n")
        (self.root / "two.md").write_text("two-v1\n")
        manifest["providers"] = {"fixture": {
            "native": False, "capabilities": ["scout"], "commands": [], "detect_commands": [],
            "invocation": {"kind": "skill", "prefix": ""},
            "minimum_version": None, "last_tested_version": None,
            "version_probe": None, "install_mode": "portable",
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
        self.assertFalse((self.home / ".claude" / "plugins" / "mlx-agent" / "commands" / "mlx-scout.md").exists())
        self.assertFalse((self.home / "plugins" / "mlx-agent" / "skills" / "mlx-scout" / "SKILL.md").exists())
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
        self.assertTrue((self.home / ".claude" / "plugins" / "mlx-agent" / "commands" / "mlx-scout.md").is_file())
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
        target = self.home / ".claude" / "plugins" / "mlx-agent" / "commands" / "mlx-scout.md"

        def mutate_then_rollback(receipt_path, **kwargs):
            target.write_text("changed after uninstall preflight\n")
            return rollback(receipt_path, **kwargs)

        installer = Installer(self.registry, project_root=self.project, rollback_func=mutate_then_rollback)
        with self.assertRaises(InstallerConflictError):
            installer.execute(removal, confirmed=removal.preview["preview_hash"])
        self.assertEqual("changed after uninstall preflight\n", target.read_text())

    def test_uninstall_rechecks_complete_receipt_binding_under_rollback_lock(self):
        install = self.installer.plan("install", ["claude"], "user", self.project)
        self.installer.execute(install, confirmed=install.preview["preview_hash"])
        removal = self.installer.plan("uninstall", ["claude"], "user", self.project)
        target = self.home / ".claude" / "plugins" / "mlx-agent" / "commands" / "mlx-scout.md"

        def mutate_receipt_then_rollback(receipt_path, **kwargs):
            value = json.loads(Path(receipt_path).read_text())
            alternate_lock_root = self.root / "alternate-locks"
            alternate_lock_root.mkdir()
            value["lock_root"] = str(alternate_lock_root)
            Path(receipt_path).write_text(json.dumps(value))
            return rollback(receipt_path, **kwargs)

        installer = Installer(
            self.registry,
            project_root=self.project,
            rollback_func=mutate_receipt_then_rollback,
        )
        with self.assertRaisesRegex(InstallerConflictError, "receipt changed"):
            installer.execute(removal, confirmed=removal.preview["preview_hash"])
        self.assertTrue(target.is_file())

    def test_uninstall_later_failed_child_is_journaled_and_requires_recovery(self):
        install = self.installer.plan("install", ["claude", "codex"], "user", self.project)
        self.installer.execute(install, confirmed=install.preview["preview_hash"])
        removal = self.installer.plan("uninstall", ["claude", "codex"], "user", self.project)
        calls = []

        def fail_second_rollback(receipt_path, **kwargs):
            calls.append(receipt_path)
            result = rollback(receipt_path, **kwargs)
            if len(calls) == 2:
                result.status = "rollback_failed"
            return result

        installer = Installer(self.registry, project_root=self.project, rollback_func=fail_second_rollback)
        with self.assertRaises(InstallerConflictError):
            installer.execute(removal, confirmed=removal.preview["preview_hash"])
        self.assertTrue((self.home / ".claude" / "plugins" / "mlx-agent" / "commands" / "mlx-scout.md").is_file())
        self.assertFalse((self.home / "plugins" / "mlx-agent" / "skills" / "mlx-scout" / "SKILL.md").exists())
        journals = [json.loads(path.read_text()) for path in (self.config / "mlx-agent" / "installer-receipts" / "batches").glob("*/batch.json")]
        self.assertIn("recovery_required", [item["status"] for item in journals])
        journal = next(item for item in journals if item["status"] == "recovery_required")
        self.assertEqual(["rolled_back", "rollback_failed"], [item["status"] for item in journal["children"]])

    def test_uninstall_first_failed_child_is_journaled_and_requires_recovery(self):
        install = self.installer.plan("install", ["claude", "codex"], "user", self.project)
        self.installer.execute(install, confirmed=install.preview["preview_hash"])
        removal = self.installer.plan("uninstall", ["claude", "codex"], "user", self.project)

        def fail_first_rollback(receipt_path, **kwargs):
            result = rollback(receipt_path, **kwargs)
            result.status = "recovery_required"
            return result

        installer = Installer(self.registry, project_root=self.project, rollback_func=fail_first_rollback)
        with self.assertRaises(InstallerConflictError):
            installer.execute(removal, confirmed=removal.preview["preview_hash"])
        self.assertFalse((self.home / ".claude" / "plugins" / "mlx-agent" / "commands" / "mlx-scout.md").exists())
        self.assertTrue((self.home / "plugins" / "mlx-agent" / "skills" / "mlx-scout" / "SKILL.md").is_file())
        journals = [json.loads(path.read_text()) for path in (self.config / "mlx-agent" / "installer-receipts" / "batches").glob("*/batch.json")]
        self.assertIn("recovery_required", [item["status"] for item in journals])
        journal = next(item for item in journals if item["status"] == "recovery_required")
        self.assertEqual(["recovery_required"], [item["status"] for item in journal["children"]])

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
        target = self.home / ".claude" / "plugins" / "mlx-agent" / "commands" / "mlx-scout.md"
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

    def test_crash_after_child_receipt_write_leaves_batch_with_deterministic_active_receipt(self):
        class SimulatedCrash(BaseException):
            pass

        calls = []

        def transaction_factory(**kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                kwargs["fault_injector"] = lambda point: (_ for _ in ()).throw(SimulatedCrash()) if point == "after_pending_receipt" else None
            return Transaction(**kwargs)

        installer = Installer(self.registry, project_root=self.project, transaction_factory=transaction_factory)
        plan = installer.plan("install", ["claude"], "user", self.project)
        with self.assertRaises(SimulatedCrash):
            installer.execute(plan, confirmed=plan.preview["preview_hash"])
        journals = [json.loads(path.read_text()) for path in (self.config / "mlx-agent" / "installer-receipts" / "batches").glob("*/batch.json")]
        journal = journals[0]
        self.assertEqual("pending", journal["status"])
        self.assertEqual("claude", journal["active"]["provider"])
        self.assertTrue(Path(journal["active"]["expected_receipt_path"]).is_file())
        self.assertEqual(journal["active"]["transaction_id"], json.loads(Path(journal["active"]["expected_receipt_path"]).read_text())["transaction_id"])
        doctor = installer.execute(installer.plan("doctor", ["claude"], "user", self.project))
        self.assertFalse(doctor["healthy"])
        self.assertIn("batch_recovery_required", [item["code"] for item in doctor["problems"]])

    def test_recovery_required_child_is_recorded_and_keeps_batch_recovery_required(self):
        factory_calls, writes = [], []

        def transaction_factory(**kwargs):
            factory_calls.append(kwargs)
            if len(factory_calls) == 2:
                def writer(path, value):
                    writes.append(value["status"])
                    if len(writes) == 1:
                        path.write_text(json.dumps(value))
                    else:
                        raise OSError("receipt transition unavailable")
                kwargs["receipt_writer"] = writer
            return Transaction(**kwargs)

        installer = Installer(self.registry, project_root=self.project, transaction_factory=transaction_factory)
        plan = installer.plan("install", ["claude"], "user", self.project)
        receipt = installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertEqual("recovery_required", receipt.status)
        journal = json.loads(Path(receipt.batch_path).read_text())
        self.assertEqual("recovery_required", journal["status"])
        self.assertEqual("recovery_required", journal["children"][0]["status"])
        self.assertTrue(journal["remediation"])

    def test_retry_after_compensated_batch_uses_a_new_child_receipt_namespace(self):
        calls = []

        def transaction_factory(**kwargs):
            calls.append(kwargs)
            if len(calls) == 4:
                kwargs["fault_injector"] = lambda point: (_ for _ in ()).throw(RuntimeError("first attempt interrupted")) if point == "before_replace:0" else None
            return Transaction(**kwargs)

        installer = Installer(self.registry, project_root=self.project, transaction_factory=transaction_factory)
        plan = installer.plan("install", ["claude", "codex"], "user", self.project)
        first = installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertEqual("rolled_back", first.status)
        retry = installer.execute(plan, confirmed=plan.preview["preview_hash"])
        self.assertEqual("applied", retry.status)
        first_batch = json.loads(Path(first.batch_path).read_text())
        retry_batch = json.loads(Path(retry.batch_path).read_text())
        self.assertNotEqual(first_batch["batch_id"], retry_batch["batch_id"])
        self.assertTrue({item["transaction_id"] for item in first_batch["operations"]}.isdisjoint({item["transaction_id"] for item in retry_batch["operations"]}))

    def test_uninstall_compensation_does_not_overwrite_user_edit_after_completed_rollback(self):
        install = self.installer.plan("install", ["claude", "codex"], "user", self.project)
        self.installer.execute(install, confirmed=install.preview["preview_hash"])
        removal = self.installer.plan("uninstall", ["claude", "codex"], "user", self.project)
        target = self.home / ".claude" / "plugins" / "mlx-agent" / "commands" / "mlx-scout.md"
        calls = []

        def edit_then_fail(receipt_path, **kwargs):
            calls.append(receipt_path)
            if len(calls) == 2:
                target.write_text("user edit after completed rollback\n")
                return SimpleNamespace(status="rollback_failed")
            return rollback(receipt_path, **kwargs)

        installer = Installer(self.registry, project_root=self.project, rollback_func=edit_then_fail)
        with self.assertRaises(InstallerConflictError):
            installer.execute(removal, confirmed=removal.preview["preview_hash"])
        self.assertEqual("user edit after completed rollback\n", target.read_text())
        journals = [json.loads(path.read_text()) for path in (self.config / "mlx-agent" / "installer-receipts" / "batches").glob("*/batch.json")]
        journal = next(item for item in journals if item["status"] == "recovery_required")
        self.assertEqual("recovery_required", journal["status"])
        self.assertTrue(any(str(target.resolve()) in item["paths"] for item in journal["remediation"]))

    def test_uninstall_compensation_checks_rollback_state_under_its_mutation_lock(self):
        install = self.installer.plan("install", ["claude", "codex"], "user", self.project)
        self.installer.execute(install, confirmed=install.preview["preview_hash"])
        removal = self.installer.plan("uninstall", ["claude", "codex"], "user", self.project)
        target = self.home / ".claude" / "plugins" / "mlx-agent" / "commands" / "mlx-scout.md"
        factory_calls, expected_states = [], []

        class EditingCompensationTransaction(Transaction):
            def apply(self, confirmation, expected_current=None):
                if expected_current is not None:
                    expected_states.append(expected_current)
                    target.write_text("user edit between compensation preview and lock\n")
                return super().apply(confirmation) if expected_current is None else super().apply(confirmation, expected_current=expected_current)

        def transaction_factory(**kwargs):
            factory_calls.append(kwargs)
            if len(factory_calls) == 1:
                return EditingCompensationTransaction(**kwargs)
            return Transaction(**kwargs)

        calls = []

        def fail_second_rollback(receipt_path, **kwargs):
            calls.append(receipt_path)
            if len(calls) == 2:
                return SimpleNamespace(status="rollback_failed")
            return rollback(receipt_path, **kwargs)

        installer = Installer(self.registry, project_root=self.project, transaction_factory=transaction_factory, rollback_func=fail_second_rollback)
        with self.assertRaises(InstallerConflictError):
            installer.execute(removal, confirmed=removal.preview["preview_hash"])
        self.assertTrue(expected_states)
        self.assertEqual("user edit between compensation preview and lock\n", target.read_text())
        journal = next(json.loads(path.read_text()) for path in (self.config / "mlx-agent" / "installer-receipts" / "batches").glob("*/batch.json") if json.loads(path.read_text())["status"] == "recovery_required")
        self.assertTrue(any(str(target.resolve()) in item["paths"] for item in journal["remediation"]))
