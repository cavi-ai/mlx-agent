import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_agent.installer import Installer
from mlx_agent.providers import ProviderRegistry


ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES = ("scout", "adopt", "wire", "bench", "doctor", "watch", "fleet", "task")


class GeminiAdapterTests(unittest.TestCase):
    def test_generated_gemini_surface_is_zero_bloat_ten_files(self):
        gemini_dir = ROOT / "providers" / "gemini"
        files = sorted(str(path.relative_to(gemini_dir)) for path in gemini_dir.rglob("*") if path.is_file())
        self.assertEqual(10, len(files))
        self.assertIn("gemini-extension.json", files)
        self.assertIn(".mlx-agent-generated-files.json", files)
        self.assertFalse((gemini_dir / "src").exists())
        manifest = json.loads((gemini_dir / "gemini-extension.json").read_text(encoding="utf-8"))
        self.assertEqual("mlx-agent", manifest["name"])
        self.assertEqual(json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))["version"], manifest["version"])
        for capability in CAPABILITIES:
            skill_md = gemini_dir / "skills" / "mlx-{0}".format(capability) / "SKILL.md"
            self.assertTrue(skill_md.is_file())
            self.assertIn("<plugin-root>/scripts/mlx-agent", skill_md.read_text(encoding="utf-8"))

    def test_gemini_user_and_project_lifecycle_coexists_with_agy_and_runs_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "home"
            config = root / "config"
            project = root / "project"
            unrelated = root / "unrelated"
            for path in (home, config, project, unrelated):
                path.mkdir(parents=True)
            registry = ProviderRegistry(ROOT / "plugin.json", home=home, config_root=config)
            installer = Installer(registry, project_root=project)

            # User scope coexistence with agy
            plan = installer.plan("install", ["agy", "gemini"], "user", project)
            installer.execute(plan, confirmed=plan.preview["preview_hash"])
            gemini_user = home / ".gemini" / "extensions" / "mlx-agent"
            agy_user = home / ".gemini" / "config" / "plugins" / "mlx-agent"
            self.assertTrue((gemini_user / "gemini-extension.json").is_file())
            self.assertTrue((gemini_user / "skills" / "mlx-scout" / "SKILL.md").is_file())
            self.assertTrue((gemini_user / "scripts" / "mlx-agent").is_file())
            self.assertTrue((agy_user / "plugin.json").is_file())

            # Launcher executes from unrelated CWD with fixture
            env = dict(os.environ, MLX_AGENT_FIXTURE=str(ROOT / "tests" / "fixtures" / "scout_responses.json"))
            env.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, str(gemini_user / "scripts" / "mlx-agent"), "discover", "--limit", "1", "--json"],
                cwd=unrelated, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual("ok", json.loads(result.stdout)["status"])

            # Project scope install & uninstall
            proj_plan = installer.plan("install", ["gemini"], "project", project)
            installer.execute(proj_plan, confirmed=proj_plan.preview["preview_hash"])
            gemini_proj = project / ".gemini" / "extensions" / "mlx-agent"
            self.assertTrue((gemini_proj / "gemini-extension.json").is_file())

            rm_proj = installer.plan("uninstall", ["gemini"], "project", project)
            installer.execute(rm_proj, confirmed=rm_proj.preview["preview_hash"])
            self.assertFalse((gemini_proj / "gemini-extension.json").exists())
            self.assertEqual([], [p for p in gemini_proj.rglob("*") if p.is_file()] if gemini_proj.exists() else [])

            # Uninstall gemini user scope leaves agy intact
            rm_user = installer.plan("uninstall", ["gemini"], "user", project)
            installer.execute(rm_user, confirmed=rm_user.preview["preview_hash"])
            self.assertFalse((gemini_user / "gemini-extension.json").exists())
            self.assertEqual([], [p for p in gemini_user.rglob("*") if p.is_file()] if gemini_user.exists() else [])
            self.assertTrue((agy_user / "plugin.json").is_file())

    def test_gemini_layout_validation_rejects_wrong_user_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
            bad_gemini = dict(manifest["providers"]["gemini"], user_root="{home}/.gemini/wrong")
            registry = ProviderRegistry(ROOT / "plugin.json", home=root / "home", config_root=root / "config")
            with self.assertRaisesRegex(ValueError, "provider gemini.user_root"):
                registry._definition("gemini", bad_gemini)


if __name__ == "__main__":
    unittest.main()
