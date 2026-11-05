"""Run the Test workflow pin-guard step against changes in a scratch repository."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COPIED = ("scripts/pin_guard.py", ".github/workflows/test.yml", "pyproject.toml",
          "requirements-dev.txt")
COMMIT = "547b4b1" + "0" * 33
CLIENT = "https://github.com/cavi-ai/cavi-api-client.git"


class PinGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "gitconfig").write_text(
            "[user]\n\tname = Pin Test\n\temail = pin@example.invalid\n")
        self.env = dict(os.environ, GIT_CONFIG_GLOBAL=str(self.root / "gitconfig"),
                        GIT_CONFIG_NOSYSTEM="1")
        self.repo = self.root / "repo"
        for name in COPIED:
            (self.repo / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / name, self.repo / name)
        self.git("init", "-q", "-b", "main")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "base")
        self.base = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, env=self.env, check=True,
                              capture_output=True, text=True).stdout.strip()

    def edit(self, name, old, new):
        path = self.repo / name
        text = path.read_text()
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1))

    def move_client(self):
        path = self.repo / "requirements-dev.txt"
        lines = [f"cavi-api-client @ git+{CLIENT}@{COMMIT}#subdirectory=python"
                 if line.startswith("cavi-api-client") else line
                 for line in path.read_text().splitlines()]
        path.write_text("\n".join(lines) + "\n")

    def change(self, change):
        self.git("checkout", "-q", "-b", "change", self.base)
        change()
        self.git("add", "-A", "--", ".", ":(exclude)vendor")
        self.git("commit", "-q", "-m", "change")
        workflow = (ROOT / ".github/workflows/test.yml").read_text()
        block = workflow.split("  pin-guard:\n", 1)[1].split("        run: |\n", 1)[1]
        script = "\n".join(line[10:] for line in
                           block.split("\n\n  contracts-and-install-lifecycle:", 1)[0].splitlines())
        runner = self.root / "runner"
        runner.mkdir(exist_ok=True)
        head = self.git("rev-parse", "HEAD")
        return subprocess.run(["bash", "-e", "-c", script], cwd=self.repo, capture_output=True,
                              text=True, check=False,
                              env=dict(self.env, BASE=self.base, HEAD=head,
                                       RUNNER_TEMP=str(runner)))

    def test_moving_a_git_requirement_fails(self):
        result = self.change(self.move_client)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("::error file=requirements-dev.txt::", result.stdout)

    def test_submodule_fails(self):
        def add_submodule():
            (self.repo / ".gitmodules").write_text(
                f'[submodule "vendor/client"]\n\tpath = vendor/client\n\turl = {CLIENT}\n')
            self.git("update-index", "--add", "--cacheinfo", f"160000,{COMMIT},vendor/client")
        result = self.change(add_submodule)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("::error file=.gitmodules::", result.stdout)
        self.assertIn("::error file=vendor/client::", result.stdout)

    def test_consumer_dispatch_is_never_changed(self):
        result = self.change(lambda: self.edit(
            ".github/workflows/test.yml", "repos/cavi-ai/mlx-workbench/dispatches",
            "repos/cavi-ai/other/dispatches"))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("changes the pin-consumers job", result.stdout)

    def test_main_guard_judges_a_rewritten_guard(self):
        def disable_guard():
            (self.repo / "scripts/pin_guard.py").write_text("import sys\nsys.exit(0)\n")
            self.move_client()
        result = self.change(disable_guard)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("::error file=scripts/pin_guard.py::", result.stdout)
        self.assertIn("::error file=requirements-dev.txt::", result.stdout)

    def test_version_range_and_other_jobs_pass(self):
        def update():
            self.edit("requirements-dev.txt", "coverage>=7.16,<8", "coverage>=7.17,<8")
            self.edit(".github/workflows/test.yml", "timeout-minutes: 60", "timeout-minutes: 70")
        result = self.change(update)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
