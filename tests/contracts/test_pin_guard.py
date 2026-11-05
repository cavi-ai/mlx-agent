"""Run the pin-guard and pin-integrity steps against changes in a scratch repository."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COPIED = ("scripts/pin_guard.py", ".github/workflows/test.yml", ".github/workflows/pin-client.yml",
          ".github/workflows/pin-integrity.yml", "pyproject.toml", "requirements-dev.txt")
COMMIT = "547b4b1" + "0" * 33
CLIENT = "https://github.com/cavi-ai/cavi-api-client.git"
BOT = ("github-actions[bot]", "41898282+github-actions[bot]@users.noreply.github.com")
PERSON = ("Pin Test", "pin@example.invalid")


def step(workflow, job, after):
    """The run script of a job's last step, as the workflow file holds it."""
    text = (ROOT / ".github/workflows" / workflow).read_text()
    block = text.split(f"  {job}:\n", 1)[1].split("        run: |\n", 1)[1]
    return "\n".join(line[10:] for line in block.split(after, 1)[0].splitlines())


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

    def git(self, *args, who=PERSON):
        name, email = who
        env = dict(self.env, GIT_AUTHOR_NAME=name, GIT_AUTHOR_EMAIL=email,
                   GIT_COMMITTER_NAME=name, GIT_COMMITTER_EMAIL=email)
        return subprocess.run(["git", *args], cwd=self.repo, env=env, check=True,
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

    def commit(self, branch, change, who=PERSON):
        self.git("checkout", "-q", "-B", branch, self.base)
        change()
        self.git("add", "-A", "--", ".", ":(exclude)vendor")
        self.git("commit", "-q", "-m", "change", who=who)
        return self.git("rev-parse", "HEAD")

    def run_step(self, script, **env):
        runner = self.root / "runner"
        runner.mkdir(exist_ok=True)
        return subprocess.run(["bash", "-e", "-c", script], cwd=self.repo, capture_output=True,
                              text=True, check=False,
                              env={**self.env, "BASE": self.base, "RUNNER_TEMP": str(runner),
                                   **env})

    def guard(self, branch, change, who=PERSON):
        head = self.commit(branch, change, who)
        return self.run_step(step("test.yml", "pin-guard",
                                  "\n\n  contracts-and-install-lifecycle:"),
                             HEAD=head, BRANCH=branch)

    def integrity(self, change):
        """Run pin-integrity as pull_request_target does: main checked out, the PR fetched."""
        head = self.commit("change", change)
        origin = self.root / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True, env=self.env)
        self.git("push", "-q", str(origin), "main", f"{head}:refs/pull/7/head")
        self.git("remote", "add", "origin", str(origin))
        self.git("checkout", "-q", "main")
        return self.run_step(step("pin-integrity.yml", "pin-integrity", "\0"),
                             HEAD=head, NUMBER="7")

    def test_moving_a_git_requirement_by_hand_fails(self):
        result = self.guard("fix/client", self.move_client)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("::error file=requirements-dev.txt::", result.stdout)

    def test_pin_bot_moves_the_client(self):
        result = self.guard("bot/pin-client-547b4b1", self.move_client, who=BOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_bot_branch_from_a_person_fails(self):
        result = self.guard("bot/pin-client-547b4b1", self.move_client)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_submodule_fails(self):
        def add_submodule():
            (self.repo / ".gitmodules").write_text(
                f'[submodule "vendor/client"]\n\tpath = vendor/client\n\turl = {CLIENT}\n')
            self.git("update-index", "--add", "--cacheinfo", f"160000,{COMMIT},vendor/client")
        result = self.guard("feat/vendor", add_submodule)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("::error file=vendor/client::", result.stdout)

    def test_dispatch_and_pin_bot_are_never_changed(self):
        def change_both():
            self.edit(".github/workflows/test.yml", "repos/cavi-ai/mlx-workbench/dispatches",
                      "repos/cavi-ai/other/dispatches")
            self.edit(".github/workflows/pin-client.yml", "refs/heads/main", "refs/heads/dev")
        result = self.guard("bot/pin-client-547b4b1", change_both, who=BOT)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("changes the pin-consumers job", result.stdout)
        self.assertIn("::error file=.github/workflows/pin-client.yml::", result.stdout)

    def test_main_guard_judges_a_rewritten_guard(self):
        def disable_guard():
            (self.repo / "scripts/pin_guard.py").write_text("import sys\nsys.exit(0)\n")
            self.move_client()
        result = self.guard("fix/client", disable_guard)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("::error file=scripts/pin_guard.py::", result.stdout)
        self.assertIn("::error file=requirements-dev.txt::", result.stdout)

    def test_integrity_fails_a_gutted_guard_job(self):
        result = self.integrity(lambda: self.edit(
            ".github/workflows/test.yml", 'python3 "$RUNNER_TEMP/pin_guard.py"', "true #"))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("changes the pin-guard job", result.stdout)

    def test_integrity_passes_an_unrelated_change(self):
        result = self.integrity(lambda: self.edit(
            ".github/workflows/test.yml", "timeout-minutes: 60", "timeout-minutes: 70"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_version_range_and_other_jobs_pass(self):
        def update():
            self.edit("requirements-dev.txt", "coverage>=7.16,<8", "coverage>=7.17,<8")
            self.edit(".github/workflows/test.yml", "timeout-minutes: 60", "timeout-minutes: 70")
        result = self.guard("ci/coverage", update)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
