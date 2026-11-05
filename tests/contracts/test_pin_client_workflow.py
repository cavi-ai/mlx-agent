"""Execute the client pin step against local repositories."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ClientPinTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "gitconfig").write_text(
            "[user]\n\tname = Pin Test\n\temail = pin@example.invalid\n")
        self.env = dict(os.environ, GIT_CONFIG_GLOBAL=str(self.root / "gitconfig"),
                        GIT_CONFIG_NOSYSTEM="1")

    def git(self, *args, cwd):
        return subprocess.run(["git", *args], cwd=cwd, env=self.env, check=True,
                              capture_output=True, text=True).stdout.strip()

    def run_sync(self, pinned):
        client = self.root / "client"
        client.mkdir()
        self.git("init", "-q", "-b", "main", cwd=client)
        commits = []
        for name in ("first", "second"):
            (client / name).write_text(name)
            self.git("add", name, cwd=client)
            self.git("commit", "-q", "-m", name, cwd=client)
            commits.append(self.git("rev-parse", "HEAD", cwd=client))
        consumer = self.root / "consumer"
        consumer.mkdir()
        lines = (ROOT / "requirements-dev.txt").read_text().splitlines()
        (consumer / "requirements-dev.txt").write_text("\n".join(
            f"cavi-api-client @ git+https://example.invalid/client.git@{commits[pinned]}"
            "#subdirectory=python" if line.startswith("cavi-api-client") else line
            for line in lines) + "\n")
        workflow = (ROOT / ".github/workflows/pin-client.yml").read_text()
        block = workflow.split("      - id: sync\n", 1)[1].split("        run: |\n", 1)[1]
        script = "\n".join(line[10:] for line in block.split("\n      - ", 1)[0].splitlines())
        output = self.root / "output"
        output.touch()
        result = subprocess.run(["bash", "-e", "-c", script], cwd=consumer, capture_output=True,
                                text=True, check=False,
                                env=dict(self.env, GITHUB_OUTPUT=str(output), CLIENT=str(client)))
        self.assertEqual(result.returncode, 0, result.stderr)
        return commits, output.read_text(), (consumer / "requirements-dev.txt").read_text()

    def test_sync_moves_the_client_to_main_and_names_the_pin(self):
        commits, output, requirements = self.run_sync(pinned=0)
        self.assertIn(f"client.git@{commits[1]}#subdirectory=python", requirements)
        self.assertNotIn(commits[0], requirements)
        self.assertIn("coverage>=", requirements)
        self.assertEqual(output, f"branch=bot/pin-client-{commits[1][:7]}\n"
                                 f"body=- cavi-api-client {commits[0][:7]} -> {commits[1][:7]}\n")

    def test_sync_proposes_nothing_when_the_client_is_on_main(self):
        commits, output, requirements = self.run_sync(pinned=1)
        self.assertIn(f"client.git@{commits[1]}#subdirectory=python", requirements)
        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
