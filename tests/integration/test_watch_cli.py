import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "scout_responses.json"


class WatchCliTests(unittest.TestCase):
    def setUp(self):
        os.environ["MLX_AGENT_FIXTURE"] = str(FIXTURE)
        self.addCleanup(lambda: os.environ.pop("MLX_AGENT_FIXTURE", None))

    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_diff_without_baseline(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "watch", "diff", "--state-dir", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "missing_baseline")

    def test_snapshot_then_diff_is_quiet(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "watch", "snapshot", "--state-dir", directory, "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertGreater(payload["data"]["candidates"], 0)
            state = Path(payload["data"]["state"])
            self.assertTrue(state.is_file())

            code, output = self._run([
                "watch", "diff", "--state-dir", directory, "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["data"]["findings"], [])

    def test_second_snapshot_rotates_previous(self):
        with TemporaryDirectory() as directory:
            for _ in range(2):
                code, output = self._run([
                    "watch", "snapshot", "--state-dir", directory, "--json",
                ])
                self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertTrue(payload["data"]["rotated_previous"])


if __name__ == "__main__":
    unittest.main()
