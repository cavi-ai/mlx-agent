import io
import json
import os
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "scout_responses.json"


class WatchCliTests(unittest.TestCase):
    def setUp(self):
        self.owned_patch = patch("mlx_agent.watch.collect_owned", return_value=[])
        self.owned = self.owned_patch.start()
        self.addCleanup(self.owned_patch.stop)
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

    def test_scope_flags_forward_exact_roots_and_explicit_repositories(self):
        with TemporaryDirectory() as directory:
            client = str(Path(directory) / "client.json")
            receipts = str(Path(directory) / "receipts")
            code, output = self._run(["watch", "snapshot", "--state-dir", directory,
                "--wired-root", client, "--wired-root", receipts, "--no-hf-cache",
                "--owned-repo", "managed/model", "--owned-repo", "managed/other", "--json"])
        self.assertEqual(code, 0, output)
        kwargs = self.owned.call_args.kwargs
        self.assertEqual(kwargs["wired_roots"], [client, receipts])
        self.assertFalse(kwargs["include_hf_cache"])
        self.assertEqual(kwargs["owned_repos"], ["managed/model", "managed/other"])

    def test_selected_files_and_managed_repos_work_from_outside_project(self):
        self.owned_patch.stop()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client = root / "opencode.json"
            client.write_text(json.dumps({"mlx_agent_wire": {"marker": "MLX_AGENT_WIRE", "version": "2.0",
                "runtime": "mlx_lm", "model": "wired/model", "provider": {"base_url": "http://127.0.0.1:8080/v1"}}}))
            excluded = root / "unselected"; excluded.mkdir()
            (excluded / "opencode.json").write_text(client.read_text().replace("wired/model", "excluded/model"))
            state = root / "state"
            with patch("mlx_agent.cli._watch_runtime_clients", return_value=[]), \
                 patch("mlx_agent.watch.default_hf_cache") as default, \
                 patch("mlx_agent.watch.inspect_hf_cache") as cache, \
                 patch("mlx_agent.watch.Path.cwd", return_value=Path("/")):
                code, output = self._run(["watch", "snapshot", "--state-dir", str(state),
                    "--wired-root", str(client), "--no-hf-cache", "--owned-repo", "managed/model", "--json"])
            default.assert_not_called()
            cache.assert_not_called()
            self.assertEqual(code, 0, output)
            payload = json.loads(output)
            saved = json.loads(Path(payload["data"]["state"]).read_text())
            self.assertEqual({item["id"] for item in saved["owned"]}, {"wired/model", "managed/model"})
            self.assertEqual(saved["scope"]["wired_roots"], [str(client.resolve())])
            self.assertEqual(payload["data"]["scope"], saved["scope"])
            self.assertEqual(saved["schema_version"], "1.0")

    def test_malformed_owned_repository_fails_without_discovery_or_scan(self):
        with patch("mlx_agent.cli._discovery_service_from_environment") as discovery:
            code, output = self._run(["watch", "snapshot", "--owned-repo", "org/model/checkpoint", "--json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output)["error"]["code"], "invalid_owned_repo")
        discovery.assert_not_called()
        self.owned.assert_not_called()

    def test_cache_selection_flags_are_mutually_exclusive(self):
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as caught:
            main(["watch", "snapshot", "--hf-cache", "/selected/cache", "--no-hf-cache"])
        self.assertEqual(caught.exception.code, 2)
        self.owned.assert_not_called()

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
