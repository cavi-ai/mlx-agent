import io
import json
import subprocess
import sys
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mlx_agent import lab
from mlx_agent.compute import ComputeBusy


def _completed(stdout="", stderr="", code=0):
    return subprocess.CompletedProcess(["engine"], code, stdout, stderr)


@contextmanager
def _no_lease(_name):
    yield None


class LabRunnerTests(unittest.TestCase):
    def run_lab(self, argv, child=None, legacy=False):
        output, errors = io.StringIO(), io.StringIO()
        with patch("mlx_agent.lab._run_child", side_effect=child) as spawned, \
                patch("mlx_agent.compute.ComputeLease", _no_lease), \
                redirect_stdout(output), redirect_stderr(errors):
            code = lab.run(argv, legacy=legacy)
        return code, output.getvalue(), errors.getvalue(), spawned

    def test_operation_skips_option_values(self):
        self.assertEqual("experiment", lab._operation(["--database", "db", "--artifacts=a", "--json", "experiment", "list"]))
        self.assertIsNone(lab._operation(["--reviewer-id", "r", "--json"]))

    def test_state_root_follows_the_platform(self):
        with patch.object(lab.sys, "platform", "darwin"):
            self.assertEqual(Path.home() / "Library/Application Support/mlx-agent/lab", lab.state_root())
        with patch.object(lab.sys, "platform", "linux"), patch.dict(lab.os.environ, {"XDG_STATE_HOME": "/state"}):
            self.assertEqual(Path("/state/mlx-agent/lab"), lab.state_root())

    def test_run_routes_operations_and_fills_default_stores(self):
        for argv, module in ((["workspace", "--json"], "mlx_agent.lab_workspace"),
                             (["queue", "--json"], "mlx_agent.queue_cli"),
                             (["training-data", "--json"], "video_lab.training_data_cli"),
                             (["experiment", "list", "--json"], "video_lab.cli")):
            with self.subTest(argv=argv):
                _code, _out, _err, spawned = self.run_lab(argv, lambda *a, **k: _completed('{"ok": true, "data": {}}'))
                command, env = spawned.call_args.args[:2]
                self.assertEqual([sys.executable, "-m", module], command[:3])
                self.assertIn("--database", command)
                self.assertIn("--artifacts", command)
                self.assertTrue(env["PYTHONPATH"].startswith(str(lab.ENGINE_ROOT)))
        _code, _out, _err, spawned = self.run_lab([], lambda *a, **k: _completed("usage"))
        self.assertIn("--help", spawned.call_args.args[0])

    def test_json_results_become_lab_envelopes(self):
        cases = (
            ('{"ok": true, "schema_version": "e/1", "data": [{"id": 1}]}', 0, ["experiment", "list", "--json"],
             0, {"schema": "lab/1", "engine_schema": "e/1", "status": "ok", "data": {"experiments": [{"id": 1}]}}),
            ('{"status": "ok", "data": [1]}', 0, ["queue", "list", "--json"],
             0, {"schema": "lab/1", "engine_schema": None, "status": "ok", "data": {"items": [1]}}),
            ('{"ok": false, "error": {"code": "x", "message": "y"}}', 3, ["experiment", "show", "--json"],
             3, {"schema": "lab/1", "engine_schema": None, "status": "error", "data": None, "error": {"code": "x", "message": "y"}}),
            ('{"ok": true}', 4, ["experiment", "show", "--json"],
             4, {"schema": "lab/1", "engine_schema": None, "status": "error", "data": None,
                 "error": {"code": "lab_failed", "message": "Experiment operation failed."}}),
        )
        for stdout, returncode, argv, code, envelope in cases:
            with self.subTest(stdout=stdout):
                result = self.run_lab(argv, lambda *a, **k: _completed(stdout, code=returncode))
                self.assertEqual(code, result[0])
                self.assertEqual(envelope, json.loads(result[1]))

    def test_unusable_engine_output_is_reported_not_passed_through(self):
        for stdout, code in (("not json", "lab_unavailable"), ("[1, 2]", "lab_invalid_output")):
            with self.subTest(stdout=stdout):
                result = self.run_lab(["experiment", "list", "--json"], lambda *a, **k: _completed(stdout))
                self.assertEqual(2, result[0])
                self.assertEqual(code, json.loads(result[1])["error"]["code"])
        with patch.object(lab, "MAX_OUTPUT_BYTES", 4):
            result = self.run_lab(["experiment", "list", "--json"], lambda *a, **k: _completed('{"ok": true}'))
        self.assertEqual("lab_output_limit", json.loads(result[1])["error"]["code"])

    def test_engine_failures_map_to_error_codes(self):
        for error, code in ((OverflowError("too much"), "lab_output_limit"), (TimeoutError(), "lab_timeout"),
                            (subprocess.TimeoutExpired("engine", 1), "lab_timeout"), (OSError("missing"), "lab_unavailable")):
            with self.subTest(error=error):
                result = self.run_lab(["experiment", "list", "--json"], error)
                self.assertEqual(2, result[0])
                self.assertEqual(code, json.loads(result[1])["error"]["code"])

    def test_plain_and_legacy_output_pass_through_with_the_exit_status(self):
        for argv, legacy in ((["experiment", "list"], False), (["experiment", "list", "--json"], True)):
            with self.subTest(argv=argv, legacy=legacy):
                code, out, err, _spawned = self.run_lab(argv, lambda *a, **k: _completed("rows\n", "note\n", 5), legacy=legacy)
                self.assertEqual((5, "rows\n", "note\n"), (code, out, err))

    def test_heavy_operations_report_a_busy_compute_lease(self):
        def busy(_name):
            raise ComputeBusy("owned by another task")
        output = io.StringIO()
        with patch("mlx_agent.compute.ComputeLease", busy), patch("mlx_agent.lab._run_child") as spawned, redirect_stdout(output):
            self.assertEqual(2, lab.run(["worker", "run-once", "--json"]))
            self.assertEqual(2, lab.worker_main(["run-once"]))
        spawned.assert_not_called()
        self.assertEqual(["compute_busy", "compute_busy"],
                         [json.loads(line)["error"]["code"] for line in output.getvalue().splitlines()])

    def test_worker_passes_engine_output_through(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch("mlx_agent.compute.ComputeLease", _no_lease), \
                patch("mlx_agent.lab._run_child", return_value=_completed("done\n", "warn\n", 0)) as spawned, \
                redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(0, lab.worker_main(["--help"]))
        self.assertEqual([sys.executable, "-m", "video_lab.worker", "--help"], spawned.call_args.args[0])
        self.assertEqual(("done\n", "warn\n"), (output.getvalue(), errors.getvalue()))

    def test_child_output_is_collected_bounded_and_timed(self):
        script = "import sys; sys.stdout.write('out'); sys.stderr.write('err')"
        result = lab._run_child([sys.executable, "-c", script], None)
        self.assertEqual((0, "out", "err"), (result.returncode, result.stdout, result.stderr))
        with patch.object(lab, "MAX_OUTPUT_BYTES", 16), self.assertRaises(OverflowError):
            lab._run_child([sys.executable, "-c", "print('x' * 64)"], None)
        with self.assertRaises(TimeoutError):
            lab._run_child([sys.executable, "-c", "import time; time.sleep(30)"], None, timeout=0.5)


if __name__ == "__main__":
    unittest.main()
