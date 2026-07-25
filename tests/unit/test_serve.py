import json
import signal
import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.serve import (
    ServeError,
    load_recipes,
    plan_start,
    start_serve,
    status_serve,
    stop_serve,
    wired_port_claim,
)


RECIPES = load_recipes()


def _plan(**overrides):
    values = {"repo": "pub/model", "runtime": "mlx_lm"}
    values.update(overrides)
    return plan_start(values.pop("repo"), values.pop("runtime"), RECIPES, **values)


class PlanStartTests(unittest.TestCase):
    def test_default_plan(self):
        plan = _plan()
        self.assertEqual(plan["port"], 8080)
        self.assertEqual(
            plan["argv"],
            ["mlx_lm.server", "--model", "pub/model", "--port", "8080", "--max-tokens", "8192"],
        )
        self.assertEqual(plan["readiness"], "http://127.0.0.1:8080/v1/models")
        self.assertEqual(plan["bind"], "127.0.0.1")
        self.assertEqual(len(plan["preview_hash"]), 64)

    def test_preview_hash_changes_with_plan(self):
        self.assertNotEqual(_plan()["preview_hash"], _plan(port=9090)["preview_hash"])

    def test_vlm_recipe(self):
        plan = _plan(runtime="mlx-vlm")
        self.assertEqual(plan["port"], 8083)
        self.assertEqual(plan["argv"][0], "mlx_vlm.server")

    def test_adapter_path_appends_adapter_argv(self):
        plan = _plan(adapter_path="/tmp/adapters")
        self.assertEqual(plan["argv"][-2:], ["--adapter-path", "/tmp/adapters"])

    def test_adapter_rejected_for_vlm(self):
        with self.assertRaises(ServeError) as caught:
            _plan(runtime="mlx-vlm", adapter_path="/tmp/adapters")
        self.assertEqual(caught.exception.code, "unsupported_runtime")

    def test_unsupported_runtime(self):
        with self.assertRaises(ServeError) as caught:
            plan_start("pub/model", "ollama", RECIPES)
        self.assertEqual(caught.exception.code, "unsupported_runtime")

    def test_invalid_repo(self):
        with self.assertRaises(ServeError) as caught:
            plan_start("noslash", "mlx_lm", RECIPES)
        self.assertEqual(caught.exception.code, "invalid_repo")

    def test_port_bounds(self):
        with self.assertRaises(ServeError):
            _plan(port=0)
        with self.assertRaises(ServeError):
            _plan(port=70000)


class StartServeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.spawned = []

    def _spawn(self, argv, log_path):
        self.spawned.append((argv, log_path))
        return 4242

    def _start(self, plan, **overrides):
        kwargs = {
            "receipts_dir": str(self.root),
            "confirm": True,
            "preview_hash": plan["preview_hash"],
            "which": lambda executable: "/usr/local/bin/" + executable,
            "model_present": lambda repo: True,
            "port_free": lambda port: True,
            "spawn": self._spawn,
            "readiness": lambda url, deadline: True,
            "now": lambda: "2026-07-25T00:00:00+00:00",
        }
        kwargs.update(overrides)
        return start_serve(plan, **kwargs)

    def test_preview_without_confirm(self):
        outcome = start_serve(_plan(), receipts_dir=str(self.root))
        self.assertEqual(outcome["status"], "preview")
        self.assertTrue(outcome["requires_confirmation"])
        self.assertEqual(self.spawned, [])

    def test_confirm_requires_preview_hash(self):
        with self.assertRaises(ServeError) as caught:
            self._start(_plan(), preview_hash=None)
        self.assertEqual(caught.exception.code, "preview_hash_required")

    def test_stale_preview_hash_is_rejected(self):
        with self.assertRaises(ServeError) as caught:
            self._start(_plan(), preview_hash="0" * 64)
        self.assertEqual(caught.exception.code, "preview_stale")
        self.assertEqual(self.spawned, [])

    def test_missing_runtime_executable(self):
        with self.assertRaises(ServeError) as caught:
            self._start(_plan(), which=lambda executable: None)
        self.assertEqual(caught.exception.code, "runtime_not_installed")

    def test_model_gate(self):
        with self.assertRaises(ServeError) as caught:
            self._start(_plan(), model_present=lambda repo: False)
        self.assertEqual(caught.exception.code, "model_not_local")

    def test_port_gate(self):
        with self.assertRaises(ServeError) as caught:
            self._start(_plan(), port_free=lambda port: False)
        self.assertEqual(caught.exception.code, "port_in_use")

    def test_wired_claim_gate(self):
        with self.assertRaises(ServeError) as caught:
            self._start(_plan(), wired_claims=lambda port, runtime: True)
        self.assertEqual(caught.exception.code, "port_in_use")

    def test_missing_adapter_path(self):
        with self.assertRaises(ServeError) as caught:
            self._start(_plan(adapter_path="/tmp/definitely-missing-adapters"))
        self.assertEqual(caught.exception.code, "invalid_arguments")

    def test_readiness_timeout_terminates_child(self):
        terminated = []

        def terminate(pid, sig=signal.SIGTERM):
            terminated.append(pid)
            return True

        with self.assertRaises(ServeError) as caught:
            with unittest.mock.patch("mlx_agent.serve._terminate_pid", terminate):
                self._start(_plan(), readiness=lambda url, deadline: False)
        self.assertEqual(caught.exception.code, "readiness_timeout")
        self.assertEqual(terminated, [4242])

    def test_success_writes_receipt(self):
        outcome = self._start(_plan())
        self.assertEqual(outcome["status"], "started")
        receipt = outcome["receipt"]
        self.assertEqual(receipt["pid"], 4242)
        self.assertEqual(receipt["kind"], "serve")
        receipt_path = self.root / ".mlx-agent-receipts" / "serve" / "8080.json"
        self.assertTrue(receipt_path.is_file())
        persisted = json.loads(receipt_path.read_text())
        self.assertEqual(persisted["argv"], receipt["argv"])

    def test_live_receipt_blocks_second_start(self):
        self._start(_plan(), pid_alive=lambda pid: True)
        with self.assertRaises(ServeError) as caught:
            self._start(_plan(), pid_alive=lambda pid: True)
        self.assertEqual(caught.exception.code, "port_in_use")


class StatusAndStopTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.spawned = []

    def _start_one(self):
        plan = _plan()
        return start_serve(
            plan,
            receipts_dir=str(self.root),
            confirm=True,
            preview_hash=plan["preview_hash"],
            which=lambda executable: "/bin/" + executable,
            spawn=lambda argv, log_path: 4242,
            readiness=lambda url, deadline: True,
            now=lambda: "2026-07-25T00:00:00+00:00",
        )

    _COMMAND = "mlx_lm.server --model pub/model --port 8080 --max-tokens 8192"

    def test_status_reports_live_server(self):
        self._start_one()
        entries = status_serve(
            str(self.root),
            pid_alive=lambda pid: True,
            pid_command=lambda pid: self._COMMAND,
        )
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["alive"])
        self.assertTrue(entries[0]["argv_match"])

    def test_status_marks_dead_server(self):
        self._start_one()
        entries = status_serve(str(self.root), pid_alive=lambda pid: False)
        self.assertFalse(entries[0]["alive"])
        self.assertFalse(entries[0]["argv_match"])

    def test_stop_requires_receipt(self):
        with self.assertRaises(ServeError) as caught:
            stop_serve(8080, str(self.root))
        self.assertEqual(caught.exception.code, "receipt_not_found")

    def test_stop_refuses_argv_mismatch(self):
        self._start_one()
        with self.assertRaises(ServeError) as caught:
            stop_serve(
                8080, str(self.root),
                pid_alive=lambda pid: True,
                pid_command=lambda pid: "some-other-server --port 9999",
            )
        self.assertEqual(caught.exception.code, "pid_argv_mismatch")
        receipt = self.root / ".mlx-agent-receipts" / "serve" / "8080.json"
        self.assertTrue(receipt.is_file())

    def test_stop_terminates_matching_process(self):
        self._start_one()
        signals = []

        def terminate(pid, sig=signal.SIGTERM):
            signals.append(sig)
            return True

        alive = {"value": True}

        def pid_alive(pid):
            return alive["value"]

        def terminate_and_kill(pid, sig=signal.SIGTERM):
            signals.append(sig)
            alive["value"] = False
            return True

        outcome = stop_serve(
            8080, str(self.root),
            pid_alive=pid_alive,
            pid_command=lambda pid: self._COMMAND,
            terminate=terminate_and_kill,
        )
        self.assertEqual(outcome["status"], "stopped")
        self.assertEqual(signals, [signal.SIGTERM])
        receipt = self.root / ".mlx-agent-receipts" / "serve" / "8080.json"
        self.assertFalse(receipt.exists())

    def test_stop_reports_already_stopped(self):
        self._start_one()
        outcome = stop_serve(8080, str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(outcome["status"], "already_stopped")


class WiredPortClaimTests(unittest.TestCase):
    def test_claim_detects_other_runtime_on_port(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wired.json").write_text(json.dumps({
                "mlx_agent_wire": {
                    "marker": "MLX_AGENT_WIRE",
                    "runtime": "lmstudio",
                    "model": "pub/model",
                    "provider": {"base_url": "http://127.0.0.1:8080/v1"},
                }
            }))
            claims = wired_port_claim([root])
            self.assertTrue(claims(8080, "mlx_lm"))
            self.assertFalse(claims(8080, "lmstudio"))
            self.assertFalse(claims(9999, "mlx_lm"))


if __name__ == "__main__":
    unittest.main()
