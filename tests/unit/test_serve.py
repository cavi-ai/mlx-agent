import json
import hashlib
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


def _adapter(root):
    root.mkdir()
    (root / "adapter_config.json").write_text("{}")
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (root / "adapters.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"1234")
    return root


def _plan(**overrides):
    values = {"repo": "pub/model", "runtime": "mlx_lm"}
    values.update(overrides)
    return plan_start(values.pop("repo"), values.pop("runtime"), RECIPES, **values)


class PlanStartTests(unittest.TestCase):
    def test_local_vq_evidence_is_reviewed_without_treating_default_bits_as_uniform(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "config.json"
            config.write_text(json.dumps({"quantization": {"bits": 8, "mode": "affine", "layer": {"bits": 4}},
                                          "vq_modules": {"expert": {"bits": 2}}}))
            (root / "model.safetensors").write_bytes(b"local")
            plan = _plan(repo=None, path=str(root), runtime="mlx-vlm")
            self.assertEqual(plan["weight_format"], "safetensors")
            self.assertEqual(plan["weight_quantization"], "VQ + per-tensor affine")
            self.assertNotIn("weights_preserved", plan)
            config.write_text(json.dumps({"quantization": {"bits": 4, "mode": "affine"}}))
            changed = _plan(repo=None, path=str(root), runtime="mlx-vlm")
            self.assertEqual(changed["weight_quantization"], "affine 4-bit")
            self.assertNotEqual(changed["preview_hash"], plan["preview_hash"])

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

    def test_thinking_off_is_a_reviewed_chat_template_argument(self):
        default = _plan()
        off = _plan(thinking="off")
        self.assertNotIn("thinking", default)
        self.assertEqual(off["thinking"], "off")
        self.assertEqual(off["argv"][-2:], ["--chat-template-args", '{"enable_thinking":false}'])
        self.assertNotEqual(default["preview_hash"], off["preview_hash"])
        vlm = _plan(runtime="mlx-vlm", thinking="off")
        self.assertEqual(vlm["thinking"], "unsupported")
        self.assertNotIn("--chat-template-args", vlm["argv"])
        with self.assertRaises(ServeError) as caught:
            _plan(thinking="maybe")
        self.assertEqual(caught.exception.code, "invalid_arguments")

    def test_preview_hash_changes_with_plan(self):
        self.assertNotEqual(_plan()["preview_hash"], _plan(port=9090)["preview_hash"])

    def test_vlm_recipe(self):
        plan = _plan(runtime="mlx-vlm")
        self.assertEqual(plan["port"], 8083)
        self.assertEqual(plan["argv"][0], "mlx_vlm.server")

    def test_adapter_path_appends_adapter_argv(self):
        with TemporaryDirectory() as folder:
            adapter = _adapter(Path(folder) / "adapter")
            plan = _plan(adapter_path=str(adapter))
            self.assertEqual(plan["argv"][-2:], ["--adapter-path", str(adapter)])

    def test_adapter_content_changes_the_reviewed_hash(self):
        with TemporaryDirectory() as folder:
            adapter = _adapter(Path(folder) / "adapter")
            first = _plan(adapter_path=str(adapter))
            self.assertIn("adapters.safetensors", first["adapter_identity"])
            (adapter / "adapter_config.json").write_text('{"rank":4}')
            second = _plan(adapter_path=str(adapter))
            self.assertNotEqual(first["preview_hash"], second["preview_hash"])
            tensor = adapter / "adapters.safetensors"
            tensor.write_bytes(tensor.read_bytes()[:-4] + b"5678")
            self.assertNotEqual(second["preview_hash"], _plan(adapter_path=str(adapter))["preview_hash"])

    def test_invalid_and_linked_adapters_are_refused_at_preview(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            adapter = _adapter(root / "adapter")
            link = root / "linked"; link.symlink_to(adapter, target_is_directory=True)
            for path in [root, root / "missing", link]:
                with self.assertRaises(ServeError) as caught:
                    _plan(adapter_path=str(path))
                self.assertEqual(caught.exception.code, "invalid_adapter")

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

    def test_path_plan(self):
        plan = _plan(repo=None, path="/models/foo-MLX-4bit")
        self.assertIsNone(plan["repo"])
        self.assertEqual(plan["path"], "/models/foo-MLX-4bit")
        self.assertEqual(
            plan["argv"],
            ["mlx_lm.server", "--model", "/models/foo-MLX-4bit", "--port", "8080", "--max-tokens", "8192"],
        )

    def test_relative_path_is_absolutized(self):
        plan = _plan(repo=None, path="models/foo")
        self.assertTrue(plan["path"].startswith("/"))
        self.assertTrue(plan["path"].endswith("models/foo"))

    def test_repo_and_path_together_are_rejected(self):
        with self.assertRaises(ServeError) as caught:
            plan_start("pub/model", "mlx_lm", RECIPES, path="/models/foo")
        self.assertEqual(caught.exception.code, "invalid_arguments")

    def test_neither_repo_nor_path_is_rejected(self):
        with self.assertRaises(ServeError) as caught:
            plan_start(None, "mlx_lm", RECIPES)
        self.assertEqual(caught.exception.code, "invalid_arguments")

    def test_path_plan_hash_differs_from_repo_plan(self):
        self.assertNotEqual(
            _plan()["preview_hash"],
            _plan(repo=None, path="/pub/model")["preview_hash"],
        )

    def test_port_bounds(self):
        with self.assertRaises(ServeError):
            _plan(port=0)
        with self.assertRaises(ServeError):
            _plan(port=70000)


class TrainingReceiptServeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "base"; self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"llama"}')
        (self.model / "model.safetensors").write_bytes(b"local-base")
        self.adapter = _adapter(self.root / "adapter")
        from mlx_agent.lora import adapter_identity
        from mlx_agent.jit_serve import model_fingerprint
        self.receipt = self.root / "training.json"
        self.record = {"kind": "lora", "pid": 9, "argv": [], "repo": "pub/model", "run_id": "run-a",
            "out": str(self.adapter), "exit_status": "done", "adapter_identity": adapter_identity(self.adapter),
            "base_identity": {"path": str(self.model), "fingerprint": model_fingerprint(self.model)}}
        self.receipt.write_text(json.dumps(self.record))

    def plan(self, **options):
        return _plan(repo=None, training_receipt=str(self.receipt), **options)

    def test_completed_receipt_selects_the_pinned_base_and_adapter(self):
        plan = self.plan()
        self.assertIsNone(plan["repo"])
        self.assertEqual(plan["path"], str(self.model))
        self.assertEqual(plan["adapter_path"], str(self.adapter))
        binding = plan["training_binding"]
        self.assertEqual(binding["training_receipt"], str(self.receipt))
        self.assertEqual(binding["training_receipt_sha256"], hashlib.sha256(self.receipt.read_bytes()).hexdigest())
        self.assertEqual(binding["adapter_identity"], plan["adapter_identity"])
        self.assertEqual(binding["base_fingerprint"], self.record["base_identity"]["fingerprint"])

    def test_evaluation_and_serving_use_the_same_receipt_artifacts(self):
        from mlx_agent.training_evaluation import preview
        evaluation = preview(str(self.receipt), "A controlled prompt")
        serving = self.plan()["training_binding"]
        for key in ["training_receipt", "training_receipt_sha256", "base_path", "adapter_path", "adapter_identity"]:
            self.assertEqual(evaluation[key], serving[key])

    def test_receipt_cannot_be_combined_with_other_model_or_adapter_inputs(self):
        for options in [{"repo": "other/model"}, {"path": str(self.model)}, {"adapter_path": str(self.adapter)}]:
            with self.assertRaises(ServeError) as caught:
                plan_start(options.pop("repo", None), "mlx_lm", RECIPES, training_receipt=str(self.receipt), **options)
            self.assertEqual(caught.exception.code, "invalid_arguments")

    def test_unfinished_historical_or_changed_artifacts_cannot_be_selected(self):
        for change in [{"exit_status": "running"}, {"base_identity": None}, {"adapter_identity": {}}]:
            self.receipt.write_text(json.dumps(dict(self.record, **change)))
            with self.assertRaises(ServeError) as caught:
                self.plan()
            self.assertEqual(caught.exception.code, "invalid_training_receipt")
        self.receipt.write_text(json.dumps(self.record))
        (self.model / "config.json").write_text('{"changed":true}')
        with self.assertRaises(ServeError): self.plan()

    def test_changed_receipt_revokes_a_reviewed_plan_before_runtime_checks(self):
        plan = self.plan()
        self.receipt.write_text(json.dumps(dict(self.record, run_id="changed")))
        with self.assertRaises(ServeError) as caught:
            start_serve(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan["preview_hash"], which=lambda _: None)
        self.assertEqual(caught.exception.code, "training_receipt_changed")
        self.assertFalse((self.root / ".mlx-agent-receipts").exists())

    def test_changed_base_revokes_a_reviewed_plan(self):
        plan = self.plan()
        (self.model / "config.json").write_text('{"changed":true}')
        with self.assertRaises(ServeError) as caught:
            start_serve(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan["preview_hash"], which=lambda _: None)
        self.assertEqual(caught.exception.code, "training_receipt_changed")

    def test_training_binding_is_retained_by_jit_config_receipt_and_status(self):
        plan = self.plan(jit=True, receipts_dir=str(self.root))
        outcome = start_serve(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan["preview_hash"],
            which=lambda _: "/python", port_free=lambda _: True, spawn=lambda *_: 4242, readiness=lambda *_: True)
        receipt = outcome["receipt"]
        config = json.loads(Path(plan["control_config"]).read_text())
        status = status_serve(str(self.root), pid_alive=lambda _: False)[0]
        for record in [receipt, config, status]:
            self.assertEqual(record["training_binding"], plan["training_binding"])

    def test_receipt_changes_during_startup_do_not_publish_a_server(self):
        plan = self.plan()
        def ready(*_):
            self.receipt.write_text(json.dumps(dict(self.record, run_id="changed")))
            return True
        with unittest.mock.patch("mlx_agent.serve._terminate_pid") as stop:
            with self.assertRaises(ServeError) as caught:
                start_serve(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan["preview_hash"],
                    which=lambda _: "/python", port_free=lambda _: True, spawn=lambda *_: 4242, readiness=ready)
        self.assertEqual(caught.exception.code, "training_receipt_changed")
        stop.assert_called_once_with(4242)
        self.assertFalse((self.root / ".mlx-agent-receipts/serve/8080.json").exists())


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

    def test_path_gate_rejects_a_missing_directory(self):
        with self.assertRaises(ServeError) as caught:
            self._start(_plan(repo=None, path="/definitely/not/a/model"))
        self.assertEqual(caught.exception.code, "model_not_local")

    def test_path_gate_via_injection(self):
        with self.assertRaises(ServeError) as caught:
            self._start(_plan(repo=None, path="/models/foo"), path_present=lambda path: False)
        self.assertEqual(caught.exception.code, "model_not_local")

    def test_repo_gate_is_not_applied_to_path_plans(self):
        model_dir = self.root / "converted"
        model_dir.mkdir()
        outcome = self._start(
            _plan(repo=None, path=str(model_dir)),
            model_present=lambda repo: False,
        )
        self.assertEqual(outcome["status"], "started")

    def test_path_success_writes_receipt(self):
        model_dir = self.root / "converted"
        model_dir.mkdir()
        outcome = self._start(_plan(repo=None, path=str(model_dir)))
        self.assertEqual(outcome["status"], "started")
        receipt = outcome["receipt"]
        self.assertIsNone(receipt["repo"])
        self.assertEqual(receipt["path"], str(model_dir))
        self.assertIn(str(model_dir), receipt["argv"])

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
            _plan(adapter_path=str(self.root / "missing-adapter"))
        self.assertEqual(caught.exception.code, "invalid_adapter")

    def test_adapter_drift_refuses_confirmation_before_spawn(self):
        adapter = _adapter(self.root / "adapter")
        plan = _plan(adapter_path=str(adapter))
        (adapter / "adapter_config.json").write_text('{"rank":8}')
        with self.assertRaises(ServeError) as caught:
            self._start(plan)
        self.assertEqual(caught.exception.code, "adapter_changed")
        self.assertEqual(self.spawned, [])

    def test_adapter_drift_during_load_stops_only_the_started_process(self):
        adapter = _adapter(self.root / "adapter")
        plan = _plan(adapter_path=str(adapter))
        def ready(*_):
            (adapter / "adapter_config.json").write_text('{"rank":8}')
            return True
        with unittest.mock.patch("mlx_agent.serve._terminate_pid") as stop:
            with self.assertRaises(ServeError) as caught:
                self._start(plan, readiness=ready)
        self.assertEqual(caught.exception.code, "adapter_changed")
        stop.assert_called_once_with(4242)
        self.assertFalse((self.root / ".mlx-agent-receipts/serve/8080.json").exists())

    def test_adapter_identity_survives_receipts_status_and_jit_config(self):
        adapter = _adapter(self.root / "adapter")
        model = self.root / "model"; model.mkdir()
        (model / "model.safetensors").write_bytes(b"local")
        plan = _plan(repo=None, path=str(model), adapter_path=str(adapter), jit=True, receipts_dir=str(self.root))
        receipt = self._start(plan)["receipt"]
        config = json.loads(Path(plan["control_config"]).read_text())
        for record in [receipt, config]:
            self.assertEqual(record["adapter_path"], str(adapter))
            self.assertEqual(record["adapter_identity"], plan["adapter_identity"])
        status = status_serve(str(self.root), pid_alive=lambda _: False)[0]
        self.assertEqual(status["adapter_path"], str(adapter))
        self.assertEqual(status["adapter_identity"], plan["adapter_identity"])
        receipt_file = Path(status["receipt"])
        legacy = json.loads(receipt_file.read_text())
        legacy.pop("adapter_path"); legacy.pop("adapter_identity")
        receipt_file.write_text(json.dumps(legacy))
        legacy_status = status_serve(str(self.root), pid_alive=lambda _: False)[0]
        self.assertEqual(legacy_status["adapter_path"], str(adapter))
        self.assertIsNone(legacy_status["adapter_identity"])
        Path(plan["control_config"]).unlink()
        self.assertEqual(status_serve(str(self.root), pid_alive=lambda _: False)[0]["adapter_status"], "unknown")

    def test_eager_adapter_receipt_requires_adapter_arguments_in_the_process(self):
        adapter = _adapter(self.root / "adapter")
        plan = _plan(adapter_path=str(adapter))
        self._start(plan)
        command = " ".join(plan["argv"][:-2])
        self.assertFalse(status_serve(str(self.root), pid_alive=lambda _: True,
            pid_command=lambda _: command)[0]["argv_match"])
        command = " ".join(plan["argv"])
        self.assertTrue(status_serve(str(self.root), pid_alive=lambda _: True,
            pid_command=lambda _: command)[0]["argv_match"])

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

    def _start_path_one(self, model_dir):
        plan = _plan(repo=None, path=str(model_dir))
        return plan, start_serve(
            plan,
            receipts_dir=str(self.root),
            confirm=True,
            preview_hash=plan["preview_hash"],
            which=lambda executable: "/bin/" + executable,
            spawn=lambda argv, log_path: 4343,
            readiness=lambda url, deadline: True,
            now=lambda: "2026-07-25T00:00:00+00:00",
        )

    def test_status_includes_path_and_matches_argv(self):
        model_dir = self.root / "converted"
        model_dir.mkdir()
        self._start_path_one(model_dir)
        command = "mlx_lm.server --model {0} --port 8080 --max-tokens 8192".format(model_dir)
        entries = status_serve(
            str(self.root),
            pid_alive=lambda pid: True,
            pid_command=lambda pid: command,
        )
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["repo"])
        self.assertEqual(entries[0]["path"], str(model_dir))
        self.assertTrue(entries[0]["argv_match"])

    def test_stop_terminates_path_served_process(self):
        model_dir = self.root / "converted"
        model_dir.mkdir()
        self._start_path_one(model_dir)
        command = "mlx_lm.server --model {0} --port 8080 --max-tokens 8192".format(model_dir)
        alive = {"value": True}

        def terminate(pid, sig=signal.SIGTERM):
            alive["value"] = False
            return True

        outcome = stop_serve(
            8080, str(self.root),
            pid_alive=lambda pid: alive["value"],
            pid_command=lambda pid: command,
            terminate=terminate,
        )
        self.assertEqual(outcome["status"], "stopped")
        self.assertEqual(outcome["pid"], 4343)


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
