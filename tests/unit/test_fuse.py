import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.fuse import (
    FuseError,
    plan_fuse,
    start_fuse,
    status_fuse,
    validate_adapter,
)


def _adapter(root):
    path = Path(root) / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps({"fine_tune_type": "lora"}))
    return path


class ValidateAdapterTests(unittest.TestCase):
    def test_valid_adapter(self):
        with TemporaryDirectory() as directory:
            adapter = _adapter(directory)
            self.assertEqual(validate_adapter(adapter), {"adapter_config": True})

    def test_missing_directory(self):
        with self.assertRaises(FuseError) as caught:
            validate_adapter("/tmp/definitely-not-here")
        self.assertEqual(caught.exception.code, "adapter_invalid")

    def test_missing_config(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FuseError) as caught:
                validate_adapter(directory)
            self.assertIn("adapter_config.json", str(caught.exception))

    def test_broken_config(self):
        with TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("nope")
            with self.assertRaises(FuseError):
                validate_adapter(adapter)


class PlanFuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.adapter = _adapter(self.directory.name)

    def test_default_plan(self):
        plan = plan_fuse("pub/model", str(self.adapter))
        self.assertEqual(plan["out"], "model-fused")
        self.assertEqual(
            plan["argv"],
            ["mlx_lm.fuse", "--model", "pub/model", "--adapter-path", str(self.adapter), "--save-path", "model-fused"],
        )
        self.assertEqual(len(plan["preview_hash"]), 64)

    def test_invalid_repo(self):
        with self.assertRaises(FuseError) as caught:
            plan_fuse("bad", str(self.adapter))
        self.assertEqual(caught.exception.code, "invalid_repo")


class StartFuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.adapter = _adapter(self.directory.name)
        self.plan = plan_fuse(
            "pub/model", str(self.adapter), out=str(self.root / "fused")
        )

    def _start(self, **overrides):
        kwargs = {
            "receipts_dir": str(self.root),
            "confirm": True,
            "preview_hash": self.plan["preview_hash"],
            "which": lambda executable: "/usr/local/bin/" + executable,
            "model_present": lambda repo: True,
            "spawn": lambda argv, log_path: 9999,
            "now": lambda: "2026-07-27T00:00:00+00:00",
        }
        kwargs.update(overrides)
        return start_fuse(self.plan, **kwargs)

    def test_preview_without_confirm(self):
        outcome = start_fuse(self.plan, receipts_dir=str(self.root))
        self.assertEqual(outcome["status"], "preview")

    def test_confirm_requires_preview_hash(self):
        with self.assertRaises(FuseError) as caught:
            self._start(preview_hash=None)
        self.assertEqual(caught.exception.code, "preview_hash_required")

    def test_stale_preview_hash(self):
        with self.assertRaises(FuseError) as caught:
            self._start(preview_hash="0" * 64)
        self.assertEqual(caught.exception.code, "preview_stale")

    def test_missing_executable(self):
        with self.assertRaises(FuseError) as caught:
            self._start(which=lambda executable: None)
        self.assertEqual(caught.exception.code, "runtime_not_installed")

    def test_model_gate(self):
        with self.assertRaises(FuseError) as caught:
            self._start(model_present=lambda repo: False)
        self.assertEqual(caught.exception.code, "model_not_local")

    def test_output_exists_gate(self):
        (self.root / "fused").mkdir()
        with self.assertRaises(FuseError) as caught:
            self._start()
        self.assertEqual(caught.exception.code, "output_exists")

    def test_success_writes_receipt(self):
        outcome = self._start()
        self.assertEqual(outcome["status"], "started")
        self.assertEqual(outcome["receipt"]["kind"], "fuse")
        receipts = list((self.root / ".mlx-agent-receipts" / "fuse").glob("*.json"))
        self.assertEqual(len(receipts), 1)

    def test_running_job_blocks_second_start(self):
        self._start(pid_alive=lambda pid: True)
        with self.assertRaises(FuseError) as caught:
            self._start(pid_alive=lambda pid: True)
        self.assertEqual(caught.exception.code, "job_in_progress")


class StatusFuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        adapter = _adapter(self.directory.name)
        plan = plan_fuse("pub/model", str(adapter), out=str(self.root / "fused"))
        start_fuse(
            plan,
            receipts_dir=str(self.root),
            confirm=True,
            preview_hash=plan["preview_hash"],
            which=lambda executable: "/bin/x",
            spawn=lambda argv, log_path: 9999,
            now=lambda: "2026-07-27T00:00:00+00:00",
        )

    _COMMAND = "mlx_lm.fuse --model pub/model --adapter-path adapter --save-path fused"

    def test_running_job(self):
        entries = status_fuse(
            str(self.root),
            pid_alive=lambda pid: True,
            pid_command=lambda pid: self._COMMAND,
        )
        self.assertEqual(entries[0]["state"], "running")

    def test_completed_marks_done(self):
        (self.root / "fused").mkdir()
        entries = status_fuse(str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(entries[0]["state"], "done")

    def test_missing_output_marks_failed(self):
        entries = status_fuse(str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(entries[0]["state"], "failed")


if __name__ == "__main__":
    unittest.main()
