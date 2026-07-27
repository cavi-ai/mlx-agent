import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.convert import (
    ConvertError,
    plan_convert,
    start_convert,
    status_convert,
)


class PlanConvertTests(unittest.TestCase):
    def test_default_plan(self):
        plan = plan_convert("pub/model")
        self.assertEqual(plan["q_bits"], 4)
        self.assertEqual(plan["out"], "model-MLX-4bit")
        self.assertEqual(
            plan["argv"],
            ["mlx_lm.convert", "--hf-path", "pub/model", "--mlx-path", "model-MLX-4bit", "--q-bits", "4"],
        )
        self.assertEqual(len(plan["preview_hash"]), 64)

    def test_explicit_out_and_bits(self):
        plan = plan_convert("pub/model", q_bits=8, out="/tmp/out-8")
        self.assertEqual(plan["out"], "/tmp/out-8")
        self.assertIn("--q-bits", plan["argv"])
        self.assertEqual(plan["argv"][-1], "8")

    def test_invalid_repo(self):
        with self.assertRaises(ConvertError) as caught:
            plan_convert("../evil")
        self.assertEqual(caught.exception.code, "invalid_repo")

    def test_invalid_q_bits(self):
        with self.assertRaises(ConvertError) as caught:
            plan_convert("pub/model", q_bits=5)
        self.assertEqual(caught.exception.code, "invalid_arguments")


class StartConvertTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.spawned = []

    def _spawn(self, argv, log_path):
        self.spawned.append((argv, log_path))
        return 7777

    def _start(self, plan, **overrides):
        kwargs = {
            "receipts_dir": str(self.root),
            "confirm": True,
            "preview_hash": plan["preview_hash"],
            "which": lambda executable: "/usr/local/bin/" + executable,
            "model_present": lambda repo: True,
            "spawn": self._spawn,
            "now": lambda: "2026-07-26T00:00:00+00:00",
        }
        kwargs.update(overrides)
        return start_convert(plan, **kwargs)

    def test_preview_without_confirm(self):
        outcome = start_convert(plan_convert("pub/model"), receipts_dir=str(self.root))
        self.assertEqual(outcome["status"], "preview")
        self.assertEqual(self.spawned, [])

    def test_confirm_requires_preview_hash(self):
        with self.assertRaises(ConvertError) as caught:
            self._start(plan_convert("pub/model"), preview_hash=None)
        self.assertEqual(caught.exception.code, "preview_hash_required")

    def test_stale_preview_hash(self):
        with self.assertRaises(ConvertError) as caught:
            self._start(plan_convert("pub/model"), preview_hash="0" * 64)
        self.assertEqual(caught.exception.code, "preview_stale")

    def test_missing_executable(self):
        with self.assertRaises(ConvertError) as caught:
            self._start(plan_convert("pub/model"), which=lambda executable: None)
        self.assertEqual(caught.exception.code, "runtime_not_installed")

    def test_model_gate(self):
        with self.assertRaises(ConvertError) as caught:
            self._start(plan_convert("pub/model"), model_present=lambda repo: False)
        self.assertEqual(caught.exception.code, "model_not_local")

    def test_output_exists_gate(self):
        with TemporaryDirectory() as other:
            plan = plan_convert("pub/model", out=other)
            with self.assertRaises(ConvertError) as caught:
                self._start(plan)
            self.assertEqual(caught.exception.code, "output_exists")

    def test_success_writes_receipt(self):
        outcome = self._start(plan_convert("pub/model", out=str(self.root / "out-new")))
        self.assertEqual(outcome["status"], "started")
        receipt = outcome["receipt"]
        self.assertEqual(receipt["pid"], 7777)
        self.assertEqual(receipt["kind"], "convert")
        receipts = list((self.root / ".mlx-agent-receipts" / "convert").glob("*.json"))
        self.assertEqual(len(receipts), 1)

    def test_running_job_blocks_second_start(self):
        plan = plan_convert("pub/model", out=str(self.root / "out-new"))
        self._start(plan, pid_alive=lambda pid: True)
        second = plan_convert("pub/other", out=str(self.root / "out-2"))
        with self.assertRaises(ConvertError) as caught:
            start_convert(
                second,
                receipts_dir=str(self.root),
                confirm=True,
                preview_hash=second["preview_hash"],
                which=lambda executable: "/bin/x",
                spawn=self._spawn,
                pid_alive=lambda pid: True,
            )
        self.assertEqual(caught.exception.code, "job_in_progress")


class StatusConvertTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def _start_one(self, out):
        plan = plan_convert("pub/model", out=str(out))
        return start_convert(
            plan,
            receipts_dir=str(self.root),
            confirm=True,
            preview_hash=plan["preview_hash"],
            which=lambda executable: "/bin/x",
            spawn=lambda argv, log_path: 7777,
            now=lambda: "2026-07-26T00:00:00+00:00",
        )

    _COMMAND = "mlx_lm.convert --hf-path pub/model --mlx-path out-new --q-bits 4"

    def test_running_job(self):
        out = self.root / "out-new"
        self._start_one(out)
        entries = status_convert(
            str(self.root),
            pid_alive=lambda pid: True,
            pid_command=lambda pid: self._COMMAND,
        )
        self.assertEqual(entries[0]["state"], "running")

    def test_argv_mismatch_is_unknown(self):
        out = self.root / "out-new"
        self._start_one(out)
        entries = status_convert(
            str(self.root),
            pid_alive=lambda pid: True,
            pid_command=lambda pid: "python something-else",
        )
        self.assertEqual(entries[0]["state"], "unknown")

    def test_completed_job_marks_done_once(self):
        out = self.root / "out-new"
        self._start_one(out)
        out.mkdir()
        entries = status_convert(str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(entries[0]["state"], "done")
        self.assertIsNotNone(entries[0]["completed_at"])
        again = status_convert(str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(again[0]["state"], "done")

    def test_missing_output_marks_failed(self):
        out = self.root / "out-new"
        self._start_one(out)
        entries = status_convert(str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(entries[0]["state"], "failed")


if __name__ == "__main__":
    unittest.main()
