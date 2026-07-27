import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.lora import (
    LoraError,
    plan_lora,
    start_lora,
    status_lora,
    validate_dataset,
)


def _dataset(root, lines=None, name="train.jsonl"):
    path = Path(root) / name
    path.write_text("\n".join(lines if lines is not None else ['{"text": "hello world"}']) + "\n")
    return path


class DatasetValidationTests(unittest.TestCase):
    def test_text_dataset(self):
        with TemporaryDirectory() as directory:
            _dataset(directory)
            summary = validate_dataset(directory)
            self.assertEqual(summary["train_lines"], 1)
            self.assertEqual(summary["valid_lines"], 0)

    def test_messages_dataset(self):
        with TemporaryDirectory() as directory:
            _dataset(directory, [
                json.dumps({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]})
            ])
            summary = validate_dataset(directory)
            self.assertEqual(summary["train_lines"], 1)

    def test_valid_jsonl_counted(self):
        with TemporaryDirectory() as directory:
            _dataset(directory)
            _dataset(directory, name="valid.jsonl")
            summary = validate_dataset(directory)
            self.assertEqual(summary["valid_lines"], 1)

    def test_missing_train(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(LoraError) as caught:
                validate_dataset(directory)
            self.assertEqual(caught.exception.code, "dataset_invalid")

    def test_bad_json_reports_line(self):
        with TemporaryDirectory() as directory:
            _dataset(directory, ['{"text": "ok"}', "not json"])
            with self.assertRaises(LoraError) as caught:
                validate_dataset(directory)
            self.assertIn("line 2", str(caught.exception))

    def test_malformed_message(self):
        with TemporaryDirectory() as directory:
            _dataset(directory, [json.dumps({"messages": [{"role": "user"}]})])
            with self.assertRaises(LoraError) as caught:
                validate_dataset(directory)
            self.assertEqual(caught.exception.code, "dataset_invalid")

    def test_empty_file(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "train.jsonl").write_text("")
            with self.assertRaises(LoraError):
                validate_dataset(directory)


class PlanLoraTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        _dataset(self.directory.name)

    def test_default_plan(self):
        plan = plan_lora("pub/model", self.directory.name)
        self.assertEqual(plan["out"], "model-lora")
        self.assertEqual(plan["argv"][0], "mlx_lm.lora")
        self.assertIn("--adapter-path", plan["argv"])
        self.assertEqual(plan["dataset"]["train_lines"], 1)
        self.assertEqual(len(plan["preview_hash"]), 64)

    def test_invalid_repo(self):
        with self.assertRaises(LoraError) as caught:
            plan_lora("bad", self.directory.name)
        self.assertEqual(caught.exception.code, "invalid_repo")

    def test_hyperparameter_bounds(self):
        with self.assertRaises(LoraError):
            plan_lora("pub/model", self.directory.name, iters=0)
        with self.assertRaises(LoraError):
            plan_lora("pub/model", self.directory.name, batch_size=0)
        with self.assertRaises(LoraError):
            plan_lora("pub/model", self.directory.name, learning_rate=1.0)
        with self.assertRaises(LoraError):
            plan_lora("pub/model", self.directory.name, num_layers=1000)


class StartLoraTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "data").mkdir()
        _dataset(str(self.root / "data"))
        self.plan = plan_lora(
            "pub/model", str(self.root / "data"), out=str(self.root / "adapter")
        )
        self.spawned = []

    def _spawn(self, argv, log_path):
        self.spawned.append(argv)
        return 8888

    def _start(self, **overrides):
        kwargs = {
            "receipts_dir": str(self.root),
            "confirm": True,
            "preview_hash": self.plan["preview_hash"],
            "which": lambda executable: "/usr/local/bin/" + executable,
            "model_present": lambda repo: True,
            "spawn": self._spawn,
            "now": lambda: "2026-07-26T00:00:00+00:00",
        }
        kwargs.update(overrides)
        return start_lora(self.plan, **kwargs)

    def test_preview_without_confirm(self):
        outcome = start_lora(self.plan, receipts_dir=str(self.root))
        self.assertEqual(outcome["status"], "preview")
        self.assertEqual(self.spawned, [])

    def test_confirm_requires_preview_hash(self):
        with self.assertRaises(LoraError) as caught:
            self._start(preview_hash=None)
        self.assertEqual(caught.exception.code, "preview_hash_required")

    def test_stale_preview_hash(self):
        with self.assertRaises(LoraError) as caught:
            self._start(preview_hash="0" * 64)
        self.assertEqual(caught.exception.code, "preview_stale")

    def test_missing_executable(self):
        with self.assertRaises(LoraError) as caught:
            self._start(which=lambda executable: None)
        self.assertEqual(caught.exception.code, "runtime_not_installed")

    def test_model_gate(self):
        with self.assertRaises(LoraError) as caught:
            self._start(model_present=lambda repo: False)
        self.assertEqual(caught.exception.code, "model_not_local")

    def test_output_exists_gate(self):
        (self.root / "adapter").mkdir()
        with self.assertRaises(LoraError) as caught:
            self._start()
        self.assertEqual(caught.exception.code, "output_exists")

    def test_success_writes_receipt(self):
        outcome = self._start()
        self.assertEqual(outcome["status"], "started")
        self.assertEqual(outcome["receipt"]["kind"], "lora")
        receipts = list((self.root / ".mlx-agent-receipts" / "lora").glob("*.json"))
        self.assertEqual(len(receipts), 1)

    def test_running_job_blocks_second_start(self):
        self._start(pid_alive=lambda pid: True)
        with self.assertRaises(LoraError) as caught:
            self._start(pid_alive=lambda pid: True)
        self.assertEqual(caught.exception.code, "job_in_progress")


class StatusLoraTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "data").mkdir()
        _dataset(str(self.root / "data"))

    def _start_one(self, out):
        plan = plan_lora("pub/model", str(self.root / "data"), out=str(out))
        return start_lora(
            plan,
            receipts_dir=str(self.root),
            confirm=True,
            preview_hash=plan["preview_hash"],
            which=lambda executable: "/bin/x",
            spawn=lambda argv, log_path: 8888,
            now=lambda: "2026-07-26T00:00:00+00:00",
        )

    _COMMAND = "mlx_lm.lora --model pub/model --train --data data --adapter-path adapter"

    def test_running_job(self):
        self._start_one(self.root / "adapter")
        entries = status_lora(
            str(self.root),
            pid_alive=lambda pid: True,
            pid_command=lambda pid: self._COMMAND,
        )
        self.assertEqual(entries[0]["state"], "running")

    def test_completed_marks_done(self):
        out = self.root / "adapter"
        self._start_one(out)
        out.mkdir()
        entries = status_lora(str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(entries[0]["state"], "done")

    def test_missing_output_marks_failed(self):
        self._start_one(self.root / "adapter")
        entries = status_lora(str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(entries[0]["state"], "failed")


if __name__ == "__main__":
    unittest.main()
