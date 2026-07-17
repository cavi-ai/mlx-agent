import hashlib
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from mlx_agent.cli import main
from mlx_agent.transactions import Transaction, rollback


class TransactionTests(unittest.TestCase):
    def _change(self, path, content, endpoint=None):
        return {"path": str(path), "content": content, "runtime": "mlx_lm", "endpoint": endpoint}

    def test_preview_refuses_symlinks_and_apply_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"providers": []}\n')
            linked = root / "linked.json"
            linked.symlink_to(target)
            transaction = Transaction(receipts_dir=root / "receipts")
            with self.assertRaises(ValueError):
                transaction.preview([self._change(linked, '{"providers": []}\n')])
            preview = transaction.preview([self._change(target, '{"providers": [{"id": "mlxlm"}]}\n')])
            self.assertIn("---", preview["diff"])
            with self.assertRaises(PermissionError):
                transaction.apply(False)
            self.assertEqual('{"providers": []}\n', target.read_text())

    def test_apply_preserves_mode_and_manual_rollback_restores_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            before = b'{"providers": [], "token": "do-not-store"}\n'
            target.write_bytes(before)
            os.chmod(target, 0o640)
            transaction = Transaction(receipts_dir=root / "receipts")
            transaction.preview([self._change(target, '{"providers": [{"id": "mlxlm"}]}\n')])
            receipt = transaction.apply(True)
            self.assertEqual(0o640, target.stat().st_mode & 0o777)
            self.assertTrue(Path(receipt.backup_paths[str(target)]).is_file())
            self.assertNotIn("do-not-store", json.dumps(receipt.to_dict()))
            restored = rollback(receipt.receipt_path)
            self.assertEqual("rolled_back", restored.status)
            self.assertEqual(before, target.read_bytes())
            self.assertEqual(hashlib.sha256(before).hexdigest(), restored.after_hashes[str(target)])

    def test_failed_health_check_automatically_restores_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            before = b'{"providers": []}\n'
            target.write_bytes(before)
            transaction = Transaction(receipts_dir=root / "receipts", health_checker=lambda endpoint: False)
            transaction.preview([self._change(target, '{"providers": [{"id": "mlxlm"}]}\n', "http://127.0.0.1:1/health")])
            receipt = transaction.apply(True)
            self.assertEqual("rolled_back", receipt.status)
            self.assertEqual(before, target.read_bytes())
            self.assertTrue(receipt.validations["health_check"]["passed"] is False)

    def test_post_apply_validation_failure_restores_the_current_target(self):
        class FailingPostValidation:
            def __init__(self):
                self.calls = 0

            def validate(self, content):
                self.calls += 1
                if self.calls > 2:
                    raise ValueError("simulated live validation failure")
                return True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            before = b'{"providers": []}\n'
            target.write_bytes(before)
            adapter = FailingPostValidation()
            transaction = Transaction(receipts_dir=root / "receipts")
            transaction.preview([{
                "path": str(target), "content": '{"providers": [{"id": "mlxlm"}]}\n',
                "adapter": adapter,
            }])
            receipt = transaction.apply(True)
            self.assertEqual("rolled_back", receipt.status)
            self.assertEqual(before, target.read_bytes())
            self.assertEqual(receipt.before_hashes[str(target)], receipt.after_hashes[str(target)])

    def test_cli_wire_apply_requires_confirm_and_writes_a_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text("{}\n")
            preview = StringIO()
            with redirect_stdout(preview):
                self.assertEqual(2, main(["wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target)]))
            self.assertIn("---", preview.getvalue())
            self.assertEqual("{}\n", target.read_text())
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(["wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target), "--confirm", "--json"]))
            receipt = json.loads(output.getvalue())["data"]["receipt"]
            self.assertEqual("applied", receipt["status"])
            self.assertTrue(Path(receipt["receipt_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
