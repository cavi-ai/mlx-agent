import hashlib
import json
import os
import fcntl
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from mlx_agent.cli import main
from mlx_agent.transactions import (
    ConcurrentTransactionError, Receipt, Transaction, _target_lock_digest,
    _target_lock_name, legacy_lock_problem, rollback,
)


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

    def test_explicit_lock_root_is_persisted_outside_target_and_reused_by_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            target = artifacts / "providers.json"
            target.write_text('{"providers": []}\n')
            locks = root / "installer-locks"
            transaction = Transaction(receipts_dir=root / "receipts", lock_root=locks)
            transaction.preview([self._change(target, '{"providers": [{"id": "mlxlm"}]}\n')])
            receipt = transaction.apply(True)
            self.assertEqual(str(locks), receipt.lock_root)
            self.assertFalse(list(artifacts.glob(".mlx-agent-wire-*.lock")))
            self.assertTrue(list(locks.glob(".mlx-agent-wire-*.lock")))
            self.assertEqual("rolled_back", rollback(receipt.receipt_path).status)
            self.assertFalse(list(artifacts.glob(".mlx-agent-wire-*.lock")))

    def test_explicit_lock_root_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text("{}\n")
            outside = root / "outside"
            outside.mkdir()
            linked = root / "locks"
            linked.symlink_to(outside, target_is_directory=True)
            transaction = Transaction(receipts_dir=root / "receipts", lock_root=linked)
            transaction.preview([self._change(target, '{"mlx_agent_wire": {}}\n')])
            with self.assertRaises(ValueError):
                transaction.apply(True)

    def test_scoped_lock_root_refuses_a_busy_legacy_target_lock_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "artifacts" / "providers.json"
            target.parent.mkdir()
            target.write_text('{"before": true}\n')
            legacy = target.parent / _target_lock_name(target)
            legacy.write_text("legacy\n")
            descriptor = os.open(legacy, os.O_RDWR | os.O_NOFOLLOW)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                transaction = Transaction(receipts_dir=root / "receipts", lock_root=root / "locks")
                transaction.preview([self._change(target, '{"after": true}\n')])
                with self.assertRaisesRegex(ConcurrentTransactionError, "legacy_lock_busy"):
                    transaction.apply(True)
                self.assertEqual('{"before": true}\n', target.read_text())
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def test_scoped_lock_root_migrates_stale_legacy_locks_and_rollback_rejects_recreated_legacy_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "artifacts" / "providers.json"
            target.parent.mkdir()
            target.write_text('{"before": true}\n')
            legacy = target.parent / _target_lock_name(target)
            legacy.write_text("stale legacy lock\n")
            transaction = Transaction(receipts_dir=root / "receipts", lock_root=root / "locks")
            transaction.preview([self._change(target, '{"after": true}\n')])
            receipt = transaction.apply(True)
            self.assertFalse(legacy.exists())
            self.assertEqual("legacy-target-locks-v1", receipt.lock_migration)
            self.assertTrue((root / "locks" / "legacy-lock-migration-v1.json").is_file())
            legacy.write_text("recreated legacy lock\n")
            with self.assertRaisesRegex(ConcurrentTransactionError, "legacy_lock_recreated"):
                rollback(receipt.receipt_path)
            self.assertEqual('{"after": true}\n', target.read_text())

    def test_scoped_lock_root_refuses_a_symlinked_legacy_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "artifacts" / "providers.json"
            target.parent.mkdir()
            target.write_text('{"before": true}\n')
            outside = root / "outside.lock"
            outside.write_text("outside\n")
            (target.parent / _target_lock_name(target)).symlink_to(outside)
            transaction = Transaction(receipts_dir=root / "receipts", lock_root=root / "locks")
            transaction.preview([self._change(target, '{"after": true}\n')])
            with self.assertRaises(ValueError):
                transaction.apply(True)
            self.assertEqual('{"before": true}\n', target.read_text())

    def test_legacy_migration_tracks_targets_independently_and_doctor_distinguishes_recreation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / "scoped-locks"
            first = root / "a" / "providers.json"
            second = root / "b" / "providers.json"
            for target in (first, second):
                target.parent.mkdir()
                target.write_text('{"before": true}\n')
                (target.parent / _target_lock_name(target)).write_text("stale\n")

            transaction = Transaction(receipts_dir=root / "receipts-a", lock_root=locks)
            transaction.preview([self._change(first, '{"first": true}\n')])
            transaction.apply(True)
            state_path = locks / "legacy-lock-migration-v1.json"
            state = json.loads(state_path.read_text())
            self.assertEqual({_target_lock_digest(first)}, set(state["targets"]))
            self.assertTrue((second.parent / _target_lock_name(second)).exists())

            (first.parent / _target_lock_name(first)).write_text("recreated\n")
            problems = legacy_lock_problem([str(first), str(second)], locks)
            self.assertEqual({"legacy_lock_recreated", "legacy_lock_migration_required"}, {item["code"] for item in problems})

            transaction = Transaction(receipts_dir=root / "receipts-b", lock_root=locks)
            transaction.preview([self._change(second, '{"second": true}\n')])
            transaction.apply(True)
            state = json.loads(state_path.read_text())
            self.assertEqual({_target_lock_digest(first), _target_lock_digest(second)}, set(state["targets"]))
            (first.parent / _target_lock_name(first)).unlink()
            self.assertEqual([], legacy_lock_problem([str(first), str(second)], locks))

    def test_legacy_migration_merges_multi_target_state_and_rejects_corrupt_or_symlinked_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a.json", root / "b.json"
            for target in (first, second):
                target.write_text('{"before": true}\n')
                (target.parent / _target_lock_name(target)).write_text("stale\n")
            locks = root / "locks"
            transaction = Transaction(receipts_dir=root / "receipts", lock_root=locks)
            transaction.preview([
                self._change(first, '{"first": true}\n'),
                self._change(second, '{"second": true}\n'),
            ])
            transaction.apply(True)
            self.assertEqual({_target_lock_digest(first), _target_lock_digest(second)}, set(json.loads((locks / "legacy-lock-migration-v1.json").read_text())["targets"]))

            corrupt = root / "corrupt-locks"
            corrupt.mkdir()
            (corrupt / "legacy-lock-migration-v1.json").write_text("not-json\n")
            target = root / "corrupt.json"
            target.write_text('{"before": true}\n')
            transaction = Transaction(receipts_dir=root / "corrupt-receipts", lock_root=corrupt)
            transaction.preview([self._change(target, '{"after": true}\n')])
            with self.assertRaises(ValueError):
                transaction.apply(True)
            self.assertEqual('{"before": true}\n', target.read_text())

            linked = root / "linked-locks"
            linked.mkdir()
            outside = root / "outside-state.json"
            outside.write_text("{}\n")
            (linked / "legacy-lock-migration-v1.json").symlink_to(outside)
            transaction = Transaction(receipts_dir=root / "linked-receipts", lock_root=linked)
            transaction.preview([self._change(target, '{"after": true}\n')])
            with self.assertRaises(ValueError):
                transaction.apply(True)

    def test_target_specific_migration_uses_the_same_digest_for_macos_var_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if not (str(root).startswith("/var/") and os.path.islink("/var") and os.readlink("/var") == "private/var"):
                self.skipTest("requires the macOS /var compatibility alias")
            physical_root = Path("/private/var") / root.relative_to("/var")
            target = root / "providers.json"
            physical_target = physical_root / "providers.json"
            target.write_text('{"before": true}\n')
            (physical_target.parent / _target_lock_name(physical_target)).write_text("stale\n")
            transaction = Transaction(receipts_dir=root / "receipts", lock_root=root / "locks")
            transaction.preview([self._change(target, '{"after": true}\n')])
            transaction.apply(True)
            state = json.loads((physical_root / "locks" / "legacy-lock-migration-v1.json").read_text())
            self.assertEqual(_target_lock_digest(target), _target_lock_digest(physical_target))
            self.assertIn(_target_lock_digest(physical_target), state["targets"])

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
            preview_json = StringIO()
            with redirect_stdout(preview_json):
                self.assertEqual(2, main(["wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target), "--json"]))
            preview_hash = json.loads(preview_json.getvalue())["data"]["preview"]["preview_hash"]
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(["wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target), "--confirm", "--preview-hash", preview_hash, "--json"]))
            receipt = json.loads(output.getvalue())["data"]["receipt"]
            self.assertEqual("applied", receipt["status"])
            self.assertTrue(Path(receipt["receipt_path"]).is_file())
            self.assertIn("preview", json.loads(output.getvalue())["data"])

    def test_cli_health_failure_returns_nonzero_with_rolled_back_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text("{}\n")
            preview = StringIO()
            with redirect_stdout(preview):
                self.assertEqual(2, main(["wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target), "--endpoint", "http://127.0.0.1:1/health", "--json"]))
            preview_hash = json.loads(preview.getvalue())["data"]["preview"]["preview_hash"]
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(2, main([
                    "wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target),
                    "--endpoint", "http://127.0.0.1:1/health", "--confirm", "--preview-hash", preview_hash, "--json",
                ]))
            payload = json.loads(output.getvalue())
            self.assertEqual("rolled_back", payload["data"]["receipt"]["status"])
            self.assertEqual("{}\n", target.read_text())

    def test_pending_journal_survives_crash_after_first_of_two_replacements(self):
        class SimulatedCrash(BaseException):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "one.json", root / "two.json"
            first.write_text('{"before": 1}\n')
            second.write_text('{"before": 2}\n')
            def crash(point):
                if point == "after_replace:0":
                    raise SimulatedCrash()
            transaction = Transaction(receipts_dir=root / "receipts", fault_injector=crash)
            transaction.preview([
                self._change(first, '{"after": 1}\n'),
                self._change(second, '{"after": 2}\n'),
            ])
            with self.assertRaises(SimulatedCrash):
                transaction.apply(True)
            journals = list((root / "receipts").glob("*/receipt.json"))
            self.assertEqual(1, len(journals))
            self.assertEqual("pending", json.loads(journals[0].read_text())["status"])
            receipt = rollback(journals[0])
            self.assertEqual("rolled_back", receipt.status)
            self.assertEqual('{"before": 1}\n', first.read_text())
            self.assertEqual('{"before": 2}\n', second.read_text())

    def test_pending_receipt_write_failure_changes_no_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            before = b'{"before": true}\n'
            target.write_bytes(before)
            def reject_receipt(path, value):
                raise OSError("receipt disk full")
            transaction = Transaction(receipts_dir=root / "receipts", receipt_writer=reject_receipt)
            transaction.preview([self._change(target, '{"after": true}\n')])
            receipt = transaction.apply(True)
            self.assertEqual("failed", receipt.status)
            self.assertEqual(before, target.read_bytes())

    def test_receipt_transition_failure_restores_and_leaves_pending_journal_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            before = b'{"before": true}\n'
            target.write_bytes(before)
            calls = []
            def writer(path, value):
                calls.append(value["status"])
                if len(calls) == 1:
                    path.write_text(json.dumps(value))
                else:
                    raise OSError("receipt transition unavailable")
            transaction = Transaction(receipts_dir=root / "receipts", receipt_writer=writer)
            transaction.preview([self._change(target, '{"after": true}\n')])
            receipt = transaction.apply(True)
            self.assertEqual("recovery_required", receipt.status)
            self.assertEqual(before, target.read_bytes())
            self.assertEqual("pending", json.loads(Path(receipt.receipt_path).read_text())["status"])
            self.assertEqual("rolled_back", rollback(receipt.receipt_path).status)

    def test_rollback_rejects_tampered_or_missing_backup_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            transaction = Transaction(receipts_dir=root / "receipts")
            transaction.preview([self._change(target, '{"after": true}\n')])
            receipt = transaction.apply(True)
            Path(receipt.backup_paths[str(target)]).write_text("tampered\n")
            restored = rollback(receipt.receipt_path)
            self.assertIn(restored.status, {"rollback_failed", "recovery_required"})

    def test_rollback_restore_write_failure_never_claims_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            transaction = Transaction(receipts_dir=root / "receipts")
            transaction.preview([self._change(target, '{"after": true}\n')])
            receipt = transaction.apply(True)
            from mlx_agent.transactions import _atomic_in_directory as real_atomic
            def fail_target(directory, name, *args, **kwargs):
                if Path(directory) == target.parent and name == target.name:
                    raise OSError("simulated restore write failure")
                return real_atomic(directory, name, *args, **kwargs)
            with patch("mlx_agent.transactions._atomic_in_directory", side_effect=fail_target):
                restored = rollback(receipt.receipt_path)
            self.assertIn(restored.status, {"rollback_failed", "recovery_required"})
            self.assertEqual('{"after": true}\n', target.read_text())
            self.assertEqual('{"after": true}\n', target.read_text())
            Path(receipt.backup_paths[str(target)]).unlink()
            restored = rollback(receipt.receipt_path)
            self.assertIn(restored.status, {"rollback_failed", "recovery_required"})

    def test_receipt_rejects_ancestor_symlink_traversal_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                Transaction(receipts_dir=root / "receipts").preview([self._change(linked / "providers.json", "{}\n")])
            receipts_link = root / "receipts-link"
            receipts_link.symlink_to(real, target_is_directory=True)
            transaction = Transaction(receipts_dir=receipts_link)
            target = real / "providers.json"
            with self.assertRaises(ValueError):
                transaction.preview([self._change(target, "{}\n")])
                transaction.apply(True)
            bad = root / "bad.json"
            bad.write_text(json.dumps({"schema_version": "1.0", "unexpected": True}))
            with self.assertRaises(ValueError):
                Receipt.from_dict(json.loads(bad.read_text()), str(bad))

    def test_receipt_loader_rejects_unknown_status_and_backup_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            transaction = Transaction(receipts_dir=root / "receipts")
            transaction.preview([self._change(target, '{"after": true}\n')])
            receipt = transaction.apply(True)
            value = json.loads(Path(receipt.receipt_path).read_text())
            value["status"] = "anything"
            with self.assertRaises(ValueError):
                Receipt.from_dict(value, receipt.receipt_path)
            value["status"] = "applied"
            value["backup_paths"][str(target)] = str(root / "outside-backup.bin")
            with self.assertRaises(ValueError):
                Receipt.from_dict(value, receipt.receipt_path)

    def test_apply_aborts_if_previewed_before_hash_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": 1}\n')
            transaction = Transaction(receipts_dir=root / "receipts")
            transaction.preview([self._change(target, '{"after": 1}\n')])
            target.write_text('{"changed": true}\n')
            with self.assertRaises(ValueError):
                transaction.apply(True)
            self.assertEqual('{"changed": true}\n', target.read_text())

    def test_cli_status_and_confirmed_rollback_report_receipt_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text("{}\n")
            preview = StringIO()
            with redirect_stdout(preview):
                self.assertEqual(2, main(["wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target), "--json"]))
            preview_hash = json.loads(preview.getvalue())["data"]["preview"]["preview_hash"]
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(["wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target), "--confirm", "--preview-hash", preview_hash, "--json"]))
            receipt_path = json.loads(output.getvalue())["data"]["receipt"]["receipt_path"]
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(["wire", "status", receipt_path, "--json"]))
            self.assertEqual("applied", json.loads(output.getvalue())["data"]["receipt"]["status"])
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(["wire", "rollback", receipt_path, "--confirm", "--json"]))
            self.assertEqual("rolled_back", json.loads(output.getvalue())["data"]["receipt"]["status"])

    def test_cli_apply_requires_prior_preview_hash_across_invocations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text("{}\n")
            output = StringIO()
            request = ["wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target), "--json"]
            with redirect_stdout(output):
                self.assertEqual(2, main(request))
            preview = json.loads(output.getvalue())["data"]["preview"]
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(2, main(request[:-1] + ["--confirm", "--json"]))
            self.assertEqual("{}\n", target.read_text())
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(request[:-1] + ["--confirm", "--preview-hash", preview["preview_hash"], "--json"]))
            self.assertEqual("applied", json.loads(output.getvalue())["data"]["receipt"]["status"])

    def test_cli_preview_hash_rejects_stale_render_or_current_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text("{}\n")
            request = ["wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target), "--json"]
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(2, main(request))
            preview_hash = json.loads(output.getvalue())["data"]["preview"]["preview_hash"]
            target.write_text('{"external": true}\n')
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(2, main(request[:-1] + ["--confirm", "--preview-hash", preview_hash, "--json"]))
            self.assertEqual("preview_stale", json.loads(output.getvalue())["error"]["code"])

    def test_cli_preview_hash_rejects_render_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text("{}\n")
            initial = ["wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target), "--json"]
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(2, main(initial))
            preview_hash = json.loads(output.getvalue())["data"]["preview"]["preview_hash"]
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(2, main([
                    "wire", "apply", "mlx-community/Other-4bit", "--target", "mlx_lm", "--path", str(target),
                    "--confirm", "--preview-hash", preview_hash, "--json",
                ]))
            self.assertEqual("preview_stale", json.loads(output.getvalue())["error"]["code"])

    def test_repeated_rollback_rechecks_current_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            before = b'{"before": true}\n'
            target.write_bytes(before)
            transaction = Transaction(receipts_dir=root / "receipts")
            transaction.preview([self._change(target, '{"after": true}\n')])
            receipt = transaction.apply(True)
            self.assertEqual("rolled_back", rollback(receipt.receipt_path).status)
            target.write_text('{"later": true}\n')
            self.assertEqual("rolled_back", rollback(receipt.receipt_path).status)
            self.assertEqual(before, target.read_bytes())

    def test_ancestor_swap_hook_refuses_without_external_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safe = root / "safe"
            child = safe / "child"
            external = root / "external"
            child.mkdir(parents=True)
            external.mkdir()
            target = child / "providers.json"
            external_target = external / "providers.json"
            target.write_text('{"before": true}\n')
            def swap(parent, component):
                if component == "child":
                    child.rename(safe / "old-child")
                    child.symlink_to(external, target_is_directory=True)
            transaction = Transaction(receipts_dir=root / "receipts", path_race_hook=swap)
            with self.assertRaises(ValueError):
                transaction.preview([self._change(target, '{"after": true}\n')])
            self.assertFalse(external_target.exists())

    def test_prepare_journal_rechecks_absent_target_before_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "new.json"
            def create_after_preview(point):
                if point == "before_journal_capture":
                    target.write_text('{"external": true}\n')
            transaction = Transaction(receipts_dir=root / "receipts", fault_injector=create_after_preview)
            transaction.preview([self._change(target, '{"after": true}\n')])
            with self.assertRaises(ValueError):
                transaction.apply(True)
            self.assertEqual('{"external": true}\n', target.read_text())

    def test_journal_capture_rechecks_content_after_initial_capture_before_pending_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            def mutate_after_capture(point):
                if point == "after_journal_capture":
                    target.write_text('{"external": true}\n')
            transaction = Transaction(receipts_dir=root / "receipts", fault_injector=mutate_after_capture)
            transaction.preview([self._change(target, '{"after": true}\n')])
            with self.assertRaises(ValueError):
                transaction.apply(True)
            self.assertEqual('{"external": true}\n', target.read_text())
            self.assertEqual([], list((root / "receipts").glob("*/receipt.json")))

    def test_transaction_root_creation_uses_opened_receipts_directory_fd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            receipts = root / "receipts"
            external = root / "external"
            external.mkdir()
            def swap_receipts(point):
                if point == "before_transaction_root_create":
                    receipts.rename(root / "old-receipts")
                    receipts.symlink_to(external, target_is_directory=True)
            transaction = Transaction(receipts_dir=receipts, fault_injector=swap_receipts)
            transaction.preview([self._change(target, '{"after": true}\n')])
            with self.assertRaises(ValueError):
                transaction.apply(True)
            self.assertEqual([], list(external.iterdir()))
            self.assertEqual('{"before": true}\n', target.read_text())

    def test_mode_only_change_after_preview_aborts_before_pending_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            os.chmod(target, 0o640)
            def chmod_after_preview(point):
                if point == "before_journal_capture":
                    os.chmod(target, 0o600)
            transaction = Transaction(receipts_dir=root / "receipts", fault_injector=chmod_after_preview)
            transaction.preview([self._change(target, '{"after": true}\n')])
            with self.assertRaises(ValueError):
                transaction.apply(True)
            self.assertEqual(0o600, target.stat().st_mode & 0o777)

    def test_repeated_rollback_restores_mode_after_later_chmod(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            os.chmod(target, 0o640)
            transaction = Transaction(receipts_dir=root / "receipts")
            transaction.preview([self._change(target, '{"after": true}\n')])
            receipt = transaction.apply(True)
            self.assertEqual("rolled_back", rollback(receipt.receipt_path).status)
            os.chmod(target, 0o600)
            self.assertEqual("rolled_back", rollback(receipt.receipt_path).status)
            self.assertEqual(0o640, target.stat().st_mode & 0o777)

    def test_cli_render_refuses_leaf_and_ancestor_symlink_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            target = real / "providers.json"
            target.write_text("{}\n")
            leaf = root / "leaf.json"
            leaf.symlink_to(target)
            ancestor = root / "ancestor"
            ancestor.symlink_to(real, target_is_directory=True)
            for candidate in (leaf, ancestor / "providers.json"):
                with self.subTest(candidate=candidate):
                    output = StringIO()
                    with redirect_stdout(output):
                        self.assertEqual(2, main(["wire", "render", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(candidate), "--json"]))
                    self.assertEqual("wire_failed", json.loads(output.getvalue())["error"]["code"])

    def test_cooperative_transaction_contention_fails_before_mutation_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            first = Transaction(receipts_dir=root / "receipts")
            second = Transaction(receipts_dir=root / "receipts")
            first.preview([self._change(target, '{"first": true}\n')])
            second.preview([self._change(target, '{"second": true}\n')])
            with first._advisory_lock():
                with self.assertRaises(ConcurrentTransactionError):
                    second.apply(True)
                self.assertEqual('{"before": true}\n', target.read_text())
            receipt = second.apply(True)
            self.assertEqual("applied", receipt.status)
            self.assertEqual('{"second": true}\n', target.read_text())

    def test_lock_covers_after_check_writer_and_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            contender = Transaction(receipts_dir=root / "receipts")
            contender.preview([self._change(target, '{"contender": true}\n')])
            attempted = []
            def after_check(point):
                if point == "before_replace:0":
                    attempted.append(True)
                    with self.assertRaises(ConcurrentTransactionError):
                        contender.apply(True)
                    raise ValueError("induced failure after final check")
            first = Transaction(receipts_dir=root / "receipts", fault_injector=after_check)
            first.preview([self._change(target, '{"first": true}\n')])
            receipt = first.apply(True)
            self.assertEqual([True], attempted)
            self.assertEqual("rolled_back", receipt.status)
            self.assertEqual('{"before": true}\n', target.read_text())
            receipt = contender.apply(True)
            self.assertEqual("applied", receipt.status)
            self.assertIn("concurrency", receipt.validations)

    def test_cli_reports_cooperative_concurrency_warning_and_classified_contention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text("{}\n")
            preview = StringIO()
            with redirect_stdout(preview):
                self.assertEqual(2, main(["wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target), "--json"]))
            payload = json.loads(preview.getvalue())
            self.assertEqual("cooperative_concurrency", payload["warnings"][0]["code"])

            preview_hash = payload["data"]["preview"]["preview_hash"]
            holder = Transaction(receipts_dir=root / ".mlx-agent-receipts")
            holder.preview([self._change(target, '{"holder": true}\n')])
            with holder._advisory_lock():
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(2, main([
                        "wire", "apply", "mlx-community/Test-4bit", "--target", "mlx_lm", "--path", str(target),
                        "--confirm", "--preview-hash", preview_hash, "--json",
                    ]))
                self.assertEqual("cooperative_lock_busy", json.loads(output.getvalue())["error"]["code"])

    def test_same_target_different_receipt_directories_contend_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            first = Transaction(receipts_dir=root / "receipts-a")
            second = Transaction(receipts_dir=root / "receipts-b")
            first.preview([self._change(target, '{"first": true}\n')])
            second.preview([self._change(target, '{"second": true}\n')])
            with first._advisory_lock():
                with self.assertRaises(ConcurrentTransactionError):
                    second.apply(True)
                self.assertEqual('{"before": true}\n', target.read_text())
            self.assertEqual("applied", second.apply(True).status)

    def test_macos_var_aliases_contend_for_the_same_physical_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if not (str(root).startswith("/var/") and os.path.islink("/var") and os.readlink("/var") == "private/var"):
                self.skipTest("requires the macOS /var compatibility alias")
            physical_root = Path("/private/var") / root.relative_to("/var")
            target = root / "providers.json"
            alias_target = physical_root / "providers.json"
            target.write_text('{"before": true}\n')
            first = Transaction(receipts_dir=root / "receipts-a")
            second = Transaction(receipts_dir=physical_root / "receipts-b")
            first.preview([self._change(target, '{"first": true}\n')])
            second.preview([self._change(alias_target, '{"second": true}\n')])
            with first._advisory_lock():
                with self.assertRaises(ConcurrentTransactionError):
                    second.apply(True)
                self.assertEqual('{"before": true}\n', target.read_text())
            self.assertEqual("applied", second.apply(True).status)

    def test_rollback_contends_with_apply_using_different_receipt_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            original = Transaction(receipts_dir=root / "receipts-a")
            original.preview([self._change(target, '{"after": true}\n')])
            receipt = original.apply(True)
            holder = Transaction(receipts_dir=root / "receipts-b")
            holder.preview([self._change(target, '{"next": true}\n')])
            with holder._advisory_lock():
                with self.assertRaises(ConcurrentTransactionError):
                    rollback(receipt.receipt_path)
            self.assertEqual("rolled_back", rollback(receipt.receipt_path).status)

    def test_overlapping_multi_target_locks_sort_and_fail_without_partial_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path, shared_path, second_path = root / "a.json", root / "b.json", root / "c.json"
            for path in (first_path, shared_path, second_path):
                path.write_text('{"before": true}\n')
            first = Transaction(receipts_dir=root / "receipts-a")
            second = Transaction(receipts_dir=root / "receipts-b")
            first.preview([self._change(shared_path, '{"first": true}\n'), self._change(first_path, '{"first": true}\n')])
            second.preview([self._change(second_path, '{"second": true}\n'), self._change(shared_path, '{"second": true}\n')])
            self.assertEqual([str(first_path), str(shared_path)], first._lock_targets())
            with first._advisory_lock():
                with self.assertRaises(ConcurrentTransactionError):
                    second.apply(True)
                self.assertEqual('{"before": true}\n', second_path.read_text())
                self.assertEqual('{"before": true}\n', shared_path.read_text())
            self.assertEqual("applied", second.apply(True).status)

    def test_cli_status_refuses_swapped_receipt_ancestor_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "providers.json"
            target.write_text('{"before": true}\n')
            transaction = Transaction(receipts_dir=root / "receipts")
            transaction.preview([self._change(target, '{"after": true}\n')])
            receipt = transaction.apply(True)
            receipts = root / "receipts"
            receipts.rename(root / "old-receipts")
            external = root / "external"
            external.mkdir()
            receipts.symlink_to(external, target_is_directory=True)
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(2, main(["wire", "status", receipt.receipt_path, "--json"]))
            self.assertEqual("wire_failed", json.loads(output.getvalue())["error"]["code"])


if __name__ == "__main__":
    unittest.main()
