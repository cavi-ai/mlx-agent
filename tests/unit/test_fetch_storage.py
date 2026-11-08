"""Storage admission uses synthetic bytes, never downloaded model weights."""
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from mlx_agent import fetch_runner
from mlx_agent import cli
from mlx_agent.intake_fetch import FetchError, plan_fetch, start_fetch


COMMIT = "a" * 40


class FetchStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.storage = self.root / "storage"
        self.launched = []

    def plan(self, directory="canonical", **kwargs):
        return plan_fetch("org/model", revision=COMMIT, file="weights.safetensors",
                          local_dir=str(self.root / directory), storage_root=self.storage, **kwargs)

    def start(self, plan, receipts="receipts", spawn=None):
        return start_fetch(plan, receipts_dir=self.root / receipts, confirm=True,
                           preview_hash=plan["preview_hash"], module_present=lambda _: [],
                           spawn=spawn or (lambda argv, log, env: self.launched.append(argv) or 999999))

    def complete(self, outcome):
        def materialize(**kwargs):
            path = Path(kwargs["local_dir"])
            for name in kwargs["allow_patterns"]:
                item = path / name
                item.parent.mkdir(parents=True, exist_ok=True)
                item.write_bytes(b"synthetic fixture")
            return str(path)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(fetch_runner.main(self.launched[-1][2:], download=materialize), 0)
        return Path(outcome["receipt"]["local_dir"])

    def test_preview_is_read_only(self):
        plan = self.plan()
        self.assertEqual(plan["storage"]["mode"], "download")
        self.assertFalse(self.storage.exists())
        self.assertFalse((self.root / "canonical").exists())

    def test_download_approval_survives_admission_and_removed_failed_staging(self):
        original = self.plan()
        legacy = plan_fetch("org/model", revision=COMMIT, file="weights.safetensors", local_dir=str(self.root / "canonical"))
        self.assertEqual(original["preview_hash"], legacy["preview_hash"])
        first = self.start(original)
        def fail(**kwargs):
            (Path(kwargs["local_dir"]) / "partial").write_bytes(b"synthetic partial")
            raise OSError("synthetic failure")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(fetch_runner.main(self.launched[-1][2:], download=fail), 1)
        canonical = Path(first["receipt"]["local_dir"])
        canonical.rename(self.root / "quarantined-fixture")
        retry = self.plan()
        self.assertEqual(retry["preview_hash"], original["preview_hash"])
        self.start(retry)
        self.assertEqual(len(self.launched), 2)

    def test_other_sources_cannot_claim_overlapping_destinations(self):
        first = self.start(self.plan())
        for destination in ["canonical", "canonical/nested", "."]:
            with self.subTest(destination=destination), self.assertRaises(FetchError) as raised:
                plan_fetch("other/model", revision=COMMIT, local_dir=str(self.root / destination), storage_root=self.storage)
            self.assertEqual(raised.exception.code, "storage_destination_conflict")
        self.assertEqual(len(self.launched), 1)

    def test_distinct_sources_share_admission_for_parent_child_destinations(self):
        original = self.plan()
        child = plan_fetch("other/model", revision=COMMIT, local_dir=str(self.root / "canonical/nested"), storage_root=self.storage)
        self.start(original)
        with self.assertRaises(FetchError) as raised:
            self.start(child)
        self.assertEqual(raised.exception.code, "storage_changed")
        self.assertEqual(len(self.launched), 1)

    def test_cli_reuses_one_source_without_a_second_worker_or_destination(self):
        def call(directory, preview_hash=None):
            arguments = ["intake", "fetch", "org/model", "--revision", COMMIT,
                         "--file", "weights.safetensors", "--local-dir", str(self.root / directory),
                         "--storage-root", str(self.storage), "--receipts-dir", str(self.root / "receipts"), "--json"]
            if preview_hash:
                arguments += ["--confirm", "--preview-hash", preview_hash]
            output = io.StringIO()
            with redirect_stdout(output):
                code = cli.main(arguments)
            return code, json.loads(output.getvalue())["data"]
        with patch("mlx_agent.intake_fetch.spawn_with_env", side_effect=lambda argv, *_: self.launched.append(argv) or 999999), \
                patch("mlx_agent.intake_fetch._default_module_present", return_value=[]):
            code, preview = call("canonical")
            self.assertEqual(code, 2)
            code, first = call("canonical", preview["plan"]["preview_hash"])
            self.assertEqual(code, 0)
            self.complete(first)
            _, review = call("copy")
            self.assertEqual(review["plan"]["storage"]["mode"], "reuse")
            code, reused = call("copy", review["plan"]["preview_hash"])
        self.assertEqual(code, 0)
        self.assertEqual(reused["status"], "reused")
        self.assertEqual(reused["reused_fetch"]["state"], "done")
        self.assertEqual(reused["receipt"], first["receipt"])
        self.assertEqual(len(self.launched), 1)
        self.assertFalse((self.root / "copy").exists())

    def test_overlapping_snapshot_and_companion_sets_cannot_duplicate_files(self):
        self.start(self.plan(additional_files=["projector.gguf"]))
        for file, extras in [(None, []), ("projector.gguf", []), ("other.gguf", ["weights.safetensors"])]:
            with self.subTest(file=file), self.assertRaises(FetchError) as raised:
                plan_fetch("org/model", revision=COMMIT, file=file, additional_files=extras,
                           local_dir=str(self.root / "copy"), storage_root=self.storage)
            self.assertEqual(raised.exception.code, "storage_selection_conflict")

    def test_reuse_rechecks_file_evidence_at_confirmation(self):
        first = self.start(self.plan())
        canonical = self.complete(first)
        reuse = self.plan("copy")
        (canonical / "weights.safetensors").write_bytes(b"changed after preview")
        with self.assertRaises(FetchError) as raised:
            self.start(reuse)
        self.assertEqual(raised.exception.code, "storage_source_changed")
        self.assertEqual(len(self.launched), 1)

    def test_existing_provider_snapshot_blocks_a_managed_copy(self):
        cached = self.root / "hub" / "models--org--model" / "snapshots" / COMMIT
        cached.mkdir(parents=True)
        (cached / "weights.safetensors").write_bytes(b"synthetic fixture")
        with patch("mlx_agent.model_doctor.default_hf_cache", return_value=self.root / "hub"):
            with self.assertRaises(FetchError) as raised:
                self.plan()
        self.assertEqual(raised.exception.code, "source_already_cached")
        self.assertFalse(self.storage.exists())

    def test_provider_cache_is_rechecked_before_launch(self):
        preview = self.plan()
        with patch("mlx_agent.convert.cached_snapshot", return_value=self.root / "cached"):
            with self.assertRaises(FetchError) as raised:
                self.start(preview)
        self.assertEqual(raised.exception.code, "source_already_cached")
        self.assertEqual(self.launched, [])

    def test_registration_survives_receipt_publication_failure_after_spawn(self):
        from mlx_agent.intake_fetch import _write_receipt
        calls = []
        def publish(*args):
            calls.append(args)
            if len(calls) == 2:
                raise OSError("publication interrupted")
            return _write_receipt(*args)
        with patch("mlx_agent.intake_fetch._write_receipt", side_effect=publish), self.assertRaises(FetchError):
            self.start(self.plan())
        with self.assertRaises(FetchError):
            self.plan("copy")
        self.assertEqual(len(self.launched), 1)

    def test_registration_survives_an_interrupted_launcher(self):
        def interrupted(*_):
            raise OSError("launcher interrupted")
        with self.assertRaises(FetchError):
            self.start(self.plan(), spawn=interrupted)
        with self.assertRaises(FetchError):
            self.plan("copy")
        self.assertEqual(self.launched, [])

    def test_missing_source_can_retry_only_its_original_location_after_terminal_receipt(self):
        first = self.start(self.plan())
        canonical = self.complete(first)
        (canonical / "weights.safetensors").unlink()
        canonical.rmdir()
        retry = self.plan("copy")
        self.assertEqual(retry["local_dir"], str(canonical))
        self.assertEqual(retry["storage"]["mode"], "download")
        self.start(retry)
        self.assertEqual(len(self.launched), 2)

    def test_completed_source_is_reused_across_destinations_and_receipt_roots(self):
        first = self.start(self.plan())
        canonical = self.complete(first)
        before = (canonical / "weights.safetensors").stat()
        next_plan = self.plan("another-copy")
        self.assertEqual(next_plan["local_dir"], str(canonical))
        self.assertEqual(next_plan["storage"]["mode"], "reuse")
        self.assertEqual(next_plan["argv"], [])
        self.assertEqual(next_plan["network"], [])
        result = self.start(next_plan, receipts="other-receipts")
        self.assertEqual(result["status"], "reused")
        self.assertEqual(result["receipt"], first["receipt"])
        self.assertEqual(len(self.launched), 1)
        self.assertFalse((self.root / "another-copy").exists())
        self.assertEqual((canonical / "weights.safetensors").stat(), before)

    def test_changed_completed_source_blocks_a_second_copy(self):
        first = self.start(self.plan())
        canonical = self.complete(first)
        (canonical / "weights.safetensors").write_bytes(b"changed")
        with self.assertRaises(FetchError) as raised:
            self.plan("another-copy")
        self.assertEqual(raised.exception.code, "storage_source_changed")
        self.assertFalse((self.root / "another-copy").exists())

    def test_another_companion_selection_cannot_duplicate_the_primary(self):
        first = self.start(self.plan())
        self.complete(first)
        with self.assertRaises(FetchError) as raised:
            self.plan("another-copy", additional_files=["projector.gguf"])
        self.assertEqual(raised.exception.code, "storage_selection_conflict")

    def test_confirmation_rechecks_admission_after_preview(self):
        first_plan = self.plan()
        stale_plan = self.plan("another-copy")
        self.start(first_plan)
        with self.assertRaises(FetchError) as raised:
            self.start(stale_plan, receipts="other-receipts")
        self.assertEqual(raised.exception.code, "storage_changed")
        self.assertEqual(len(self.launched), 1)

    def test_a_concurrent_confirmer_cannot_start_a_second_worker(self):
        first_plan, second_plan = self.plan(), self.plan("another-copy")
        entered, release = threading.Event(), threading.Event()
        failures = []
        def spawn(argv, log, env):
            self.launched.append(argv)
            entered.set()
            release.wait(3)
            return 999999
        def run():
            try:
                self.start(first_plan, spawn=spawn)
            except Exception as error:
                failures.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaises(FetchError) as raised:
                self.start(second_plan, receipts="other-receipts")
            self.assertEqual(raised.exception.code, "storage_busy")
        finally:
            release.set()
            thread.join(4)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(self.launched), 1)

    def test_corrupt_or_linked_index_is_not_silently_reset(self):
        first = self.start(self.plan())
        self.complete(first)
        entry = next(self.storage.glob("*.json"))
        original = entry.read_bytes()
        entry.write_text("not JSON")
        with self.assertRaises(FetchError):
            self.plan("another-copy")
        entry.unlink()
        outside = self.root / "outside.json"
        outside.write_bytes(original)
        entry.symlink_to(outside)
        with self.assertRaises(FetchError):
            self.plan("another-copy")
        self.assertEqual(outside.read_bytes(), original)

    def test_other_revisions_keep_distinct_canonical_sources(self):
        first = self.start(self.plan())
        self.complete(first)
        other = plan_fetch("org/model", revision="b" * 40, file="weights.safetensors",
                           local_dir=str(self.root / "other-revision"), storage_root=self.storage)
        self.assertEqual(other["storage"]["mode"], "download")
        self.start(other)
        self.assertEqual(len(self.launched), 2)
