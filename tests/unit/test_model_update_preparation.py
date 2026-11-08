"""Managed update contracts use tiny files/workers, never model runtimes."""
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from mlx_agent import fetch_runner, queue_bridge, queue_cli
from mlx_agent.convert import ConvertError, _finalize_plan, plan_convert, plan_gguf_convert, start_convert, status_convert
from mlx_agent.convert_queue import ConvertQueue, QueueStore
from mlx_agent.intake_fetch import FetchError, plan_fetch, start_fetch, status_fetch
from mlx_agent.source_evidence import completed_fetch, identities
from .test_gguf import write_gguf, _kv, _UINT32, _STRING

COMMIT = "a" * 40
BASE_COMMIT = "d26bb61231c349cf6b7896fa83353113880e1ba3"


class ManagedUpdatePreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.sequence = 0
        self.processes = []
        self.addCleanup(self._stop_workers)

    def _stop_workers(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)

    def fetch(self, repo="org/model", files=None, primary=None, additional=None, subfolder=None, revision=COMMIT, model_type=None, opener=None):
        self.sequence += 1
        destination = self.root / ("download-" + str(self.sequence))
        source = repo + ("/" + subfolder if subfolder else "")
        plan = plan_fetch(source, revision=revision, file=primary, additional_files=additional,
                          local_dir=str(destination), model_type=model_type)
        launched = []
        outcome = start_fetch(plan, receipts_dir=self.root, confirm=True, preview_hash=plan["preview_hash"],
                              spawn=lambda argv, log, env: launched.append(argv) or 99999,
                              module_present=lambda _: [])
        def download(**kwargs):
            target = Path(kwargs["local_dir"])
            for name, data in (files or {"config.json": b"{}", "model.safetensors": b"weights"}).items():
                location = target / name
                location.parent.mkdir(parents=True, exist_ok=True)
                location.write_bytes(data)
            return str(target)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(fetch_runner.main(launched[0][2:], download=download, opener=opener), 0)
        receipt = self.root / ".mlx-agent-receipts/fetch" / (outcome["receipt"]["slug"] + ".json")
        return destination, receipt, plan

    def test_bundle_is_exact_pinned_and_returns_bound_file_identities(self):
        destination, receipt, plan = self.fetch(primary="main.gguf", additional=["mmproj.gguf"],
            files={"main.gguf": b"main", "mmproj.gguf": b"vision"})
        self.assertEqual(plan["additional_files"], ["mmproj.gguf"])
        self.assertEqual(completed_fetch(receipt, destination / "main.gguf")["origin"], {
            "repo": "org/model", "revision": COMMIT, "file": "main.gguf", "subfolder": None,
            "additional_files": ["mmproj.gguf"]})
        status = status_fetch(self.root, pid_alive=lambda _: False)[0]
        self.assertEqual(status["state"], "done")
        self.assertEqual(status["files"], identities(destination))
        self.assertEqual(status["preview_hash"], plan["preview_hash"])
        self.assertEqual(status["path"], str(destination))
        self.assertEqual({value["path"] for value in status["files"]}, {"main.gguf", "mmproj.gguf"})

    def test_completed_fetch_status_rechecks_file_identities(self):
        for primary in (None, "main.gguf", "weights.safetensors"):
            with self.subTest(primary=primary):
                files = {primary: b"weights"} if primary else None
                destination, receipt, _ = self.fetch(primary=primary, files=files)
                recorded = json.loads(receipt.read_text())
                status = next(item for item in status_fetch(self.root) if item["receipt"] == str(receipt))
                self.assertEqual((status["state"], status["path"]), ("done", str(destination)))
                target = destination / (primary or "model.safetensors")
                target.write_bytes(b"changed")
                status = next(item for item in status_fetch(self.root) if item["receipt"] == str(receipt))
                self.assertEqual((status["state"], status["path"], status["files"]), ("failed", None, []))
                self.assertEqual(json.loads(receipt.read_text()), recorded)

    def test_completed_fetch_status_refuses_missing_or_linked_payload(self):
        for replacement in ("missing", "symlink"):
            with self.subTest(replacement=replacement):
                destination, receipt, _ = self.fetch()
                target = destination / "model.safetensors"
                target.unlink()
                if replacement == "symlink":
                    target.symlink_to(destination / "config.json")
                status = next(item for item in status_fetch(self.root) if item["receipt"] == str(receipt))
                self.assertEqual((status["state"], status["path"], status["files"]), ("failed", None, []))

    def test_bundle_allow_list_and_no_overwrite_are_enforced_in_runner(self):
        destination = self.root / "bundle"
        marker = self.root / "done.json"
        observed = []
        def download(**kwargs):
            observed.append(kwargs)
            for name in kwargs["allow_patterns"]:
                (Path(kwargs["local_dir"]) / name).write_bytes(b"fixture")
            return kwargs["local_dir"]
        args = ["--repo", "org/model", "--revision", COMMIT, "--file", "main.gguf", "--additional-file", "mmproj.gguf",
                "--local-dir", str(destination), "--fresh-destination", "--capture-identities", "--proof-token", "token", "--marker", str(marker)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(fetch_runner.main(args, download=download), 0)
            self.assertEqual(fetch_runner.main(args, download=download), 1)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["allow_patterns"], ["main.gguf", "mmproj.gguf"])
        self.assertEqual((destination / "main.gguf").read_bytes(), b"fixture")

    def test_bundle_rejects_mutable_revision_cache_and_duplicate_files(self):
        for changes in ({"revision": "main"}, {"hf_cache": "/shared/cache"},
                        {"additional_files": ["main.gguf"]}, {"additional_files": ["mmproj.gguf", "mmproj.gguf"]}):
            values = dict(revision=COMMIT, file="main.gguf", additional_files=["mmproj.gguf"], local_dir=str(self.root / "new"))
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaises((FetchError, ValueError)):
                plan_fetch("org/model", **values)
        with self.assertRaises(ValueError):
            plan_fetch("org/model", revision=COMMIT, file="main.gguf", additional_files=["../escape.gguf"], local_dir=str(self.root / "new"))

    def test_existing_destination_is_rejected_before_spawn(self):
        destination = self.root / "existing"; destination.mkdir()
        (destination / "owned").write_text("keep")
        plan = plan_fetch("org/model", revision=COMMIT, file="main.gguf", local_dir=str(destination))
        with patch("mlx_agent.intake_fetch.spawn_with_env") as spawn, self.assertRaises(FetchError):
            start_fetch(plan, receipts_dir=self.root, confirm=True, preview_hash=plan["preview_hash"], module_present=lambda _: [])
        spawn.assert_not_called()
        self.assertEqual((destination / "owned").read_text(), "keep")

    def test_changed_download_cannot_supply_origin_or_conversion_preview(self):
        destination, receipt, _ = self.fetch()
        (destination / "model.safetensors").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            completed_fetch(receipt)
        with self.assertRaises(ConvertError):
            plan_convert("org/model", source_path=str(destination), fetch_receipt=str(receipt), require_completion=True)

    def test_selected_checkpoint_path_preserves_relative_origin(self):
        destination, receipt, _ = self.fetch(subfolder="checkpoints/small", files={"checkpoints/small/config.json": b"{}", "checkpoints/small/model.safetensors": b"weights"})
        selected = destination / "checkpoints/small"
        plan = plan_convert("org/model", source_path=str(selected), subfolder="checkpoints/small", fetch_receipt=str(receipt), require_completion=True)
        self.assertEqual(plan["source"]["path"], str(selected))
        self.assertEqual(plan["source"]["origin"]["subfolder"], "checkpoints/small")
        with self.assertRaises(ConvertError):
            plan_convert("org/model", source_path=str(destination), subfolder="checkpoints/small", fetch_receipt=str(receipt))

    def test_managed_recipe_uses_primary_lora_and_pinned_dependency_without_cache(self):
        repo = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
        lora = "qwen-image-2.1-uncensored-lora.safetensors"
        main, main_receipt, fetch_plan = self.fetch(repo=repo, model_type="qwen_image_21", files={lora: b"lora"})
        base, base_receipt, _ = self.fetch(repo="Qwen/Qwen-Image-2.1", revision=BASE_COMMIT)
        self.assertNotIn("--base-repo", fetch_plan["argv"])
        with patch("mlx_agent.convert.cached_snapshot", side_effect=AssertionError("shared cache used")):
            plan = plan_convert(repo, backend="mflux", model_type="qwen_image_21", source_path=str(main),
                fetch_receipt=str(main_receipt), dependency_fetch_receipts=[str(base_receipt)], require_completion=True)
        self.assertEqual(plan["argv"][plan["argv"].index("--hf-path") + 1], str(base))
        self.assertEqual(plan["argv"][plan["argv"].index("--lora") + 1], str(main / lora))
        self.assertEqual(plan["source"]["dependencies"][0]["origin"]["revision"], BASE_COMMIT)
        with patch("mlx_agent.convert.cached_snapshot", side_effect=AssertionError("shared cache used")), self.assertRaises(ConvertError):
            plan_convert(repo, backend="mflux", model_type="qwen_image_21", source_path=str(main), fetch_receipt=str(main_receipt))

    def test_required_projector_is_explicit_bound_and_cannot_be_lost(self):
        fixtures = self.root / "fixtures"; fixtures.mkdir()
        model = write_gguf(fixtures / "main.gguf", architecture="qwen35", name="Qwen 27B", block_count=65, tensor_count=866,
            extra=(_kv("qwen35.nextn_predict_layers", _UINT32, 1), _kv("qwen35.embedding_length", _UINT32, 5120)))
        projector = write_gguf(fixtures / "mmproj.gguf", architecture="clip", name="Qwen 27B", block_count=None, tensor_count=334,
            extra=(_kv("clip.projector_type", _STRING, "qwen3vl_merger"), _kv("clip.vision.projection_dim", _UINT32, 5120)))
        folder, receipt, _ = self.fetch(primary="main.gguf", additional=["mmproj.gguf"],
            files={"main.gguf": model.read_bytes(), "mmproj.gguf": projector.read_bytes()})
        plan = plan_gguf_convert(folder / "main.gguf", projector=str(folder / "mmproj.gguf"), require_projector=True,
            fetch_receipt=str(receipt), require_completion=True)
        self.assertEqual(plan["source"]["projector"]["origin"]["file"], "mmproj.gguf")
        self.assertEqual(len(plan["projector"]["signature"]), 64)
        (folder / "mmproj.gguf").unlink()
        with self.assertRaises(ConvertError):
            plan_gguf_convert(folder / "main.gguf", require_projector=True)

    def _worker_plan(self, exit_code=0, script=None):
        self.sequence += 1
        out = self.root / ("output-" + str(self.sequence))
        source = self.root / "source"; source.mkdir(exist_ok=True)
        body = script or "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); (p/'config.json').write_text('{}'); (p/'model.safetensors').write_bytes(b'weights'); sys.exit(" + str(exit_code) + ")"
        return _finalize_plan({"repo": "fixture/model", "source": {"kind": "hf-cache", "repo": "fixture/model", "path": str(source)},
            "slug": "fixture-" + str(self.sequence), "q_bits": 4, "out": str(out), "argv": [sys.executable, "-c", body, str(out)], "require_completion": True})

    def _start_worker(self, plan, **launch):
        def spawn(argv, log):
            with open(log, "ab") as stream:
                child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.processes.append(child)
            return child.pid
        outcome = start_convert(plan, receipts_dir=self.root, confirm=True, preview_hash=plan["preview_hash"],
            which=lambda _: sys.executable, spawn=spawn, **launch)
        return outcome, self.processes[-1]

    def test_later_same_recipe_conversion_preserves_completed_receipt(self):
        first = self._worker_plan()
        result_a, worker_a = self._start_worker(first, attempt_id="a" * 32)
        self.assertEqual(worker_a.wait(timeout=10), 0)
        receipt_a = Path(result_a["receipt_path"])
        original = receipt_a.read_bytes()
        second = self._worker_plan(); second["slug"] = first["slug"]; _finalize_plan(second)
        result_b, worker_b = self._start_worker(second, attempt_id="b" * 32)
        self.assertEqual(worker_b.wait(timeout=10), 0)
        self.assertNotEqual(result_a["receipt_path"], result_b["receipt_path"])
        self.assertNotEqual(result_a["receipt"]["log_path"], result_b["receipt"]["log_path"])
        self.assertEqual(receipt_a.read_bytes(), original)
        jobs = status_convert(self.root)
        self.assertEqual({job["out"] for job in jobs}, {first["out"], second["out"]})
        self.assertEqual({job["attempt_id"] for job in jobs}, {"a" * 32, "b" * 32})
        self.assertTrue(all(job["state"] == "done" for job in jobs))
        self.assertEqual(result_a["receipt"]["preview_hash"], first["preview_hash"])
        self.assertNotIn("attempt_id", first)

    def test_legacy_completed_recipe_can_convert_again_to_a_fresh_output(self):
        first = self._worker_plan(); first["require_completion"] = False; _finalize_plan(first)
        result_a, worker_a = self._start_worker(first)
        self.assertEqual(worker_a.wait(timeout=10), 0)
        self.assertEqual(status_convert(self.root)[0]["state"], "done")
        second = self._worker_plan(); second.update(require_completion=False, slug=first["slug"]); _finalize_plan(second)
        result_b, worker_b = self._start_worker(second)
        self.assertEqual(worker_b.wait(timeout=10), 0)
        self.assertEqual(result_a["receipt"]["slug"], result_b["receipt"]["slug"])
        self.assertEqual(status_convert(self.root)[0]["out"], second["out"])
        self.assertTrue(Path(first["out"]).is_dir())
        self.assertTrue(Path(second["out"]).is_dir())

    def _historical_queue_receipt(self, item, token):
        directory = self.root / ".mlx-agent-receipts/convert"; directory.mkdir(parents=True, exist_ok=True)
        path = directory / ("old-" + token + ".json")
        path.write_text(json.dumps({"kind": "convert", "pid": 99999, "argv": ["fixture"],
            "repo": item["repo"], "source": {"kind": "hf-cache", "repo": item["repo"]},
            "preview_hash": item["preview_hash"], "out": item["out"], "q_bits": item["q_bits"],
            "require_completion": True, "attempt_id": token, "completed_at": "finished", "exit_status": "failed"}))
        return path

    def test_prepared_queue_run_does_not_consume_historical_failed_attempt(self):
        for restored_start in (False, True):
            with self.subTest(restored_start=restored_start):
                path = self.root / ("queue-" + str(restored_start) + ".json")
                queue = ConvertQueue(path)
                item = queue.enqueue("repo", "p" * 64, 4, repo="org/model", out="/fresh-output", require_completion=True)
                if restored_start:
                    item.update(state="starting", attempt_id="a" * 32)
                    QueueStore(path).save([item])
                previous = self._historical_queue_receipt(item, "b" * 32)
                old_bytes = previous.read_bytes()
                launched = []
                def start(*args, **kwargs):
                    durable = QueueStore(path).load()[0]
                    self.assertEqual(durable["state"], "starting")
                    self.assertRegex(durable["attempt_id"], r"^[0-9a-f]{32}$")
                    self.assertEqual(kwargs["attempt_id"], durable["attempt_id"])
                    self.assertNotIn(durable["attempt_id"], ("a" * 32, "b" * 32))
                    launched.append(durable["attempt_id"])
                    return {"status": "started", "receipt_path": "/actual/new-receipt.json", "receipt": {"attempt_id": durable["attempt_id"]}}
                with patch("mlx_agent.queue_bridge.jobs", return_value={"jobs": [{"state": "failed", "receipt": str(previous)}]}), \
                     patch("mlx_agent.queue_bridge.start_repo", side_effect=start):
                    result = queue_cli.execute({"operation": "run"}, path=path, receipts_dir=self.root)
                self.assertEqual(result["status"], "started")
                self.assertEqual(len(launched), 1)
                self.assertEqual(result["item"]["attempt_id"], launched[0])
                self.assertEqual(result["result"]["receipt_path"], "/actual/new-receipt.json")
                self.assertEqual(QueueStore(path).load(), [])
                self.assertEqual(previous.read_bytes(), old_bytes)

    def test_prepared_start_recovery_requires_exact_attempt_and_returns_path(self):
        path = self.root / "recover-prepared.json"
        queue = ConvertQueue(path)
        item = queue.enqueue("repo", "p" * 64, 4, repo="org/model", out="/fresh-output", require_completion=True)
        item.update(state="starting", attempt_id="a" * 32)
        QueueStore(path).save([item])
        old = self._historical_queue_receipt(item, "b" * 32)
        current = self._historical_queue_receipt(item, "a" * 32)
        with patch("mlx_agent.queue_bridge.jobs", return_value={"jobs": [
            {"state": "failed", "receipt": str(old)}, {"state": "failed", "receipt": str(current)}]}), \
             patch("mlx_agent.queue_bridge.start_repo") as launch:
            result = ConvertQueue(path).try_start_next(str(self.root))
        self.assertEqual(result["status"], "recovered")
        self.assertEqual(result["result"]["receipt_path"], str(current))
        self.assertEqual(result["result"]["receipt"]["attempt_id"], "a" * 32)
        launch.assert_not_called()
        self.assertEqual(QueueStore(path).load(), [])

    def test_prepared_retry_generates_a_new_durable_attempt(self):
        path = self.root / "retry-prepared.json"
        queue = ConvertQueue(path)
        queue.enqueue("repo", "p" * 64, 4, repo="org/model", out="/fresh-output", require_completion=True)
        with patch("mlx_agent.queue_bridge.jobs", return_value={"jobs": []}), \
             patch("mlx_agent.queue_bridge.start_repo", side_effect=queue_bridge.BridgeError("fixture_failed", "failed", "review")):
            self.assertEqual(queue.try_start_next(str(self.root))["status"], "failed")
        failed = queue.snapshot()[0]
        previous = self._historical_queue_receipt(failed, failed["attempt_id"])
        retried = queue.retry(failed["id"])
        self.assertNotIn("attempt_id", retried)
        def start(*args, **kwargs):
            durable = QueueStore(path).load()[0]
            self.assertNotEqual(durable["attempt_id"], failed["attempt_id"])
            self.assertEqual(kwargs["attempt_id"], durable["attempt_id"])
            return {"status": "started", "receipt_path": "/new-attempt.json", "receipt": {"attempt_id": durable["attempt_id"]}}
        with patch("mlx_agent.queue_bridge.jobs", return_value={"jobs": [{"state": "failed", "receipt": str(previous)}]}), \
             patch("mlx_agent.queue_bridge.start_repo", side_effect=start):
            result = queue_cli.execute({"operation": "run"}, path=path, receipts_dir=self.root)
        self.assertEqual(result["status"], "started")
        self.assertNotEqual(result["item"]["attempt_id"], failed["attempt_id"])

    def test_completed_attempt_token_cannot_be_reused_for_another_launch(self):
        plan = self._worker_plan()
        first, child = self._start_worker(plan, attempt_id="a" * 32)
        self.assertEqual(child.wait(timeout=10), 0)
        original = Path(first["receipt_path"]).read_bytes()
        second = self._worker_plan()
        with patch("mlx_agent.convert.spawn_with_env") as launch, self.assertRaises(ConvertError) as error:
            start_convert(second, receipts_dir=self.root, confirm=True, preview_hash=second["preview_hash"],
                          which=lambda _: sys.executable, attempt_id="a" * 32)
        self.assertEqual(error.exception.code, "attempt_exists")
        launch.assert_not_called()
        self.assertEqual(Path(first["receipt_path"]).read_bytes(), original)

    def test_previous_receipt_cannot_claim_same_path_retry_output(self):
        plan = self._worker_plan()
        first, worker_a = self._start_worker(plan, attempt_id="a" * 32)
        self.assertEqual(worker_a.wait(timeout=10), 0)
        original = Path(first["receipt_path"]).read_bytes()
        Path(plan["out"]).rename(self.root / "previous-output")
        second, worker_b = self._start_worker(plan, attempt_id="b" * 32)
        self.assertEqual(worker_b.wait(timeout=10), 0)
        self.assertEqual(first["receipt"]["preview_hash"], second["receipt"]["preview_hash"])
        jobs = status_convert(self.root)
        states = {job["attempt_id"]: job["state"] for job in jobs}
        self.assertEqual(states, {"a" * 32: "failed", "b" * 32: "done"})
        self.assertEqual(Path(first["receipt_path"]).read_bytes(), original)

    def test_real_worker_success_records_observed_exit_and_output_provenance(self):
        plan = self._worker_plan()
        outcome, child = self._start_worker(plan)
        self.assertEqual(child.wait(timeout=10), 0)
        status = next(item for item in status_convert(self.root) if item["out"] == plan["out"])
        self.assertEqual((status["state"], status["completion_observed"], status["exit_code"]), ("done", True, 0))
        self.assertEqual(status["preview_hash"], plan["preview_hash"])
        self.assertTrue(status["processes_quiescent"])
        marker = json.loads((Path(plan["out"]) / "mlx-converter.json").read_text())
        self.assertEqual(marker["source"], plan["source"])
        self.assertEqual(marker["preview_hash"], plan["preview_hash"])
        self.assertFalse(list((self.root / ".mlx-agent-receipts/convert").glob(".monitor-*.json")))

    def test_real_nonzero_worker_with_output_is_failed_not_output_exists_success(self):
        plan = self._worker_plan(exit_code=7)
        _, child = self._start_worker(plan)
        self.assertEqual(child.wait(timeout=10), 1)
        status = next(item for item in status_convert(self.root) if item["out"] == plan["out"])
        self.assertTrue(Path(plan["out"]).exists())
        self.assertEqual((status["state"], status["completion_observed"], status["exit_code"]), ("failed", True, 7))
        self.assertFalse((Path(plan["out"]) / "mlx-converter.json").exists())

    def test_completed_output_drift_cannot_be_adopted_after_restart(self):
        for change in ("weights", "missing-marker", "wrong-preview"):
            with self.subTest(change=change):
                plan = self._worker_plan()
                _, child = self._start_worker(plan)
                self.assertEqual(child.wait(timeout=10), 0)
                output = Path(plan["out"])
                marker = output / "mlx-converter.json"
                if change == "weights":
                    (output / "model.safetensors").write_bytes(b"changed")
                elif change == "missing-marker":
                    marker.unlink()
                else:
                    value = json.loads(marker.read_text()); value["preview_hash"] = "other"
                    marker.write_text(json.dumps(value))
                status = next(item for item in status_convert(self.root) if item["out"] == plan["out"])
                self.assertEqual(status["state"], "failed")
                self.assertTrue(status["completion_observed"])
                self.assertTrue(status["processes_quiescent"])

    def test_dead_monitor_retains_live_worker_as_nonquiescent(self):
        release = self.root / "orphan-release"
        ready = self.root / "orphan-ready"
        self.addCleanup(release.touch)
        script = "from pathlib import Path; import sys,time; Path(sys.argv[1]).mkdir(); Path(" + repr(str(ready)) + ").touch();\nwhile not Path(" + repr(str(release)) + ").exists(): time.sleep(.01)"
        plan = self._worker_plan(script=script)
        _, monitor = self._start_worker(plan)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(ready.exists())
        monitor.kill(); monitor.wait(timeout=5)
        try:
            status = status_convert(self.root)[0]
            self.assertEqual(status["state"], "failed")
            self.assertFalse(status["processes_quiescent"])
        finally:
            release.touch()
            deadline = time.monotonic() + 5
            while not status_convert(self.root)[0].get("processes_quiescent") and time.monotonic() < deadline:
                time.sleep(.01)
        self.assertTrue(status_convert(self.root)[0]["processes_quiescent"])

    def test_quiescence_is_false_when_group_inspection_is_unavailable(self):
        plan = self._worker_plan(exit_code=7)
        _, child = self._start_worker(plan)
        self.assertEqual(child.wait(timeout=10), 1)
        with patch("mlx_agent.compute.group_members", side_effect=OSError("process table unavailable")):
            status = status_convert(self.root)[0]
        self.assertEqual(status["state"], "failed")
        self.assertFalse(status["processes_quiescent"])

    def test_dead_monitor_without_completion_never_uses_output_existence(self):
        plan = self._worker_plan()
        outcome = start_convert(plan, receipts_dir=self.root, confirm=True, preview_hash=plan["preview_hash"],
            which=lambda _: sys.executable, spawn=lambda *_: 99999)
        Path(plan["out"]).mkdir()
        status = status_convert(self.root, pid_alive=lambda _: False)[0]
        self.assertEqual(status["state"], "failed")
        self.assertFalse(status["completion_observed"])
        self.assertIsNone(status["completed_at"])
        self.assertIsNone(outcome["receipt"]["exit_code"])

    def test_monitor_waits_for_its_initial_receipt_before_launching_worker(self):
        from mlx_agent.convert import _write_receipt
        plan = self._worker_plan()
        def delayed(root, receipt, filename, **kwargs):
            time.sleep(0.1)
            self.assertFalse(Path(plan["out"]).exists())
            return _write_receipt(root, receipt, filename, **kwargs)
        with patch("mlx_agent.convert._write_receipt", side_effect=delayed):
            _, child = self._start_worker(plan)
        self.assertEqual(child.wait(timeout=10), 0)

    def test_late_monitor_cannot_publish_into_replaced_receipt(self):
        barrier = self.root / "release"
        entered = self.root / "entered"
        script = "from pathlib import Path; import sys,time; Path(" + repr(str(entered)) + ").write_text('ready');\nwhile not Path(" + repr(str(barrier)) + ").exists(): time.sleep(.01)\nPath(sys.argv[1]).mkdir()"
        plan = self._worker_plan(script=script)
        outcome, child = self._start_worker(plan)
        deadline = time.monotonic() + 5
        while not entered.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(entered.exists())
        receipt = Path(outcome["receipt_path"])
        replacement = json.loads(receipt.read_text())
        replacement["launch_id"] = "b" * 32
        receipt.write_text(json.dumps(replacement))
        barrier.write_text("release")
        self.assertEqual(child.wait(timeout=10), 2)
        self.assertEqual(json.loads(receipt.read_text()), replacement)
        self.assertFalse((Path(plan["out"]) / "mlx-converter.json").exists())

    def test_running_monitor_has_receipt_bound_process_identity(self):
        release = self.root / "release-running"
        script = "from pathlib import Path; import sys,time;\nwhile not Path(" + repr(str(release)) + ").exists(): time.sleep(.01)\nPath(sys.argv[1]).mkdir()"
        plan = self._worker_plan(script=script)
        _, child = self._start_worker(plan)
        status = status_convert(self.root)[0]
        self.assertEqual(status["state"], "running")
        self.assertFalse(status["completion_observed"])
        release.write_text("go")
        self.assertEqual(child.wait(timeout=10), 0)

    def test_worker_source_drift_blocks_success_provenance(self):
        source, receipt, _ = self.fetch()
        plan = plan_convert("org/model", source_path=str(source), fetch_receipt=str(receipt),
            require_completion=True, out=str(self.root / "drifted-output"))
        plan["argv"] = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(); Path(sys.argv[2]).write_bytes(b'changed')", plan["out"], str(source / "model.safetensors")]
        _finalize_plan(plan)
        _, child = self._start_worker(plan)
        self.assertEqual(child.wait(timeout=10), 1)
        status = next(item for item in status_convert(self.root) if item["out"] == plan["out"])
        self.assertEqual((status["state"], status["completion_observed"], status["exit_code"]), ("failed", True, 0))
        self.assertFalse((Path(plan["out"]) / "mlx-converter.json").exists())

    def test_wrong_dependency_revision_is_not_resolved_through_cache(self):
        repo = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
        lora = "qwen-image-2.1-uncensored-lora.safetensors"
        source, receipt, _ = self.fetch(repo=repo, files={lora: b"lora"})
        _, dependency, _ = self.fetch(repo="Qwen/Qwen-Image-2.1", revision="b" * 40)
        with patch("mlx_agent.convert.cached_snapshot", side_effect=AssertionError("shared cache used")), self.assertRaises(ConvertError):
            plan_convert(repo, backend="mflux", model_type="qwen_image_21", source_path=str(source),
                fetch_receipt=str(receipt), dependency_fetch_receipts=[str(dependency)], require_completion=True)

    def test_fetch_marker_from_another_launch_cannot_claim_completion(self):
        _, receipt, _ = self.fetch()
        data = json.loads(receipt.read_text())
        marker = Path(data["marker"])
        payload = json.loads(marker.read_text()); payload["proof_token"] = "other-launch"
        marker.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            completed_fetch(receipt)
        status = status_fetch(self.root, pid_alive=lambda _: False)[0]
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["files"], [])

    def test_later_same_selection_fetch_does_not_replace_queued_source_proof(self):
        first, receipt_a, _ = self.fetch(primary="main.gguf", files={"main.gguf": b"first"})
        original = receipt_a.read_bytes()
        proof = completed_fetch(receipt_a)
        second, receipt_b, _ = self.fetch(primary="main.gguf", files={"main.gguf": b"second"}, revision="b" * 40)
        self.assertNotEqual(receipt_a, receipt_b)
        self.assertNotEqual(first, second)
        self.assertEqual(receipt_a.read_bytes(), original)
        self.assertEqual(completed_fetch(receipt_a), proof)
        self.assertEqual(completed_fetch(receipt_b)["origin"]["revision"], "b" * 40)

    def test_success_waits_for_late_writing_worker_descendant(self):
        descendant = "from pathlib import Path; import sys,time; time.sleep(.25); (Path(sys.argv[1])/'model.safetensors').write_bytes(b'final-weights')"
        script = "from pathlib import Path; import sys,subprocess; p=Path(sys.argv[1]); p.mkdir(); (p/'model.safetensors').write_bytes(b'initial'); subprocess.Popen([sys.executable,'-c'," + repr(descendant) + ",sys.argv[1]])"
        plan = self._worker_plan(script=script)
        _, child = self._start_worker(plan)
        self.assertEqual(child.wait(timeout=10), 0)
        output = Path(plan["out"])
        self.assertEqual((output / "model.safetensors").read_bytes(), b"final-weights")
        status = status_convert(self.root)[0]
        self.assertEqual((status["state"], status["completion_observed"]), ("done", True))
        from mlx_agent.source_evidence import fingerprint
        self.assertEqual(json.loads((output / "mlx-converter.json").read_text())["output_fingerprint"], fingerprint(output, exclude=("mlx-converter.json",)))

    def test_canonical_worker_waits_for_compute_registration_barrier(self):
        from mlx_agent.convert import register_monitor_lease
        plan = self._worker_plan()
        def spawn(argv, log):
            with open(log, "ab") as stream:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.processes.append(process)
            return process.pid
        result = start_convert(plan, receipts_dir=self.root, confirm=True, preview_hash=plan["preview_hash"],
            which=lambda _: sys.executable, spawn=spawn, wait_for_compute_lease=True)
        time.sleep(.1)
        self.assertFalse(Path(plan["out"]).exists())
        register_monitor_lease(result["receipt_path"], result["receipt"])
        self.assertEqual(self.processes[-1].wait(timeout=10), 0)

    def test_port_dependencies_are_explicit_managed_file_bundles(self):
        source, receipt, plan = self.fetch(repo="microsoft/TRELLIS.2-4B", model_type="trellis2")
        self.assertNotIn("--require", plan["argv"])
        expected = {item["repo"]: item for item in plan["required_dependencies"]}
        self.assertIsNone(expected["facebook/dinov3-vitl16-pretrain-lvd1689m"]["revision"])
        first, first_receipt, _ = self.fetch(repo="facebook/dinov3-vitl16-pretrain-lvd1689m", primary="config.json", additional=["model.safetensors"])
        second_files = {"ckpts/ss_dec_conv3d_16l8_fp16.json": b"{}", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors": b"weights"}
        second, second_receipt, _ = self.fetch(repo="microsoft/TRELLIS-image-large", primary="ckpts/ss_dec_conv3d_16l8_fp16.json",
            additional=["ckpts/ss_dec_conv3d_16l8_fp16.safetensors"], files=second_files)
        with patch("mlx_agent.convert.cached_snapshot", side_effect=AssertionError("shared cache used")):
            conversion = plan_convert("microsoft/TRELLIS.2-4B", backend="mlx-3d", model_type="trellis2", q_bits=32,
                source_path=str(source), fetch_receipt=str(receipt), dependency_fetch_receipts=[str(first_receipt), str(second_receipt)], require_completion=True)
        self.assertEqual(conversion["requires"]["facebook/dinov3-vitl16-pretrain-lvd1689m"], str(first))
        self.assertEqual(conversion["requires"]["microsoft/TRELLIS-image-large"], str(second))
        self.assertEqual(conversion["source"]["dependencies"][0]["origin"]["file"], "config.json")
        self.assertEqual(conversion["source"]["dependencies"][0]["path"], str(first))

    def test_http_support_stays_in_managed_source_and_is_bound_to_conversion(self):
        payload = b"fixture support"
        asset = {"url": "https://example.org/files/naf.pth", "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        with patch("mlx_agent.intake_fetch.port_downloads", return_value=[asset]), \
             patch("mlx_agent.intake_fetch.downloads_root", side_effect=AssertionError("default downloads used")):
            source, receipt, plan = self.fetch(repo="TencentARC/Pixal3D", model_type="pixal3d", opener=lambda _: io.BytesIO(payload))
        support = source / "support"
        self.assertEqual(plan["downloads_dir"], str(support))
        target = support / asset["sha256"] / "naf.pth"
        self.assertEqual(target.read_bytes(), payload)
        proof = completed_fetch(receipt)
        self.assertEqual(proof["support"]["downloads"], [asset])
        self.assertIn(str(target.relative_to(source)), {value["path"] for value in status_fetch(self.root)[0]["files"]})
        with patch("mlx_agent.convert.port_downloads", return_value=[asset]), \
             patch("mlx_agent.convert.port_requirements", return_value={}), \
             patch("mlx_agent.backends.downloads_root", side_effect=AssertionError("default downloads used")):
            conversion = plan_convert("TencentARC/Pixal3D", backend="mlx-3d", model_type="pixal3d", q_bits=32,
                source_path=str(source), fetch_receipt=str(receipt), require_completion=True)
        self.assertEqual(conversion["downloads_dir"], str(support))
        self.assertEqual(conversion["source"]["support"], proof["support"])
        with patch("mlx_agent.intake_fetch.port_downloads", return_value=[asset]), self.assertRaises(FetchError):
            plan_fetch("TencentARC/Pixal3D", revision=COMMIT, model_type="pixal3d", local_dir=str(self.root / "new"), downloads_dir=str(self.root / "outside"))

    def test_queue_preparation_intent_survives_storage_and_exact_cli_forwarding(self):
        queue = ConvertQueue(self.root / "queue.json")
        intent = dict(backend="mflux", model_type="qwen_image_21", subfolder="checkpoint", source_path="/managed/source",
            fetch_receipt="/state/primary.json", dependency_fetch_receipts=["/state/base.json"], require_completion=True, downloads_dir="/managed/source/support")
        item = queue.enqueue("repo", "hash", 32, repo="org/model", out="/managed/new", **intent)
        self.assertEqual(QueueStore(self.root / "queue.json").load()[0], item)
        commands = []
        def runner(argv, timeout):
            commands.append(argv)
            return {"stdout": json.dumps({"status": "ok", "data": {"status": "started"}})}
        queue_bridge.start_repo("/agent", "org/model", "hash", q_bits=32, out="/managed/new", runner=runner, **intent)
        argv = commands[0]
        for flag, value in (("--backend", "mflux"), ("--source-path", "/managed/source"), ("--fetch-receipt", "/state/primary.json"),
                            ("--dependency-fetch-receipt", "/state/base.json"), ("--downloads-dir", "/managed/source/support")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--require-completion", argv)
        saved = json.loads((self.root / "queue.json").read_text())
        saved["schema_version"] = "1.1"
        (self.root / "queue.json").write_text(json.dumps(saved))
        self.assertEqual(QueueStore(self.root / "queue.json").load()[0], item)


if __name__ == "__main__":
    unittest.main()
