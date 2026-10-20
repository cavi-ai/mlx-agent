"""Exercise real task processes with fixture backends, not model inference."""

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import unittest
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.backends import INSTALL_MARKER, load_manifests
from mlx_agent.compute import ComputeBusy, ComputeLease, group_members, process_identity
from mlx_agent.task_contracts import TaskError
from mlx_agent.task_runtime import cancel_task, start_task, supervise, task_result, task_status, _stop
from mlx_agent.task_store import TaskStore


BACKEND = r'''
import json, os, signal, subprocess, sys, time, wave
import base64
from pathlib import Path
args = sys.argv[1:]
png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGPgEpEDAABoAD1UCKP3AAAAAElFTkSuQmCC")
if args and args[0].endswith("task_png_validator.py"):
    image = Path(args[1]).read_bytes()
    sys.exit(0 if image == png else 1)
if args and args[0].endswith("task_mp4_validator.py"):
    sys.exit(0 if Path(args[1]).read_bytes() == b"fixture video output" else 1)
model = Path(args[args.index("--model") + 1])
config = json.loads((model / "config.json").read_text())
behavior = config.get("fixture_behavior", "success")
if behavior == "descendant":
    child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    (model.parent / "descendant.pid").write_text(str(child.pid))
if behavior == "ignore-term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
(model.parent / "started").write_text(str(os.getpid()))
if behavior in ("sleep", "descendant", "ignore-term"):
    time.sleep(60)
if "--audio" in args:
    while behavior == "hold-transcription" and not (model.parent / "release-transcription").exists():
        time.sleep(.01)
    audio = Path(args[args.index("--audio") + 1])
    print(json.dumps({"text": audio.read_bytes().decode("utf-8"), "seconds": .1, "audio_seconds": .25}))
elif "--out" in args:
    target = args[args.index("--out") + 1]
    if target.endswith(".png"):
        Path(target).write_bytes(b"not png" if behavior == "invalid-image" else png)
        print(json.dumps({"path": target, "width": 1, "height": 1, "steps": 40, "seed": 42, "seconds": .1}))
    elif target.endswith(".mp4"):
        Path(target).write_bytes(b"invalid video" if behavior == "invalid-video" else b"fixture video output")
        print(json.dumps({"path": target, "width": 832, "height": 480, "frames": 81, "fps": 16,
                          "duration_seconds": 5.0625, "steps": 50, "seed": 42, "seconds": .1,
                          "seconds_per_frame": .001, "peak_memory_gb": 1.0, "pixel_std": 1.0}))
    elif behavior == "invalid-audio":
        Path(target).write_bytes(b"not audio")
    else:
        with wave.open(target, "wb") as output:
            output.setparams((1, 2, 8000, 2000, "NONE", "not compressed"))
            output.writeframes(b"\x00\x01" * 2000)
        if behavior == "truncated-audio":
            Path(target).write_bytes(Path(target).read_bytes()[:46])
    if not target.endswith(".png"):
        print(json.dumps({"path": target, "sample_rate": 8000, "audio_seconds": .25, "seconds": .1}))
elif "--image" in args:
    image = Path(args[args.index("--image") + 1])
    while behavior == "hold-image-description" and not (model.parent / "release-image-description").exists():
        time.sleep(.01)
    print(json.dumps({"text": "A detailed description. " + ("More scene detail. " * 150),
                      "prompt_tokens": 12, "generation_tokens": 300, "seconds": .1}))
elif "--video" in args:
    while behavior == "hold-video-description" and not (model.parent / "release-video-description").exists():
        time.sleep(.01)
    if behavior == "video-unsupported":
        print(json.dumps({"error": "video_unsupported", "detail": "fixture backend has no video decoder"}))
        raise SystemExit(3)
    video = Path(args[args.index("--video") + 1])
    print(json.dumps({"text": "A detailed video description. " + ("The scene changes over time. " * 100),
                      "prompt_tokens": 14, "generation_tokens": 400, "seconds": .1}))
else:
    request = json.loads(Path(args[args.index("--request") + 1]).read_text())
    print(json.dumps({"answers": {"route": "billing"}, "seconds": .1, "usage": {"questions": len(request["questions"])}}))
'''


def launch_inference_supervisor(state, compute, argv, **kwargs):
    identifier = argv[argv.index("--id") + 1]
    code = ("from functools import partial; from pathlib import Path; "
            "from mlx_agent.compute import ComputeLease; from mlx_agent.task_runtime import supervise; "
            "raise SystemExit(supervise({!r}, Path({!r}), lease_factory=partial(ComputeLease, root=Path({!r}))))").format(identifier, str(state), str(compute))
    return subprocess.Popen([sys.executable, "-c", code], **kwargs)


class TaskProcessTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.state = self.root / "tasks"
        self.compute = self.root / "compute"
        self.processes = []
        self.addCleanup(self.stop_processes)
        self.environment = patch.dict(os.environ, {"XDG_DATA_HOME": str(self.root / "data")})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.model = self.root / "model"
        self.model.mkdir()
        for identifier in ("mlx-audio", "mlx-embeddings", "mflux", "mlx-vlm", "mlx-video"):
            manifest = load_manifests()[identifier]
            backend = self.root / "data/mlx-workbench/backends" / identifier
            (backend / "bin").mkdir(parents=True)
            executable = backend / "bin/python"
            executable.write_text("#!" + sys.executable + "\n" + BACKEND)
            executable.chmod(0o700)
            (backend / INSTALL_MARKER).write_text(json.dumps({"id": identifier, "version": manifest["version"]}))
            (backend / "lib/python3.12/site-packages" / manifest["package"]).mkdir(parents=True)

    def stop_processes(self):
        # Only process groups created by this test are eligible for cleanup.
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    if group_members(process.pid):
                        raise
            process.wait(timeout=10)
        if self.state.exists():
            for record in TaskStore(self.state).list(100):
                execution = record.get("execution")
                if execution and group_members(execution["pid"]):
                    try:
                        os.killpg(execution["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def launch(self, argv, **kwargs):
        process = launch_inference_supervisor(self.state, self.compute, argv, **kwargs)
        self.processes.append(process)
        return process

    def prepare_request(self, kind="classification", behavior="success"):
        model_type = {"classification": "laya", "music_generation": "minimax_music3",
                      "transcribe": "whisper", "speak": "kokoro", "image_generation": "qwen_image_21",
                      "image_understanding": "qwen2_5_vl", "video_understanding": "qwen2_5_vl",
                      "video_generation": "t2v"}[kind]
        (self.model / "config.json").write_text(json.dumps({"model_type": model_type, "fixture_behavior": behavior}))
        (self.model / "tokenizer.json").write_text("{}")
        schema = "task-request/1"
        if kind == "classification":
            params = {"state": "refund", "questions": {"route": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": ["billing", "other"]}}}
        elif kind == "music_generation":
            params = {"caption": "Quiet piano"}
        elif kind == "transcribe":
            self.audio = self.root / "input.wav"
            self.audio.write_bytes(b"original speech snapshot")
            params = {"audio_path": str(self.audio)}
            schema = "task-request/2"
        elif kind == "speak":
            params = {"text": "Hello there."}
            schema = "task-request/2"
        elif kind == "image_generation":
            params = {"prompt": "A quiet forest"}
            schema = "task-request/2"
        elif kind == "image_understanding":
            self.image = self.root / "input.png"
            self.image.write_bytes(b"original image snapshot")
            params = {"image_path": str(self.image), "prompt": "Describe the scene."}
            schema = "task-request/2"
        elif kind == "video_understanding":
            self.video = self.root / "input.MOV"
            self.video.write_bytes(b"original video snapshot")
            params = {"video_path": str(self.video), "prompt": "What changes?"}
            schema = "task-request/2"
        elif kind == "video_generation":
            params = {"prompt": "A slow camera pan."}
            schema = "task-request/2"
        request = self.root / "request.json"
        request.write_text(json.dumps({"schema": schema, "kind": kind, "model_path": str(self.model), "parameters": params}))
        return request

    def start(self, kind="classification", behavior="success", launcher=None):
        return start_task(self.prepare_request(kind, behavior), self.state, launcher=launcher or self.launch)

    def wait_for(self, condition, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = condition()
            if value:
                return value
            time.sleep(.05)
        diagnostics = {
            "tasks": TaskStore(self.state).list(100),
            "supervisors": [(process.pid, process.poll()) for process in self.processes],
        }
        self.fail("Task condition was not reached before its deadline: {!r}".format(diagnostics))

    def terminal(self, identifier):
        value = task_status(identifier, self.state)
        return value if value["state"] in ("succeeded", "failed", "cancelled") else None

    def test_real_workers_preserve_classifier_and_music_results(self):
        for kind in ("classification", "music_generation"):
            with self.subTest(kind=kind):
                task = self.start(kind)
                result = self.wait_for(lambda: self.terminal(task["id"]))
                self.assertEqual(result["state"], "succeeded", result)
                evidence = task_result(task["id"], self.state)
                self.assertEqual(evidence["result"]["schema"], "decide/1" if kind == "classification" else "music/1")
                self.assertEqual(evidence["result"]["model"], str(self.model))
                self.assertTrue(Path(evidence["artifact"]["path"]).is_file())
                self.processes[-1].wait(timeout=10)

    def test_transcription_uses_its_fsynced_snapshot_after_original_changes(self):
        task = self.start("transcribe", behavior="hold-transcription")
        self.wait_for(lambda: (self.root / "started").exists())
        self.audio.write_bytes(b"mutated original")
        self.audio.unlink()
        (self.root / "release-transcription").touch()
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "succeeded", result)
        evidence = task_result(task["id"], self.state)["result"]
        self.assertEqual(evidence["schema"], "transcribe/1")
        self.assertEqual(evidence["text"], "original speech snapshot")
        self.assertEqual(evidence["input_artifact"], result["input_artifact"])
        self.assertEqual(Path(result["input_artifact"]["path"]).read_bytes(), b"original speech snapshot")

    def test_speech_synthesis_returns_only_a_validated_wav_artifact(self):
        task = self.start("speak")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "succeeded", result)
        evidence = task_result(task["id"], self.state)["result"]
        self.assertEqual(evidence["schema"], "speak/1")
        self.assertIsNone(result["input_artifact"])
        output = evidence["output_artifact"]
        self.assertEqual(Path(output["path"]).name, "output.wav")
        self.assertTrue(Path(output["path"]).is_file())

    def test_speech_synthesis_rejects_invalid_wav_before_success(self):
        task = self.start("speak", behavior="invalid-audio")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(result["error"]["code"], "invalid_audio_output")

    def test_speech_synthesis_rejects_truncated_wav_payload_before_success(self):
        task = self.start("speak", behavior="truncated-audio")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(result["error"]["code"], "invalid_audio_output")

    def test_image_description_uses_its_fsynced_snapshot_and_preserves_full_text(self):
        task = self.start("image_understanding", behavior="hold-image-description")
        self.wait_for(lambda: (self.root / "started").exists())
        self.image.write_bytes(b"changed image")
        self.image.unlink()
        (self.root / "release-image-description").touch()
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "succeeded", result)
        evidence = task_result(task["id"], self.state)
        self.assertEqual(evidence["result"]["schema"], "describe/1")
        self.assertGreater(len(evidence["result"]["text"]), 2000)
        self.assertEqual(evidence["result"]["input_artifact"], result["input_artifact"])
        self.assertEqual(Path(result["input_artifact"]["path"]).read_bytes(), b"original image snapshot")

    def test_image_generation_returns_only_a_decodable_png_artifact(self):
        task = self.start("image_generation")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "succeeded", result)
        evidence = task_result(task["id"], self.state)["result"]
        self.assertEqual(evidence["schema"], "generate/1")
        self.assertIsNone(result["input_artifact"])
        self.assertEqual(Path(evidence["output_artifact"]["path"]).name, "output.png")

    def test_image_generation_rejects_undecodable_png_before_success(self):
        task = self.start("image_generation", behavior="invalid-image")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(result["error"]["code"], "invalid_image_output")

    def test_video_description_uses_its_fsynced_snapshot_and_preserves_full_text(self):
        task = self.start("video_understanding", behavior="hold-video-description")
        self.wait_for(lambda: (self.root / "started").exists())
        self.video.write_bytes(b"changed video")
        self.video.unlink()
        (self.root / "release-video-description").touch()
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "succeeded", result)
        evidence = task_result(task["id"], self.state)
        self.assertEqual(evidence["result"]["schema"], "describe/1")
        self.assertGreater(len(evidence["result"]["text"]), 2000)
        self.assertEqual(evidence["result"]["input_artifact"], result["input_artifact"])
        self.assertEqual(Path(result["input_artifact"]["path"]).read_bytes(), b"original video snapshot")

    def test_video_description_preserves_explicit_unsupported_error(self):
        task = self.start("video_understanding", behavior="video-unsupported")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(result["error"]["code"], "video_unsupported")

    def test_video_generation_returns_mp4_only_after_backend_validation(self):
        task = self.start("video_generation")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "succeeded", result)
        evidence = task_result(task["id"], self.state)["result"]
        self.assertEqual(evidence["schema"], "video/1")
        self.assertIsNone(result["input_artifact"])
        self.assertEqual(Path(evidence["output_artifact"]["path"]).name, "output.mp4")

    def test_video_generation_rejects_invalid_mp4_before_success(self):
        task = self.start("video_generation", behavior="invalid-video")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(result["error"]["code"], "invalid_video_output")

    def test_video_model_constraints_fail_before_task_acceptance(self):
        request_path = self.prepare_request("video_generation")
        request = json.loads(request_path.read_text())
        request["parameters"]["frames"] = 18
        request_path.write_text(json.dumps(request))
        with self.assertRaises(TaskError) as caught:
            start_task(request_path, self.state, launcher=self.launch)
        self.assertEqual(caught.exception.code, "invalid_arguments")
        self.assertEqual(TaskStore(self.state).list(), [])

    def test_invalid_speech_parameters_fail_in_planner_before_acceptance(self):
        request_path = self.prepare_request("speak")
        request = json.loads(request_path.read_text())
        request["parameters"]["speed"] = 3.0
        request_path.write_text(json.dumps(request))
        with self.assertRaises(TaskError) as caught:
            start_task(request_path, self.state, launcher=self.launch)
        self.assertEqual(caught.exception.code, "invalid_arguments")
        self.assertEqual(TaskStore(self.state).list(), [])

    def test_cancel_waits_for_descendants(self):
        task = self.start(behavior="descendant")
        self.wait_for(lambda: (self.root / "started").exists())
        descendant = int((self.root / "descendant.pid").read_text())
        requested = cancel_task(task["id"], self.state)
        self.assertTrue(requested["cancel_requested"])
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "cancelled", result)
        self.assertFalse(result["execution_unresolved"])
        from mlx_agent.compute import group_members
        self.assertNotIn(descendant, group_members(result["execution"]["pid"]))

    def test_supervisor_loss_keeps_compute_busy(self):
        task = self.start(behavior="sleep")
        self.wait_for(lambda: (self.root / "started").exists())
        supervisor = self.processes[-1]
        supervisor.kill()
        supervisor.wait(timeout=10)
        result = task_status(task["id"], self.state)
        self.assertEqual(result["state"], "failed")
        self.assertTrue(result["execution_unresolved"])
        with self.assertRaises(ComputeBusy):
            with ComputeLease("another model", self.compute):
                pass

    def test_cancel_during_spawn(self):
        task = self.start(behavior="sleep")
        cancel_task(task["id"], self.state)
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "cancelled", result)
        self.assertFalse((self.root / "started").exists())

    def test_caller_exit_does_not_end_task(self):
        request = self.prepare_request()
        code = ("import json; from functools import partial; from pathlib import Path; "
                "from tests.integration.test_task_processes import launch_inference_supervisor; "
                "from mlx_agent.task_runtime import start_task; "
                "print(json.dumps(start_task(Path({!r}), Path({!r}), launcher=partial(launch_inference_supervisor, {!r}, {!r}))))").format(str(request), str(self.state), str(self.state), str(self.compute))
        caller = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        self.assertEqual(caller.returncode, 0, caller.stderr)
        record = json.loads(caller.stdout)
        result = self.wait_for(lambda: self.terminal(record["id"]))
        self.assertEqual(result["state"], "succeeded", result)
        self.assertEqual(task_result(record["id"], self.state)["result"]["answers"], {"route": "billing"})

    def test_term_escalates_to_kill(self):
        record = self.start(behavior="ignore-term")
        self.wait_for(lambda: (self.root / "started").exists())
        cancel_task(record["id"], self.state)
        result = self.wait_for(lambda: self.terminal(record["id"]))
        self.assertEqual(result["state"], "cancelled", result)
        self.assertEqual(group_members(result["execution"]["pid"]), [])

    def test_cancel_after_python_launcher_reexec_preserves_worker_identity(self):
        import mlx_agent.task_runtime as task_runtime

        record = self.start(behavior="ignore-term", launcher=lambda *args, **kwargs: None)
        launcher = self.root / "python-launcher"
        launcher.write_text("#!/bin/sh\n/bin/sleep 0.5\nexec {0} \"$@\"\n".format(shlex.quote(sys.executable)))
        launcher.chmod(0o700)
        original_spawn = task_runtime._spawn

        def launch_through_shim(argv, **kwargs):
            process = subprocess.Popen([str(launcher), *argv[1:]], **kwargs)
            self.processes.append(process)
            return process

        def spawn(role, identifier, store, process_launcher=subprocess.Popen):
            if role == "execute":
                return original_spawn(role, identifier, store, launcher=launch_through_shim)
            return original_spawn(role, identifier, store, launcher=process_launcher)

        cancel_errors = []
        identity_observations = []

        def cancel_after_backend_starts():
            try:
                self.wait_for(lambda: (self.root / "started").exists())
                execution = task_status(record["id"], self.state)["execution"]
                identity_observations.append((execution["identity"], process_identity(execution["pid"])))
                cancel_task(record["id"], self.state)
            except BaseException as error:
                cancel_errors.append(error)

        cancellation = threading.Thread(target=cancel_after_backend_starts)
        cancellation.start()
        with patch.object(task_runtime, "_spawn", spawn):
            task_runtime.supervise(record["id"], self.state, lease_factory=partial(ComputeLease, root=self.compute))
        cancellation.join(timeout=10)
        self.assertFalse(cancellation.is_alive(), "Cancellation request did not finish")
        self.assertEqual(cancel_errors, [])
        self.assertEqual(len(identity_observations), 1)
        self.assertEqual(identity_observations[0][0], identity_observations[0][1])
        result = task_status(record["id"], self.state)
        self.assertEqual(result["state"], "cancelled", result)
        self.assertFalse(result["execution_unresolved"])
        self.assertEqual(group_members(result["execution"]["pid"]), [])

    def test_pid_reuse_never_signals_unrelated_process(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        self.processes.append(process)
        self.assertFalse(_stop(process, "a different process identity", {}, grace=.01))
        self.assertIsNone(process.poll())

    def test_busy_lease_is_a_retained_failure(self):
        with ComputeLease("existing model", self.compute):
            record = self.start()
            result = self.wait_for(lambda: self.terminal(record["id"]))
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"]["code"], "compute_busy")
        self.assertFalse((self.root / "started").exists())

    def test_launch_failure_is_retained(self):
        def unavailable(*args, **kwargs):
            raise OSError("fixture launch failure")
        result = self.start(launcher=unavailable)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"]["code"], "task_launch_failed")
        self.assertEqual(TaskStore(self.state).get(result["id"]), result)

    def test_changed_identity_before_execution_is_refused(self):
        def changed(argv, **kwargs):
            (self.model / "config.json").write_text('{"model_type":"changed"}')
            return self.launch(argv, **kwargs)
        task = self.start(launcher=changed)
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"]["code"], "identity_changed")
        self.assertFalse((self.root / "started").exists())

    def test_invalid_wav_is_not_success(self):
        task = self.start("music_generation", "invalid-audio")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "failed")
        self.assertIsNone(result["result"])

    def test_result_storage_failure_is_not_success(self):
        task = self.start(launcher=lambda *args, **kwargs: None)
        with patch.object(TaskStore, "write_result", side_effect=TaskError("task_storage_failed", "disk full")):
            supervise(task["id"], self.state, lease_factory=partial(ComputeLease, root=self.compute))
        result = task_status(task["id"], self.state)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"]["code"], "task_storage_failed")

    def test_success_cancel_race(self):
        task = self.start(launcher=lambda *args, **kwargs: None)
        original_write = TaskStore.write_result
        def cancel_after_write(store, identifier, value):
            reference = original_write(store, identifier, value)
            store.request_cancel(identifier)
            return reference
        with patch.object(TaskStore, "write_result", cancel_after_write):
            supervise(task["id"], self.state, lease_factory=partial(ComputeLease, root=self.compute))
        result = task_status(task["id"], self.state)
        self.assertEqual(result["state"], "cancelled")
        self.assertIsNone(result["result"])

    def test_changed_music_output_is_not_returned_as_valid_evidence(self):
        task = self.start("music_generation")
        result = self.wait_for(lambda: self.terminal(task["id"]))
        self.assertEqual(result["state"], "succeeded", result)
        evidence = task_result(task["id"], self.state)
        Path(evidence["result"]["path"]).write_bytes(b"changed")
        with self.assertRaises(TaskError):
            task_result(task["id"], self.state)

    def test_cancel_while_execution_identity_is_being_bound(self):
        from mlx_agent import task_runtime
        task = self.start(launcher=lambda *args, **kwargs: None)
        spawn = task_runtime._spawn
        def cancel_after_spawn(role, identifier, store, **kwargs):
            process = spawn(role, identifier, store, **kwargs)
            store.request_cancel(identifier)
            process.wait(timeout=10)
            return process
        with patch.object(task_runtime, "_spawn", cancel_after_spawn):
            supervise(task["id"], self.state, lease_factory=partial(ComputeLease, root=self.compute))
        result = task_status(task["id"], self.state)
        self.assertEqual(result["state"], "cancelled", result)
        self.assertFalse((self.root / "started").exists())
