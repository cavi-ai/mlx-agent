import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from mlx_agent.task_contracts import TaskError
from mlx_agent.task_kinds import KINDS, check_mp4, check_png
from mlx_agent.task_runtime import cancel_task, start_task, supervise, task_result, task_status
from mlx_agent.task_mp4_validator import valid_mp4
from mlx_agent.task_store import TaskStore
from tests.task_fixtures import identity_evidence


def _plan(request, directory, input_path=None):
    return KINDS[request["kind"]].plan(request, directory, input_path)


class TaskRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.state = self.root / "tasks"
        self.store = TaskStore(self.state)
        self.request = {"schema": "task-request/1", "kind": "classification", "model_path": str(self.root / "missing-model"), "parameters": {"state": "refund", "questions": {"route": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": ["billing", "other"]}}}}

    def test_speech_planners_receive_only_supplied_optional_values(self):
        with patch("mlx_agent.task_kinds.plan_transcribe", return_value={}) as transcribe, patch(
                "mlx_agent.task_kinds.plan_speak", return_value={}) as speak:
            _plan({"kind": "transcribe", "model_path": "/model", "parameters": {"audio_path": "/audio.wav"}},
                  self.root, "/owned/input.wav")
            transcribe.assert_called_once_with("/model", "/owned/input.wav")
            _plan({"kind": "speak", "model_path": "/model", "parameters": {"text": "Hello."}}, self.root)
            speak.assert_called_once_with("/model", "Hello.", str(self.root / "output.wav"))

    def test_image_planners_receive_snapshots_and_only_supplied_optional_values(self):
        with patch("mlx_agent.task_kinds.plan_generate", return_value={}) as generate, patch(
                "mlx_agent.task_kinds.plan_describe", return_value={}) as describe:
            _plan({"kind": "image_generation", "model_path": "/model",
                   "parameters": {"prompt": "A quiet forest"}}, self.root)
            generate.assert_called_once_with("/model", "A quiet forest", str(self.root / "output.png"))
            _plan({"kind": "image_understanding", "model_path": "/vision",
                   "parameters": {"image_path": "/original/source.jpeg", "prompt": "Describe."}},
                  self.root, "/owned/input.jpeg")
            describe.assert_called_once_with("/vision", "Describe.", image="/owned/input.jpeg")

    def test_video_planners_receive_snapshots_and_only_supplied_optional_values(self):
        with patch("mlx_agent.task_kinds.plan_video", return_value={}) as generate, patch(
                "mlx_agent.task_kinds.plan_describe", return_value={}) as describe:
            _plan({"kind": "video_generation", "model_path": "/video",
                   "parameters": {"prompt": "A slow camera pan."}}, self.root)
            generate.assert_called_once_with("/video", "A slow camera pan.", str(self.root / "output.mp4"))
            _plan({"kind": "video_understanding", "model_path": "/vision",
                   "parameters": {"video_path": "/original/source.mov", "prompt": "What changes?"}},
                  self.root, "/owned/input.mov")
            describe.assert_called_once_with("/vision", "What changes?", video="/owned/input.mov")
            _plan({"kind": "video_generation", "model_path": "/video",
                   "parameters": {"prompt": "A pan.", "width": 256, "height": 256, "frames": 17,
                                  "fps": 8, "steps": 8, "seed": 42}}, self.root)
            generate.assert_called_with("/video", "A pan.", str(self.root / "output.mp4"),
                                        width=256, height=256, frames=17, fps=8, steps=8, seed=42)
            _plan({"kind": "video_understanding", "model_path": "/vision",
                   "parameters": {"video_path": "/original/source.mp4", "prompt": "What changes?",
                                  "fps": 2, "temperature": .25, "max_tokens": 32, "max_pixels": 4096}},
                  self.root, "/owned/input.mp4")
            describe.assert_called_with("/vision", "What changes?", video="/owned/input.mp4",
                                        fps=2, max_tokens=32, temperature=.25, max_pixels=4096)

    def test_mp4_validation_runs_in_selected_backend_and_fails_closed(self):
        plan = {"argv": ["/backend/python"]}
        with patch("mlx_agent.task_kinds.backend_environment", return_value={}) as environment:
            runner = unittest.mock.Mock(return_value=unittest.mock.Mock(returncode=0))
            check_mp4(self.root / "output.mp4", plan, runner=runner)
            self.assertEqual(runner.call_args.args[0][0:2], [
                "/backend/python", str(Path(__file__).resolve().parents[2] / "src/mlx_agent/task_mp4_validator.py")])
            environment.assert_called_once_with()
            with self.assertRaises(TaskError) as caught:
                check_mp4(self.root / "bad.mp4", plan,
                          runner=unittest.mock.Mock(return_value=unittest.mock.Mock(returncode=1)))
            self.assertEqual(caught.exception.code, "invalid_video_output")

    def test_mp4_validator_rejects_truncated_frame_stream(self):
        class Capture:
            def __init__(self, _path):
                self.frames = iter(([SimpleNamespace(size=1)], []))

            def isOpened(self):
                return True

            def get(self, property_id):
                return {1: 8.0, 2: 4.0, 3: 64.0, 4: 64.0}[property_id]

            def read(self):
                frame = next(self.frames)
                return (bool(frame), frame[0] if frame else None)

            def release(self):
                pass

        fake_cv2 = SimpleNamespace(VideoCapture=Capture, CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2,
                                   CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4)
        path = self.root / "truncated.mp4"
        path.write_bytes(b"fixture")
        with patch.dict("sys.modules", {"cv2": fake_cv2}):
            self.assertFalse(valid_mp4(path))

    def test_png_validation_runs_in_the_selected_backend_and_fails_closed(self):
        plan = {"argv": ["/backend/python"]}
        with patch("mlx_agent.task_kinds.backend_environment", return_value={}) as environment:
            runner = unittest.mock.Mock(return_value=unittest.mock.Mock(returncode=0))
            check_png(self.root / "output.png", plan, runner=runner)
            self.assertEqual(runner.call_args.args[0][0:2], ["/backend/python", str(Path(__file__).resolve().parents[2] / "src/mlx_agent/task_png_validator.py")])
            environment.assert_called_once_with()
            with self.assertRaises(TaskError) as caught:
                check_png(self.root / "bad.png", plan,
                          runner=unittest.mock.Mock(return_value=unittest.mock.Mock(returncode=1)))
            self.assertEqual(caught.exception.code, "invalid_image_output")

    def create(self):
        return self.store.create(self.request, identity_evidence(self.request["model_path"]))

    def test_missing_model_does_not_create_a_running_task(self):
        path = self.root / "request.json"
        path.write_text(json.dumps(self.request))
        with self.assertRaises(TaskError):
            start_task(path, self.state)
        self.assertEqual(self.store.list(), [])

    def test_repeated_cancel_is_intent_until_execution_stops(self):
        record = self.create()
        first = cancel_task(record["id"], self.state)
        second = cancel_task(record["id"], self.state)
        self.assertTrue(first["cancel_requested"])
        self.assertTrue(second["cancel_requested"])
        self.assertEqual(second["state"], "starting")

    def test_lost_supervisor_is_failed_without_signaling_any_pid(self):
        record = self.create()
        self.store.transition(record["id"], ("starting",), {"supervisor": {"pid": 4242, "identity": "old"}})
        self.store.transition(record["id"], ("starting",), {
            "state": "running", "execution": {"pid": 4343, "identity": "worker"}})
        with patch("mlx_agent.task_runtime.process_identity", return_value="reused"), patch("mlx_agent.task_runtime.process_alive", return_value=True):
            result = task_status(record["id"], self.state)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"]["code"], "task_interrupted")

    def test_reading_unfinished_result_does_not_run_model(self):
        record = self.create()
        with self.assertRaises(TaskError) as caught:
            task_result(record["id"], self.state)
        self.assertEqual(caught.exception.code, "task_not_succeeded")
        self.assertEqual(self.store.get(record["id"])["state"], "starting")

    def test_lost_claim_does_not_fail_another_supervisors_task(self):
        record = self.create()
        original = TaskStore.transition
        def raced_claim(store, identifier, expected, changes):
            if "supervisor" in changes:
                original(store, identifier, expected, {"supervisor": {"pid": 4242, "identity": "other-supervisor"}})
                raise TaskError("task_state_changed", "Already claimed.")
            return original(store, identifier, expected, changes)
        with patch.object(TaskStore, "transition", raced_claim):
            supervise(record["id"], self.state)
        current = self.store.get(record["id"])
        self.assertEqual(current["state"], "starting")
        self.assertEqual(current["supervisor"]["identity"], "other-supervisor")

    def test_process_probe_failure_is_actionable_and_does_not_rewrite_record(self):
        record = self.create()
        self.store.transition(record["id"], ("starting",), {"supervisor": {"pid": 4242, "identity": "supervisor"}})
        before = self.store.database.read_bytes()
        with patch("mlx_agent.task_runtime.process_identity", side_effect=OSError("process table unavailable")):
            with self.assertRaises(TaskError) as caught:
                task_status(record["id"], self.state)
        self.assertEqual(caught.exception.code, "task_process_inspection_failed")
        self.assertTrue(caught.exception.remediation)
        self.assertEqual(self.store.database.read_bytes(), before)

    def test_unavailable_live_process_identity_preserves_running_record(self):
        record = self.create()
        self.store.transition(record["id"], ("starting",), {"supervisor": {"pid": 4242, "identity": "supervisor"}})
        self.store.transition(record["id"], ("starting",), {
            "state": "running", "execution": {"pid": 4343, "identity": "worker"}})
        before = self.store.database.read_bytes()
        with patch("mlx_agent.task_runtime.process_identity", return_value=None), patch(
                "mlx_agent.task_runtime.process_alive", return_value=True):
            with self.assertRaises(TaskError) as caught:
                task_status(record["id"], self.state)
        self.assertEqual(caught.exception.code, "task_identity_unavailable")
        self.assertEqual(self.store.database.read_bytes(), before)
        self.assertEqual(self.store.get(record["id"])["state"], "running")
