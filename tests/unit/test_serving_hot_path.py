"""Repeated work on the serving and task paths stays at one inspection per boundary."""
import importlib.util
import inspect
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlx_agent.backends import load_manifests
from mlx_agent.identity_pass import recheck_bound_identity
from mlx_agent.jit_serve import model_fingerprint
from mlx_agent.memory_budget import SCHEMA, account
from mlx_agent.task_contracts import validate_record
from mlx_agent.task_identity import capture_identity
from mlx_agent.task_serving import TaskModelWorker
from mlx_agent.task_store import TaskStore
from mlx_agent import task_runtime
from tests.task_fixtures import identity_evidence


def _ps_runs(function):
    calls = []
    real_run = subprocess.run

    def counting(argv, *args, **kwargs):
        if isinstance(argv, (list, tuple)) and argv and argv[0] == "/bin/ps":
            calls.append(list(argv))
        return real_run(argv, *args, **kwargs)

    with patch("mlx_agent.process_table.subprocess.run", counting):
        function()
    return calls


class ProcessInspectionTests(unittest.TestCase):
    def test_group_membership_is_one_ps_per_inspection(self):
        process = types.SimpleNamespace(pid=10)

        def inspect():
            with patch("mlx_agent.task_runtime.group_members", return_value=[10, 11, 12]):
                task_runtime._known_members(process, "leader", {11: "child"})

        calls = _ps_runs(inspect)
        self.assertEqual(len(calls), 1)
        self.assertIn("10,11,12", calls[0][-1])

    def test_idle_release_is_one_ps_for_the_whole_group(self):
        from mlx_agent.compute import ComputeLease

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve() / "compute"
        leader = os.getpid()
        members = {leader: "leader", 4242: "child"}

        def release():
            with ComputeLease("task", root, group_probe=lambda _pid: list(members)) as lease:
                lease.execution_group = {"pgid": leader, "leader_identity": "leader"}
                lease.record = dict(lease.record, schema=2, execution_group=dict(lease.execution_group))

                def once():
                    with self.assertRaises(Exception):
                        lease.release_idle_process_group(leader, members)

                return _ps_runs(once)

        calls = release()
        self.assertEqual(len(calls), 1)
        self.assertIn(str(leader), calls[0][-1])
        self.assertIn("4242", calls[0][-1])


class CancelPollTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "tasks"
        self.store = TaskStore(self.root)
        self.request = {"schema": "task-request/1", "kind": "classification", "model_path": "/models/local",
                        "parameters": {"state": "refund", "questions": {"route": {"type": "choice",
                        "instructions": "Which team should handle this request?", "criteria": ["billing", "other"]}}}}

    def test_cancel_flag_does_not_load_the_record(self):
        task = self.store.create(self.request, identity_evidence(self.request["model_path"]))
        with patch("mlx_agent.task_store.validate_record", wraps=validate_record) as validate:
            self.assertFalse(self.store.cancel_requested(task["id"]))
        self.assertEqual(validate.call_count, 0)
        self.store.request_cancel(task["id"])
        with patch("mlx_agent.task_store.validate_record", wraps=validate_record) as validate:
            self.assertTrue(self.store.cancel_requested(task["id"]))
        self.assertEqual(validate.call_count, 0)

    def test_supervisor_and_resident_polls_do_not_load_the_record(self):
        task = self.store.create(self.request, identity_evidence(self.request["model_path"]))
        loop = inspect.getsource(task_runtime.supervise)
        poll = loop.split("while process.poll() is None:", 1)[1].split("time.sleep(POLL_SECONDS)", 1)[0]
        self.assertIn("cancel_requested", poll)
        self.assertNotIn("store.get", poll)
        execute = inspect.getsource(TaskModelWorker._execute)
        self.assertNotIn("store.get(identifier)['cancel_requested']", execute)
        self.assertIn("cancel_requested(identifier)", execute)
        worker = TaskModelWorker.__new__(TaskModelWorker)
        worker.closed = False
        worker.store = self.store
        with patch.object(TaskStore, "get", side_effect=AssertionError("full record")):
            self.assertFalse(worker._task_cancelled(task["id"]))
            self.store.request_cancel(task["id"])
            self.assertTrue(worker._task_cancelled(task["id"]))


class IdentityBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model = Path(temporary.name).resolve() / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"laya"}\n')
        (self.model / "model.safetensors").write_bytes(b"weights")
        manifest = load_manifests()["mlx-embeddings"]
        identity = capture_identity(self.model, manifest, Path(sys.executable))
        from mlx_agent.task_serving import runtime_fingerprint
        self.plan = {"runtime": "tasks", "jit": True, "local_path": str(self.model),
                     "fingerprint": model_fingerprint(self.model), "task_identity": identity,
                     "task_runtime_fingerprint": runtime_fingerprint(identity["backend"])}

    def _walks(self, function):
        walks = []
        original = Path.rglob

        def counting(path, pattern):
            if path == self.model:
                walks.append(pattern)
            return original(path, pattern)

        with patch.object(Path, "rglob", counting):
            function()
        return walks

    def test_one_recheck_walks_the_model_once(self):
        with patch("mlx_agent.backends.load_manifests", wraps=load_manifests) as manifests:
            walks = self._walks(lambda: recheck_bound_identity(self.plan))
        self.assertEqual(walks, ["*"])
        self.assertEqual(manifests.call_count, 1)
        self.assertEqual(len(self._walks(lambda: (recheck_bound_identity(self.plan), recheck_bound_identity(self.plan)))), 2)

    def test_a_matching_record_is_not_fingerprinted_again(self):
        worker = TaskModelWorker.__new__(TaskModelWorker)
        worker.config = self.plan
        seen = []
        worker._recheck_identity = lambda: seen.append("recheck")
        with patch("mlx_agent.task_serving.check_identity") as check:
            worker._confirm_identity({"identity": self.plan["task_identity"]})
            check.assert_not_called()
            worker._confirm_identity({"identity": dict(self.plan["task_identity"], model_fingerprint="0" * 64)})
            check.assert_called_once()
        self.assertEqual(seen, ["recheck", "recheck"])


class FootprintTests(unittest.TestCase):
    def test_footprint_skips_pids_outside_the_accounted_family(self):
        rows = {
            1: {"parent": 0, "group": 1},
            2: {"parent": 1, "group": 2, "identity": "worker"},
            3: {"parent": 2, "group": 2},
            9: {"parent": 1, "group": 9},
        }
        policy = {"schema": SCHEMA, "limit_bytes": 1000, "reserve_bytes": 0, "external": []}
        entries = {"job": {"phase": "resident", "owner": (1, "owner"), "resident_bytes": 10,
                           "reserved_bytes": 0, "growth_bytes": 0,
                           "worker": {"pid": 2, "identity": "worker", "pgid": 2}}}
        seen = []

        def usage(pid):
            seen.append(pid)
            return types.SimpleNamespace(resident_size=10, phys_footprint=12, exit_time=0)

        with patch("mlx_agent.memory_budget._usage", side_effect=usage), \
             patch("mlx_agent.memory_budget.sys.platform", "darwin"):
            account(policy, entries, rows, (1, "owner"))
        self.assertEqual(set(seen), {1, 2, 3})
        self.assertNotIn(9, seen)


class LTXRetainTests(unittest.TestCase):
    def test_second_generate_does_not_reload_a_retained_pipeline(self):
        path = Path(__file__).resolve().parents[2] / "src/mlx_agent/resources/ports/ltx-video/ltx2/ltx2.py"
        spec = importlib.util.spec_from_file_location("ltx_retain_fixture", path)
        port = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(port)
        for retain, expected in ((False, 2), (True, 1)):
            with self.subTest(retain=retain), TemporaryDirectory() as directory:
                root = Path(directory)
                calls = {}

                def counted(name):
                    def function(*args, **kwargs):
                        calls[name] = calls.get(name, 0) + 1
                        return object()
                    return function

                ltx = types.ModuleType("mlx_video.models.ltx_2.generate")

                class LTXModel:
                    pass

                class VideoEncoder:
                    pass

                class VideoDecoder:
                    pass

                LTXModel.from_pretrained = classmethod(lambda cls, *args, **kwargs: counted("transformer")(*args, **kwargs))
                VideoEncoder.from_pretrained = classmethod(lambda cls, *args, **kwargs: counted("video_encoder")(*args, **kwargs))
                VideoDecoder.from_pretrained = classmethod(lambda cls, *args, **kwargs: counted("video_decoder")(*args, **kwargs))
                ltx.LTXModel = LTXModel
                ltx.VideoEncoder = VideoEncoder
                ltx.VideoDecoder = VideoDecoder
                ltx.load_upsampler = counted("upsampler")
                ltx.load_audio_decoder = counted("audio_decoder")
                ltx.load_vocoder_model = counted("vocoder")
                ltx.PipelineType = types.SimpleNamespace(DISTILLED="distilled")
                ltx.mux_video_audio = lambda *args, **kwargs: True

                class LTX2TextEncoder:
                    def __init__(self, has_prompt_adaln=False):
                        self.has_prompt_adaln = has_prompt_adaln

                    def load(self, *args, **kwargs):
                        calls["text_encoder"] = calls.get("text_encoder", 0) + 1

                    def parameters(self):
                        return []

                    def __call__(self, *args, **kwargs):
                        return object(), object()

                text = types.ModuleType("mlx_video.models.ltx_2.text_encoder")
                class LanguageModel:
                    def _create_causal_mask_with_padding(self, *args, **kwargs):
                        return None

                text.LTX2TextEncoder = LTX2TextEncoder
                text.LanguageModel = LanguageModel

                class AudioEncoder:
                    pass

                AudioEncoder.from_pretrained = classmethod(lambda cls, *args, **kwargs: counted("audio_encoder")(*args, **kwargs))
                audio = types.ModuleType("mlx_video.models.ltx_2.audio_vae")
                audio.AudioEncoder = AudioEncoder
                package = types.ModuleType("mlx_video.models.ltx_2")
                package.generate = ltx

                def generate_video(**options):
                    encoder = LTX2TextEncoder(has_prompt_adaln=False)
                    encoder.load(model_path=options["model_repo"], text_encoder_path=options["text_encoder_repo"])
                    encoder.parameters()
                    ltx.LTXModel.from_pretrained(model_path=options["model_repo"] + "/transformer", strict=True)
                    ltx.VideoEncoder.from_pretrained(options["model_repo"] + "/vae/encoder")
                    ltx.load_upsampler(options["spatial_upscaler"])
                    ltx.VideoDecoder.from_pretrained(options["model_repo"] + "/vae/decoder")
                    ltx.load_audio_decoder(options["model_repo"], "distilled")
                    ltx.load_vocoder_model(options["model_repo"], "distilled")
                    class Video:
                        shape = (2, 4, 6)

                        def __len__(self):
                            return 2

                    return (Video(),)

                ltx.generate_video = generate_video
                core = types.SimpleNamespace(reset_peak_memory=lambda: None, get_peak_memory=lambda: 0)
                modules = {"mlx": types.SimpleNamespace(core=core), "mlx.core": core,
                           "mlx_video": types.ModuleType("mlx_video"),
                           "mlx_video.models": types.ModuleType("mlx_video.models"),
                           "mlx_video.models.ltx_2": package,
                           "mlx_video.models.ltx_2.generate": ltx,
                           "mlx_video.models.ltx_2.text_encoder": text,
                           "mlx_video.models.ltx_2.audio_vae": audio}
                model = port.Model(root, {"spatial_upscaler": "upscaler.safetensors"})
                model.retain_weights = retain
                with patch.dict(sys.modules, modules), patch.object(port, "_pixel_std", return_value=0):
                    for seed in (1, 2):
                        port.generate(model, "test", str(root / ("out%d.mp4" % seed)), seed=seed)
                names = ("transformer", "video_encoder", "video_decoder", "upsampler",
                         "audio_decoder", "vocoder", "text_encoder")
                self.assertEqual(calls, {name: expected for name in names})
                self.assertEqual(len(model.components), len(names) if retain else 0)
