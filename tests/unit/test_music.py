import io
import json
import subprocess
import sys
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mlx_agent import cli
from mlx_agent.backends import INSTALL_MARKER, load_manifests
from mlx_agent.music import MusicError, plan_music, run_music
from mlx_agent.music_requantize_runner import copy_supporting_files
from mlx_agent import music_runner
from mlx_agent.taxonomy import classify


class MusicTests(unittest.TestCase):
    def setUp(self):
        from tests.unit.ownership_fixtures import isolate
        isolate(self)
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifests = load_manifests()
        self.registries = {key: {"registry": value["registry"], "source": "snapshot", "installed": key == "mlx-audio"}
                           for key, value in self.manifests.items()}
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"model_type": "minimax_music3"}))
        (self.model / "tokenizer.json").write_text("{}")
        backend = self.root / "mlx-audio"
        (backend / "bin").mkdir(parents=True)
        (backend / "bin/python").touch()
        (backend / INSTALL_MARKER).write_text(json.dumps({"id": "mlx-audio", "version": self.manifests["mlx-audio"]["version"]}))
        self.out = self.root / "music.wav"
        sync = mock.patch("mlx_agent.inference_runner.sync_ports", return_value=[])
        self.sync = sync.start()
        self.addCleanup(sync.stop)

    def plan(self, **kwargs):
        return plan_music(kwargs.pop("model_path", str(self.model)), kwargs.pop("caption", "-warm piano"),
                          kwargs.pop("lyrics", "[instrumental]"), kwargs.pop("out", str(self.out)),
                          manifests=self.manifests, registries=self.registries, root=self.root, **kwargs)

    def test_music_architecture_is_not_mislabelled_as_speech(self):
        task = classify("music", model_type="minimax_music3", pipeline_tag="text-to-audio",
                        manifests=self.manifests, registries=self.registries)
        self.assertEqual(task["type"], "music_generation")
        self.assertEqual(classify("speech", model_type="kokoro", pipeline_tag="text-to-audio",
                                 manifests=self.manifests, registries=self.registries)["type"], "text_to_speech")

    def test_real_tokenizer_is_required_and_root_or_nested_layouts_are_supported(self):
        resolver = music_runner.tokenizer_directory
        self.assertEqual(resolver(self.model), self.model)
        nested = self.model / "tokenizer"
        nested.mkdir()
        (nested / "tokenizer.json").write_text("{}")
        self.assertEqual(resolver(self.model), nested)
        (nested / "tokenizer.json").unlink()
        (self.model / "tokenizer.json").unlink()
        with self.assertRaises(ValueError):
            resolver(self.model)
        with self.assertRaises(MusicError):
            self.plan()

    def test_caption_and_lyrics_use_real_encoder_without_synthetic_fallback(self):
        model = SimpleNamespace(model_type="minimax_music3", config=object(),
                                _text_ids=lambda *args: "synthetic")
        calls = []
        def encode(text, config, directory):
            self.assertEqual(directory, self.model)
            self.assertIs(config, model.config)
            calls.append(text)
            return text
        modules = {
            "mlx_audio.music.models.minimax_music3.minimax_music3": SimpleNamespace(_encode_official_text_pair=encode),
            "mlx_audio.music.models.minimax_music3.prompt": SimpleNamespace(assemble_prompt=lambda caption, lyrics: caption + "|" + lyrics),
        }
        with mock.patch.dict(sys.modules, modules):
            music_runner.configure_tokenizer(model, self.model)
            self.assertEqual(model._text_ids("jazz piano", "[instrumental]"), "jazz piano|[instrumental]")
            self.assertNotEqual(model._text_ids("jazz piano", "[instrumental]"),
                                model._text_ids("metal guitar", "[instrumental]"))
            music_runner.configure_tokenizer(model, self.model)
            self.assertEqual(model._text_ids("jazz piano", "[instrumental]"), "jazz piano|[instrumental]")
            self.assertEqual(calls, ["jazz piano|[instrumental]", "metal guitar|[instrumental]"])

    def test_prompt_cache_is_bounded_and_isolated_by_model_and_tokenizer_directory(self):
        calls = []
        def encode(text, config, directory):
            calls.append((text, config, directory))
            return text
        modules = {
            "mlx_audio.music.models.minimax_music3.minimax_music3": SimpleNamespace(_encode_official_text_pair=encode),
            "mlx_audio.music.models.minimax_music3.prompt": SimpleNamespace(assemble_prompt=lambda c,l: c + "|" + l),
        }
        a = SimpleNamespace(model_type="minimax_music3", config=object())
        b = SimpleNamespace(model_type="minimax_music3", config=object())
        with mock.patch.dict(sys.modules, modules):
            for model in (a,b):
                music_runner.configure_tokenizer(model, self.model)
                model._text_ids("piano", "[instrumental]")
            self.assertEqual(len(calls),2)
            for index in range(8):
                a._text_ids(str(index), "[instrumental]")
            a._text_ids("piano", "[instrumental]")
            self.assertEqual(len(calls),11)
            nested = self.model / "tokenizer"
            nested.mkdir()
            (nested / "tokenizer.json").write_text("{}")
            music_runner.configure_tokenizer(a, self.model)
            a._text_ids("piano", "[instrumental]")
            self.assertEqual(calls[-1][2], nested)

    def test_requantization_preserves_assets_without_copying_old_weights_or_config(self):
        for name in ("tokenizer.json", "scheduler_config.json", "chat_template.jinja",
                     "model.safetensors.index.json", "model.safetensors", "custom.py"):
            (self.model / name).write_text("source")
        (self.model / "tokenizer").mkdir()
        (self.model / "tokenizer/config.json").write_text("tokenizer")
        (self.model / "tokenizer/weights.safetensors").write_text("weights")
        destination = self.root / "converted"
        destination.mkdir()
        (destination / "config.json").write_text("new quantization")
        copy_supporting_files(self.model, destination)
        self.assertEqual((destination / "config.json").read_text(), "new quantization")
        self.assertEqual(sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()),
                         ["chat_template.jinja", "config.json", "scheduler_config.json", "tokenizer.json",
                          "tokenizer/config.json"])

    def test_local_plan_and_offline_run_preserve_parameters(self):
        plan = self.plan(duration=5, steps=12, seed=7, lyrics="[verse]\nA new day")
        self.assertIn("--caption=-warm piano", plan["argv"])
        self.assertIn("--lyrics=[verse]\nA new day", plan["argv"])
        def runner(argv, **kwargs):
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(kwargs["env"]["TRANSFORMERS_OFFLINE"], "1")
            self.out.write_bytes(b"audio")
            return subprocess.CompletedProcess(argv, 0, json.dumps({"path": str(self.out), "seconds": 2, "audio_seconds": 5, "real_time_factor": .4}), "")
        result = run_music(plan, manifests=self.manifests, root=self.root, runner=runner)
        self.assertEqual((result["duration"], result["steps"], result["seed"], result["real_time_factor"]), (5, 12, 7, .4))
        self.assertEqual((self.sync.call_args.args[0]["id"], self.sync.call_args.args[1]), ("mlx-audio", self.root))

    def test_refuses_invalid_parameters_existing_outputs_and_non_music(self):
        for args in ({"duration": float("nan")}, {"duration": 0}, {"duration": 361}, {"steps": 31}, {"seed": -1}, {"lyrics": ""}, {"caption": "bad\x00"}, {"out": "relative.wav"}):
            with self.subTest(args=args), self.assertRaises(MusicError):
                self.plan(**args)
        self.out.touch()
        with self.assertRaises(MusicError):
            self.plan()
        self.out.unlink()
        (self.model / "config.json").write_text('{"model_type":"kokoro"}')
        with self.assertRaises(MusicError) as caught:
            self.plan()
        self.assertEqual(caught.exception.code, "not_music_model")

    def test_no_file_wrong_file_failure_and_timeout_are_not_success(self):
        plan = self.plan()
        for result in (subprocess.CompletedProcess([], 0, json.dumps({"path": str(self.out)}), ""),
                       subprocess.CompletedProcess([], 0, '{"path":"/wrong.wav"}', ""),
                       subprocess.CompletedProcess([], 1, "", "failed")):
            with self.assertRaises(MusicError) as caught:
                run_music(plan, manifests=self.manifests, root=self.root, runner=lambda *a, **k: result)
            self.assertEqual(caught.exception.code, "music_failed")
        with self.assertRaises(MusicError) as caught:
            run_music(plan, manifests=self.manifests, root=self.root,
                      runner=mock.Mock(side_effect=subprocess.TimeoutExpired([], 1)))
        self.assertEqual(caught.exception.code, "music_timeout")

    def test_cli_routes_music_arguments(self):
        output = io.StringIO()
        with mock.patch("mlx_agent.music.plan_music", return_value={}) as plan, mock.patch("mlx_agent.music.run_music", return_value={"path": str(self.out)}), redirect_stdout(output):
            self.assertEqual(cli.main(["convert", "music", "--path", str(self.model), "--caption=-warm piano", "--out", str(self.out), "--json"]), 0)
        self.assertEqual(plan.call_args.args[1:3], ("-warm piano", "[instrumental]"))

    def test_served_music_omits_keep_caption_unless_requested(self):
        from mlx_agent.served import request

        parser = cli.build_parser("convert")
        argv = ["convert", "music", "--path", str(self.model), "--caption", "warm piano", "--out", str(self.out)]
        kind, default = request("music", parser.parse_args(argv))
        _, explicit = request("music", parser.parse_args([*argv, "--keep-caption"]))
        self.assertEqual(kind, "music_generation")
        self.assertNotIn("keep_caption", default)
        self.assertEqual(explicit, {**default, "keep_caption": True})


class AceStepMusicTests(unittest.TestCase):
    """ACE-Step 1.5 checkpoints: a DiT folder beside its VAE, text encoder and planner folders."""

    def setUp(self):
        from tests.unit.ownership_fixtures import isolate
        isolate(self)
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifests = load_manifests()
        self.registries = {key: {"registry": value["registry"], "source": "snapshot", "installed": key == "ace-step"}
                           for key, value in self.manifests.items()}
        checkpoints = self.root / "ACE-Step-1.5" / "checkpoints"
        self.model = checkpoints / "acestep-v15-turbo"
        self.model.mkdir(parents=True)
        (self.model / "config.json").write_text(json.dumps({"model_type": "acestep", "is_turbo": True}))
        for name in ("model.safetensors", "silence_latent.pt"):
            (self.model / name).write_bytes(b"weights")
        (self.model / "modeling_acestep_v15_turbo.py").write_text("# published model code\n")
        for name in ("vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"):
            (checkpoints / name).mkdir()
            (checkpoints / name / "config.json").write_text("{}")
        backend = self.root / "ace-step"
        (backend / "bin").mkdir(parents=True)
        (backend / "bin/python").touch()
        (backend / INSTALL_MARKER).write_text(json.dumps({"id": "ace-step", "version": self.manifests["ace-step"]["version"]}))
        self.out = self.root / "music.wav"
        sync = mock.patch("mlx_agent.inference_runner.sync_ports", return_value=[])
        sync.start()
        self.addCleanup(sync.stop)

    def plan(self, **kwargs):
        return plan_music(kwargs.pop("model_path", str(self.model)), "minimal tech house", kwargs.pop("lyrics", "[instrumental]"),
                          str(self.out), manifests=self.manifests, registries=self.registries, root=self.root, **kwargs)

    def test_turbo_checkpoint_runs_in_the_ace_step_backend_with_its_planner(self):
        plan = self.plan(duration=120, steps=8, seed=7, bpm=120, time_signature=4)
        self.assertEqual(plan["backend"], "ace-step")
        self.assertEqual(Path(plan["argv"][0]), self.root / "ace-step" / "bin" / "python")
        self.assertEqual(Path(plan["argv"][1]).name, "acestep_runner.py")
        flags = dict(zip(plan["argv"][2::2], plan["argv"][3::2]))
        self.assertEqual((flags["--shift"], flags["--bpm"], flags["--time-signature"], flags["--planner"]),
                         ("3.0", "120", "4", "acestep-5Hz-lm-1.7B"))
        def runner(argv, **kwargs):
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.out.write_bytes(b"audio")
            return subprocess.CompletedProcess(argv, 0, json.dumps({"path": str(self.out), "sample_rate": 48000, "audio_seconds": 120}), "")
        result = run_music(plan, manifests=self.manifests, root=self.root, runner=runner)
        self.assertEqual((result["schema"], result["backend"], result["bpm"], result["planner"], result["audio_seconds"]),
                         ("music/1", "ace-step", 120, "acestep-5Hz-lm-1.7B", 120))

    def test_key_and_kept_caption_reach_the_runner_and_the_take_reports_what_the_planner_used(self):
        plan = self.plan(steps=8, key="f# Minor", keep_caption=True)
        self.assertEqual(plan["argv"][plan["argv"].index("--key") + 1], "F# minor")
        self.assertEqual(plan["argv"][-1], "--keep-caption")
        planned = {"caption": "minimal tech house", "bpm": 124, "key": "F# minor", "time_signature": 4}
        def runner(argv, **kwargs):
            self.out.write_bytes(b"audio")
            return subprocess.CompletedProcess(argv, 0, json.dumps({"path": str(self.out), "audio_seconds": 15,
                                                                    "planned": planned}), "")
        result = run_music(plan, manifests=self.manifests, root=self.root, runner=runner)
        self.assertEqual((result["key"], result["keep_caption"], result["planned"]), ("F# minor", True, planned))
        self.out.unlink()
        self.assertFalse({"--key", "--keep-caption"} & set(self.plan(steps=8)["argv"]))
        for key in ("H minor", "F# dorian", "minor", "F#minor", "F## minor"):
            with self.subTest(key=key), self.assertRaises(MusicError) as caught:
                self.plan(steps=8, key=key)
            self.assertEqual(caught.exception.code, "invalid_arguments")

    def test_the_planner_report_names_the_caption_the_dit_was_given(self):
        from mlx_agent import acestep_runner
        metadata = {"caption": "rewritten by the planner", "bpm": 124, "keyscale": "F# minor", "timesignature": "4",
                    "language": "en", "duration": "N/A"}
        self.assertEqual(acestep_runner.planned(metadata, "as written", True),
                         {"bpm": 124, "key": "F# minor", "time_signature": 4, "language": "en", "caption": "as written"})
        self.assertEqual(acestep_runner.planned(metadata, "as written", False)["caption"], "rewritten by the planner")

    def test_music_plans_caption_even_with_all_musical_settings_bound(self):
        from mlx_agent import acestep_runner

        class Planner:
            def initialize(self, **kwargs):
                return "ready", True

            def has_all_metas(self, metadata):
                return {"bpm", "keyscale", "timesignature", "duration"} <= metadata.keys()

        def generate_music(*, llm_handler, params, save_dir, **kwargs):
            metadata = {"bpm": params.bpm, "keyscale": params.keyscale, "timesignature": params.timesignature,
                        "duration": params.duration}
            self.assertEqual(metadata, {"bpm": 120, "keyscale": "F# minor", "timesignature": "4", "duration": 30})
            if params.use_cot_metas and not llm_handler.has_all_metas(metadata):
                if params.use_cot_caption:
                    metadata["caption"] = "Four-on-the-floor groove with a dry kick and warm bass"
                if params.use_cot_language:
                    metadata["language"] = "en"
            path = Path(save_dir) / "music.wav"
            path.touch()
            return SimpleNamespace(success=True, audios=[{"path": str(path)}], extra_outputs={"lm_metadata": metadata})

        mx = SimpleNamespace(reset_peak_memory=lambda: None, get_peak_memory=lambda: 0)
        handler = type("Handler", (), {"initialize_service": lambda self, **kwargs: ("ready", True)})
        modules = {
            "mlx": SimpleNamespace(core=mx), "mlx.core": mx,
            "soundfile": SimpleNamespace(info=lambda path: SimpleNamespace(frames=48000, samplerate=48000)),
            "acestep.model_downloader": SimpleNamespace(_CHECKPOINT_TO_VARIANT={self.model.name: "turbo"}),
            "acestep.handler": SimpleNamespace(AceStepHandler=handler),
            "acestep.inference": SimpleNamespace(GenerationParams=SimpleNamespace, GenerationConfig=SimpleNamespace,
                                                 generate_music=generate_music),
            "acestep.llm_inference": SimpleNamespace(LLMHandler=Planner),
        }
        for keep_caption, expected in ((False, "Four-on-the-floor groove with a dry kick and warm bass"),
                                       (True, "minimal tech house")):
            with self.subTest(keep_caption=keep_caption):
                plan = self.plan(duration=30, steps=8, bpm=120, key="F# minor", time_signature=4,
                                 keep_caption=keep_caption)
                output = io.StringIO()
                with mock.patch.dict(sys.modules, modules), mock.patch.dict(acestep_runner.os.environ), \
                        mock.patch.object(acestep_runner, "socket", SimpleNamespace(socket=SimpleNamespace())), \
                        redirect_stdout(output):
                    self.assertEqual(acestep_runner.main(plan["argv"][2:]), 0)
                self.out.unlink()
                result = json.loads(output.getvalue())["planned"]
                self.assertEqual((result["caption"], result.get("language")), (expected, "en"))

    def test_cover_and_repaint_condition_ace_step_on_the_source_without_the_planner(self):
        from mlx_agent import acestep_runner

        source = self.root / "take.wav"
        source.write_bytes(b"RIFF")
        seen = {}
        source_frames = 96000
        generated_frames = 245760

        class Handler:
            sample_rate = 48000

            def process_src_audio(self, path):
                seen.setdefault("decoded", []).append(path)
                return SimpleNamespace(shape=(2, source_frames))

            def initialize_service(self, **kwargs):
                seen["loads"] = seen.get("loads", 0) + 1
                return "ready", True

            def generate_instruction(self, task_type, complete_track_classes=None):
                return "instruction for " + task_type

        def generate_music(*, dit_handler, llm_handler, params, save_dir, **kwargs):
            seen.update(planner=llm_handler, params=params)
            dit_handler.process_src_audio(params.src_audio)
            seen["frames"] = generated_frames
            path = Path(save_dir) / "music.wav"
            path.touch()
            return SimpleNamespace(success=True, audios=[{"path": str(path)}], extra_outputs={})

        def truncate(frames):
            seen["frames"] = frames

        wav = mock.MagicMock()
        wav.__enter__.return_value.truncate.side_effect = truncate
        mx = SimpleNamespace(reset_peak_memory=lambda: None, get_peak_memory=lambda: 0)
        modules = {
            "mlx": SimpleNamespace(core=mx), "mlx.core": mx,
            "soundfile": SimpleNamespace(info=lambda path: SimpleNamespace(frames=seen["frames"], samplerate=48000),
                                         SoundFile=lambda *args: wav),
            "acestep.model_downloader": SimpleNamespace(_CHECKPOINT_TO_VARIANT={self.model.name: "turbo"}),
            "acestep.handler": SimpleNamespace(AceStepHandler=Handler),
            "acestep.inference": SimpleNamespace(GenerationParams=SimpleNamespace, GenerationConfig=SimpleNamespace,
                                                 generate_music=generate_music),
            "acestep.llm_inference": SimpleNamespace(LLMHandler=object),
        }

        def run(plan, status=0):
            output = io.StringIO()
            with mock.patch.dict(sys.modules, modules), mock.patch.dict(acestep_runner.os.environ), \
                    mock.patch.object(acestep_runner, "socket", SimpleNamespace(socket=SimpleNamespace())), \
                    redirect_stdout(output):
                self.assertEqual(acestep_runner.main(plan["argv"][2:]), status)
            if status == 0:
                self.out.unlink()
            return json.loads(output.getvalue())

        cover = self.plan(steps=8, lyrics="[verse]\nNew words", task="cover", source=str(source), cover_strength=0.4)
        self.assertNotIn("--planner", cover["argv"])
        self.assertNotIn("--duration", cover["argv"])
        self.assertNotIn("planned", run(cover))
        params = seen["params"]
        self.assertIsNone(seen["planner"])
        self.assertEqual((params.task_type, params.src_audio, params.audio_cover_strength, params.instruction,
                          params.lyrics, params.thinking),
                         ("cover", str(source), 0.4, "instruction for cover", "[verse]\nNew words", False))

        self.assertTrue(getattr(params, "enable_normalization", True))

        repaint = self.plan(steps=8, task="repaint", source=str(source), repaint_start=0.45, repaint_end=1.25)
        seen["decoded"].clear()
        self.assertEqual(run(repaint)["audio_seconds"], 2.0)
        self.assertEqual(seen["decoded"], [str(source)])
        wav.__enter__.return_value.truncate.assert_called_once_with(source_frames)
        params = seen["params"]
        self.assertEqual((params.task_type, params.src_audio, params.repainting_start, params.repainting_end,
                          params.instruction), ("repaint", str(source), 0.45, 1.25, "instruction for repaint"))
        self.assertFalse(params.enable_normalization)

        outpainting = self.plan(steps=8, task="repaint", source=str(source), repaint_start=1.5, repaint_end=2.01)
        loads = seen["loads"]
        refused = run(outpainting, status=2)
        self.assertEqual(refused["error"], "invalid_arguments")
        self.assertIn("2 s source", refused["detail"])
        self.assertEqual(seen["loads"], loads)
        self.assertFalse(self.out.exists())

        def runner(argv, **kwargs):
            record = refused if argv == outpainting["argv"] else {"path": str(self.out), "audio_seconds": 2.0}
            if "path" in record:
                self.out.write_bytes(b"audio")
            return subprocess.CompletedProcess(argv, 2 if "error" in record else 0, json.dumps(record), "")
        with self.assertRaises(MusicError) as error:
            run_music(outpainting, manifests=self.manifests, root=self.root, runner=runner)
        self.assertEqual(error.exception.code, "invalid_arguments")
        self.assertIn("2 s source", str(error.exception))
        result = run_music(repaint, manifests=self.manifests, root=self.root, runner=runner)
        self.assertEqual((result["task"], result["source"], result["repaint_start"], result["repaint_end"],
                          result["duration"], result.get("planner")), ("repaint", str(source), 0.45, 1.25, None, None))
        self.out.unlink()
        self.assertNotIn("task", run_music(self.plan(steps=8), manifests=self.manifests, root=self.root,
                                           runner=runner))

    def test_cover_and_repaint_settings_are_refused_where_they_do_not_apply(self):
        source = self.root / "take.wav"
        source.write_bytes(b"RIFF")
        refused = {
            "unknown task": {"task": "extract", "source": str(source)},
            "source without a task": {"source": str(source)},
            "cover without a source": {"task": "cover"},
            "relative source": {"task": "cover", "source": "take.wav"},
            "missing source": {"task": "cover", "source": str(self.root / "gone.wav")},
            "source that is not audio": {"task": "cover", "source": str(self.model / "config.json")},
            "cover strength above 1": {"task": "cover", "source": str(source), "cover_strength": 1.5},
            "cover strength on repaint": {"task": "repaint", "source": str(source), "repaint_start": 0,
                                          "repaint_end": 2, "cover_strength": 0.5},
            "repaint without a window": {"task": "repaint", "source": str(source)},
            "repaint window ending first": {"task": "repaint", "source": str(source), "repaint_start": 5,
                                            "repaint_end": 5},
            "repaint window on cover": {"task": "cover", "source": str(source), "repaint_start": 0, "repaint_end": 2},
        }
        for name, settings in refused.items():
            with self.subTest(name), self.assertRaises(MusicError) as caught:
                self.plan(steps=8, **settings)
            self.assertEqual(caught.exception.code, "invalid_arguments")

    def base_checkpoint(self, name="acestep-v15-xl-base"):
        model = self.model.parent / name
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "acestep"}))
        for file in ("model.safetensors", "silence_latent.pt"):
            (model / file).write_bytes(b"weights")
        return model

    def test_complete_adds_the_named_tracks_to_a_source_on_base_checkpoints(self):
        from mlx_agent import acestep_runner

        source = self.root / "vocals.wav"
        source.write_bytes(b"RIFF")
        base = self.base_checkpoint()
        seen = {}

        class Handler:
            def initialize_service(self, **kwargs):
                return "ready", True

            def generate_instruction(self, task_type, complete_track_classes=None):
                return "{0} with {1}".format(task_type, complete_track_classes)

        def generate_music(*, llm_handler, params, save_dir, **kwargs):
            seen.update(planner=llm_handler, params=params)
            path = Path(save_dir) / "music.wav"
            path.touch()
            return SimpleNamespace(success=True, audios=[{"path": str(path)}], extra_outputs={})

        mx = SimpleNamespace(reset_peak_memory=lambda: None, get_peak_memory=lambda: 0)
        modules = {
            "mlx": SimpleNamespace(core=mx), "mlx.core": mx,
            "soundfile": SimpleNamespace(info=lambda path: SimpleNamespace(frames=96000, samplerate=48000)),
            "acestep.model_downloader": SimpleNamespace(_CHECKPOINT_TO_VARIANT={base.name: "xl_base"}),
            "acestep.handler": SimpleNamespace(AceStepHandler=Handler),
            "acestep.inference": SimpleNamespace(GenerationParams=SimpleNamespace, GenerationConfig=SimpleNamespace,
                                                 generate_music=generate_music),
            "acestep.llm_inference": SimpleNamespace(LLMHandler=object),
        }
        plan = self.plan(model_path=str(base), steps=50, lyrics="[verse]\nKept words", task="complete",
                         source=str(source), tracks=["drums", "bass"])
        self.assertNotIn("--planner", plan["argv"])
        self.assertNotIn("--duration", plan["argv"])
        output = io.StringIO()
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(acestep_runner.os.environ), \
                mock.patch.object(acestep_runner, "socket", SimpleNamespace(socket=SimpleNamespace())), \
                redirect_stdout(output):
            self.assertEqual(acestep_runner.main(plan["argv"][2:]), 0)
        self.out.unlink()
        params = seen["params"]
        self.assertIsNone(seen["planner"])
        self.assertEqual((params.task_type, params.src_audio, params.instruction, params.lyrics),
                         ("complete", str(source), "complete with ['drums', 'bass']", "[verse]\nKept words"))

        def runner(argv, **kwargs):
            self.out.write_bytes(b"audio")
            return subprocess.CompletedProcess(argv, 0, json.dumps({"path": str(self.out), "audio_seconds": 2}), "")
        result = run_music(plan, manifests=self.manifests, root=self.root, runner=runner)
        self.assertEqual((result["task"], result["tracks"], result["duration"], result["shift"]),
                         ("complete", ["drums", "bass"], None, 1.0))
        self.out.unlink()

        with mock.patch("mlx_agent.music.plan_music", return_value={}) as planned, \
                mock.patch("mlx_agent.music.run_music", return_value={"path": str(self.out)}), \
                mock.patch("mlx_agent.served.endpoint", return_value=object()) as served, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["convert", "music", "--path", str(base), "--caption", "neo soul band", "--out",
                                       str(self.out), "--task", "complete", "--source", str(source),
                                       "--track", "drums", "--track", "bass", "--json"]), 0)
        self.assertEqual((planned.call_args.kwargs["task"], planned.call_args.kwargs["tracks"]),
                         ("complete", ["drums", "bass"]))
        served.assert_not_called()

        sft = self.base_checkpoint("acestep-v15-base-sft-fix-inst")
        refused = {
            "a turbo checkpoint": {"task": "complete", "source": str(source)},
            "a base checkpoint tuned as SFT": {"model_path": str(sft), "task": "complete", "source": str(source)},
            "an unknown track": {"model_path": str(base), "task": "complete", "source": str(source),
                                 "tracks": ["kazoo"]},
            "a track named twice": {"model_path": str(base), "task": "complete", "source": str(source),
                                    "tracks": ["drums", "drums"]},
            "an empty track list": {"model_path": str(base), "task": "complete", "source": str(source), "tracks": []},
            "a cover strength": {"model_path": str(base), "task": "complete", "source": str(source),
                                 "cover_strength": 0.5},
            "tracks on a cover": {"task": "cover", "source": str(source), "tracks": ["drums"]},
            "tracks without a task": {"tracks": ["drums"]},
        }
        for name, settings in refused.items():
            with self.subTest(name), self.assertRaises(MusicError) as caught:
                self.plan(steps=8, **settings)
            self.assertEqual(caught.exception.code, "invalid_arguments")

    def test_cli_runs_a_cover_in_its_own_process_with_its_settings(self):
        source = self.root / "take.wav"
        source.write_bytes(b"RIFF")
        output = io.StringIO()
        with mock.patch("mlx_agent.music.plan_music", return_value={}) as plan, \
                mock.patch("mlx_agent.music.run_music", return_value={"path": str(self.out)}) as run, \
                mock.patch("mlx_agent.served.endpoint", return_value=object()) as served, redirect_stdout(output):
            self.assertEqual(cli.main(["convert", "music", "--path", str(self.model), "--caption", "trap", "--out",
                                       str(self.out), "--task", "cover", "--source", str(source),
                                       "--cover-strength", "0.3", "--json"]), 0)
        self.assertEqual({name: plan.call_args.kwargs[name] for name in ("task", "source", "cover_strength")},
                         {"task": "cover", "source": str(source), "cover_strength": 0.3})
        run.assert_called_once()
        served.assert_not_called()

    def test_compose_runs_the_planner_beside_the_checkpoint_bound_to_the_given_values(self):
        from mlx_agent.compose import plan_compose, run_compose
        def compose(**kwargs):
            return plan_compose(str(self.model), "boom bap", "[verse]\nRun it local", manifests=self.manifests,
                                registries=self.registries, root=self.root, **kwargs)
        plan = compose(bpm=96, duration=30, key="f# minor", language="en")
        self.assertEqual(Path(plan["argv"][1]).name, "acestep_compose_runner.py")
        self.assertEqual(plan["argv"][plan["argv"].index("--planner") + 1], str(self.model.parent / "acestep-5Hz-lm-1.7B"))
        self.assertEqual([item for item in plan["argv"] if item.startswith(("--bpm=", "--duration=", "--key=", "--language="))],
                         ["--bpm=96", "--duration=30", "--key=F# minor", "--language=en"])
        self.assertFalse(any(item.startswith(("--bpm", "--key")) for item in compose()["argv"]))
        answer = {"caption": "A dusty boom bap beat over warm bass", "lyrics": "[Verse]\nRun it local", "bpm": 96,
                  "key": "F# minor", "time_signature": 4}
        def runner(argv, **kwargs):
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            return subprocess.CompletedProcess(argv, 0, json.dumps(answer), "")
        result = run_compose(plan, manifests=self.manifests, root=self.root, runner=runner)
        self.assertEqual((result["schema"], result["caption"], result["key"], result["bound"]),
                         ("compose/1", answer["caption"], "F# minor",
                          {"bpm": 96, "duration": 30, "key": "F# minor", "language": "en"}))
        self.assertEqual(result["language"], "en")
        for kwargs in ({"key": "F# dorian"}, {"bpm": 20}, {"language": "e n"}, {"temperature": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(MusicError):
                compose(**kwargs)
        (self.model.parent / "acestep-5Hz-lm-1.7B" / "config.json").unlink()
        with self.assertRaises(MusicError) as caught:
            compose()
        self.assertIn("planner", str(caught.exception))

    def test_compose_duration_requires_whole_seconds(self):
        from mlx_agent.compose import plan_compose

        def compose(duration):
            return plan_compose(str(self.model), "boom bap", "[instrumental]", duration=duration,
                                manifests=self.manifests, registries=self.registries, root=self.root)

        for duration in (0.5, 30.5):
            with self.subTest(duration=duration), self.assertRaisesRegex(MusicError, "whole seconds"):
                compose(duration)
        self.assertIn("--duration=30", compose(30.0)["argv"])

    def test_ace_step_limits_and_incomplete_checkpoints_are_refused(self):
        with self.assertRaises(MusicError) as caught:
            self.plan(steps=21)
        self.assertEqual(caught.exception.code, "invalid_arguments")
        (self.model / "config.json").write_text(json.dumps({"model_type": "acestep", "is_turbo": False}))
        self.assertEqual(self.plan(steps=50)["argv"][-6:-2], ["--shift", "1.0", "--guidance", "7.0"])
        with self.assertRaises(MusicError):
            self.plan(steps=65)
        (self.model.parent / "vae" / "config.json").unlink()
        with self.assertRaises(MusicError) as caught:
            self.plan(steps=8)
        self.assertEqual(caught.exception.code, "model_unreadable")
        self.assertIn("vae", str(caught.exception))

    def test_staged_checkpoint_takes_code_writes_without_touching_the_model_folder(self):
        from mlx_agent import acestep_runner
        with TemporaryDirectory() as directory:
            checkpoints = acestep_runner.stage(self.model, "acestep-5Hz-lm-1.7B", directory)
            staged = checkpoints / self.model.name
            self.assertEqual(sorted(path.name for path in checkpoints.iterdir()),
                             ["Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B", "acestep-v15-turbo", "vae"])
            self.assertEqual((staged / "model.safetensors").resolve(), (self.model / "model.safetensors").resolve())
            self.assertFalse((staged / "modeling_acestep_v15_turbo.py").exists())
            # ACE-Step copies its own model code into the folder it loads.
            (staged / "modeling_acestep_v15_turbo.py").write_text("# synced code\n")
        self.assertEqual((self.model / "modeling_acestep_v15_turbo.py").read_text(), "# published model code\n")
        self.assertEqual((self.model / "model.safetensors").read_bytes(), b"weights")

    def test_staged_quantizer_adaptation_preserves_original_checkpoint(self):
        from mlx_agent import acestep_runner
        original_code = (self.model / "modeling_acestep_v15_turbo.py").read_bytes()
        original_config = (self.model / "config.json").read_bytes()
        original_weights = (self.model / "model.safetensors").read_bytes()
        with TemporaryDirectory() as directory:
            staged = acestep_runner.stage(self.model, "acestep-5Hz-lm-1.7B", directory) / self.model.name
            code = staged / "modeling_acestep_v15_turbo.py"
            code.write_text("from vector_quantize_pytorch import ResidualFSQ\n")
            acestep_runner.adapt_quantizer(staged)
            self.assertEqual(code.read_text(), "from acestep_fsq import ResidualFSQ\n")
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                acestep_runner.adapt_quantizer(staged)
        self.assertEqual((self.model / "modeling_acestep_v15_turbo.py").read_bytes(), original_code)
        self.assertEqual((self.model / "config.json").read_bytes(), original_config)
        self.assertEqual((self.model / "model.safetensors").read_bytes(), original_weights)
