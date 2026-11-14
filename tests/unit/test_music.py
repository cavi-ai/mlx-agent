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


class AceStepMusicTests(unittest.TestCase):
    """ACE-Step 1.5 checkpoints: a DiT folder beside its VAE, text encoder and planner folders."""

    def setUp(self):
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
