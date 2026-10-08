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
        def encode(text, config, directory):
            self.assertEqual(directory, self.model)
            self.assertIs(config, model.config)
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
        result = run_music(plan, runner=runner)
        self.assertEqual((result["duration"], result["steps"], result["seed"], result["real_time_factor"]), (5, 12, 7, .4))

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
            with self.assertRaises(MusicError):
                run_music(plan, runner=lambda *a, **k: result)
        with self.assertRaises(MusicError):
            run_music(plan, runner=mock.Mock(side_effect=subprocess.TimeoutExpired([], 1)))

    def test_cli_routes_music_arguments(self):
        output = io.StringIO()
        with mock.patch("mlx_agent.cli.plan_music", return_value={}) as plan, mock.patch("mlx_agent.cli.run_music", return_value={"path": str(self.out)}), redirect_stdout(output):
            self.assertEqual(cli.main(["convert", "music", "--path", str(self.model), "--caption=-warm piano", "--out", str(self.out), "--json"]), 0)
        self.assertEqual(plan.call_args.args[1:3], ("-warm piano", "[instrumental]"))
