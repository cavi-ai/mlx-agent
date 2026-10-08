import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.convert import (
    ConvertError,
    load_receipts,
    plan_convert,
    plan_gguf_convert,
    start_convert,
    status_convert,
)

from jsonschema import Draft202012Validator

from .test_gguf import _UINT32, _kv, write_gguf

RECEIPT_SCHEMA = Draft202012Validator(json.loads(
    (Path(__file__).resolve().parents[2] / "schemas" / "convert-receipt.schema.json").read_text(encoding="utf-8")
))


class PlanConvertTests(unittest.TestCase):
    def test_cached_source_is_passed_as_local_path(self):
        with TemporaryDirectory() as directory:
            cache = Path(directory)
            entry = cache / "models--pub--model"
            snapshot = entry / "snapshots" / "abc"
            snapshot.mkdir(parents=True)
            (entry / "refs").mkdir()
            (entry / "refs" / "main").write_text("abc")
            (snapshot / "config.json").write_text('{"model_type":"qwen2"}')
            plan = plan_convert("pub/model", hf_cache=cache)
            self.assertEqual(plan["argv"][2], str(snapshot))
            self.assertEqual(plan["source"]["path"], str(snapshot))

    def test_explicit_download_directory_is_reused_and_changes_invalidate_preview(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "download"
            source.mkdir()
            (source / "config.json").write_text('{"model_type":"qwen2"}')
            first = plan_convert("pub/model", source_path=str(source))
            self.assertEqual(first["argv"][2], str(source))
            (source / "config.json").write_text('{"model_type":"qwen3"}')
            second = plan_convert("pub/model", source_path=str(source))
            self.assertNotEqual(first["preview_hash"], second["preview_hash"])

    def test_default_plan(self):
        plan = plan_convert("pub/model")
        self.assertEqual(plan["q_bits"], 4)
        self.assertEqual(plan["out"], "model-MLX-4bit")
        self.assertEqual(
            plan["argv"],
            ["mlx_lm.convert", "--hf-path", "pub/model", "--mlx-path", "model-MLX-4bit",
             "--quantize", "--q-bits", "4"],
        )
        self.assertNotIn("backend", plan)
        self.assertEqual(len(plan["preview_hash"]), 64)

    def test_explicit_out_and_bits(self):
        plan = plan_convert("pub/model", q_bits=8, out="/tmp/out-8")
        self.assertEqual(plan["out"], "/tmp/out-8")
        self.assertIn("--q-bits", plan["argv"])
        self.assertEqual(plan["argv"][-1], "8")

    def test_invalid_repo(self):
        with self.assertRaises(ConvertError) as caught:
            plan_convert("../evil")
        self.assertEqual(caught.exception.code, "invalid_repo")

    def test_invalid_q_bits(self):
        with self.assertRaises(ConvertError) as caught:
            plan_convert("pub/model", q_bits=5)
        self.assertEqual(caught.exception.code, "invalid_arguments")


class BackendConvertTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_exported_music_uses_local_requantization_but_raw_and_speech_keep_backend_converter(self):
        source = self.root / "source"
        source.mkdir()
        for model_type, quantization, dedicated in (
                ("minimax_music3", {"bits": 8, "group_size": 64}, True),
                ("minimax_music3", None, False), ("kokoro", {"bits": 8}, False)):
            with self.subTest(model_type=model_type, quantization=quantization):
                config = {"model_type": model_type}
                if quantization:
                    config["quantization"] = quantization
                (source / "config.json").write_text(json.dumps(config))
                plan = plan_convert("pub/model", backend="mlx-audio", source_path=str(source),
                                    backends_root_dir=self.root)
                if dedicated:
                    self.assertTrue(plan["argv"][1].endswith("music_requantize_runner.py"))
                else:
                    self.assertEqual(plan["argv"][1:3], ["-m", "mlx_audio.convert"])
                self.assertEqual(plan["argv"][plan["argv"].index("--hf-path") + 1], str(source))

    def test_backend_plan_uses_the_backend_venv(self):
        plan = plan_convert("openai/whisper-tiny", backend="mlx-audio", backends_root_dir=self.root)
        self.assertEqual(plan["backend"], "mlx-audio")
        self.assertEqual(plan["argv"], [
            str(self.root / "mlx-audio" / "bin" / "python"), "-m", "mlx_audio.convert",
            "--hf-path", "openai/whisper-tiny", "--mlx-path", "whisper-tiny-MLX-4bit",
            "--quantize", "--q-bits", "4",
        ])
        self.assertNotIn("--trust-remote-code", plan["argv"])

    def test_unknown_backend_is_refused(self):
        with self.assertRaises(ConvertError) as caught:
            plan_convert("pub/model", backend="nope")
        self.assertEqual(caught.exception.code, "unknown_backend")

    def test_start_refuses_uninstalled_backend(self):
        plan = plan_convert("openai/whisper-tiny", backend="mlx-audio", backends_root_dir=self.root)
        with self.assertRaises(ConvertError) as caught:
            start_convert(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan["preview_hash"],
                          spawn=lambda argv, log: 1, model_present=lambda repo: True)
        self.assertEqual(caught.exception.code, "backend_not_installed")
        self.assertIn("backend install mlx-audio", caught.exception.remediation)

    def test_backend_plan_names_the_ports_it_installs(self):
        plan = plan_convert("Edge0/Audio8-ASR-Infinite", backend="mlx-audio", backends_root_dir=self.root)
        self.assertEqual(plan["ports"], ["audio8_asr_infinite"])
        self.assertNotIn("ports", plan_convert("pub/model", backend="mlx-vlm", backends_root_dir=self.root))

    def test_model_type_selects_a_port_converter_only_where_the_port_ships_one(self):
        plan = plan_convert("convaiinnovations/laya", q_bits=8, backend="mlx-embeddings", model_type="laya", backends_root_dir=self.root)
        self.assertEqual(plan["argv"][1:3], ["-m", "mlx_embeddings.classifiers.laya.convert"])
        self.assertEqual(plan["port_converter"], "mlx_embeddings.classifiers.laya.convert")
        self.assertEqual(plan["ports"], ["laya"])
        default = plan_convert("convaiinnovations/laya", q_bits=8, backend="mlx-embeddings", backends_root_dir=self.root)
        self.assertEqual(default["argv"][1:3], ["-m", "mlx_embeddings.convert"])
        self.assertNotIn("port_converter", default)
        self.assertNotEqual(plan["preview_hash"], default["preview_hash"])
        whisper = plan_convert("openai/whisper-tiny", backend="mlx-audio", model_type="whisper", backends_root_dir=self.root)
        self.assertEqual(whisper["argv"][1:3], ["-m", "mlx_audio.convert"])
        for bad in ("../x", "laya; rm", "", 3):
            with self.subTest(model_type=bad):
                with self.assertRaises(ConvertError) as caught:
                    plan_convert("pub/model", backend="mlx-embeddings", model_type=bad, backends_root_dir=self.root)
                self.assertEqual(caught.exception.code, "invalid_arguments")

    def test_a_port_converts_only_at_its_declared_bit_widths(self):
        with self.assertRaises(ConvertError) as caught:
            plan_convert("convaiinnovations/laya", q_bits=4, backend="mlx-embeddings", model_type="laya", backends_root_dir=self.root)
        self.assertEqual(caught.exception.code, "invalid_arguments")
        self.assertIn("--q-bits 8", caught.exception.remediation)
        self.assertEqual(plan_convert("pub/model", q_bits=4, backend="mlx-embeddings", backends_root_dir=self.root)["q_bits"], 4)

    def cache(self, repo, subfolder):
        entry = self.root / "hub" / "models--{0}".format(repo.replace("/", "--"))
        (entry / "refs").mkdir(parents=True)
        (entry / "refs" / "main").write_text("abc123\n", encoding="utf-8")
        folder = entry / "snapshots" / "abc123" / subfolder
        folder.mkdir(parents=True)
        return folder

    def test_a_recipe_converts_from_the_cached_base_and_lora(self):
        base = self.root / "hub" / "models--Qwen--Qwen-Image-2.1" / "snapshots" / "d26bb61231c349cf6b7896fa83353113880e1ba3"
        base.mkdir(parents=True)
        with self.assertRaises(ConvertError) as caught:
            plan_convert("abenzerps/Qwen-Image-2.1-Uncensored-GGUF", q_bits=8, backend="mflux", model_type="qwen_image_21",
                         hf_cache=str(self.root / "hub"), backends_root_dir=self.root)
        self.assertEqual(caught.exception.code, "recipe_not_cached")
        lora = self.cache("abenzerps/Qwen-Image-2.1-Uncensored-GGUF", ".")
        (lora / "qwen-image-2.1-uncensored-lora.safetensors").write_bytes(b"x")
        plan = plan_convert("abenzerps/Qwen-Image-2.1-Uncensored-GGUF", q_bits=8, backend="mflux", model_type="qwen_image_21",
                            out=str(self.root / "out"), hf_cache=str(self.root / "hub"), backends_root_dir=self.root)
        argv = plan["argv"]
        self.assertEqual(argv[1:3], ["-m", "mflux.mlx_agent_ports.qwen_image_21.convert"])
        self.assertEqual(argv[argv.index("--hf-path") + 1], str(base))
        self.assertEqual(Path(argv[argv.index("--lora") + 1]).resolve(), (lora / "qwen-image-2.1-uncensored-lora.safetensors").resolve())
        self.assertEqual((argv[-2], argv[-1]), ("--lora-scale", "1.0"))
        self.assertEqual(plan["recipe"]["base"], "Qwen/Qwen-Image-2.1")
        venv = self.root / "mflux"
        (venv / "lib" / "python3.12" / "site-packages" / "mflux").mkdir(parents=True)
        (venv / "bin").mkdir()
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / ".mlx-agent-backend.json").write_text(json.dumps({"id": "mflux", "version": "0.20.0"}), encoding="utf-8")
        outcome = start_convert(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan["preview_hash"],
                                spawn=lambda argv, log: 99, model_present=lambda repo: True)
        RECEIPT_SCHEMA.validate(outcome["receipt"])
        self.assertTrue((venv / "lib" / "python3.12" / "site-packages" / "mflux" / "mlx_agent_ports" / "qwen_image_21" / "convert.py").is_file())
        pipeline = plan_convert("Qwen/Qwen-Image-2.1", q_bits=4, backend="mflux", model_type="qwen_image_21", backends_root_dir=self.root)
        self.assertEqual(pipeline["argv"][pipeline["argv"].index("--hf-path") + 1], "Qwen/Qwen-Image-2.1")
        self.assertNotIn("recipe", pipeline)

    def test_a_wan_checkpoint_converts_with_the_mlx_video_port_converter(self):
        plan = plan_convert("Wan-AI/Wan2.1-T2V-1.3B", hf_cache=self.root / "empty-cache", q_bits=4, backend="mlx-video", model_type="t2v",
                            out=str(self.root / "wan"), backends_root_dir=self.root)
        argv = plan["argv"]
        self.assertEqual(argv[1:3], ["-m", "mlx_video.mlx_agent_ports.t2v.convert"])
        self.assertEqual(argv[argv.index("--hf-path") + 1], "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertEqual((argv[argv.index("--q-bits") + 1], "--quantize" in argv), ("4", True))
        self.assertEqual((plan["backend"], plan["ports"], plan["port_converter"]), ("mlx-video", ["t2v"], "mlx_video.mlx_agent_ports.t2v.convert"))
        self.assertNotIn("recipe", plan)

    def test_a_subfolder_receipt_matches_the_receipt_schema(self):
        self.cache("org/name", "chat")
        plan = plan_convert("org/name", subfolder="chat", out=str(self.root / "out"), hf_cache=str(self.root / "hub"))
        outcome = start_convert(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan["preview_hash"],
                                spawn=lambda argv, log: 99, model_present=lambda repo: True, which=lambda name: "/bin/x")
        RECEIPT_SCHEMA.validate(outcome["receipt"])
        self.assertEqual(outcome["receipt"]["source"]["subfolder"], "chat")

    def test_a_subfolder_converts_from_its_cached_folder_with_any_converter(self):
        folder = self.cache("convaiinnovations/laya", "multilingual")
        plan = plan_convert("convaiinnovations/laya", q_bits=8, backend="mlx-embeddings", model_type="laya",
                            subfolder="multilingual", hf_cache=str(self.root / "hub"), backends_root_dir=self.root)
        self.assertEqual(plan["argv"][plan["argv"].index("--hf-path") + 1], str(folder))
        self.assertEqual((plan["out"], plan["slug"]), ("laya-multilingual-MLX-8bit", "laya-multilingual-8bit"))
        self.assertEqual({key: plan["source"][key] for key in ("kind", "repo", "subfolder")}, {"kind": "hf-cache", "repo": "convaiinnovations/laya", "subfolder": "multilingual"})
        self.assertEqual(plan["source"]["path"], str(folder))
        chat = self.cache("org/name", "a/chat")
        builtin = plan_convert("org/name", subfolder="a/chat", hf_cache=str(self.root / "hub"))
        self.assertEqual(builtin["argv"][builtin["argv"].index("--hf-path") + 1], str(chat))
        self.assertEqual(builtin["out"], "name-a-chat-MLX-4bit")
        self.assertEqual(plan_convert("org/name")["source"], {"kind": "hf-cache", "repo": "org/name"})
        for subfolder, code in (("missing", "subfolder_not_cached"), ("../x", "invalid_arguments")):
            with self.subTest(subfolder=subfolder):
                with self.assertRaises(ConvertError) as caught:
                    plan_convert("org/name", subfolder=subfolder, hf_cache=str(self.root / "hub"))
                self.assertEqual(caught.exception.code, code)

    def test_start_with_installed_backend_records_it_and_syncs_ports(self):
        plan = plan_convert("openai/whisper-tiny", backend="mlx-audio", out=str(self.root / "out"), backends_root_dir=self.root)
        venv = self.root / "mlx-audio"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / ".mlx-agent-backend.json").write_text(json.dumps({"id": "mlx-audio", "version": "0.5.7"}), encoding="utf-8")
        models = venv / "lib" / "python3.12" / "site-packages" / "mlx_audio" / "stt" / "models"
        models.mkdir(parents=True)
        spawned = []
        outcome = start_convert(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan["preview_hash"],
                                spawn=lambda argv, log: spawned.append(argv) or 99, model_present=lambda repo: True)
        self.assertEqual(outcome["receipt"]["backend"], "mlx-audio")
        RECEIPT_SCHEMA.validate(outcome["receipt"])
        self.assertEqual(spawned[0], plan["argv"])
        self.assertTrue((models / "audio8_asr_infinite" / "__init__.py").is_file())
        self.assertTrue((models / "audio8_asr_infinite" / ".mlx-agent-port.json").is_file())

    def test_refused_start_leaves_the_backend_untouched(self):
        plan = plan_convert("openai/whisper-tiny", backend="mlx-audio", out=str(self.root / "out"), backends_root_dir=self.root)
        venv = self.root / "mlx-audio"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / ".mlx-agent-backend.json").write_text(json.dumps({"id": "mlx-audio", "version": "0.5.7"}), encoding="utf-8")
        models = venv / "lib" / "python3.12" / "site-packages" / "mlx_audio" / "stt" / "models"
        models.mkdir(parents=True)
        (self.root / "out").mkdir()
        with self.assertRaises(ConvertError) as caught:
            start_convert(plan, receipts_dir=str(self.root), confirm=True, preview_hash=plan["preview_hash"],
                          spawn=lambda argv, log: 99, model_present=lambda repo: True)
        self.assertEqual(caught.exception.code, "output_exists")
        self.assertEqual(list(models.iterdir()), [])


class StartConvertTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.spawned = []

    def _spawn(self, argv, log_path):
        self.spawned.append((argv, log_path))
        return 7777

    def _start(self, plan, **overrides):
        kwargs = {
            "receipts_dir": str(self.root),
            "confirm": True,
            "preview_hash": plan["preview_hash"],
            "which": lambda executable: "/usr/local/bin/" + executable,
            "model_present": lambda repo: True,
            "spawn": self._spawn,
            "now": lambda: "2026-07-26T00:00:00+00:00",
        }
        kwargs.update(overrides)
        return start_convert(plan, **kwargs)

    def test_preview_without_confirm(self):
        outcome = start_convert(plan_convert("pub/model"), receipts_dir=str(self.root))
        self.assertEqual(outcome["status"], "preview")
        self.assertEqual(self.spawned, [])

    def test_confirm_requires_preview_hash(self):
        with self.assertRaises(ConvertError) as caught:
            self._start(plan_convert("pub/model"), preview_hash=None)
        self.assertEqual(caught.exception.code, "preview_hash_required")

    def test_stale_preview_hash(self):
        with self.assertRaises(ConvertError) as caught:
            self._start(plan_convert("pub/model"), preview_hash="0" * 64)
        self.assertEqual(caught.exception.code, "preview_stale")

    def test_missing_executable(self):
        with self.assertRaises(ConvertError) as caught:
            self._start(plan_convert("pub/model"), which=lambda executable: None)
        self.assertEqual(caught.exception.code, "runtime_not_installed")

    def test_model_gate(self):
        with self.assertRaises(ConvertError) as caught:
            self._start(plan_convert("pub/model"), model_present=lambda repo: False)
        self.assertEqual(caught.exception.code, "model_not_local")

    def test_output_exists_gate(self):
        with TemporaryDirectory() as other:
            plan = plan_convert("pub/model", out=other)
            with self.assertRaises(ConvertError) as caught:
                self._start(plan)
            self.assertEqual(caught.exception.code, "output_exists")

    def test_success_writes_receipt(self):
        outcome = self._start(plan_convert("pub/model", out=str(self.root / "out-new")))
        self.assertEqual(outcome["status"], "started")
        receipt = outcome["receipt"]
        self.assertEqual(receipt["pid"], 7777)
        self.assertEqual(receipt["kind"], "convert")
        receipts = list((self.root / ".mlx-agent-receipts" / "convert").glob("*.json"))
        self.assertEqual(len(receipts), 1)

    def test_running_job_blocks_second_start(self):
        plan = plan_convert("pub/model", out=str(self.root / "out-new"))
        self._start(plan, pid_alive=lambda pid: True)
        second = plan_convert("pub/other", out=str(self.root / "out-2"))
        with self.assertRaises(ConvertError) as caught:
            start_convert(
                second,
                receipts_dir=str(self.root),
                confirm=True,
                preview_hash=second["preview_hash"],
                which=lambda executable: "/bin/x",
                spawn=self._spawn,
                pid_alive=lambda pid: True,
            )
        self.assertEqual(caught.exception.code, "job_in_progress")


class StatusConvertTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def _start_one(self, out):
        plan = plan_convert("pub/model", out=str(out))
        return start_convert(
            plan,
            receipts_dir=str(self.root),
            confirm=True,
            preview_hash=plan["preview_hash"],
            which=lambda executable: "/bin/x",
            spawn=lambda argv, log_path: 7777,
            now=lambda: "2026-07-26T00:00:00+00:00",
        )

    _COMMAND = "mlx_lm.convert --hf-path pub/model --mlx-path out-new --quantize --q-bits 4"

    def test_running_job(self):
        out = self.root / "out-new"
        self._start_one(out)
        entries = status_convert(
            str(self.root),
            pid_alive=lambda pid: True,
            pid_command=lambda pid: self._COMMAND,
        )
        self.assertEqual(entries[0]["state"], "running")

    def test_argv_mismatch_is_unknown(self):
        out = self.root / "out-new"
        self._start_one(out)
        entries = status_convert(
            str(self.root),
            pid_alive=lambda pid: True,
            pid_command=lambda pid: "python something-else",
        )
        self.assertEqual(entries[0]["state"], "unknown")

    def test_completed_job_marks_done_once(self):
        out = self.root / "out-new"
        self._start_one(out)
        out.mkdir()
        entries = status_convert(str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(entries[0]["state"], "done")
        self.assertIsNotNone(entries[0]["completed_at"])
        again = status_convert(str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(again[0]["state"], "done")

    def test_missing_output_marks_failed(self):
        out = self.root / "out-new"
        self._start_one(out)
        entries = status_convert(str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(entries[0]["state"], "failed")


class PlanGGUFConvertTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runner = self.root / "runner.py"
        self.runner.write_text("", encoding="utf-8")

    def _plan(self, filename="alpha-model-Q4_K_M.gguf", **kwargs):
        source = write_gguf(self.root / filename, name="Alpha Model")
        return plan_gguf_convert(source, runner=self.runner, **kwargs)

    def test_plan_argv_targets_the_runner(self):
        plan = self._plan()
        self.assertEqual(plan["source"]["kind"], "gguf")
        self.assertEqual(plan["q_bits"], 4)
        self.assertEqual(plan["out"], "alpha-model-Q4_K_M-MLX-4bit")
        self.assertEqual(plan["argv"][1], str(self.runner))
        self.assertIn("--gguf", plan["argv"])
        self.assertEqual(len(plan["preview_hash"]), 64)
        self.assertEqual(plan["slug"], "alpha-model-Q4_K_M-4bit")

    def test_plan_is_deterministic(self):
        self.assertEqual(self._plan()["preview_hash"], self._plan()["preview_hash"])

    def test_slug_is_filename_safe(self):
        plan = self._plan(filename="weird name (1)-Q4_K_M.gguf")
        self.assertNotIn(" ", plan["slug"])
        self.assertNotIn("/", plan["slug"])

    def test_rejects_non_gguf_suffix(self):
        other = self.root / "model.bin"
        other.write_bytes(b"x")
        with self.assertRaises(ConvertError) as caught:
            plan_gguf_convert(other, runner=self.runner)
        self.assertEqual(caught.exception.code, "invalid_source")

    def test_rejects_missing_file(self):
        with self.assertRaises(ConvertError) as caught:
            plan_gguf_convert(self.root / "absent.gguf", runner=self.runner)
        self.assertEqual(caught.exception.code, "source_not_found")

    def test_rejects_unreadable_header(self):
        broken = self.root / "broken.gguf"
        broken.write_bytes(b"NOPE" + b"\x00" * 16)
        with self.assertRaises(ConvertError) as caught:
            plan_gguf_convert(broken, runner=self.runner)
        self.assertEqual(caught.exception.code, "not_gguf")

    def test_rejects_non_first_shard(self):
        source = write_gguf(self.root / "beta-Q4_K_M-00002-of-00003.gguf", name="Beta")
        with self.assertRaises(ConvertError) as caught:
            plan_gguf_convert(source, runner=self.runner)
        self.assertEqual(caught.exception.code, "shard_not_first")

    def test_dspark_drafter_runs_the_port(self):
        source = write_gguf(self.root / "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", architecture="dflash",
                            name="DeepSeek-V4-Flash-0731", extra=(_kv("dflash.hyper_connection.count", _UINT32, 4),))
        plan = plan_gguf_convert(source, runner=self.runner)
        self.assertEqual(plan["argv"][-2:], ["--port", "deepseek_v4_dspark"])
        self.assertEqual(plan["port_converter"], "deepseek_v4_dspark")
        self.assertEqual(plan["draft"]["target"], "DeepSeek-V4-Flash-0731")
        self.assertEqual(plan["out"], "dspark-DeepSeek-V4-Flash-0731-Q8_0-MLX-4bit")

    def test_rejects_drafters_no_port_converts(self):
        source = write_gguf(self.root / "eagle3-Llama.gguf", architecture="eagle3", name="Llama 3.1 8B")
        with self.assertRaises(ConvertError) as caught:
            plan_gguf_convert(source, runner=self.runner)
        self.assertEqual(caught.exception.code, "unsupported_draft")
        self.assertIn("Llama 3.1 8B", str(caught.exception))

    def test_rejects_bad_q_bits(self):
        source = write_gguf(self.root / "gamma-Q4_K_M.gguf", name="Gamma")
        with self.assertRaises(ConvertError) as caught:
            plan_gguf_convert(source, q_bits=6, runner=self.runner)
        self.assertEqual(caught.exception.code, "invalid_arguments")


class StartGGUFConvertTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runner = self.root / "runner.py"
        self.runner.write_text("", encoding="utf-8")
        self.source = write_gguf(self.root / "delta-model-Q4_K_M.gguf", name="Delta Model")
        self.spawned = []

    def _plan(self):
        return plan_gguf_convert(
            self.source, out=str(self.root / "out-new"), runner=self.runner
        )

    def _start(self, plan, **overrides):
        kwargs = {
            "receipts_dir": str(self.root),
            "confirm": True,
            "preview_hash": plan["preview_hash"],
            "which": lambda executable: "/usr/local/bin/" + executable,
            "spawn": lambda argv, log_path: self.spawned.append(argv) or 4242,
            "now": lambda: "2026-07-28T00:00:00+00:00",
            "module_present": lambda names: [],
        }
        kwargs.update(overrides)
        return start_convert(plan, **kwargs)

    def test_missing_modules_are_gated(self):
        with self.assertRaises(ConvertError) as caught:
            self._start(self._plan(), module_present=lambda names: ["torch", "gguf"])
        self.assertEqual(caught.exception.code, "runtime_not_installed")
        self.assertIn("torch", str(caught.exception))

    def test_port_conversions_need_gguf_and_mlx_only(self):
        source = write_gguf(self.root / "dspark.gguf", architecture="dflash", name="DeepSeek-V4-Flash-0731",
                            extra=(_kv("dflash.hyper_connection.count", _UINT32, 4),))
        plan = plan_gguf_convert(source, out=str(self.root / "drafter-out"), runner=self.runner)
        asked = []
        self._start(plan, module_present=lambda names: asked.append(names) or [])
        self.assertEqual(asked, [("gguf", "mlx")])
        self.assertEqual(self.spawned[0][-2:], ["--port", "deepseek_v4_dspark"])
        with self.assertRaises(ConvertError) as caught:
            self._start(plan_gguf_convert(source, out=str(self.root / "drafter-two"), runner=self.runner),
                        module_present=lambda names: ["mlx"], pid_alive=lambda pid: False)
        self.assertIn("uv pip install gguf mlx", caught.exception.remediation)

    def test_deleted_source_is_gated(self):
        plan = self._plan()
        self.source.unlink()
        with self.assertRaises(ConvertError) as caught:
            self._start(plan)
        self.assertEqual(caught.exception.code, "source_not_found")

    def test_receipt_records_the_gguf_source(self):
        outcome = self._start(self._plan())
        receipt = outcome["receipt"]
        self.assertEqual(receipt["source"]["kind"], "gguf")
        self.assertEqual(receipt["source"]["path"], str(self.source))
        self.assertEqual(receipt["slug"], "delta-model-Q4_K_M-4bit")
        written = sorted((self.root / ".mlx-agent-receipts" / "convert").glob("*.json"))
        self.assertEqual([path.name for path in written], ["delta-model-Q4_K_M-4bit.json"])
        self.assertEqual(len(self.spawned), 1)

    def test_load_receipts_round_trip(self):
        self._start(self._plan())
        receipts = load_receipts(str(self.root))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["source"]["kind"], "gguf")

    def test_status_reports_the_source_kind(self):
        self._start(self._plan())
        entries = status_convert(str(self.root), pid_alive=lambda pid: False)
        self.assertEqual(entries[0]["source"], "gguf")
        self.assertEqual(entries[0]["state"], "failed")


if __name__ == "__main__":
    unittest.main()
