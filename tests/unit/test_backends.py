import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.backends import (
    BackendError,
    INSTALL_MARKER,
    PORT_MARKER,
    PORTS_DIR,
    backend_environment,
    backend_python,
    backend_target,
    backends_root,
    choose_backend,
    component_matches,
    is_installed,
    is_vision_type,
    load_manifests,
    load_registries,
    lookup,
    lookup_squashed,
    probe_sources_from_directory,
    probe_sources_from_wheel,
    quantize_rule,
    registry_from_sources,
    sync_ports,
    with_ports,
)

from .backend_fixtures import synthetic_manifests, synthetic_registries

AUDIO_SOURCES = {
    "stt/models/whisper/__init__.py": "from .whisper import Model, ModelConfig\n",
    "stt/models/voxtral_realtime/__init__.py": "from .voxtral_realtime import Model\n",
    "stt/models/helpers/__init__.py": "from .x import Helper\n",
    "stt/models/base.py": "class BaseModelArgs:\n    pass\n",
    "stt/models/_private/__init__.py": "from .x import Model\n",
    "stt/utils.py": "MODEL_REMAPPING = {\"parakeet_tdt\": \"parakeet\", \"glm\": \"glmasr\"}\n",
    "tts/models/kokoro/__init__.py": "from .kokoro import Model\n",
    "tts/utils.py": "MODEL_REMAPPING: dict = {}\n",
}


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.manifests = synthetic_manifests()
        self.registries = synthetic_registries(self.manifests)

    def test_registry_from_sources_keeps_model_modules_only(self):
        registry = registry_from_sources(AUDIO_SOURCES, self.manifests["mlx-audio"])
        self.assertEqual(registry["speech_to_text"]["model_types"], ["voxtral_realtime", "whisper"])
        self.assertEqual(registry["speech_to_text"]["remapping"], {"glm": "glmasr", "parakeet_tdt": "parakeet"})
        self.assertEqual(registry["text_to_speech"]["model_types"], ["kokoro"])

    def test_flat_modules_with_model_class_count(self):
        manifest = self.manifests["mlx-lm"]
        sources = {
            "models/llama.py": "class ModelArgs:\n    pass\nclass Model:\n    pass\n",
            "models/cache.py": "class KVCache:\n    pass\n",
            "utils.py": "MODEL_REMAPPING = {\"mistral\": \"llama\"}\n",
        }
        registry = registry_from_sources(sources, manifest)
        self.assertEqual(registry["text_llm"], {"model_types": ["llama"], "remapping": {"mistral": "llama"}})

    def test_wheel_and_directory_probes_read_the_same_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "mlx_audio-0.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for relative, text in AUDIO_SOURCES.items():
                    archive.writestr("mlx_audio/" + relative, text)
                archive.writestr("mlx_audio/stt/models/whisper/whisper.py", "class Model:\n    pass\n")
            package = root / "site" / "mlx_audio"
            for relative, text in AUDIO_SOURCES.items():
                location = package / relative
                location.parent.mkdir(parents=True, exist_ok=True)
                location.write_text(text, encoding="utf-8")
            manifest = self.manifests["mlx-audio"]
            from_wheel = probe_sources_from_wheel(str(wheel), manifest)
            from_directory = probe_sources_from_directory(package, manifest)
            self.assertEqual(from_wheel, from_directory)
            self.assertNotIn("stt/models/whisper/whisper.py", from_wheel)

    def test_lookup_exact_remap_and_normalization(self):
        hits = lookup("Qwen2", self.manifests, self.registries)
        self.assertEqual(
            [(h["backend"], h["category"], h["match"], h["module"]) for h in hits],
            [("mlx-lm", "text_llm", "exact", "mlx_lm.models.qwen2"),
             ("mlx-vlm", "vision_language", "exact", "mlx_vlm.models.qwen2")],
        )
        remap = lookup("mistral", self.manifests, self.registries)
        self.assertEqual([(h["backend"], h["match"], h["module"]) for h in remap], [("mlx-lm", "remap", "mlx_lm.models.llama")])
        self.assertEqual(lookup("audio8_asr_infinite", self.manifests, self.registries), [])
        self.assertEqual(lookup(None, self.manifests, self.registries), [])

    def test_component_matches_strips_role_suffixes(self):
        hits = component_matches("voxtral_realtime_encoder", self.manifests, self.registries)
        self.assertEqual(hits, [{
            "backend": "mlx-audio", "category": "speech_to_text", "match": "stripped",
            "module": "mlx_audio.stt.models.voxtral_realtime",
        }])

    def test_lookup_squashed_matches_gguf_architecture_names(self):
        hits = lookup_squashed("qwen3moe", self.manifests, self.registries)
        self.assertEqual([(h["backend"], h["match"]) for h in hits], [("mlx-lm", "squashed")])

    def test_choose_backend_follows_task_preference(self):
        qwen2 = lookup("qwen2", self.manifests, self.registries)
        self.assertEqual(choose_backend(qwen2, "text_llm"), "mlx-lm")
        self.assertEqual(choose_backend(qwen2, "text_llm", has_vision_config=True), "mlx-vlm")
        self.assertEqual(choose_backend(lookup("qwen2_vl", self.manifests, self.registries), "vision_language"), "mlx-vlm")
        self.assertEqual(choose_backend(lookup("whisper", self.manifests, self.registries), "speech_to_text"), "mlx-audio")
        self.assertIsNone(choose_backend(qwen2, "speech_to_text"))
        glm = lookup("glm", self.manifests, self.registries)
        self.assertEqual(choose_backend(glm, "text_llm"), "mlx-lm")


class VisionRegistryTests(unittest.TestCase):
    SOURCES = {
        "models/moondream3/__init__.py": "from .moondream3 import Model\n",
        "models/moondream3/image_crops.py": "def crop():\n    pass\n",
        "models/glm4_moe_lite/__init__.py": "from .glm4_moe_lite import Model\n",
        "models/glm4_moe_lite/language.py": "class LanguageModel:\n    pass\n",
        "utils.py": "MODEL_REMAPPING = {}\n",
    }

    def test_vision_types_come_from_vision_files(self):
        manifest = synthetic_manifests()["mlx-vlm"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "mlx_vlm-0.0-py3-none-any.whl"
            package = root / "site" / "mlx_vlm"
            with zipfile.ZipFile(wheel, "w") as archive:
                for relative, text in self.SOURCES.items():
                    archive.writestr("mlx_vlm/" + relative, text)
                    location = package / relative
                    location.parent.mkdir(parents=True, exist_ok=True)
                    location.write_text(text, encoding="utf-8")
            from_wheel = registry_from_sources(probe_sources_from_wheel(str(wheel), manifest), manifest)
            from_directory = registry_from_sources(probe_sources_from_directory(package, manifest), manifest)
        self.assertEqual(from_wheel, from_directory)
        entry = from_wheel["vision_language"]
        self.assertEqual(entry["model_types"], ["glm4_moe_lite", "moondream3"])
        self.assertEqual(entry["vision_types"], ["moondream3"])

    def test_is_vision_type(self):
        manifests = synthetic_manifests()
        registries = synthetic_registries(manifests)
        self.assertTrue(is_vision_type("moondream3", manifests, registries))
        self.assertTrue(is_vision_type("Qwen2-VL", manifests, registries))
        self.assertFalse(is_vision_type("glm4_moe_lite", manifests, registries))
        self.assertFalse(is_vision_type("whisper", manifests, registries))


class InstallStateTests(unittest.TestCase):
    def test_marker_and_python_make_an_optional_backend_installed(self):
        manifest = synthetic_manifests()["mlx-audio"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(is_installed(manifest, root))
            target = backend_target(manifest, root)
            (target / "bin").mkdir(parents=True)
            (target / "bin" / "python").write_text("", encoding="utf-8")
            (target / INSTALL_MARKER).write_text(json.dumps({"id": "mlx-audio", "version": "0.0-test"}), encoding="utf-8")
            self.assertTrue(is_installed(manifest, root))
            self.assertEqual(backend_python(manifest, root), target / "bin" / "python")
            (target / INSTALL_MARKER).write_text(json.dumps({"id": "mlx-audio", "version": "9.9"}), encoding="utf-8")
            self.assertFalse(is_installed(manifest, root))

    def test_builtin_installed_follows_find_spec(self):
        manifests = synthetic_manifests()
        registries = load_registries(manifests, root=Path("/nonexistent"), find_spec=lambda name: None)
        self.assertEqual(registries["mlx-lm"], {"registry": manifests["mlx-lm"]["registry"], "source": "snapshot", "installed": False})

    def test_backends_root_honours_xdg(self):
        self.assertEqual(backends_root({"XDG_DATA_HOME": "/x"}), Path("/x/mlx-workbench/backends"))
        self.assertEqual(backends_root({}), Path.home() / ".local/share/mlx-workbench/backends")

    def test_backend_environment_is_an_allowlist(self):
        env = backend_environment({"PATH": "/bin", "HOME": "/h", "HF_HOME": "/hf", "AWS_SECRET_ACCESS_KEY": "x", "OPENAI_API_KEY": "y"})
        self.assertEqual(env, {"PATH": "/bin", "HOME": "/h", "HF_HOME": "/hf"})


class ManifestTests(unittest.TestCase):
    def test_declared_manifests_load(self):
        manifests = load_manifests()
        self.assertEqual(sorted(manifests), ["mlx-audio", "mlx-lm", "mlx-vlm"])
        self.assertTrue(manifests["mlx-lm"]["builtin"])
        self.assertEqual(manifests["mlx-audio"]["version"], "0.5.7")
        self.assertEqual(manifests["mlx-vlm"]["version"], "0.7.4")

    def test_invalid_manifest_is_refused(self):
        with TemporaryDirectory() as directory:
            bad = dict(synthetic_manifests()["mlx-audio"], lock=None)
            Path(directory, "mlx-audio.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(BackendError) as caught:
                load_manifests(Path(directory))
            self.assertEqual(caught.exception.code, "manifest_invalid")

class PortTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = load_manifests()["mlx-audio"]
        self.models = self.root / "mlx-audio" / "lib" / "python3.12" / "site-packages" / "mlx_audio" / "stt" / "models"

    def install(self):
        self.models.mkdir(parents=True)

    def write_manifest(self, ports):
        directory = self.root / "manifests"
        directory.mkdir(exist_ok=True)
        manifest = dict(self.manifest, ports=ports)
        (directory / "mlx-audio.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    def test_manifest_ports_must_name_modules_in_declared_categories(self):
        self.assertEqual(self.manifest["ports"], {"speech_to_text": ["audio8_asr_infinite"]})
        for bad in ({"text_llm": ["x"]}, {"speech_to_text": "x"}, {"speech_to_text": ["../x"]}):
            with self.assertRaises(BackendError) as caught:
                load_manifests(self.write_manifest(bad))
            self.assertEqual(caught.exception.code, "manifest_invalid")

    def test_port_quantize_rules_name_declared_ports(self):
        manifests = load_manifests()
        self.assertEqual(quantize_rule("Audio8_ASR_Infinite", "mlx-audio", manifests),
                         {"include": ["language_model."], "exclude": ["ada_rms_norm"]})
        self.assertIsNone(quantize_rule("whisper", "mlx-audio", manifests))
        self.assertIsNone(quantize_rule("audio8_asr_infinite", None, manifests))
        directory = self.write_manifest(self.manifest["ports"])
        for bad in ({"whisper": {"include": ["x."]}}, {"audio8_asr_infinite": {"only": ["x."]}},
                    {"audio8_asr_infinite": {"include": "x."}}):
            manifest = dict(self.manifest, port_quantize=bad)
            (directory / "mlx-audio.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(BackendError) as caught:
                load_manifests(directory)
            self.assertEqual(caught.exception.code, "manifest_invalid")

    def test_ports_join_the_registry_whether_or_not_the_backend_is_installed(self):
        manifests = load_manifests()
        self.assertNotIn("audio8_asr_infinite", self.manifest["registry"]["speech_to_text"]["model_types"])
        merged = with_ports(self.manifest["registry"], self.manifest)
        self.assertIn("audio8_asr_infinite", merged["speech_to_text"]["model_types"])
        self.assertIn("whisper", merged["speech_to_text"]["model_types"])
        registries = load_registries(manifests, root=self.root, find_spec=lambda name: None)
        hits = lookup("audio8_asr_infinite", manifests, registries)
        self.assertEqual(hits, [{
            "backend": "mlx-audio", "category": "speech_to_text", "match": "exact",
            "module": "mlx_audio.stt.models.audio8_asr_infinite",
        }])
        self.assertEqual(lookup("audio8_asr_infinite", manifests, {}), hits)

    def test_sync_copies_once_marks_and_replaces_a_stale_copy(self):
        self.install()
        self.assertEqual(sync_ports(self.manifest, self.root), ["audio8_asr_infinite"])
        target = self.models / "audio8_asr_infinite"
        sources = sorted(path.name for path in (PORTS_DIR / "mlx-audio" / "audio8_asr_infinite").glob("*.py"))
        self.assertEqual(sorted(path.name for path in target.glob("*.py")), sources)
        marker = json.loads((target / PORT_MARKER).read_text(encoding="utf-8"))
        self.assertEqual((marker["backend"], marker["port"]), ("mlx-audio", "audio8_asr_infinite"))
        self.assertEqual(sync_ports(self.manifest, self.root), [])
        (target / "config.py").write_text("stale = True\n", encoding="utf-8")
        (target / PORT_MARKER).write_text(json.dumps(dict(marker, sha256="stale")), encoding="utf-8")
        self.assertEqual(sync_ports(self.manifest, self.root), ["audio8_asr_infinite"])
        self.assertNotIn("stale", (target / "config.py").read_text(encoding="utf-8"))
        self.assertEqual(sorted(path.name for path in self.models.iterdir()), ["audio8_asr_infinite"])

    def test_sync_refuses_a_module_the_backend_ships_and_symlinks(self):
        self.install()
        (self.models / "audio8_asr_infinite").mkdir()
        with self.assertRaises(BackendError) as caught:
            sync_ports(self.manifest, self.root)
        self.assertEqual(caught.exception.code, "port_conflict")
        (self.models / "audio8_asr_infinite").rename(self.root / "elsewhere")
        (self.models / "audio8_asr_infinite").symlink_to(self.root / "elsewhere")
        with self.assertRaises(BackendError) as caught:
            sync_ports(self.manifest, self.root)
        self.assertEqual(caught.exception.code, "port_target_symlink")

    def test_sync_needs_an_installed_backend_and_a_shipped_port(self):
        with self.assertRaises(BackendError) as caught:
            sync_ports(self.manifest, self.root)
        self.assertEqual(caught.exception.code, "backend_not_installed")
        self.install()
        with self.assertRaises(BackendError) as caught:
            sync_ports(self.manifest, self.root, ports_dir=self.root / "no-ports")
        self.assertEqual(caught.exception.code, "port_missing")
        self.assertEqual(sync_ports(load_manifests()["mlx-vlm"], self.root), [])


if __name__ == "__main__":
    unittest.main()
