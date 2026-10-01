import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.taxonomy import annotate_inventory, classify, read_config_type, use_cases_for

from .backend_fixtures import synthetic_manifests, synthetic_registries


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        self.manifests = synthetic_manifests()
        self.registries = synthetic_registries(self.manifests)

    def classify(self, name, **kwargs):
        return classify(name, manifests=self.manifests, registries=self.registries, **kwargs)

    def test_pipeline_tag_wins_and_is_confirmed(self):
        task = self.classify(
            "Edge0/Audio8-ASR-Infinite",
            tags=["streaming", "realtime", "speech-recognition"],
            pipeline_tag="automatic-speech-recognition",
            model_type="audio8_asr_infinite",
        )
        self.assertEqual(task, {
            "type": "speech_to_text",
            "use_cases": ["realtime_transcription", "transcription"],
            "source": "pipeline_tag",
            "confidence": "confirmed",
        })

    def test_registry_prefers_exact_hits_and_vision_config(self):
        self.assertEqual(self.classify("x", model_type="whisper")["type"], "speech_to_text")
        self.assertEqual(self.classify("x", model_type="glm")["type"], "text_llm")
        self.assertEqual(self.classify("x", model_type="qwen2_vl", config_keys=("vision_config",))["type"], "vision_language")
        self.assertEqual(self.classify("x", model_type="qwen2_vl")["type"], "text_llm")
        self.assertEqual(self.classify("x", model_type="qwen2")["source"], "registry")

    def test_gguf_architecture_is_likely(self):
        task = self.classify("Qwen3-30B-A3B-Q4_K_M.gguf", gguf_architecture="qwen3moe", local=True)
        self.assertEqual((task["type"], task["source"], task["confidence"]), ("text_llm", "gguf_architecture", "likely"))

    def test_name_fallbacks_and_defaults(self):
        self.assertEqual(self.classify("org/kokoro-tts-mlx")["type"], "text_to_speech")
        self.assertEqual(self.classify("org/my-whisper-asr")["type"], "speech_to_text")
        self.assertEqual(self.classify("org/bge-embedding-small")["type"], "embedding")
        self.assertEqual(self.classify("org/llava-next-vl")["type"], "vision_language")
        self.assertEqual(self.classify("org/mystery")["type"], "other")
        local = self.classify("mystery.gguf", local=True)
        self.assertEqual((local["type"], local["source"]), ("text_llm", "default"))

    def test_use_cases_are_specific_first_with_default_last(self):
        self.assertEqual(use_cases_for("text_llm", "qwen2.5-coder-32b"), ["coding", "general_chat"])
        self.assertEqual(use_cases_for("text_llm", "deepseek-r1-distill thinking"), ["reasoning", "general_chat"])
        self.assertEqual(use_cases_for("text_llm", "some-encoder-decoder"), ["general_chat"])
        self.assertEqual(use_cases_for("text_llm", "starcoder2-15b"), ["coding", "general_chat"])
        self.assertEqual(use_cases_for("vision_language", "dots.ocr document"), ["ocr_documents", "vision"])
        self.assertEqual(use_cases_for("embedding", "bge-reranker"), ["reranking", "retrieval"])
        self.assertEqual(use_cases_for("other", "anything"), [])


class AnnotateInventoryTests(unittest.TestCase):
    def test_scan_items_and_outputs_get_task_labels(self):
        manifests = synthetic_manifests()
        registries = synthetic_registries(manifests)
        with TemporaryDirectory() as directory:
            vl = Path(directory, "qwen2-vl-mlx")
            vl.mkdir()
            (vl / "config.json").write_text(json.dumps({"model_type": "qwen2_vl", "vision_config": {}}), encoding="utf-8")
            whisper = Path(directory, "whisper-tiny-mlx")
            whisper.mkdir()
            (whisper / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
            report = {
                "models": [{"name": "Qwen3-30B-A3B-Q4_K_M.gguf", "architecture": "qwen3moe"}],
                "outputs": [{"name": vl.name, "path": str(vl)}, {"name": whisper.name, "path": str(whisper)}],
            }
            annotate_inventory(report, manifests=manifests, registries=registries)
        self.assertEqual(report["models"][0]["task"]["type"], "text_llm")
        self.assertEqual(report["outputs"][0]["task"]["type"], "vision_language")
        self.assertEqual(report["outputs"][1]["task"]["type"], "speech_to_text")

    def test_read_config_type_tolerates_missing_and_bad_files(self):
        with TemporaryDirectory() as directory:
            self.assertEqual(read_config_type(directory), (None, ()))
            Path(directory, "config.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(read_config_type(directory), (None, ()))


if __name__ == "__main__":
    unittest.main()
