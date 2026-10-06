import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent import cli
from mlx_agent.backends import load_manifests, load_registries
from mlx_agent.taxonomy import TASK_TYPES, annotate_inventory, classify, read_config_type, use_cases_for

from .backend_fixtures import synthetic_manifests, synthetic_registries
from .test_gguf import write_gguf


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

    def test_vision_needs_a_vision_tower(self):
        # Only mlx-vlm implements these; that alone is not evidence of vision.
        self.assertEqual(self.classify("x", model_type="glm4_moe_lite")["type"], "text_llm")
        # A module that ships vision files is vision even without config keys.
        self.assertEqual(self.classify("x", model_type="moondream3")["type"], "vision_language")
        # GGUF main weights never carry the vision tower.
        task = self.classify("moondream3.gguf", gguf_architecture="moondream3", local=True)
        self.assertEqual((task["type"], task["source"]), ("text_llm", "gguf_architecture"))
        self.assertEqual(self.classify("x.gguf", gguf_architecture="qwen2vl", local=True)["type"], "text_llm")

    def test_drafters_are_speculative_drafts(self):
        task = self.classify("dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", gguf_architecture="dflash", local=True)
        self.assertEqual(task, {"type": "speculative_draft", "use_cases": ["speculative_decoding"],
                                "source": "gguf_architecture", "confidence": "confirmed"})
        self.assertEqual(self.classify("x.gguf", gguf_architecture="eagle3", local=True)["type"], "speculative_draft")
        converted = self.classify("DeepSeek-V4.1-Flash-DSpark-drafter", pipeline_tag="text-generation",
                                  model_type="deepseek_v41_dspark", config_keys=("dspark_target_layer_ids", "model_type"))
        self.assertEqual((converted["type"], converted["source"]), ("speculative_draft", "config"))

    def test_gguf_vision_projectors_are_not_models(self):
        task = self.classify("mmproj-Qwen3.8-27B-BF16.gguf", gguf_architecture="clip", local=True)
        self.assertEqual((task["type"], task["source"], task["use_cases"]), ("other", "gguf_architecture", []))

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


class ScanTaskCliTests(unittest.TestCase):
    def test_convert_scan_json_carries_task_labels(self):
        with TemporaryDirectory() as directory:
            write_gguf(Path(directory, "tiny-Q4_K_M.gguf"))
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["convert", "scan", "--gguf-root", directory, "--no-signature", "--json"])
        envelope = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        task = envelope["data"]["models"][0]["task"]
        self.assertIn(task["type"], ("text_llm", "other"))
        self.assertEqual(set(task), {"type", "use_cases", "source", "confidence"})


class CommittedRegistryCollisionTests(unittest.TestCase):
    def setUp(self):
        self.manifests = load_manifests()
        self.registries = load_registries(
            self.manifests, root=Path("/nonexistent"), find_spec=lambda name: None
        )

    def classify(self, name, **kwargs):
        return classify(name, manifests=self.manifests, registries=self.registries, **kwargs)

    def test_llama_and_qwen3_are_text_llm_even_though_mlx_audio_lists_them(self):
        self.assertEqual(self.classify("tiny.gguf", gguf_architecture="llama", local=True)["type"], "text_llm")
        self.assertEqual(self.classify("Qwen3-8B-mlx", model_type="qwen3", local=True)["type"], "text_llm")

    def test_audio_name_breaks_the_collision_toward_speech(self):
        self.assertEqual(self.classify("orpheus-3b-tts-mlx", model_type="llama", local=True)["type"], "text_to_speech")


class ClassificationTaxonomyTests(unittest.TestCase):
    def test_text_classification_repos_and_ported_classifiers_are_classification(self):
        manifests = load_manifests()
        registries = load_registries(manifests, root=Path("/nonexistent"), find_spec=lambda name: None)
        task = classify("convaiinnovations/laya", tags=["routing", "guardrails", "moderation"],
                        pipeline_tag="text-classification", model_type="laya", manifests=manifests, registries=registries)
        self.assertEqual(task, {"type": "classification", "use_cases": ["moderation", "routing", "classification"],
                                "source": "pipeline_tag", "confidence": "confirmed"})
        self.assertEqual(classify("x", pipeline_tag="zero-shot-classification")["type"], "classification")
        local = classify("laya-MLX-4bit", model_type="laya", manifests=manifests, registries=registries, local=True)
        self.assertEqual((local["type"], local["source"]), ("classification", "registry"))
        self.assertEqual(use_cases_for("classification", "plain"), ["classification"])
        image = classify("qwen-image-2.1-uc-MLX-8bit", model_type="qwen_image_21", manifests=manifests, registries=registries, local=True)
        self.assertEqual((image["type"], image["source"]), ("image_generation", "registry"))
        video = classify("Wan2.1-T2V-1.3B-MLX-4bit", model_type="t2v", manifests=manifests, registries=registries, local=True)
        self.assertEqual((video["type"], video["use_cases"], video["source"]), ("video_generation", ["video_generation"], "registry"))
        tagged = classify("Wan-AI/Wan2.1-T2V-1.3B", pipeline_tag="text-to-video")
        self.assertEqual((tagged["type"], tagged["source"], tagged["confidence"]), ("video_generation", "pipeline_tag", "confirmed"))
        self.assertIn("video_generation", TASK_TYPES)


if __name__ == "__main__":
    unittest.main()
