import unittest

from mlx_agent.backends import load_manifests


class CommittedSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.manifests = load_manifests()

    def types(self, backend_id, category):
        return set(self.manifests[backend_id]["registry"][category]["model_types"])

    def test_snapshots_are_populated(self):
        self.assertGreater(len(self.types("mlx-lm", "text_llm")), 80)
        self.assertGreater(len(self.types("mlx-vlm", "vision_language")), 80)
        self.assertGreater(len(self.types("mlx-audio", "speech_to_text")), 10)
        self.assertGreater(len(self.types("mlx-audio", "text_to_speech")), 3)

    def test_known_architectures(self):
        self.assertIn("qwen2", self.types("mlx-lm", "text_llm"))
        self.assertIn("voxtral_realtime", self.types("mlx-audio", "speech_to_text"))
        self.assertIn("whisper", self.types("mlx-audio", "speech_to_text"))
        self.assertIn("qwen2_vl", self.types("mlx-vlm", "vision_language"))
        for backend_id, manifest in self.manifests.items():
            for category, entry in manifest["registry"].items():
                self.assertNotIn("audio8_asr_infinite", entry["model_types"], (backend_id, category))


if __name__ == "__main__":
    unittest.main()
