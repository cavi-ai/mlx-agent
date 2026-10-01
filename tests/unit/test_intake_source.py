import unittest

from mlx_agent.intake_source import (
    IntakeSourceError,
    parse_hf_source,
    validate_file,
    validate_revision,
)

AUDIO8 = "Edge0/Audio8-ASR-Infinite"
GGUF_REPO = "unsloth/Qwen3-8B-GGUF"


class ParseHfSourceTests(unittest.TestCase):
    def test_accepted_forms(self):
        cases = {
            AUDIO8: (AUDIO8, "main", None),
            "https://huggingface.co/Edge0/Audio8-ASR-Infinite": (AUDIO8, "main", None),
            "  https://huggingface.co/Edge0/Audio8-ASR-Infinite/  ": (AUDIO8, "main", None),
            "huggingface.co/Edge0/Audio8-ASR-Infinite": (AUDIO8, "main", None),
            "https://hf.co/Edge0/Audio8-ASR-Infinite": (AUDIO8, "main", None),
            "https://WWW.HuggingFace.co/Edge0/Audio8-ASR-Infinite?library=mlx#top": (AUDIO8, "main", None),
            "https://huggingface.co/Edge0/Audio8-ASR-Infinite/tree/v1.0": (AUDIO8, "v1.0", None),
            "https://huggingface.co/Edge0/Audio8-ASR-Infinite/blob/main/README.md": (AUDIO8, "main", None),
            "https://huggingface.co/Edge0/Audio8-ASR-Infinite/discussions/3": (AUDIO8, "main", None),
            "https://huggingface.co/unsloth/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q4_K_M.gguf": (GGUF_REPO, "main", "Qwen3-8B-Q4_K_M.gguf"),
            "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf?download=true": (GGUF_REPO, "main", "Qwen3-8B-Q4_K_M.gguf"),
            "https://huggingface.co/unsloth/Qwen3-8B-GGUF/blob/main/Q8_0/Qwen3-8B-Q8_0-00001-of-00002.gguf": (GGUF_REPO, "main", "Q8_0/Qwen3-8B-Q8_0-00001-of-00002.gguf"),
        }
        for text, (repo, revision, file) in cases.items():
            with self.subTest(text=text):
                self.assertEqual(
                    parse_hf_source(text),
                    {"repo": repo, "revision": revision, "file": file},
                )

    def test_rejected_forms(self):
        rejected = [
            "", "   ", "Edge0", "a/b/c", "../etc/passwd", "org/..", "org/na me",
            "org/name\x00", "x" * 3000,
            "https://github.com/Edge0/Audio8-ASR-Infinite",
            "https://huggingface.co/datasets/org/name",
            "https://huggingface.co/spaces/org/app",
            "https://user:pw@huggingface.co/org/name",
            "https://huggingface.co:8443/org/name",
            "https://huggingface.co/org/name/tree/refs%2Fpr%2F1",
            "https://huggingface.co/org",
        ]
        for text in rejected:
            with self.subTest(text=text):
                with self.assertRaises(IntakeSourceError) as caught:
                    parse_hf_source(text)
                self.assertEqual(caught.exception.code, "invalid_source")
                self.assertIn("huggingface.co", caught.exception.remediation)

    def test_non_text_is_rejected(self):
        with self.assertRaises(IntakeSourceError):
            parse_hf_source(None)

    def test_validate_revision_and_file(self):
        self.assertEqual(validate_revision("v1.0"), "v1.0")
        self.assertEqual(validate_file("Q8_0/model.gguf"), "Q8_0/model.gguf")
        for bad in ("refs/pr/1", "", "../x"):
            with self.subTest(revision=bad):
                with self.assertRaises(IntakeSourceError):
                    validate_revision(bad)
        for bad in ("../model.gguf", "model.bin", "/abs/model.gguf", ""):
            with self.subTest(file=bad):
                with self.assertRaises(IntakeSourceError):
                    validate_file(bad)


if __name__ == "__main__":
    unittest.main()
