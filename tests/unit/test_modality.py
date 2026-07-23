"""Unit tests for foundational modality detection and profile seeding."""

import unittest

from mlx_agent.modality import (
    ALL_FACET_IDS,
    FOUNDATION_IDS,
    MODALITY_CHOICES,
    apply_modality_profile,
    detect_facets,
    detect_modalities,
    resolve_facets,
    resolve_modalities,
    validate_facets,
    validate_modalities,
)


class DetectModalitiesTests(unittest.TestCase):
    def test_empty_text_returns_empty(self):
        self.assertEqual(detect_modalities(""), ())
        self.assertEqual(detect_modalities("   "), ())

    def test_detects_audio(self):
        self.assertEqual(detect_modalities("speech recognition with whisper"), ("audio",))

    def test_detects_video(self):
        self.assertEqual(detect_modalities("video captioning pipeline"), ("video",))

    def test_detects_document_vision(self):
        self.assertEqual(detect_modalities("PDF OCR and layout parsing"), ("document-vision",))

    def test_multi_hit(self):
        hits = detect_modalities("transcribe meeting video then OCR the slides")
        self.assertIn("audio", hits)
        self.assertIn("video", hits)
        self.assertIn("document-vision", hits)

    def test_no_false_positive_on_unrelated(self):
        self.assertEqual(detect_modalities("a python web API for billing"), ())


class DetectFacetsTests(unittest.TestCase):
    def test_audio_facets(self):
        self.assertEqual(detect_facets("audio", "whisper ASR for calls"), ("asr",))
        self.assertEqual(detect_facets("audio", "text to speech narrator"), ("tts",))
        self.assertEqual(detect_facets("audio", "music generation model"), ("music",))

    def test_video_facets(self):
        self.assertEqual(detect_facets("video", "video understanding caption"), ("understanding",))
        self.assertEqual(detect_facets("video", "video generation diffusion"), ("generation",))
        self.assertEqual(detect_facets("video", "action recognition tracking"), ("action",))

    def test_document_vision_facets(self):
        self.assertEqual(detect_facets("document-vision", "ocr scanned invoices"), ("ocr",))
        self.assertEqual(detect_facets("document-vision", "document layout analysis"), ("layout",))
        self.assertEqual(detect_facets("document-vision", "general vision VLM"), ("general-vision",))


class ResolveAndValidateTests(unittest.TestCase):
    def test_cli_modalities_win_over_detect(self):
        resolved = resolve_modalities(cli=("audio",), text="PDF OCR docs")
        self.assertEqual(resolved, ("audio",))

    def test_detect_when_cli_empty(self):
        resolved = resolve_modalities(cli=(), text="music generation")
        self.assertEqual(resolved, ("audio",))

    def test_validate_modalities_rejects_unknown(self):
        with self.assertRaises(ValueError):
            validate_modalities(("audio", "smell"))

    def test_validate_facets_rejects_unknown(self):
        with self.assertRaises(ValueError):
            validate_facets(("asr", "smell"))

    def test_resolve_facets_seeds_all_when_none_detected(self):
        facets = resolve_facets(("audio",), cli=(), text="audio app")
        self.assertEqual(set(facets), {"asr", "tts", "music"})

    def test_foundation_and_facet_ids_stable(self):
        self.assertEqual(FOUNDATION_IDS, ("audio", "video", "document-vision"))
        self.assertIn("asr", ALL_FACET_IDS)
        self.assertIn("understanding", ALL_FACET_IDS)
        self.assertIn("ocr", ALL_FACET_IDS)
        self.assertEqual(len(MODALITY_CHOICES), 3)


class ApplyProfileTests(unittest.TestCase):
    def test_unions_seeds_without_dropping_user_values(self):
        roles, keywords = apply_modality_profile(
            roles=("coding",),
            keywords=("custom",),
            modalities=("document-vision",),
            facets=("ocr",),
        )
        self.assertIn("coding", roles)
        self.assertIn("vision", roles)
        self.assertIn("custom", keywords)
        self.assertIn("ocr", keywords)

    def test_audio_seeds_general_role(self):
        roles, keywords = apply_modality_profile(
            roles=(),
            keywords=(),
            modalities=("audio",),
            facets=("asr",),
        )
        self.assertIn("general", roles)
        self.assertIn("whisper", keywords)

    def test_unknown_modality_rejected(self):
        with self.assertRaises(ValueError):
            apply_modality_profile((), (), ("nope",), ())


if __name__ == "__main__":
    unittest.main()
