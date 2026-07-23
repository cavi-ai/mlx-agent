import unittest

from mlx_agent.interview import (
    DomainIntent,
    QUESTIONS,
    ROLE_CHOICES,
    build_intent,
    run_interview,
)
from mlx_agent.models import DISCOVERY_ROLES


class BuildIntentTests(unittest.TestCase):
    def test_build_intent_is_deterministic_and_validated(self):
        answers = {
            "domain": "  Legal contract review  ",
            "roles": ["Vision / OCR", "General chat"],
            "keywords": "OCR, contracts,  redaction ,OCR",
            "license": "apache-2.0, MIT",
            "memory_gb": "32",
            "notes": "on-device only",
        }
        first = build_intent(answers)
        second = build_intent(answers)
        self.assertEqual(first, second)
        self.assertIsInstance(first, DomainIntent)
        self.assertEqual(first.domain, "Legal contract review")
        self.assertEqual(first.roles, ("vision", "general"))
        self.assertIn("ocr", first.keywords)
        self.assertIn("contracts", first.keywords)
        self.assertIn("redaction", first.keywords)
        self.assertEqual(first.license_allow, ("apache-2.0", "mit"))
        self.assertEqual(first.memory_gb, 32.0)
        self.assertEqual(first.notes, "on-device only")
        self.assertEqual(first.modalities, ("document-vision",))
        self.assertIn("ocr", first.facets)

    def test_build_intent_requires_a_domain(self):
        with self.assertRaises(ValueError):
            build_intent({"domain": "   ", "roles": ["General chat"]})

    def test_build_intent_rejects_unknown_role_label(self):
        with self.assertRaises(ValueError):
            build_intent({"domain": "x", "roles": ["not a real role"]})

    def test_build_intent_accepts_canonical_role_ids(self):
        intent = build_intent({"domain": "x", "roles": ["vision", "tool-use"]})
        self.assertEqual(intent.roles, ("vision", "tool-use"))

    def test_build_intent_defaults_role_to_general(self):
        intent = build_intent({"domain": "x", "roles": []})
        self.assertEqual(intent.roles, ("general",))
        self.assertEqual(intent.modalities, ())

    def test_build_intent_rejects_non_numeric_memory(self):
        with self.assertRaises(ValueError):
            build_intent({"domain": "x", "roles": [], "memory_gb": "lots"})

    def test_all_role_choices_map_to_discovery_roles(self):
        for role in ROLE_CHOICES.values():
            self.assertIn(role, DISCOVERY_ROLES)

    def test_build_intent_rejects_non_positive_or_nonfinite_memory(self):
        for bad in ("-8", "0", "nan", "inf"):
            with self.assertRaises(ValueError):
                build_intent({"domain": "x", "roles": [], "memory_gb": bad})

    def test_explicit_modalities_seed_roles_and_keywords(self):
        intent = build_intent({
            "domain": "billing helper",
            "roles": [],
            "keywords": "",
            "modalities": ["audio"],
            "facets": ["asr"],
        })
        self.assertEqual(intent.modalities, ("audio",))
        self.assertEqual(intent.facets, ("asr",))
        self.assertIn("general", intent.roles)
        self.assertIn("whisper", intent.keywords)

    def test_rejects_unknown_modality(self):
        with self.assertRaises(ValueError):
            build_intent({"domain": "x", "modalities": ["smell"]})


class RunInterviewTests(unittest.TestCase):
    def test_run_interview_uses_reader_for_each_question(self):
        canned = {
            "domain": "Audio transcription",
            "roles": "Vision / OCR",
            "keywords": "asr, whisper",
            "license": "",
            "memory_gb": "",
            "notes": "",
        }
        asked = []

        def reader(question):
            asked.append(question["id"])
            return canned[question["id"]]

        intent = run_interview(reader)
        self.assertEqual(asked, [question["id"] for question in QUESTIONS])
        self.assertEqual(intent.domain, "Audio transcription")
        self.assertEqual(intent.modalities, ("audio",))
        self.assertIn("asr", intent.facets)
        self.assertIn("vision", intent.roles)
        self.assertIn("general", intent.roles)
        self.assertIn("asr", intent.keywords)
        self.assertIn("whisper", intent.keywords)
        self.assertIsNone(intent.memory_gb)

    def test_interview_asks_modality_when_undetected(self):
        canned = {
            "domain": "billing helper",
            "modalities": "Audio (ASR / TTS / music)",
            "facets:audio": "ASR / speech-to-text",
            "roles": "General chat",
            "keywords": "",
            "license": "",
            "memory_gb": "",
            "notes": "",
        }
        asked = []

        def reader(question):
            asked.append(question["id"])
            return canned[question["id"]]

        intent = run_interview(reader)
        self.assertIn("modalities", asked)
        self.assertIn("facets:audio", asked)
        self.assertEqual(intent.modalities, ("audio",))
        self.assertEqual(intent.facets, ("asr",))

    def test_interview_skips_ask_when_preset_modalities(self):
        canned = {
            "domain": "billing helper",
            "roles": "",
            "keywords": "",
            "license": "",
            "memory_gb": "",
            "notes": "",
        }
        asked = []

        def reader(question):
            asked.append(question["id"])
            return canned[question["id"]]

        intent = run_interview(
            reader,
            preset_modalities=("video",),
            preset_facets=("understanding",),
        )
        self.assertNotIn("modalities", asked)
        self.assertTrue(all(not item.startswith("facets:") for item in asked))
        self.assertEqual(intent.modalities, ("video",))
        self.assertEqual(intent.facets, ("understanding",))

    def test_assist_output_is_revalidated_not_trusted(self):
        canned = {
            "domain": "Legal OCR contracts",
            "roles": "General chat",
            "keywords": "contracts",
            "license": "",
            "memory_gb": "",
            "notes": "",
        }

        def reader(question):
            return canned[question["id"]]

        def assist(intent):
            return {"keywords": intent.keywords + ("extra",)}

        intent = run_interview(reader, assist=assist)
        self.assertIn("contracts", intent.keywords)
        self.assertIn("extra", intent.keywords)
        self.assertEqual(intent.modalities, ("document-vision",))

    def test_assist_cannot_inject_invalid_role(self):
        canned = {
            "domain": "Legal OCR",
            "roles": "General chat",
            "keywords": "",
            "license": "",
            "memory_gb": "",
            "notes": "",
        }

        def reader(question):
            return canned[question["id"]]

        def assist(intent):
            return {"roles": ["totally-invalid-role"]}

        with self.assertRaises(ValueError):
            run_interview(reader, assist=assist)

    def test_assist_preserves_existing_license_filter(self):
        canned = {
            "domain": "Legal OCR contracts",
            "roles": "General chat",
            "keywords": "contracts",
            "license": "apache-2.0, mit",
            "memory_gb": "",
            "notes": "",
        }

        def reader(question):
            return canned[question["id"]]

        def assist(intent):
            return {"keywords": list(intent.keywords) + ["extra"]}

        intent = run_interview(reader, assist=assist)
        self.assertEqual(intent.license_allow, ("apache-2.0", "mit"))
        self.assertIn("contracts", intent.keywords)
        self.assertIn("extra", intent.keywords)


if __name__ == "__main__":
    unittest.main()
