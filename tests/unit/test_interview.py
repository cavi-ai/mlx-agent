import unittest

from mlx_agent.interview import (
    DomainIntent,
    QUESTIONS,
    ROLE_CHOICES,
    build_intent,
    run_interview,
)


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
        self.assertEqual(first.keywords, ("ocr", "contracts", "redaction"))
        self.assertEqual(first.license_allow, ("apache-2.0", "mit"))
        self.assertEqual(first.memory_gb, 32.0)
        self.assertEqual(first.notes, "on-device only")

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

    def test_build_intent_rejects_non_numeric_memory(self):
        with self.assertRaises(ValueError):
            build_intent({"domain": "x", "roles": [], "memory_gb": "lots"})

    def test_all_role_choices_map_to_discovery_roles(self):
        from mlx_agent.models import DISCOVERY_ROLES

        for role in ROLE_CHOICES.values():
            self.assertIn(role, DISCOVERY_ROLES)


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
        self.assertEqual(intent.roles, ("vision",))
        self.assertEqual(intent.keywords, ("asr", "whisper"))
        self.assertIsNone(intent.memory_gb)

    def test_assist_output_is_revalidated_not_trusted(self):
        canned = {
            "domain": "Legal",
            "roles": "General chat",
            "keywords": "contracts",
            "license": "",
            "memory_gb": "",
            "notes": "",
        }

        def reader(question):
            return canned[question["id"]]

        def assist(intent):
            return {"keywords": intent.keywords + ("ocr",)}

        intent = run_interview(reader, assist=assist)
        self.assertEqual(intent.keywords, ("contracts", "ocr"))

    def test_assist_cannot_inject_invalid_role(self):
        canned = {
            "domain": "Legal",
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
            "domain": "Legal",
            "roles": "General chat",
            "keywords": "contracts",
            "license": "apache-2.0, mit",
            "memory_gb": "",
            "notes": "",
        }

        def reader(question):
            return canned[question["id"]]

        def assist(intent):
            return {"keywords": intent.keywords + ("ocr",)}

        intent = run_interview(reader, assist=assist)
        self.assertEqual(intent.license_allow, ("apache-2.0", "mit"))
        self.assertEqual(intent.keywords, ("contracts", "ocr"))


if __name__ == "__main__":
    unittest.main()
