"""Unit tests for MLX-native runtime preference (reporting only)."""

import unittest

from mlx_agent.runtime_preference import (
    RuntimePreference,
    prefer_runtime,
    wiring_for_preference,
)


def _host(ollama=False, lmstudio=False):
    return {"chip": "M3", "ram_gb": 32, "ollama": ollama, "lmstudio": lmstudio}


class PreferRuntimeTests(unittest.TestCase):
    def test_vision_prefers_mlx_vlm(self):
        pref = prefer_runtime(
            _host(ollama=True, lmstudio=True),
            roles=("vision",),
            modalities=("document-vision",),
        )
        self.assertEqual(pref.preferred, "mlx-vlm")
        self.assertNotIn("ollama", pref.alternates)
        self.assertIn("lmstudio", pref.alternates)
        self.assertIn("mlx-vlm", pref.rationale.lower())

    def test_video_modality_prefers_mlx_vlm(self):
        pref = prefer_runtime(_host(), roles=("general",), modalities=("video",))
        self.assertEqual(pref.preferred, "mlx-vlm")

    def test_text_prefers_lmstudio_when_up(self):
        pref = prefer_runtime(
            _host(ollama=True, lmstudio=True),
            roles=("general",),
        )
        self.assertEqual(pref.preferred, "lmstudio")
        self.assertIn("mlx_lm", pref.alternates)
        self.assertIn("ollama", pref.alternates)

    def test_text_prefers_mlx_lm_when_lmstudio_down(self):
        pref = prefer_runtime(
            _host(ollama=True, lmstudio=False),
            roles=("coding",),
        )
        self.assertEqual(pref.preferred, "mlx_lm")
        self.assertIn("ollama", pref.alternates)

    def test_audio_prefers_lmstudio_when_up_else_mlx_lm(self):
        with_lms = prefer_runtime(
            _host(lmstudio=True, ollama=True),
            roles=("general",),
            modalities=("audio",),
        )
        without = prefer_runtime(
            _host(lmstudio=False, ollama=True),
            roles=("general",),
            modalities=("audio",),
        )
        self.assertEqual(with_lms.preferred, "lmstudio")
        self.assertEqual(without.preferred, "mlx_lm")
        self.assertIn("ollama", without.alternates)
        self.assertIn("curated", without.rationale.lower())

    def test_ollama_in_alternates_when_up_for_text(self):
        pref = prefer_runtime(_host(ollama=True, lmstudio=False), roles=("general",))
        self.assertIn("ollama", pref.alternates)

    def test_host_snapshot_recorded(self):
        pref = prefer_runtime(_host(ollama=True, lmstudio=False), roles=("general",))
        self.assertEqual(pref.host_snapshot["ollama"], True)
        self.assertEqual(pref.host_snapshot["lmstudio"], False)
        self.assertIsInstance(pref, RuntimePreference)


class WiringForPreferenceTests(unittest.TestCase):
    def test_mlx_vlm_wiring(self):
        pref = RuntimePreference(
            preferred="mlx-vlm",
            alternates=("lmstudio",),
            rationale="vision",
            host_snapshot=_host(lmstudio=True),
        )
        text = wiring_for_preference("org/VLM-4bit", "vision", pref)
        self.assertIn("mlx-vlm", text)
        self.assertIn("mlxvlm/", text)

    def test_lmstudio_wiring(self):
        pref = RuntimePreference(
            preferred="lmstudio",
            alternates=("mlx_lm", "ollama"),
            rationale="native",
            host_snapshot=_host(lmstudio=True, ollama=True),
        )
        text = wiring_for_preference("org/Model-4bit", "general", pref)
        self.assertIn("LM Studio", text)
        self.assertIn("lmstudio/", text)

    def test_mlx_lm_wiring(self):
        pref = RuntimePreference(
            preferred="mlx_lm",
            alternates=("ollama",),
            rationale="native",
            host_snapshot=_host(ollama=True),
        )
        text = wiring_for_preference("org/Model-4bit", "general", pref)
        self.assertIn("mlx_lm.server", text)


class ModelsWiringAlignmentTests(unittest.TestCase):
    def test_wiring_matches_preference_helper(self):
        from mlx_agent.models import wiring

        host = _host(lmstudio=True, ollama=True)
        pref = prefer_runtime(host, roles=("general",))
        self.assertEqual(
            wiring("acme/Chat-4bit", "general", host),
            wiring_for_preference("acme/Chat-4bit", "general", pref),
        )

    def test_vision_wiring_unchanged_shape(self):
        from mlx_agent.models import wiring

        text = wiring("acme/OCR-4bit", "vision", _host(lmstudio=True))
        self.assertIn("mlx-vlm", text)


if __name__ == "__main__":
    unittest.main()
