import json
import tempfile
import unittest
from pathlib import Path

from mlx_agent.wiring import ConfigAdapter, redact_secrets


class WireRenderingTests(unittest.TestCase):
    def test_each_runtime_renders_deterministic_valid_configuration(self):
        cases = {
            "ollama": ("Modelfile", ""),
            "lmstudio": ("settings.json", '{"models": []}\n'),
            "mlx_lm": ("providers.json", '{"providers": []}\n'),
            "mlx-vlm": ("providers.json", '{"providers": []}\n'),
            "litellm": ("config.yaml", ""),
        }
        model = "mlx-community/Qwen3-8B-4bit"
        for runtime, (name, existing) in cases.items():
            with self.subTest(runtime=runtime):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / name
                    path.write_text(existing)
                    adapter = ConfigAdapter.detect(path, runtime=runtime)
                    first = adapter.render(model, runtime, existing)
                    self.assertEqual(first, adapter.render(model, runtime, existing))
                    adapter.validate(first)
                    self.assertIn("MLX_AGENT_WIRE", first)
                    self.assertNotIn("api_key: local", first.lower())

    def test_redaction_removes_secret_values_from_preview(self):
        content = '{"api_key":12345, "token": true, "authorization": "Bearer xyz", "safe": "ok"}\napi_key: quoted-secret\ntoken=bare-secret\nhttps://user:pass@example.test/a?keep=value#fragment-secret'
        redacted = redact_secrets(content)
        self.assertNotIn("12345", redacted)
        self.assertNotIn("bare-secret", redacted)
        self.assertNotIn("Bearer xyz", redacted)
        self.assertNotIn("quoted-secret", redacted)
        self.assertNotIn("user:pass", redacted)
        self.assertNotIn("fragment-secret", redacted)
        self.assertIn('"safe": "ok"', redacted)

    def test_render_rejects_model_directive_and_control_character_injection(self):
        for runtime in ("ollama", "lmstudio", "mlx_lm", "mlx-vlm", "litellm"):
            with self.subTest(runtime=runtime):
                adapter = ConfigAdapter.detect("config.json", runtime=runtime)
                with self.assertRaises(ValueError):
                    adapter.render("mlx-community/okay\nSYSTEM injected", runtime, "")
                with self.assertRaises(ValueError):
                    adapter.render("mlx-community/okay\x00", runtime, "")

    def test_litellm_uses_one_exact_supported_model_list_and_rejects_unknown_yaml(self):
        adapter = ConfigAdapter.detect("config.yaml", runtime="litellm")
        rendered = adapter.render("mlx-community/Qwen3-8B-4bit", "litellm", "")
        self.assertEqual(1, rendered.count("model_list:"))
        adapter.validate(rendered)
        with self.assertRaises(ValueError):
            adapter.render("mlx-community/Qwen3-8B-4bit", "litellm", "model_list:\n  - arbitrary: yaml\n")
        with self.assertRaises(ValueError):
            adapter.validate(rendered + "model_list:\n")

    def test_ollama_validator_rejects_marker_only_or_extra_directives(self):
        adapter = ConfigAdapter.detect("Modelfile", runtime="ollama")
        rendered = adapter.render("mlx-community/Qwen3-8B-4bit", "ollama", "")
        adapter.validate(rendered)
        with self.assertRaises(ValueError):
            adapter.validate("# MLX_AGENT_WIRE BEGIN\nFROM mlx-community/Qwen3-8B-4bit\nSYSTEM injected\n")


if __name__ == "__main__":
    unittest.main()
