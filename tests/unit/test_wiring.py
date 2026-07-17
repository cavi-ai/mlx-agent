import json
import tempfile
import unittest
from pathlib import Path

from mlx_agent.wiring import ConfigAdapter, redact_secrets


class WireRenderingTests(unittest.TestCase):
    def test_each_runtime_renders_deterministic_valid_configuration(self):
        cases = {
            "ollama": ("Modelfile", "# existing\n"),
            "lmstudio": ("settings.json", '{"models": []}\n'),
            "mlx_lm": ("providers.json", '{"providers": []}\n'),
            "mlx-vlm": ("providers.json", '{"providers": []}\n'),
            "litellm": ("config.yaml", "model_list:\n"),
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
        content = '{"api_key":"super-secret", "token": "token-value", "authorization": "Bearer xyz", "safe": "ok"}\nendpoint?secret=query-secret&api_key=url-secret'
        redacted = redact_secrets(content)
        self.assertNotIn("super-secret", redacted)
        self.assertNotIn("token-value", redacted)
        self.assertNotIn("Bearer xyz", redacted)
        self.assertNotIn("query-secret", redacted)
        self.assertNotIn("url-secret", redacted)
        self.assertIn('"safe": "ok"', redacted)


if __name__ == "__main__":
    unittest.main()
