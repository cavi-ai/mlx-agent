import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from mlx_agent.wiring import (
    ConfigAdapter,
    _SameLoopbackOriginRedirect,
    redact_secrets,
    validate_health_endpoint,
)


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

    def test_render_fails_closed_on_resolved_secrets_but_allows_environment_references(self):
        adapter = ConfigAdapter.detect("providers.json", runtime="mlx_lm")
        for existing in (
            '{"api_key": "resolved-value", "providers": []}\n',
            '{"nested": {"authorization": "Bearer resolved-value"}}\n',
            '{"base_url": "http://user:resolved-value@127.0.0.1:8080/v1"}\n',
            '{"client_secret": "resolved-value", "providers": []}\n',
            '{"api_token": "resolved-value", "providers": []}\n',
            '{"session_token": "resolved-value", "providers": []}\n',
            '{"x-api-key": "resolved-value", "providers": []}\n',
            '{"aws_secret_access_key": "resolved-value", "providers": []}\n',
            '{"auth": "Bearer resolved-value", "providers": []}\n',
            '{"base_url": "http://127.0.0.1:8080/v1?access_token=resolved-value"}\n',
            '{"command": "curl -H Authorization:resolved-value http://127.0.0.1"}\n',
        ):
            with self.subTest(existing=existing):
                with self.assertRaisesRegex(ValueError, "resolved secret-bearing fields") as captured:
                    adapter.render("mlx-community/Qwen3-8B-4bit", "mlx_lm", existing)
                self.assertNotIn("resolved-value", str(captured.exception))

        safe = adapter.render(
            "mlx-community/Qwen3-8B-4bit",
            "mlx_lm",
            '{"api_key": "os.environ/LOCAL_API_KEY", "providers": []}\n',
        )
        self.assertIn("os.environ/LOCAL_API_KEY", safe)

    def test_health_endpoints_are_credential_free_loopback_reads_only(self):
        accepted = (
            "http://127.0.0.1:8080/health",
            "https://localhost:8443/v1/models",
            "http://[::1]:11434/api/tags",
        )
        for endpoint in accepted:
            with self.subTest(endpoint=endpoint):
                self.assertEqual(endpoint, validate_health_endpoint(endpoint))

        rejected = (
            "http://169.254.169.254/latest/meta-data",
            "http://127.0.0.1:8080/delete",
            "http://user:pass@127.0.0.1:8080/health",
            "http://127.0.0.1:8080/health?mutate=true",
            "http://127.0.0.1:8080/health#fragment",
            "http://127.0.0.1:70000/health",
        )
        for endpoint in rejected:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ValueError):
                    validate_health_endpoint(endpoint)

    def test_health_check_rejects_cross_origin_redirects(self):
        adapter = ConfigAdapter("mlx_lm")

        class RedirectingOpener:
            def open(self, request, timeout):
                del request, timeout
                raise AssertionError("unsafe opener should not be reached")

        with patch("mlx_agent.wiring.urllib.request.build_opener", return_value=RedirectingOpener()):
            self.assertFalse(adapter.health_check("http://example.test/health"))

        handler = _SameLoopbackOriginRedirect(("http", "127.0.0.1", 8080))
        request = urllib.request.Request("http://127.0.0.1:8080/health")
        with self.assertRaises(urllib.error.URLError):
            handler.redirect_request(
                request, None, 302, "Found", {}, "http://127.0.0.1:8081/health"
            )
        with self.assertRaises(ValueError):
            handler.redirect_request(
                request, None, 302, "Found", {}, "http://127.0.0.1:8080/restart"
            )

    def test_health_check_disables_environment_proxy_routing(self):
        adapter = ConfigAdapter("mlx_lm")

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_arguments):
                return False

        class Opener:
            def open(self, request, timeout):
                del request, timeout
                return Response()

        with patch(
            "mlx_agent.wiring.urllib.request.build_opener", return_value=Opener()
        ) as build_opener:
            self.assertTrue(adapter.health_check("http://127.0.0.1:8080/health"))
        proxy_handlers = [
            item for item in build_opener.call_args.args
            if isinstance(item, urllib.request.ProxyHandler)
        ]
        self.assertEqual(1, len(proxy_handlers))
        self.assertEqual({}, proxy_handlers[0].proxies)

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
