import http.client
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.port_analysis import PortAnalysisError
from mlx_agent.port_plan import build_prompt, draft_port_plan

from .backend_fixtures import synthetic_manifests, synthetic_registries
from .test_port_analysis import golden_toy, toy_client

FIXED_NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)


class PortPlanTests(unittest.TestCase):
    def setUp(self):
        self.manifests = synthetic_manifests()
        self.registries = synthetic_registries(self.manifests)
        self.calls = []

    def post(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        return {"choices": [{"message": {"content": "## Summary\nPort it."}}]}

    def draft(self, out_dir, endpoint="http://127.0.0.1:8090", context_tokens=32768):
        return draft_port_plan("org/toy-asr", endpoint, "qwen-local", out_dir=out_dir,
                               context_tokens=context_tokens, client=toy_client(), manifests=self.manifests,
                               registries=self.registries, post=self.post, now=lambda: FIXED_NOW)

    def test_writes_a_labeled_draft(self):
        with TemporaryDirectory() as directory:
            result = self.draft(directory)
            path = Path(result["path"])
            text = path.read_text(encoding="utf-8")
            self.assertEqual(path.name, "org--toy-asr-20261001T120000Z.md")
            self.assertTrue(text.startswith("# Port plan draft: org/toy-asr"))
            self.assertIn("Draft by qwen-local", text)
            self.assertIn(result["analysis_sha256"], text)
            self.assertIn("Port it.", text)
        url, payload, timeout = self.calls[0]
        self.assertEqual(url, "http://127.0.0.1:8090/v1/chat/completions")
        self.assertEqual(payload["model"], "qwen-local")
        self.assertEqual(payload["temperature"], 0)
        prompt = payload["messages"][-1]["content"]
        self.assertIn("mlx_lm.models.qwen2", prompt)
        self.assertIn("class ToyProjector", prompt)

    def test_refuses_non_loopback_endpoints(self):
        with TemporaryDirectory() as directory:
            for endpoint in ("http://10.0.0.5:8080", "https://api.example.com"):
                with self.subTest(endpoint=endpoint):
                    with self.assertRaises(PortAnalysisError) as caught:
                        self.draft(directory, endpoint=endpoint)
                    self.assertEqual(caught.exception.code, "endpoint_not_loopback")
        self.assertEqual(self.calls, [])

    def test_empty_draft_is_an_error(self):
        self.post = lambda url, payload, timeout: {"choices": [{"message": {"content": "  "}}]}
        with TemporaryDirectory() as directory:
            with self.assertRaises(PortAnalysisError) as caught:
                self.draft(directory)
        self.assertEqual(caught.exception.code, "draft_empty")

    def test_prompt_truncates_sources_but_keeps_analysis(self):
        payload = golden_toy()
        prompt, truncated = build_prompt(payload, {"modeling_toy_asr.py": "x = 1\n" * 5000}, budget_chars=4000)
        self.assertTrue(truncated)
        self.assertIn('"schema": "port-analysis/1"', prompt)

    def test_endpoint_transport_failures_are_classified(self):
        def failing(error):
            def post(url, payload, timeout):
                raise error
            return post

        for error in (http.client.HTTPException("local runtime returned HTTP status 500"), ConnectionRefusedError("refused")):
            with self.subTest(error=type(error).__name__):
                self.post = failing(error)
                with TemporaryDirectory() as directory:
                    with self.assertRaises(PortAnalysisError) as caught:
                        self.draft(directory)
                    self.assertEqual(list(Path(directory).iterdir()), [])
                self.assertEqual(caught.exception.code, "draft_failed")


if __name__ == "__main__":
    unittest.main()
