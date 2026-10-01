import http.client
import unittest

from mlx_agent.huggingface import (
    RAW_TEXT_MAX_BYTES,
    HuggingFaceClient,
    HuggingFaceHTTPError,
    http_json,
    http_raw_text,
)

from .test_huggingface_card import FakeCardConnection, FakeCardResponse


def _factory(response):
    connection = FakeCardConnection(response)

    def factory(host, port, timeout):
        del host, port, timeout
        return connection

    return factory, connection


class HttpRawTextTests(unittest.TestCase):
    def test_reads_allowed_raw_files(self):
        for name in ("config.json", "model.safetensors.index.json", "modeling_toy.py"):
            with self.subTest(name=name):
                factory, connection = _factory(FakeCardResponse(body=b'{"model_type": "toy"}'))
                text = http_raw_text(
                    "https://huggingface.co/org/name/raw/main/{0}".format(name),
                    connection_factory=factory,
                )
                self.assertIn("toy", text)
                self.assertEqual(connection.requested[1], "/org/name/raw/main/{0}".format(name))

    def test_rejects_other_paths_and_hosts(self):
        for url in (
            "https://huggingface.co/org/name/raw/main/README.md",
            "https://huggingface.co/org/name/raw/main/sub/config.json",
            "https://huggingface.co/org/name/resolve/main/config.json",
            "https://huggingface.co/org/name/raw/refs%2Fpr%2F1/config.json",
            "https://huggingface.co/org/name/raw/main/model.safetensors",
            "https://evil.example/org/name/raw/main/config.json",
            "http://huggingface.co/org/name/raw/main/config.json",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    http_raw_text(url)

    def test_status_errors_carry_the_status(self):
        factory, _ = _factory(FakeCardResponse(status=401))
        with self.assertRaises(HuggingFaceHTTPError) as caught:
            http_raw_text("https://huggingface.co/org/name/raw/main/config.json", connection_factory=factory)
        self.assertEqual(caught.exception.status, 401)
        self.assertIsInstance(caught.exception, http.client.HTTPException)

    def test_declared_oversize_is_refused(self):
        response = FakeCardResponse(body=b"{}", headers={"Content-Length": str(RAW_TEXT_MAX_BYTES + 1)})
        factory, _ = _factory(response)
        with self.assertRaises(ValueError):
            http_raw_text("https://huggingface.co/org/name/raw/main/config.json", connection_factory=factory)


class HttpJsonStatusTests(unittest.TestCase):
    def test_json_status_error_is_typed_and_backward_compatible(self):
        factory, _ = _factory(FakeCardResponse(status=404))
        with self.assertRaises(HuggingFaceHTTPError) as caught:
            http_json("https://huggingface.co/api/models/org/name", connection_factory=factory)
        self.assertEqual(caught.exception.status, 404)
        self.assertIsInstance(caught.exception, http.client.HTTPException)


class ClientUrlTests(unittest.TestCase):
    def test_model_info_and_raw_urls(self):
        seen = []

        def http_get(url, timeout=None):
            seen.append(url)
            return {"id": "org/name"}

        def raw_get(url, timeout=None):
            seen.append(url)
            return "{}"

        client = HuggingFaceClient(http_get=http_get, raw_get=raw_get)
        client.fetch_model_info("org/name")
        client.fetch_model_info("org/name", revision="v1.0")
        client.fetch_raw_text("org/name", "main", "config.json")
        self.assertEqual(seen, [
            "https://huggingface.co/api/models/org/name?blobs=true",
            "https://huggingface.co/api/models/org/name/revision/v1.0?blobs=true",
            "https://huggingface.co/org/name/raw/main/config.json",
        ])


if __name__ == "__main__":
    unittest.main()
