"""Tests for Hugging Face datasets API and dataset card transport."""

import unittest
import urllib.parse

from mlx_agent.huggingface import (
    HF_DATASETS_API,
    MODEL_CARD_MAX_BYTES,
    HuggingFaceClient,
    http_card_text,
    http_json,
)


class FakeResponse:
    def __init__(self, body=b"{}", headers=None, status=200):
        self.body = body
        self.headers = headers or {"Content-Length": str(len(body))}
        self.status = status
        self.offset = 0

    def read(self, size):
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


class FakeSocket:
    def settimeout(self, timeout):
        del timeout


class FakeConnection:
    def __init__(self, response):
        self._response = response
        self.sock = FakeSocket()
        self.requested = None
        self.closed = False

    def request(self, method, target, headers=None):
        self.requested = (method, target, headers)

    def getresponse(self):
        return self._response

    def close(self):
        self.closed = True


class HttpDatasetsApiTests(unittest.TestCase):
    def _factory(self, response):
        connection = FakeConnection(response)

        def factory(host, port, timeout):
            del host, port, timeout
            return connection

        return factory, connection

    def test_accepts_datasets_api_path(self):
        body = b'[{"id": "acme/legal-data"}]'
        response = FakeResponse(body=body)
        factory, connection = self._factory(response)
        payload = http_json(
            "https://huggingface.co/api/datasets?search=legal&limit=5",
            connection_factory=factory,
        )
        self.assertEqual(payload[0]["id"], "acme/legal-data")
        self.assertTrue(connection.requested[1].startswith("/api/datasets"))

    def test_rejects_non_allowlisted_api_path(self):
        with self.assertRaises(ValueError):
            http_json("https://huggingface.co/api/spaces")


class DatasetCardTests(unittest.TestCase):
    def _factory(self, response):
        connection = FakeConnection(response)

        def factory(host, port, timeout):
            del host, port, timeout
            return connection

        return factory, connection

    def test_reads_dataset_card_path(self):
        response = FakeResponse(body=b"# Dataset\nContracts OCR labels.\n")
        factory, _ = self._factory(response)
        text = http_card_text(
            "https://huggingface.co/datasets/acme/legal/raw/main/README.md",
            connection_factory=factory,
        )
        self.assertIn("Contracts OCR", text)

    def test_rejects_malformed_dataset_card_path(self):
        with self.assertRaises(ValueError):
            http_card_text(
                "https://huggingface.co/datasets/acme/legal/resolve/main/data.parquet"
            )


class CatalogListHelpersTests(unittest.TestCase):
    def test_list_adapters_url_includes_peft_filter_and_search(self):
        url = HuggingFaceClient.list_adapters_url(search="ocr", limit_fetch=10)
        parsed = urllib.parse.urlsplit(url)
        self.assertEqual(parsed.path, "/api/models")
        query = urllib.parse.parse_qs(parsed.query)
        self.assertEqual(query["filter"], ["peft"])
        self.assertEqual(query["search"], ["ocr"])
        self.assertEqual(query["limit"], ["10"])

    def test_list_datasets_url_targets_datasets_api(self):
        url = HuggingFaceClient.list_datasets_url(search="contracts", limit_fetch=7)
        self.assertTrue(url.startswith(HF_DATASETS_API))
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        self.assertEqual(query["search"], ["contracts"])
        self.assertEqual(query["limit"], ["7"])

    def test_fetch_dataset_card_uses_datasets_prefix(self):
        calls = []

        def card_get(url, timeout=8):
            calls.append(url)
            return "# Dataset card"

        client = HuggingFaceClient(card_get=card_get)
        text = client.fetch_dataset_card("acme/legal")
        self.assertEqual(text, "# Dataset card")
        self.assertEqual(
            calls[0],
            "https://huggingface.co/datasets/acme/legal/raw/main/README.md",
        )

    def test_fetch_dataset_card_returns_none_on_failure(self):
        def card_get(url, timeout=8):
            raise OSError("boom")

        client = HuggingFaceClient(card_get=card_get)
        self.assertIsNone(client.fetch_dataset_card("acme/legal"))

    def test_list_adapters_and_datasets_call_http_get(self):
        calls = []

        def http_get(url, timeout=10):
            calls.append(url)
            return [{"id": "acme/x", "downloads": 1, "tags": ["peft"]}]

        client = HuggingFaceClient(http_get=http_get)
        adapters = client.list_adapters(search="ocr", limit_fetch=3)
        datasets = client.list_datasets(search="legal", limit_fetch=3)
        self.assertEqual(len(adapters), 1)
        self.assertEqual(len(datasets), 1)
        self.assertIn("filter=peft", calls[0])
        self.assertIn("/api/datasets", calls[1])


if __name__ == "__main__":
    unittest.main()
