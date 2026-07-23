import unittest

from mlx_agent.huggingface import (
    MODEL_CARD_MAX_BYTES,
    HuggingFaceClient,
    http_card_text,
)


class FakeCardResponse:
    def __init__(self, body=b"", headers=None, status=200):
        self.body = body
        self.headers = headers or {}
        self.status = status
        self.offset = 0

    def read(self, size):
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


class FakeCardSocket:
    def settimeout(self, timeout):
        del timeout


class FakeCardConnection:
    def __init__(self, response):
        self._response = response
        self.sock = FakeCardSocket()
        self.requested = None
        self.closed = False

    def request(self, method, target, headers=None):
        self.requested = (method, target, headers)

    def getresponse(self):
        return self._response

    def close(self):
        self.closed = True


class HttpCardTextTests(unittest.TestCase):
    def _factory(self, response):
        connection = FakeCardConnection(response)

        def factory(host, port, timeout):
            del host, port, timeout
            return connection

        return factory, connection

    def test_reads_bounded_card_text(self):
        response = FakeCardResponse(body=b"# Model\nGood OCR model.\n")
        factory, connection = self._factory(response)
        text = http_card_text(
            "https://huggingface.co/acme/ocr-model/raw/main/README.md",
            connection_factory=factory,
        )
        self.assertIn("Good OCR model", text)
        self.assertEqual(connection.requested[0], "GET")
        self.assertTrue(connection.closed)

    def test_rejects_non_huggingface_host(self):
        with self.assertRaises(ValueError):
            http_card_text("https://evil.example/acme/x/raw/main/README.md")

    def test_rejects_non_readme_path(self):
        with self.assertRaises(ValueError):
            http_card_text("https://huggingface.co/acme/x/resolve/main/model.bin")

    def test_rejects_redirect(self):
        response = FakeCardResponse(status=302, headers={"Location": "https://evil"})
        factory, _ = self._factory(response)
        with self.assertRaises(Exception):
            http_card_text(
                "https://huggingface.co/acme/x/raw/main/README.md",
                connection_factory=factory,
            )

    def test_rejects_oversized_declared_length(self):
        response = FakeCardResponse(
            headers={"Content-Length": str(MODEL_CARD_MAX_BYTES + 1)}
        )
        factory, _ = self._factory(response)
        with self.assertRaises(ValueError):
            http_card_text(
                "https://huggingface.co/acme/x/raw/main/README.md",
                connection_factory=factory,
            )


class FetchModelCardTests(unittest.TestCase):
    def test_fetch_returns_text_via_injected_getter(self):
        calls = []

        def card_get(url, timeout=8):
            calls.append((url, timeout))
            return "# Card\nDetails."

        client = HuggingFaceClient(card_get=card_get)
        card = client.fetch_model_card("acme/ocr-model")
        self.assertEqual(card, "# Card\nDetails.")
        self.assertEqual(
            calls[0][0],
            "https://huggingface.co/acme/ocr-model/raw/main/README.md",
        )

    def test_fetch_returns_none_on_failure(self):
        def card_get(url, timeout=8):
            raise OSError("boom")

        client = HuggingFaceClient(card_get=card_get)
        self.assertIsNone(client.fetch_model_card("acme/ocr-model"))


if __name__ == "__main__":
    unittest.main()
