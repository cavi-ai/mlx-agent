import http.client
import json
import struct
import unittest

from mlx_agent.huggingface import (
    SAFETENSORS_HEADER_MAX_BYTES,
    HuggingFaceClient,
    HuggingFaceHTTPError,
    http_safetensors_header,
)

from .test_huggingface_card import FakeCardSocket

URL = "https://huggingface.co/org/name/resolve/main/model.safetensors"
CDN = "https://us.aws.cdn.hf.co/xet-bridge-us/abc?Expires=1&Signature=s"


class RangeResponse:
    def __init__(self, status=206, body=b"", location=None):
        self.status = status
        self.body = body
        self.location = location
        self.headers = {}
        self.offset = 0

    def getheader(self, name):
        return self.location if name == "Location" else None

    def read(self, size):
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


class RangeConnection:
    def __init__(self, host, file_bytes, redirect_to=None):
        self.host = host
        self.file_bytes = file_bytes
        self.redirect_to = redirect_to
        self.sock = FakeCardSocket()
        self.requests = []
        self.closed = False

    def request(self, method, target, headers=None):
        self.requests.append((method, target, dict(headers or {})))

    def getresponse(self):
        method, target, headers = self.requests[-1]
        if self.redirect_to is not None:
            return RangeResponse(status=302, location=self.redirect_to)
        start, end = (int(value) for value in headers["Range"][len("bytes="):].split("-"))
        return RangeResponse(body=self.file_bytes[start:end + 1])

    def close(self):
        self.closed = True


def safetensors_bytes(header, padding=b"\0" * 64):
    encoded = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + padding


def scripted(redirects, file_bytes):
    """Connections by host: the Hub redirects per `redirects`, storage hosts serve ranges."""
    opened = []

    def factory(host, port, timeout):
        del port, timeout
        connection = RangeConnection(host, file_bytes, redirect_to=redirects.get(host))
        opened.append(connection)
        return connection

    return factory, opened


HEADER = {"__metadata__": {"format": "pt"}, "a.weight": {"dtype": "BF16", "shape": [64, 64], "data_offsets": [0, 8192]}}


class SafetensorsHeaderTests(unittest.TestCase):
    def test_follows_one_redirect_to_hub_storage_and_reads_only_the_header(self):
        factory, opened = scripted({"huggingface.co": CDN}, safetensors_bytes(HEADER))
        self.assertEqual(http_safetensors_header(URL, connection_factory=factory), HEADER)
        self.assertEqual([connection.host for connection in opened], ["huggingface.co", "us.aws.cdn.hf.co"])
        self.assertEqual(opened[0].requests[0][1], "/org/name/resolve/main/model.safetensors")
        self.assertEqual(opened[1].requests[0][1], "/xet-bridge-us/abc?Expires=1&Signature=s")
        self.assertTrue(all(request[2]["Range"].startswith("bytes=0-") for connection in opened for request in connection.requests))
        self.assertTrue(all(connection.closed for connection in opened))

    def test_large_header_takes_a_second_range(self):
        big = dict(HEADER, __metadata__={"pad": "x" * (1024 * 1024 + 10)})
        factory, opened = scripted({"huggingface.co": CDN}, safetensors_bytes(big))
        self.assertEqual(http_safetensors_header(URL, connection_factory=factory), big)
        self.assertEqual(len(opened), 3)
        self.assertTrue(opened[2].requests[0][2]["Range"].startswith("bytes=1048576-"))

    def test_redirects_off_the_hub_or_twice_are_refused(self):
        for redirects, error in (
            ({"huggingface.co": "https://evil.example/file"}, ValueError),
            ({"huggingface.co": "http://us.aws.cdn.hf.co/file"}, ValueError),
            ({"huggingface.co": "https://user:pw@us.aws.cdn.hf.co/file"}, ValueError),
            ({"huggingface.co": CDN, "us.aws.cdn.hf.co": "https://other.hf.co/file"}, http.client.HTTPException),
        ):
            with self.subTest(redirects=redirects):
                factory, _ = scripted(redirects, safetensors_bytes(HEADER))
                with self.assertRaises(error):
                    http_safetensors_header(URL, connection_factory=factory)

    def test_oversized_or_truncated_headers_are_refused(self):
        oversized = struct.pack("<Q", SAFETENSORS_HEADER_MAX_BYTES + 1) + b"{}"
        for file_bytes in (oversized, b"\x01\x02"):
            with self.subTest(size=len(file_bytes)):
                factory, _ = scripted({"huggingface.co": CDN}, file_bytes)
                with self.assertRaises(ValueError):
                    http_safetensors_header(URL, connection_factory=factory)

    def test_status_errors_carry_the_status(self):
        def factory(host, port, timeout):
            connection = RangeConnection(host, b"")
            connection.getresponse = lambda: RangeResponse(status=404)
            return connection

        with self.assertRaises(HuggingFaceHTTPError) as caught:
            http_safetensors_header(URL, connection_factory=factory)
        self.assertEqual(caught.exception.status, 404)

    def test_subfolder_weights_and_configs_are_hub_paths_but_traversal_is_not(self):
        from mlx_agent.huggingface import _is_valid_raw_path, _is_valid_resolve_path

        self.assertTrue(_is_valid_resolve_path("/org/name/resolve/main/multilingual/model.safetensors"))
        self.assertTrue(_is_valid_raw_path("/org/name/raw/main/a/b/config.json"))
        deep = "/org/name/resolve/main/" + "d/" * 9 + "model.safetensors"
        for path in ("/org/name/resolve/main/../model.safetensors", "/org/name/resolve/main/./model.safetensors",
                     "/org/name/resolve/main//model.safetensors", "/org/name/resolve/main/%2e%2e/model.safetensors", deep,
                     "/org/name/raw/main/sub/model.safetensors"):
            with self.subTest(path=path):
                self.assertFalse(_is_valid_resolve_path(path) and _is_valid_raw_path(path))
                self.assertFalse(_is_valid_resolve_path(path))

    def test_only_safetensors_on_the_hub_host(self):
        for url in (
            "https://huggingface.co/org/name/resolve/main/config.json",
            "https://huggingface.co/org/name/resolve/main/../model.safetensors",
            "https://huggingface.co/org/name/raw/main/model.safetensors",
            "https://huggingface.co/org/name/resolve/main/model.safetensors?download=1",
            "https://evil.example/org/name/resolve/main/model.safetensors",
            "http://huggingface.co/org/name/resolve/main/model.safetensors",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    http_safetensors_header(url)

    def test_client_builds_the_resolve_url(self):
        seen = []
        client = HuggingFaceClient(header_get=lambda url, timeout=None: seen.append(url) or {})
        client.fetch_safetensors_header("org/name", "abc123", "model-00001-of-00002.safetensors")
        self.assertEqual(seen, ["https://huggingface.co/org/name/resolve/abc123/model-00001-of-00002.safetensors"])


if __name__ == "__main__":
    unittest.main()
