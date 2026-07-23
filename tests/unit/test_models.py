import http.client
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from mlx_agent.discovery import DiscoveryRequest, DiscoveryService
from mlx_agent.host import HostInventory
from mlx_agent.huggingface import (
    HF_RESPONSE_MAX_BYTES,
    HuggingFaceClient,
    http_json,
)
from mlx_agent.models import classify, classify_roles


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "scout_responses.json"


def fixture_http_get(url, timeout=10.0):
    del timeout
    payload = json.loads(FIXTURE.read_text())
    if "/tree/main" in url:
        repo = url.split("/api/models/", 1)[1].split("/tree/main", 1)[0]
        return payload["trees"].get(repo, [])
    if "/api/models/" in url:
        repo = url.split("/api/models/", 1)[1]
        return payload["details"].get(repo, {})
    return payload["models"]


class FakeHttpResponse:
    def __init__(self, body=b"", headers=None, status=200, chunks=None, on_read=None):
        self.body = body
        self.headers = headers or {}
        self.status = status
        self.chunks = list(chunks) if chunks is not None else None
        self.on_read = on_read
        self.read_calls = []
        self.offset = 0

    def getheader(self, name):
        return self.headers.get(name)

    def read(self, size):
        self.read_calls.append(size)
        if self.on_read is not None:
            self.on_read()
        if self.chunks is not None:
            return self.chunks.pop(0) if self.chunks else b""
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


class FakeSocket:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)


class FakeHttpsConnection:
    def __init__(self, response):
        self.response = response
        self.sock = FakeSocket()
        self.requests = []
        self.closed = False

    def request(self, method, target, headers=None):
        self.requests.append((method, target, headers))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class DiscoveryModelTests(unittest.TestCase):
    def setUp(self):
        host = HostInventory(**json.loads(FIXTURE.read_text())["host"])
        self.service = DiscoveryService(
            host=host,
            huggingface=HuggingFaceClient(http_get=fixture_http_get),
        )

    def test_quant_variants_keep_highest_ranked_candidate_per_logical_model(self):
        result = self.service.discover(DiscoveryRequest(limit=4)).to_dict()

        coding = result["data"]["roles"]["coding"]
        self.assertEqual([item["repo"] for item in coding], [
            "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-Q8"
        ])

    def test_template_capable_coder_appears_in_both_roles(self):
        repo = "community/Template-Coder-7B-4bit"

        def http_get(url, timeout=10.0):
            del timeout
            if "/api/models?" in url:
                return [{"id": repo, "downloads": 20, "likes": 2}]
            if "/tree/main" in url:
                return []
            return {
                "tags": ["mlx"],
                "config": {
                    "tokenizer_config": {
                        "chat_template": "{% if tools %}<tool_call>{{ tool_calls }}{% endif %}"
                    }
                },
            }

        service = DiscoveryService(
            host=HostInventory(ram_gb=32),
            huggingface=HuggingFaceClient(http_get=http_get),
        )
        result = service.discover(DiscoveryRequest(limit=4)).to_dict()

        self.assertIn(repo, [
            item["repo"] for item in result["data"]["roles"]["coding"]
        ])
        self.assertIn(repo, [
            item["repo"] for item in result["data"]["roles"]["tool-use"]
        ])

    def test_reasoning_detection_prefers_chat_template_over_tags_and_name(self):
        result = self.service.discover(DiscoveryRequest(role="general", limit=2)).to_dict()

        model = result["data"]["roles"]["general"][0]
        self.assertTrue(model["reasoning"])
        self.assertEqual(model["reason_src"], "chat_template")

    def test_classify_roles_preserves_primary_role_and_adds_tool_use(self):
        enrichment = {
            "tool_use": True,
            "tool_use_src": "chat_template",
            "tool_use_confidence": "explicit",
        }

        roles, signal = classify_roles(
            "community/Code-Coder-7B-4bit", enrichment
        )

        self.assertEqual(classify("community/Code-Coder-7B-4bit"), "coding")
        self.assertEqual(roles, ("coding", "tool-use"))
        self.assertEqual(signal, {
            "supported": True,
            "source": "chat_template",
            "confidence": "explicit",
        })

    def test_classify_roles_uses_narrow_weak_name_fallback(self):
        roles, signal = classify_roles(
            "community/Assistant-Tool_Call-7B-4bit", {}
        )

        self.assertEqual(roles, ("general", "tool-use"))
        self.assertEqual(signal, {
            "supported": True,
            "source": "name",
            "confidence": "weak",
        })
        self.assertEqual(classify_roles("community/Functionary-7B", {})[0], ("general",))
        self.assertEqual(classify_roles("community/FC-7B", {})[0], ("general",))

    def test_explicit_false_tool_use_beats_name_fallback(self):
        roles, signal = classify_roles(
            "community/Tool-Use-Assistant-7B",
            {"tool_use": False, "tool_use_src": "checked"},
        )

        self.assertEqual(roles, ("general",))
        self.assertEqual(signal, {
            "supported": False,
            "source": "checked",
            "confidence": "explicit",
        })

    def test_huggingface_tool_use_template_precedes_tags_and_name(self):
        def http_get(url, timeout=10.0):
            del timeout
            if "/tree/main" in url:
                return []
            return {
                "tags": ["tools"],
                "config": {
                    "tokenizer_config": {
                        "chat_template": "{% if tools %}<tool_call>{{ tool_calls }}{% endif %}"
                    }
                },
            }

        inspected = HuggingFaceClient(http_get=http_get).inspect_model(
            "community/Function-Calling-7B"
        )

        self.assertTrue(inspected["tool_use"])
        self.assertEqual(inspected["tool_use_src"], "chat_template")
        self.assertEqual(inspected["tool_use_confidence"], "explicit")

    def test_huggingface_tool_use_tags_precede_weak_name(self):
        def http_get(url, timeout=10.0):
            del timeout
            if "/tree/main" in url:
                return []
            return {
                "tags": ["function_calling"],
                "config": {"tokenizer_config": {"chat_template": ""}},
            }

        inspected = HuggingFaceClient(http_get=http_get).inspect_model(
            "community/ToolUse-7B"
        )

        self.assertTrue(inspected["tool_use"])
        self.assertEqual(inspected["tool_use_src"], "tags")
        self.assertEqual(inspected["tool_use_confidence"], "explicit")

    def test_metadata_only_inspection_never_fetches_repository_tree(self):
        requests = []

        def http_get(url, timeout=10.0):
            requests.append((url, timeout))
            return {
                "tags": ["function_calling"],
                "config": {"tokenizer_config": {"chat_template": ""}},
            }

        inspected = HuggingFaceClient(http_get=http_get).inspect_model_metadata(
            "community/Assistant-7B",
            timeout=2.5,
        )

        self.assertTrue(inspected["metadata_available"])
        self.assertTrue(inspected["tool_use"])
        self.assertFalse(inspected["tree_available"])
        self.assertEqual(len(requests), 1)
        self.assertNotIn("/tree/", requests[0][0])
        self.assertEqual(requests[0][1], 2.5)

    def test_huggingface_http_json_rejects_declared_oversize_before_reading(self):
        response = FakeHttpResponse(
            b'{"ignored": true}',
            headers={"Content-Length": str(HF_RESPONSE_MAX_BYTES + 1)},
        )
        connection = FakeHttpsConnection(response)

        with patch(
            "mlx_agent.huggingface.json.loads",
            side_effect=AssertionError("oversized response must not be parsed"),
        ):
            with self.assertRaises(ValueError):
                http_json(
                    "https://huggingface.co/api/models",
                    timeout=1.25,
                    connection_factory=lambda *args: connection,
                    clock=lambda: 0.0,
                )

        self.assertEqual(response.read_calls, [])
        self.assertTrue(connection.closed)

    def test_huggingface_http_json_bounds_stream_before_json_parsing(self):
        response = FakeHttpResponse(b"x" * (HF_RESPONSE_MAX_BYTES + 1))
        connection = FakeHttpsConnection(response)
        connection_args = []

        def connection_factory(host, port, timeout):
            connection_args.append((host, port, timeout))
            return connection

        with patch(
            "mlx_agent.huggingface.json.loads",
            side_effect=AssertionError("oversized response must not be parsed"),
        ):
            with self.assertRaises(ValueError):
                http_json(
                    "https://huggingface.co/api/models?filter=mlx",
                    timeout=1.25,
                    connection_factory=connection_factory,
                    clock=lambda: 0.0,
                )

        self.assertEqual(connection_args, [("huggingface.co", 443, 1.25)])
        self.assertEqual(
            connection.requests[0][0:2],
            ("GET", "/api/models?filter=mlx"),
        )
        self.assertGreater(len(response.read_calls), 1)
        self.assertTrue(connection.closed)

    def test_huggingface_http_json_enforces_one_monotonic_deadline(self):
        clock = FakeClock()
        response = FakeHttpResponse(
            chunks=[b'{"value":', b"1}", b""],
            on_read=lambda: clock.advance(0.6),
        )
        connection = FakeHttpsConnection(response)

        with self.assertRaises(TimeoutError):
            http_json(
                "https://huggingface.co/api/models",
                timeout=1.0,
                connection_factory=lambda *args: connection,
                clock=clock,
            )

        self.assertGreaterEqual(len(response.read_calls), 2)
        self.assertGreaterEqual(len(connection.sock.timeouts), 2)
        self.assertLess(
            connection.sock.timeouts[-1],
            connection.sock.timeouts[0],
        )
        self.assertTrue(connection.closed)

    def test_huggingface_deadline_covers_request_and_response_headers(self):
        for stage in ("request", "headers"):
            with self.subTest(stage=stage):
                clock = FakeClock()

                class AdvancingConnection(FakeHttpsConnection):
                    def request(self, method, target, headers=None):
                        super().request(method, target, headers=headers)
                        if stage == "request":
                            clock.advance(1.1)

                    def getresponse(self):
                        if stage == "headers":
                            clock.advance(1.1)
                        return super().getresponse()

                connection = AdvancingConnection(
                    FakeHttpResponse(body=b'{"value": 1}')
                )
                with self.assertRaises(TimeoutError):
                    http_json(
                        "https://huggingface.co/api/models",
                        timeout=1.0,
                        connection_factory=lambda *args: connection,
                        clock=clock,
                    )
                self.assertTrue(connection.closed)

    def test_huggingface_hard_deadline_abandons_blocked_daemon_worker(self):
        entered_headers = threading.Event()
        release_headers = threading.Event()
        late_result = threading.Event()

        class BlockingHeadersConnection(FakeHttpsConnection):
            def __init__(self):
                super().__init__(FakeHttpResponse(body=b'{"value": 1}'))
                self.close_calls = 0
                self.worker_was_daemon = None

            def getresponse(self):
                self.worker_was_daemon = threading.current_thread().daemon
                entered_headers.set()
                release_headers.wait()
                late_result.set()
                return super().getresponse()

            def close(self):
                self.close_calls += 1
                release_headers.set()
                super().close()

        connection = BlockingHeadersConnection()
        completion_events = []

        def force_timeout(completion, remaining):
            completion_events.append(completion)
            self.assertGreater(remaining, 0)
            self.assertTrue(entered_headers.wait(1.0))
            self.assertFalse(completion.is_set())
            return False

        with self.assertRaises(TimeoutError):
            http_json(
                "https://huggingface.co/api/models",
                timeout=1.0,
                connection_factory=lambda *args: connection,
                clock=lambda: 0.0,
                completion_wait=force_timeout,
            )

        self.assertGreaterEqual(connection.close_calls, 1)
        self.assertTrue(connection.worker_was_daemon)
        self.assertTrue(late_result.wait(1.0))
        self.assertTrue(completion_events[0].wait(1.0))

    def test_huggingface_http_json_rejects_redirects_errors_and_other_origins(self):
        invalid_urls = (
            "http://huggingface.co/api/models",
            "https://example.com/api/models",
            "https://huggingface.co:444/api/models",
            "https://user:pass@huggingface.co/api/models",
            "https://huggingface.co/api/models#fragment",
            "https://huggingface.co/api/modelsevil",
        )
        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    http_json(
                        url,
                        connection_factory=lambda *args: self.fail(
                            "invalid URL must not connect"
                        ),
                    )

        for status in (302, 500):
            with self.subTest(status=status):
                response = FakeHttpResponse(status=status)
                connection = FakeHttpsConnection(response)
                with self.assertRaises(http.client.HTTPException):
                    http_json(
                        "https://huggingface.co/api/models",
                        connection_factory=lambda *args: connection,
                    )
                self.assertEqual(response.read_calls, [])
                self.assertTrue(connection.closed)

    def test_weight_size_uses_actual_tree_bytes_when_available(self):
        result = self.service.discover(DiscoveryRequest(role="general", limit=2)).to_dict()

        model = result["data"]["roles"]["general"][0]
        self.assertEqual(model["est_ram_gb"], 4.2)
        self.assertEqual(model["ram_src"], "estimated_from_weight_bytes")
        self.assertEqual(model["facts"]["weight_bytes"], 4200000000)

    def test_discovery_data_retains_legacy_report_keys(self):
        result = self.service.discover(DiscoveryRequest(limit=2, fast=True)).to_dict()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result["data"]), {"host", "fast", "roles"})

    def test_list_provenance_records_the_exact_filtered_ranked_request_url(self):
        requested = []

        def capture_http_get(url, timeout=10.0):
            del timeout
            requested.append(url)
            return [{"id": "allowed/Open-Coder-7B-4bit", "downloads": 3, "likes": 2}]

        service = DiscoveryService(
            host=HostInventory(ram_gb=32, chip="Apple Test"),
            huggingface=HuggingFaceClient(http_get=capture_http_get),
        )
        candidate = service.discover(
            DiscoveryRequest(role="coding", fast=True, new=True, limit=1)
        ).to_dict()["data"]["roles"]["coding"][0]
        source = next(
            record for record in candidate["provenance"]
            if record["source"] == "huggingface_model_list"
        )
        self.assertEqual(requested[0], source["url"])
        self.assertEqual(
            {
                "filter": ["mlx"],
                "sort": ["lastModified"],
                "direction": ["-1"],
                "limit": ["300"],
            },
            parse_qs(urlsplit(source["url"]).query),
        )
