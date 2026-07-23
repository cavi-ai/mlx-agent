import http.client
import json
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from mlx_agent.verification import (
    EvidenceStrength,
    INVENTORY_RESPONSE_MAX_BYTES,
    LMStudioRuntimeClient,
    OllamaRuntimeClient,
    OpenAICompatibleRuntimeClient,
    PROBE_RESPONSE_MAX_BYTES,
    TOOL_USE_ARGUMENTS,
    TOOL_USE_PROBE_ID,
    TOOL_USE_PROMPT,
    TOOL_USE_TOOL_NAME,
    TOOL_USE_TOOLS,
    VerificationEvidence,
    VerificationStatus,
    Verifier,
    _http_json_get,
    _http_json_request,
    normalize_tool_call,
)

ROOT = Path(__file__).resolve().parents[2]
TOOL_USE_FIXTURE = ROOT / "tests" / "fixtures" / "tool_use_responses.json"


class FakeRuntimeClient:
    name = "fake-runtime"

    def __init__(
        self,
        installed,
        responses=None,
        failures=None,
        inventory_failure=None,
    ):
        self.installed = list(installed)
        self.responses = responses or {}
        self.failures = failures or {}
        self.inventory_failure = inventory_failure
        self.inventory_calls = 0
        self._inventory_lock = threading.Lock()
        self.generated = []
        self.downloads = []

    def list_models(self):
        with self._inventory_lock:
            self.inventory_calls += 1
        if self.inventory_failure is not None:
            raise RuntimeError(self.inventory_failure)
        return list(self.installed)

    def generate(self, model, prompt, max_tokens):
        self.generated.append((model, prompt, max_tokens))
        if model in self.failures:
            raise RuntimeError(self.failures[model])
        return self.responses.get(model, {"message": {"content": "ready"}})

    def download(self, model):
        self.downloads.append(model)
        raise AssertionError("verification must never download a model")


class FakeToolRuntimeClient(FakeRuntimeClient):
    def __init__(self, installed, probe_response=None, probe_error=None, name="fake-runtime"):
        super().__init__(installed)
        self.name = name
        self.probe_response = probe_response
        self.probe_error = probe_error
        self.probed = []

    def probe_tool_use(self, model):
        self.probed.append(model)
        if self.probe_error is not None:
            raise self.probe_error
        return self.probe_response


class FakeMetadataClient:
    def __init__(self, records=None, failures=None):
        self.records = records or {}
        self.failures = failures or {}
        self.inspected = []

    def inspect_model(self, repo):
        self.inspected.append(repo)
        if repo in self.failures:
            raise RuntimeError(self.failures[repo])
        return self.records.get(repo, {"metadata_available": True, "tags": []})


class FakeSocket:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)


class FakeHTTPResponse:
    def __init__(self, status=200, chunks=None, headers=None, on_read=None):
        self.status = status
        self._chunks = list(chunks or [])
        self._headers = headers or {}
        self._on_read = on_read
        self.read_calls = 0

    def getheader(self, name):
        return self._headers.get(name)

    def read(self, size):
        del size
        self.read_calls += 1
        if self._on_read is not None:
            self._on_read()
        return self._chunks.pop(0) if self._chunks else b""


class FakeHTTPConnection:
    def __init__(self, response):
        self.response = response
        self.sock = FakeSocket()
        self.requests = []
        self.closed = False

    def request(self, method, path, body=None, headers=None):
        self.requests.append((method, path, body, headers))

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


class VerificationTests(unittest.TestCase):
    def test_tool_use_probe_contract_is_fixed_and_synthetic(self):
        self.assertEqual(TOOL_USE_PROBE_ID, "tool-use-v1")
        self.assertEqual(TOOL_USE_TOOL_NAME, "lookup_widget")
        self.assertEqual(TOOL_USE_ARGUMENTS, {"widget_id": "widget-42"})
        self.assertEqual(
            TOOL_USE_PROMPT,
            (
                "Call lookup_widget exactly once for widget-42. "
                "Do not answer directly and do not call any other tool."
            ),
        )
        self.assertEqual(
            TOOL_USE_TOOLS,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_widget",
                        "description": "Look up one synthetic widget by its identifier.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "widget_id": {
                                    "type": "string",
                                    "description": "Synthetic widget identifier.",
                                }
                            },
                            "required": ["widget_id"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        )

    def test_normalize_tool_call_fixture_cases(self):
        fixture = json.loads(TOOL_USE_FIXTURE.read_text(encoding="utf-8"))

        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(normalize_tool_call(case["response"]), case["expected"])

    def test_normalize_tool_call_handles_malformed_shapes_without_raising(self):
        invalid_response_cases = [
            None,
            "not an object",
            [],
            {"message": []},
            {"choices": "not a list"},
            {"choices": [None]},
            {"choices": [{"message": []}]},
            {"message": {"tool_calls": "not a list"}},
            {"message": {"tool_calls": [None]}},
            {"message": {"tool_calls": [{"function": []}]}},
            {"message": {"tool_calls": [{"function": {"name": ["lookup_widget"]}}]}},
        ]
        expected = {
            "valid": False,
            "tool_name": None,
            "arguments": None,
            "reason": "invalid_response",
        }

        for response in invalid_response_cases:
            with self.subTest(response=response):
                self.assertEqual(normalize_tool_call(response), expected)

        self.assertEqual(
            normalize_tool_call(
                {"message": {"tool_calls": [{"function": {"name": "lookup_widget"}}]}}
            ),
            {
                "valid": False,
                "tool_name": None,
                "arguments": None,
                "reason": "malformed_arguments",
            },
        )

    def test_normalize_tool_call_does_not_retain_failure_evidence(self):
        unexpected_name = "x" * 100
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": unexpected_name,
                            "arguments": {
                                "widget_id": "widget-42",
                                "secret": "must-not-survive",
                            },
                        }
                    }
                ]
            }
        }

        normalized = normalize_tool_call(response)

        self.assertEqual(
            normalized,
            {
                "valid": False,
                "tool_name": None,
                "arguments": None,
                "reason": "wrong_tool",
            },
        )
        self.assertNotIn(unexpected_name, repr(normalized))
        self.assertNotIn("must-not-survive", repr(normalized))

    def test_normalize_tool_call_rejects_oversized_arguments_without_parsing(self):
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup_widget",
                            "arguments": "{" + (" " * 512) + "}",
                        }
                    }
                ]
            }
        }

        with patch(
            "mlx_agent.verification.json.loads",
            side_effect=AssertionError("oversized arguments must not be parsed"),
        ):
            normalized = normalize_tool_call(response)

        self.assertEqual(
            normalized,
            {
                "valid": False,
                "tool_name": None,
                "arguments": None,
                "reason": "malformed_arguments",
            },
        )

    def test_normalize_tool_call_handles_pathological_json_without_raising(self):
        deeply_nested = ("[" * 240) + "0" + ("]" * 240)
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup_widget",
                            "arguments": deeply_nested,
                        }
                    }
                ]
            }
        }
        self.assertEqual(
            normalize_tool_call(response),
            {
                "valid": False,
                "tool_name": None,
                "arguments": None,
                "reason": "malformed_arguments",
            },
        )

        with patch(
            "mlx_agent.verification.json.loads",
            side_effect=RecursionError("pathological nesting"),
        ):
            normalized = normalize_tool_call(
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "lookup_widget",
                                    "arguments": '{"widget_id":"widget-42"}',
                                }
                            }
                        ]
                    }
                }
            )
        self.assertEqual(
            normalized,
            {
                "valid": False,
                "tool_name": None,
                "arguments": None,
                "reason": "malformed_arguments",
            },
        )

    def test_tool_use_expected_arguments_cannot_be_mutated(self):
        with self.assertRaises(TypeError):
            TOOL_USE_ARGUMENTS["widget_id"] = "attacker-controlled"

        normalized = normalize_tool_call(
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup_widget",
                                "arguments": {"widget_id": "widget-42"},
                            }
                        }
                    ]
                }
            }
        )
        self.assertEqual(
            normalized,
            {
                "valid": True,
                "tool_name": "lookup_widget",
                "arguments": {"widget_id": "widget-42"},
                "reason": "valid",
            },
        )

    def test_builtin_runtime_clients_use_only_inventory_and_generation_endpoints(self):
        gets = []
        posts = []

        def http_get(url, timeout=3.0):
            gets.append((url, timeout))
            if url.endswith("/api/tags"):
                return {"models": [{"name": "local/ollama-model"}]}
            return {"data": [{"id": "local/lmstudio-model"}]}

        def http_post(url, payload, timeout=10.0):
            posts.append((url, payload, timeout))
            return {"message": {"content": "ready"}}

        ollama = OllamaRuntimeClient(http_get=http_get, http_post=http_post)
        lmstudio = LMStudioRuntimeClient(http_get=http_get, http_post=http_post)

        self.assertEqual(ollama.list_models()["models"][0]["name"], "local/ollama-model")
        self.assertEqual(lmstudio.list_models()["data"][0]["id"], "local/lmstudio-model")
        ollama.generate("local/ollama-model", "ready?", max_tokens=12)
        lmstudio.generate("local/lmstudio-model", "ready?", max_tokens=12)

        called_urls = [item[0] for item in gets + posts]
        self.assertEqual(called_urls, [
            "http://127.0.0.1:11434/api/tags",
            "http://127.0.0.1:1234/v1/models",
            "http://127.0.0.1:11434/api/chat",
            "http://127.0.0.1:1234/v1/chat/completions",
        ])
        self.assertFalse(any("pull" in url or "download" in url for url in called_urls))
        self.assertEqual(posts[0][1]["options"]["num_predict"], 12)
        self.assertEqual(posts[1][1]["max_tokens"], 12)

    def test_default_transport_ignores_proxy_environment(self):
        response = FakeHTTPResponse(chunks=[b'{"models": []}', b""])
        connection = FakeHTTPConnection(response)

        with patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://example.com:9999",
                "HTTPS_PROXY": "http://example.com:9999",
                "NO_PROXY": "",
            },
            clear=False,
        ), patch(
            "mlx_agent.verification.http.client.HTTPConnection",
            return_value=connection,
        ) as connection_type:
            result = _http_json_get("http://127.0.0.1:11434/api/tags")

        self.assertEqual(result, {"models": []})
        connection_type.assert_called_once()
        connection_args, connection_kwargs = connection_type.call_args
        self.assertEqual(connection_args, ("127.0.0.1", 11434))
        self.assertGreater(connection_kwargs["timeout"], 0)
        self.assertLessEqual(connection_kwargs["timeout"], 3.0)
        self.assertEqual(connection.requests[0][0:2], ("GET", "/api/tags"))
        self.assertTrue(connection.closed)

    def test_default_transport_rejects_redirects_without_following(self):
        response = FakeHTTPResponse(
            status=302,
            headers={"Location": "http://example.com/escaped"},
        )
        connection = FakeHTTPConnection(response)

        with patch(
            "mlx_agent.verification.http.client.HTTPConnection",
            return_value=connection,
        ) as connection_type:
            with self.assertRaises(http.client.HTTPException):
                _http_json_get("http://127.0.0.1:11434/api/tags")

        connection_type.assert_called_once()
        self.assertEqual(response.read_calls, 0)
        self.assertTrue(connection.closed)

    def test_default_transport_validates_loopback_before_connecting(self):
        with patch(
            "mlx_agent.verification.http.client.HTTPConnection"
        ) as connection_type:
            with self.assertRaises(ValueError):
                _http_json_get("http://example.com:11434/api/tags")

        connection_type.assert_not_called()

    def test_transport_rejects_declared_and_streamed_oversized_bodies(self):
        declared = FakeHTTPResponse(
            chunks=[b"should-not-be-read"],
            headers={"Content-Length": str(INVENTORY_RESPONSE_MAX_BYTES + 1)},
        )
        declared_connection = FakeHTTPConnection(declared)
        with self.assertRaises(ValueError):
            _http_json_request(
                "http://127.0.0.1:8080/v1/models",
                "GET",
                timeout=3.0,
                max_response_bytes=INVENTORY_RESPONSE_MAX_BYTES,
                connection_factory=lambda *args: declared_connection,
            )
        self.assertEqual(declared.read_calls, 0)

        streamed = FakeHTTPResponse(chunks=[b"12345", b"6789"])
        streamed_connection = FakeHTTPConnection(streamed)
        with self.assertRaises(ValueError):
            _http_json_request(
                "http://127.0.0.1:8080/v1/models",
                "GET",
                timeout=3.0,
                max_response_bytes=8,
                connection_factory=lambda *args: streamed_connection,
            )
        self.assertEqual(streamed.read_calls, 2)

    def test_transport_enforces_overall_monotonic_deadline_while_reading(self):
        clock = FakeClock()
        response = FakeHTTPResponse(
            chunks=[b'{"a":', b"1}", b""],
            on_read=lambda: clock.advance(0.6),
        )
        connection = FakeHTTPConnection(response)

        with self.assertRaises(TimeoutError):
            _http_json_request(
                "http://127.0.0.1:8080/v1/models",
                "GET",
                timeout=1.0,
                max_response_bytes=PROBE_RESPONSE_MAX_BYTES,
                connection_factory=lambda *args: connection,
                clock=clock,
            )

        self.assertGreaterEqual(response.read_calls, 2)
        self.assertTrue(connection.sock.timeouts)
        self.assertLess(connection.sock.timeouts[-1], connection.sock.timeouts[0])
        self.assertTrue(connection.closed)

    def test_ollama_tool_probe_uses_bounded_chat_contract(self):
        posts = []

        def http_post(url, payload, timeout=10.0):
            posts.append((url, payload, timeout))
            return {"message": {"tool_calls": []}}

        client = OllamaRuntimeClient(http_post=http_post)
        client.probe_tool_use("local/tools")

        self.assertEqual(posts[0][0], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(posts[0][2], 10.0)
        payload = posts[0][1]
        self.assertEqual(payload["model"], "local/tools")
        self.assertEqual(
            payload["messages"], [{"role": "user", "content": TOOL_USE_PROMPT}]
        )
        self.assertEqual(payload["tools"], TOOL_USE_TOOLS)
        self.assertIsNot(payload["tools"], TOOL_USE_TOOLS)
        self.assertFalse(payload["stream"])
        self.assertLessEqual(payload["options"]["num_predict"], 64)
        json.dumps(payload)

    def test_openai_compatible_tool_probe_uses_bounded_completion_contract(self):
        gets = []
        posts = []

        def http_get(url, timeout=3.0):
            gets.append((url, timeout))
            return {"data": []}

        def http_post(url, payload, timeout=10.0):
            posts.append((url, payload, timeout))
            return {"choices": []}

        client = OpenAICompatibleRuntimeClient(
            "mlx_lm",
            "http://127.0.0.1:8080",
            http_get=http_get,
            http_post=http_post,
        )
        client.list_models()
        client.generate("local/model", "ready?", max_tokens=12)
        client.probe_tool_use("local/model")

        self.assertEqual(gets, [("http://127.0.0.1:8080/v1/models", 3.0)])
        self.assertEqual(posts[0][0], "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(
            posts[0][1],
            {
                "model": "local/model",
                "messages": [{"role": "user", "content": "ready?"}],
                "max_tokens": 12,
            },
        )
        self.assertEqual(posts[1][0], "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(posts[1][2], 10.0)
        probe_payload = posts[1][1]
        self.assertEqual(
            probe_payload["messages"],
            [{"role": "user", "content": TOOL_USE_PROMPT}],
        )
        self.assertEqual(probe_payload["tools"], TOOL_USE_TOOLS)
        self.assertIsNot(probe_payload["tools"], TOOL_USE_TOOLS)
        self.assertEqual(probe_payload["tool_choice"], "auto")
        self.assertLessEqual(probe_payload["max_tokens"], 64)
        json.dumps(probe_payload)

    def test_default_verifier_has_four_local_runtime_clients(self):
        verifier = Verifier()
        self.assertEqual(
            [client.name for client in verifier._runtime_clients],
            ["ollama", "lmstudio", "mlx_lm", "litellm"],
        )

    def test_runtime_base_urls_must_be_credential_free_loopback_origins(self):
        valid = [
            "http://localhost:8080",
            "https://127.0.0.1:8080/",
            "http://[::1]:8080",
        ]
        for base_url in valid:
            with self.subTest(valid=base_url):
                OpenAICompatibleRuntimeClient("local", base_url)

        invalid = [
            "ftp://127.0.0.1:8080",
            "http://user:pass@127.0.0.1:8080",
            "http://example.com:8080",
            "http://192.168.1.10:8080",
            "http://127.0.0.1:8080/v1",
            "http://127.0.0.1:8080?token=secret",
            "http://127.0.0.1:8080#fragment",
        ]
        for base_url in invalid:
            with self.subTest(invalid=base_url):
                with self.assertRaises(ValueError):
                    OpenAICompatibleRuntimeClient("local", base_url)

        with self.assertRaises(ValueError):
            OllamaRuntimeClient(base_url="http://example.com:11434")

    def test_lmstudio_remains_openai_compatible_with_legacy_defaults(self):
        client = LMStudioRuntimeClient()
        self.assertIsInstance(client, OpenAICompatibleRuntimeClient)
        self.assertEqual(client.name, "lmstudio")
        self.assertEqual(client._base_url, "http://127.0.0.1:1234")

    def test_evidence_strength_exposes_exact_contract_names(self):
        self.assertEqual(EvidenceStrength.runtime_tested.value, "runtime_tested")
        self.assertEqual(EvidenceStrength.runtime_inventory.value, "runtime_inventory")
        self.assertEqual(EvidenceStrength.metadata_only.value, "metadata_only")
        self.assertEqual(EvidenceStrength.heuristic_only.value, "heuristic_only")

    def test_verification_status_exposes_exact_contract_names_and_serializes(self):
        self.assertEqual(VerificationStatus.verified.value, "verified")
        self.assertEqual(VerificationStatus.metadata_only.value, "metadata-only")
        self.assertEqual(VerificationStatus.failed.value, "failed")
        self.assertEqual(
            VerificationStatus.unsupported_runtime.value, "unsupported-runtime"
        )
        evidence = VerificationEvidence(
            repo="local/model",
            role="general",
            strength=EvidenceStrength.RUNTIME_TESTED,
            status=VerificationStatus.VERIFIED,
            available_locally=True,
            loads=True,
            reasoning_confirmed=False,
            runtime="fake-runtime",
            note="verified",
        )
        self.assertEqual(evidence.to_dict()["status"], "verified")

    def test_only_installed_models_are_generated_and_missing_models_use_metadata(self):
        runtime = FakeRuntimeClient(
            ["local/normal-7b"],
            responses={"local/normal-7b": {"message": {"content": "pong"}}},
        )
        metadata = FakeMetadataClient(
            {"remote/reasoner-7b": {"metadata_available": True, "tags": ["reasoning"]}}
        )
        verifier = Verifier(runtime_clients=[runtime], metadata_client=metadata)

        installed = verifier.verify(
            {"repo": "local/normal-7b", "role": "general", "reasoning": False},
            {"ram_gb": 32},
        )
        missing = verifier.verify(
            {"repo": "remote/reasoner-7b", "role": "reasoning", "reasoning": True},
            {"ram_gb": 32},
        )

        self.assertEqual(installed.strength, EvidenceStrength.RUNTIME_TESTED)
        self.assertEqual(installed.status, VerificationStatus.VERIFIED)
        self.assertFalse(installed.reasoning_confirmed)
        self.assertEqual(missing.strength, EvidenceStrength.METADATA_ONLY)
        self.assertEqual(missing.status, VerificationStatus.METADATA_ONLY)
        self.assertFalse(missing.available_locally)
        self.assertEqual([call[0] for call in runtime.generated], ["local/normal-7b"])
        self.assertEqual(metadata.inspected, ["remote/reasoner-7b"])
        self.assertEqual(runtime.downloads, [])

    def test_runtime_inventory_is_cached_until_explicitly_cleared(self):
        runtime = FakeRuntimeClient(["local/one", "local/two"])
        verifier = Verifier(runtime_clients=[runtime])

        first = verifier.verify(
            {"repo": "local/one", "role": "tool-use"}, {}, allow_network=False
        )
        second = verifier.verify(
            {"repo": "local/two", "role": "tool-use"}, {}, allow_network=False
        )

        self.assertTrue(first.available_locally)
        self.assertTrue(second.available_locally)
        self.assertEqual(runtime.inventory_calls, 1)

        verifier.clear_inventory_cache()
        verifier.verify(
            {"repo": "local/one", "role": "tool-use"}, {}, allow_network=False
        )
        self.assertEqual(runtime.inventory_calls, 2)

    def test_runtime_inventory_cache_is_thread_safe_and_does_not_cache_probes(self):
        outcome = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": TOOL_USE_TOOL_NAME,
                            "arguments": {"widget_id": "widget-42"},
                        }
                    }
                ]
            }
        }
        runtime = FakeToolRuntimeClient(["local/tools"], probe_response=outcome)
        verifier = Verifier(runtime_clients=[runtime])

        def verify_candidate(_index):
            return verifier.verify(
                {"repo": "local/tools", "role": "tool-use"},
                {},
                allow_network=False,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            evidence = list(executor.map(verify_candidate, range(16)))

        self.assertTrue(all(item.loads for item in evidence))
        self.assertEqual(runtime.inventory_calls, 1)
        self.assertEqual(len(runtime.probed), 16)

    def test_runtime_inventory_errors_are_cached_and_bounded(self):
        runtime = FakeRuntimeClient([], inventory_failure="x" * 1000)
        verifier = Verifier(runtime_clients=[runtime])

        first = verifier.verify(
            {"repo": "missing/one", "role": "general"}, {}, allow_network=False
        )
        second = verifier.verify(
            {"repo": "missing/two", "role": "general"}, {}, allow_network=False
        )

        self.assertEqual(runtime.inventory_calls, 1)
        self.assertEqual(first.strength, EvidenceStrength.HEURISTIC_ONLY)
        self.assertEqual(second.strength, EvidenceStrength.HEURISTIC_ONLY)
        self.assertLessEqual(len(first.note), 300)
        self.assertLessEqual(len(second.note), 300)

    def test_valid_tool_use_probe_returns_runtime_tested_evidence(self):
        outcome = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": TOOL_USE_TOOL_NAME,
                            "arguments": '{"widget_id":"widget-42"}',
                        }
                    }
                ]
            }
        }
        runtime = FakeToolRuntimeClient(["local/tools"], probe_response=outcome)

        evidence = Verifier(runtime_clients=[runtime]).verify(
            {"repo": "local/tools", "role": "tool-use"},
            {"ram_gb": 16},
            allow_network=False,
        )

        self.assertEqual(evidence.strength, EvidenceStrength.RUNTIME_TESTED)
        self.assertEqual(evidence.status, VerificationStatus.VERIFIED)
        self.assertTrue(evidence.available_locally)
        self.assertTrue(evidence.loads)
        self.assertIsNone(evidence.reasoning_confirmed)
        self.assertIn("schema-valid synthetic tool call", evidence.note)
        self.assertEqual(evidence.details["probe_id"], TOOL_USE_PROBE_ID)
        self.assertEqual(evidence.details["outcome"], normalize_tool_call(outcome))
        self.assertEqual(runtime.probed, ["local/tools"])
        self.assertEqual(runtime.generated, [])
        self.assertEqual(runtime.downloads, [])

    def test_invalid_tool_use_probe_still_confirms_model_loaded(self):
        outcome = {"choices": [{"message": {"content": "I cannot call tools"}}]}
        runtime = FakeToolRuntimeClient(["local/tools"], probe_response=outcome)

        evidence = Verifier(runtime_clients=[runtime]).verify(
            {"repo": "local/tools", "role": "tool-use"},
            {},
            allow_network=False,
        )

        self.assertEqual(evidence.strength, EvidenceStrength.RUNTIME_TESTED)
        self.assertEqual(evidence.status, VerificationStatus.FAILED)
        self.assertTrue(evidence.loads)
        self.assertIn("did not return a valid call", evidence.note)
        self.assertEqual(
            evidence.details,
            {
                "probe_id": TOOL_USE_PROBE_ID,
                "outcome": {
                    "valid": False,
                    "tool_name": None,
                    "arguments": None,
                    "reason": "missing_tool_call",
                },
            },
        )

    def test_unsupported_tool_runtime_uses_inventory_without_generation(self):
        runtime = FakeRuntimeClient(["local/tools"])

        evidence = Verifier(runtime_clients=[runtime]).verify(
            {"repo": "local/tools", "role": "tool-use"},
            {},
            allow_network=False,
        )

        self.assertEqual(evidence.strength, EvidenceStrength.RUNTIME_INVENTORY)
        self.assertEqual(evidence.status, VerificationStatus.UNSUPPORTED_RUNTIME)
        self.assertTrue(evidence.available_locally)
        self.assertIsNone(evidence.loads)
        self.assertIsNone(evidence.reasoning_confirmed)
        self.assertEqual(evidence.runtime, "fake-runtime")
        self.assertEqual(
            evidence.details,
            {
                "probe_id": TOOL_USE_PROBE_ID,
                "outcome": {"reason": "unsupported_runtime"},
            },
        )
        self.assertEqual(runtime.generated, [])

    def test_mlx_vlm_is_unsupported_for_tool_probe_even_if_method_exists(self):
        runtime = FakeToolRuntimeClient(
            ["local/vision-tools"],
            probe_response={"message": {"tool_calls": []}},
            name="mlx-vlm",
        )

        evidence = Verifier(runtime_clients=[runtime]).verify(
            {"repo": "local/vision-tools", "role": "tool-use"},
            {},
            allow_network=False,
        )

        self.assertEqual(evidence.strength, EvidenceStrength.RUNTIME_INVENTORY)
        self.assertEqual(evidence.status, VerificationStatus.UNSUPPORTED_RUNTIME)
        self.assertEqual(
            evidence.details["outcome"]["reason"], "unsupported_runtime"
        )
        self.assertEqual(runtime.probed, [])
        self.assertEqual(runtime.generated, [])

    def test_tool_probe_exception_is_bounded_and_does_not_leak_sensitive_context(self):
        sensitive = (
            "secret-response https://127.0.0.1:8080/v1/chat/completions "
            + TOOL_USE_PROMPT
        )
        runtime = FakeToolRuntimeClient(
            ["local/tools"], probe_error=RuntimeError(sensitive)
        )

        evidence = Verifier(runtime_clients=[runtime]).verify(
            {"repo": "local/tools", "role": "tool-use"},
            {},
            allow_network=False,
        )

        self.assertEqual(evidence.strength, EvidenceStrength.RUNTIME_INVENTORY)
        self.assertEqual(evidence.status, VerificationStatus.FAILED)
        self.assertFalse(evidence.loads)
        self.assertEqual(set(evidence.details), {"probe_id", "error"})
        self.assertEqual(evidence.details["probe_id"], TOOL_USE_PROBE_ID)
        self.assertLessEqual(len(evidence.details["error"]), 160)
        self.assertNotIn("secret-response", repr(evidence.to_dict()))
        self.assertNotIn(TOOL_USE_PROMPT, repr(evidence.to_dict()))
        self.assertNotIn("/v1/chat/completions", repr(evidence.to_dict()))

    def test_non_tool_role_retains_existing_generate_behavior(self):
        runtime = FakeToolRuntimeClient(
            ["local/general"],
            probe_response={"message": {"tool_calls": []}},
        )
        runtime.responses["local/general"] = {"message": {"content": "ready"}}

        evidence = Verifier(runtime_clients=[runtime]).verify(
            {"repo": "local/general", "role": "general"},
            {},
            allow_network=False,
        )

        self.assertEqual(evidence.strength, EvidenceStrength.RUNTIME_TESTED)
        self.assertEqual(evidence.status, VerificationStatus.VERIFIED)
        self.assertEqual(
            runtime.generated,
            [("local/general", "Reply with the single word ready.", 24)],
        )
        self.assertEqual(runtime.probed, [])

    def test_tool_use_fallback_mentions_probe_contract_without_claiming_it_ran(self):
        metadata = FakeMetadataClient(
            {"remote/tools": {"metadata_available": True, "tags": ["tools"]}}
        )
        verifier = Verifier(runtime_clients=[], metadata_client=metadata)

        metadata_evidence = verifier.verify(
            {"repo": "remote/tools", "role": "tool-use"}, {}, allow_network=True
        )
        heuristic_evidence = verifier.verify(
            {"repo": "offline/tools", "role": "tool-use"}, {}, allow_network=False
        )

        self.assertEqual(metadata_evidence.details["probe_id"], TOOL_USE_PROBE_ID)
        self.assertEqual(heuristic_evidence.details["probe_id"], TOOL_USE_PROBE_ID)
        self.assertEqual(metadata_evidence.status, VerificationStatus.METADATA_ONLY)
        self.assertEqual(heuristic_evidence.status, VerificationStatus.METADATA_ONLY)
        self.assertNotIn("ran", metadata_evidence.note.lower())
        self.assertNotIn("ran", heuristic_evidence.note.lower())

    def test_hidden_reasoning_with_empty_content_is_runtime_confirmed(self):
        runtime = FakeRuntimeClient(
            ["local/thinking-7b"],
            responses={
                "local/thinking-7b": {
                    "choices": [
                        {"message": {"content": "", "reasoning_content": "working"}}
                    ]
                }
            },
        )
        evidence = Verifier(runtime_clients=[runtime]).verify(
            {"repo": "local/thinking-7b", "role": "general"},
            {"ram_gb": 16},
            allow_network=False,
        )

        self.assertEqual(evidence.strength, EvidenceStrength.RUNTIME_TESTED)
        self.assertTrue(evidence.reasoning_confirmed)
        self.assertTrue(evidence.loads)

    def test_visible_content_does_not_override_runtime_reasoning_signal(self):
        runtime = FakeRuntimeClient(
            ["local/thinking-7b"],
            responses={
                "local/thinking-7b": {
                    "choices": [
                        {
                            "message": {
                                "content": "final answer",
                                "reasoning_content": "hidden chain of thought",
                            }
                        }
                    ]
                }
            },
        )

        evidence = Verifier(runtime_clients=[runtime]).verify(
            {"repo": "local/thinking-7b", "role": "general"},
            {"ram_gb": 16},
            allow_network=False,
        )

        self.assertTrue(evidence.reasoning_confirmed)
        self.assertEqual(evidence.details["reasoning_evidence"], "runtime_hidden")

    def test_metadata_reasoning_tag_confirms_reasoning_for_missing_model(self):
        metadata = FakeMetadataClient(
            {"remote/reasoner-7b": {"metadata_available": True, "tags": ["reasoning"]}}
        )

        evidence = Verifier(runtime_clients=[], metadata_client=metadata).verify(
            {"repo": "remote/reasoner-7b", "role": "general", "reasoning": False},
            {"ram_gb": 16},
        )

        self.assertEqual(evidence.strength, EvidenceStrength.METADATA_ONLY)
        self.assertTrue(evidence.reasoning_confirmed)
        self.assertEqual(evidence.details["reasoning_evidence"], "metadata_tags")

    def test_runtime_and_metadata_failures_become_bounded_evidence(self):
        runtime = FakeRuntimeClient(
            ["local/broken-7b"], failures={"local/broken-7b": "generation failed"}
        )
        metadata = FakeMetadataClient(failures={"remote/broken-7b": "metadata failed"})
        verifier = Verifier(runtime_clients=[runtime], metadata_client=metadata)

        local = verifier.verify(
            {"repo": "local/broken-7b", "role": "coding"}, {"ram_gb": 64}
        )
        remote = verifier.verify(
            {"repo": "remote/broken-7b", "role": "coding", "reasoning": False},
            {"ram_gb": 64},
        )

        self.assertEqual(local.strength, EvidenceStrength.RUNTIME_INVENTORY)
        self.assertEqual(local.status, VerificationStatus.FAILED)
        self.assertFalse(local.loads)
        self.assertIn("generation failed", local.note)
        self.assertEqual(remote.strength, EvidenceStrength.HEURISTIC_ONLY)
        self.assertEqual(remote.status, VerificationStatus.METADATA_ONLY)
        self.assertIn("metadata failed", remote.note)
        self.assertLessEqual(len(local.note), 300)
        self.assertLessEqual(len(remote.note), 300)

    def test_generation_failure_redacts_credentials_before_serialization(self):
        sentinel = "generation-secret-sentinel"
        runtime = FakeRuntimeClient(
            ["local/broken"],
            failures={
                "local/broken": (
                    "token={0} at "
                    "https://agent:{0}@runtime.example/v1?password={0}"
                ).format(sentinel)
            },
        )

        evidence = Verifier(runtime_clients=[runtime]).verify(
            {"repo": "local/broken", "role": "coding"},
            {},
            allow_network=False,
        )
        serialized = json.dumps(evidence.to_dict(), sort_keys=True)

        self.assertEqual(evidence.status, VerificationStatus.FAILED)
        self.assertNotIn(sentinel, serialized)
        self.assertNotIn("agent:", serialized)
        self.assertNotIn("?password=", serialized)
        self.assertIn("RuntimeError", evidence.note)

    def test_inventory_and_metadata_fallbacks_redact_credentials(self):
        inventory_sentinel = "inventory-secret-sentinel"
        metadata_sentinel = "metadata-secret-sentinel"
        runtime = FakeRuntimeClient(
            [],
            inventory_failure=(
                "inventory failed at "
                "https://user:{0}@runtime.example/models?token={0}"
            ).format(inventory_sentinel),
        )
        metadata = FakeMetadataClient(
            failures={
                "remote/model": "password={0} metadata unavailable".format(
                    metadata_sentinel
                )
            }
        )

        evidence = Verifier(
            runtime_clients=[runtime], metadata_client=metadata
        ).verify(
            {"repo": "remote/model", "role": "coding"},
            {},
            allow_network=True,
        )
        serialized = json.dumps(evidence.to_dict(), sort_keys=True)

        self.assertEqual(evidence.status, VerificationStatus.METADATA_ONLY)
        self.assertNotIn(inventory_sentinel, serialized)
        self.assertNotIn(metadata_sentinel, serialized)
        self.assertNotIn("user:", serialized)
        self.assertNotIn("?token=", serialized)
        self.assertIn("RuntimeError", evidence.note)
        self.assertIn("<redacted>", evidence.note)

    def test_offline_missing_model_uses_heuristics_without_metadata_request(self):
        runtime = FakeRuntimeClient([])
        metadata = FakeMetadataClient()
        evidence = Verifier(runtime_client=runtime, metadata_client=metadata).verify(
            {"repo": "remote/model-7b", "role": "general", "reasoning": False},
            {"ram_gb": 8},
            allow_network=False,
        )

        self.assertEqual(evidence.strength, EvidenceStrength.HEURISTIC_ONLY)
        self.assertEqual(metadata.inspected, [])
        self.assertEqual(runtime.generated, [])


if __name__ == "__main__":
    unittest.main()
