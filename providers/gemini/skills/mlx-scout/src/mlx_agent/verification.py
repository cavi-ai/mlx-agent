"""Safe, bounded verification of discovered local-model candidates."""

from __future__ import annotations

import http.client
import ipaddress
import json
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from .wiring import redact_secrets


TOOL_USE_PROBE_ID = "tool-use-v1"
TOOL_USE_TOOL_NAME = "lookup_widget"
TOOL_USE_ARGUMENTS = MappingProxyType({"widget_id": "widget-42"})
TOOL_USE_PROMPT = (
    "Call lookup_widget exactly once for widget-42. "
    "Do not answer directly and do not call any other tool."
)
TOOL_USE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_USE_TOOL_NAME,
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
]
_TOOL_USE_ARGUMENTS_MAX_CHARS = 512
INVENTORY_RESPONSE_MAX_BYTES = 1024 * 1024
PROBE_RESPONSE_MAX_BYTES = 256 * 1024
_HTTP_READ_CHUNK_BYTES = 16 * 1024


def normalize_tool_call(response):
    """Normalize one synthetic tool call without retaining raw model output."""
    try:
        message = _tool_call_message(response)
        if message is None:
            return _tool_call_result("invalid_response")

        if "tool_calls" not in message or message["tool_calls"] is None:
            return _tool_call_result("missing_tool_call")
        tool_calls = message["tool_calls"]
        if not isinstance(tool_calls, list):
            return _tool_call_result("invalid_response")
        if not tool_calls:
            return _tool_call_result("missing_tool_call")
        if len(tool_calls) != 1:
            return _tool_call_result("multiple_tool_calls")

        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            return _tool_call_result("invalid_response")
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return _tool_call_result("invalid_response")
        raw_name = function.get("name")
        if not isinstance(raw_name, str):
            return _tool_call_result("invalid_response")
        if raw_name != TOOL_USE_TOOL_NAME:
            return _tool_call_result("wrong_tool")

        if "arguments" not in function:
            return _tool_call_result("malformed_arguments")
        arguments = function["arguments"]
        if isinstance(arguments, str):
            if len(arguments) > _TOOL_USE_ARGUMENTS_MAX_CHARS:
                return _tool_call_result("malformed_arguments")
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, RecursionError):
                return _tool_call_result("malformed_arguments")
        if not isinstance(arguments, dict):
            return _tool_call_result("malformed_arguments")
        if arguments != TOOL_USE_ARGUMENTS:
            return _tool_call_result("schema_invalid")
        return _tool_call_result("valid")
    except Exception:
        return _tool_call_result("invalid_response")


def _tool_call_message(response):
    if not isinstance(response, dict):
        return None
    if "message" in response:
        message = response["message"]
        return message if isinstance(message, dict) else None
    if "choices" not in response:
        return None
    choices = response["choices"]
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    return message if isinstance(message, dict) else None


def _tool_call_result(reason):
    valid = reason == "valid"
    return {
        "valid": valid,
        "tool_name": TOOL_USE_TOOL_NAME if valid else None,
        "arguments": dict(TOOL_USE_ARGUMENTS) if valid else None,
        "reason": reason,
    }


class EvidenceStrength(str, Enum):
    """Ordered labels describing how directly a candidate was verified."""

    runtime_tested = "runtime_tested"
    runtime_inventory = "runtime_inventory"
    metadata_only = "metadata_only"
    heuristic_only = "heuristic_only"

    # Conventional aliases preserve a readable internal style while the exact
    # lowercase contract names remain available to API consumers.
    RUNTIME_TESTED = runtime_tested
    RUNTIME_INVENTORY = runtime_inventory
    METADATA_ONLY = metadata_only
    HEURISTIC_ONLY = heuristic_only


class VerificationStatus(str, Enum):
    """Outcome labels independent from the source strength of the evidence."""

    verified = "verified"
    metadata_only = "metadata-only"
    failed = "failed"
    unsupported_runtime = "unsupported-runtime"

    VERIFIED = verified
    METADATA_ONLY = metadata_only
    FAILED = failed
    UNSUPPORTED_RUNTIME = unsupported_runtime


@dataclass(frozen=True)
class VerificationEvidence:
    """Portable evidence from one candidate verification attempt."""

    repo: str
    role: str
    strength: EvidenceStrength
    status: VerificationStatus
    available_locally: bool
    loads: Optional[bool]
    reasoning_confirmed: Optional[bool]
    runtime: Optional[str]
    note: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["strength"] = self.strength.value
        value["status"] = self.status.value
        return value


def _http_json_get(url, timeout=3.0):
    """Read a local runtime endpoint without installing or starting anything."""
    return _http_json_request(
        url,
        "GET",
        timeout=timeout,
        max_response_bytes=INVENTORY_RESPONSE_MAX_BYTES,
    )


def _http_json_post(url, payload, timeout=10.0):
    """Send the single bounded generation probe used by verification."""
    return _http_json_request(
        url,
        "POST",
        payload=payload,
        timeout=timeout,
        max_response_bytes=PROBE_RESPONSE_MAX_BYTES,
    )


def _http_json_request(
    url,
    method,
    payload=None,
    timeout=10.0,
    max_response_bytes=PROBE_RESPONSE_MAX_BYTES,
    connection_factory=None,
    clock=time.monotonic,
):
    """Exchange bounded JSON with one validated loopback endpoint."""
    parsed = _validated_loopback_url(url)
    deadline = clock() + timeout
    remaining = _deadline_remaining(deadline, clock)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if connection_factory is None:
        connection_type = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(parsed.hostname, port, timeout=remaining)
    else:
        connection = connection_factory(
            parsed.scheme,
            parsed.hostname,
            port,
            remaining,
        )
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    try:
        _set_connection_timeout(connection, _deadline_remaining(deadline, clock))
        connection.request(method, parsed.path or "/", body=body, headers=headers)
        _set_connection_timeout(connection, _deadline_remaining(deadline, clock))
        response = connection.getresponse()
        _deadline_remaining(deadline, clock)
        if 300 <= response.status < 400:
            raise http.client.HTTPException(
                "redirect responses are not allowed for local runtime requests"
            )
        if not 200 <= response.status < 300:
            raise http.client.HTTPException(
                "local runtime returned HTTP status {0}".format(response.status)
            )
        response_body = _read_bounded_response(
            response,
            connection,
            max_response_bytes,
            deadline,
            clock,
        )
        return json.loads(response_body.decode("utf-8"))
    finally:
        connection.close()


def _read_bounded_response(response, connection, limit, deadline, clock):
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("response size limit must be a positive integer")
    content_length = response.getheader("Content-Length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid Content-Length from local runtime") from error
        if declared_length < 0:
            raise ValueError("invalid Content-Length from local runtime")
        if declared_length > limit:
            raise ValueError("local runtime response exceeds size limit")

    chunks = []
    total = 0
    while True:
        remaining = _deadline_remaining(deadline, clock)
        _set_connection_timeout(connection, remaining)
        chunk = response.read(min(_HTTP_READ_CHUNK_BYTES, limit - total + 1))
        _deadline_remaining(deadline, clock)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ValueError("local runtime response exceeds size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _deadline_remaining(deadline, clock):
    remaining = deadline - clock()
    if remaining <= 0:
        raise TimeoutError("local runtime request exceeded its overall deadline")
    return remaining


def _set_connection_timeout(connection, timeout):
    sock = getattr(connection, "sock", None)
    if sock is not None:
        sock.settimeout(timeout)


def _local_runtime_origin(base_url):
    """Return a normalized credential-free HTTP(S) loopback origin."""
    parsed = _validated_loopback_url(base_url)
    if parsed.path not in ("", "/"):
        raise ValueError("runtime base URL must be an origin without a path")
    return base_url.rstrip("/")


def _validated_loopback_url(url):
    if not isinstance(url, str) or url != url.strip():
        raise ValueError("runtime URL must be a trimmed string")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("runtime URL must use HTTP(S)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("runtime URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("runtime URL must not contain a query or fragment")
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("runtime URL must include a host")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("runtime URL has an invalid port") from error
    if hostname.casefold() != "localhost":
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as error:
            raise ValueError("runtime URL must use a loopback host") from error
        if not address.is_loopback:
            raise ValueError("runtime URL must use a loopback host")
    return parsed


def _tool_use_payload_tools():
    """Copy the fixed schema into ordinary JSON-serializable containers."""
    return deepcopy(TOOL_USE_TOOLS)


class OllamaRuntimeClient:
    """Read-only inventory and bounded generation adapter for an existing Ollama."""

    name = "ollama"

    def __init__(self, http_get=None, http_post=None, base_url="http://127.0.0.1:11434"):
        self._http_get = http_get or _http_json_get
        self._http_post = http_post or _http_json_post
        self._base_url = _local_runtime_origin(base_url)

    def list_models(self):
        return self._http_get("{0}/api/tags".format(self._base_url), timeout=3.0)

    def generate(self, model, prompt, max_tokens):
        return self._http_post(
            "{0}/api/chat".format(self._base_url),
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=10.0,
        )

    def probe_tool_use(self, model):
        return self._http_post(
            "{0}/api/chat".format(self._base_url),
            {
                "model": model,
                "messages": [{"role": "user", "content": TOOL_USE_PROMPT}],
                "tools": _tool_use_payload_tools(),
                "stream": False,
                "options": {"num_predict": 64},
            },
            timeout=10.0,
        )


class OpenAICompatibleRuntimeClient:
    """Bounded adapter for an existing local OpenAI-compatible runtime."""

    def __init__(self, name, base_url, http_get=None, http_post=None):
        self.name = name
        self._http_get = http_get or _http_json_get
        self._http_post = http_post or _http_json_post
        self._base_url = _local_runtime_origin(base_url)

    def list_models(self):
        return self._http_get("{0}/v1/models".format(self._base_url), timeout=3.0)

    def generate(self, model, prompt, max_tokens):
        return self._http_post(
            "{0}/v1/chat/completions".format(self._base_url),
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=10.0,
        )

    def probe_tool_use(self, model):
        return self._http_post(
            "{0}/v1/chat/completions".format(self._base_url),
            {
                "model": model,
                "messages": [{"role": "user", "content": TOOL_USE_PROMPT}],
                "tools": _tool_use_payload_tools(),
                "tool_choice": "auto",
                "max_tokens": 64,
            },
            timeout=10.0,
        )


class LMStudioRuntimeClient(OpenAICompatibleRuntimeClient):
    """Backward-compatible adapter for an existing LM Studio runtime."""

    name = "lmstudio"

    def __init__(self, http_get=None, http_post=None, base_url="http://127.0.0.1:1234"):
        super().__init__(
            "lmstudio",
            base_url,
            http_get=http_get,
            http_post=http_post,
        )


class Verifier:
    """Verify installed candidates without ever installing or downloading them.

    Runtime clients implement ``list_models()`` and ``generate(model, prompt,
    max_tokens)``; tool-capable clients also implement ``probe_tool_use(model)``.
    Metadata clients implement ``inspect_model(repo)``. Keeping these protocols
    implicit allows provider adapters and tests to supply clients without adding
    runtime dependencies.
    """

    def __init__(self, runtime_clients=None, metadata_client=None, runtime_client=None):
        if runtime_clients is None:
            runtime_clients = (
                []
                if runtime_client is not None
                else [
                    OllamaRuntimeClient(),
                    LMStudioRuntimeClient(),
                    OpenAICompatibleRuntimeClient("mlx_lm", "http://127.0.0.1:8080"),
                    OpenAICompatibleRuntimeClient("litellm", "http://127.0.0.1:4000"),
                ]
            )
        if isinstance(runtime_clients, dict):
            runtime_clients = list(runtime_clients.values())
        self._runtime_clients = list(runtime_clients)
        if runtime_client is not None:
            self._runtime_clients.append(runtime_client)
        self._metadata_client = metadata_client
        self._inventory_cache = {}
        self._inventory_cache_lock = threading.Lock()

    def clear_inventory_cache(self):
        """Start a new verification phase with fresh runtime inventories."""
        with self._inventory_cache_lock:
            self._inventory_cache.clear()

    def _runtime_inventory(self, runtime):
        cache_key = id(runtime)
        with self._inventory_cache_lock:
            if cache_key not in self._inventory_cache:
                try:
                    result = (installed_model_ids(runtime.list_models()), None)
                except Exception as error:
                    result = (None, _safe_error(error))
                self._inventory_cache[cache_key] = result
            return self._inventory_cache[cache_key]

    def verify(self, candidate, host, allow_network=True) -> VerificationEvidence:
        """Return the strongest available evidence without mutating runtimes."""
        del host  # Reserved for runtime adapters whose probes depend on host facts.
        repo = _candidate_repo(candidate)
        role = str(candidate.get("role") or candidate.get("heuristics", {}).get("role") or "general")
        inventory_errors = []

        for runtime in self._runtime_clients:
            installed, inventory_error = self._runtime_inventory(runtime)
            if inventory_error is not None:
                inventory_errors.append(
                    "{0}: {1}".format(_runtime_name(runtime), inventory_error)
                )
                continue
            if not _is_installed(repo, installed):
                continue
            runtime_name = _runtime_name(runtime)
            if role == "tool-use":
                return _verify_tool_use(repo, role, runtime_name, runtime)
            try:
                response = runtime.generate(
                    repo,
                    "Reply with the single word ready.",
                    max_tokens=24,
                )
                content, hidden_reasoning = _generation_text(response)
                reasoning_confirmed, reasoning_evidence = _reasoning_confirmation(
                    candidate, hidden_reasoning
                )
                return VerificationEvidence(
                    repo=repo,
                    role=role,
                    strength=EvidenceStrength.RUNTIME_TESTED,
                    status=VerificationStatus.VERIFIED,
                    available_locally=True,
                    loads=True,
                    reasoning_confirmed=reasoning_confirmed,
                    runtime=runtime_name,
                    note="Runtime generation completed without downloading the model.",
                    details={
                        "visible_content": bool(content.strip()),
                        "hidden_reasoning": bool(hidden_reasoning),
                        "reasoning_evidence": reasoning_evidence,
                    },
                )
            except Exception as error:
                safe_error = _safe_error(error)
                return VerificationEvidence(
                    repo=repo,
                    role=role,
                    strength=EvidenceStrength.RUNTIME_INVENTORY,
                    status=VerificationStatus.FAILED,
                    available_locally=True,
                    loads=False,
                    reasoning_confirmed=None,
                    runtime=runtime_name,
                    note=_bounded(
                        "Installed runtime inventory found the model, but "
                        "generation failed: {0}".format(safe_error)
                    ),
                    details={"error": safe_error},
                )

        if allow_network and self._metadata_client is not None:
            try:
                metadata = self._metadata_client.inspect_model(repo)
                reasoning_confirmed, reasoning_evidence = _reasoning_confirmation(
                    candidate, metadata
                )
                details = _bounded_metadata(metadata)
                details["reasoning_evidence"] = reasoning_evidence
                if role == "tool-use":
                    details["probe_id"] = TOOL_USE_PROBE_ID
                return VerificationEvidence(
                    repo=repo,
                    role=role,
                    strength=EvidenceStrength.METADATA_ONLY,
                    status=VerificationStatus.METADATA_ONLY,
                    available_locally=False,
                    loads=None,
                    reasoning_confirmed=reasoning_confirmed,
                    runtime=None,
                    note="Model is not installed; repository metadata was inspected without downloading it.",
                    details=details,
                )
            except Exception as error:
                inventory_errors.append("metadata: {0}".format(_safe_error(error)))

        note = "Model is not installed; only candidate heuristics were available."
        if inventory_errors:
            note += " Probe errors: {0}".format("; ".join(inventory_errors))
        details = {
            "reasoning_heuristic": candidate.get("reasoning"),
            "reasoning_source": candidate.get("reason_src"),
        }
        if role == "tool-use":
            details["probe_id"] = TOOL_USE_PROBE_ID
        return VerificationEvidence(
            repo=repo,
            role=role,
            strength=EvidenceStrength.HEURISTIC_ONLY,
            status=VerificationStatus.METADATA_ONLY,
            available_locally=False,
            loads=None,
            reasoning_confirmed=None,
            runtime=None,
            note=_bounded(note),
            details=details,
        )


def _verify_tool_use(repo, role, runtime_name, runtime):
    if runtime_name == "mlx-vlm" or not callable(getattr(runtime, "probe_tool_use", None)):
        return VerificationEvidence(
            repo=repo,
            role=role,
            strength=EvidenceStrength.RUNTIME_INVENTORY,
            status=VerificationStatus.UNSUPPORTED_RUNTIME,
            available_locally=True,
            loads=None,
            reasoning_confirmed=None,
            runtime=runtime_name,
            note=_bounded(
                "Installed runtime inventory found the model, but this runtime "
                "does not support the bounded tool-use probe."
            ),
            details={
                "probe_id": TOOL_USE_PROBE_ID,
                "outcome": {"reason": "unsupported_runtime"},
            },
        )
    try:
        outcome = normalize_tool_call(runtime.probe_tool_use(repo))
    except Exception as error:
        error_name = _redacted_error(error)
        return VerificationEvidence(
            repo=repo,
            role=role,
            strength=EvidenceStrength.RUNTIME_INVENTORY,
            status=VerificationStatus.FAILED,
            available_locally=True,
            loads=False,
            reasoning_confirmed=None,
            runtime=runtime_name,
            note=_bounded(
                "Installed runtime inventory found the model, but the bounded "
                "tool-use probe failed: {0}".format(error_name)
            ),
            details={
                "probe_id": TOOL_USE_PROBE_ID,
                "error": _bounded(error_name, 160),
            },
        )
    if outcome["valid"]:
        note = "Runtime returned a schema-valid synthetic tool call."
    else:
        note = "Runtime responded, but the probe did not return a valid call."
    return VerificationEvidence(
        repo=repo,
        role=role,
        strength=EvidenceStrength.RUNTIME_TESTED,
        status=(
            VerificationStatus.VERIFIED
            if outcome["valid"]
            else VerificationStatus.FAILED
        ),
        available_locally=True,
        loads=True,
        reasoning_confirmed=None,
        runtime=runtime_name,
        note=note,
        details={"probe_id": TOOL_USE_PROBE_ID, "outcome": outcome},
    )


def _redacted_error(error):
    name = error.__class__.__name__
    return _bounded(name if name else "runtime error", 160)


def _safe_error(error):
    name = error.__class__.__name__ or "Error"
    message = redact_secrets(str(error))
    value = "{0}: {1}".format(name, message) if message else name
    return _bounded(value, 160)


def _candidate_repo(candidate):
    if not isinstance(candidate, dict):
        raise TypeError("candidate must be an object")
    repo = candidate.get("repo") or candidate.get("repository")
    if not isinstance(repo, str) or not repo:
        raise ValueError("candidate.repo must be a non-empty string")
    return repo


def _runtime_name(runtime):
    return str(getattr(runtime, "name", runtime.__class__.__name__))


def installed_model_ids(records) -> set:
    """Return model IDs from a supported runtime's read-only inventory response."""
    if isinstance(records, dict):
        records = records.get("models", records.get("data", []))
    models = set()
    for record in records or []:
        if isinstance(record, str):
            models.add(record)
            continue
        if isinstance(record, dict):
            value = record.get("repo") or record.get("id") or record.get("name") or record.get("model")
            if isinstance(value, str) and value:
                models.add(value)
    return models


def _is_installed(repo, installed):
    normalized = repo.casefold()
    return any(item.casefold() == normalized for item in installed)


def _generation_text(response):
    if isinstance(response, str):
        return response, ""
    if not isinstance(response, dict):
        return "", ""
    message = response.get("message")
    if message is None:
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or choices[0]
    if not isinstance(message, dict):
        message = response
    content = message.get("content") or message.get("response") or ""
    hidden = (
        message.get("reasoning_content")
        or message.get("thinking")
        or message.get("reasoning")
        or ""
    )
    return str(content), str(hidden)


def _reasoning_confirmation(candidate, signal):
    """Prefer any positive runtime or metadata signal over an inconclusive probe."""
    if isinstance(signal, str) and signal.strip():
        return True, "runtime_hidden"
    if isinstance(signal, dict):
        if signal.get("reasoning") is True:
            source = signal.get("reason_src") or "field"
            return True, "metadata_{0}".format(source)
        tags = signal.get("tags") or []
        if any(str(tag).lower() in ("reasoning", "thinking", "chain-of-thought") for tag in tags):
            return True, "metadata_tags"
        if signal.get("chat_template"):
            return True, "metadata_chat_template"
    if candidate.get("reasoning") is True and candidate.get("reason_src") in (
        "chat_template", "tags",
    ):
        return True, "discovery_{0}".format(candidate["reason_src"])
    return False, "none"


def _bounded(value, limit=300):
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _bounded_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}
    allowed = ("metadata_available", "gated", "license", "tags", "reasoning", "reason_src", "weight_bytes")
    value = {key: metadata[key] for key in allowed if key in metadata}
    if isinstance(value.get("tags"), list):
        value["tags"] = value["tags"][:20]
    return value
