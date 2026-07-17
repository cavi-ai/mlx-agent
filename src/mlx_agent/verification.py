"""Safe, bounded verification of discovered local-model candidates."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


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


@dataclass(frozen=True)
class VerificationEvidence:
    """Portable evidence from one candidate verification attempt."""

    repo: str
    role: str
    strength: EvidenceStrength
    available_locally: bool
    loads: Optional[bool]
    reasoning_confirmed: Optional[bool]
    runtime: Optional[str]
    note: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["strength"] = self.strength.value
        return value


def _http_json_get(url, timeout=3.0):
    """Read a local runtime endpoint without installing or starting anything."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _http_json_post(url, payload, timeout=10.0):
    """Send the single bounded generation probe used by verification."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class OllamaRuntimeClient:
    """Read-only inventory and bounded generation adapter for an existing Ollama."""

    name = "ollama"

    def __init__(self, http_get=None, http_post=None, base_url="http://127.0.0.1:11434"):
        self._http_get = http_get or _http_json_get
        self._http_post = http_post or _http_json_post
        self._base_url = base_url.rstrip("/")

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


class LMStudioRuntimeClient:
    """Read-only inventory and bounded generation adapter for an existing LM Studio."""

    name = "lmstudio"

    def __init__(self, http_get=None, http_post=None, base_url="http://127.0.0.1:1234"):
        self._http_get = http_get or _http_json_get
        self._http_post = http_post or _http_json_post
        self._base_url = base_url.rstrip("/")

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


class Verifier:
    """Verify installed candidates without ever installing or downloading them.

    Runtime clients implement ``list_models()`` and ``generate(model, prompt,
    max_tokens)``. Metadata clients implement ``inspect_model(repo)``. Keeping
    these small protocols implicit allows provider adapters and tests to supply
    clients without adding runtime dependencies.
    """

    def __init__(self, runtime_clients=None, metadata_client=None, runtime_client=None):
        if runtime_clients is None:
            runtime_clients = (
                [] if runtime_client is not None else [OllamaRuntimeClient(), LMStudioRuntimeClient()]
            )
        if isinstance(runtime_clients, dict):
            runtime_clients = list(runtime_clients.values())
        self._runtime_clients = list(runtime_clients)
        if runtime_client is not None:
            self._runtime_clients.append(runtime_client)
        self._metadata_client = metadata_client

    def verify(self, candidate, host, allow_network=True) -> VerificationEvidence:
        """Return the strongest available evidence without mutating runtimes."""
        del host  # Reserved for runtime adapters whose probes depend on host facts.
        repo = _candidate_repo(candidate)
        role = str(candidate.get("role") or candidate.get("heuristics", {}).get("role") or "general")
        inventory_errors = []

        for runtime in self._runtime_clients:
            try:
                installed = _installed_models(runtime.list_models())
            except Exception as error:
                inventory_errors.append("{0}: {1}".format(_runtime_name(runtime), error))
                continue
            if not _is_installed(repo, installed):
                continue
            runtime_name = _runtime_name(runtime)
            try:
                response = runtime.generate(
                    repo,
                    "Reply with the single word ready.",
                    max_tokens=24,
                )
                content, hidden_reasoning = _generation_text(response)
                reasoning_confirmed = bool(hidden_reasoning and not content.strip())
                if content.strip():
                    reasoning_confirmed = False
                return VerificationEvidence(
                    repo=repo,
                    role=role,
                    strength=EvidenceStrength.RUNTIME_TESTED,
                    available_locally=True,
                    loads=True,
                    reasoning_confirmed=reasoning_confirmed,
                    runtime=runtime_name,
                    note="Runtime generation completed without downloading the model.",
                    details={"visible_content": bool(content.strip()), "hidden_reasoning": bool(hidden_reasoning)},
                )
            except Exception as error:
                return VerificationEvidence(
                    repo=repo,
                    role=role,
                    strength=EvidenceStrength.RUNTIME_INVENTORY,
                    available_locally=True,
                    loads=False,
                    reasoning_confirmed=None,
                    runtime=runtime_name,
                    note=_bounded("Installed runtime inventory found the model, but generation failed: {0}".format(error)),
                    details={"error": _bounded(str(error), 160)},
                )

        if allow_network and self._metadata_client is not None:
            try:
                metadata = self._metadata_client.inspect_model(repo)
                return VerificationEvidence(
                    repo=repo,
                    role=role,
                    strength=EvidenceStrength.METADATA_ONLY,
                    available_locally=False,
                    loads=None,
                    reasoning_confirmed=None,
                    runtime=None,
                    note="Model is not installed; repository metadata was inspected without downloading it.",
                    details=_bounded_metadata(metadata),
                )
            except Exception as error:
                inventory_errors.append("metadata: {0}".format(error))

        note = "Model is not installed; only candidate heuristics were available."
        if inventory_errors:
            note += " Probe errors: {0}".format("; ".join(inventory_errors))
        return VerificationEvidence(
            repo=repo,
            role=role,
            strength=EvidenceStrength.HEURISTIC_ONLY,
            available_locally=False,
            loads=None,
            reasoning_confirmed=None,
            runtime=None,
            note=_bounded(note),
            details={
                "reasoning_heuristic": candidate.get("reasoning"),
                "reasoning_source": candidate.get("reason_src"),
            },
        )


def _candidate_repo(candidate):
    if not isinstance(candidate, dict):
        raise TypeError("candidate must be an object")
    repo = candidate.get("repo") or candidate.get("repository")
    if not isinstance(repo, str) or not repo:
        raise ValueError("candidate.repo must be a non-empty string")
    return repo


def _runtime_name(runtime):
    return str(getattr(runtime, "name", runtime.__class__.__name__))


def _installed_models(records) -> set:
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
