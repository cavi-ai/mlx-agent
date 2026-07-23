"""Opt-in smoke test for installed tool-use models on Apple Silicon."""

from __future__ import annotations

import json
import os
import platform
import re
import time
import unittest

from mlx_agent.huggingface import HuggingFaceClient
from mlx_agent.models import TOOL_USE_HINTS, classify_roles
from mlx_agent.verification import (
    EvidenceStrength,
    LMStudioRuntimeClient,
    OllamaRuntimeClient,
    OpenAICompatibleRuntimeClient,
    TOOL_USE_PROBE_ID,
    TOOL_USE_PROMPT,
    VerificationEvidence,
    VerificationStatus,
    Verifier,
    installed_model_ids,
)


MAX_INSTALLED_MODELS_PER_RUNTIME = 6
MAX_INSTALLED_MODELS_GLOBAL = 12
MAX_METADATA_INSPECTIONS = 3
LIVE_SELECTION_TIMEOUT_SECONDS = 15.0
HF_METADATA_TIMEOUT_SECONDS = 3.0
_HF_REPOSITORY_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_QWEN3_NAME_HINT = re.compile(r"(?:^|[/_.-])qwen3(?:$|[:/_.-])", re.IGNORECASE)
_NORMALIZED_OUTCOME_FIELDS = {"valid", "tool_name", "arguments", "reason"}
_LOCAL_MODEL_PATH_PREFIXES = ("model/", "models/", "weight/", "weights/")


def _direct_local_runtime_clients():
    return (
        OllamaRuntimeClient(),
        LMStudioRuntimeClient(),
        OpenAICompatibleRuntimeClient("mlx_lm", "http://127.0.0.1:8080"),
    )


def _huggingface_repository_id(model_id):
    if not isinstance(model_id, str):
        return None
    value = model_id.strip()
    if value != model_id or value.startswith(("/", "./", "../", "~")):
        return None
    if value.lower().startswith(_LOCAL_MODEL_PATH_PREFIXES):
        return None
    if "\\" in value:
        return None
    if value.lower().startswith("hf.co/"):
        value = value[6:]
    if value.count("/") != 1:
        return None
    owner, repository = value.split("/", 1)
    repository = repository.split(":", 1)[0]
    normalized = "{0}/{1}".format(owner, repository)
    return normalized if _HF_REPOSITORY_ID.fullmatch(normalized) else None


def _has_tool_use_name_hint(model_id):
    return bool(TOOL_USE_HINTS.search(model_id) or _QWEN3_NAME_HINT.search(model_id))


def _candidate(model_id, metadata):
    roles, tool_use = classify_roles(model_id, metadata)
    if "tool-use" not in roles:
        return None
    return {
        "repo": model_id,
        "role": "tool-use",
        "roles": list(roles),
        "tool_use": tool_use,
    }


def _inventory_direct_runtimes(runtime_clients, deadline, clock=time.monotonic):
    inventories = []
    total = 0
    for runtime in runtime_clients:
        if clock() >= deadline or total >= MAX_INSTALLED_MODELS_GLOBAL:
            break
        try:
            model_ids = sorted(installed_model_ids(runtime.list_models()))
        except Exception:
            continue
        remaining = MAX_INSTALLED_MODELS_GLOBAL - total
        bounded = model_ids[: min(MAX_INSTALLED_MODELS_PER_RUNTIME, remaining)]
        total += len(bounded)
        inventories.append((runtime, bounded))
    return inventories


def _select_tool_use_candidate(inventories, metadata_client, deadline, clock=time.monotonic):
    for runtime, model_ids in inventories:
        for model_id in model_ids:
            if clock() >= deadline:
                return None
            if _has_tool_use_name_hint(model_id):
                candidate = _candidate(
                    model_id,
                    {
                        "tool_use": True,
                        "tool_use_src": "name",
                        "tool_use_confidence": "weak",
                    },
                )
                if candidate is not None:
                    return runtime, candidate

    inspected = set()
    for runtime, model_ids in inventories:
        for model_id in model_ids:
            if len(inspected) >= MAX_METADATA_INSPECTIONS:
                return None
            repository = _huggingface_repository_id(model_id)
            if repository is None or repository in inspected:
                continue
            remaining = deadline - clock()
            if remaining <= 0:
                return None
            inspected.add(repository)
            try:
                metadata = metadata_client.inspect_model_metadata(
                    repository,
                    timeout=min(HF_METADATA_TIMEOUT_SECONDS, remaining),
                )
            except Exception:
                continue
            if metadata.get("metadata_available") is not True:
                continue
            candidate = _candidate(model_id, metadata)
            if candidate is not None:
                return runtime, candidate
    return None


def _is_successful_verification_smoke(evidence):
    if evidence.status == VerificationStatus.VERIFIED:
        return True
    if evidence.status != VerificationStatus.FAILED:
        return False
    outcome = evidence.details.get("outcome")
    return (
        evidence.available_locally is True
        and evidence.loads is True
        and evidence.details.get("probe_id") == TOOL_USE_PROBE_ID
        and isinstance(outcome, dict)
        and set(outcome) == _NORMALIZED_OUTCOME_FIELDS
        and outcome.get("valid") is False
        and isinstance(outcome.get("reason"), str)
    )


class LiveToolUseHelperTests(unittest.TestCase):
    def test_auto_selection_uses_only_direct_local_runtime_clients(self):
        self.assertEqual(
            [runtime.name for runtime in _direct_local_runtime_clients()],
            ["ollama", "lmstudio", "mlx_lm"],
        )

    def test_common_runtime_identifiers_preserve_only_safe_hf_repository_ids(self):
        cases = (
            ("owner/repo", "owner/repo"),
            ("owner/repo:q4_K_M", "owner/repo"),
            ("hf.co/owner/repo:q4_K_M", "owner/repo"),
            ("qwen3:8b", None),
            ("/models/owner/repo", None),
            ("models/Qwen3-8B", None),
            ("./owner/repo", None),
            ("../owner/repo", None),
            ("owner/repo/weights", None),
        )
        for model_id, expected in cases:
            with self.subTest(model_id=model_id):
                self.assertEqual(_huggingface_repository_id(model_id), expected)

    def test_narrow_name_hints_cover_qwen3_and_local_paths(self):
        cases = (
            ("qwen3:8b", True),
            ("/models/Qwen3-8B-Instruct", True),
            ("models/Qwen3-8B-Instruct", True),
            ("owner/Assistant-Tool-Calling-7B", True),
            ("/models/ordinary-assistant", False),
            ("owner/Functionary-7B", False),
        )
        for model_id, expected in cases:
            with self.subTest(model_id=model_id):
                self.assertEqual(_has_tool_use_name_hint(model_id), expected)

    def test_inventory_is_bounded_per_runtime_and_globally(self):
        class Runtime:
            def __init__(self, name):
                self.name = name

            def list_models(self):
                return ["{0}/model-{1}".format(self.name, index) for index in range(20)]

        inventories = _inventory_direct_runtimes(
            [Runtime("one"), Runtime("two"), Runtime("three")],
            deadline=10.0,
            clock=lambda: 0.0,
        )

        self.assertLessEqual(
            sum(len(model_ids) for _runtime, model_ids in inventories),
            MAX_INSTALLED_MODELS_GLOBAL,
        )
        self.assertTrue(
            all(
                len(model_ids) <= MAX_INSTALLED_MODELS_PER_RUNTIME
                for _runtime, model_ids in inventories
            )
        )

    def test_name_hint_precedes_remote_metadata_inspection(self):
        class Metadata:
            def __init__(self):
                self.calls = []

            def inspect_model_metadata(self, repository, timeout):
                self.calls.append((repository, timeout))
                raise AssertionError("name hint must avoid metadata lookup")

        runtime = object()
        metadata = Metadata()
        selected = _select_tool_use_candidate(
            [(runtime, ["owner/plain-model", "qwen3:8b"])],
            metadata,
            deadline=10.0,
            clock=lambda: 0.0,
        )

        self.assertIs(selected[0], runtime)
        self.assertEqual(selected[1]["repo"], "qwen3:8b")
        self.assertEqual(metadata.calls, [])

    def test_metadata_normalization_preserves_runtime_id_for_probe(self):
        class Metadata:
            def __init__(self):
                self.calls = []

            def inspect_model_metadata(self, repository, timeout):
                self.calls.append((repository, timeout))
                return {
                    "metadata_available": True,
                    "tool_use": True,
                    "tool_use_src": "chat_template",
                    "tool_use_confidence": "explicit",
                }

        runtime = object()
        metadata = Metadata()
        selected = _select_tool_use_candidate(
            [(runtime, ["hf.co/owner/repo:q4_K_M"])],
            metadata,
            deadline=10.0,
            clock=lambda: 0.0,
        )

        self.assertEqual(metadata.calls[0][0], "owner/repo")
        self.assertEqual(selected[1]["repo"], "hf.co/owner/repo:q4_K_M")

    def test_remote_metadata_inspection_is_capped_and_deadline_aware(self):
        class Metadata:
            def __init__(self):
                self.calls = []

            def inspect_model_metadata(self, repository, timeout):
                self.calls.append((repository, timeout))
                return {"metadata_available": False}

        runtime = object()
        metadata = Metadata()
        model_ids = ["owner/model-{0}".format(index) for index in range(8)]
        selected = _select_tool_use_candidate(
            [(runtime, model_ids)],
            metadata,
            deadline=10.0,
            clock=lambda: 0.0,
        )

        self.assertIsNone(selected)
        self.assertEqual(len(metadata.calls), MAX_METADATA_INSPECTIONS)
        self.assertTrue(
            all(timeout <= HF_METADATA_TIMEOUT_SECONDS for _repo, timeout in metadata.calls)
        )

        expired = Metadata()
        self.assertIsNone(
            _select_tool_use_candidate(
                [(runtime, ["qwen3:8b"])],
                expired,
                deadline=1.0,
                clock=lambda: 1.0,
            )
        )
        self.assertEqual(expired.calls, [])

    def test_only_response_backed_failed_evidence_counts_as_successful_smoke(self):
        def evidence(status, loads, details, available=True):
            return VerificationEvidence(
                repo="local/model",
                role="tool-use",
                strength=EvidenceStrength.RUNTIME_TESTED,
                status=status,
                available_locally=available,
                loads=loads,
                reasoning_confirmed=None,
                runtime="direct",
                note="test",
                details=details,
            )

        invalid_response = evidence(
            VerificationStatus.FAILED,
            True,
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
        runtime_exception = evidence(
            VerificationStatus.FAILED,
            False,
            {"probe_id": TOOL_USE_PROBE_ID, "error": "RuntimeError"},
        )
        unsupported = evidence(
            VerificationStatus.UNSUPPORTED_RUNTIME,
            None,
            {
                "probe_id": TOOL_USE_PROBE_ID,
                "outcome": {"reason": "unsupported_runtime"},
            },
        )
        verified = evidence(
            VerificationStatus.VERIFIED,
            True,
            {"probe_id": TOOL_USE_PROBE_ID},
        )

        self.assertTrue(_is_successful_verification_smoke(verified))
        self.assertTrue(_is_successful_verification_smoke(invalid_response))
        self.assertFalse(_is_successful_verification_smoke(runtime_exception))
        self.assertFalse(_is_successful_verification_smoke(unsupported))


@unittest.skipUnless(
    os.environ.get("MLX_AGENT_LIVE_TOOL_USE") == "1",
    "set MLX_AGENT_LIVE_TOOL_USE=1 to probe installed local models",
)
class LiveToolUseTests(unittest.TestCase):
    def test_first_installed_tool_use_candidate_exercises_verification(self):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            self.skipTest("live tool-use smoke requires Apple Silicon")

        deadline = time.monotonic() + LIVE_SELECTION_TIMEOUT_SECONDS
        inventories = _inventory_direct_runtimes(
            _direct_local_runtime_clients(),
            deadline,
        )

        if not inventories:
            self.skipTest("no direct local loopback runtime is running")

        selected = _select_tool_use_candidate(
            inventories,
            HuggingFaceClient(),
            deadline,
        )

        if selected is None:
            self.skipTest(
                "selection deadline expired or no installed direct-local model "
                "has tool-use metadata or a narrow name signal"
            )

        runtime, candidate = selected
        evidence = Verifier(runtime_clients=[runtime]).verify(
            candidate,
            {},
            allow_network=False,
        )

        self.assertTrue(
            _is_successful_verification_smoke(evidence),
            "tool-use smoke requires verified evidence or a normalized invalid "
            "tool call from an actual local response: {0}".format(evidence.status.value),
        )
        self.assertEqual(evidence.details["probe_id"], TOOL_USE_PROBE_ID)

        serialized = json.dumps(evidence.to_dict(), sort_keys=True).lower()
        self.assertNotIn(TOOL_USE_PROMPT.lower(), serialized)
        for forbidden in (
            '"prompt"',
            '"response"',
            '"endpoint"',
            "127.0.0.1",
            "localhost",
            "/api/chat",
            "/v1/chat/completions",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
