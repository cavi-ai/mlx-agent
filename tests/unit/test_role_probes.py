import unittest

from mlx_agent.adoption import (
    ROLE_PROBE_BONUS,
    AdoptionState,
    AdoptionWorkflow,
    EVIDENCE_SCORES,
)
from mlx_agent.verification import (
    CODING_PROBE_ID,
    EMBEDDING_PROBE_ID,
    REASONING_PROBE_ID,
    ROLE_PROBE_IDS,
    VISION_PROBE_ID,
    EvidenceStrength,
    VerificationStatus,
    Verifier,
    validate_coding_response,
    validate_embedding_response,
    validate_reasoning_response,
    validate_vision_response,
)


class FakeRoleRuntimeClient:
    name = "fake-role-runtime"

    def __init__(self, installed, generate_response=None, embed_response=None,
                 vision_response=None, generate_error=None):
        self.installed = list(installed)
        self.generate_response = generate_response or {"message": {"content": "ready"}}
        self.embed_response = embed_response
        self.vision_response = vision_response
        self.generate_error = generate_error
        self.prompts = []

    def list_models(self):
        return list(self.installed)

    def generate(self, model, prompt, max_tokens):
        self.prompts.append(prompt)
        if self.generate_error is not None and prompt != "Reply with the single word ready.":
            raise self.generate_error
        return self.generate_response


class FakeEmbeddingRuntimeClient(FakeRoleRuntimeClient):
    def embed(self, model, inputs):
        return self.embed_response


class FakeVisionRuntimeClient(FakeRoleRuntimeClient):
    name = "mlx-vlm"

    def generate_vision(self, model, prompt, image_base64, max_tokens):
        return self.vision_response


class CodingProbeValidatorTests(unittest.TestCase):
    def test_plain_function_is_valid(self):
        response = {"message": {"content": "def add(a, b):\n    return a + b"}}
        self.assertEqual(validate_coding_response(response)["reason"], "valid")

    def test_fenced_function_is_valid(self):
        response = {"message": {"content": "```python\ndef add(a, b):\n    return a + b\n```"}}
        self.assertEqual(validate_coding_response(response)["reason"], "valid")

    def test_prose_only_reports_missing_content(self):
        response = {"message": {"content": "Sure! Here is how addition works."}}
        self.assertEqual(validate_coding_response(response)["reason"], "parse_failure")

    def test_empty_response_reports_missing_content(self):
        response = {"message": {"content": ""}}
        self.assertEqual(validate_coding_response(response)["reason"], "missing_content")

    def test_wrong_function_name_reports_missing_function(self):
        response = {"message": {"content": "def sum_two(a, b):\n    return a + b"}}
        self.assertEqual(validate_coding_response(response)["reason"], "missing_function")

    def test_wrong_answer_is_detected(self):
        response = {"message": {"content": "def add(a, b):\n    return a * b"}}
        self.assertEqual(validate_coding_response(response)["reason"], "wrong_answer")

    def test_unsafe_content_is_rejected_before_exec(self):
        response = {"message": {"content": "def add(a, b):\n    import os\n    return a + b"}}
        self.assertEqual(validate_coding_response(response)["reason"], "unsafe_content")

    def test_loop_content_is_rejected_before_exec(self):
        response = {"message": {"content": "def add(a, b):\n    while True:\n        pass"}}
        self.assertEqual(validate_coding_response(response)["reason"], "unsafe_content")

    def test_non_dict_response_is_not_valid(self):
        self.assertFalse(validate_coding_response(42)["valid"])


class ReasoningProbeValidatorTests(unittest.TestCase):
    def test_exact_answer_is_valid(self):
        self.assertEqual(
            validate_reasoning_response({"message": {"content": "9"}})["reason"], "valid"
        )

    def test_answer_with_text_is_valid(self):
        response = {"message": {"content": "The farmer has 9 sheep left."}}
        self.assertEqual(validate_reasoning_response(response)["reason"], "valid")

    def test_wrong_answer_is_detected(self):
        response = {"message": {"content": "8"}}
        self.assertEqual(validate_reasoning_response(response)["reason"], "wrong_answer")

    def test_empty_response_reports_missing_content(self):
        response = {"message": {"content": "no idea"}}
        self.assertEqual(validate_reasoning_response(response)["reason"], "missing_content")


class VisionProbeValidatorTests(unittest.TestCase):
    def test_expected_text_is_valid(self):
        response = {"message": {"content": "MLX42"}}
        self.assertEqual(validate_vision_response(response)["reason"], "valid")

    def test_case_and_whitespace_insensitive(self):
        response = {"message": {"content": "m l x 4 2"}}
        self.assertEqual(validate_vision_response(response)["reason"], "valid")

    def test_wrong_text_is_detected(self):
        response = {"message": {"content": "HELLO"}}
        self.assertEqual(validate_vision_response(response)["reason"], "wrong_answer")


class EmbeddingProbeValidatorTests(unittest.TestCase):
    def test_correct_ordering_is_valid(self):
        response = {"embeddings": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]}
        self.assertEqual(validate_embedding_response(response)["reason"], "valid")

    def test_inverted_ordering_is_detected(self):
        response = {"embeddings": [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]}
        self.assertEqual(validate_embedding_response(response)["reason"], "order_inverted")

    def test_openai_shape_is_supported(self):
        response = {
            "data": [
                {"index": 2, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 1, "embedding": [0.9, 0.1]},
            ]
        }
        self.assertEqual(validate_embedding_response(response)["reason"], "valid")

    def test_zero_norm_vector_is_invalid(self):
        response = {"embeddings": [[0.0, 0.0], [0.9, 0.1], [0.0, 1.0]]}
        self.assertEqual(validate_embedding_response(response)["reason"], "invalid_response")

    def test_wrong_vector_count_is_invalid(self):
        response = {"embeddings": [[1.0, 0.0], [0.9, 0.1]]}
        self.assertEqual(validate_embedding_response(response)["reason"], "invalid_response")


class VerifierRoleProbeTests(unittest.TestCase):
    def test_coding_probe_attached_to_evidence(self):
        client = FakeRoleRuntimeClient(
            ["pub/model"],
            generate_response={"message": {"content": "def add(a, b):\n    return a + b"}},
        )
        verifier = Verifier(runtime_clients=[client])
        evidence = verifier.verify({"repo": "pub/model", "role": "coding"}, {})
        self.assertEqual(evidence.status, VerificationStatus.VERIFIED)
        probes = evidence.details["probes"]
        self.assertEqual(probes[0]["probe_id"], CODING_PROBE_ID)
        self.assertTrue(probes[0]["outcome"]["valid"])

    def test_failing_coding_probe_is_recorded(self):
        client = FakeRoleRuntimeClient(
            ["pub/model"],
            generate_response={"message": {"content": "def add(a, b):\n    return a - b"}},
        )
        verifier = Verifier(runtime_clients=[client])
        evidence = verifier.verify({"repo": "pub/model", "role": "coding"}, {})
        self.assertFalse(evidence.details["probes"][0]["outcome"]["valid"])
        self.assertEqual(
            evidence.details["probes"][0]["outcome"]["reason"], "wrong_answer"
        )

    def test_probe_exception_is_bounded(self):
        client = FakeRoleRuntimeClient(
            ["pub/model"], generate_error=RuntimeError("boom")
        )
        verifier = Verifier(runtime_clients=[client])
        evidence = verifier.verify({"repo": "pub/model", "role": "reasoning"}, {})
        self.assertEqual(
            evidence.details["probes"][0]["outcome"]["reason"], "probe_error"
        )
        self.assertEqual(evidence.status, VerificationStatus.VERIFIED)

    def test_embedding_without_support_is_unsupported(self):
        client = FakeRoleRuntimeClient(["pub/model"])
        verifier = Verifier(runtime_clients=[client])
        evidence = verifier.verify({"repo": "pub/model", "role": "embedding"}, {})
        self.assertEqual(
            evidence.details["probes"][0]["outcome"]["reason"], "unsupported_runtime"
        )

    def test_embedding_probe_runs_when_supported(self):
        client = FakeEmbeddingRuntimeClient(
            ["pub/model"],
            embed_response={"embeddings": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]},
        )
        verifier = Verifier(runtime_clients=[client])
        evidence = verifier.verify({"repo": "pub/model", "role": "embedding"}, {})
        self.assertEqual(evidence.details["probes"][0]["probe_id"], EMBEDDING_PROBE_ID)
        self.assertTrue(evidence.details["probes"][0]["outcome"]["valid"])

    def test_vision_probe_uses_vision_client(self):
        client = FakeVisionRuntimeClient(
            ["pub/model"], vision_response={"message": {"content": "MLX42"}}
        )
        verifier = Verifier(runtime_clients=[client])
        evidence = verifier.verify({"repo": "pub/model", "role": "vision"}, {})
        self.assertEqual(evidence.details["probes"][0]["probe_id"], VISION_PROBE_ID)
        self.assertTrue(evidence.details["probes"][0]["outcome"]["valid"])

    def test_general_role_has_no_probe(self):
        client = FakeRoleRuntimeClient(["pub/model"])
        verifier = Verifier(runtime_clients=[client])
        evidence = verifier.verify({"repo": "pub/model", "role": "general"}, {})
        self.assertNotIn("probes", evidence.details)

    def test_probe_ids_cover_expected_roles(self):
        self.assertEqual(
            set(ROLE_PROBE_IDS), {"coding", "reasoning", "vision", "embedding"}
        )
        self.assertEqual(ROLE_PROBE_IDS["reasoning"], REASONING_PROBE_ID)


class _WorkflowStateFixture:
    """Minimal state object for direct compare-phase testing."""

    def __init__(self, shortlist, evidence):
        self.request = {"roles": [item["role"] for item in shortlist], "fast": False}
        self.shortlist = shortlist
        self.evidence = evidence
        self.comparisons = []


def _compare(shortlist, evidence):
    workflow = AdoptionWorkflow(verifier=Verifier(runtime_clients=[]))
    state = _WorkflowStateFixture(shortlist, evidence)
    workflow._phase_compare(state)
    return state.comparisons


def _evidence(repo, role, probes=None):
    details = {"probes": probes} if probes is not None else {}
    return {
        "repo": repo,
        "role": role,
        "strength": EvidenceStrength.RUNTIME_TESTED.value,
        "status": VerificationStatus.VERIFIED.value,
        "available_locally": True,
        "loads": True,
        "reasoning_confirmed": None,
        "runtime": "fake",
        "note": "ok",
        "details": details,
    }


class CompareProbeScoringTests(unittest.TestCase):
    def test_valid_probe_adds_bonus(self):
        shortlist = [{"repo": "pub/a", "role": "coding", "rank_score": 0}]
        evidence = [_evidence("pub/a", "coding", [
            {"probe_id": CODING_PROBE_ID, "outcome": {"valid": True, "reason": "valid"}}
        ])]
        comparison = _compare(shortlist, evidence)[0]
        self.assertEqual(
            comparison["score"],
            EVIDENCE_SCORES[EvidenceStrength.RUNTIME_TESTED.value] + ROLE_PROBE_BONUS,
        )
        self.assertTrue(comparison["eligible"])

    def test_failed_probe_rejects_candidate(self):
        shortlist = [{"repo": "pub/a", "role": "coding", "rank_score": 0}]
        evidence = [_evidence("pub/a", "coding", [
            {"probe_id": CODING_PROBE_ID, "outcome": {"valid": False, "reason": "wrong_answer"}}
        ])]
        comparison = _compare(shortlist, evidence)[0]
        self.assertFalse(comparison["eligible"])
        self.assertIn("role_probe_failed", comparison["rejection_reasons"])

    def test_unsupported_probe_neither_bonuses_nor_rejects(self):
        shortlist = [{"repo": "pub/a", "role": "embedding", "rank_score": 0}]
        evidence = [_evidence("pub/a", "embedding", [
            {"probe_id": EMBEDDING_PROBE_ID, "outcome": {"valid": False, "reason": "unsupported_runtime"}}
        ])]
        comparison = _compare(shortlist, evidence)[0]
        self.assertTrue(comparison["eligible"])
        self.assertEqual(
            comparison["score"], EVIDENCE_SCORES[EvidenceStrength.RUNTIME_TESTED.value]
        )

    def test_runtime_measured_strength_outranks_runtime_tested(self):
        self.assertGreater(
            EVIDENCE_SCORES["runtime_measured"],
            EVIDENCE_SCORES[EvidenceStrength.RUNTIME_TESTED.value],
        )


if __name__ == "__main__":
    unittest.main()
