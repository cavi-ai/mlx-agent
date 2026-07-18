import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from mlx_agent.adoption import (
    PHASES,
    AdoptionRequest,
    AdoptionState,
    AdoptionStateConflictError,
    AdoptionWorkflow,
)
from mlx_agent.cli import main
from mlx_agent.contracts import ResultEnvelope
from mlx_agent.host import HostInventory
from mlx_agent.verification import EvidenceStrength, VerificationEvidence
from mlx_agent.transactions import _target_locks


def candidate(repo, role, reasoning=False, rank_score=1):
    return {
        "repo": repo,
        "role": role,
        "reasoning": reasoning,
        "reason_src": "name",
        "rank_score": rank_score,
        "fits": True,
        "trusted": True,
        "est_ram_gb": 4.0,
        "wiring": "mlx_lm.server --model {0}".format(repo),
    }


class FakeDiscoveryService:
    def __init__(self, ram_gb, candidates):
        self.host = HostInventory(ram_gb=ram_gb, chip="Test Apple", ollama=True)
        self.candidates = candidates
        self.requests = []

    def discover(self, request):
        self.requests.append(request)
        roles = {}
        for item in self.candidates:
            roles.setdefault(item["role"], []).append(dict(item))
        return ResultEnvelope.ok(
            "discover",
            {"host": self.host.to_dict(), "fast": False, "roles": roles},
        )


class TrackingVerifier:
    def __init__(self, reasoning_repos=()):
        self.reasoning_repos = set(reasoning_repos)
        self.active = 0
        self.maximum_active = 0
        self.calls = []
        self.lock = threading.Lock()

    def verify(self, item, host, allow_network=True):
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.calls.append((item["repo"], allow_network))
        time.sleep(0.025)
        with self.lock:
            self.active -= 1
        reasoning = item["repo"] in self.reasoning_repos
        return VerificationEvidence(
            repo=item["repo"],
            role=item["role"],
            strength=EvidenceStrength.RUNTIME_TESTED,
            available_locally=True,
            loads=True,
            reasoning_confirmed=reasoning,
            runtime="fake-runtime",
            note="runtime generation completed",
        )


class MetadataReasoningVerifier:
    def verify(self, item, host, allow_network=True):
        del host, allow_network
        return VerificationEvidence(
            repo=item["repo"],
            role=item["role"],
            strength=EvidenceStrength.METADATA_ONLY,
            available_locally=False,
            loads=None,
            reasoning_confirmed=True,
            runtime=None,
            note="Repository metadata tag confirms reasoning behavior.",
            details={"reasoning_evidence": "metadata_tags"},
        )


class AdoptionWorkflowTests(unittest.TestCase):
    def _advance_to(self, workflow, state, phase):
        while state.phase != phase:
            state = workflow.advance(state)
        return state

    def test_saved_state_resumes_at_first_incomplete_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adoption.json"
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(
                    32, [candidate("local/normal", "general")]
                ),
                verifier=TrackingVerifier(),
            )
            state = workflow.start(
                AdoptionRequest(roles=("general",), state_path=path, allow_network=False)
            )
            state = self._advance_to(workflow, state, "verify")

            saved = json.loads(path.read_text())
            self.assertEqual(saved["phase"], "verify")
            self.assertEqual(saved["completed_phases"], ["inspect", "discover", "shortlist"])
            resumed = workflow.resume(path)
            self.assertEqual(resumed.phase, "verify")
            self.assertEqual(resumed.completed_phases, ["inspect", "discover", "shortlist"])

    def test_state_revision_uses_compare_and_swap_for_concurrent_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adoption.json"
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(32, []),
                verifier=TrackingVerifier(),
            )
            started = workflow.start(
                AdoptionRequest(roles=("general",), state_path=path)
            )
            self.assertEqual(1, started.revision)
            first = workflow.resume(path)
            stale = workflow.resume(path)
            workflow.advance(first)
            durable = path.read_bytes()
            self.assertEqual(2, json.loads(durable)["revision"])

            with self.assertRaises(AdoptionStateConflictError):
                workflow.advance(stale)
            self.assertEqual(durable, path.read_bytes())

    def test_state_lock_contention_fails_without_mutating_the_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adoption.json"
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(32, []),
                verifier=TrackingVerifier(),
            )
            workflow.start(AdoptionRequest(roles=("general",), state_path=path))
            state = workflow.resume(path)
            durable = path.read_bytes()
            with _target_locks([path]):
                with self.assertRaises(AdoptionStateConflictError):
                    workflow.advance(state)
            self.assertEqual(durable, path.read_bytes())

    def test_state_paths_never_follow_leaf_or_ancestor_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(32, []),
                verifier=TrackingVerifier(),
            )
            referent = root / "referent.json"
            referent.write_text("external bytes\n")
            leaf = root / "state.json"
            leaf.symlink_to(referent)
            with self.assertRaises(ValueError):
                workflow.resume(leaf)
            with self.assertRaises((ValueError, AdoptionStateConflictError)):
                workflow.start(AdoptionRequest(roles=("general",), state_path=leaf))

            outside = root / "outside"
            outside.mkdir()
            ancestor = root / "linked"
            ancestor.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                workflow.start(AdoptionRequest(
                    roles=("general",), state_path=ancestor / "state.json"
                ))
            self.assertEqual("external bytes\n", referent.read_text())
            self.assertEqual([], list(outside.iterdir()))

    def test_request_and_saved_state_reject_unbounded_or_secret_fields(self):
        self.assertEqual(AdoptionRequest(roles="general").roles, ("general",))
        with self.assertRaises(ValueError):
            AdoptionRequest(roles=("general", "coding", "reasoning", "vision", "embedding", "extra"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(16, []), verifier=TrackingVerifier()
            )
            state = workflow.start(AdoptionRequest(roles=("general",), state_path=path))
            value = state.to_dict()
            value["request"]["api_key"] = "must-not-persist"
            with self.assertRaises(ValueError):
                AdoptionState.from_dict(value, path)

    def test_verification_concurrency_is_bounded_by_host_ram(self):
        for ram_gb, expected in ((8, 1), (32, 2), (128, 4)):
            with self.subTest(ram_gb=ram_gb), tempfile.TemporaryDirectory() as directory:
                records = [candidate("local/model-{0}".format(index), "coding") for index in range(8)]
                verifier = TrackingVerifier()
                workflow = AdoptionWorkflow(
                    discovery_service=FakeDiscoveryService(ram_gb, records),
                    verifier=verifier,
                )
                state = workflow.start(
                    AdoptionRequest(
                        roles=("coding",),
                        state_path=Path(directory) / "state.json",
                        shortlist_limit=8,
                    )
                )
                state = self._advance_to(workflow, state, "verify")
                state = workflow.advance(state)

                self.assertEqual(state.phase, "compare")
                self.assertEqual(verifier.maximum_active, expected)
                self.assertEqual(len(state.evidence), 8)

    def test_per_candidate_failure_does_not_abort_verify_phase(self):
        class FailingVerifier(TrackingVerifier):
            def verify(self, item, host, allow_network=True):
                if item["repo"].endswith("broken"):
                    raise RuntimeError("candidate exploded")
                return super().verify(item, host, allow_network)

        with tempfile.TemporaryDirectory() as directory:
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(
                    32,
                    [candidate("local/good", "coding"), candidate("local/broken", "coding")],
                ),
                verifier=FailingVerifier(),
            )
            state = workflow.start(
                AdoptionRequest(roles=("coding",), state_path=Path(directory) / "state.json")
            )
            state = self._advance_to(workflow, state, "verify")
            state = workflow.advance(state)

            self.assertEqual(state.phase, "compare")
            self.assertEqual(len(state.evidence), 2)
            broken = next(item for item in state.evidence if item["repo"] == "local/broken")
            self.assertEqual(broken["strength"], "heuristic_only")
            self.assertIn("candidate exploded", broken["note"])

    def test_recommendation_rejects_confirmed_reasoner_for_general_role(self):
        with tempfile.TemporaryDirectory() as directory:
            reasoner = candidate("local/reasoner", "general", reasoning=True, rank_score=100)
            normal = candidate("local/normal", "general", reasoning=False, rank_score=10)
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(64, [reasoner, normal]),
                verifier=TrackingVerifier(reasoning_repos={"local/reasoner"}),
            )
            state = workflow.start(
                AdoptionRequest(roles=("general",), state_path=Path(directory) / "state.json")
            )
            state = self._advance_to(workflow, state, "complete")

            self.assertEqual(state.status, "complete")
            self.assertEqual(state.recommendations[0]["repo"], "local/normal")
            rejected = next(item for item in state.comparisons if item["repo"] == "local/reasoner")
            self.assertFalse(rejected["eligible"])
            self.assertIn("confirmed_reasoner_for_utility_role", rejected["rejection_reasons"])
            self.assertEqual(tuple(PHASES), (
                "inspect", "discover", "shortlist", "verify", "compare", "recommend", "complete"
            ))

    def test_recommendation_rejects_metadata_confirmed_reasoner_for_fast_role(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(
                    64, [candidate("remote/reasoner", "coding", reasoning=True, rank_score=100)]
                ),
                verifier=MetadataReasoningVerifier(),
            )
            state = workflow.start(
                AdoptionRequest(
                    roles=("coding",),
                    state_path=Path(directory) / "state.json",
                    fast=True,
                )
            )
            state = self._advance_to(workflow, state, "complete")

            self.assertEqual(state.recommendations, [])
            rejected = state.comparisons[0]
            self.assertFalse(rejected["eligible"])
            self.assertIn("confirmed_reasoner_for_utility_role", rejected["rejection_reasons"])

    def test_resume_rejects_noncontiguous_duplicate_and_phase_mismatched_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(32, [candidate("local/normal", "general")]),
                verifier=TrackingVerifier(),
            )
            state = workflow.start(AdoptionRequest(roles=("general",), state_path=path))
            state = self._advance_to(workflow, state, "verify")
            saved = json.loads(path.read_text())

            for completed, phase, status in (
                (["inspect", "shortlist"], "verify", "running"),
                (["inspect", "inspect"], "discover", "running"),
                (["inspect"], "verify", "running"),
                (list(PHASES[:-1]), "complete", "running"),
                (list(PHASES), "complete", "complete"),
            ):
                with self.subTest(completed=completed, phase=phase, status=status):
                    corrupt = dict(saved)
                    corrupt["completed_phases"] = completed
                    corrupt["phase"] = phase
                    corrupt["status"] = status
                    path.write_text(json.dumps(corrupt))
                    with self.assertRaises(ValueError):
                        workflow.resume(path)

    def test_resume_rejects_state_that_claims_verification_without_current_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(32, [candidate("local/normal", "general")]),
                verifier=TrackingVerifier(),
            )
            state = workflow.start(AdoptionRequest(roles=("general",), state_path=path))
            state = self._advance_to(workflow, state, "verify")
            saved = json.loads(path.read_text())
            saved["completed_phases"].append("verify")
            saved["phase"] = "compare"
            saved["evidence"] = []
            path.write_text(json.dumps(saved))

            with self.assertRaises(ValueError):
                workflow.resume(path)

    def test_resume_rejects_invalid_nested_schema_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(32, []), verifier=TrackingVerifier()
            )
            state = workflow.start(AdoptionRequest(roles=("general",), state_path=path))
            saved = state.to_dict()

            invalid_values = []
            invalid_evidence = dict(saved)
            invalid_evidence["evidence"] = [{
                "repo": "local/model", "role": "general", "strength": "invalid",
                "available_locally": True, "loads": True, "reasoning_confirmed": False,
                "runtime": "fake", "note": "x", "details": {},
            }]
            invalid_values.append(invalid_evidence)
            missing_evidence_field = dict(saved)
            missing_evidence_field["evidence"] = [{"repo": "local/model"}]
            invalid_values.append(missing_evidence_field)
            invalid_timestamp = dict(saved)
            invalid_timestamp["created_at"] = "not-a-timestamp"
            invalid_values.append(invalid_timestamp)
            too_many_warnings = dict(saved)
            too_many_warnings["warnings"] = [{}] * 51
            invalid_values.append(too_many_warnings)
            empty_roles = dict(saved)
            empty_roles["request"] = dict(saved["request"])
            empty_roles["request"]["roles"] = []
            invalid_values.append(empty_roles)

            for corrupt in invalid_values:
                with self.subTest(corrupt=corrupt):
                    path.write_text(json.dumps(corrupt))
                    with self.assertRaises((TypeError, ValueError)):
                        workflow.resume(path)

    def test_schema_and_cli_status_expose_bounded_structured_state(self):
        root = Path(__file__).resolve().parents[2]
        schema = json.loads((root / "schemas" / "adoption-state.schema.json").read_text())
        self.assertEqual(schema["properties"]["phase"]["enum"], list(PHASES))
        self.assertFalse(schema["additionalProperties"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            workflow = AdoptionWorkflow(
                discovery_service=FakeDiscoveryService(16, []), verifier=TrackingVerifier()
            )
            workflow.start(AdoptionRequest(roles=("general",), state_path=path))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["adopt", "status", "--state", str(path), "--json"])
            envelope = json.loads(output.getvalue())

            self.assertEqual(code, 0)
            self.assertEqual(envelope["operation"], "adopt-status")
            self.assertEqual(envelope["data"]["state"]["phase"], "inspect")

    def test_cli_start_and_resume_complete_with_fixture_backed_evidence(self):
        root = Path(__file__).resolve().parents[2]
        fixture = root / "tests" / "fixtures" / "scout_responses.json"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            original_fixture = os.environ.get("MLX_AGENT_FIXTURE")
            os.environ["MLX_AGENT_FIXTURE"] = str(fixture)
            try:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    start_code = main([
                        "adopt", "start", "--state", str(path), "--role", "general",
                        "--shortlist-limit", "1", "--json",
                    ])
                started = json.loads(output.getvalue())
                self.assertEqual(start_code, 0)
                self.assertEqual(started["operation"], "adopt-start")
                self.assertEqual(started["data"]["state"]["phase"], "complete")
                self.assertEqual(len(started["data"]["state"]["evidence"]), 1)
                self.assertEqual(
                    started["data"]["state"]["evidence"][0]["strength"],
                    "metadata_only",
                )

                saved = json.loads(path.read_text())
                saved["phase"] = "compare"
                saved["status"] = "running"
                saved["completed_phases"] = ["inspect", "discover", "shortlist", "verify"]
                saved["comparisons"] = []
                saved["recommendations"] = []
                path.write_text(json.dumps(saved))

                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    resume_code = main(["adopt", "resume", "--state", str(path), "--json"])
                resumed = json.loads(output.getvalue())
                self.assertEqual(resume_code, 0)
                self.assertEqual(resumed["operation"], "adopt-resume")
                self.assertEqual(resumed["data"]["state"]["phase"], "complete")
            finally:
                if original_fixture is None:
                    os.environ.pop("MLX_AGENT_FIXTURE", None)
                else:
                    os.environ["MLX_AGENT_FIXTURE"] = original_fixture

    def test_cli_invalid_fixture_uses_adopt_operation_name(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "invalid-fixture.json"
            fixture.write_text("[]")
            state = Path(directory) / "state.json"
            original_fixture = os.environ.get("MLX_AGENT_FIXTURE")
            os.environ["MLX_AGENT_FIXTURE"] = str(fixture)
            try:
                for command in ("start", "resume"):
                    with self.subTest(command=command):
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            code = main(["adopt", command, "--state", str(state), "--json"])
                        envelope = json.loads(output.getvalue())
                        self.assertEqual(code, 2)
                        self.assertEqual(envelope["operation"], "adopt-{0}".format(command))
                        self.assertEqual(envelope["error"]["code"], "invalid_fixture")
            finally:
                if original_fixture is None:
                    os.environ.pop("MLX_AGENT_FIXTURE", None)
                else:
                    os.environ["MLX_AGENT_FIXTURE"] = original_fixture


if __name__ == "__main__":
    unittest.main()
