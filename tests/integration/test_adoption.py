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
    AdoptionWorkflow,
)
from mlx_agent.cli import main
from mlx_agent.contracts import ResultEnvelope
from mlx_agent.host import HostInventory
from mlx_agent.verification import EvidenceStrength, VerificationEvidence


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


if __name__ == "__main__":
    unittest.main()
