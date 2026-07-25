import unittest

from mlx_agent.adoption import (
    ADOPTION_SCHEMA_VERSION,
    MEASURE_LIMIT,
    AdoptionWorkflow,
    _migrate_state,
)
from mlx_agent.verification import EvidenceStrength, VerificationStatus


class FakeStreamingClient:
    name = "mlx_lm"

    def __init__(self, installed=("pub/model",), fail=False):
        self.installed = list(installed)
        self.fail = fail

    def list_models(self):
        if self.fail:
            raise RuntimeError("inventory down")
        return list(self.installed)

    def stream_generate(self, model, prompt, max_tokens, timeout=120.0):
        for _ in range(8):
            yield {"choices": [{"delta": {"content": "tok"}}]}
        yield {
            "message": {"content": ""},
            "eval_count": 64,
            "eval_duration": int(2e9),
            "prompt_eval_count": 96,
            "prompt_eval_duration": int(1e9),
        }


class FakeVerifier:
    def __init__(self, clients):
        self._clients = list(clients)

    def runtime_clients(self):
        return tuple(self._clients)


def _state(request_measure, shortlist, evidence):
    class State:
        pass

    state = State()
    state.request = {"measure": request_measure}
    state.host = {"chip": "Apple M5 Max"}
    state.shortlist = shortlist
    state.evidence = evidence
    state.warnings = []
    return state


def _verified_evidence(repo, runtime="mlx_lm"):
    return {
        "repo": repo,
        "role": "coding",
        "strength": EvidenceStrength.RUNTIME_TESTED.value,
        "status": VerificationStatus.VERIFIED.value,
        "available_locally": True,
        "loads": True,
        "reasoning_confirmed": None,
        "runtime": runtime,
        "note": "ok",
        "details": {"probes": [{"probe_id": "coding-v1", "outcome": {"valid": True, "reason": "valid"}}]},
    }


class MeasurePhaseTests(unittest.TestCase):
    def test_measure_off_is_a_noop(self):
        evidence = [_verified_evidence("pub/model")]
        state = _state(False, [{"repo": "pub/model", "role": "coding"}], evidence)
        workflow = AdoptionWorkflow(verifier=FakeVerifier([FakeStreamingClient()]))
        workflow._phase_measure(state)
        self.assertEqual(
            state.evidence[0]["strength"], EvidenceStrength.RUNTIME_TESTED.value
        )

    def test_verified_candidate_is_upgraded_to_runtime_measured(self):
        evidence = [_verified_evidence("pub/model")]
        state = _state(True, [{"repo": "pub/model", "role": "coding"}], evidence)
        workflow = AdoptionWorkflow(verifier=FakeVerifier([FakeStreamingClient()]))
        workflow._phase_measure(state)
        upgraded = state.evidence[0]
        self.assertEqual(upgraded["strength"], "runtime_measured")
        self.assertEqual(upgraded["details"]["bench"]["decode_toks"], 32.0)
        self.assertEqual(
            upgraded["details"]["probes"][0]["probe_id"], "coding-v1"
        )
        self.assertEqual(upgraded["details"]["bench"]["chip"], "Apple M5 Max")

    def test_bench_failure_warns_and_preserves_evidence(self):
        evidence = [_verified_evidence("pub/model")]
        state = _state(True, [{"repo": "pub/model", "role": "coding"}], evidence)
        workflow = AdoptionWorkflow(
            verifier=FakeVerifier([FakeStreamingClient(installed=[])])
        )
        workflow._phase_measure(state)
        self.assertEqual(
            state.evidence[0]["strength"], EvidenceStrength.RUNTIME_TESTED.value
        )
        self.assertEqual(state.warnings[0]["code"], "measure_skipped")

    def test_non_verified_evidence_is_skipped(self):
        evidence = _verified_evidence("pub/model")
        evidence["status"] = VerificationStatus.FAILED.value
        state = _state(True, [{"repo": "pub/model", "role": "coding"}], [evidence])
        workflow = AdoptionWorkflow(verifier=FakeVerifier([FakeStreamingClient()]))
        workflow._phase_measure(state)
        self.assertEqual(
            state.evidence[0]["strength"], EvidenceStrength.RUNTIME_TESTED.value
        )
        self.assertEqual(state.warnings, [])

    def test_measure_limit_bounds_measurements(self):
        count = MEASURE_LIMIT + 2
        shortlist = [
            {"repo": "pub/model", "role": "coding"} for _ in range(count)
        ]
        evidence = [_verified_evidence("pub/model") for _ in range(count)]
        state = _state(True, shortlist, evidence)
        workflow = AdoptionWorkflow(verifier=FakeVerifier([FakeStreamingClient()]))
        workflow._phase_measure(state)
        upgraded = [
            item for item in state.evidence
            if item["strength"] == "runtime_measured"
        ]
        self.assertEqual(len(upgraded), MEASURE_LIMIT)


class MeasureMigrationTests(unittest.TestCase):
    def test_1_2_state_gains_measure_phase_and_request_flag(self):
        legacy = {
            "schema_version": "1.2",
            "request": {"roles": ["general"], "fast": False},
            "completed_phases": ["inspect", "discover", "shortlist", "verify", "compare"],
            "evidence": [],
        }
        migrated = _migrate_state(legacy)
        self.assertEqual(migrated["schema_version"], ADOPTION_SCHEMA_VERSION)
        self.assertEqual(migrated["request"]["measure"], False)
        self.assertEqual(
            migrated["completed_phases"],
            ["inspect", "discover", "shortlist", "verify", "measure", "compare"],
        )

    def test_1_1_state_migrates_through_to_current(self):
        legacy = {
            "schema_version": "1.1",
            "request": {"roles": ["general"]},
            "completed_phases": ["inspect", "discover", "shortlist", "verify"],
            "evidence": [],
        }
        migrated = _migrate_state(legacy)
        self.assertEqual(migrated["schema_version"], ADOPTION_SCHEMA_VERSION)
        self.assertIn("measure", migrated["completed_phases"])

    def test_current_state_is_untouched(self):
        current = {"schema_version": ADOPTION_SCHEMA_VERSION}
        self.assertIs(_migrate_state(current), current)

    def test_unsupported_version_is_rejected(self):
        with self.assertRaises(ValueError):
            _migrate_state({"schema_version": "9.9"})


if __name__ == "__main__":
    unittest.main()
