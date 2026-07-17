import unittest

from mlx_agent.contracts import ResultEnvelope
from scripts.validate_contracts import validate_result


class ResultEnvelopeTests(unittest.TestCase):
    def test_success_envelope_is_versioned(self):
        value = ResultEnvelope.ok("inspect-host", {"chip": "Apple M4"}).to_dict()
        self.assertEqual(value["schema_version"], "1.0")
        self.assertEqual(value["operation"], "inspect-host")
        self.assertEqual(value["status"], "ok")
        self.assertEqual(value["data"]["chip"], "Apple M4")
        self.assertEqual(value["warnings"], [])

    def test_error_envelope_exposes_remediation(self):
        value = ResultEnvelope.fail(
            "discover", "network_unavailable", "HF unavailable",
            "Retry with --offline to use the last cache.", retryable=True,
        ).to_dict()
        self.assertEqual(value["status"], "error")
        self.assertTrue(value["error"]["retryable"])
        self.assertIn("--offline", value["error"]["remediation"])

    def test_result_validator_rejects_error_without_remediation(self):
        errors = validate_result(
            {
                "schema_version": "1.0",
                "operation": "discover",
                "status": "error",
                "data": {},
                "warnings": [],
                "error": {
                    "code": "network_unavailable",
                    "message": "HF unavailable",
                    "retryable": True,
                },
                "generated_at": "2026-07-17T00:00:00+00:00",
            }
        )
        self.assertIn("error.remediation must be a string", errors)
