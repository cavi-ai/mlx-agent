import contextlib
import io
import unittest
from unittest.mock import patch

from mlx_agent.contracts import ResultEnvelope
from mlx_agent.cli import main
from mlx_agent.host import HostInventory
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

    def test_factory_output_conforms_to_result_contract(self):
        success = ResultEnvelope.ok(
            "inspect-host",
            {"chip": "Apple M4"},
            warnings=[{"code": "cached", "message": "Using cached data."}],
        ).to_dict()
        failure = ResultEnvelope.fail(
            "discover",
            "network_unavailable",
            "HF unavailable",
            "Retry with --offline to use the last cache.",
            retryable=True,
        ).to_dict()
        self.assertEqual(validate_result(success), [])
        self.assertEqual(validate_result(failure), [])

    def test_success_factory_rejects_schema_invalid_shapes(self):
        with self.assertRaises(TypeError):
            ResultEnvelope.ok("inspect-host", [], warnings=[])
        with self.assertRaises(TypeError):
            ResultEnvelope.ok("inspect-host", {}, warnings="cached")
        with self.assertRaises(TypeError):
            ResultEnvelope.ok("inspect-host", {}, warnings=["cached"])
        with self.assertRaises(TypeError):
            ResultEnvelope.ok("inspect-host", {}, warnings=[{"code": 1}])
        with self.assertRaises(ValueError):
            ResultEnvelope.ok("", {})

    def test_error_factory_rejects_schema_invalid_shapes(self):
        with self.assertRaises(ValueError):
            ResultEnvelope.fail("discover", "", "HF unavailable", "Retry later.")
        with self.assertRaises(ValueError):
            ResultEnvelope.fail("discover", "network_unavailable", "", "Retry later.")
        with self.assertRaises(TypeError):
            ResultEnvelope.fail(
                "discover", "network_unavailable", "HF unavailable", "Retry later.", "yes"
            )

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

    def test_result_validator_enforces_schema_only_constraints(self):
        invalid = {
            "schema_version": "1.0",
            "generated_at": "not-a-date",
            "operation": "",
            "status": "error",
            "data": {},
            "warnings": [{"code": "cached", "attempts": 1}],
            "error": {
                "code": "",
                "message": "HF unavailable",
                "remediation": "Retry later.",
                "retryable": True,
                "unexpected": "value",
            },
            "unexpected": True,
        }
        errors = validate_result(invalid)
        self.assertIn("result.generated_at must be an ISO-8601 date-time", errors)
        self.assertIn("result.operation must not be empty", errors)
        self.assertIn("warnings[0] must be an object of strings", errors)
        self.assertIn("error.code must not be empty", errors)
        self.assertIn("error has unexpected keys: ['unexpected']", errors)
        self.assertIn("result has unexpected keys: ['unexpected']", errors)

    def test_result_validator_rejects_error_for_success_and_non_object_warning(self):
        value = ResultEnvelope.ok("discover", {}).to_dict()
        value["error"] = {}
        value["warnings"] = ["cached"]
        errors = validate_result(value)
        self.assertIn("error is only allowed when status is 'error'", errors)
        self.assertIn("warnings[0] must be an object of strings", errors)

    def test_inspect_host_cli_returns_versioned_inventory_and_classified_probe_warnings(self):
        output = io.StringIO()
        warnings = [{"code": "runtime_probe_unavailable", "probe": "ollama", "message": "Local runtime probe unavailable."}]
        with patch.object(HostInventory, "inspect", return_value=(HostInventory(ram_gb=32, chip="Apple Test"), warnings)), contextlib.redirect_stdout(output):
            self.assertEqual(0, main(["inspect-host", "--json"]))
        payload = __import__("json").loads(output.getvalue())
        self.assertEqual("inspect-host", payload["operation"])
        self.assertEqual({"ram_gb": 32, "chip": "Apple Test", "ollama": False, "lmstudio": False}, payload["data"])
        self.assertEqual(warnings, payload["warnings"])
        self.assertEqual([], validate_result(payload))
