import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main


class DoctorModelsCliTests(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_doctor_models_json_envelope(self):
        with TemporaryDirectory() as project:
            code, output = self._run([
                "doctor", "models",
                "--project", project,
                "--wired-root", project,
                "--hf-cache", str(Path(project) / "hf"),
                "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["operation"], "doctor-models")
            self.assertEqual(payload["status"], "ok")
            self.assertIn("inventory", payload["data"])
            self.assertIn("findings", payload["data"])
            self.assertEqual(payload["data"]["summary"]["wired_configs"], 0)

    def test_doctor_models_reports_wired_drift(self):
        with TemporaryDirectory() as project:
            root = Path(project)
            (root / "wired.json").write_text(json.dumps({
                "mlx_agent_wire": {
                    "marker": "MLX_AGENT_WIRE",
                    "runtime": "mlx_lm",
                    "model": "pub/gone",
                    "provider": {"base_url": "http://127.0.0.1:8080/v1"},
                }
            }))
            code, output = self._run([
                "doctor", "models",
                "--project", project,
                "--wired-root", project,
                "--hf-cache", str(root / "hf"),
                "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            codes = [finding["code"] for finding in payload["data"]["findings"]]
            self.assertIn("drift_missing_model", codes)
            warning_codes = [warning["code"] for warning in payload["warnings"]]
            self.assertIn("drift_missing_model", warning_codes)

    def test_provider_doctor_still_works(self):
        code, output = self._run(["providers", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
