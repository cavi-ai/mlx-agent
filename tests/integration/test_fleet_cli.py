import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.cli import main


class FleetCliTests(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_render_outputs_router_yaml(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "router.yaml"
            code, output = self._run([
                "fleet", "render", "--path", str(target),
                "--assign", "coding=pub/coder", "--assign", "vision=pub/see",
                "--allow-missing", "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            config = payload["data"]["config"]
            self.assertIn("model_name: coding", config)
            self.assertIn("model_name: vision", config)
            self.assertIn("8083", config)
            warning_codes = [item["code"] for item in payload["warnings"]]
            self.assertIn("model_not_local", warning_codes)

    def test_render_fails_for_missing_models_without_flag(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "router.yaml"
            code, output = self._run([
                "fleet", "render", "--path", str(target),
                "--assign", "coding=pub/definitely-missing-model",
                "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "model_not_local")

    def test_assign_and_from_adoption_are_exclusive(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "router.yaml"
            code, output = self._run([
                "fleet", "render", "--path", str(target),
                "--assign", "coding=pub/coder",
                "--from-adoption", str(Path(directory) / "state.json"),
                "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "invalid_arguments")

    def test_apply_requires_confirmation_then_applies(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "router.yaml"
            argv = [
                "fleet", "apply", "--path", str(target),
                "--assign", "coding=pub/coder", "--allow-missing",
                "--receipts-dir", str(Path(directory) / "receipts"), "--json",
            ]
            code, output = self._run(argv)
            self.assertEqual(code, 2)
            preview = json.loads(output)["data"]["preview"]
            self.assertFalse(target.exists())

            code, output = self._run(argv + ["--confirm"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output)["error"]["code"], "preview_hash_required")

            code, output = self._run(argv + [
                "--confirm", "--preview-hash", preview["preview_hash"],
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["data"]["receipt"]["status"], "applied")
            content = target.read_text()
            self.assertIn("model_name: coding", content)

    def test_from_adoption_consumes_recommendations(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text(json.dumps({
                "recommendations": [{"role": "coding", "repo": "pub/coder"}]
            }))
            target = Path(directory) / "router.yaml"
            code, output = self._run([
                "fleet", "render", "--path", str(target),
                "--from-adoption", str(state), "--allow-missing", "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["data"]["assignments"], {"coding": "pub/coder"})


if __name__ == "__main__":
    unittest.main()
