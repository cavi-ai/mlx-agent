import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mlx_agent.cli import main


class ServeCliTests(unittest.TestCase):
    def test_memory_policy_is_bound_to_preview_and_rejected_for_eager_serving(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / 'model.safetensors').write_bytes(b'weights')
            args = ['serve', 'start', '--path', directory, '--runtime', 'mlx_lm', '--json']
            code, output = self._run(args + ['--idle-timeout', '600'])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output)['error']['code'], 'jit_required')
            code, output = self._run(args + ['--jit', '--idle-timeout', '600', '--min-headroom-gb', '2'])
            plan = json.loads(output)['data']['plan']
            self.assertEqual(code, 2)
            self.assertEqual(plan['memory_policy'], {'idle_timeout_seconds': 600, 'keep_loaded': False, 'minimum_headroom_gb': 2})
            _, other = self._run(args + ['--jit', '--idle-timeout', '300', '--min-headroom-gb', '2'])
            self.assertNotEqual(plan['preview_hash'], json.loads(other)['data']['plan']['preview_hash'])
            code, output = self._run(args + ['--jit', '--min-headroom-gb', 'nan'])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output)['error']['code'], 'invalid_arguments')

    def test_policy_command_passes_full_policy_and_requires_expected_pid(self):
        with mock.patch('mlx_agent.cli.configure_serve_memory', return_value={'status': 'configured'}) as configure:
            code, _ = self._run(['serve', 'policy', '--port', '8080', '--expected-pid', '123', '--idle-timeout', '300', '--keep-loaded', '--json'])
            self.assertEqual(code, 0)
            configure.assert_called_once_with(8080, {'idle_timeout_seconds': 300, 'keep_loaded': True, 'minimum_headroom_gb': None}, None, 123)
        with self.assertRaises(SystemExit):
            self._run(['serve', 'policy', '--port', '8080', '--json'])

    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_start_preview_requires_confirmation(self):
        code, output = self._run([
            "serve", "start", "--repo", "pub/model", "--runtime", "mlx_lm",
        ])
        self.assertEqual(code, 2)
        self.assertIn("preview_hash", output)
        self.assertIn("Confirmation required", output)

    def test_start_preview_json(self):
        code, output = self._run([
            "serve", "start", "--repo", "pub/model", "--runtime", "mlx_lm", "--json",
        ])
        self.assertEqual(code, 2)
        payload = json.loads(output)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["data"]["requires_confirmation"])
        self.assertEqual(payload["data"]["plan"]["port"], 8080)

    def test_confirm_without_hash_is_rejected(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "serve", "start", "--repo", "pub/model", "--runtime", "mlx_lm",
                "--confirm", "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "preview_hash_required")

    def test_status_empty(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "serve", "status", "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["data"]["servers"], [])

    def test_stop_without_receipt(self):
        with TemporaryDirectory() as directory:
            code, output = self._run([
                "serve", "stop", "--port", "8080",
                "--receipts-dir", directory, "--json",
            ])
            self.assertEqual(code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["error"]["code"], "receipt_not_found")

    def test_unsupported_runtime_choices_are_rejected(self):
        with self.assertRaises(SystemExit):
            self._run(["serve", "start", "--repo", "pub/model", "--runtime", "ollama"])


if __name__ == "__main__":
    unittest.main()
