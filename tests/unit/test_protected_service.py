import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from mlx_agent.cli import main


class ProtectedServiceTests(unittest.TestCase):
    def test_foreground_capability_is_metadata_only(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['serve', 'protected-service', '--describe']), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value['protocol'], 'protected-service-v1')
        self.assertEqual(len(value['source_sha256']), 64)

    def test_service_requires_a_private_key_and_local_model(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['serve', 'protected-service', '--model', '/missing/model'])

    def test_foreground_exit_closes_worker_and_restores_signal_handlers(self):
        import signal
        from mlx_agent.protected_service import main as service_main
        for failure in (False, True):
            with self.subTest(failure=failure), TemporaryDirectory() as folder:
                root = Path(folder)
                model = root / 'model'
                model.mkdir()
                (model / 'config.json').write_text('{}')
                (model / 'model.safetensors').write_bytes(b'fixture')
                key = root / 'key'
                key.write_text('a' * 43)
                key.chmod(0o600)
                server = Mock()
                if failure:
                    server.serve_forever.side_effect = RuntimeError('service interrupted')
                handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
                def construct(config, port):
                    self.assertEqual(port, 19001)
                    self.assertEqual(config['private_worker']['protocol'], 'inherited-unix-v1')
                    self.assertEqual(Path(config['state_path']).parent, Path(config['state_path']).parent.resolve())
                    self.assertNotEqual(config['control_token'], key.read_text())
                    return server
                with patch('mlx_agent.jit_serve.make_server', side_effect=construct):
                    args = ['--model', str(model), '--api-key-file', str(key), '--port', '19001']
                    if failure:
                        with self.assertRaises(RuntimeError):
                            service_main(args)
                    else:
                        self.assertEqual(service_main(args), 0)
                server.controller.close.assert_called_once()
                server.server_close.assert_called_once()
                self.assertEqual(handlers, {sig: signal.getsignal(sig) for sig in handlers})
