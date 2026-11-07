"""Exercise launch, cross-directory discovery and stop through the shipped CLI."""
import ast
import json
import os
import socket
import signal
import time
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

CLI = Path(__file__).resolve().parents[2] / 'scripts/mlx-agent'
SERVER = '''#!/usr/bin/env python3
import argparse, json, os, sys, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import TCPServer
parser = argparse.ArgumentParser()
parser.add_argument('--model'); parser.add_argument('--port', type=int)
parser.add_argument('--fixture-not-ready', action='store_true')
args, _ = parser.parse_known_args()
marker = Path(args.model).parent / 'fixture-worker.json'
temporary = marker.with_suffix('.tmp')
temporary.write_text(json.dumps({'pid': os.getpid(), 'argv': sys.argv, 'model': args.model, 'port': args.port}))
os.replace(temporary, marker)
if args.fixture_not_ready:
    while True: time.sleep(1)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({'data': [{'id': args.model}]}).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *args): pass
class Server(HTTPServer):
    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
Server(('127.0.0.1', args.port), Handler).serve_forever()
'''


def stop_fixture(marker, executable, model, port):
    """Only the private fixture process is eligible, including before a serve receipt exists."""
    from mlx_agent.serve import _argv_matches, _pid_command
    if not marker.is_file():
        return
    record = json.loads(marker.read_text())
    pid, argv = record['pid'], record['argv']
    if (type(pid) is not int or pid <= 0 or not argv
            or Path(argv[0]).resolve() != executable.resolve()
            or Path(record['model']).resolve() != model.resolve() or record['port'] != port):
        raise AssertionError('Fixture process ownership changed')
    expected = {'argv': argv, 'path': record['model'], 'port': port}
    try:
        if os.getpgid(pid) == pid and _argv_matches(expected, _pid_command(pid)):
            os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


class ServiceDiscoveryCliTests(unittest.TestCase):
    def test_fixture_listener_does_not_resolve_loopback_hostname(self):
        module = ast.parse(SERVER)
        definitions = [node for node in module.body
                       if isinstance(node, (ast.Import, ast.ImportFrom))
                       or isinstance(node, ast.ClassDef) and node.name == 'Server']
        namespace = {}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<service fixture>', 'exec'), namespace)
        with mock.patch('socket.getfqdn', side_effect=AssertionError('Loopback startup must not query DNS')):
            with namespace['Server'](('127.0.0.1', 0), namespace['BaseHTTPRequestHandler']) as server:
                self.assertEqual(server.server_name, '127.0.0.1')
                self.assertEqual(server.server_port, server.socket.getsockname()[1])

    def test_fixture_cleanup_works_before_readiness_and_serve_receipt(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / 'model'; model.mkdir()
            executable = root / 'fixture-server.py'; executable.write_text(SERVER)
            marker = root / 'fixture-worker.json'
            process = subprocess.Popen([sys.executable, str(executable), '--model', str(model),
                '--port', '12345', '--fixture-not-ready'], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.is_file())
                self.assertIsNone(process.poll())
                original = marker.read_text()
                changed = json.loads(original); changed['port'] = 12346
                marker.write_text(json.dumps(changed))
                with self.assertRaises(AssertionError):
                    stop_fixture(marker, executable, model, 12345)
                self.assertIsNone(process.poll())
                marker.write_text(original)
                stop_fixture(marker, executable, model, 12345)
                process.wait(timeout=5)
                self.assertEqual(process.returncode, -signal.SIGTERM)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)

    def test_launch_is_discovered_from_another_directory_and_stop_checks_selected_pid(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / 'model'; model.mkdir()
            (model / 'config.json').write_text('{"model_type":"qwen2"}')
            (model / 'model.safetensors').write_bytes(b'fixture weights; not loaded')
            bin_dir = root / 'bin'; bin_dir.mkdir()
            executable = bin_dir / 'mlx_lm.server'
            executable.write_text(SERVER); executable.chmod(0o755)
            receipts = root / 'service'; receipts.mkdir()
            elsewhere = root / 'elsewhere'; elsewhere.mkdir()
            environment = dict(os.environ, XDG_STATE_HOME=str(root / 'state'),
                               PATH=str(bin_dir) + os.pathsep + str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
            def run(arguments, cwd=elsewhere):
                result = subprocess.run([sys.executable, str(CLI), *arguments, '--json'],
                                        cwd=cwd, env=environment, capture_output=True, text=True, timeout=25)
                self.assertTrue(result.stdout, result.stderr)
                return result.returncode, json.loads(result.stdout)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
            start = ['serve', 'start', '--path', str(model), '--runtime', 'mlx_lm', '--port', str(port), '--receipts-dir', str(receipts)]
            pid = None
            receipt = receipts / ".mlx-agent-receipts" / "serve" / (str(port) + ".json")
            try:
                code, preview = run(start)
                self.assertEqual(code, 2)
                code, launched = run(start + ['--confirm', '--preview-hash', preview['data']['plan']['preview_hash']])
                self.assertEqual(code, 0, launched)
                pid = launched['data']['receipt']['pid']
                code, found = run(['serve', 'discover'])
                self.assertEqual(code, 0, found)
                matching = [value for value in found['data']['services'] if value['pid'] == pid]
                self.assertEqual(len(matching), 1)
                self.assertEqual(Path(matching[0]['receipts_dir']), receipts.resolve())
                self.assertTrue(matching[0]['alive'])
                self.assertTrue(matching[0]['argv_match'])
                self.assertTrue(matching[0]['process_started_at'])
                code, replaced = run(['serve', 'stop', '--port', str(port), '--expected-pid', str(pid), '--expected-start', 'different process start', '--receipts-dir', str(receipts)])
                self.assertEqual(code, 2)
                self.assertEqual(replaced['error']['code'], 'server_changed')
                code, refused = run(['serve', 'stop', '--port', str(port), '--expected-pid', str(pid + 1), '--receipts-dir', str(receipts)])
                self.assertEqual(code, 2)
                self.assertEqual(refused['error']['code'], 'server_changed')
            finally:
                if pid is None and receipt.is_file():
                    pid = json.loads(receipt.read_text())['pid']
                try:
                    if pid is not None:
                        code, stopped = run(['serve', 'stop', '--port', str(port), '--expected-pid', str(pid), '--receipts-dir', str(receipts)])
                        self.assertEqual(code, 0, stopped)
                finally:
                    stop_fixture(root / 'fixture-worker.json', executable, model, port)
