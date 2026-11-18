import hashlib
import http.client
import json
import threading
import unittest
from unittest import mock

from mlx_agent import tailnet


BINDING = {"host": "node.example.ts.net", "https_port": 8443, "endpoint_port": 4321,
           "model_id": "model", "api_key_sha256": hashlib.sha256(b"secret").hexdigest()}


def session(port=8443, proxy=12345):
    authority = "node.example.ts.net:{}".format(port)
    return {"TCP": {str(port): {"HTTPS": True}},
            "Web": {authority: {"Handlers": {"/": {"Proxy": "http://127.0.0.1:{}".format(proxy)}}}}}


def request(server, method, path, headers, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    result = response.status, response.read()
    connection.close()
    return result


class TailnetTests(unittest.TestCase):

    def test_conflicts_and_owned_session(self):
        assert tailnet.occupied({"Foreground": {"one": session()}}, 8443)
        assert tailnet.occupied({"AllowFunnel": {"node.example.ts.net:8443": True}}, 8443)
        assert not tailnet.occupied({"Foreground": {"one": session(443)}}, 8443)
        assert tailnet.matching_session({"Foreground": {"one": session()}}, BINDING, 12345)
        assert not tailnet.matching_session({"Foreground": {"one": session(), "two": session()}}, BINDING, 12345)
        assert not tailnet.matching_session({"Foreground": {"one": dict(session(), AllowFunnel={"other:8443": True})}}, BINDING, 12345)

    def test_preview_is_bound_and_read_only(self):
        endpoint = dict(BINDING, pid=1, process_started_at="birth", receipt="/receipt")
        with mock.patch.object(tailnet, "select_endpoint", return_value=endpoint), \
             mock.patch.object(tailnet, "executable", return_value="tailscale"), \
             mock.patch.object(tailnet, "qualified_version"), \
             mock.patch.object(tailnet, "node_status", return_value=BINDING["host"]), \
             mock.patch.object(tailnet, "read_json", return_value={}):
            first, _, _ = tailnet.preview(4321)
            second, _, _ = tailnet.preview(4321, 9443)
        assert first["state"] == "ready"
        assert first["preview_hash"] != second["preview_hash"]
        assert first["api_base"] == "https://node.example.ts.net:8443/v1"

    def test_relay_authority_auth_routes_and_model(self):
        server = tailnet.RelayServer(BINDING)
        server.admitted.set()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        good = {"Host": "node.example.ts.net:8443", "Authorization": "Bearer secret"}
        try:
            with mock.patch.object(tailnet, "endpoint_matches", return_value=True):
                assert request(server, "GET", "/v1/models", good)[0] == 200
                assert request(server, "GET", "/health", good)[0] == 404
                assert request(server, "GET", "/v1/models", dict(good, Host="wrong"))[0] == 403
                assert request(server, "GET", "/v1/models", dict(good, Origin="https://other"))[0] == 403
                assert request(server, "GET", "/v1/models", {"Host": good["Host"]})[0] == 401
                assert request(server, "POST", "/v1/chat/completions", good, json.dumps({"model": "other"}))[0] == 400
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_streaming_connection_close_and_busy(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b'data: {"model":"/Users/franco/private","choices":[{"delta":{"content":"first"}}]}\n\n')
                self.wfile.flush()
                self.wfile.write(b'data: {"model":"/Users/franco/private","choices":[{"delta":{"content":"second"}}]}\n\n')
                self.wfile.flush()
                self.close_connection = True

        upstream = HTTPServer(("127.0.0.1", 0), Upstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        server = tailnet.RelayServer(dict(BINDING, endpoint_port=upstream.server_port))
        server.admitted.set()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        good = {"Host": "node.example.ts.net:8443", "Authorization": "Bearer secret"}
        payload = json.dumps({"model": "model", "stream": True})
        try:
            with mock.patch.object(tailnet, "endpoint_matches", return_value=True):
                status, body = request(server, "POST", "/v1/chat/completions", good, payload)
                assert status == 200
                events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b'data: ')]
                assert [event['choices'][0]['delta']['content'] for event in events] == ['first', 'second']
                assert [event['model'] for event in events] == ['model', 'model']
                assert server.generation.acquire(blocking=False)
                try:
                    assert request(server, "POST", "/v1/chat/completions", good, payload)[0] == 429
                finally:
                    server.generation.release()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join()

    def test_stale_cleaned_status_and_session_fence(self):
        record = {"public": {"share_id": "one"}}
        with mock.patch.object(tailnet, "read_state", return_value=record), \
             mock.patch.object(tailnet, "owner_matches", return_value=False), \
             mock.patch.object(tailnet, "stale_cleaned", return_value=True):
            assert tailnet.status() == {"state": "off"}
        with mock.patch.object(tailnet, "read_state", return_value=record):
            try:
                tailnet.stop("another")
            except tailnet.TailnetError as error:
                assert error.code == "tailnet_session_changed"
            else:
                raise AssertionError("replacement share accepted")

    def test_default_https_port_host(self):
        server = tailnet.RelayServer(dict(BINDING, https_port=443))
        server.admitted.set()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with mock.patch.object(tailnet, "endpoint_matches", return_value=True):
                for host in ("node.example.ts.net", "node.example.ts.net:443"):
                    assert request(server, "GET", "/v1/models", {"Host": host, "Authorization": "Bearer secret"})[0] == 200
                assert request(server, "GET", "/v1/models", {"Host": "node.example.ts.net:8443", "Authorization": "Bearer secret"})[0] == 403
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_tasks_endpoint_is_ineligible(self):
        entry = dict(BINDING, port=4321, pid=9, alive=True, argv_match=True, jit=True,
                     runtime="tasks", receipt="/x", process_started_at="birth")
        receipt = {"process_started_at": "birth"}
        with mock.patch.object(tailnet, "discover_services", return_value={"services": [entry]}), \
             mock.patch.object(tailnet, "_read_receipt", return_value=receipt):
            try:
                tailnet.select_endpoint(4321)
            except tailnet.TailnetError as error:
                assert error.code == "tailnet_endpoint_ineligible"
            else:
                raise AssertionError("task endpoint accepted")

    def test_guard_import_without_caller_pythonpath(self):
        import os
        import socket
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "tailscale"
            marker = Path(tmp) / "started"
            binary.write_text("#!/bin/sh\necho started > " + str(marker) + "\nexec /bin/sleep 30\n")
            binary.chmod(0o700)
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            env = dict(os.environ, PYTHONPATH=str(Path(tailnet.__file__).resolve().parent.parent))
            try:
                child = subprocess.Popen([sys.executable, "-m", "mlx_agent.tailnet", "--guard", str(binary),
                                          "8443", str(listener.getsockname()[1]), str(listener.fileno())],
                                         cwd=tmp, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, pass_fds=(listener.fileno(),))
                identity = json.loads(child.stdout.readline())
                assert identity["pid"] > 1 and identity["birth"] and identity["argv"]
                child.stdin.close()
                child.wait(timeout=10)
                assert child.returncode != 0
                assert not marker.exists()
            finally:
                listener.close()
                if child.poll() is None:
                    child.kill()
                    child.wait()
                child.stdout.close()
                child.stderr.close()

    def test_cli_preview_does_not_start(self):
        import contextlib
        import io
        from mlx_agent.cli import main

        public = {"state": "ready", "api_base": "https://node.example.ts.net:8443/v1", "model_id": "model",
                  "endpoint_port": 4321, "https_port": 8443, "preview_hash": "abc"}
        output = io.StringIO()
        with mock.patch.object(tailnet, "preview", return_value=(public, BINDING, "tailscale")), \
             mock.patch.object(tailnet, "run_session") as run, contextlib.redirect_stdout(output):
            assert main(["serve", "tailnet", "start", "--port", "4321", "--json"]) == 0
            run.assert_not_called()
        assert json.loads(output.getvalue())["data"] == public

    def test_cli_confirm_hash_drift(self):
        import contextlib
        import io
        from mlx_agent.cli import main

        output = io.StringIO()
        with mock.patch.object(tailnet, "preview", return_value=({"preview_hash": "new"}, BINDING, "tailscale")), \
             mock.patch.object(tailnet, "run_session") as run, contextlib.redirect_stdout(output):
            assert main(["serve", "tailnet", "start", "--port", "4321", "--json", "--confirm", "--preview-hash", "old"]) == 2
            run.assert_not_called()
        assert json.loads(output.getvalue())["error"]["code"] == "tailnet_preview_changed"

    def test_guard_release_and_cleanup(self):
        import os
        import socket
        import subprocess
        import sys
        import tempfile
        import time
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "started"
            binary = Path(tmp) / "tailscale"
            binary.write_text("#!/bin/sh\necho started > " + str(marker) + "\nexec /bin/sleep 30\n")
            binary.chmod(0o700)
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            env = dict(os.environ, PYTHONPATH=str(Path(tailnet.__file__).resolve().parent.parent))
            child = subprocess.Popen([sys.executable, "-m", "mlx_agent.tailnet", "--guard", str(binary),
                                      "8443", str(listener.getsockname()[1]), str(listener.fileno())],
                                     cwd=tmp, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, pass_fds=(listener.fileno(),))
            try:
                identity = json.loads(child.stdout.readline())
                assert identity["pid"] > 1 and identity["birth"] and identity["argv"]
                child.stdin.write(b"1")
                child.stdin.flush()
                for _ in range(100):
                    if marker.exists():
                        break
                    time.sleep(0.05)
                assert marker.exists()
                child.stdin.close()
                child.wait(timeout=10)
                assert child.returncode != 0
            finally:
                listener.close()
                if child.poll() is None:
                    child.kill()
                    child.wait()
                child.stdout.close()
                child.stderr.close()

    def test_native_binary_resolution_rejects_shell_shims(self):
        import os
        import struct
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.write_bytes(b"\xcf\xfa\xed\xfe" + struct.pack("<III", 0, 0, 2))
            native.chmod(0o700)
            alias = root / "tailscale"
            alias.symlink_to(native)
            shim = root / "shim"
            shim.write_text("#!/bin/sh\nexit 0\n")
            shim.chmod(0o700)
            assert tailnet.native_executable(alias) == str(native.resolve())
            assert tailnet.native_executable(shim) is None
            assert tailnet.native_executable(root / "missing") is None

    def test_cli_normalizes_expected_share_id(self):
        import contextlib
        import io
        from mlx_agent.cli import main

        value = "ABCDEF12-3456-4789-ABCD-1234567890AB"
        with mock.patch.object(tailnet, "stop", return_value={"state": "off"}) as stop, \
             contextlib.redirect_stdout(io.StringIO()):
            assert main(["serve", "tailnet", "stop", "--expected-share-id", value, "--json"]) == 0
        stop.assert_called_once_with(value.lower())

    def test_relay_port_reserved_during_cleanup_readback(self):
        import errno
        import os
        from pathlib import Path
        import socket
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "tailscale"
            binary.write_text("#!/bin/sh\nexec /bin/sleep 30\n")
            binary.chmod(0o700)
            observed = []

            def readback(config, binding, port):
                probe = socket.socket()
                try:
                    try:
                        probe.bind(("127.0.0.1", port))
                    except OSError as error:
                        assert error.errno == errno.EADDRINUSE
                        observed.append(port)
                    else:
                        raise AssertionError("relay port was released before readback")
                finally:
                    probe.close()
                return False

            def stop_after_ready(data):
                os.kill(os.getpid(), __import__("signal").SIGTERM)

            binding = dict(BINDING)
            public = {"state": "ready", "preview_hash": "preview"}
            with mock.patch.object(tailnet, "state_root", return_value=root / "state"), \
                 mock.patch.object(tailnet, "endpoint_matches", return_value=True), \
                 mock.patch.object(tailnet, "node_status", return_value=binding["host"]), \
                 mock.patch.object(tailnet, "read_json", return_value={}), \
                 mock.patch.object(tailnet, "occupied", return_value=False), \
                 mock.patch.object(tailnet, "matching_session", return_value=True), \
                 mock.patch.object(tailnet, "owned_mapping_present", side_effect=readback):
                tailnet.run_session(public, binding, str(binary), stop_after_ready)
            assert len(observed) == 1
            probe = socket.socket()
            try:
                probe.bind(("127.0.0.1", observed[0]))
            finally:
                probe.close()

    def test_upstream_responses_hide_private_model_paths_and_errors(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                case = body.get('case')
                if case == 'error':
                    self.send_response(503)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"failed at /Users/franco/private"}}')
                    return
                self.send_response(200)
                self.send_header('Connection', 'close')
                if case in ('stream', 'bad-frame', 'stream-error'):
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    frame = {'model': '/Users/franco/private/snapshot', 'choices': [{'delta': {'content': 'hello'}}]}
                    self.wfile.write(b'data: ' + json.dumps(frame).encode() + b'\n\n')
                    if case == 'bad-frame':
                        self.wfile.write(b'data: broken /Users/franco/private\n\n')
                    elif case == 'stream-error':
                        self.wfile.write(b'data: {"error":{"message":"/Users/franco/private"}}\n\n')
                    else:
                        self.wfile.write(b'data: [DONE]\n\n')
                else:
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'model': '/Users/franco/private/snapshot', 'choices': [{'message': {'content': 'hello'}}]}).encode())
                self.close_connection = True

        upstream = HTTPServer(('127.0.0.1', 0), Upstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        server = tailnet.RelayServer(dict(BINDING, endpoint_port=upstream.server_port))
        server.admitted.set()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        headers = {'Host': 'node.example.ts.net:8443', 'Authorization': 'Bearer secret'}
        try:
            with mock.patch.object(tailnet, 'endpoint_matches', return_value=True):
                for case in ('json', 'stream', 'bad-frame', 'stream-error', 'error'):
                    status, raw = request(server, 'POST', '/v1/chat/completions', headers,
                                          json.dumps({'model': 'model', 'case': case, 'stream': case in ('stream', 'bad-frame', 'stream-error')}))
                    assert b'/Users/franco' not in raw
                    if case == 'error':
                        assert status == 503
                        assert b'failed at' not in raw
                    elif case == 'json':
                        assert status == 200 and json.loads(raw)['model'] == 'model'
                        assert json.loads(raw)['choices'][0]['message']['content'] == 'hello'
                    else:
                        assert status == 200
                        assert json.loads(raw.splitlines()[0][len(b'data: '):])['model'] == 'model'
                        if case == 'stream':
                            assert raw.endswith(b'data: [DONE]\n\n')
                        else:
                            assert b'broken' not in raw
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join()

    def test_endpoint_selection_requires_matching_authenticated_descriptor(self):
        endpoint = {'origin': 'http://127.0.0.1:4321', 'api_base': 'http://127.0.0.1:4321/v1',
                    'service_id': 'a' * 64, 'runtime': 'mlx_lm', 'state': 'ready', 'auth_required': True}
        entry = dict(BINDING, port=4321, pid=9, alive=True, argv_match=True, jit=True,
                     runtime='mlx_lm', receipt='/tmp/receipts/.mlx-agent-receipts/serve/4321.json',
                     receipts_dir='/tmp/receipts', process_started_at='birth', repo='model', path=None,
                     endpoint=endpoint, service_id='a' * 64)
        receipt = {'process_started_at': 'birth'}
        with mock.patch.object(tailnet, '_read_receipt', return_value=receipt):
            for state in ('ready', 'unknown'):
                selected = dict(entry, endpoint=dict(endpoint, state=state))
                with mock.patch.object(tailnet, 'discover_services', return_value={'services': [selected]}):
                    binding = tailnet.select_endpoint(4321)
                    assert binding['service_id'] == 'a' * 64
                    assert binding['receipts_dir'] == '/tmp/receipts' and binding['runtime'] == 'mlx_lm'
            invalid = [None, dict(endpoint, origin='http://localhost:4321'),
                       dict(endpoint, api_base='http://127.0.0.1:4322/v1'),
                       dict(endpoint, auth_required=False), dict(endpoint, service_id='z' * 64),
                       dict(endpoint, runtime='tasks'), dict(endpoint, state='starting')]
            for descriptor in invalid:
                selected = dict(entry, endpoint=descriptor)
                with mock.patch.object(tailnet, 'discover_services', return_value={'services': [selected]}):
                    try:
                        tailnet.select_endpoint(4321)
                    except tailnet.TailnetError as error:
                        assert error.code == 'tailnet_endpoint_ineligible'
                    else:
                        raise AssertionError('invalid endpoint descriptor accepted')

    def test_endpoint_generation_drift_invalidates_binding(self):
        binding = dict(BINDING, receipt='/tmp/receipt', receipts_dir='/tmp', pid=9,
                       process_started_at='birth', runtime='mlx_lm', service_id='a' * 64)
        receipt = {'pid': 9, 'port': 4321, 'process_started_at': 'birth',
                   'api_key_sha256': BINDING['api_key_sha256'], 'runtime': 'mlx_lm', 'jit': True}
        descriptor = {'origin': 'http://127.0.0.1:4321', 'api_base': 'http://127.0.0.1:4321/v1',
                      'service_id': 'a' * 64, 'runtime': 'mlx_lm', 'state': 'ready', 'auth_required': True}
        with mock.patch.object(tailnet, '_read_receipt', return_value=receipt), \
             mock.patch.object(tailnet, '_pid_start', return_value='birth'), \
             mock.patch.object(tailnet, '_pid_command', return_value='command'), \
             mock.patch.object(tailnet, '_argv_matches', return_value=True), \
             mock.patch('mlx_agent.model_serving.resolve_receipt_endpoint', return_value=descriptor) as resolve:
            assert tailnet.endpoint_matches(binding)
            resolve.return_value = dict(descriptor, service_id='b' * 64)
            assert not tailnet.endpoint_matches(binding)
            resolve.return_value = None
            assert not tailnet.endpoint_matches(binding)

    def test_tailscale_prerequisite_failures_are_specific(self):
        base = {'BackendState': 'Running', 'Version': '1.104.2-test',
                'CurrentTailnet': {'MagicDNSEnabled': True},
                'Self': {'DNSName': 'node.example.ts.net.'}, 'CertDomains': ['node.example.ts.net']}
        cases = [({'BackendState': 'Stopped'}, 'tailscale_disconnected'),
                 ({'Version': '1.100.0'}, 'tailnet_unavailable'),
                 ({'CurrentTailnet': {'MagicDNSEnabled': False}}, 'tailscale_dns_required'),
                 ({'Self': {'DNSName': 'bad host'}}, 'tailscale_dns_required'),
                 ({'CertDomains': []}, 'tailscale_https_required')]
        with mock.patch.object(tailnet, 'read_json') as read:
            read.return_value = base
            self.assertEqual(tailnet.node_status('tailscale'), 'node.example.ts.net')
            for changes, code in cases:
                with self.subTest(code=code):
                    read.return_value = dict(base, **changes)
                    with self.assertRaises(tailnet.TailnetError) as caught:
                        tailnet.node_status('tailscale')
                    self.assertEqual(caught.exception.code, code)

    def test_tailscale_version_and_status_json_are_bounded(self):
        import subprocess
        import sys
        from types import SimpleNamespace

        with mock.patch.object(sys, 'platform', 'darwin'), mock.patch.object(tailnet, 'read_json') as read:
            read.return_value = {'majorMinorPatch': '1.104.2'}
            tailnet.qualified_version('tailscale')
            for value in ('1.104.1', 'garbage'):
                read.return_value = {'majorMinorPatch': value}
                with self.assertRaises(tailnet.TailnetError):
                    tailnet.qualified_version('tailscale')
        with mock.patch.object(sys, 'platform', 'linux'):
            with self.assertRaises(tailnet.TailnetError):
                tailnet.qualified_version('tailscale')
        with mock.patch.object(tailnet.subprocess, 'run') as run:
            run.return_value = SimpleNamespace(stdout=b'{"ok":true}')
            self.assertEqual(tailnet.read_json('/native', 'status'), {'ok': True})
            for output in (b'[]', b'{broken', b'x' * 2_000_001):
                run.return_value = SimpleNamespace(stdout=output)
                with self.assertRaises(tailnet.TailnetError):
                    tailnet.read_json('/native', 'status')
            run.side_effect = subprocess.TimeoutExpired('tailscale', 5)
            with self.assertRaises(tailnet.TailnetError):
                tailnet.read_json('/native', 'status')

    def test_status_and_owned_stop_verify_live_mapping(self):
        import signal
        record = {'pid': 12345, 'binary': '/native', 'binding': BINDING,
                  'relay_port': 12345, 'public': {'state': 'sharing', 'share_id': 'one'}}
        with mock.patch.object(tailnet, 'read_state', return_value=record), \
             mock.patch.object(tailnet, 'owner_matches', return_value=True), \
             mock.patch.object(tailnet, 'endpoint_matches', return_value=True), \
             mock.patch.object(tailnet, 'node_status', return_value=BINDING['host']), \
             mock.patch.object(tailnet, 'read_json', return_value={'Foreground': {'one': session()}}):
            self.assertEqual(tailnet.status(), record['public'])
        with mock.patch.object(tailnet, 'read_state', return_value=record), \
             mock.patch.object(tailnet, 'owner_matches', side_effect=[True, False]), \
             mock.patch.object(tailnet, 'read_json', return_value={}), \
             mock.patch.object(tailnet.time, 'sleep'), mock.patch.object(tailnet.os, 'kill') as kill:
            self.assertEqual(tailnet.stop('one'), {'state': 'off'})
            kill.assert_called_once_with(record['pid'], signal.SIGTERM)
        with mock.patch.object(tailnet, 'read_state', return_value=record), \
             mock.patch.object(tailnet, 'owner_matches', return_value=True), \
             mock.patch.object(tailnet, 'endpoint_matches', return_value=False):
            with self.assertRaises(tailnet.TailnetError):
                tailnet.status()

    def test_unowned_and_malformed_mappings_fail_closed(self):
        owned = {'Foreground': {'one': session()}}
        self.assertTrue(tailnet.owned_mapping_present(owned, BINDING, 12345))
        self.assertFalse(tailnet.owned_mapping_present(owned, BINDING, 12346))
        for bad in ({'Foreground': []}, {'Web': ['invalid']}, {'Web': {'node.example.ts.net:8443': []}}):
            with self.subTest(bad=bad), self.assertRaises(tailnet.TailnetError):
                tailnet.owned_mapping_present(bad, BINDING, 12345)
        self.assertFalse(tailnet.matching_session({'TCP': {'8443': {'HTTPS': True}}}, BINDING, 12345))
