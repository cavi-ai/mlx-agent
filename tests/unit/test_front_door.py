import contextlib
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

from mlx_agent.cli import main

AUTH_PROBE_VALUE = 'front-' + 'k' * 40
DIGEST = hashlib.sha256(AUTH_PROBE_VALUE.encode()).hexdigest()


class Upstream(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        self.seen = []
        super().__init__(('127.0.0.1', 0), UpstreamHandler)


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.server.seen.append({'path': self.path, 'body': body, 'host': self.headers['Host'],
            'origin': self.headers.get('Origin'), 'authorization': self.headers.get('Authorization'),
            'x-api-key': self.headers.get('x-api-key')})
        self.send_response(200)
        self.send_header('Connection', 'close')
        if json.loads(body).get('stream'):
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for event in (b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n',
                          b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n', b'data: [DONE]\n\n'):
                self.wfile.write(event)
                self.wfile.flush()
            return
        value = json.dumps({'served_by': self.server.server_port, 'path': self.path}).encode()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(value)))
        self.end_headers()
        self.wfile.write(value)


class FrontDoorTests(unittest.TestCase):
    def setUp(self):
        from mlx_agent.front_door import FrontDoorServer
        self.upstreams = {}
        for name in ('chat', 'embed', 'decide', 'open', 'other', 'dead', 'twin_a', 'twin_b'):
            upstream = Upstream()
            thread = threading.Thread(target=upstream.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(upstream.server_close)
            self.addCleanup(upstream.shutdown)
            self.upstreams[name] = upstream

        def entry(name, path, **values):
            return dict({'path': path, 'repo': None, 'runtime': 'mlx-vlm', 'jit': True, 'api_key_sha256': DIGEST,
                         'port': self.upstreams[name].server_port, 'alive': True, 'argv_match': True}, **values)

        self.services = [
            entry('chat', '/models/Chat-9B-4bit'),
            entry('embed', '/models/Embed-4B', runtime='tasks', jit=None, task_kinds=['embedding']),
            entry('open', '/models/Open-1B', api_key_sha256=None),
            entry('other', '/models/Other-1B', api_key_sha256='0' * 64),
            entry('dead', '/models/Dead-1B', alive=False, argv_match=False),
            entry('twin_a', '/a/Twin-1B'),
            entry('twin_b', '/b/Twin-1B'),
        ]
        self.server = FrontDoorServer(('127.0.0.1', 0), {'api_key_sha256': DIGEST},
                                      lambda: {'services': self.services, 'errors': []})
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, value=None, headers=None, key='bearer'):
        credentials = {'bearer': {'Authorization': 'Bearer ' + AUTH_PROBE_VALUE}, 'x-api-key': {'x-api-key': AUTH_PROBE_VALUE},
                       'wrong': {'Authorization': 'Bearer ' + 'w' * 46}, None: {}}[key]
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            client.request(method, path, json.dumps(value) if value is not None else None,
                           dict(credentials, **(headers or {})))
            response = client.getresponse()
            return response.status, response.read()
        finally:
            client.close()

    def forwarded(self):
        return sum(len(upstream.seen) for upstream in self.upstreams.values())

    def test_models_lists_stable_names_of_routes_sharing_the_key(self):
        status, raw = self.request('GET', '/v1/models')
        self.assertEqual(status, 200, raw)
        models = json.loads(raw)['data']
        self.assertEqual([model['id'] for model in models], ['Chat-9B-4bit', 'Embed-4B'])
        self.assertEqual(models[1]['task_kinds'], ['embedding'])
        self.assertNotIn(b'/models/', raw)

    def test_routes_by_model_and_forwards_the_client_credential_to_the_route_port(self):
        body = {'model': 'Embed-4B', 'input': ['a', 'b']}
        status, raw = self.request('POST', '/v1/embeddings', body)
        self.assertEqual((status, json.loads(raw)), (200, {'served_by': self.upstreams['embed'].server_port, 'path': '/v1/embeddings'}))
        seen = self.upstreams['embed'].seen[-1]
        self.assertEqual(seen['body'], json.dumps(body).encode())
        self.assertEqual(seen['host'], '127.0.0.1:{}'.format(self.upstreams['embed'].server_port))
        self.assertEqual((seen['authorization'], seen['x-api-key']), ('Bearer ' + AUTH_PROBE_VALUE, None))
        status, raw = self.request('POST', '/v1/chat/completions', {'model': 'Chat-9B-4bit', 'messages': []},
                                   headers={'Origin': 'http://127.0.0.1:{}'.format(self.server.server_port)}, key='x-api-key')
        self.assertEqual(json.loads(raw)['served_by'], self.upstreams['chat'].server_port)
        seen = self.upstreams['chat'].seen[-1]
        self.assertEqual((seen['authorization'], seen['x-api-key'], seen['origin']), (None, AUTH_PROBE_VALUE, None))

    def test_routes_decisions_only_to_classification_routes(self):
        self.services.append(dict(self.services[1], path='/models/Kev-4B-MLX-8bit', task_kinds=['classification'],
                                  port=self.upstreams['decide'].server_port))
        body = {'model': 'Kev-4B-MLX-8bit', 'state': 'charged twice', 'questions': {'refund': {'type': 'noul', 'instructions': 'refund?'}}}
        status, raw = self.request('POST', '/v1/systemone', body)
        self.assertEqual((status, json.loads(raw)), (200, {'served_by': self.upstreams['decide'].server_port, 'path': '/v1/systemone'}))
        self.assertEqual(self.upstreams['decide'].seen[-1]['body'], json.dumps(body).encode())
        for model, path in (('Embed-4B', '/v1/systemone'), ('Chat-9B-4bit', '/v1/systemone'),
                            ('Kev-4B-MLX-8bit', '/v1/embeddings'), ('Kev-4B-MLX-8bit', '/v1/chat/completions')):
            with self.subTest(model=model, path=path):
                self.assertEqual(self.request('POST', path, {'model': model})[0], 400)
        self.assertEqual(self.forwarded(), 1)

    def test_streams_route_events_through(self):
        status, raw = self.request('POST', '/v1/chat/completions', {'model': 'Chat-9B-4bit', 'stream': True, 'messages': []})
        self.assertEqual(status, 200)
        self.assertEqual(raw.count(b'data: '), 3)
        self.assertTrue(raw.endswith(b'data: [DONE]\n\n'))

    def test_never_forwards_to_routes_without_the_same_key_or_with_ambiguous_names(self):
        for model, path, expected in (('Open-1B', '/v1/chat/completions', 404), ('Other-1B', '/v1/chat/completions', 404),
                                      ('Dead-1B', '/v1/chat/completions', 404), ('Twin-1B', '/v1/chat/completions', 409),
                                      ('/models/Chat-9B-4bit', '/v1/chat/completions', 404),
                                      ('Chat-9B-4bit', '/v1/embeddings', 400), ('Embed-4B', '/v1/chat/completions', 400)):
            with self.subTest(model=model, path=path):
                self.assertEqual(self.request('POST', path, {'model': model})[0], expected)
        self.assertEqual(self.forwarded(), 0)

    def test_requires_the_key_and_loopback_host_and_origin(self):
        port = self.server.server_port
        self.assertEqual(self.request('GET', '/health', key=None)[0], 200)
        for headers, key, expected in (({}, None, 401), ({}, 'wrong', 401),
                                       ({'Host': 'evil.invalid'}, 'bearer', 403),
                                       ({'Host': '127.0.0.1:{}'.format(port + 1)}, 'bearer', 403),
                                       ({'Origin': 'http://evil.invalid'}, 'bearer', 403),
                                       ({'Origin': 'null'}, 'bearer', 403)):
            with self.subTest(headers=headers, key=key):
                status, raw = self.request('POST', '/v1/chat/completions', {'model': 'Chat-9B-4bit'}, headers, key)
                self.assertEqual(status, expected, raw)
                self.assertEqual(self.request('GET', '/v1/models', None, headers, key)[0], expected)
        status, raw = self.request('POST', '/v1/chat/completions', {'model': 'Chat-9B-4bit'},
                                   {'Host': 'evil.invalid'})
        self.assertEqual(json.loads(raw)['error']['code'], 'invalid_host')
        status, raw = self.request('POST', '/v1/chat/completions', {'model': 'Chat-9B-4bit'},
                                   {'Origin': 'http://evil.invalid'})
        self.assertEqual(json.loads(raw)['error']['code'], 'invalid_origin')
        self.assertEqual(self.forwarded(), 0)

    def test_front_door_requires_one_key_source(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['serve', 'front-door', '--port', '18492'])
            with self.assertRaises(SystemExit):
                main(['serve', 'front-door', '--port', '18492', '--api-key-sha256', DIGEST, '--api-key-file', '/missing'])


class FrontDoorRelayTests(unittest.TestCase):
    def serve(self, server):
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    @staticmethod
    def route(port, name="chat"):
        return {"path": "/models/" + name, "runtime": "mlx_lm", "jit": True, "port": port,
                "alive": True, "argv_match": True, "api_key_sha256": DIGEST}

    def front(self, discover, limits=None):
        from mlx_agent.front_door import FrontDoorServer
        server = FrontDoorServer(("127.0.0.1", 0), {"api_key_sha256": DIGEST}, discover)
        if limits is not None:
            server.relay_limits = limits
        return self.serve(server)

    def request(self, server, method="POST", path="/v1/chat/completions", body=None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        self.addCleanup(conn.close)
        conn.request(method, path, body, {"Authorization": "Bearer " + AUTH_PROBE_VALUE})
        return conn

    def test_disconnect_closes_unanswered_upstream_connection(self):
        received, disconnected = threading.Event(), threading.Event()

        class WaitingUpstream(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                received.set()
                self.connection.settimeout(1.5)
                try:
                    if self.connection.recv(1) == b"":
                        disconnected.set()
                except TimeoutError:
                    pass

        upstream = self.serve(ThreadingHTTPServer(("127.0.0.1", 0), WaitingUpstream))
        front = self.front(lambda: {"services": [self.route(upstream.server_port)]})
        client = self.request(front, body='{"model":"chat"}')
        self.assertTrue(received.wait(1))
        client.close()
        self.assertTrue(disconnected.wait(2))

    def test_headers_deadline_returns_504(self):
        from mlx_agent.http_relay import RelayLimits

        class WaitingUpstream(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.connection.settimeout(1.5)
                try:
                    self.connection.recv(1)
                except TimeoutError:
                    pass

        upstream = self.serve(ThreadingHTTPServer(("127.0.0.1", 0), WaitingUpstream))
        front = self.front(lambda: {"services": [self.route(upstream.server_port)]}, RelayLimits(headers=.1))
        response = self.request(front, body='{"model":"chat"}').getresponse()
        self.assertEqual(response.status, 504)
        self.assertEqual(json.loads(response.read())["error"]["code"], "route_timeout")

    def test_idle_deadline_ends_a_stalled_stream(self):
        from mlx_agent.http_relay import RelayLimits

        class StalledStream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: first\n\n")
                self.connection.settimeout(1.5)
                try:
                    self.connection.recv(1)
                except TimeoutError:
                    pass

        upstream = self.serve(ThreadingHTTPServer(("127.0.0.1", 0), StalledStream))
        front = self.front(lambda: {"services": [self.route(upstream.server_port)]}, RelayLimits(idle=.1))
        reply = self.request(front, body='{"model":"chat","stream":true}').getresponse()
        self.assertEqual(reply.status, 200)
        self.assertEqual(reply.read(), b"data: first\n\n")

    def test_failed_connect_refreshes_route_before_post(self):
        dead = socket.socket()
        dead.bind(("127.0.0.1", 0))
        self.addCleanup(dead.close)
        upstream = self.serve(Upstream())
        services = [self.route(dead.getsockname()[1])]
        front = self.front(lambda: {"services": services})
        front.routes()
        front.attempted_at -= 2
        services[:] = [self.route(upstream.server_port)]
        dead.close()

        response = self.request(front, body='{"model":"chat"}').getresponse()
        self.assertEqual(json.loads(response.read())["served_by"], upstream.server_port)
        self.assertEqual(len(upstream.seen), 1)

    def test_failed_reply_never_replays_post(self):
        seen = []

        class DroppedReply(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(self.rfile.read(int(self.headers["Content-Length"])))

        upstream = self.serve(ThreadingHTTPServer(("127.0.0.1", 0), DroppedReply))
        replacement = self.serve(Upstream())
        services = [self.route(upstream.server_port)]
        front = self.front(lambda: {"services": services})
        front.routes()
        front.attempted_at -= 2
        services[:] = [self.route(replacement.server_port)]

        response = self.request(front, body='{"model":"chat"}').getresponse()
        self.assertEqual(response.status, 503)
        response.read()
        self.assertEqual(seen, [b'{"model":"chat"}'])
        self.assertFalse(replacement.seen)

    def test_route_refresh_keeps_last_complete_table_and_coalesces_readers(self):
        refreshing, release = threading.Event(), threading.Event()
        upstream = self.serve(Upstream())
        calls = []

        def discover():
            calls.append(1)
            if len(calls) == 1:
                return {"services": [self.route(upstream.server_port)]}
            refreshing.set()
            release.wait(2)
            return {"services": [], "errors": [{"message": "deadline"}]}

        front = self.front(discover)
        self.addCleanup(release.set)
        front.routes()
        front.attempted_at -= 6
        for _ in range(2):
            response = self.request(front, "GET", "/v1/models").getresponse()
            self.assertEqual([row["id"] for row in json.loads(response.read())["data"]], ["chat"])
        self.assertTrue(refreshing.wait(1))
        self.assertEqual(len(calls), 2)
        release.set()
        self.assertIn("chat", front.routes(wait=True))
        self.assertTrue(front.degraded)


class FrontDoorSessionTests(unittest.TestCase):
    """Run the real command: its signal handlers and parent watch need a process of their own."""

    COMMAND = [sys.executable, str(Path(__file__).resolve().parents[2] / 'scripts/mlx-agent'), 'serve', 'front-door']
    # Owns the front door as its child and exits when its input closes.
    LAUNCHER = 'import subprocess, sys; subprocess.Popen(sys.argv[1:]); sys.stdin.read()'

    def setUp(self):
        state = tempfile.TemporaryDirectory()
        self.addCleanup(state.cleanup)
        self.environment = dict(os.environ, XDG_STATE_HOME=state.name)
        self.orphan = None
        self.addCleanup(self.stop_orphan)

    def stop_orphan(self):
        if self.orphan is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.orphan, signal.SIGKILL)

    def launch(self, command, **options):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   env=self.environment, **options)
        for stream in (process.stdout, process.stdin):
            if stream is not None:
                self.addCleanup(stream.close)
        self.addCleanup(process.wait)
        self.addCleanup(process.kill)
        return process

    def read(self, process, seconds=60):
        """The next output line, or b'' once every writer closed the pipe."""
        if not select.select([process.stdout], [], [], seconds)[0]:
            self.fail('No output within {} seconds.'.format(seconds))
        return process.stdout.readline()

    def health(self, port):
        client = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            client.request('GET', '/health')
            return client.getresponse().status
        finally:
            client.close()

    def free_port(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            return probe.getsockname()[1]

    def test_json_session_reports_the_listening_port_and_stops_on_sigterm(self):
        port = self.free_port()
        process = self.launch(self.COMMAND + ['--port', str(port), '--api-key-sha256', DIGEST, '--json'])
        ready = json.loads(self.read(process))
        self.assertEqual((ready['operation'], ready['status']), ('serve-front-door', 'ok'))
        self.assertEqual(ready['data'], {'state': 'listening', 'port': port,
                                         'api_base': 'http://127.0.0.1:{}/v1'.format(port), 'pid': process.pid})
        self.assertEqual(self.health(port), 200)
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=30), 0)
        self.assertEqual(process.stdout.read(), b'')

    def test_watch_parent_stops_the_front_door_when_its_launcher_exits(self):
        port = self.free_port()
        launcher = self.launch([sys.executable, '-c', self.LAUNCHER] + self.COMMAND + [
            '--port', str(port), '--api-key-sha256', DIGEST, '--json', '--watch-parent'], stdin=subprocess.PIPE)
        ready = json.loads(self.read(launcher))['data']
        self.orphan = ready['pid']
        self.assertNotEqual(ready['pid'], launcher.pid)
        self.assertEqual(self.health(port), 200)
        launcher.stdin.close()
        self.assertEqual(launcher.wait(timeout=30), 0)
        # The front door holds the last write end of the pipe: end of output is its exit.
        self.assertEqual(self.read(launcher, 30), b'')
        self.orphan = None
        with self.assertRaises(OSError):
            self.health(port)

    def test_json_session_names_a_port_that_is_in_use(self):
        with socket.socket() as taken:
            taken.bind(('127.0.0.1', 0))
            taken.listen(1)
            process = self.launch(self.COMMAND + ['--port', str(taken.getsockname()[1]),
                                                  '--api-key-sha256', DIGEST, '--json'])
            self.assertEqual(process.wait(timeout=60), 2)
        result = json.loads(process.stdout.read())
        self.assertEqual((result['operation'], result['status'], result['error']['code']),
                         ('serve-front-door', 'error', 'front_door_port_unavailable'))

    def test_json_session_names_an_unusable_key(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            status = main(['serve', 'front-door', '--port', '18492', '--api-key-sha256', 'not-a-digest', '--json'])
        result = json.loads(output.getvalue())
        self.assertEqual((status, result['status'], result['error']['code']),
                         (2, 'error', 'front_door_key_unavailable'))
