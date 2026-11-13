import contextlib
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import unittest

from mlx_agent.cli import main

TOKEN = 'front-' + 'k' * 40
DIGEST = hashlib.sha256(TOKEN.encode()).hexdigest()


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
        for name in ('chat', 'embed', 'open', 'other', 'dead', 'twin_a', 'twin_b'):
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
        credentials = {'bearer': {'Authorization': 'Bearer ' + TOKEN}, 'x-api-key': {'x-api-key': TOKEN},
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
        self.assertEqual((seen['authorization'], seen['x-api-key']), ('Bearer ' + TOKEN, None))
        status, raw = self.request('POST', '/v1/chat/completions', {'model': 'Chat-9B-4bit', 'messages': []},
                                   headers={'Origin': 'http://127.0.0.1:{}'.format(self.server.server_port)}, key='x-api-key')
        self.assertEqual(json.loads(raw)['served_by'], self.upstreams['chat'].server_port)
        seen = self.upstreams['chat'].seen[-1]
        self.assertEqual((seen['authorization'], seen['x-api-key'], seen['origin']), (None, TOKEN, None))

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
