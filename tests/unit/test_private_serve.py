import http.client
import json
import os
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.private_serve import private_server_class, guarded_handler, PRIVATE_PROTOCOL


class PrivateServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='mlx-private-test-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / 'worker.sock')
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.path)
        self.listener.listen()
        self.addCleanup(self.listener.close)
        self.calls = []
        calls = self.calls

        class InferenceHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append(body)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

        handler = guarded_handler(InferenceHandler, '/local/model', max_tokens=32)
        server_type = private_server_class(os.dup(self.listener.fileno()))
        self.server = server_type(('127.0.0.1', 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method='POST', path='/v1/chat/completions', body=None, headers=None):
        values = {'Host': '127.0.0.1', 'Content-Type': 'application/json', 'Connection': 'close'}
        values.update(headers or {})
        payload = json.dumps(body if body is not None else {'messages': []}).encode() if method == 'POST' else b''
        if method == 'POST':
            values['Content-Length'] = str(len(payload))
        lines = [f'{method} {path} HTTP/1.1'] + [f'{name}: {value}' for name, value in values.items()]
        frame = ('\r\n'.join(lines) + '\r\n\r\n').encode() + payload
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(3)
            connection.connect(self.path)
            # Header refusals can close immediately. Queue the complete small
            # request before reading so a later body write cannot race the close.
            connection.sendall(frame)
            with http.client.HTTPResponse(connection) as response:
                response.begin()
                return response.status, response.getheader('X-MLX-Agent-Transport'), response.read()

    def test_inherited_listener_and_fixed_model_are_enforced(self):
        self.assertEqual(self.server.socket.family, socket.AF_UNIX)
        status, protocol, body = self.request(body={'messages': [], 'model': 'default_model', 'max_tokens': 1000})
        self.assertEqual((status, protocol), (200, PRIVATE_PROTOCOL))
        self.assertEqual(json.loads(body)['model'], '/local/model')
        self.assertEqual(json.loads(body)['max_tokens'], 32)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.request('GET', '/_mlx/private')[0], 200)
        self.assertEqual(self.request('GET', '/v1/models')[0], 200)
        self.assertEqual(len(self.calls), 1)

    def test_browser_headers_and_invalid_models_never_reach_inference(self):
        for headers in ({'Host': 'attacker.invalid'}, {'Origin': 'https://attacker.invalid'},
                        {'Origin': 'null'}, {'Origin': ''}):
            self.assertEqual(self.request(headers=headers)[0], 403)
        for body in ({'model': 'another/model', 'messages': []},
                     {'adapter_path': '/another/adapter', 'messages': []},
                     {'draft_model': 'another/model', 'messages': []},
                     {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': 'https://outside.invalid'}]}]},
                     {'messages': [], 'stream': 'true'}, {'messages': [], 'max_tokens': True}):
            self.assertEqual(self.request(body=body)[0], 400)
        self.assertEqual(self.request(headers={'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.request(path='/v1/completions')[0], 404)
        self.assertEqual(self.request('OPTIONS')[0], 405)
        self.assertEqual(self.calls, [])

    def test_duplicate_headers_and_body_framing_are_rejected(self):
        for headers in ([('Host', 'attacker.invalid')], [('Content-Length', '2')], [('Transfer-Encoding', 'chunked')]):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(3)
                connection.connect(self.path)
                lines = ['POST /v1/chat/completions HTTP/1.1', 'Host: 127.0.0.1',
                         'Content-Type: application/json', 'Content-Length: 2']
                lines.extend(f'{name}: {value}' for name, value in headers)
                # A rejecting server may close immediately after the headers;
                # send the small complete frame before waiting for its reply.
                connection.sendall(('\r\n'.join(lines) + '\r\n\r\n{}').encode())
                response = http.client.HTTPResponse(connection)
                self.addCleanup(response.close)
                response.begin()
                self.assertIn(response.status, (400, 403))
                response.read()
        self.assertEqual(self.calls, [])

    def test_tcp_descriptor_is_refused(self):
        with socket.socket() as tcp:
            tcp.bind(('127.0.0.1', 0))
            tcp.listen()
            with self.assertRaises(ValueError):
                private_server_class(os.dup(tcp.fileno()))
