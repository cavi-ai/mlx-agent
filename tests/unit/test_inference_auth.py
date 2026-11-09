import asyncio
import hashlib
from email.message import Message
import json
import os
import http.client
import socket
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import unittest

from mlx_agent.inference_auth import AuthenticatedASGI, authorized_headers, key_file_digest, validate_digest, configuration_digest
from mlx_agent.serve import ServeError, load_recipes, plan_start
from mlx_agent.protected_worker import command, WORKER_DIGEST


class InferenceAuthenticationTests(unittest.TestCase):
    token = 'client-' + 'c' * 40

    def test_empty_or_ambiguous_key_configuration_never_disables_authentication(self):
        self.assertIsNone(configuration_digest())
        with self.assertRaises(OSError): configuration_digest(key_file='')
        with self.assertRaises(ValueError): configuration_digest(key_sha256='')
        with self.assertRaises(ValueError): configuration_digest(key_file='key', key_sha256='a' * 64)

    def test_credentials_reject_duplicates_and_ambiguous_headers(self):
        digest = hashlib.sha256(self.token.encode()).hexdigest()
        for values, expected in [([('Authorization', 'Bearer ' + self.token)], True),
                ([('x-api-key', self.token)], True), ([], False),
                ([('Authorization', 'Bearer wrong')], False),
                ([('Authorization', 'Bearer ' + self.token)] * 2, False),
                ([('Authorization', 'Bearer ' + self.token), ('x-api-key', self.token)], False)]:
            headers = Message()
            for name, value in values: headers[name] = value
            self.assertEqual(authorized_headers(headers, digest), expected)
        for invalid in ('', 'plaintext-key', 'A' * 64, None):
            with self.assertRaises(ValueError): validate_digest(invalid)

    def test_private_key_file_is_bounded_and_never_follows_a_link(self):
        with tempfile.TemporaryDirectory() as folder:
            key = Path(folder) / 'key'
            key.write_text(self.token + '\n'); key.chmod(0o600)
            self.assertEqual(key_file_digest(key), hashlib.sha256(self.token.encode()).hexdigest())
            link = key.with_name('link'); link.symlink_to(key)
            with self.assertRaises(OSError): key_file_digest(link)
            key.chmod(0o644)
            with self.assertRaises(ValueError): key_file_digest(key)
            key.chmod(0o600); key.write_text('x' * 512 + '\r\nextra')
            with self.assertRaises(ValueError): key_file_digest(key)
            key.write_text('short')
            with self.assertRaises(ValueError): key_file_digest(key)

    def test_asgi_authentication_precedes_application_and_covers_websockets(self):
        calls, sent = [], []
        async def app(scope, receive, send): calls.append(scope['type'])
        async def receive(): return {}
        async def send(message): sent.append(message)
        protected = AuthenticatedASGI(app, hashlib.sha256(self.token.encode()).hexdigest())
        for kind in ('http', 'websocket'):
            asyncio.run(protected({'type': kind, 'headers': []}, receive, send))
        self.assertEqual(calls, [])
        self.assertEqual(sent[0]['status'], 401)
        self.assertEqual(sent[-1]['code'], 4401)
        asyncio.run(protected({'type': 'http', 'headers': [(b'authorization', ('Bearer ' + self.token).encode())]}, receive, send))
        asyncio.run(protected({'type': 'lifespan'}, receive, send))
        self.assertEqual(calls, ['http', 'lifespan'])

    def test_authenticated_preview_requires_jit_and_binds_verifier_without_key(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'model'; model.mkdir()
            (model / 'config.json').write_text('{}')
            (model / 'model.safetensors').write_bytes(b'fixture')
            digest = hashlib.sha256(self.token.encode()).hexdigest()
            with self.assertRaises(ServeError) as error:
                plan_start(None, 'mlx_lm', load_recipes(), path=str(model), api_key_sha256=digest)
            self.assertEqual(error.exception.code, 'jit_required')
            plan = plan_start(None, 'mlx_lm', load_recipes(), path=str(model), jit=True, api_key_sha256=digest)
            self.assertEqual(plan['api_key_sha256'], digest)
            self.assertIn('/_mlx/status', plan['readiness'])
            self.assertNotIn(self.token, json.dumps(plan))
            self.assertNotEqual(plan['preview_hash'], plan_start(None, 'mlx_lm', load_recipes(), path=str(model), jit=True)['preview_hash'])

    @unittest.skipUnless(os.environ.get('MLX_AGENT_AUTH_UVICORN_PYTHON'), 'Explicit installed Uvicorn runtime check')
    def test_real_uvicorn_worker_refuses_unauthorized_http_before_application(self):
        with tempfile.TemporaryDirectory() as folder:
            entry = Path(folder)/'asgi.py'
            entry.write_text('''import sys, uvicorn
async def app(scope, receive, send):
    assert not any(k in (b'authorization', b'x-api-key') for k,v in scope['headers'])
    await send({'type':'http.response.start','status':200,'headers':[]})
    await send({'type':'http.response.body','body':b'{}'})
uvicorn.run(app,host='127.0.0.1',port=int(sys.argv[1]),lifespan='off',log_level='error')
''')
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0)); port = reservation.getsockname()[1]
            argv = command([os.environ['MLX_AGENT_AUTH_UVICORN_PYTHON'], str(entry), str(port)], 'mlx-vlm')
            environment = dict(os.environ, **{WORKER_DIGEST: hashlib.sha256(self.token.encode()).hexdigest()})
            process = subprocess.Popen(argv, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                for _ in range(100):
                    try:
                        with socket.create_connection(('127.0.0.1', port), timeout=.1): break
                    except OSError:
                        if process.poll() is not None: self.fail(process.stderr.read().decode())
                        time.sleep(.05)
                for headers, expected in [({}, 401), ({'Authorization':'Bearer wrong'}, 401),
                        ({'Authorization':'Bearer '+self.token}, 200), ({'x-api-key':self.token}, 200)]:
                    client = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
                    try:
                        client.request('GET', '/v1/models', headers=headers)
                        response = client.getresponse(); self.assertEqual(response.status, expected); response.read()
                    finally: client.close()
            finally:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
                process.stderr.close()
