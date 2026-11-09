"""Optional bearer-key verification shared by local gateways and their workers."""
import hashlib
import hmac
from pathlib import Path
import re


def validate_digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('API key verification requires a lowercase SHA-256 digest.')
    return value


def key_file_digest(filename):
    if __package__:
        from .serving_access import read_access_key
    else:
        from serving_access import read_access_key
    return read_access_key(filename)[1]


def configuration_digest(key_file=None, key_sha256=None):
    if key_file is not None:
        if key_sha256 is not None:
            raise ValueError('Select one API key source.')
        return key_file_digest(key_file)
    return validate_digest(key_sha256) if key_sha256 is not None else None


def authorized(authorization, api_keys, digest):
    """Reject ambiguous credentials and compare only fixed-length verification hashes."""
    if len(authorization) + len(api_keys) != 1:
        return False
    if authorization:
        scheme, separator, token = authorization[0].partition(b' ')
        if scheme.lower() != b'bearer' or not separator:
            return False
    else:
        token = api_keys[0]
    if not 1 <= len(token) <= 512 or any(b < 33 or b > 126 for b in token):
        return False
    return hmac.compare_digest(hashlib.sha256(token).hexdigest(), digest)


def authorized_headers(headers, digest):
    return authorized([v.encode('latin-1') for v in headers.get_all('Authorization', [])],
                      [v.encode('latin-1') for v in headers.get_all('x-api-key', [])], digest)


def runtime_fingerprint():
    digest = hashlib.sha256()
    for name in ('inference_auth.py', 'protected_worker.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def loopback_opener():
    import urllib.request

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


class AuthenticatedASGI:
    def __init__(self, app, digest):
        self.app, self.digest = app, validate_digest(digest)

    async def __call__(self, scope, receive, send):
        if scope['type'] in ('http', 'websocket'):
            headers = scope.get('headers', [])
            if not authorized([v for k, v in headers if k.lower() == b'authorization'],
                              [v for k, v in headers if k.lower() == b'x-api-key'], self.digest):
                if scope['type'] == 'websocket':
                    await send({'type': 'websocket.close', 'code': 4401})
                else:
                    body = b'{"error":{"code":"authentication_required","message":"A valid API key is required."}}'
                    await send({'type': 'http.response.start', 'status': 401, 'headers': [
                        (b'content-type', b'application/json'), (b'content-length', str(len(body)).encode()),
                        (b'www-authenticate', b'Bearer'), (b'connection', b'close')]})
                    await send({'type': 'http.response.body', 'body': body})
                return
        if scope['type'] in ('http', 'websocket'):
            scope = dict(scope, headers=[(k, v) for k, v in scope.get('headers', [])
                                        if k.lower() not in (b'authorization', b'x-api-key')])
        await self.app(scope, receive, send)


def protect_worker(digest, asgi=False):
    """Install enforcement before importing the selected server, in its own process."""
    from http.server import BaseHTTPRequestHandler
    validate_digest(digest)
    parse = BaseHTTPRequestHandler.parse_request

    def parse_request(handler):
        if not parse(handler):
            return False
        if authorized_headers(handler.headers, digest):
            del handler.headers['Authorization']
            del handler.headers['x-api-key']
            return True
        body = b'{"error":{"code":"authentication_required","message":"A valid API key is required."}}'
        handler.send_response(401)
        handler.send_header('WWW-Authenticate', 'Bearer')
        handler.send_header('Content-Type', 'application/json')
        handler.send_header('Content-Length', str(len(body)))
        handler.send_header('Connection', 'close')
        handler.end_headers()
        handler.close_connection = True
        if handler.command != 'HEAD':
            handler.wfile.write(body)
        return False

    BaseHTTPRequestHandler.parse_request = parse_request
    if asgi:
        import uvicorn
        load = uvicorn.Config.load

        def load_authenticated(config):
            load(config)
            config.loaded_app = AuthenticatedASGI(config.loaded_app, digest)

        uvicorn.Config.load = load_authenticated
