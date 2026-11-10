"""Loopback HTTP interface for typed model jobs and model residency controls."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import signal
import sys
import threading

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_agent.jit_serve import GatewayError, GatewayHandler, GatewayServer, MAX_BODY
from mlx_agent.task_contracts import TaskError
from mlx_agent.task_runtime import task_result
from mlx_agent.task_serving import TaskModelWorker
from mlx_agent import task_kinds


class TaskHandler(GatewayHandler):
    def _artifact(self, identifier, name):
        record = self.server.controller.record(identifier)
        outcome = task_result(identifier, self.server.controller.store.root)
        if name == 'result':
            reference, filename, content_type = outcome['artifact'], 'result.json', 'application/json'
        elif name == 'output' and task_kinds.get(record['kind']).output:
            result = outcome['result']
            if result is None:
                result = json.loads(Path(outcome['artifact']['path']).read_bytes())
            reference = result['output_artifact']
            filename = task_kinds.get(record['kind']).output
            content_type = {'.wav': 'audio/wav', '.png': 'image/png', '.mp4': 'video/mp4', '.glb': 'model/gltf-binary'}[Path(filename).suffix]
        else:
            raise GatewayError(404, 'not_found', 'No artifact with that name.')
        path = self.server.controller.store.artifact_path(identifier, filename)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(descriptor, 'rb') as stream:
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: stream.read(65536), b''):
                digest.update(chunk)
                size += len(chunk)
            if size != reference['bytes'] or digest.hexdigest() != reference['sha256']:
                raise TaskError('artifact_changed', 'The saved task artifact changed.')
            stream.seek(0)
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(size))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            for chunk in iter(lambda: stream.read(65536), b''):
                self.wfile.write(chunk)

    def do_GET(self):
        try:
            self._guard(control=self.path == '/_mlx/status')
            controller = self.server.controller
            if self.path == '/_mlx/status':
                self._json(200, controller.status())
            elif self.path == '/v1/models':
                self._json(200, {'object': 'list', 'data': [{'id': self.server.config['model'], 'object': 'model',
                    'task_kinds': self.server.config['task_kinds'], 'api': '/v1/tasks'}]})
            elif self.path.startswith('/v1/tasks/'):
                parts = self.path.split('/')
                identifier = parts[3]
                controller.record(identifier)
                if len(parts) == 4:
                    self._json(200, controller.record(identifier))
                elif len(parts) == 6 and parts[4] == 'artifacts':
                    self._artifact(identifier, parts[5])
                elif len(parts) == 5 and parts[4] == 'result':
                    self._json(200, task_result(identifier, controller.store.root))
                else:
                    raise GatewayError(404, 'not_found', 'Unknown task endpoint.')
            else:
                raise GatewayError(404, 'not_found', 'Unknown endpoint.')
        except (GatewayError, TaskError) as error:
            self._error(error if isinstance(error, GatewayError) else GatewayError(404, error.code, str(error)))

    def do_POST(self):
        try:
            self._guard(control=self.path in ('/_mlx/unload', '/_mlx/policy', '/_mlx/cache'))
            if self.headers.get('Transfer-Encoding'):
                raise GatewayError(400, 'invalid_body', 'Use a bounded Content-Length request.')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 1024 * 1024:
                    raise ValueError()
                body = json.loads(self.rfile.read(length)) if length else {}
                if not isinstance(body, dict):
                    raise ValueError()
            except (ValueError, UnicodeError):
                raise GatewayError(400, 'invalid_body', 'Expected a JSON object of at most 1 MiB.') from None
            controller = self.server.controller
            if self.path == '/_mlx/unload':
                self._json(200, controller.unload())
            elif self.path == '/_mlx/cache':
                if body:
                    raise GatewayError(400, 'invalid_body', 'Cache reclaim requires an empty object.')
                self._json(200, controller.reclaim_cache())
            elif self.path == '/_mlx/policy':
                self._json(200, controller.configure(body))
            elif self.path == '/v1/tasks':
                self._json(202, controller.submit(body))
            elif self.path.startswith('/v1/tasks/') and self.path.endswith('/cancel'):
                parts = self.path.split('/')
                if len(parts) != 5 or body:
                    raise GatewayError(400, 'invalid_request', 'Cancel requires an empty object and a task ID.')
                controller.record(parts[3])
                self._json(200, controller.store.request_cancel(parts[3]))
            else:
                raise GatewayError(404, 'not_found', 'Unknown inference endpoint.')
        except (GatewayError, TaskError) as error:
            self._error(error if isinstance(error, GatewayError) else GatewayError(400, error.code, str(error)))
        except OSError:
            self._error(GatewayError(503, 'task_unavailable', 'The local model job could not be started.'))


class TaskServer(GatewayServer):
    def __init__(self, address, config):
        super().__init__(address, config)
        self.RequestHandlerClass = TaskHandler
        self.controller = TaskModelWorker(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    path = Path(args.config)
    if path.is_symlink():
        raise ValueError('Refusing a symbolic-link endpoint configuration.')
    config = json.loads(path.read_text())
    if args.model != config['model'] or args.port != config['port']:
        raise ValueError('Endpoint configuration identity changed.')
    server = TaskServer(('127.0.0.1', args.port), config)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    finally:
        server.controller.close()
        server.server_close()


if __name__ == '__main__':
    main()
