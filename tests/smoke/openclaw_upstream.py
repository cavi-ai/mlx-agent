"""Serve a front-door model list on loopback for the OpenClaw provider smoke."""
import http.server
import json
import os
import sys
from pathlib import Path

MODELS = {
    "object": "list",
    "data": [
        {"id": "smoke-chat-model", "object": "model", "owned_by": "mlx-agent"},
        # A name without "embed" so only task_kinds can keep it out of the chat catalog.
        {"id": "smoke-vector-model", "object": "model", "owned_by": "mlx-agent", "task_kinds": ["embedding"]},
    ],
}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        expected = "Bearer " + os.environ["MLX_AGENT_API_KEY"]
        if self.path != "/v1/models":
            self._json(404, {"error": {"code": "not_found"}})
        elif self.headers.get("Authorization") != expected:
            self._json(401, {"error": {"code": "access_denied"}})
        else:
            self._json(200, MODELS)

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Path(sys.argv[1]).write_text(str(server.server_address[1]))
    server.serve_forever()


if __name__ == "__main__":
    main()
