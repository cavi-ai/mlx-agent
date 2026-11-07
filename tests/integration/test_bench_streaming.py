"""Exercise streaming availability over a real loopback HTTP connection."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from mlx_agent.verification import _http_event_stream


class BenchStreamingTests(unittest.TestCase):
    def test_first_event_is_available_before_the_server_finishes(self):
        release = threading.Event()
        buffered = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(json.dumps({"message": {"content": "first"}}).encode() + b"\n")
                self.wfile.flush()
                if not release.wait(2):
                    buffered.set()
                self.wfile.write(b'{"done":true,"eval_count":1,"eval_duration":1000000}\n')
                self.wfile.flush()

        server = HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        stream = _http_event_stream("http://127.0.0.1:{0}/stream".format(server.server_port), {}, timeout=5)
        try:
            self.assertEqual(next(stream)["message"]["content"], "first")
            self.assertFalse(buffered.is_set(), "the reader waited for stream completion before yielding content")
            release.set()
            self.assertEqual(list(stream)[0]["eval_count"], 1)
        finally:
            release.set()
            stream.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)
