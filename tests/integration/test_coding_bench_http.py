"""Exercise the coding workload through MCP stdio and real loopback SSE."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer


class CodingHTTPTests(unittest.TestCase):
    def test_mcp_preserves_tool_results_feedback_and_cache_provenance(self):
        requests = []
        first = "def clamp(value, lower, upper):\n    return max(lower, min(value, upper))\n"
        second = "def clamp(value, lower, upper):\n    if lower > upper:\n        raise ValueError('bounds')\n    return max(lower, min(value, upper))\n"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[{"id":"org/model"}]}')

            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                index = len(requests) - 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b": keepalive 0/100\n\n: keepalive 100/100\n\n")
                delta = {"tool_calls": [{"index": 0, "id": "call-1", "type": "function", "function": {
                    "name": "read_file", "arguments": '{"path":"clamp.py"}'}}]} if index == 0 else {"content": (first, second)[index - 1]}
                for event in ({"choices": [{"delta": delta}]}, {"choices": [], "usage": {
                        "prompt_tokens": 100 * (index + 1), "completion_tokens": 20}}):
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

        server = HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "mlx_agent_execute", "arguments": {"capability": "bench", "arguments":
                    "coding --repo org/model --base-url http://127.0.0.1:{0} --json".format(server.server_port)}}}
            process = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/mlx-agent-mcp")],
                input=json.dumps(request) + "\n", capture_output=True, text=True, timeout=10, env=os.environ.copy())
            self.assertEqual(process.returncode, 0, process.stderr)
            tool = json.loads(process.stdout)["result"]["structuredContent"]
            self.assertEqual(tool["exit_code"], 0, tool)
            self.assertFalse(tool["stdout_truncated"])
            result = json.loads(tool["stdout"])["data"]
            self.assertTrue(result["quality_passed"])
            self.assertEqual(result["gen_tokens"], 512)
            self.assertEqual([t["cached_tokens"] for t in result["turns"]], [0, 100, 200])
            self.assertEqual(len(requests), 3)
            self.assertIn("tools", requests[0])
            self.assertNotIn("tools", requests[1])
            self.assertNotIn("tools", requests[2])
            self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
            self.assertEqual(requests[2]["messages"][-2]["content"], first)
            self.assertTrue(json.loads(requests[2]["messages"][-1]["content"])["test_results"]["passed"])
            self.assertTrue(all(r["seed"] == 0 and r["temperature"] == 0 for r in requests))
            for turn, request in zip(result["turns"], requests):
                digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                self.assertEqual(turn["request_sha256"], digest)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)
