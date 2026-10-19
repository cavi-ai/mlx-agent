"""Exercise framing and recovery through the real stdio entry point."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
FRAME_LIMIT = 65536


def ping(request_id):
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": "ping"}).encode() + b"\n"


class MCPStdioTests(unittest.TestCase):
    def exchange(self, payload):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "mlx-agent-mcp")],
            input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
            env=dict(os.environ, PYTHONINTMAXSTRDIGITS="4300"),
        )
        self.assertEqual(0, result.returncode, result.stderr.decode())
        return [json.loads(line) for line in result.stdout.splitlines()]

    def test_oversized_frame_suffix_is_never_executed_as_a_request(self):
        responses = self.exchange(b"x" * (FRAME_LIMIT + 1) + ping(17) + ping(18))
        self.assertEqual([None, 18], [response["id"] for response in responses])
        self.assertEqual(-32600, responses[0]["error"]["code"])
        self.assertEqual({}, responses[1]["result"])

    def test_large_frame_is_rejected_once_and_the_next_frame_survives(self):
        responses = self.exchange(b"x" * 1000000 + b"\n" + ping(18))
        self.assertEqual([None, 18], [response["id"] for response in responses])
        self.assertEqual(-32600, responses[0]["error"]["code"])
        self.assertEqual({}, responses[1]["result"])

    def test_oversized_frame_ending_at_read_boundary_preserves_the_next_frame(self):
        responses = self.exchange(b"x" * FRAME_LIMIT + b"\n" + ping(18))
        self.assertEqual([None, 18], [response["id"] for response in responses])

    @unittest.skipUnless(hasattr(sys, "get_int_max_str_digits"), "Python integer parsing limit is unavailable")
    def test_integer_parse_limit_does_not_terminate_the_server(self):
        request = b'{"jsonrpc":"2.0","id":' + b"1" * 5000 + b',"method":"ping"}\n'
        responses = self.exchange(request + ping(18))
        self.assertEqual([None, 18], [response["id"] for response in responses])
        self.assertEqual(-32700, responses[0]["error"]["code"])
        self.assertEqual({}, responses[1]["result"])

    def test_nonfinite_numbers_are_rejected_without_losing_the_session(self):
        for constant in (b"NaN", b"Infinity", b"-Infinity", b"1e9999"):
            with self.subTest(constant=constant):
                request = b'{"jsonrpc":"2.0","id":' + constant + b',"method":"ping"}\n'
                responses = self.exchange(request + ping(18))
                self.assertEqual([None, 18], [response["id"] for response in responses])
                self.assertEqual(-32700, responses[0]["error"]["code"])
                self.assertEqual({}, responses[1]["result"])

    def test_excessive_json_nesting_does_not_terminate_the_server(self):
        request = b'{"jsonrpc":"2.0","id":17,"method":"ping","params":{"nested":' + b"[" * 30000 + b"0" + b"]" * 30000 + b'}}\n'
        responses = self.exchange(request + ping(18))
        self.assertEqual(2, len(responses))
        # Parser recursion limits differ across supported Python versions.
        if "error" in responses[0]:
            self.assertIsNone(responses[0]["id"])
            self.assertEqual(-32700, responses[0]["error"]["code"])
        else:
            self.assertEqual({"jsonrpc": "2.0", "id": 17, "result": {}}, responses[0])
        self.assertEqual({"jsonrpc": "2.0", "id": 18, "result": {}}, responses[1])

    def test_complete_frame_at_the_size_limit_is_accepted(self):
        request = ping(17)
        responses = self.exchange(b" " * (FRAME_LIMIT - len(request)) + request + ping(18))
        self.assertEqual([17, 18], [response["id"] for response in responses])
        self.assertTrue(all(response["result"] == {} for response in responses))

    def test_oversized_unterminated_frame_is_rejected_once_at_eof(self):
        responses = self.exchange(b"x" * 1000000)
        self.assertEqual([None], [response["id"] for response in responses])
        self.assertEqual(-32600, responses[0]["error"]["code"])

    def test_invalid_request_envelopes_are_rejected_without_losing_the_session(self):
        for request in (
            {"jsonrpc": "2.0", "id": [], "method": "ping"},
            {"jsonrpc": "2.0", "id": {}, "method": "ping"},
            {"jsonrpc": "2.0", "id": True, "method": "ping"},
            {"jsonrpc": "2.0", "id": 1},
            {"jsonrpc": "2.0", "method": ["ping"]},
            {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": "invalid"},
        ):
            with self.subTest(request=request):
                responses = self.exchange(json.dumps(request).encode() + b"\n" + ping(18))
                self.assertEqual([None, 18], [response["id"] for response in responses])
                self.assertEqual(-32600, responses[0]["error"]["code"])
                self.assertEqual({}, responses[1]["result"])

    def test_string_and_integer_ids_and_notifications_keep_their_response_contract(self):
        notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode() + b"\n"
        responses = self.exchange(ping("client-a") + notification + ping(0) + ping(-1))
        self.assertEqual(["client-a", 0, -1], [response["id"] for response in responses])
        self.assertTrue(all(response["result"] == {} for response in responses))


if __name__ == "__main__":
    unittest.main()
