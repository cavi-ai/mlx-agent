import io
import json
import types
import unittest
from unittest.mock import patch

from mlx_agent import mcp_server
from mlx_agent.mcp_server import MAX_REQUEST_BYTES, _BoundedWriter, handle_request


def _call(arguments, core=None, name=mcp_server.TOOL_NAME):
    return handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": name, "arguments": arguments}}, core=core)


class McpProtocolTests(unittest.TestCase):
    def test_bounded_writer_truncates_and_marks_the_cut(self):
        writer = _BoundedWriter(limit=5)
        self.assertTrue(writer.writable())
        self.assertEqual(3, writer.write("abc"))
        self.assertEqual(4, writer.write("defg"))
        self.assertTrue(writer.truncated)
        self.assertEqual("abcde\n... [output truncated]", writer.value())

    def test_invalid_envelopes_are_invalid_requests(self):
        for request in ([], {"jsonrpc": "1.0", "id": 1, "method": "ping"},
                        {"jsonrpc": "2.0", "id": float("nan"), "method": "ping"},
                        {"jsonrpc": "2.0", "id": [1], "method": "ping"},
                        {"jsonrpc": "2.0", "id": 1, "method": 7},
                        {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": "x"}):
            with self.subTest(request=request):
                self.assertEqual({"code": -32600, "message": "Invalid Request"}, handle_request(request)["error"])
        self.assertIsNone(handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_methods_negotiate_answer_and_refuse(self):
        initialize = handle_request({"jsonrpc": "2.0", "id": "a", "method": "initialize", "params": {"protocolVersion": "1999-01-01"}})
        self.assertEqual(mcp_server.SUPPORTED_PROTOCOL_VERSIONS[-1], initialize["result"]["protocolVersion"])
        self.assertEqual({}, handle_request({"jsonrpc": "2.0", "id": 2, "method": "ping"})["result"])
        names = [tool["name"] for tool in handle_request({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})["result"]["tools"]]
        self.assertEqual([mcp_server.TOOL_NAME, mcp_server.CAPABILITIES_TOOL_NAME], names)
        self.assertEqual(-32601, handle_request({"jsonrpc": "2.0", "id": 4, "method": "resources/list"})["error"]["code"])
        invalid = {"code": -32602, "message": "Invalid params", "data": {"retryable": False}}
        self.assertEqual(invalid, handle_request({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": [1]})["error"])
        self.assertEqual(invalid, _call({"x": 1}, name=mcp_server.CAPABILITIES_TOOL_NAME)["error"])
        self.assertEqual(invalid, _call({}, name="shell")["error"])

    def test_tool_arguments_are_bounded_text(self):
        invalid = {"code": -32602, "message": "Invalid params", "data": {"retryable": False}}
        for arguments in ("scout --json", {"capability": "scout"}, {"capability": "scout", "arguments": 7},
                          {"capability": "scout", "arguments": "\udcff"},
                          {"capability": "scout", "arguments": "--json " * 1000}):
            with self.subTest(arguments=str(arguments)[:40]):
                self.assertEqual(invalid, _call(arguments, core=lambda argv: 0)["error"])

    def test_core_exit_paths_become_tool_results(self):
        def raising(value):
            def core(_argv):
                raise value
            return core

        cases = ((lambda argv: print("ok"), 0, "ok\n", None), (raising(SystemExit(None)), 0, "", None),
                 (raising(SystemExit(3)), 3, "", None), (raising(SystemExit("usage")), 1, "", None),
                 (raising(RuntimeError("boom")), 1, "", "MLX-agent execution failed"))
        for core, exit_code, stdout, error in cases:
            with self.subTest(exit_code=exit_code, error=error):
                result = _call({"capability": "scout", "arguments": "--json"}, core=core)["result"]
                payload = result["structuredContent"]
                self.assertEqual((exit_code, stdout, exit_code != 0), (payload["exit_code"], payload["stdout"], result["isError"]))
                self.assertEqual(error, payload.get("error"))

    def test_stdio_loop_rejects_oversized_unterminated_and_malformed_frames(self):
        frames = b"".join((
            b"x" * (MAX_REQUEST_BYTES + 10) + b"\n",
            b"{bad json\n",
            b'{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"n": NaN}}\n',
            b'{"jsonrpc": "2.0", "method": "notifications/initialized"}\n',
            b'{"jsonrpc": "2.0", "id": 2, "method": "ping"}\n',
            b'{"jsonrpc": "2.0", "id": 3, "method": "ping"}',
        ))
        output = io.StringIO()
        with patch.object(mcp_server.sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(frames))), \
                patch.object(mcp_server.sys, "stdout", output):
            self.assertEqual(0, mcp_server.main())
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([-32600, -32700, -32700], [item["error"]["code"] for item in responses[:3]])
        self.assertEqual({"jsonrpc": "2.0", "id": 2, "result": {}}, responses[3])
        self.assertEqual(-32600, responses[4]["error"]["code"])
        self.assertEqual(5, len(responses))


if __name__ == "__main__":
    unittest.main()
