import json
import unittest

from mlx_agent.mcp_server import TOOL_NAME, handle_request


class MCPServerTests(unittest.TestCase):
    def test_tools_list_exposes_one_constrained_execution_tool(self):
        response = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = response["result"]["tools"]
        self.assertEqual([TOOL_NAME], [item["name"] for item in tools])
        schema = tools[0]["inputSchema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(["capability", "arguments"], schema["required"])

    def test_valid_tool_call_reaches_core_as_an_argv_array_and_propagates_nonzero(self):
        calls = []

        def core(argv):
            calls.append(argv)
            print("bounded core output")
            return 7

        response = handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": TOOL_NAME,
                "arguments": {"capability": "scout", "arguments": "--limit 1 --json"},
            },
        }, core=core)
        self.assertEqual([["discover", "--limit", "1", "--json"]], calls)
        self.assertTrue(response["result"]["isError"])
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual("error", payload["status"])
        self.assertEqual(7, payload["exit_code"])
        self.assertIn("bounded core output", payload["stdout"])

    def test_hostile_tool_arguments_are_rejected_before_core(self):
        calls = []
        cases = (
            ("scout", "--limit 1; touch owned"),
            (
                "wire",
                "apply mlx-community/Test-4bit --path providers.json --endpoint http://[::1",
            ),
            ("scout", "\ud800"),
        )
        for capability, arguments in cases:
            with self.subTest(capability=capability, arguments=repr(arguments)):
                response = handle_request({
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": TOOL_NAME,
                        "arguments": {"capability": capability, "arguments": arguments},
                    },
                }, core=lambda argv: calls.append(argv))
                self.assertEqual(-32602, response["error"]["code"])
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
