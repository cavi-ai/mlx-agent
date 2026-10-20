import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jsonschema import Draft202012Validator

from mlx_agent.task_store import TaskStore

from mlx_agent.mcp_server import TOOL_NAME, handle_request
from tests.task_fixtures import identity_evidence


class MCPServerTests(unittest.TestCase):
    def test_initialize_negotiates_only_supported_protocol_versions(self):
        for requested, expected in (
            ("2024-11-05", "2024-11-05"),
            ("2025-03-26", "2025-03-26"),
            ("2025-06-18", "2025-06-18"),
            ("2099-01-01", "2025-06-18"),
        ):
            with self.subTest(requested=requested):
                response = handle_request({
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": requested},
                })
                self.assertEqual(expected, response["result"]["protocolVersion"])

    def test_catalog_describes_arguments_and_declares_possible_mutation(self):
        tool = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"][0]
        self.assertFalse(tool["annotations"]["readOnlyHint"])
        self.assertTrue(tool["annotations"]["destructiveHint"])
        self.assertFalse(tool["annotations"]["idempotentHint"])
        self.assertTrue(tool["annotations"]["openWorldHint"])
        for property_schema in tool["inputSchema"]["properties"].values():
            self.assertTrue(property_schema["description"])
            self.assertEqual("string", property_schema["type"])

    def test_real_capability_result_matches_output_schema_and_legacy_text(self):
        tool = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"][0]
        result = self.task_call("capabilities")["result"]
        payload = result["structuredContent"]
        Draft202012Validator.check_schema(tool["outputSchema"])
        Draft202012Validator(tool["outputSchema"]).validate(payload)
        self.assertEqual(payload, json.loads(result["content"][0]["text"]))
        self.assertEqual("ok", payload["status"])
        self.assertEqual(0, payload["exit_code"])
        self.assertFalse(result["isError"])
        self.assertEqual("task.capabilities", json.loads(payload["stdout"])["operation"])

    def test_cli_exit_is_a_tool_result_and_does_not_end_the_session(self):
        for code, expected in ((None, 0), (0, 0), (2, 2), (True, 1), ("failed", 1)):
            with self.subTest(code=code):
                def core(argv):
                    print("CLI output")
                    raise SystemExit(code)

                result = self.task_call("capabilities", core=core)["result"]
                self.assertEqual(expected != 0, result["isError"])
                self.assertEqual(expected, result["structuredContent"]["exit_code"])
                self.assertIs(int, type(result["structuredContent"]["exit_code"]))
                self.assertIn("CLI output", result["structuredContent"]["stdout"])
                self.assertEqual({}, handle_request({"jsonrpc": "2.0", "id": 3, "method": "ping"})["result"])

    def task_call(self, arguments, core=None):
        return handle_request({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": TOOL_NAME, "arguments": {"capability": "task", "arguments": arguments}}}, core=core)

    def test_task_capability_is_discoverable_and_uses_real_cli(self):
        response = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertIn("task", response["result"]["tools"][0]["inputSchema"]["properties"]["capability"]["enum"])
        response = self.task_call("capabilities")
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertFalse(payload["stdout_truncated"])
        capabilities = json.loads(payload["stdout"])
        self.assertEqual(capabilities["operation"], "task.capabilities")
        tasks = {task["kind"]: task for task in capabilities["data"]["tasks"]}
        self.assertEqual(set(tasks), {"classification", "music_generation", "transcribe", "speak", "image_generation", "image_understanding",
                                      "video_generation", "video_understanding"})
        self.assertEqual(tasks["transcribe"]["request_schema"], "task-request/2")
        self.assertEqual(tasks["image_generation"]["run_schema"], "task-run/2")
        self.assertEqual(tasks["video_generation"]["result_schema"], "video/1")

    def test_task_paths_ids_results_and_cancellation_use_the_same_store(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "tasks with spaces"
            store = TaskStore(root)
            request = {"schema": "task-request/1", "kind": "music_generation", "model_path": "/model", "parameters": {"caption": "piano"}}
            record = store.create(request, identity_evidence(request["model_path"]))
            for action in ("status", "cancel"):
                response = self.task_call('{0} --id {1} --state-dir "{2}" --json'.format(action, record["id"], root))
                payload = json.loads(response["result"]["content"][0]["text"])
                envelope = json.loads(payload["stdout"])
                self.assertEqual(envelope["data"]["task"]["id"], record["id"])
                self.assertEqual(payload["exit_code"], 0)
            self.assertTrue(store.get(record["id"])["cancel_requested"])
            response = self.task_call('list --state-dir "{0}" --limit 1'.format(root))
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertEqual(json.loads(payload["stdout"])["data"]["tasks"][0]["id"], record["id"])

    def test_large_task_result_is_a_valid_artifact_reference_not_truncated_json(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "tasks"
            store = TaskStore(root)
            request = {"schema": "task-request/1", "kind": "classification", "model_path": "/model", "parameters": {"state": "x", "questions": {"score": {"type": "score", "instructions": "score"}}}}
            record = store.create(request, identity_evidence(request["model_path"]))
            store.transition(record["id"], ("starting",), {"supervisor": {"pid": 42, "identity": "supervisor"}})
            record = store.transition(record["id"], ("starting",), {
                "state": "running", "execution": {"pid": 43, "identity": "worker"}})
            reference = store.write_result(record["id"], {"answers": {"text": "x" * 30000}})
            store.transition(record["id"], ("running",), {"state": "succeeded", "result": reference})
            response = self.task_call('result --id {0} --state-dir "{1}"'.format(record["id"], root))
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertFalse(payload["stdout_truncated"])
            data = json.loads(payload["stdout"])["data"]
            self.assertFalse(data["inline"])
            self.assertEqual(data["artifact"], reference)

    def test_task_grammar_rejects_malformed_ids_paths_and_shell_tokens(self):
        calls = []
        for arguments in ("status --id bad", "list --limit 101", "list --state-dir relative", "start --request ../request.json",
                          "capabilities; echo bad", "status --id $(id)", "cancel --id 11111111-1111-4111-8111-111111111111 --id 11111111-1111-4111-8111-111111111111"):
            with self.subTest(arguments=arguments):
                response = self.task_call(arguments, core=lambda argv: calls.append(argv))
                self.assertEqual(response["error"]["code"], -32602)
        self.assertEqual(calls, [])

    def test_task_start_errors_keep_the_cli_error_envelope(self):
        with TemporaryDirectory() as directory:
            request = Path(directory).resolve() / "missing.json"
            response = self.task_call('start --request "{0}"'.format(request))
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertEqual(payload["exit_code"], 2)
            self.assertEqual(json.loads(payload["stdout"])["error"]["code"], "invalid_request")

    def test_tools_list_exposes_execution_and_read_only_capabilities(self):
        response = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = response["result"]["tools"]
        self.assertEqual([TOOL_NAME, "mlx_agent_capabilities"], [item["name"] for item in tools])
        schema = tools[0]["inputSchema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(["capability", "arguments"], schema["required"])

    def test_capabilities_tool_only_runs_fixed_read_only_discovery(self):
        tool = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"][-1]
        self.assertTrue(tool["annotations"]["readOnlyHint"])
        self.assertFalse(tool["annotations"]["destructiveHint"])
        self.assertFalse(tool["annotations"]["openWorldHint"])
        calls = []
        for arguments in ({"arguments": "cancel"}, {"capability": "task"}, [], None):
            response = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "mlx_agent_capabilities", "arguments": arguments}}, core=lambda argv: calls.append(argv))
            self.assertEqual(-32602, response["error"]["code"])
            self.assertFalse(response["error"]["data"]["retryable"])
        self.assertEqual([], calls)
        response = handle_request({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "mlx_agent_capabilities", "arguments": {}}})
        result = response["result"]
        Draft202012Validator(tool["outputSchema"]).validate(result["structuredContent"])
        self.assertEqual("task.capabilities", json.loads(result["structuredContent"]["stdout"])["operation"])

    def test_invalid_arguments_declare_permanent_failure_in_error_data(self):
        response = self.task_call("capabilities --unsupported")
        self.assertEqual(-32602, response["error"]["code"])
        self.assertEqual({"retryable": False}, response["error"]["data"])

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
