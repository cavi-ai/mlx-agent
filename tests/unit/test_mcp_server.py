import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.task_store import TaskStore

from mlx_agent.mcp_server import TOOL_NAME, handle_request
from tests.task_fixtures import identity_evidence


class MCPServerTests(unittest.TestCase):
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
