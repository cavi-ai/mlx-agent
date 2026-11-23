"""Exercise framing and recovery through the real stdio entry point."""

import json
import os
from pathlib import Path
import select
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
FRAME_LIMIT = 65536


def ping(request_id):
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": "ping"}).encode() + b"\n"


def cancel(request_id):
    return json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled",
                       "params": {"requestId": request_id}}).encode() + b"\n"


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

    @staticmethod
    def tool(request_id):
        return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                           "params": {"name": "mlx_agent_capabilities"}}).encode() + b"\n"

    def test_capabilities_execute_through_bundled_cli(self):
        result, = self.exchange(self.tool(1))
        payload = result["result"]["structuredContent"]
        self.assertEqual(payload["exit_code"], 0, payload)
        self.assertEqual(json.loads(payload["stdout"])["operation"], "task.capabilities")

    def exercise_slow_tool(self, directory, timeout):
        code = """
import os, sys
from mlx_agent import mcp_server as server
worker = 'import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)'
execute_cli = server._execute_cli
def execute(argv, cancelled):
    if os.path.exists(sys.argv[1]):
        return execute_cli(argv, cancelled)
    return server._run_process([sys.executable, '-c', worker, sys.argv[1]],
                               cancelled, float(sys.argv[2]), os.environ)
server._execute_cli = execute
raise SystemExit(server.main())
"""
        marker = Path(directory) / "worker.pid"
        process = subprocess.Popen([sys.executable, "-c", code, str(marker), str(timeout)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   bufsize=0, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        self.addCleanup(process.stderr.close)
        self.addCleanup(process.stdout.close)
        self.addCleanup(process.stdin.close)

        def cleanup():
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)

        self.addCleanup(cleanup)
        process.stdin.write(self.tool(1))
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(marker.exists())
        pid = int(marker.read_text())
        return process, pid

    def reply(self, process):
        self.assertTrue(select.select([process.stdout], [], [], 3)[0], "no stdio response")
        return json.loads(process.stdout.readline())

    def test_ping_and_cancel_while_tool_is_executing(self):
        with TemporaryDirectory() as directory:
            process, pid = self.exercise_slow_tool(directory, 15)
            process.stdin.write(ping(2))
            self.assertEqual(self.reply(process), {"jsonrpc": "2.0", "id": 2, "result": {}})
            process.stdin.write(cancel(1))
            process.stdin.close()
            self.assertEqual(process.wait(timeout=3), 0, process.stderr.read())
            self.assertEqual(process.stdout.read(), b"")
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_cancelling_queued_tools_frees_slots_while_the_active_tool_runs(self):
        with TemporaryDirectory() as directory:
            process, pid = self.exercise_slow_tool(directory, 15)
            queued = range(2, 9)
            process.stdin.write(b"".join(self.tool(identifier) for identifier in queued) + self.tool(9))
            rejected = self.reply(process)
            self.assertEqual((rejected["id"], rejected["error"]["code"]), (9, -32000))

            process.stdin.write(b"".join(cancel(identifier) for identifier in queued) + self.tool(11) + ping(10))
            self.assertEqual(self.reply(process), {"jsonrpc": "2.0", "id": 10, "result": {}})
            os.kill(pid, 0)
            process.stdin.write(cancel(1))
            process.stdin.close()
            result = self.reply(process)
            self.assertEqual(result["id"], 11)
            payload = result["result"]["structuredContent"]
            self.assertEqual(json.loads(payload["stdout"])["operation"], "task.capabilities")
            self.assertEqual(process.wait(timeout=3), 0, process.stderr.read())
            self.assertEqual(process.stdout.read(), b"")

    def test_cli_output_written_between_selector_timeout_and_exit_poll_is_captured(self):
        from mlx_agent import mcp_server

        with TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            worker = """
import sys, time
from pathlib import Path
while not Path(sys.argv[1]).exists():
    time.sleep(.01)
print('{"result": "ready"}')
print('final diagnostic', file=sys.stderr)
"""

            class ExitOnPoll(subprocess.Popen):
                def poll(self):
                    release.touch()
                    return self.wait(timeout=3)

            # The real child writes only after the first empty select wakes the parent.
            with patch.object(mcp_server.subprocess, "Popen", ExitOnPoll):
                result = mcp_server._run_process([sys.executable, "-c", worker, str(release)],
                                                 lambda: False, 5, os.environ)
            payload = result["structuredContent"]
            self.assertEqual(payload["exit_code"], 0, payload)
            self.assertEqual(payload["stdout"], '{"result": "ready"}\n')
            self.assertEqual(payload["stderr"], "final diagnostic\n")

    def test_timed_out_tool_reports_failure_and_session_survives(self):
        with TemporaryDirectory() as directory:
            process, pid = self.exercise_slow_tool(directory, 1.5)
            response = self.reply(process)
            self.assertEqual(response["id"], 1)
            self.assertEqual(response["result"]["structuredContent"]["exit_code"], 124)
            self.assertTrue(response["result"]["isError"])
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            process.stdin.write(ping(2))
            self.assertEqual(self.reply(process), {"jsonrpc": "2.0", "id": 2, "result": {}})

    def test_server_shutdown_reaps_the_active_tool(self):
        with TemporaryDirectory() as directory:
            process, pid = self.exercise_slow_tool(directory, 15)
            process.stdin.close()
            process.terminate()
            self.assertEqual(process.wait(timeout=3), 143, process.stderr.read())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
