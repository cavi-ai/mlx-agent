import copy
import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mlx_agent import cli
from mlx_agent.bench import BenchError
from mlx_agent.coding_bench import check_edit, run_coding_workload
from mlx_agent.command_args import CommandArgumentError, parse_command_arguments
from mlx_agent.verification import _parse_stream_line


FIRST = "def clamp(value, lower, upper):\n    return max(lower, min(value, upper))\n"
SECOND = ("def clamp(value, lower, upper):\n"
          "    if lower > upper:\n        raise ValueError('reversed bounds')\n"
          "    return max(lower, min(value, upper))\n")


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class ChatRuntime:
    name = "mlx_lm"

    def __init__(self, clock, sources=(FIRST, SECOND), path="clamp.py", usage=True):
        self.clock, self.sources, self.path, self.usage = clock, sources, path, usage
        self.requests = []

    def list_models(self):
        return {"data": [{"id": "org/model"}]}

    def stream_chat(self, model, messages, max_tokens, **options):
        self.requests.append(copy.deepcopy({"model": model, "messages": messages,
                                            "max_tokens": max_tokens, **options}))
        index = len(self.requests) - 1
        self.clock.value += 0.1
        yield {"prompt_progress": {"processed": 0, "total": 100, "source": "mlx_lm_keepalive"}}
        self.clock.value += 0.1
        yield {"prompt_progress": {"processed": 100, "total": 100, "source": "mlx_lm_keepalive"}}
        if index == 0:
            arguments = json.dumps({"path": self.path})
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1",
                "type": "function", "function": {"name": "read_file", "arguments": arguments[:5]}}]}}]}
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0,
                "function": {"arguments": arguments[5:]}}]}}]}
        else:
            yield {"choices": [{"delta": {"content": self.sources[index - 1]}}]}
        self.clock.value += 0.1
        if self.usage:
            yield {"choices": [], "usage": {"prompt_tokens": 100 + index * 100,
                                               "completion_tokens": 20}}


class EditQualityTests(unittest.TestCase):
    def test_boundary_fix_and_followup_are_checked_by_running_fixed_cases(self):
        self.assertTrue(check_edit(FIRST)["passed"])
        self.assertFalse(check_edit("def clamp(value, lower, upper):\n    return min(value, upper)")["passed"])
        self.assertFalse(check_edit(FIRST, reversed_bounds=True)["passed"])
        result = check_edit(SECOND, reversed_bounds=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["cases_passed"], result["cases_total"])

    def test_unsupported_code_is_rejected_before_execution(self):
        sources = ["import os\n" + FIRST,
                   "def clamp(value, lower, upper):\n    while True: pass",
                   "def clamp(value, lower, upper):\n    return clamp(value, lower, upper)",
                   "def clamp(value, lower, upper):\n    return value.__class__",
                   "def clamp(value, lower, upper=open('secret')):\n    return value",
                   "@print('side effect')\n" + FIRST,
                   "def clamp(value, lower, upper):\n    return 'x' * 1000000",
                   "def clamp(value, lower, upper):\n    return open('secret').read()"]
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(check_edit(source)["reason"], "unsupported_code")

    def test_oversized_or_multiple_code_blocks_are_not_partially_accepted(self):
        self.assertFalse(check_edit("x" * 5000)["passed"])
        self.assertFalse(check_edit("```python\n" + FIRST + "```\n```python\n" + SECOND + "```")["passed"])
        self.assertTrue(check_edit("```python\n" + FIRST + "```")["passed"])


class CodingWorkloadTests(unittest.TestCase):
    def test_tool_call_actual_source_and_test_feedback_form_one_conversation(self):
        clock = Clock()
        runtime = ChatRuntime(clock)
        result = run_coding_workload("org/model", runtime, clock=clock)
        self.assertTrue(result["quality_passed"])
        self.assertEqual(len(runtime.requests), 3)
        first, edit, followup = runtime.requests
        self.assertEqual(first["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(edit["messages"][-1]["role"], "tool")
        self.assertEqual(edit["messages"][-1]["tool_call_id"], "call-1")
        self.assertEqual(json.loads(edit["messages"][-1]["content"])["path"], "clamp.py")
        self.assertEqual(followup["messages"][-2], {"role": "assistant", "content": FIRST})
        feedback = json.loads(followup["messages"][-1]["content"])
        self.assertTrue(feedback["test_results"]["passed"])
        self.assertTrue(all(r["temperature"] == 0 and r["seed"] == 0 for r in runtime.requests))
        self.assertEqual([v["cached_tokens"] for v in result["turns"]], [0, 100, 200])
        self.assertEqual(result["turns"][1]["cache_source"], "mlx_lm_keepalive")

    def test_invalid_read_request_stops_without_reading_another_path(self):
        clock = Clock()
        runtime = ChatRuntime(clock, path="../../secret")
        result = run_coding_workload("org/model", runtime, clock=clock)
        self.assertFalse(result["quality_passed"])
        self.assertEqual(result["reason"], "invalid_tool_call")
        self.assertEqual(len(runtime.requests), 1)

    def test_incorrect_edit_cannot_be_reported_as_a_quality_passing_run(self):
        clock = Clock()
        runtime = ChatRuntime(clock, sources=("def clamp(value, lower, upper):\n    return upper", SECOND))
        result = run_coding_workload("org/model", runtime, clock=clock)
        self.assertFalse(result["quality_passed"])
        self.assertFalse(result["turns"][1]["quality"]["passed"])

    def test_unknown_usage_does_not_become_token_or_cache_counts(self):
        clock = Clock()
        result = run_coding_workload("org/model", ChatRuntime(clock, usage=False), clock=clock)
        self.assertTrue(result["quality_passed"])
        self.assertTrue(all(v["completion_tokens"] is None and v["cached_tokens"] is None for v in result["turns"]))

    def test_one_total_deadline_limits_the_whole_workflow(self):
        clock = Clock()
        runtime = ChatRuntime(clock)
        with self.assertRaises(BenchError) as caught:
            run_coding_workload("org/model", runtime, timeout=0.5, clock=clock)
        self.assertEqual(caught.exception.code, "measurement_timeout")
        self.assertLessEqual(len(runtime.requests), 2)

    def test_bad_progress_does_not_manufacture_cache_hits(self):
        class BadProgress(ChatRuntime):
            def stream_chat(self, *args, **kwargs):
                for event in super().stream_chat(*args, **kwargs):
                    yield event
                    if "prompt_progress" in event:
                        yield {"prompt_progress": {"processed": 50, "total": 100, "source": "mlx_lm_keepalive"}}
        clock = Clock()
        result = run_coding_workload("org/model", BadProgress(clock), clock=clock)
        self.assertTrue(all(t["cached_tokens"] is None for t in result["turns"]))

    def test_inventory_and_stream_failures_are_classified(self):
        for method, error, code in (("list_models", OSError("offline"), "runtime_unreachable"),
                                    ("stream_chat", OSError("closed"), "measurement_failed"),
                                    ("stream_chat", TimeoutError(), "measurement_timeout")):
            runtime = ChatRuntime(Clock())
            with patch.object(runtime, method, side_effect=error), self.assertRaises(BenchError) as caught:
                run_coding_workload("org/model", runtime)
            self.assertEqual(caught.exception.code, code)


class CodingTransportTests(unittest.TestCase):
    def test_native_progress_is_explicit_opt_in_and_bounded(self):
        line = b": keepalive 10/100"
        self.assertEqual(_parse_stream_line(line), (None, False))
        value, done = _parse_stream_line(line, include_progress=True)
        self.assertEqual(value["prompt_progress"], {"processed": 10, "total": 100, "source": "mlx_lm_keepalive"})
        self.assertFalse(done)
        for invalid in (b": keepalive 101/100", b": keepalive -1/100", b": keepalive 0/999999999999"):
            value, _ = _parse_stream_line(invalid, include_progress=True)
            self.assertTrue(value["prompt_progress"]["invalid"])

    def test_agent_argument_parser_exposes_only_bounded_coding_options(self):
        args = parse_command_arguments("bench", "coding --repo org/model --base-url http://127.0.0.1:18890 --gen-tokens 512 --timeout 120 --json")
        self.assertEqual(args[:2], ["bench", "coding"])
        with self.assertRaises(CommandArgumentError):
            parse_command_arguments("bench", "coding --repo org/model --gen-tokens 999999 --json")


class CodingCliTests(unittest.TestCase):
    def run_cli(self, *extra, sources=(FIRST, SECOND)):
        output = io.StringIO()
        with redirect_stdout(output), patch("mlx_agent.cli._bench_runtime_client", return_value=ChatRuntime(Clock(), sources=sources)):
            code = cli.main(["bench", "coding", "--repo", "org/model", "--json", *extra])
        return code, json.loads(output.getvalue())

    def test_trace_is_private_exclusive_and_not_in_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "trace.json"
            code, value = self.run_cli("--out", str(destination))
            self.assertEqual(code, 0, value)
            self.assertNotIn("transcript", value["data"])
            self.assertEqual(destination.stat().st_mode & 0o777, 0o600)
            original = destination.read_bytes()
            self.assertEqual(len(json.loads(original)["transcript"]), 3)
            code, value = self.run_cli("--out", str(destination))
            self.assertEqual(code, 2)
            self.assertEqual(destination.read_bytes(), original)

    def test_quality_failure_has_a_distinct_exit_status(self):
        code, value = self.run_cli(sources=("def clamp(value, lower, upper):\n    return upper", SECOND))
        self.assertEqual(code, 1)
        self.assertFalse(value["data"]["quality_passed"])

    def test_direct_cli_rejects_invalid_bounds_before_contacting_the_runtime(self):
        for extra in (("--gen-tokens", "999999"), ("--timeout", "nan")):
            with patch("mlx_agent.coding_bench.run_coding_workload") as measure:
                code, value = self.run_cli(*extra)
            self.assertEqual(code, 2, value)
            measure.assert_not_called()
