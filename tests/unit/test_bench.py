import io
import json
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
from unittest.mock import patch

from mlx_agent import cli
from mlx_agent.bench import (
    BENCH_PROBE_ID,
    BENCH_PROMPT,
    BenchError,
    measure_runtime,
    validate_bench_bounds,
)
from mlx_agent.verification import EvidenceStrength, VerificationStatus


class FakeClock:
    def __init__(self, step=0.05):
        self.value = 1000.0
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


class FakeStreamingRuntime:
    name = "fake-stream"

    def __init__(self, installed, events=None, include_ollama_usage=False,
                 include_openai_usage=False):
        self.installed = list(installed)
        self.events = events if events is not None else [
            {"choices": [{"delta": {"content": "word"}}]} for _ in range(10)
        ]
        self.include_ollama_usage = include_ollama_usage
        self.include_openai_usage = include_openai_usage
        self.calls = 0

    def list_models(self):
        return list(self.installed)

    def stream_generate(self, model, prompt, max_tokens, timeout=120.0):
        self.calls += 1
        for event in self.events:
            yield event
        if self.include_ollama_usage:
            yield {
                "message": {"content": ""},
                "done": True,
                "eval_count": 100,
                "eval_duration": int(2e9),
                "prompt_eval_count": 96,
                "prompt_eval_duration": int(0.5e9),
            }
        if self.include_openai_usage:
            yield {
                "choices": [{"delta": {}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 10},
            }


class NonStreamingRuntime:
    name = "no-stream"

    def list_models(self):
        return ["pub/model"]


class BenchBoundsTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        validate_bench_bounds(3, 128, 120.0)

    def test_runs_out_of_range(self):
        with self.assertRaises(ValueError):
            validate_bench_bounds(0, 128, 120.0)
        with self.assertRaises(ValueError):
            validate_bench_bounds(11, 128, 120.0)

    def test_gen_tokens_out_of_range(self):
        with self.assertRaises(ValueError):
            validate_bench_bounds(3, 8, 120.0)

    def test_timeout_out_of_range(self):
        with self.assertRaises(ValueError):
            validate_bench_bounds(3, 128, 0.1)


class BenchMeasurementTests(unittest.TestCase):
    def test_model_not_serving_is_classified(self):
        runtime = FakeStreamingRuntime(["other/model"])
        with self.assertRaises(BenchError) as caught:
            measure_runtime("pub/model", runtime)
        self.assertEqual(caught.exception.code, "model_not_serving")

    def test_unsupported_runtime_is_classified(self):
        with self.assertRaises(BenchError) as caught:
            measure_runtime("pub/model", NonStreamingRuntime())
        self.assertEqual(caught.exception.code, "unsupported_runtime")

    def test_invalid_repo_is_classified(self):
        with self.assertRaises(BenchError) as caught:
            measure_runtime("  ", FakeStreamingRuntime(["pub/model"]))
        self.assertEqual(caught.exception.code, "invalid_repo")

    def test_event_count_fallback_measurement(self):
        clock = FakeClock(step=0.05)
        runtime = FakeStreamingRuntime(["pub/model"])
        measurement = measure_runtime(
            "pub/model", runtime, runs=2, gen_tokens=64, timeout=120.0,
            chip="Apple M5 Max", clock=clock,
        )
        self.assertEqual(measurement.runs, 2)
        self.assertEqual(measurement.runtime, "fake-stream")
        self.assertGreater(measurement.decode_toks, 0)
        self.assertIsNotNone(measurement.ttft_ms)
        self.assertEqual(measurement.chip, "Apple M5 Max")
        self.assertEqual(len(measurement.samples), 2)
        self.assertEqual(runtime.calls, 3)

    def test_ollama_usage_fields_drive_measurement(self):
        clock = FakeClock(step=0.01)
        runtime = FakeStreamingRuntime(["pub/model"], include_ollama_usage=True)
        measurement = measure_runtime(
            "pub/model", runtime, runs=1, gen_tokens=128, timeout=120.0, clock=clock
        )
        self.assertEqual(measurement.decode_toks, 50.0)
        self.assertEqual(measurement.prefill_toks, 192.0)

    def test_openai_usage_prompt_tokens_beat_the_estimate(self):
        clock = FakeClock(step=0.05)
        runtime = FakeStreamingRuntime(["pub/model"], include_openai_usage=True)
        measurement = measure_runtime(
            "pub/model", runtime, runs=1, gen_tokens=64, timeout=120.0, clock=clock
        )
        # The server-reported count (120) replaces the fixed estimate (96)...
        self.assertEqual(measurement.prompt_tokens, 120)
        self.assertEqual(measurement.samples[0]["prompt_tokens"], 120)
        # ...and prefill derives from the real count over measured TTFT:
        # first content lands one 0.05s step in, so 120 / 0.05 = 2400.
        self.assertEqual(measurement.samples[0]["prefill_toks"], 2400.0)
        self.assertEqual(measurement.prefill_toks, 2400.0)

    def test_empty_stream_is_classified(self):
        runtime = FakeStreamingRuntime(["pub/model"], events=[])
        with self.assertRaises(BenchError) as caught:
            measure_runtime("pub/model", runtime, runs=1, timeout=120.0)
        self.assertEqual(caught.exception.code, "empty_stream")

    def test_evidence_uses_runtime_measured_strength(self):
        clock = FakeClock()
        runtime = FakeStreamingRuntime(["pub/model"])
        measurement = measure_runtime(
            "pub/model", runtime, runs=1, timeout=120.0, clock=clock
        )
        evidence = measurement.to_evidence(role="coding")
        self.assertEqual(evidence.strength, EvidenceStrength.RUNTIME_MEASURED)
        self.assertEqual(evidence.status, VerificationStatus.VERIFIED)
        self.assertEqual(evidence.details["probe_id"], BENCH_PROBE_ID)
        record = evidence.to_dict()
        self.assertEqual(record["strength"], "runtime_measured")
        self.assertEqual(record["details"]["outcome"], "measured")

    def test_spread_is_zero_for_single_run(self):
        clock = FakeClock()
        runtime = FakeStreamingRuntime(["pub/model"])
        measurement = measure_runtime(
            "pub/model", runtime, runs=1, timeout=120.0, clock=clock
        )
        self.assertEqual(measurement.spread_pct, 0.0)


def _shared_prefix(left, right):
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


class _LoopbackRuntime(ThreadingHTTPServer):
    """An OpenAI-compatible server that records each prompt and the peak number of open streams."""

    daemon_threads = True

    def __init__(self, delay=0.0):
        super().__init__(("127.0.0.1", 0), _LoopbackHandler)
        self.prompts, self.lock, self.active, self.peak, self.delay = [], threading.Lock(), 0, 0, delay

    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class _LoopbackHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "org/model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        prompt = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["messages"][-1]["content"]
        server = self.server
        with server.lock:
            # Like a prompt cache: reuse the longest prefix shared with an earlier prompt, at 4 characters a token.
            cached = max((_shared_prefix(prompt, seen) for seen in server.prompts), default=0) // 4
            server.prompts.append(prompt)
            server.active += 1
            server.peak = max(server.peak, server.active)
        try:
            time.sleep(server.delay)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for _ in range(4):
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"tok"}}]}\n\n')
                self.wfile.flush()
                time.sleep(0.01)
            usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": 4,
                     "prompt_tokens_details": {"cached_tokens": cached}}
            self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        finally:
            with server.lock:
                server.active -= 1


class BenchWorkloadTests(unittest.TestCase):
    def start(self, delay=0.0):
        self.runtime = _LoopbackRuntime(delay)
        thread = threading.Thread(target=self.runtime.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.runtime.server_close)
        self.addCleanup(self.runtime.shutdown)

    def bench(self, *extra):
        output = io.StringIO()
        base = "http://127.0.0.1:{0}".format(self.runtime.server_address[1])
        with redirect_stdout(output), patch("mlx_agent.host.HostInventory.inspect", side_effect=RuntimeError("no host")):
            code = cli.main(["bench", "run", "--repo", "org/model", "--runtime", "mlx_lm", "--base-url", base,
                             "--runs", "2", "--json", *extra])
        return code, json.loads(output.getvalue())

    def test_base_url_measures_the_runtime_at_that_origin(self):
        self.start()
        code, result = self.bench()
        self.assertEqual(code, 0, result)
        self.assertEqual(self.runtime.prompts, [BENCH_PROMPT] * 3)
        self.assertEqual(result["data"]["measurement"]["workload"], "chat")
        self.assertEqual(result["data"]["evidence"]["strength"], "runtime_measured")

    def test_prefix_workload_repeats_a_long_shared_prefix_and_reports_cached_tokens(self):
        self.start()
        code, result = self.bench("--workload", "prefix")
        self.assertEqual(code, 0, result)
        first, *timed = self.runtime.prompts
        self.assertEqual(len(timed), 2)
        shared = min(_shared_prefix(first, prompt) for prompt in timed)
        self.assertGreater(shared, 60000)
        self.assertTrue(all(len(prompt) - shared < 100 for prompt in self.runtime.prompts))
        self.assertEqual(len(set(self.runtime.prompts)), 3)
        measurement = result["data"]["measurement"]
        self.assertGreater(measurement["prompt_tokens"], 15000)
        self.assertGreater(measurement["cached_tokens"], 15000)
        self.assertIsNotNone(measurement["first_ttft_ms"])
        self.assertIsNone(result["data"]["evidence"])
        # A later invocation starts from a prefix the server has not seen.
        self.bench("--workload", "prefix")
        self.assertLess(_shared_prefix(first, self.runtime.prompts[3]), 100)

    def test_prefill_workload_sends_long_prompts_without_a_shared_prefix(self):
        self.start()
        code, result = self.bench("--workload", "prefill")
        self.assertEqual(code, 0, result)
        prompts = self.runtime.prompts
        self.assertEqual(len(prompts), 3)
        self.assertTrue(all(_shared_prefix(a, b) < 100 for a in prompts for b in prompts if a is not b))
        measurement = result["data"]["measurement"]
        self.assertGreater(measurement["prompt_tokens"], 30000)
        self.assertLess(measurement["cached_tokens"], 30)

    def test_concurrent_workload_keeps_the_requested_streams_open_together(self):
        self.start(delay=0.3)
        code, result = self.bench("--workload", "concurrent", "--concurrency", "4")
        self.assertEqual(code, 0, result)
        self.assertEqual(self.runtime.peak, 4)
        self.assertEqual(len(self.runtime.prompts), 12)
        measurement = result["data"]["measurement"]
        self.assertEqual(measurement["concurrency"], 4)
        self.assertEqual([len(round_["streams"]) for round_ in measurement["samples"]], [4, 4])
        for round_ in measurement["samples"]:
            # Four streams of four tokens each over the round's wall time.
            self.assertAlmostEqual(round_["aggregate_decode_toks"], 16 / (round_["elapsed_ms"] / 1000), delta=1)
        self.assertGreater(measurement["aggregate_decode_toks"], 0)

    def test_workload_arguments_are_refused_where_they_do_not_apply(self):
        self.start()
        for extra in (("--workload", "prefix", "--export", "out.jsonl"), ("--workload", "cold"),
                      ("--workload", "concurrent", "--concurrency", "9")):
            code, result = self.bench(*extra)
            self.assertEqual(code, 2, extra)
            self.assertEqual(result["error"]["code"], "invalid_arguments")
        self.assertEqual(self.runtime.prompts, [])


if __name__ == "__main__":
    unittest.main()
