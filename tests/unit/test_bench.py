import unittest

from mlx_agent.bench import (
    BENCH_PROBE_ID,
    BenchError,
    _timed_run,
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

    def test_stream_without_token_usage_reports_latency_only(self):
        clock = FakeClock(step=0.05)
        runtime = FakeStreamingRuntime(["pub/model"])
        measurement = measure_runtime(
            "pub/model", runtime, runs=2, gen_tokens=64, timeout=120.0,
            chip="Apple M5 Max", clock=clock,
        )
        self.assertEqual(measurement.runs, 2)
        self.assertEqual(measurement.runtime, "fake-stream")
        self.assertIsNone(measurement.decode_toks)
        self.assertIsNone(measurement.end_to_end_toks)
        self.assertIsNone(measurement.prompt_tokens)
        self.assertEqual(measurement.measurement_method, "latency_only")
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

    def test_openai_usage_does_not_turn_ttft_into_prefill_speed(self):
        clock = FakeClock(step=0.05)
        runtime = FakeStreamingRuntime(["pub/model"], include_openai_usage=True)
        measurement = measure_runtime(
            "pub/model", runtime, runs=1, gen_tokens=64, timeout=120.0, clock=clock
        )
        self.assertEqual(measurement.prompt_tokens, 120)
        self.assertEqual(measurement.samples[0]["prompt_tokens"], 120)
        self.assertIsNone(measurement.decode_toks)
        self.assertIsNone(measurement.prefill_toks)
        self.assertGreater(measurement.end_to_end_toks, 0)
        self.assertEqual(measurement.measurement_method, "client_wall_clock")

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

    def test_spread_is_unknown_without_decode_timing(self):
        clock = FakeClock()
        runtime = FakeStreamingRuntime(["pub/model"])
        measurement = measure_runtime(
            "pub/model", runtime, runs=1, timeout=120.0, clock=clock
        )
        self.assertIsNone(measurement.spread_pct)


class ManualClock:
    value = 0.0

    def __call__(self):
        return self.value


class TimedRuntime:
    name = "timed-runtime"

    def __init__(self, clock, events, elapsed=2.0):
        self.clock, self.events, self.elapsed = clock, events, elapsed
        self.timeouts = []

    def list_models(self):
        return ["pub/model"]

    def stream_generate(self, model, prompt, max_tokens, timeout):
        self.timeouts.append(timeout)
        started = self.clock.value
        for offset, event in self.events:
            self.clock.value = started + offset
            yield event
        self.clock.value = started + self.elapsed


class MetricIntegrityTests(unittest.TestCase):
    def sample(self, events, elapsed=2.0):
        clock = ManualClock()
        runtime = TimedRuntime(clock, events, elapsed)
        return _timed_run(runtime.stream_generate, "pub/model", 128, 10.0, clock)

    def test_multi_token_chunks_use_reported_completion_count_and_full_elapsed(self):
        sample = self.sample([
            (0.5, {"choices": [{"delta": {"content": "many tokens in this chunk"}}]}),
            (1.5, {"choices": [{"delta": {"content": " and more"}}]}),
            (2.0, {"choices": [], "usage": {"completion_tokens": 20, "prompt_tokens": 120}}),
        ])
        self.assertIsNone(sample["decode_toks"])
        self.assertIsNone(sample["prefill_toks"])
        self.assertEqual(sample["end_to_end_toks"], 10.0)
        self.assertEqual(sample["ttft_ms"], 500.0)
        self.assertEqual(sample["elapsed_ms"], 2000.0)
        self.assertEqual(sample["completion_tokens"], 20)

    def test_partial_usage_is_merged_without_losing_counts(self):
        sample = self.sample([
            (0.5, {"choices": [{"delta": {"content": "reply"}}], "usage": {"completion_tokens": 8}}),
            (1.0, {"choices": [], "usage": {"prompt_tokens": 80}}),
        ])
        self.assertEqual(sample["end_to_end_toks"], 4.0)
        self.assertEqual(sample["prompt_tokens"], 80)

    def test_invalid_counts_never_manufacture_token_rates(self):
        for value in (True, -1, 1.5, float("nan"), float("inf"), "12", 10 ** 1000):
            with self.subTest(value=value):
                sample = self.sample([(0.5, {
                    "choices": [{"delta": {"content": "reply"}}],
                    "usage": {"completion_tokens": value, "prompt_tokens": value},
                    "eval_count": value, "eval_duration": 1e9,
                })])
                self.assertIsNone(sample["decode_toks"])
                self.assertIsNone(sample["end_to_end_toks"])
                self.assertIsNone(sample["prompt_tokens"])

    def test_stage_timings_are_separate_from_client_elapsed(self):
        sample = self.sample([(0.5, {
            "message": {"content": "reply"}, "eval_count": 20, "eval_duration": 1e9,
            "prompt_eval_count": 100, "prompt_eval_duration": 0.25e9,
        })])
        self.assertEqual(sample["decode_toks"], 20.0)
        self.assertEqual(sample["prefill_toks"], 400.0)
        self.assertEqual(sample["end_to_end_toks"], 10.0)
        self.assertEqual(sample["measurement_method"], "server_timings")

    def test_invalid_stage_durations_remain_unknown(self):
        for value in (True, 0, -1, float("nan"), float("inf"), 10 ** 1000, 1e-300):
            with self.subTest(value=value):
                sample = self.sample([(0.5, {"message": {"content": "reply"},
                    "eval_count": 20, "eval_duration": value,
                    "prompt_eval_count": 100, "prompt_eval_duration": value})])
                self.assertIsNone(sample["decode_toks"])
                self.assertIsNone(sample["prefill_toks"])
                self.assertEqual(sample["end_to_end_toks"], 10.0)

    def test_prompt_usage_without_output_is_an_empty_stream(self):
        with self.assertRaises(BenchError) as caught:
            self.sample([(0.5, {"choices": [], "usage": {"prompt_tokens": 80}})])
        self.assertEqual(caught.exception.code, "empty_stream")

    def test_cached_prompt_tokens_are_excluded_from_uncached_prefill_rate(self):
        sample = self.sample([(0.5, {"message": {"content": "reply"},
            "prompt_eval_count": 100, "prompt_eval_cached_count": 80,
            "prompt_eval_duration": 0.25e9})])
        self.assertEqual(sample["prompt_tokens"], 100)
        self.assertEqual(sample["prefill_toks"], 80.0)
        self.assertEqual(sample["usage"]["prompt_eval_cached_count"], 80)

    def test_inconsistent_cached_prompt_count_has_no_prefill_rate(self):
        sample = self.sample([(0.5, {"message": {"content": "reply"},
            "prompt_eval_count": 100, "prompt_eval_cached_count": 120,
            "prompt_eval_duration": 0.25e9})])
        self.assertIsNone(sample["prefill_toks"])

    def test_zero_completion_is_distinct_from_missing_usage(self):
        sample = self.sample([(0.5, {"choices": [], "usage": {"completion_tokens": 0, "prompt_tokens": 80}})])
        self.assertEqual(sample["completion_tokens"], 0)
        self.assertEqual(sample["end_to_end_toks"], 0.0)
        self.assertIsNone(sample["ttft_ms"])
        self.assertIsNone(sample["decode_toks"])

    def test_warmup_and_samples_share_one_total_deadline(self):
        clock = ManualClock()
        runtime = TimedRuntime(clock, [(0.1, {"message": {"content": "reply"}})], elapsed=0.2)
        measure_runtime("pub/model", runtime, runs=3, timeout=1.0, clock=clock)
        self.assertEqual(len(runtime.timeouts), 4)
        self.assertAlmostEqual(runtime.timeouts[0], 0.25)
        self.assertAlmostEqual(runtime.timeouts[1], 0.8 / 3)
        self.assertAlmostEqual(runtime.timeouts[2], 0.3)
        self.assertAlmostEqual(runtime.timeouts[3], 0.4)

    def test_stream_exceeding_its_deadline_fails_without_more_runs(self):
        clock = ManualClock()
        runtime = TimedRuntime(clock, [(2.0, {"message": {"content": "late"}})], elapsed=2.0)
        with self.assertRaises(BenchError) as caught:
            measure_runtime("pub/model", runtime, runs=1, timeout=1.0, clock=clock)
        self.assertEqual(caught.exception.code, "measurement_timeout")
        self.assertEqual(len(runtime.timeouts), 1)


if __name__ == "__main__":
    unittest.main()
