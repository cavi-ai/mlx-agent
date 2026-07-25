import unittest

from mlx_agent.bench import (
    BENCH_PROBE_ID,
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

    def __init__(self, installed, events=None, include_ollama_usage=False):
        self.installed = list(installed)
        self.events = events if events is not None else [
            {"choices": [{"delta": {"content": "word"}}]} for _ in range(10)
        ]
        self.include_ollama_usage = include_ollama_usage
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


if __name__ == "__main__":
    unittest.main()
