import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mlx_agent import __version__, cli
from mlx_agent.bench import BenchError


def run(argv, inspect=None):
    output = io.StringIO()
    with redirect_stdout(output), patch("mlx_agent.cli.HostInventory.inspect", side_effect=inspect,
                                        return_value=types.SimpleNamespace(chip="Apple M4")):
        code = cli.main(argv)
    return code, output.getvalue()


class _Evidence:
    def to_dict(self):
        return {"strength": "runtime_measured"}


class _Measurement:
    repo, runtime, runs, decode_toks, spread_pct = "org/model", "mlx_lm", 3, 41.5, 2.0

    def __init__(self, ttft_ms=None, prefill_toks=None):
        self.ttft_ms, self.prefill_toks = ttft_ms, prefill_toks

    def to_dict(self):
        return {"repo": self.repo, "decode_toks": self.decode_toks}

    def to_evidence(self, role):
        self.role = role
        return _Evidence()


class BenchCliTests(unittest.TestCase):
    def test_runtime_clients_cover_every_supported_runtime(self):
        for name in cli._BENCH_RUNTIMES:
            with self.subTest(name=name):
                self.assertIsNotNone(cli._bench_runtime_client(name))
        with self.assertRaisesRegex(ValueError, "unsupported bench runtime: vllm"):
            cli._bench_runtime_client("vllm")

    def test_run_reports_measurement_and_evidence(self):
        measurement = _Measurement(ttft_ms=120, prefill_toks=900.0)
        with patch("mlx_agent.cli.measure_runtime", return_value=measurement) as measure:
            code, text = run(["bench", "run", "--repo", "org/model", "--runtime", "mlx_lm", "--role", "coding", "--json"],
                             inspect=RuntimeError("no host"))
            self.assertEqual(0, code)
            self.assertIsNone(measure.call_args.kwargs["chip"])
            self.assertEqual({"measurement": {"repo": "org/model", "decode_toks": 41.5},
                              "evidence": {"strength": "runtime_measured"}}, json.loads(text)["data"])
            self.assertEqual("coding", measurement.role)
            code, text = run(["bench", "run", "--repo", "org/model", "--runtime", "mlx_lm"])
        self.assertEqual(["Bench: org/model on mlx_lm (3 runs)", "  decode:  41.5 tok/s (spread 2.0%)",
                          "  ttft:    120 ms", "  prefill: 900.0 tok/s"], text.strip().splitlines())
        with patch("mlx_agent.cli.measure_runtime", return_value=_Measurement()):
            text = run(["bench", "run", "--repo", "org/model", "--runtime", "ollama"])[1]
        self.assertIn("  ttft:    n/a", text)
        self.assertIn("  prefill: n/a", text)

    def test_run_can_export_an_anonymized_record(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("mlx_agent.cli.measure_runtime", return_value=_Measurement()), \
                patch("mlx_agent.bench.export_record", return_value={"row": 1}) as record, \
                patch("mlx_agent.bench.append_export", return_value=Path(directory) / "bench.jsonl") as append:
            code, text = run(["bench", "run", "--repo", "org/model", "--runtime", "mlx_lm",
                              "--export", str(Path(directory) / "bench.jsonl"), "--json"])
        self.assertEqual(0, code)
        self.assertTrue(json.loads(text)["data"]["exported"].endswith("bench.jsonl"))
        self.assertEqual({"row": 1}, append.call_args.args[1])
        self.assertEqual(__version__, record.call_args.args[1])

    def test_run_failures_are_classified(self):
        for error, code in ((BenchError("runtime_unreachable", "no server", "Start it."), "runtime_unreachable"),
                            (ValueError("bad runs"), "invalid_arguments")):
            with self.subTest(code=code), patch("mlx_agent.cli.measure_runtime", side_effect=error):
                status, text = run(["bench", "run", "--repo", "org/model", "--runtime", "mlx_lm", "--json"])
                self.assertEqual((2, code), (status, json.loads(text)["error"]["code"]))
                status, text = run(["bench", "run", "--repo", "org/model", "--runtime", "mlx_lm"])
                self.assertTrue(text.startswith("bench failed [{0}]".format(code)))

    def test_aggregate_reads_export_files_and_can_write_the_result(self):
        with tempfile.TemporaryDirectory() as directory:
            exports = Path(directory) / "exports"
            exports.mkdir()
            out = Path(directory) / "aggregate.json"
            code, text = run(["bench", "aggregate", "--exports", str(exports), "--out", str(out)])
            self.assertEqual((0, "Aggregate: 0 entries from 0 export file(s) -> {0}".format(out)), (code, text.strip()))
            self.assertEqual([], json.loads(out.read_text())["entries"])
            code, text = run(["bench", "aggregate", "--exports", str(exports), "--json"])
            self.assertEqual({"entries": [], "sources": 0, "written": None}, json.loads(text)["data"])
            code, text = run(["bench", "aggregate", "--exports", str(exports / "missing"), "--json"])
        self.assertEqual((2, "invalid_arguments"), (code, json.loads(text)["error"]["code"]))


if __name__ == "__main__":
    unittest.main()
