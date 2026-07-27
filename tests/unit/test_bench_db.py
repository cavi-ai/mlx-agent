import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.bench import (
    BenchMeasurement,
    aggregate_exports,
    append_export,
    export_record,
    load_community_bench,
    validate_export_record,
)


def _measurement(decode=38.2, chip="Apple M5 Max"):
    return BenchMeasurement(
        repo="pub/model",
        runtime="mlx_lm",
        runs=3,
        prompt_tokens=96,
        gen_tokens=128,
        ttft_ms=412.0,
        decode_toks=decode,
        prefill_toks=910.0,
        spread_pct=2.1,
        chip=chip,
        measured_at="2026-07-27T12:00:00+00:00",
        samples=[],
    )


class ExportRecordTests(unittest.TestCase):
    def test_record_is_bounded_and_anonymized(self):
        record = export_record(_measurement(), "0.4.0")
        self.assertEqual(set(record), {
            "schema_version", "repo", "chip", "runtime", "ttft_ms",
            "decode_toks", "prefill_toks", "measured_at", "tool_version",
        })
        self.assertNotIn("samples", record)
        self.assertNotIn("log_path", record)

    def test_decode_toks_required(self):
        record = export_record(_measurement(), "0.4.0")
        record["decode_toks"] = None
        with self.assertRaises(ValueError):
            validate_export_record(record)

    def test_out_of_bounds_numbers_rejected(self):
        record = export_record(_measurement(), "0.4.0")
        record["decode_toks"] = -1
        with self.assertRaises(ValueError):
            validate_export_record(record)

    def test_append_export_creates_jsonl(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "exports.jsonl"
            record = export_record(_measurement(), "0.4.0")
            append_export(target, record)
            append_export(target, record)
            lines = target.read_text().strip().split("\n")
            self.assertEqual(len(lines), 2)
            json.loads(lines[0])


class AggregateTests(unittest.TestCase):
    def test_dedupes_to_newest_and_medians(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "exports.jsonl"
            older = export_record(_measurement(decode=30.0), "0.4.0")
            older["measured_at"] = "2026-07-26T12:00:00+00:00"
            newer = export_record(_measurement(decode=40.0), "0.4.0")
            other_chip = export_record(_measurement(decode=20.0, chip="Apple M4"), "0.4.0")
            for record in (older, newer, other_chip):
                append_export(target, record)
            aggregate = aggregate_exports([target])
            entries = {
                (entry["repo"], entry["chip"]): entry
                for entry in aggregate["entries"]
            }
            self.assertEqual(entries[("pub/model", "Apple M5 Max")]["decode_toks"], 40.0)
            self.assertEqual(entries[("pub/model", "Apple M4")]["decode_toks"], 20.0)

    def test_invalid_lines_are_skipped(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "exports.jsonl"
            target.write_text('{"garbage": true}\nnot json\n')
            aggregate = aggregate_exports([target])
            self.assertEqual(aggregate["entries"], [])


class CommunityBenchTests(unittest.TestCase):
    def test_empty_stub_yields_no_entries(self):
        self.assertEqual(load_community_bench(), {})

    def test_entries_are_indexed_by_repo_and_chip(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bench.json"
            path.write_text(json.dumps({
                "schema_version": "1.0",
                "entries": [{
                    "repo": "pub/model", "chip": "Apple M5 Max",
                    "decode_toks": 38.2, "prefill_toks": 910.0,
                    "ttft_ms": 412.0, "samples": 3,
                    "updated_at": "2026-07-27T12:00:00+00:00",
                }],
            }))
            entries = load_community_bench(path)
            self.assertEqual(
                entries[("pub/model", "Apple M5 Max")]["decode_toks"], 38.2
            )


if __name__ == "__main__":
    unittest.main()
