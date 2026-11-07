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
        end_to_end_toks=25.0,
        measurement_method="server_timings",
    )


class ExportRecordTests(unittest.TestCase):
    def test_record_is_bounded_and_anonymized(self):
        record = export_record(_measurement(), "0.4.0")
        self.assertEqual(set(record), {
            "schema_version", "repo", "chip", "runtime", "ttft_ms",
            "decode_toks", "prefill_toks", "measured_at", "tool_version",
            "probe_id", "measurement_method", "gen_tokens", "prompt_tokens", "end_to_end_toks",
        })
        self.assertNotIn("samples", record)
        self.assertNotIn("log_path", record)

    def test_missing_decode_is_preserved_with_wall_clock_method(self):
        record = export_record(_measurement(), "0.4.0")
        record["decode_toks"] = None
        record["prefill_toks"] = None
        record["measurement_method"] = "client_wall_clock"
        self.assertIsNone(validate_export_record(record)["decode_toks"])

    def test_method_cannot_claim_stage_timings_for_wall_clock_rate(self):
        record = export_record(_measurement(), "0.4.0")
        record["measurement_method"] = "client_wall_clock"
        with self.assertRaises(ValueError):
            validate_export_record(record)

    def test_latency_only_record_cannot_claim_token_throughput(self):
        record = export_record(_measurement(), "0.4.0")
        record["measurement_method"] = "latency_only"
        record["decode_toks"] = record["prefill_toks"] = None
        with self.assertRaises(ValueError):
            validate_export_record(record)

    def test_legacy_export_retains_its_version_without_invented_method(self):
        legacy = {"schema_version": "1.0", "repo": "pub/model", "chip": "Apple M5 Max",
                  "runtime": "mlx_lm", "ttft_ms": 412.0, "decode_toks": 100.0,
                  "prefill_toks": 900.0, "measured_at": "2026-07-27T12:00:00+00:00",
                  "tool_version": "0.4.0"}
        self.assertEqual(validate_export_record(legacy), legacy)
        self.assertNotIn("measurement_method", legacy)

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
    def test_runtime_method_and_workload_are_separate_groups(self):
        first = export_record(_measurement(), "0.4.0")
        variants = [dict(first), dict(first, runtime="ollama"), dict(first, gen_tokens=256),
                    dict(first, measurement_method="client_wall_clock", decode_toks=None, prefill_toks=None)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "exports.jsonl"
            for record in variants:
                append_export(path, record)
            aggregate = aggregate_exports([path])
        self.assertEqual(aggregate["schema_version"], "2.0")
        self.assertEqual(len(aggregate["entries"]), 4)
        self.assertEqual({v["runtime"] for v in aggregate["entries"]}, {"mlx_lm", "ollama"})
        self.assertTrue(all(v["samples"] == 1 for v in aggregate["entries"]))

    def test_legacy_measurements_are_explicitly_separate(self):
        current = export_record(_measurement(), "0.4.0")
        legacy = {key: current[key] for key in ("repo", "chip", "runtime", "ttft_ms", "decode_toks",
                                              "prefill_toks", "measured_at", "tool_version")}
        legacy.update(schema_version="1.0", decode_toks=1000.0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "exports.jsonl"
            for record in (current, legacy):
                append_export(path, record)
            entries = aggregate_exports([path])["entries"]
        self.assertEqual(len(entries), 2)
        methods = {v["measurement_method"]: v["decode_toks"] for v in entries}
        self.assertEqual(methods, {"server_timings": 38.2, "legacy_unclassified": 1000.0})

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
    def test_multiple_runtime_groups_do_not_choose_arbitrary_annotation(self):
        first = export_record(_measurement(), "0.4.0")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "exports.jsonl"
            append_export(path, first)
            append_export(path, dict(first, runtime="ollama"))
            aggregate = aggregate_exports([path])
            out = Path(directory) / "aggregate.json"
            out.write_text(json.dumps(aggregate))
            self.assertEqual(load_community_bench(out), {})

    def test_single_current_group_retains_method_and_runtime(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "exports.jsonl"
            append_export(path, export_record(_measurement(), "0.4.0"))
            out = Path(directory) / "aggregate.json"
            out.write_text(json.dumps(aggregate_exports([path])))
            entry = load_community_bench(out)[("pub/model", "Apple M5 Max")]
        self.assertEqual(entry["measurement_method"], "server_timings")
        self.assertEqual(entry["runtime"], "mlx_lm")

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
