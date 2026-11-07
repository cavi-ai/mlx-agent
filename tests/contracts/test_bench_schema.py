import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from mlx_agent.bench import aggregate_exports, validate_export_record
from tests.unit.test_bench_db import _measurement
from mlx_agent.bench import export_record


ROOT = Path(__file__).resolve().parents[2]


class BenchSchemaTests(unittest.TestCase):
    def test_export_schema_and_runtime_agree_on_metric_provenance(self):
        schema = json.loads((ROOT / "schemas/bench-export.schema.json").read_text())
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        current = export_record(_measurement(), "0.11.1")
        legacy = {key: current[key] for key in ("repo", "chip", "runtime", "ttft_ms", "decode_toks",
                                              "prefill_toks", "measured_at", "tool_version")}
        legacy["schema_version"] = "1.0"
        wall_clock = dict(current, measurement_method="client_wall_clock", decode_toks=None, prefill_toks=None)
        latency = dict(wall_clock, measurement_method="latency_only", end_to_end_toks=None)
        integral_numbers = dict(current, gen_tokens=128.0, prompt_tokens=96.0)
        for value in (current, legacy, wall_clock, latency, integral_numbers):
            with self.subTest(valid=value["schema_version"], method=value.get("measurement_method")):
                validator.validate(value)
                self.assertEqual(validate_export_record(value), value)
        invalid = [dict(current, measurement_method="client_wall_clock"),
                   dict(current, gen_tokens=True), dict(current, prompt_tokens=1.5),
                   dict(current, measured_at="yesterday"), dict(current, measured_at="2026-11-06"),
                   dict(current, measured_at="2026-07-27T12:00:00+00:60"),
                   dict(current, measured_at="2026-07-27T12:00:00+24:00"),
                   dict(current, private_path="/private/local"),
                   dict(current, probe_id="bench-v1"),
                   dict(latency, end_to_end_toks=20.0), dict(wall_clock, end_to_end_toks=None),
                   dict(legacy, decode_toks=None)]
        for value in invalid:
            with self.subTest(invalid=value):
                self.assertFalse(validator.is_valid(value))
                with self.assertRaises(ValueError):
                    validate_export_record(value)

    def test_aggregate_schema_accepts_current_and_legacy_records(self):
        import tempfile
        schema = json.loads((ROOT / "schemas/bench-aggregate.schema.json").read_text())
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        current = export_record(_measurement(), "0.11.1")
        legacy = {key: current[key] for key in ("repo", "chip", "runtime", "ttft_ms", "decode_toks",
                                              "prefill_toks", "measured_at", "tool_version")}
        legacy["schema_version"] = "1.0"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exports.jsonl"
            path.write_text(json.dumps(current) + "\n" + json.dumps(legacy) + "\n")
            value = aggregate_exports([path])
        validator.validate(value)
        older = {"schema_version": "1.0", "entries": [{
            key: value["entries"][0][key] for key in (
                "repo", "chip", "decode_toks", "prefill_toks", "ttft_ms", "samples", "updated_at")
        }]}
        validator.validate(older)
        broken = copy.deepcopy(value)
        broken["entries"][0]["runtime"] = None
        self.assertFalse(validator.is_valid(broken))
