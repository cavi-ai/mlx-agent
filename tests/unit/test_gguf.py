import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.gguf import (
    GGUFError,
    describe_gguf,
    file_signature,
    group_duplicates,
    inventory,
    pair_conversions,
    read_gguf_header,
    scan_gguf,
    scan_mlx_outputs,
)


_STRING = 8
_UINT32 = 4


def _string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _kv(key, value_type, value):
    payload = _string(key) + struct.pack("<I", value_type)
    if value_type == _STRING:
        return payload + _string(value)
    return payload + struct.pack("<I", value)


def write_gguf(path, architecture="llama", name=None, file_type=15,
               block_count=32, tensor_count=291, version=3, padding=b"",
               extra=()):
    """Write a minimal but structurally valid GGUF header."""
    pairs = [_kv("general.architecture", _STRING, architecture)]
    if name is not None:
        pairs.append(_kv("general.name", _STRING, name))
    if file_type is not None:
        pairs.append(_kv("general.file_type", _UINT32, file_type))
    if block_count is not None:
        pairs.append(_kv("{0}.block_count".format(architecture), _UINT32, block_count))
    pairs.extend(extra)
    body = b"".join(pairs)
    header = (
        b"GGUF"
        + struct.pack("<I", version)
        + struct.pack("<Q", tensor_count)
        + struct.pack("<Q", len(pairs))
        + body
    )
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_bytes(header + padding)
    return location


def write_mlx_output(directory, quantization=True, provenance=None):
    location = Path(directory)
    location.mkdir(parents=True, exist_ok=True)
    config = {"model_type": "llama"}
    if quantization:
        config["quantization"] = {"group_size": 64, "bits": 4}
    (location / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (location / "model.safetensors").write_bytes(b"\x00")
    if provenance is not None:
        (location / "mlx-converter.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
    return location


class ReadHeaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_reads_interesting_metadata(self):
        path = write_gguf(self.root / "a.gguf", name="Test Model", file_type=18)
        header = read_gguf_header(path)
        self.assertEqual(header["version"], 3)
        self.assertEqual(header["tensor_count"], 291)
        self.assertEqual(header["metadata"]["general.architecture"], "llama")
        self.assertEqual(header["metadata"]["general.name"], "Test Model")
        self.assertEqual(header["metadata"]["llama.block_count"], 32)

    def test_skips_uninteresting_keys(self):
        extra = (_kv("tokenizer.ggml.model", _STRING, "gpt2"),)
        path = write_gguf(self.root / "b.gguf", extra=extra)
        header = read_gguf_header(path)
        self.assertNotIn("tokenizer.ggml.model", header["metadata"])
        self.assertEqual(header["metadata"]["general.architecture"], "llama")

    def test_rejects_non_gguf(self):
        path = self.root / "c.gguf"
        path.write_bytes(b"NOPE" + b"\x00" * 32)
        with self.assertRaises(GGUFError) as caught:
            read_gguf_header(path)
        self.assertEqual(caught.exception.code, "not_gguf")

    def test_rejects_unsupported_version(self):
        path = write_gguf(self.root / "d.gguf", version=99)
        with self.assertRaises(GGUFError) as caught:
            read_gguf_header(path)
        self.assertEqual(caught.exception.code, "unsupported_gguf_version")

    def test_rejects_truncated_header(self):
        path = write_gguf(self.root / "e.gguf")
        payload = path.read_bytes()
        path.write_bytes(payload[:20])
        with self.assertRaises(GGUFError) as caught:
            read_gguf_header(path)
        self.assertEqual(caught.exception.code, "gguf_header_unreadable")

    def test_missing_file(self):
        with self.assertRaises(GGUFError) as caught:
            read_gguf_header(self.root / "missing.gguf")
        self.assertEqual(caught.exception.code, "gguf_unreadable")


class DescribeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_quantization_from_file_type(self):
        entry = describe_gguf(write_gguf(self.root / "some-model.gguf", file_type=15))
        self.assertEqual(entry["quantization"], "Q4_K_M")
        self.assertTrue(entry["readable"])
        self.assertIsNotNone(entry["signature"])

    def test_quantization_falls_back_to_name(self):
        entry = describe_gguf(
            write_gguf(self.root / "some-model-Q6_K.gguf", file_type=None)
        )
        self.assertEqual(entry["quantization"], "Q6_K")

    def test_model_key_strips_quantization_and_shard(self):
        entry = describe_gguf(
            write_gguf(self.root / "big-model-Q4_K_M-00001-of-00003.gguf", file_type=None)
        )
        self.assertEqual(entry["model_key"], "big-model")
        self.assertEqual(entry["shard"], {"index": 1, "total": 3})

    def test_general_name_wins_over_filename(self):
        entry = describe_gguf(
            write_gguf(self.root / "unhelpful.gguf", name="Qwen3 Coder 30B")
        )
        self.assertEqual(entry["model_key"], "qwen3-coder-30b")

    def test_companion_projector(self):
        entry = describe_gguf(
            write_gguf(self.root / "mmproj-F32.gguf", architecture="clip", name="Base")
        )
        self.assertTrue(entry["companion"])

    def test_unreadable_header_is_reported_not_raised(self):
        path = self.root / "broken.gguf"
        path.write_bytes(b"GGUF" + b"\x00" * 8)
        entry = describe_gguf(path)
        self.assertFalse(entry["readable"])
        self.assertEqual(entry["error"]["code"], "unsupported_gguf_version")

    def test_signature_matches_for_identical_bytes(self):
        first = write_gguf(self.root / "one.gguf", name="Same Model")
        second = self.root / "two.gguf"
        second.write_bytes(first.read_bytes())
        self.assertEqual(file_signature(first), file_signature(second))


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_scan_finds_files_and_skips_hidden_directories(self):
        write_gguf(self.root / "models" / "alpha-model-Q4_K_M.gguf", name="Alpha Model")
        write_gguf(self.root / ".cache" / "hidden.gguf")
        (self.root / "models" / "notes.txt").write_text("x", encoding="utf-8")
        entries = scan_gguf([self.root])
        self.assertEqual([entry["name"] for entry in entries], ["alpha-model-Q4_K_M.gguf"])

    def test_limit_is_respected(self):
        for index in range(4):
            write_gguf(self.root / "m{0}.gguf".format(index), name="Model {0}".format(index))
        self.assertEqual(len(scan_gguf([self.root], limit=2)), 2)

    def test_mlx_outputs_require_quantization_or_provenance(self):
        write_mlx_output(self.root / "converted-model-MLX-4bit")
        write_mlx_output(self.root / "plain-torch-model", quantization=False)
        outputs = scan_mlx_outputs([self.root])
        self.assertEqual([item["name"] for item in outputs], ["converted-model-MLX-4bit"])

    def test_mlx_output_provenance_is_read(self):
        write_mlx_output(
            self.root / "out",
            quantization=False,
            provenance={"source": {"kind": "gguf", "path": "/x/y.gguf"}},
        )
        outputs = scan_mlx_outputs([self.root])
        self.assertEqual(outputs[0]["provenance"]["source"]["path"], "/x/y.gguf")


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def _entry(self, filename, **kwargs):
        return describe_gguf(write_gguf(self.root / filename, **kwargs))

    def test_provenance_beats_name(self):
        entry = self._entry("alpha-model-Q4_K_M.gguf", name="Alpha Model")
        outputs = [{
            "path": "/mlx/whatever",
            "name": "whatever",
            "model_key": "whatever",
            "provenance": {"source": {"signature": entry["signature"]}},
        }]
        paired = pair_conversions([entry], outputs)
        self.assertEqual(paired[0]["status"], "converted")
        self.assertEqual(paired[0]["evidence"], "provenance")

    def test_name_match_requires_a_real_prefix(self):
        entry = self._entry("alpha-model-Q4_K_M.gguf", name="Alpha Model")
        near = [{"path": "/mlx/a", "name": "a", "model_key": "alpha-modelling", "provenance": None}]
        exact = [{"path": "/mlx/b", "name": "b", "model_key": "alpha-model-mlx-4bit", "provenance": None}]
        self.assertEqual(pair_conversions([entry], near)[0]["status"], "pending")
        self.assertEqual(pair_conversions([entry], exact)[0]["status"], "converted")

    def test_short_keys_never_match_by_name(self):
        entry = self._entry("ab.gguf", name="ab", file_type=None)
        outputs = [{"path": "/mlx/ab", "name": "ab", "model_key": "ab-mlx-4bit", "provenance": None}]
        self.assertEqual(pair_conversions([entry], outputs)[0]["status"], "pending")

    def test_receipt_pairs_when_output_exists(self):
        entry = self._entry("beta-model-Q8_0.gguf", name="Beta Model")
        out = write_mlx_output(self.root / "beta-out")
        receipts = [{
            "source": {"kind": "gguf", "signature": entry["signature"]},
            "out": str(out),
        }]
        paired = pair_conversions([entry], [], receipts)
        self.assertEqual(paired[0]["status"], "converted")
        self.assertEqual(paired[0]["evidence"], "receipt")

    def test_non_first_shards_and_companions_are_not_pending(self):
        shard = self._entry("gamma-model-Q4_K_M-00002-of-00003.gguf", name="Gamma Model")
        companion = self._entry("mmproj-F32.gguf", architecture="clip", name="Gamma Model")
        paired = pair_conversions([shard, companion], [])
        self.assertEqual([item["status"] for item in paired], ["shard", "companion"])


class DuplicateTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_identical_files_group_as_exact(self):
        first = write_gguf(self.root / "a" / "delta-model-Q4_K_M.gguf", name="Delta Model")
        second = self.root / "b" / "delta-model-Q4_K_M.gguf"
        second.parent.mkdir(parents=True)
        second.write_bytes(first.read_bytes())
        entries = scan_gguf([self.root])
        groups = group_duplicates(entries)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["kind"], "exact")
        self.assertEqual(len(groups[0]["redundant"]), 1)
        self.assertGreater(groups[0]["reclaimable_bytes"], 0)
        self.assertNotIn(groups[0]["keep"], groups[0]["redundant"])

    def test_different_quantizations_group_as_variant(self):
        write_gguf(self.root / "epsilon-model-Q4_K_M.gguf", name="Epsilon Model", file_type=15)
        write_gguf(self.root / "epsilon-model-Q8_0.gguf", name="Epsilon Model", file_type=7)
        groups = group_duplicates(scan_gguf([self.root]))
        self.assertEqual([group["kind"] for group in groups], ["variant"])
        self.assertEqual(groups[0]["quantizations"], ["Q4_K_M", "Q8_0"])
        self.assertEqual(groups[0]["reclaimable_bytes"], 0)

    def test_distinct_models_are_not_grouped(self):
        write_gguf(self.root / "zeta-model-Q4_K_M.gguf", name="Zeta Model")
        write_gguf(self.root / "eta-model-Q4_K_M.gguf", name="Eta Model", block_count=48)
        self.assertEqual(group_duplicates(scan_gguf([self.root])), [])

    def test_converted_copy_is_kept(self):
        first = write_gguf(self.root / "a" / "theta-model-Q4_K_M.gguf", name="Theta Model")
        second = self.root / "b" / "theta-model-Q4_K_M.gguf"
        second.parent.mkdir(parents=True)
        second.write_bytes(first.read_bytes())
        entries = scan_gguf([self.root])
        entries[0]["status"] = "pending"
        entries[1]["status"] = "converted"
        groups = group_duplicates(entries)
        self.assertEqual(groups[0]["keep"], entries[1]["path"])


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_report_shape(self):
        write_gguf(self.root / "iota-model-Q4_K_M.gguf", name="Iota Model")
        source = write_gguf(self.root / "kappa-model-Q4_K_M.gguf", name="Kappa Model")
        entry = describe_gguf(source)
        write_mlx_output(
            self.root / "kappa-out",
            provenance={"source": {"kind": "gguf", "signature": entry["signature"]}},
        )
        report = inventory([self.root])
        self.assertEqual(report["totals"]["gguf"], 2)
        self.assertEqual(report["totals"]["pending"], 1)
        self.assertEqual(report["totals"]["converted"], 1)
        self.assertEqual(report["pending"], [str(self.root / "iota-model-Q4_K_M.gguf")])
        self.assertEqual(report["duplicates"], [])
        self.assertGreater(report["totals"]["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
