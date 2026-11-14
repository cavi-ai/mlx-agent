import hashlib
import io
import json
import struct
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent import cli
from mlx_agent.gguf import (
    GGUFError,
    describe_gguf,
    families,
    file_signature,
    group_duplicates,
    group_mlx_duplicates,
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

    def test_dspark_drafter_names_its_port_and_target(self):
        entry = describe_gguf(write_gguf(
            self.root / "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", architecture="dflash",
            name="DeepSeek-V4-Flash-0731", block_count=3, extra=(
                _kv("dflash.hyper_connection.count", _UINT32, 4), _kv("dflash.block_size", _UINT32, 5),
            ),
        ))
        self.assertEqual(entry["draft"], {"port": "deepseek_v4_dspark", "target": "DeepSeek-V4-Flash-0731", "block_size": 5})
        self.assertEqual(entry["model_key"], "deepseek-v4-flash-0731-draft")

    def test_other_drafters_have_no_port(self):
        eagle = describe_gguf(write_gguf(self.root / "eagle3-Llama.gguf", architecture="eagle3", name="Llama 3.1 8B"))
        self.assertEqual(eagle["draft"], {"port": None, "target": "Llama 3.1 8B", "block_size": None})
        dflash = describe_gguf(write_gguf(self.root / "glimmer.gguf", architecture="dflash", name="Glimmer"))
        self.assertIsNone(dflash["draft"]["port"])

    def test_regular_models_are_not_drafters(self):
        self.assertIsNone(describe_gguf(write_gguf(self.root / "plain.gguf", name="Plain"))["draft"])

    def test_entries_report_the_widths_they_convert_at(self):
        def bits(filename, **fields):
            return describe_gguf(write_gguf(self.root / filename, **fields))["q_bits"]
        self.assertEqual(bits("plain.gguf", name="Plain"), [4, 5, 6, 8])
        self.assertEqual(bits("Qwen3.5-9B-Q4_K_M.gguf", architecture="qwen35", name="Qwen3.5 9B"), [4, 8])
        self.assertEqual(bits("dspark.gguf", architecture="dflash", name="DeepSeek-V4-Flash-0731",
                              extra=(_kv("dflash.hyper_connection.count", _UINT32, 4),)), [4, 8])
        self.assertEqual(bits("eagle3-Llama.gguf", architecture="eagle3", name="Llama 3.1 8B"), [])
        self.assertEqual(bits("mmproj-F32.gguf", architecture="clip", name="Base"), [])
        broken = self.root / "broken.gguf"
        broken.write_bytes(b"GGUF" + b"\x00" * 8)
        self.assertEqual(describe_gguf(broken)["q_bits"], [])

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
    def test_weight_symlink_cannot_escape_selected_root(self):
        selected = self.root / 'selected'; selected.mkdir()
        outside = write_gguf(self.root / 'excluded' / 'model.gguf')
        (selected / 'model.gguf').symlink_to(outside)
        self.assertEqual(scan_gguf([selected]), [])


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

    def test_unquantized_mlx_community_snapshots_are_outputs(self):
        hub = self.root / "hub"
        write_mlx_output(hub / "models--mlx-community--Kokoro-82M-bf16" / "snapshots" / ("a" * 40), quantization=False)
        write_mlx_output(hub / "models--openai--whisper-base" / "snapshots" / ("b" * 40), quantization=False)
        write_mlx_output(self.root / "models--mlx-community--loose", quantization=False)
        outputs = scan_mlx_outputs([self.root])
        self.assertEqual([Path(item["path"]).parts[-3] for item in outputs], ["models--mlx-community--Kokoro-82M-bf16"])
        self.assertIsNone(outputs[0]["quantization"])

    def test_cache_snapshots_with_missing_weight_blobs_are_not_outputs(self):
        snapshot = self.root / "hub" / "models--mlx-community--partial-4bit" / "snapshots" / ("c" * 40)
        write_mlx_output(snapshot)
        (snapshot / "model.safetensors").unlink()
        (snapshot / "model.safetensors").symlink_to("../../blobs/missing")
        self.assertEqual(scan_mlx_outputs([self.root]), [])

    def test_component_layout_outputs_need_weights_in_every_listed_component(self):
        def write(name, components, weights):
            directory = self.root / name
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps(
                {"model_type": "qwen_image_21", "components": components, "quantization": {"bits": 8}}), encoding="utf-8")
            for component in weights:
                (directory / component).mkdir()
                (directory / component / "0.safetensors").write_bytes(b"x")
            (directory / "transformer" / "config.json").write_text("{}", encoding="utf-8") if "transformer" in weights else None

        write("image-MLX-8bit", ["transformer", "text_encoder", "vae"], ["transformer", "text_encoder", "vae"])
        write("missing-vae-MLX-8bit", ["transformer", "vae"], ["transformer"])
        write("escape-MLX-8bit", ["../x"], [])
        outputs = scan_mlx_outputs([self.root])
        self.assertEqual([(item["name"], item["quantization"]["model_type"]) for item in outputs],
                         [("image-MLX-8bit", "qwen_image_21")])

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

    def test_a_drafter_is_not_a_duplicate_of_its_target(self):
        write_gguf(self.root / "DeepSeek-V4-Flash-Q2-0731.gguf", name="DeepSeek-V4-Flash-0731", file_type=10)
        write_gguf(self.root / "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", architecture="dflash",
                   name="DeepSeek-V4-Flash-0731", file_type=7)
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


def write_mlx_model(directory, bits=4, weights=b"weights", hidden_size=1024):
    location = Path(directory)
    location.mkdir(parents=True, exist_ok=True)
    config = {"model_type": "qwen3", "hidden_size": hidden_size, "num_hidden_layers": 28,
              "quantization": {"group_size": 64, "bits": bits}}
    (location / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (location / "model.safetensors").write_bytes(weights)
    return location


class MLXDuplicateTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_identical_mlx_folders_group_as_exact_and_keep_the_cache_snapshot(self):
        snapshot = write_mlx_model(self.root / "hub" / "models--mlx-community--Lambda-Model-4bit" / "snapshots" / "abc")
        copy = write_mlx_model(self.root / "lmstudio" / "org" / "Lambda-Model-MLX-4bit")
        report = inventory([self.root])
        exact = [group for group in report["duplicates"] if group["kind"] == "exact"]
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0]["model_key"], "lambda-model")
        self.assertEqual(exact[0]["quantization"], "4-bit")
        self.assertEqual(exact[0]["keep"], str(snapshot))
        self.assertEqual(exact[0]["redundant"], [str(copy)])
        self.assertEqual(exact[0]["reclaimable_bytes"], sum(item.stat().st_size for item in copy.iterdir()))
        self.assertEqual(report["totals"]["reclaimable_bytes"], exact[0]["reclaimable_bytes"])
        self.assertTrue(exact[0]["group_id"].startswith("exact:mlx:"))

    def test_mlx_bit_widths_of_one_model_group_as_variant(self):
        write_mlx_model(self.root / "Mu-Model-MLX-4bit", bits=4, weights=b"four")
        write_mlx_model(self.root / "Mu-Model-MLX-8bit", bits=8, weights=b"eight-bit")
        write_mlx_model(self.root / "Mu-Model-Wide-MLX-8bit", bits=8, hidden_size=2048)
        groups = group_mlx_duplicates(scan_mlx_outputs([self.root]))
        self.assertEqual([(group["kind"], group["model_key"], group["quantizations"]) for group in groups],
                         [("variant", "mu-model", ["4-bit", "8-bit"])])
        self.assertEqual(groups[0]["reclaimable_bytes"], 0)

    def test_same_size_weights_with_different_bytes_are_not_duplicates(self):
        write_mlx_model(self.root / "a" / "Nu-Model-MLX-4bit", weights=b"tuned-a")
        write_mlx_model(self.root / "b" / "Nu-Model-MLX-4bit", weights=b"tuned-b")
        self.assertEqual(group_mlx_duplicates(scan_mlx_outputs([self.root])), [])

    def test_cache_revisions_sharing_blobs_are_one_copy(self):
        blob = self.root / "hub" / "models--org--Xi-Model-4bit" / "blobs" / "weights"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"weights")
        for revision in ("one", "two"):
            snapshot = write_mlx_model(self.root / "hub" / "models--org--Xi-Model-4bit" / "snapshots" / revision)
            (snapshot / "model.safetensors").unlink()
            (snapshot / "model.safetensors").symlink_to(blob)
        self.assertEqual(group_mlx_duplicates(scan_mlx_outputs([self.root])), [])


def write_ollama_model(store, model, tag, layers, config, host="registry.ollama.ai", tree="manifests"):
    """An Ollama store manifest whose layers are (media type suffix, bytes, tensor or file name)."""
    blobs = Path(store) / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)

    def blob(content):
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        (blobs / digest.replace(":", "-")).write_bytes(content)
        return digest

    entries = []
    for kind, content, name in layers:
        layer = {"mediaType": "application/vnd.ollama.image." + kind, "digest": blob(content), "size": len(content)}
        if name:
            layer["name"] = name
        entries.append(layer)
    encoded = json.dumps(config).encode("utf-8")
    manifest = {
        "schemaVersion": 2,
        "config": {"mediaType": "application/vnd.docker.container.image.v1+json",
                   "digest": blob(encoded), "size": len(encoded)},
        "layers": entries,
    }
    location = Path(store) / tree / host / "library" / model / tag
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps(manifest), encoding="utf-8")
    return location


class OllamaTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_an_ollama_copy_of_a_gguf_file_is_kept_and_the_file_is_redundant(self):
        source = write_gguf(self.root / "lmstudio" / "rho-model-Q4_K_M.gguf", name="Rho Model", padding=b"x" * 64)
        manifest = write_ollama_model(self.root / "ollama", "rho", "latest", [("model", source.read_bytes(), None)],
                                      {"model_format": "gguf", "file_type": "Q4_K_M"})
        report = inventory([self.root])
        self.assertEqual([(item["name"], item["format"], item["path"]) for item in report["ollama"]],
                         [("rho:latest", "gguf", str(manifest))])
        exact = [group for group in report["duplicates"] if group["kind"] == "exact"]
        self.assertEqual([(group["keep"], group["redundant"]) for group in exact], [(str(manifest), [str(source)])])
        self.assertEqual(exact[0]["reclaimable_bytes"], source.stat().st_size)

    def test_an_ollama_mlx_model_groups_with_other_bit_widths_of_that_model(self):
        folder = write_mlx_model(self.root / "lmstudio" / "Sigma-Model-27B-MLX-8bit", bits=8)
        config = {"model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28}
        manifest = write_ollama_model(
            self.root / "ollama", "sigma-model", "27b-mlx",
            [("json", json.dumps(config).encode("utf-8"), "config.json"), ("tensor", b"tensor", "lm_head.weight")],
            {"model_format": "safetensors", "file_type": "nvfp4"},
        )
        report = inventory([self.root])
        self.assertEqual([(item["name"], item["format"], item["quantization"]) for item in report["ollama"]],
                         [("sigma-model:27b-mlx", "safetensors", "nvfp4")])
        self.assertEqual([(group["kind"], group["quantizations"], group["members"]) for group in report["duplicates"]],
                         [("variant", ["8-bit", "nvfp4"], sorted([str(folder), str(manifest)]))])

    def test_a_model_in_both_manifest_trees_is_listed_once(self):
        content = write_gguf(self.root / "source" / "tau.gguf", name="Tau Model").read_bytes()
        older = write_ollama_model(self.root / "ollama", "tau", "latest",
                                   [("model", content, None), ("license", b"license", None)], {"model_format": "gguf"})
        current = write_ollama_model(self.root / "ollama", "tau", "latest", [("model", content, None)],
                                     {"model_format": "gguf"}, host="ollama.com", tree="manifests-v2")
        report = inventory([self.root / "ollama"])
        self.assertTrue(older.exists())
        self.assertEqual([(item["name"], item["path"]) for item in report["ollama"]], [("tau:latest", str(current))])
        self.assertEqual(report["duplicates"], [])

    def test_a_manifest_with_a_missing_blob_is_skipped(self):
        manifest = write_ollama_model(self.root / "ollama", "phi", "latest", [("model", b"GGUF-bytes", None)],
                                      {"model_format": "gguf"})
        for blob in (self.root / "ollama" / "blobs").iterdir():
            if blob.read_bytes() == b"GGUF-bytes":
                blob.unlink()
        self.assertTrue(manifest.exists())
        self.assertEqual(inventory([self.root])["ollama"], [])


def write_shaped_model(directory, layers, width, heads, bits=4, weights=b"weights"):
    location = Path(directory)
    location.mkdir(parents=True, exist_ok=True)
    config = {"model_type": "qwen3_5", "num_hidden_layers": layers, "hidden_size": width,
              "num_attention_heads": heads, "quantization": {"group_size": 64, "bits": bits}}
    (location / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (location / "model.safetensors").write_bytes(weights)
    return location


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_copies_of_one_model_in_every_format_form_one_family(self):
        mlx = write_shaped_model(self.root / "lmstudio" / "Pi-Model-27B-MLX-8bit", 64, 5120, 24, bits=8)
        tuned = write_shaped_model(self.root / "lmstudio" / "Pi-Model-27B-Uncensored-MLX", 64, 5120, 24, weights=b"tuned")
        gguf = write_gguf(self.root / "lmstudio" / "tune" / "Pi-Tune-27B-MTP-Q6_K.gguf", architecture="qwen35",
                          name="Pi Tune", block_count=65, extra=(
                              _kv("qwen35.nextn_predict_layers", _UINT32, 1),
                              _kv("qwen35.embedding_length", _UINT32, 5120),
                              _kv("qwen35.attention.head_count", _UINT32, 24)))
        config = {"model_type": "qwen3_5",
                  "text_config": {"num_hidden_layers": 64, "hidden_size": 5120, "num_attention_heads": 24}}
        manifest = write_ollama_model(
            self.root / "ollama", "pi-model", "27b-mlx",
            [("json", json.dumps(config).encode("utf-8"), "config.json"), ("tensor", b"tensor", "lm_head.weight")],
            {"model_format": "safetensors", "file_type": "nvfp4"},
        )
        smaller = write_shaped_model(self.root / "lmstudio" / "Pi-Model-9B-MLX-4bit", 32, 4096, 16, weights=b"small")
        out = io.StringIO()
        with TemporaryDirectory() as receipts, redirect_stdout(out):
            code = cli.main(["convert", "scan", "--gguf-root", str(self.root), "--receipts-dir", receipts,
                             "--no-signature", "--json"])
        self.assertEqual(code, 0, out.getvalue())
        found = {family["name"]: family["members"] for family in json.loads(out.getvalue())["data"]["families"]}
        self.assertEqual(found, {
            "Pi-Model-27B": sorted(str(path) for path in (mlx, tuned, gguf, manifest)),
            "Pi-Model-9B": [str(smaller)],
        })

    def test_an_embedding_model_is_not_a_copy_of_a_generator_with_its_architecture(self):
        chat = write_shaped_model(self.root / "Qwen-4B-MLX-4bit", 36, 2560, 32, weights=b"chat")
        embedding = write_shaped_model(self.root / "Qwen-Embedding-4B-MLX-8bit", 36, 2560, 32, bits=8, weights=b"embed")
        items = inventory([self.root])["storage"]["items"]
        self.assertEqual([family["members"] for family in families(items)], [sorted([str(chat), str(embedding)])])
        split = families(items, {str(embedding): "embedding"})
        self.assertEqual(sorted(family["members"] for family in split), [[str(chat)], [str(embedding)]])

    def test_files_that_several_models_share_count_once(self):
        with TemporaryDirectory() as source:
            content = write_gguf(Path(source, "rho.gguf"), name="Rho Model", padding=b"x" * 4096).read_bytes()
        store = self.root / "ollama"
        tags = [write_ollama_model(store, "rho", tag, [("model", content, None), ("template", template, None)],
                                   {"model_format": "gguf"}) for tag, template in (("latest", b"chat"), ("q4", b"plain"))]
        blob = self.root / "hub" / "models--org--Xi-Model-4bit" / "blobs" / "weights"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"w" * 1000)
        for revision in ("one", "two"):
            snapshot = write_mlx_model(self.root / "hub" / "models--org--Xi-Model-4bit" / "snapshots" / revision)
            (snapshot / "model.safetensors").unlink()
            (snapshot / "model.safetensors").symlink_to(blob)
        report = inventory([self.root])
        stored = report["storage"]
        on_disk = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file() and not path.is_symlink())
        self.assertEqual(stored["disk_bytes"], on_disk)
        self.assertEqual(sum(store["bytes"] for store in stored["stores"]), on_disk)
        self.assertGreater(sum(item["bytes"] for item in stored["items"]), on_disk)
        self.assertEqual(stored["roots"], [{"root": str(self.root), "bytes": on_disk, "total_bytes": on_disk, "other": []}])
        ollama = [{file["id"] for file in item["files"]} for item in stored["items"] if item["kind"] == "ollama"]
        self.assertEqual([item["path"] for item in stored["items"] if item["kind"] == "ollama"], sorted(map(str, tags)))
        self.assertTrue(ollama[0] & ollama[1])
        self.assertEqual([(output["repo"], output["bytes"]) for output in report["outputs"]],
                         [("org/Xi-Model-4bit", 1000 + (snapshot / "config.json").stat().st_size)] * 2)

    def test_standalone_files_are_accounted_as_assets(self):
        write_shaped_model(self.root / "lmstudio" / "org" / "Pi-Model-MLX-4bit", 28, 1024, 16)
        unknown = self.root / "lmstudio" / "org" / "Unknown-bf16"
        unknown.mkdir()
        (unknown / "model.safetensors").write_bytes(b"u" * 100)
        (self.root / "comfyui" / "checkpoints").mkdir(parents=True)
        (self.root / "comfyui" / "checkpoints" / "model.ckpt").write_bytes(b"c" * 50)
        notes = self.root / "notes.txt"
        notes.write_bytes(b"notes")
        root = inventory([self.root])["storage"]["roots"][0]
        self.assertEqual(root["other"], [{"path": str(notes), "bytes": 5}])
        self.assertEqual(root["total_bytes"], root["bytes"] + 5)

    def test_a_split_gguf_is_one_item_with_the_projector_beside_it(self):
        folder = self.root / "Omega-GGUF"
        first = write_gguf(folder / "Omega-Q4_K_M-00001-of-00002.gguf", name="Omega")
        second = write_gguf(folder / "Omega-Q4_K_M-00002-of-00002.gguf", name="Omega")
        projector = write_gguf(folder / "mmproj-Omega-F16.gguf", architecture="clip")
        items = inventory([self.root])["storage"]["items"]
        files = (first, second, projector)
        self.assertEqual([(item["path"], sorted(stored["path"] for stored in item["files"]), item["bytes"]) for item in items],
                         [(str(first), sorted(map(str, files)), sum(path.stat().st_size for path in files))])


class InventoryTests(unittest.TestCase):
    def test_missing_root_is_reported_without_losing_healthy_inventory(self):
        selected = self.root / 'selected'
        write_gguf(selected / 'model.gguf')
        missing = self.root / 'missing'
        result = inventory([missing, selected])
        self.assertEqual(len(result['models']), 1)
        self.assertEqual(result['root_errors'][0]['path'], str(missing))


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
