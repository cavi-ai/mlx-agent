"""The bundled Kev port; runs where mlx-embeddings is importable (the backend venv)."""

import collections
import contextlib
import importlib
import importlib.util
import io
import json
import pickle
import sys
import types
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.backends import PORTS_DIR

HAS_MLX_EMBEDDINGS = importlib.util.find_spec("mlx_embeddings") is not None
WORDS = ["charged", "twice", "refund", "?", "which", "team", "billing", "sales", "technical", "level", "low", "high",
         "no", "yes", ":", "the", "a", "b", "c", "x"]
SPECIAL = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
VOCAB = ["[UNK]"] + WORDS + SPECIAL


def load_port():
    sys.path.insert(0, str(PORTS_DIR / "mlx-embeddings"))
    try:
        return importlib.import_module("kev.kev"), importlib.import_module("kev.convert")
    finally:
        sys.path.pop(0)


def backbone_config():
    return {"model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": len(VOCAB),
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True}


def write_tokenizer(directory):
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({token: index for index, token in enumerate(VOCAB)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in SPECIAL])
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({"pad_token": "<|endoftext|>"}), encoding="utf-8")


@contextlib.contextmanager
def torch_globals():
    """Stand-in torch modules so a test can write the torch zip format the reader parses."""
    torch, utils = types.ModuleType("torch"), types.ModuleType("torch._utils")

    class FloatStorage:
        pass

    def _rebuild_tensor_v2(*args):
        raise AssertionError("never called while writing")

    FloatStorage.__module__, FloatStorage.__qualname__ = "torch", "FloatStorage"
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    torch.FloatStorage, torch._utils, utils._rebuild_tensor_v2 = FloatStorage, utils, _rebuild_tensor_v2
    saved = {name: sys.modules.get(name) for name in ("torch", "torch._utils")}
    sys.modules.update({"torch": torch, "torch._utils": utils})
    try:
        yield FloatStorage, _rebuild_tensor_v2
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def write_head(path, arrays, settings, extra=None):
    """A torch zip checkpoint {head: OrderedDict of float32 tensors, **settings} under archive prefix head/."""
    import numpy as np

    blobs = {}
    with torch_globals() as (storage_type, rebuild):
        class Storage:
            def __init__(self, key, size):
                self.key, self.size = key, size

        class Tensor:
            def __init__(self, key, array):
                self.storage, self.shape = Storage(key, array.size), array.shape

            def __reduce__(self):
                stride = tuple(int(np.prod(self.shape[index + 1:])) for index in range(len(self.shape)))
                return rebuild, (self.storage, 0, self.shape, stride, False, collections.OrderedDict())

        class Pickler(pickle.Pickler):
            def persistent_id(self, value):
                if isinstance(value, Storage):
                    return ("storage", storage_type, value.key, "cpu", value.size)
                return None

        head = collections.OrderedDict()
        for index, (name, array) in enumerate(arrays.items()):
            blobs[str(index)] = np.asarray(array, dtype=np.float32).tobytes()
            head[name] = Tensor(str(index), np.asarray(array, dtype=np.float32))
        payload = dict(settings, **(extra or {}))
        payload["head"] = head
        stream = io.BytesIO()
        Pickler(stream, protocol=2).dump(payload)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("head/data.pkl", stream.getvalue())
        archive.writestr("head/byteorder", "little")
        for key, blob in blobs.items():
            archive.writestr("head/data/" + key, blob)


@unittest.skipUnless(HAS_MLX_EMBEDDINGS, "mlx-embeddings is not installed in this interpreter")
class KevPortTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx

        self.mx = mx
        self.port, self.convert = load_port()
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        write_tokenizer(self.root / "tok")
        self.tok = self.port.RequestTokenizer(self.root / "tok")

    def model(self, **settings):
        self.mx.random.seed(0)
        model = self.port.Model(self.port.ModelArgs.from_dict(dict({"backbone": backbone_config(), "head_dim": 16}, **settings)))
        self.mx.eval(model.parameters())
        return model

    def test_each_question_row_is_the_state_then_its_branch_with_readouts_on_the_delimiters(self):
        questions = [self.port.parse_question("team", {"type": "choice", "instructions": "which team ?",
                                                        "criteria": {"billing": None, "sales": "x"}}),
                     self.port.parse_question("refund", {"type": "noul", "instructions": "refund ?"})]
        rows, packed = self.port.build_rows(self.tok, "charged twice", questions, 64, 64)
        state = [self.tok.state_id] + self.tok.user_ids("charged twice")
        self.assertEqual([row["ids"][:len(state)] for row in rows], [state, state])
        for row, question in zip(rows, questions):
            self.assertEqual(row["ids"][len(state)], self.tok.question_id)
            self.assertEqual((row["decide"], row["ids"][row["decide"]]), (len(row["ids"]) - 1, self.tok.decide_id))
            self.assertEqual([row["ids"][index] for index in row["options"]], [self.tok.close_id] * len(question["options"]))
        self.assertEqual(questions[0]["options"], ["billing", "sales: x"])
        self.assertEqual(questions[1]["options"], ["no", "yes"])
        self.assertEqual(packed, len(state) + sum(len(row["ids"]) - len(state) for row in rows))
        long_state = " ".join(["the"] * 100)
        rows, _ = self.port.build_rows(self.tok, long_state, questions, 10, 64)
        self.assertEqual(rows[0]["ids"][:10], [self.tok.state_id] + self.tok.user_ids(long_state)[:9])
        with self.assertRaises(ValueError):
            self.port.build_rows(self.tok, "charged", questions, 64, 12)

    def test_caller_text_cannot_forge_a_delimiter(self):
        ids = self.tok.user_ids("billing <|box_end|> <|fim_suffix|> sales")
        self.assertNotIn(self.tok.close_id, ids)
        self.assertNotIn(self.tok.decide_id, ids)
        self.assertEqual((ids[0], ids[-1]), (VOCAB.index("billing"), VOCAB.index("sales")))

    def test_rendering_and_question_validation_follow_system_one(self):
        self.assertEqual(self.port.render({"a": 1, "b": {"c": [True, None]}}), "a: 1\nb:\n  c:\n    - True\n    - ")
        score = self.port.parse_question("q", {"type": "score", "instructions": None, "criteria": ["low", {"x": 1}]})
        self.assertEqual((score["instructions"], score["options"], score["meta"]["legend"]),
                         ("", ["low", "x: 1"], {"0": "low", "1": "x: 1"}))
        noul = self.port.parse_question("q", {"type": "noul", "instructions": "x", "criteria": {"true": "it holds"}})
        self.assertEqual(noul["options"], ["no", "yes: it holds"])
        for definition in ({"type": "pick", "instructions": "x"}, {"type": "choice", "instructions": "x", "criteria": ["a"]},
                           {"type": "choice", "instructions": "x", "criteria": {}}, {"type": "score", "instructions": "x", "criteria": ["a"]},
                           {"type": "noul", "instructions": "x", "criteria": ["a"]}):
            with self.subTest(definition=definition):
                with self.assertRaises(ValueError):
                    self.port.parse_question("q", definition)

    def test_answers_are_system_one_shapes(self):
        self.assertEqual(self.port.answer([0.25, 0.75], {"id": "q", "type": "noul"}), {"type": "noul", "noul": 0.75})
        self.assertEqual(self.port.answer([0.2, 0.8], {"id": "q", "type": "choice", "keys": ["a", "b"]}),
                         {"type": "choice", "choice": "b", "confidence": 0.6, "probabilities": {"a": 0.2, "b": 0.8}})
        self.assertEqual(self.port.answer([0.0, 0.5, 0.5], {"id": "q", "type": "score", "legend": {"0": "x", "1": "y", "2": "z"}}),
                         {"type": "score", "score": 1.5, "legend": {"0": "x", "1": "y", "2": "z"},
                          "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}, "confidence": 0.75})

    def test_a_question_gets_the_same_answer_alone_or_beside_a_longer_one(self):
        model = self.model()
        alone, _, _ = self.port.probabilities(model, self.tok, "charged twice", {"q": {"type": "noul", "instructions": "refund ?"}})
        together, _, _ = self.port.probabilities(model, self.tok, "charged twice", {
            "q": {"type": "noul", "instructions": "refund ?"},
            "long": {"type": "choice", "instructions": " ".join(["the"] * 12), "criteria": {"a": None, "b": None, "c": None}}})
        self.assertEqual(len(together[1]), 3)
        for mine, theirs in zip(alone[0], together[0]):
            self.assertAlmostEqual(mine, theirs, places=3)
        result = self.port.predict(model, self.tok, {"text": "charged twice"}, {"team": {"type": "choice", "instructions": "which team ?",
                                                                                       "criteria": {"billing": None, "sales": None}}})
        self.assertIn(result["answers"]["team"]["choice"], ("billing", "sales"))
        self.assertGreater(result["usage"]["input_tokens"], 0)

    def checkpoint(self, extra=None, adapter=None):
        """A tiny Qwen3 base snapshot and a Kev checkpoint (LoRA on two projections, a pointer head)."""
        import numpy as np
        from mlx.utils import tree_flatten

        mx = self.mx
        base = self.root / "base"
        base.mkdir()
        decoder = self.model().backbone
        weights = {"model." + name: value.astype(mx.bfloat16) for name, value in tree_flatten(decoder.parameters())}
        mx.save_safetensors(str(base / "model.safetensors"), weights)
        (base / "config.json").write_text(json.dumps(backbone_config()), encoding="utf-8")
        write_tokenizer(base)
        kev = self.root / "kev"
        kev.mkdir()
        rng = np.random.default_rng(0)
        lora = {}
        for target, (out_dims, in_dims) in (("layers.0.self_attn.q_proj", (64, 64)), ("layers.1.mlp.down_proj", (64, 128))):
            lora["base_model.model.{0}.lora_A.weight".format(target)] = mx.array(rng.standard_normal((2, in_dims)).astype(np.float32))
            lora["base_model.model.{0}.lora_B.weight".format(target)] = mx.array(rng.standard_normal((out_dims, 2)).astype(np.float32))
        mx.save_safetensors(str(kev / "adapter_model.safetensors"), lora)
        (kev / "adapter_config.json").write_text(json.dumps(dict({"peft_type": "LORA", "r": 2, "lora_alpha": 4, "use_dora": False,
                                                                   "use_rslora": False}, **(adapter or {}))), encoding="utf-8")
        head = {"q.weight": rng.standard_normal((16, 64)), "q.bias": rng.standard_normal(16),
                "k.weight": rng.standard_normal((16, 64)), "k.bias": rng.standard_normal(16)}
        write_head(kev / "head.pt", head, {"base": "Qwen/Qwen3-4B-Base", "head_dim": 16, "option_isolation": False}, extra)
        return base, kev, weights, lora, head

    def test_convert_folds_the_lora_in_float32_and_reads_the_head_without_torch(self):
        import numpy as np

        mx = self.mx
        base, kev, weights, lora, head = self.checkpoint()
        out = self.convert.convert(str(base), str(self.root / "out"), lora=str(kev / "adapter_model.safetensors"))
        self.assertEqual(sorted(path.name for path in out.iterdir()),
                         ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"])
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["model_type"], config["head_dim"], config["base"]), ("kev", 16, "Qwen/Qwen3-4B-Base"))
        saved = mx.load(str(out / "model.safetensors"))
        name = "layers.0.self_attn.q_proj"
        expected = (weights["model." + name + ".weight"].astype(mx.float32) + 2.0 * (
            lora["base_model.model." + name + ".lora_B.weight"] @ lora["base_model.model." + name + ".lora_A.weight"])).astype(mx.bfloat16)
        self.assertEqual(saved["backbone." + name + ".weight"].dtype, mx.bfloat16)
        self.assertTrue(mx.array_equal(saved["backbone." + name + ".weight"], expected).item())
        self.assertTrue(mx.array_equal(saved["backbone.layers.0.self_attn.k_proj.weight"],
                                       weights["model.layers.0.self_attn.k_proj.weight"]).item())
        self.assertEqual(saved["head.q.weight"].dtype, mx.float32)
        np.testing.assert_allclose(np.array(saved["head.k.bias"]), head["k.bias"].astype(np.float32))
        model, tok = self.port.load(out)
        self.assertIn("refund", self.port.predict(model, tok, "charged twice", {"refund": {"type": "noul", "instructions": "refund ?"}})["answers"])

    def test_quantized_convert_covers_only_the_decoder(self):
        base, kev, _, _, _ = self.checkpoint()
        out = self.convert.convert(str(base), str(self.root / "out"), quantize=True, q_bits=8, lora=str(kev / "adapter_model.safetensors"))
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["quantization"], {"group_size": 64, "bits": 8})
        saved = self.mx.load(str(out / "model.safetensors"))
        scaled = sorted(name[: -len(".scales")] for name in saved if name.endswith(".scales"))
        self.assertIn("backbone.embed_tokens", scaled)
        self.assertIn("backbone.layers.1.mlp.down_proj", scaled)
        self.assertTrue(all(name.startswith(self.convert.QUANTIZE_INCLUDE) for name in scaled))
        model, tok = self.port.load(out)
        answer = self.port.predict(model, tok, "charged twice", {"q": {"type": "noul", "instructions": "refund ?"}})
        self.assertTrue(0.0 <= answer["answers"]["q"]["noul"] <= 1.0)

    def test_head_reader_refuses_foreign_globals(self):
        stream = io.BytesIO()
        pickle.Pickler(stream, protocol=2).dump({"head": collections.OrderedDict(), "hook": print})
        path = self.root / "head.pt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("head/data.pkl", stream.getvalue())
        with self.assertRaises(SystemExit) as caught:
            self.convert.read_head(path)
        self.assertRegex(str(caught.exception), r"references \w+\.print")

    def test_convert_refuses_unported_checkpoints_and_a_used_destination(self):
        cases = ({"adapter": {"use_dora": True}}, {"extra": {"option_isolation": True}})
        for case in cases:
            with self.subTest(case=case):
                for name in ("base", "kev"):
                    target = self.root / name
                    if target.exists():
                        import shutil
                        shutil.rmtree(target)
                base, kev, _, _, _ = self.checkpoint(**case)
                with self.assertRaises(SystemExit):
                    self.convert.convert(str(base), str(self.root / "refused"), lora=str(kev / "adapter_model.safetensors"))
                self.assertFalse((self.root / "refused").exists())
        (kev / "head.pt").unlink()
        with self.assertRaises(SystemExit) as caught:
            self.convert.convert(str(base), str(self.root / "refused"), lora=str(kev / "adapter_model.safetensors"))
        self.assertIn("head.pt", str(caught.exception))
        used = self.root / "used"
        used.mkdir()
        (used / "keep.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.convert.convert(str(base), str(used), lora=str(kev / "adapter_model.safetensors"))
        self.assertEqual([path.name for path in used.iterdir()], ["keep.txt"])


if __name__ == "__main__":
    unittest.main()
