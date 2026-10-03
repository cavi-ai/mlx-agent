"""The bundled Laya port; runs where mlx-embeddings is importable (the backend venv)."""

import importlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.backends import PORTS_DIR

HAS_MLX_EMBEDDINGS = importlib.util.find_spec("mlx_embeddings") is not None
WORDS = ["billing", "technical", "sales", "refund", "charged", "twice", "question", "choice", "score", "noul",
         "level", "false", "true", "no", "yes", "the", "statement", "does", "not", "hold", "holds", "which",
         "team", "?", ":", ",", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]


def load_port():
    sys.path.insert(0, str(PORTS_DIR / "mlx-embeddings"))
    try:
        return importlib.import_module("laya.laya"), importlib.import_module("laya.convert")
    finally:
        sys.path.pop(0)


def tiny_encoder(**overrides):
    config = {
        "model_type": "modernbert", "vocab_size": 64, "hidden_size": 64, "num_hidden_layers": 3,
        "intermediate_size": 64, "num_attention_heads": 2, "global_attn_every_n_layers": 3, "local_attention": 4,
        "hidden_activation": "gelu", "norm_eps": 1e-5,
        "rope_parameters": {"full_attention": {"rope_theta": 160000.0}, "sliding_attention": {"rope_theta": 10000.0}},
    }
    config.update(overrides)
    return config


def tiny_settings():
    return {"head_layers": 1, "max_len": 40, "head_max_len": 24, "act_costs": {"escalate": 0.5},
            "temperature": [1.0, 2.0, 1.0], "temperature_by_options": {"choice:3-5": 2.0}, "amp_dtype": "bf16"}


def write_tokenizer(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {token: index for index, token in enumerate(["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"] + WORDS)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]", "mask_token": "[MASK]", "unk_token": "[UNK]",
    }), encoding="utf-8")


@unittest.skipUnless(HAS_MLX_EMBEDDINGS, "mlx-embeddings is not installed in this interpreter")
class LayaPortTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx

        self.mx = mx
        self.port, self.convert = load_port()
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        write_tokenizer(self.root / "tok")
        self.tok = self.port.RequestTokenizer(self.root / "tok")

    def model(self, **encoder):
        mx = self.mx
        mx.random.seed(0)
        config = self.convert.converted_config(tiny_settings(), tiny_encoder(**encoder))
        model = self.port.Model(self.port.ModelArgs.from_dict(config))
        mx.eval(model.parameters())
        return model

    def checkpoint(self, model):
        """The model's weights under the checkpoint's own names (packed attention, Sequential indices)."""
        from mlx.utils import tree_flatten

        weights = {}
        for name, value in tree_flatten(model.parameters()):
            name = name.replace("self_attn.in_proj.weight", "self_attn.in_proj_weight")
            name = name.replace("self_attn.in_proj.bias", "self_attn.in_proj_bias")
            name = name.replace("scorer.layers.", "scorer.").replace("act_head.layers.", "act_head.")
            weights[name] = value
        weights["temperature"] = self.mx.ones((3,))
        return weights

    def test_encoder_args_read_rope_per_layer_type_and_gelu_is_exact(self):
        model = self.model()
        attention = model.encoder.layers[1].attn
        self.assertEqual((model.encoder.layers[0].attn.rotary_emb.base, attention.rotary_emb.base), (160000.0, 10000.0))
        self.assertEqual({layer.mlp.act._approx for layer in model.encoder.layers}, {"none"})
        tanh = self.model(hidden_activation="gelu_pytorch_tanh")
        self.assertEqual({layer.mlp.act._approx for layer in tanh.encoder.layers}, {"tanh"})

    def test_sanitize_maps_checkpoint_names_and_drops_the_temperature_buffer(self):
        model = self.model()
        weights = self.port.Model.sanitize(self.checkpoint(model))
        self.assertNotIn("temperature", weights)
        self.assertIn("head.layers.0.self_attn.in_proj.weight", weights)
        self.assertIn("scorer.layers.3.weight", weights)
        self.assertIn("act_head.layers.2.bias", weights)
        model.load_weights(list(weights.items()), strict=True)

    def test_sequence_layout_puts_one_mask_marker_before_each_option(self):
        question = self.port.internal_question({"type": "choice", "instructions": "which team ?", "criteria": ["billing", "sales"]})
        ids, markers = self.port.build_sequence(self.tok, "charged twice", question, 40, 24)
        self.assertEqual(ids[0], self.tok.cls_id)
        self.assertEqual([ids[marker] for marker in markers], [self.tok.mask_id, self.tok.mask_id])
        self.assertEqual(ids[markers[0] + 1], self.tok.tokenizer.token_to_id("billing"))
        self.assertEqual(ids[-1], self.tok.sep_id)
        long_state = " ".join(["the"] * 100)
        ids, markers = self.port.build_sequence(self.tok, long_state, question, 40, 24)
        self.assertEqual((len(ids), ids[-1], len(markers)), (40, self.tok.sep_id, 2))
        many = self.port.internal_question({"type": "choice", "instructions": "x", "criteria": ["a b c d e f g"] * 1 + list("hijklmnop")})
        ids, markers = self.port.build_sequence(self.tok, "", many, 40, 24)
        self.assertEqual(len(markers), 10)
        self.assertLessEqual(markers[-1], 40)

    def test_question_definitions_are_validated(self):
        bad = [{"type": "pick", "instructions": "x"}, {"type": "choice", "instructions": "x", "criteria": ["only"]},
               {"type": "score", "instructions": "x", "criteria": {"a": 1}}, {"type": "noul", "instructions": "x", "criteria": ["a"]}]
        for definition in bad:
            with self.subTest(definition=definition):
                with self.assertRaises(ValueError):
                    self.port.internal_question(definition)
        noul = self.port.internal_question({"type": "noul", "instructions": {"ask": "does it hold"}})
        self.assertEqual(self.port.option_texts(noul)[0], "false: no, the statement does not hold")
        self.assertEqual(json.loads(noul["ins"]), {"ask": "does it hold"})

    def test_predict_answers_every_question_with_calibrated_distributions(self):
        model = self.model()
        questions = {
            "team": {"type": "choice", "instructions": "which team ?", "criteria": {"billing": None, "technical": "x", "sales": None}},
            "level": {"type": "score", "instructions": "score", "criteria": ["a", "b"]},
            "refund": {"type": "noul", "instructions": "refund ?"},
        }
        result = self.port.predict(model, self.tok, {"text": "charged twice"}, questions)
        answers = result["answers"]
        self.assertEqual(set(answers), {"team", "level", "refund"})
        self.assertIn(answers["team"]["choice"], ("billing", "technical", "sales"))
        self.assertAlmostEqual(sum(answers["team"]["probabilities"].values()), 1.0, places=3)
        self.assertEqual(answers["level"]["legend"], {"0": "a", "1": "b"})
        self.assertTrue(0.0 <= answers["refund"]["noul"] <= 1.0)
        self.assertGreater(result["usage"]["input_tokens"], 0)
        with self.assertRaises(ValueError):
            self.port.predict(model, self.tok, "x", {})

    def test_padding_does_not_change_an_answer(self):
        model = self.model()
        short = {"q": {"type": "noul", "instructions": "refund ?"}}
        alone = self.port.predict(model, self.tok, "charged twice", short)["answers"]["q"]["noul"]
        batched = dict(short, long={"type": "choice", "instructions": " ".join(["the"] * 12), "criteria": ["a", "b", "c", "d"]})
        together = self.port.predict(model, self.tok, "charged twice", batched)["answers"]["q"]["noul"]
        self.assertAlmostEqual(alone, together, places=3)

    def write_checkpoint(self, model):
        source = self.root / "source"
        (source / "encoder").mkdir(parents=True)
        self.mx.save_safetensors(str(source / "model.safetensors"), self.checkpoint(model))
        (source / "encoder" / "config.json").write_text(json.dumps(tiny_encoder()), encoding="utf-8")
        (source / "rl_agent_config.json").write_text(json.dumps(tiny_settings()), encoding="utf-8")
        write_tokenizer(source / "tokenizer")
        return source

    def test_convert_round_trip_quantizes_only_the_ported_layers(self):
        model = self.model()
        source = self.write_checkpoint(model)
        out = self.convert.convert(str(source), str(self.root / "out"), quantize=True, q_bits=4)
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["model_type"], config["quantization"]), ("laya", {"group_size": 32, "bits": 4}))
        self.assertNotIn("amp_dtype", config)
        self.assertEqual(sorted(path.name for path in out.iterdir()),
                         ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"])
        weights = self.mx.load(str(out / "model.safetensors"))
        scaled = sorted(name[: -len(".scales")] for name in weights if name.endswith(".scales"))
        self.assertTrue(scaled)
        self.assertTrue(all(name.startswith(self.convert.QUANTIZE_INCLUDE) for name in scaled))
        self.assertIn("head.layers.0.self_attn.in_proj", scaled)
        loaded, tok = self.port.load(out)
        answer = self.port.predict(loaded, tok, "charged twice", {"q": {"type": "noul", "instructions": "refund ?"}})
        self.assertIn("q", answer["answers"])

    def test_convert_refuses_a_partial_checkpoint_or_a_used_destination(self):
        source = self.write_checkpoint(self.model())
        (source / "rl_agent_config.json").unlink()
        with self.assertRaises(SystemExit) as caught:
            self.convert.convert(str(source), str(self.root / "out"))
        self.assertIn("rl_agent_config.json", str(caught.exception))
        used = self.root / "used"
        used.mkdir()
        (used / "keep.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.convert.convert(str(source), str(used))
        self.assertEqual([path.name for path in used.iterdir()], ["keep.txt"])


if __name__ == "__main__":
    unittest.main()
