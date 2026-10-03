"""Laya: a ModernBERT encoder with an option-marker decision head (convaiinnovations/laya).

Each question is rendered as ``[CLS] <type> question: <instructions> [SEP] [MASK] option ... [SEP] state [SEP]``;
the head scores the hidden state at every option's ``[MASK]`` and a softmax over the question's
options gives its answer distribution. All questions of a request run in one batch. Weight names
are the checkpoint's own, apart from the head's packed attention and ``nn.Sequential`` indices.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_embeddings.models.modernbert import ModelArgs as EncoderArgs
from mlx_embeddings.models.modernbert import ModernBertModel

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {value: key for key, value in QTYPES.items()}
OPTION_TOKENS = 48
MIN_OPTION_BUDGET = 16
MASKED_LOGIT = -1e4
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
# transformers' "gelu" is the exact erf form; mlx-embeddings' ModernBERT MLP uses the tanh approximation.
TANH_GELU = {"gelu_new": "tanh", "gelu_pytorch_tanh": "tanh", "gelu_fast": "tanh"}


@dataclass
class ModelArgs:
    encoder: dict
    model_type: str = "laya"
    head_layers: int = 2
    max_len: int = 512
    head_max_len: int = 192
    act_costs: Dict[str, float] = field(default_factory=lambda: {"escalate": 0.5})
    temperature: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    temperature_by_options: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        names = cls.__dataclass_fields__
        return cls(**{key: value for key, value in params.items() if key in names})

    @property
    def n_act(self) -> int:
        return len(self.act_costs) + 1


def encoder_args(config: dict) -> EncoderArgs:
    """ModernBERT arguments from a transformers 4 or 5 encoder config (``rope_parameters`` per layer type)."""
    values = dict(config)
    rope = values.get("rope_parameters")
    if isinstance(rope, dict):
        full = rope.get("full_attention") or {}
        local = rope.get("sliding_attention") or {}
        if "rope_theta" in full:
            values["global_rope_theta"] = full["rope_theta"]
        if "rope_theta" in local:
            values["local_rope_theta"] = local["rope_theta"]
    return EncoderArgs.from_dict(values)


class HeadAttention(nn.Module):
    """``torch.nn.MultiheadAttention`` with a packed input projection and a key padding mask."""

    def __init__(self, dims: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj = nn.Linear(dims, 3 * dims, bias=True)
        self.out_proj = nn.Linear(dims, dims, bias=True)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        batch, length, dims = x.shape
        head_dim = dims // self.heads
        qkv = self.in_proj(x).reshape(batch, length, 3, self.heads, head_dim).transpose(2, 0, 3, 1, 4)
        out = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=head_dim ** -0.5, mask=mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, dims))


class HeadLayer(nn.Module):
    """``torch.nn.TransformerEncoderLayer(norm_first=True)``: pre-norm attention, then a ReLU feed-forward."""

    def __init__(self, dims: int):
        super().__init__()
        self.self_attn = HeadAttention(dims, max(1, dims // 64))
        self.linear1 = nn.Linear(dims, 4 * dims)
        self.linear2 = nn.Linear(4 * dims, dims)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.self_attn(self.norm1(x), mask)
        return x + self.linear2(nn.relu(self.linear1(self.norm2(x))))


class Head(nn.Module):
    def __init__(self, dims: int, count: int):
        super().__init__()
        self.layers = [HeadLayer(dims) for _ in range(count)]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.encoder = ModernBertModel(encoder_args(args.encoder))
        approx = TANH_GELU.get(args.encoder.get("hidden_activation", "gelu"), "none")
        for layer in self.encoder.layers:
            layer.mlp.act = nn.GELU(approx=approx)
        dims = self.encoder.config.hidden_size
        self.head = Head(dims, args.head_layers)
        self.type_emb = nn.Embedding(3, dims)
        self.scorer = nn.Sequential(nn.LayerNorm(dims), nn.Linear(dims, dims), nn.GELU(), nn.Linear(dims, 1))
        self.act_head = nn.Sequential(nn.Linear(dims + 4, 256), nn.GELU(), nn.Linear(256, args.n_act))

    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        dtype = self.type_emb.weight.dtype
        h = self.encoder(input_ids, attention_mask=attention_mask.astype(dtype))["last_hidden_state"]
        h = h + self.type_emb(qtype)[:, None, :]
        key_mask = mx.where(attention_mask[:, None, None, :] > 0, 0.0, -mx.inf).astype(h.dtype)
        for layer in self.head.layers:
            h = layer(h, key_mask)
        picked = mx.take_along_axis(h, marker_pos[:, :, None], axis=1)
        logits = self.scorer(picked).squeeze(-1).astype(mx.float32)
        logits = mx.where(marker_mask, logits, MASKED_LOGIT)
        p = mx.softmax(logits, axis=-1)
        k = mx.maximum(marker_mask.sum(-1), 2).astype(mx.float32)
        entropy = -(p * mx.log(mx.maximum(p, 1e-9))).sum(-1) / mx.log(k)
        top = mx.sort(p, axis=-1)
        features = mx.stack([top[:, -1], top[:, -1] - top[:, -2], entropy, k / 255.0], axis=-1)
        pooled = h[:, 0].astype(mx.float32)
        act_logits = self.act_head(mx.concatenate([pooled, features], axis=-1).astype(dtype))
        return logits, act_logits.astype(mx.float32)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Checkpoint names to module names: packed head attention, ``nn.Sequential`` indices, no buffers."""
        result = {}
        for name, value in weights.items():
            if name == "temperature":
                continue
            name = name.replace("self_attn.in_proj_weight", "self_attn.in_proj.weight")
            name = name.replace("self_attn.in_proj_bias", "self_attn.in_proj.bias")
            for sequential in ("scorer.", "act_head."):
                if name.startswith(sequential) and not name.startswith(sequential + "layers."):
                    name = sequential + "layers." + name[len(sequential):]
            result[name] = value
        return result


# ----------------------------------------------------------------------------- requests


class RequestTokenizer:
    """The tokenizer calls request rendering needs, over a ``tokenizers`` file (no transformers)."""

    def __init__(self, directory: Path):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        settings = json.loads((Path(directory) / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.mask_token = settings.get("mask_token", "[MASK]")
        ids = {}
        for role in ("cls", "sep", "pad", "mask"):
            token = settings.get(role + "_token", "[{0}]".format(role.upper()))
            value = self.tokenizer.token_to_id(token)
            if value is None:
                raise ValueError("the tokenizer has no {0} token {1!r}".format(role, token))
            ids[role] = value
        self.cls_id, self.sep_id, self.pad_id, self.mask_id = ids["cls"], ids["sep"], ids["pad"], ids["mask"]

    def ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids


def internal_question(definition: dict) -> dict:
    """A request question ({type, instructions, criteria}) in the rendering form ({t, ins, crit})."""
    kind = definition.get("type")
    if kind not in QTYPES:
        raise ValueError("question type must be one of {0}".format(sorted(QTYPES)))
    criteria = definition.get("criteria")
    if kind == "choice" and isinstance(criteria, list):
        criteria = {str(item): None for item in criteria}
    if kind == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2):
        raise ValueError("a choice question needs at least two criteria")
    if kind == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
        raise ValueError("a score question needs a list of at least two levels")
    if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
        raise ValueError("noul criteria are an object with optional true and false texts")
    instructions = definition.get("instructions")
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions)
    return {"t": kind, "ins": instructions, "crit": criteria}


def option_texts(question: dict) -> List[str]:
    """Option texts in label order; noul is always [false, true]."""
    kind, criteria = question["t"], question.get("crit")
    if kind == "choice":
        return [key if not value else "{0}: {1}".format(key, value) for key, value in criteria.items()]
    if kind == "score":
        return ["level {0}: {1}".format(index, text) for index, text in enumerate(criteria)]
    criteria = criteria or {}
    return [
        "false: " + (criteria.get("false") or "no, the statement does not hold"),
        "true: " + (criteria.get("true") or "yes, the statement holds"),
    ]


def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def build_sequence(tok: RequestTokenizer, state, question: dict, max_len: int, head_max_len: int):
    """Token ids for one question and the positions of its option markers."""
    mask = tok.mask_token
    head = tok.ids("{0} question: {1}".format(question["t"], str(question["ins"]).replace(mask, " ")))
    options = [[tok.mask_id] + tok.ids(" " + text.replace(mask, " "))[:OPTION_TOKENS] for text in option_texts(question)]
    budget = head_max_len - sum(len(option) for option in options)
    if budget < MIN_OPTION_BUDGET:
        per_option = max(4, (head_max_len - MIN_OPTION_BUDGET) // max(1, len(options)))
        options = [option[:per_option] for option in options]
        budget = head_max_len - sum(len(option) for option in options)
    ids = [tok.cls_id] + head[:max(8, budget)] + [tok.sep_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tok.sep_id)
    room = max(0, max_len - len(ids) - 1)
    ids = ids + tok.ids(state_text(state).replace(mask, " "))[:room] + [tok.sep_id]
    return ids[:max_len], [marker for marker in markers if marker < max_len]


def temperature_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "{0}:{1}".format(QTYPE_NAMES[qtype], size)


def confidence(p: List[float]) -> float:
    """One minus the normalized entropy of the answer distribution."""
    if len(p) < 2:
        return 1.0
    entropy = -sum(value * math.log(min(max(value, 1e-12), 1.0)) for value in p)
    return 1.0 - entropy / math.log(len(p))


def batch(items: List[dict], pad_id: int):
    length = max(len(item["ids"]) for item in items)
    width = max(len(item["markers"]) for item in items)
    ids, attention, positions, present = [], [], [], []
    for item in items:
        padding = length - len(item["ids"])
        ids.append(item["ids"] + [pad_id] * padding)
        attention.append([1] * len(item["ids"]) + [0] * padding)
        k = len(item["markers"])
        positions.append(item["markers"] + [0] * (width - k))
        present.append([True] * k + [False] * (width - k))
    return (
        mx.array(ids, dtype=mx.int32), mx.array(attention, dtype=mx.int32),
        mx.array(positions, dtype=mx.int32), mx.array(present),
        mx.array([item["qtype"] for item in items], dtype=mx.int32),
    )


def predict(model: Model, tok: RequestTokenizer, state, questions: Dict[str, dict]) -> dict:
    """Answer every question about one state in one forward pass (calibrated by the shipped temperatures)."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object of question definitions")
    args = model.args
    names, parsed, items = list(questions), [], []
    for name in names:
        question = internal_question(questions[name])
        ids, markers = build_sequence(tok, state, question, args.max_len, args.head_max_len)
        if len(markers) != len(option_texts(question)):
            raise ValueError("question {0!r}: options do not fit in {1} tokens".format(name, args.head_max_len))
        parsed.append(question)
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[question["t"]]})
    logits, act = model(*batch(items, tok.pad_id))
    act = mx.softmax(act, axis=-1)
    mx.eval(logits, act)
    logits, act = logits.tolist(), act.tolist()
    answers = {}
    for row, (name, question) in enumerate(zip(names, parsed)):
        k, qtype = len(items[row]["markers"]), QTYPES[question["t"]]
        scale = args.temperature_by_options.get(temperature_bucket(qtype, k), args.temperature[qtype])
        z = [value / scale for value in logits[row][:k]]
        top = max(z)
        weights = [math.exp(value - top) for value in z]
        total = sum(weights)
        p = [value / total for value in weights]
        extra = {"act_probability": round(act[row][0], 4)}
        if question["t"] == "choice":
            keys = list(question["crit"])
            answers[name] = {
                "type": "choice", "choice": keys[p.index(max(p))],
                "probabilities": {key: round(value, 4) for key, value in zip(keys, p)},
                "confidence": round(confidence(p), 4), "laya": extra,
            }
        elif question["t"] == "score":
            answers[name] = {
                "type": "score", "score": round(sum(index * value for index, value in enumerate(p)), 4),
                "legend": {str(index): text for index, text in enumerate(question["crit"])},
                "probabilities": {str(index): round(value, 4) for index, value in enumerate(p)},
                "confidence": round(confidence(p), 4), "laya": extra,
            }
        else:
            answers[name] = {"type": "noul", "noul": round(p[1], 4), "laya": extra}
    tokens = sum(len(item["ids"]) for item in items)
    return {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}


def load(path, dtype: Optional[mx.Dtype] = None):
    """A converted Laya directory: (model, tokenizer)."""
    directory = Path(path)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = Model(ModelArgs.from_dict(config))
    weights = {}
    for shard in sorted(directory.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    quantization = config.get("quantization")
    if quantization:
        nn.quantize(
            model, group_size=quantization["group_size"], bits=quantization["bits"],
            class_predicate=lambda name, module: "{0}.scales".format(name) in weights,
        )
    model.load_weights(list(weights.items()), strict=True)
    if dtype is not None:
        model.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())
    return model, RequestTokenizer(directory)
