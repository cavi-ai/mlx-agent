"""Kev: a Qwen3 decoder with a pointer head over option spans (jaredpalmer/kev-4b).

A request renders the state once, then each question as ``<q> instructions``, one
``<opt> option </opt>`` span per option and a closing ``<decide>`` token. Every question runs
as its own causal row of the state tokens followed by its branch, which matches the
checkpoint's block-causal packed layout token for token and position for position. The head
scores the hidden state at each ``</opt>`` against the ``<decide>`` state; a softmax over the
options gives the answer distribution.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_embeddings.models.qwen3 import ModelArgs as BackboneArgs
from mlx_embeddings.models.qwen3 import Qwen3Model

# Existing Qwen special tokens reused as delimiters: state, question, option open, option close, decide.
DELIMITERS = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
QUESTION_TYPES = ("noul", "choice", "score")
MAX_OPTIONS = 255
_CONTROL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


@dataclass
class ModelArgs:
    backbone: dict
    model_type: str = "kev"
    head_dim: int = 256
    max_state: int = 8192
    max_branch: int = 8192

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        names = cls.__dataclass_fields__
        return cls(**{key: value for key, value in params.items() if key in names})


class PointerHead(nn.Module):
    """Scaled dot product between the ``<decide>`` state and each option's closing state."""

    def __init__(self, dims: int, pointer: int):
        super().__init__()
        self.q = nn.Linear(dims, pointer)
        self.k = nn.Linear(dims, pointer)
        self.scale = pointer ** -0.5

    def __call__(self, decide: mx.array, options: mx.array) -> mx.array:
        query = self.q(decide)[:, :, None]
        return (self.k(options) @ query).squeeze(-1) * self.scale


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.backbone = Qwen3Model(BackboneArgs.from_dict(args.backbone))
        self.head = PointerHead(self.backbone.hidden_size, args.head_dim)

    def __call__(self, ids, lengths, decide, options, present):
        """Option logits [rows, options] for right-padded rows; absent options are -inf."""
        attention = (mx.arange(ids.shape[1])[None, :] < lengths[:, None]).astype(mx.int32)
        hidden = self.backbone(ids, attention_mask=attention).astype(mx.float32)
        decided = hidden[mx.arange(ids.shape[0]), decide]
        spans = mx.take_along_axis(hidden, options[:, :, None], axis=1)
        return mx.where(present, self.head(decided, spans), -mx.inf)


# ----------------------------------------------------------------------------- requests


class RequestTokenizer:
    """Delimiter ids and caller-text tokenization over a ``tokenizers`` file (no transformers)."""

    def __init__(self, directory: Path):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        ids = [self.tokenizer.token_to_id(token) for token in DELIMITERS]
        missing = [token for token, value in zip(DELIMITERS, ids) if value is None]
        if missing:
            raise ValueError("the tokenizer has no delimiter tokens {0}".format(", ".join(missing)))
        self.state_id, self.question_id, self.open_id, self.close_id, self.decide_id = ids
        settings = json.loads((Path(directory) / "tokenizer_config.json").read_text(encoding="utf-8"))
        pad = settings.get("pad_token")
        pad = pad.get("content") if isinstance(pad, dict) else pad
        self.pad_id = (self.tokenizer.token_to_id(pad) if isinstance(pad, str) else None) or 0

    def user_ids(self, text: str) -> List[int]:
        """Caller text never yields a delimiter: ``<|name|>`` is rewritten to ``<¦name¦>`` first."""
        return self.tokenizer.encode(_CONTROL.sub(r"<¦\1¦>", text), add_special_tokens=False).ids

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


def render(value, indent: int = 0) -> str:
    """Text, objects and arrays flattened to the text the model reads; field names stay as labels."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join("{0}- {1}".format(pad, render(item, indent + 1).lstrip()) for item in value)
    return "\n".join(
        "{0}{1}:\n{2}".format(pad, key, render(item, indent + 1)) if isinstance(item, (dict, list))
        else "{0}{1}: {2}".format(pad, key, render(item))
        for key, item in value.items()
    )


def option_text(name: str, description) -> str:
    return name if description is None or description == "" else "{0}: {1}".format(name, render(description))


def parse_question(name: str, definition: dict) -> dict:
    """A request question ({type, instructions, criteria}) as instruction text, option texts and answer metadata."""
    if not isinstance(definition, dict) or definition.get("type") not in QUESTION_TYPES:
        raise ValueError("question {0!r} needs a type of {1}".format(name, ", ".join(QUESTION_TYPES)))
    kind, criteria = definition["type"], definition.get("criteria")
    if kind == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise ValueError("question {0!r}: noul criteria are an object with optional true and false texts".format(name))
        criteria = criteria or {}
        options = [option_text("no", criteria.get("false")), option_text("yes", criteria.get("true"))]
        meta = {"id": name, "type": "noul"}
    elif kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError("question {0!r}: choice criteria are an object of 1 to {1} options".format(name, MAX_OPTIONS))
        options = [option_text(key, value) for key, value in criteria.items()]
        meta = {"id": name, "type": "choice", "keys": list(criteria)}
    else:
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError("question {0!r}: score criteria are a list of 2 to {1} levels".format(name, MAX_OPTIONS))
        options = [render(level) for level in criteria]
        meta = {"id": name, "type": "score", "legend": {str(index): render(level) for index, level in enumerate(criteria)}}
    return {"instructions": render(definition.get("instructions")), "options": options, "meta": meta}


def build_rows(tok: RequestTokenizer, state, parsed: List[dict], max_state: int, max_branch: int):
    """One causal row per question (state + branch) with its readout offsets, and the packed token count."""
    prefix = [tok.state_id] + tok.user_ids(render(state))[: max_state - 1]
    rows, packed = [], len(prefix)
    for question in parsed:
        branch = [tok.question_id] + tok.user_ids(question["instructions"])
        closes = []
        for text in question["options"]:
            branch += [tok.open_id] + tok.user_ids(text) + [tok.close_id]
            closes.append(len(prefix) + len(branch) - 1)
        branch.append(tok.decide_id)
        if len(branch) > max_branch - len(prefix):
            raise ValueError("question {0!r}: branch too long: {1} tokens".format(question["meta"]["id"], len(branch)))
        rows.append({"ids": prefix + branch, "decide": len(prefix) + len(branch) - 1, "options": closes})
        packed += len(branch)
    return rows, packed


def batch(rows: List[dict], pad_id: int):
    length = max(len(row["ids"]) for row in rows)
    width = max(len(row["options"]) for row in rows)
    ids, options, present = [], [], []
    for row in rows:
        ids.append(row["ids"] + [pad_id] * (length - len(row["ids"])))
        count = len(row["options"])
        options.append(row["options"] + [0] * (width - count))
        present.append([True] * count + [False] * (width - count))
    return (
        mx.array(ids, dtype=mx.int32), mx.array([len(row["ids"]) for row in rows], dtype=mx.int32),
        mx.array([row["decide"] for row in rows], dtype=mx.int32),
        mx.array(options, dtype=mx.int32), mx.array(present),
    )


def r2(value: float) -> float:
    return round(float(value), 2)


def choice_confidence(p: List[float]) -> float:
    k = len(p)
    return 1.0 if k == 1 else (max(p) - 1 / k) / (1 - 1 / k)


def score_confidence(p: List[float]) -> float:
    """One minus the expected distance from the modal level, as a fraction of the scale."""
    levels = len(p)
    mode = max(range(levels), key=lambda index: p[index])
    return 1.0 - sum(value * abs(index - mode) for index, value in enumerate(p)) / (levels - 1)


def answer(p: List[float], meta: dict) -> dict:
    if meta["type"] == "noul":
        return {"type": "noul", "noul": r2(p[1])}
    if meta["type"] == "choice":
        keys = meta["keys"]
        return {"type": "choice", "choice": keys[max(range(len(p)), key=lambda index: p[index])],
                "confidence": r2(choice_confidence(p)), "probabilities": {key: r2(value) for key, value in zip(keys, p)}}
    return {"type": "score", "score": r2(sum(index * value for index, value in enumerate(p))), "legend": meta["legend"],
            "probabilities": {str(index): r2(value) for index, value in enumerate(p)}, "confidence": r2(score_confidence(p))}


def probabilities(model: Model, tok: RequestTokenizer, state, questions: Dict[str, dict]):
    """Option distributions in question order, the parsed questions, and the packed token count."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object of question definitions")
    parsed = [parse_question(name, definition) for name, definition in questions.items()]
    rows, packed = build_rows(tok, state, parsed, model.args.max_state, model.args.max_branch)
    p = mx.softmax(model(*batch(rows, tok.pad_id)), axis=-1)
    mx.eval(p)
    return [values[:len(row["options"])] for values, row in zip(p.tolist(), rows)], parsed, packed


def predict(model: Model, tok: RequestTokenizer, state, questions: Dict[str, dict]) -> dict:
    """Answer every question about one state (System One answer shapes)."""
    distributions, parsed, packed = probabilities(model, tok, state, questions)
    answers = {question["meta"]["id"]: answer(p, question["meta"]) for p, question in zip(distributions, parsed)}
    return {"answers": answers, "usage": {"input_tokens": packed, "output_tokens": tok.count(json.dumps(answers))}}


def load(path, dtype: Optional[mx.Dtype] = None):
    """A converted Kev directory: (model, tokenizer)."""
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
        model.backbone.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())
    return model, RequestTokenizer(directory)
