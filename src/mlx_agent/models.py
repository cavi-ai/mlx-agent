"""Pure model classification, sizing, wiring, and rendering helpers."""

from __future__ import annotations

import json
import re


ROLES = [
    ("vision", ["-vl", "vision", "vlm", "ocr", "-omni"], "Vision / OCR (needs mlx-vlm)"),
    ("embedding", ["embed", "embedding", "bge", "nomic", "gte"], "Embeddings"),
    ("coding", ["coder", "-code", "devstral", "codestral", "starcoder"], "Coding"),
    ("reasoning", ["gpt-oss", "-a3b", "thinking", "reason", "qwq", "-r1", "deepseek-r"], "Reasoning"),
    ("general", ["instruct", "-it", "chat", "mistral", "gemma", "qwen", "llama", "phi"], "General"),
]
REASONER_HINTS = re.compile(r"(thinking|reason|-a3b\b|qwq|-r1\b|deepseek-r|gpt-oss)", re.I)
TEMPLATE_REASON = re.compile(r"(reasoning_effort|<think>|<\|think|channel\|>analysis)", re.I)
REPUTABLE = {"mlx-community", "lmstudio-community", "unsloth", "mlx-omni", "mlxvlm", "qwen", "google", "mistralai", "meta-llama", "nvidia"}
QUANT_TOK = re.compile(r"[-_.]?(mlx|mxfp4|nvfp4|dwq|optiq|turboquant|rotorquant|oq4e?|mtp|q\d(_k(_[ms])?)?|int[48]|fp16|fp8|bf16|\d+\.?\d*bpw|\d+bit|e[24]b|4bit|8bit)$", re.I)
PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z])", re.I)
QUANT_GB_PER_B = [
    (re.compile(r"(fp16|bf16|-16bit|f16)", re.I), 2.0),
    (re.compile(r"(8bit|-q8|int8|fp8|mxfp8)", re.I), 1.0),
    (re.compile(r"(6bit|-q6)", re.I), 0.75),
    (re.compile(r"(4bit|-q4|mxfp4|nvfp4|int4)", re.I), 0.55),
    (re.compile(r"(3bit|-q3)", re.I), 0.45),
    (re.compile(r"(2bit|-q2)", re.I), 0.35),
]
QUANT_RANK = [("bf16", 6), ("fp16", 6), ("8bit", 5), ("q8", 5), ("6bit", 4), ("q6", 4), ("4bit", 3), ("q4", 3), ("mxfp4", 3), ("3bit", 2), ("2bit", 1)]


def infer_quantization(value):
    """Return a normalized name-derived quantization, never a measured property."""
    low = value.lower()
    for name, patterns in [
        ("fp16", ("fp16", "bf16", "f16")),
        ("8bit", ("8bit", "q8", "int8", "fp8")),
        ("6bit", ("6bit", "q6")),
        ("4bit", ("4bit", "q4", "int4", "mxfp4", "nvfp4")),
        ("3bit", ("3bit", "q3")),
        ("2bit", ("2bit", "q2")),
    ]:
        if any(pattern in low for pattern in patterns):
            return name
    return None


def name_ram_gb(repo):
    match = PARAM_RE.search(repo)
    if not match:
        return None
    for pattern, value in QUANT_GB_PER_B:
        if pattern.search(repo):
            return round(float(match.group(1)) * value, 1)
    return round(float(match.group(1)) * 0.55, 1)


def classify(repo):
    low = repo.lower()
    for role, keywords, _label in ROLES:
        if any(keyword in low for keyword in keywords):
            return role
    return "general"


def base_name(repo):
    name = repo.split("/")[-1]
    previous = None
    while name != previous:
        previous = name
        name = QUANT_TOK.sub("", name).rstrip("-_.")
    return name.lower()


def quant_rank(repo):
    low = repo.lower()
    for token, rank in QUANT_RANK:
        if token in low:
            return rank
    return 0


def resolve_ram(repo, enrichment):
    if enrichment.get("weight_bytes"):
        return round(enrichment["weight_bytes"] / 1e9, 1), "actual"
    if enrichment.get("params_total"):
        for pattern, value in QUANT_GB_PER_B:
            if pattern.search(repo):
                return round(enrichment["params_total"] * value / 1e9, 1), "est"
        return round(enrichment["params_total"] * 0.55 / 1e9, 1), "est"
    estimate = name_ram_gb(repo)
    return (estimate, "est") if estimate is not None else (None, None)


def resolve_reasoning(repo, enrichment):
    if enrichment.get("reasoning") is not None:
        return enrichment["reasoning"], enrichment.get("reason_src")
    return bool(REASONER_HINTS.search(repo)), "name"


def wiring(repo, role, host):
    short = repo.split("/")[-1].lower()
    if role == "vision":
        return "mlx-vlm server → `mlxvlm/{0}`".format(repo.split("/")[-1])
    if host.get("lmstudio"):
        return "LM Studio → `lmstudio/{0}`".format(short)
    return "`mlx_lm.server --model {0}` → custom provider".format(repo)


def render_md(report):
    host = report["host"]
    out = ["# mlx-scout report", ""]
    out.append("**Host:** {0} · {1}GB · Ollama {2} · LM Studio {3}".format(
        host.get("chip") or "?", host.get("ram_gb") or "?", "✓" if host["ollama"] else "✗", "✓" if host["lmstudio"] else "✗") + ("  _(fast mode: name heuristics)_" if report.get("fast") else ""))
    if report.get("error"):
        return "\n".join(out + ["\n> ⚠ {0}".format(report["error"])])
    labels = {role: label for role, _keywords, label in ROLES}
    for role, items in report["roles"].items():
        out.extend(["\n## {0}".format(labels.get(role, role)), "| model | RAM | reasoning | fits | license | wiring |", "|---|---|---|---|---|---|"])
        for item in items:
            ram = "{0}GB".format(item["est_ram_gb"]) if item.get("est_ram_gb") is not None else "?"
            if item.get("ram_src") == "actual":
                ram += "*"
            reason = "⚠ {0}".format(item.get("reason_src") or "") if item.get("reasoning") else "no"
            license_name = (item.get("license") or "?") + (" 🔒" if item.get("gated") else "")
            out.append("| `{0}`{1} | {2} | {3} | {4} | {5} | {6} |".format(item["repo"], " ⭐" if item.get("trusted") else "", ram, reason, "✓" if item["fits"] else "✗", license_name, item["wiring"]))
    out.append("\n_`*` = real download size (HF); else estimated. Add KV-cache headroom (~1–4GB) beyond weights._")
    out.append("_`reasoning ⚠` sources: chat_template > tags > name (weakest). Verify with a test-generate._")
    return "\n".join(out)


def wire(repo, target, port):
    short = repo.split("/")[-1]
    if target == "ollama":
        return "\n".join(["# Ollama runs curated tags / HF GGUF repos (not arbitrary mlx-community weights).", "ollama pull hf.co/{0}    # GGUF repos; else use the matching library tag".format(repo), "# agent ref: ollama/<tag>"])
    if target == "lmstudio":
        return "\n".join(["lms get {0}      # download into LM Studio".format(repo), "lms server start    # OpenAI-compatible API on http://localhost:1234/v1", "# agent ref: lmstudio/{0}".format(short.lower())])
    if target == "mlx_lm":
        return "\n".join(["pip install mlx-lm", "mlx_lm.server --model {0} --port {1} --max-tokens 8192".format(repo, port), "# openai-compatible provider:", json.dumps({"id": "mlxlm", "type": "openai", "baseURL": "http://127.0.0.1:{0}/v1".format(port), "apiKey": "local"}, indent=2), "# agent ref: mlxlm/{0}".format(short)])
    if target == "mlx-vlm":
        return "\n".join(["pip install mlx-vlm", "mlx_vlm.server --model {0} --port {1}".format(repo, port), json.dumps({"id": "mlxvlm", "type": "openai", "baseURL": "http://127.0.0.1:{0}/v1".format(port), "apiKey": "local"}, indent=2), "# agent ref: mlxvlm/{0}".format(short)])
    if target == "litellm":
        return "\n".join(["# litellm config.yaml (run a native server first, e.g. mlx_lm.server):", "model_list:", "  - model_name: {0}".format(short.lower()), "    litellm_params:", "      model: openai/{0}".format(repo), "      api_base: http://127.0.0.1:{0}/v1".format(port), "      api_key: local"])
    return "unknown target: {0}".format(target)
