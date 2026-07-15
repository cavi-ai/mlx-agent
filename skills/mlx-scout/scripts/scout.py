#!/usr/bin/env python3
"""mlx-scout — discover MLX-optimized models on HuggingFace suited to THIS host + role.

Self-contained (Python 3 stdlib only). Detects the local machine (Apple Silicon RAM +
available runtimes), queries the HuggingFace Hub for models carrying the `mlx` library
tag, buckets them by role, and — for the top candidates — enriches each with real
download size (HF tree API), reasoning detection (chat-template + tags), license/gated
status, and quant-variant dedup. Prints ready-to-use model refs.

Usage:
  python3 scout.py                 # all roles (enriched)
  python3 scout.py --role coding   # one role
  python3 scout.py --new           # sort by most-recently-updated
  python3 scout.py --fast          # skip per-model enrichment (name heuristics only)
  python3 scout.py --json          # machine-readable
  python3 scout.py --limit 8       # results per role
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, urllib.parse, urllib.request

HF_API = "https://huggingface.co/api/models"
UA = {"User-Agent": "mlx-scout/0.2 (+https://github.com/sasan1200/mlx-agent)"}

# role -> (name-match keywords, human label). Order matters: first match wins.
ROLES: list[tuple[str, list[str], str]] = [
    ("vision",     ["-vl", "vision", "vlm", "ocr", "-omni"],                "Vision / OCR (needs mlx-vlm)"),
    ("embedding",  ["embed", "embedding", "bge", "nomic", "gte"],           "Embeddings"),
    ("coding",     ["coder", "-code", "devstral", "codestral", "starcoder"],"Coding"),
    ("reasoning",  ["gpt-oss", "-a3b", "thinking", "reason", "qwq", "-r1", "deepseek-r"], "Reasoning"),
    ("general",    ["instruct", "-it", "chat", "mistral", "gemma", "qwen", "llama", "phi"], "General"),
]
REASONER_HINTS = re.compile(r"(thinking|reason|-a3b\b|qwq|-r1\b|deepseek-r|gpt-oss)", re.I)
# strong reasoning signals found inside a model's chat template
TEMPLATE_REASON = re.compile(r"(reasoning_effort|<think>|<\|think|channel\|>analysis)", re.I)
REPUTABLE = {"mlx-community", "lmstudio-community", "unsloth", "mlx-omni", "mlxvlm",
             "qwen", "google", "mistralai", "meta-llama", "nvidia"}
# quant / format tokens stripped to find a model's logical base name (for dedup)
QUANT_TOK = re.compile(
    r"[-_.]?(mlx|mxfp4|nvfp4|dwq|optiq|turboquant|rotorquant|oq4e?|mtp|"
    r"q\d(_k(_[ms])?)?|int[48]|fp16|fp8|bf16|\d+\.?\d*bpw|\d+bit|e[24]b|4bit|8bit)$", re.I)
PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z])", re.I)
QUANT_GB_PER_B = [
    (re.compile(r"(fp16|bf16|-16bit|f16)", re.I), 2.0),
    (re.compile(r"(8bit|-q8|int8|fp8|mxfp8)", re.I), 1.0),
    (re.compile(r"(6bit|-q6)", re.I), 0.75),
    (re.compile(r"(4bit|-q4|mxfp4|nvfp4|int4)", re.I), 0.55),
    (re.compile(r"(3bit|-q3)", re.I), 0.45),
    (re.compile(r"(2bit|-q2)", re.I), 0.35),
]
QUANT_RANK = [("bf16", 6), ("fp16", 6), ("8bit", 5), ("q8", 5), ("6bit", 4), ("q6", 4),
              ("4bit", 3), ("q4", 3), ("mxfp4", 3), ("3bit", 2), ("2bit", 1)]


def http_json(url: str, timeout: float = 10.0):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def detect_host() -> dict:
    host = {"ram_gb": None, "chip": None, "ollama": False, "lmstudio": False}
    try:
        host["ram_gb"] = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 1073741824)
    except Exception:
        pass
    try:
        host["chip"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    try:
        http_json("http://127.0.0.1:11434/api/tags", timeout=3); host["ollama"] = True
    except Exception:
        pass
    try:
        http_json("http://localhost:1234/v1/models", timeout=3); host["lmstudio"] = True
    except Exception:
        pass
    return host


def name_ram_gb(repo: str):
    m = PARAM_RE.search(repo)
    if not m:
        return None
    params = float(m.group(1))
    gb_per_b = 0.55
    for rx, v in QUANT_GB_PER_B:
        if rx.search(repo):
            gb_per_b = v
            break
    return round(params * gb_per_b, 1)


def classify(repo: str) -> str:
    low = repo.lower()
    for role, kws, _ in ROLES:
        if any(k in low for k in kws):
            return role
    return "general"


def base_name(repo: str) -> str:
    name = repo.split("/")[-1]
    prev = None
    while name != prev:
        prev = name
        name = QUANT_TOK.sub("", name).rstrip("-_.")
    return name.lower()


def quant_rank(repo: str) -> int:
    low = repo.lower()
    for tok, rank in QUANT_RANK:
        if tok in low:
            return rank
    return 0


def enrich(repo: str) -> dict:
    """Real size + metadata for one repo. Degrades gracefully on any failure."""
    out: dict = {"weight_bytes": None, "tags": [], "gated": False, "license": None,
                 "reasoning": None, "reason_src": None, "params_total": None}
    quoted = urllib.parse.quote(repo)
    try:
        m = http_json(f"{HF_API}/{quoted}", timeout=8)
        tags = m.get("tags", []) or []
        out["tags"] = tags
        out["gated"] = bool(m.get("gated"))
        cfg = m.get("config") or {}
        cd = m.get("cardData") or {}
        out["license"] = cd.get("license") or next(
            (t.split("license:", 1)[1] for t in tags if t.startswith("license:")), None)
        out["params_total"] = (m.get("safetensors") or {}).get("total")
        ct = ((cfg.get("tokenizer_config") or {}).get("chat_template") or "")
        low_tags = [t.lower() for t in tags]
        if TEMPLATE_REASON.search(ct):
            out["reasoning"], out["reason_src"] = True, "chat_template"
        elif any(t in ("reasoning", "thinking", "chain-of-thought") for t in low_tags):
            out["reasoning"], out["reason_src"] = True, "tags"
        elif REASONER_HINTS.search(repo):
            out["reasoning"], out["reason_src"] = True, "name"
        else:
            out["reasoning"], out["reason_src"] = False, "checked"
    except Exception:
        pass
    try:
        tree = http_json(f"{HF_API}/{quoted}/tree/main?recursive=true", timeout=8)
        wb = sum(f.get("size", 0) for f in tree
                 if f.get("path", "").endswith((".safetensors", ".gguf", ".bin")))
        out["weight_bytes"] = wb or None
    except Exception:
        pass
    return out


def resolve_ram(repo: str, enr: dict):
    wb = enr.get("weight_bytes")
    if wb:
        return round(wb / 1e9, 1), "actual"
    p = enr.get("params_total")
    if p:
        gb_per_b = 0.55
        for rx, v in QUANT_GB_PER_B:
            if rx.search(repo):
                gb_per_b = v; break
        return round(p * gb_per_b / 1e9, 1), "est"
    n = name_ram_gb(repo)
    return (n, "est") if n is not None else (None, None)


def resolve_reasoning(repo: str, enr: dict):
    if enr.get("reasoning") is not None:
        return enr["reasoning"], enr.get("reason_src")
    return bool(REASONER_HINTS.search(repo)), "name"


def fetch_mlx_models(sort: str = "trendingScore", limit_fetch: int = 300) -> list[dict]:
    q = urllib.parse.urlencode({"filter": "mlx", "sort": sort, "direction": "-1", "limit": limit_fetch})
    return http_json(f"{HF_API}?{q}")


def wiring(repo: str, role: str, host: dict) -> str:
    short = repo.split("/")[-1].lower()
    if role == "vision":
        return f"mlx-vlm server → `mlxvlm/{repo.split('/')[-1]}`"
    if host.get("lmstudio"):
        return f"LM Studio → `lmstudio/{short}`"
    return f"`mlx_lm.server --model {repo}` → custom provider"


def build_report(limit: int, only_role: str | None, new: bool = False, fast: bool = False) -> dict:
    host = detect_host()
    try:
        raw = fetch_mlx_models(sort="lastModified" if new else "trendingScore")
    except Exception as e:
        return {"host": host, "error": f"HuggingFace query failed: {e}", "roles": {}}

    buckets: dict[str, list[dict]] = {r[0]: [] for r in ROLES}
    seen_repo, seen_base = set(), {}
    for m in raw:
        rid = m.get("id") or m.get("modelId")
        if not rid or rid in seen_repo:
            continue
        seen_repo.add(rid)
        role = classify(rid)
        if only_role and role != only_role:
            continue
        item = {"repo": rid, "downloads": m.get("downloads", 0), "likes": m.get("likes", 0),
                "trusted": rid.split("/")[0].lower() in REPUTABLE, "base": base_name(rid),
                "qrank": quant_rank(rid)}
        # quant-variant dedup: keep the best-ranked quant per (role, base_name)
        key = (role, item["base"])
        if key in seen_base:
            cur = seen_base[key]
            better = (item["trusted"], item["qrank"], item["downloads"]) > \
                     (cur["trusted"], cur["qrank"], cur["downloads"])
            if better:
                buckets[role].remove(cur); buckets[role].append(item); seen_base[key] = item
            continue
        seen_base[key] = item
        buckets[role].append(item)

    for role in buckets:
        buckets[role].sort(key=lambda x: (x["trusted"], x["downloads"], x["likes"]), reverse=True)
        buckets[role] = buckets[role][:limit]

    # enrich survivors with real size / reasoning / license
    for role, items in buckets.items():
        for it in items:
            enr = {} if fast else enrich(it["repo"])
            ram, ram_src = resolve_ram(it["repo"], enr)
            reasoning, rsrc = resolve_reasoning(it["repo"], enr)
            it["est_ram_gb"] = ram
            it["ram_src"] = ram_src if not fast else "est"
            it["fits"] = (ram is None or host.get("ram_gb") is None or ram < host["ram_gb"] * 0.8)
            it["reasoning"] = reasoning
            it["reason_src"] = rsrc
            it["gated"] = enr.get("gated", False)
            it["license"] = enr.get("license")
            it["wiring"] = wiring(it["repo"], role, host)

    return {"host": host, "fast": fast, "roles": {k: v for k, v in buckets.items() if v}}


def render_md(rep: dict) -> str:
    h = rep["host"]
    out = ["# mlx-scout report", ""]
    out.append(f"**Host:** {h.get('chip') or '?'} · {h.get('ram_gb') or '?'}GB · "
               f"Ollama {'✓' if h['ollama'] else '✗'} · LM Studio {'✓' if h['lmstudio'] else '✗'}"
               + ("  _(fast mode: name heuristics)_" if rep.get("fast") else ""))
    if rep.get("error"):
        return "\n".join(out + [f"\n> ⚠ {rep['error']}"])
    labels = {r[0]: r[2] for r in ROLES}
    for role, items in rep["roles"].items():
        out.append(f"\n## {labels.get(role, role)}")
        out.append("| model | RAM | reasoning | fits | license | wiring |")
        out.append("|---|---|---|---|---|---|")
        for it in items:
            star = " ⭐" if it.get("trusted") else ""
            ram = f"{it['est_ram_gb']}GB" if it.get("est_ram_gb") is not None else "?"
            if it.get("ram_src") == "actual":
                ram += "*"
            reason = ("⚠ " + (it.get("reason_src") or "")) if it.get("reasoning") else "no"
            lic = (it.get("license") or "?") + (" 🔒" if it.get("gated") else "")
            out.append(f"| `{it['repo']}`{star} | {ram} | {reason} | "
                       f"{'✓' if it['fits'] else '✗'} | {lic} | {it['wiring']} |")
    out.append("\n_`*` = real download size (HF); else estimated. Add KV-cache headroom (~1–4GB) beyond weights._")
    out.append("_`reasoning ⚠` sources: chat_template > tags > name (weakest). Verify with a test-generate._")
    return "\n".join(out)


def wire(repo: str, target: str, port: int) -> str:
    """Emit setup commands + a ready config block to run `repo` under `target`."""
    short = repo.split("/")[-1]
    if target == "ollama":
        return "\n".join([
            "# Ollama runs curated tags / HF GGUF repos (not arbitrary mlx-community weights).",
            f"ollama pull hf.co/{repo}    # GGUF repos; else use the matching library tag",
            f"# agent ref: ollama/<tag>"])
    if target == "lmstudio":
        return "\n".join([
            f"lms get {repo}      # download into LM Studio",
            "lms server start    # OpenAI-compatible API on http://localhost:1234/v1",
            f"# agent ref: lmstudio/{short.lower()}"])
    if target == "mlx_lm":
        return "\n".join([
            "pip install mlx-lm",
            f"mlx_lm.server --model {repo} --port {port} --max-tokens 8192",
            "# openai-compatible provider:",
            json.dumps({"id": "mlxlm", "type": "openai",
                        "baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": "local"}, indent=2),
            f"# agent ref: mlxlm/{short}"])
    if target == "mlx-vlm":
        return "\n".join([
            "pip install mlx-vlm",
            f"mlx_vlm.server --model {repo} --port {port}",
            json.dumps({"id": "mlxvlm", "type": "openai",
                        "baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": "local"}, indent=2),
            f"# agent ref: mlxvlm/{short}"])
    if target == "litellm":
        return "\n".join([
            "# litellm config.yaml (run a native server first, e.g. mlx_lm.server):",
            "model_list:",
            f"  - model_name: {short.lower()}",
            "    litellm_params:",
            f"      model: openai/{repo}",
            f"      api_base: http://127.0.0.1:{port}/v1",
            "      api_key: local"])
    return f"unknown target: {target}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Discover MLX models on HuggingFace for this host.")
    ap.add_argument("--role", choices=[r[0] for r in ROLES])
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--new", action="store_true", help="sort by most-recently-updated")
    ap.add_argument("--fast", action="store_true", help="skip per-model enrichment (name heuristics only)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--wire", metavar="REPO", help="emit setup + config for a model, instead of discovering")
    ap.add_argument("--target", choices=["ollama", "lmstudio", "mlx_lm", "mlx-vlm", "litellm"], default="mlx_lm")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    if a.wire:
        print(wire(a.wire, a.target, a.port))
        return 0
    rep = build_report(a.limit, a.role, new=a.new, fast=a.fast)
    print(json.dumps(rep, indent=2) if a.json else render_md(rep))
    return 0 if not rep.get("error") else 2


if __name__ == "__main__":
    sys.exit(main())
