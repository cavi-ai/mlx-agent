"""Deterministic facts for porting an unsupported architecture to MLX."""

from __future__ import annotations

import ast
import json

from .backends import load_manifests, load_registries
from .huggingface import HuggingFaceClient, HuggingFaceHTTPError
from .intake import TRANSPORT_ERRORS, components, summarize_files
from .intake_source import parse_hf_source, validate_revision

PORT_ANALYSIS_SCHEMA = "port-analysis/1"
MAX_CODE_BYTES = 512 * 1024
_TEXT_MARKERS = ("language_model", "lm_head", "embed_tokens")


class PortAnalysisError(RuntimeError):
    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def _prefix(key):
    parts = key.split(".")
    if parts[0] == "model" and len(parts) > 2:
        return ".".join(parts[:2])
    return parts[0]


def _prefix_role(prefix, roles):
    lowered = prefix.lower()
    if "audio" in lowered and "audio" in roles:
        return "audio"
    if ("vision" in lowered or "visual" in lowered) and "vision" in roles:
        return "vision"
    if any(marker in lowered for marker in _TEXT_MARKERS) or lowered in ("model.layers", "model.norm", "model.embed_tokens"):
        return "text" if "text" in roles else "model"
    return None


def weight_prefixes(index, roles):
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        return []
    counts = {}
    for key in weight_map:
        if isinstance(key, str):
            prefix = _prefix(key)
            counts[prefix] = counts.get(prefix, 0) + 1
    return [
        {"prefix": prefix, "tensors": counts[prefix], "component": _prefix_role(prefix, roles)}
        for prefix in sorted(counts)
    ]


def _classes(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {"parsed": False, "classes": []}
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "bases": [ast.unparse(base) for base in node.bases],
                "methods": [item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))],
                "lineno": node.lineno,
            })
    return {"parsed": True, "classes": classes}


def _code(client, repo, revision, names, warnings):
    ordered = sorted(names, key=lambda name: (not name.startswith("modeling_"), name))
    files, sources, used, truncated = [], {}, 0, False
    for name in ordered:
        try:
            text = client.fetch_raw_text(repo, revision, name)
        except TRANSPORT_ERRORS as error:
            warnings.append("{0} unreadable: {1}".format(name, error))
            continue
        size = len(text.encode("utf-8"))
        if used + size > MAX_CODE_BYTES:
            truncated = True
            warnings.append("{0} skipped: code budget of {1} bytes reached".format(name, MAX_CODE_BYTES))
            continue
        used += size
        sources[name] = text
        entry = {"name": name, "bytes": size}
        entry.update(_classes(text))
        files.append(entry)
    return {"files": files, "truncated": truncated}, sources


def _fetch_json(client, repo, revision, name, warnings):
    try:
        value = json.loads(client.fetch_raw_text(repo, revision, name))
    except TRANSPORT_ERRORS as error:
        warnings.append("{0} unreadable: {1}".format(name, error))
        return None
    return value if isinstance(value, dict) else None


def analyze_with_sources(text, revision=None, client=None, manifests=None, registries=None):
    source = parse_hf_source(text)
    if source.get("subfolder"):
        raise PortAnalysisError(
            "invalid_source", "Port analysis reads the repository root, not {0}/.".format(source["subfolder"]),
            "Pass the repository without the folder.",
        )
    if revision:
        source["revision"] = validate_revision(revision)
    client = client or HuggingFaceClient()
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests) if registries is None else registries
    repo, rev = source["repo"], source["revision"]
    try:
        info = client.fetch_model_info(repo, revision=rev)
    except HuggingFaceHTTPError as error:
        if error.status in (401, 403, 404, 410):
            raise PortAnalysisError("not_found_or_private", str(error), "Check the repository id and that it is public.") from error
        raise PortAnalysisError("hub_unreachable", str(error), "Check your network and retry.") from error
    except TRANSPORT_ERRORS as error:
        raise PortAnalysisError("hub_unreachable", str(error), "Check your network and retry.") from error
    if not isinstance(info, dict):
        raise PortAnalysisError("hub_unreachable", "The model API returned a non-object document.", "Retry later.")
    if info.get("gated"):
        raise PortAnalysisError("gated", "{0} is gated.".format(repo), "Gated repositories need a token, which intake does not handle.")
    files = summarize_files(info)
    if "config.json" not in files["names"]:
        raise PortAnalysisError("no_config", "{0} has no config.json.".format(repo), "Port analysis needs a transformers-style config.json.")
    warnings = []
    config = _fetch_json(client, repo, rev, "config.json", warnings)
    if config is None:
        raise PortAnalysisError("no_config", "config.json could not be read.", "Retry; if it persists the repository config is malformed.")
    model_type = config.get("model_type") if isinstance(config.get("model_type"), str) else None
    comps = components(model_type, config, manifests, registries)
    for component in comps:
        component["status"] = "exists" if component["matches"] else "missing"
    roles = {component["role"] for component in comps}
    index = None
    if "model.safetensors.index.json" in files["names"]:
        index = _fetch_json(client, repo, rev, "model.safetensors.index.json", warnings)
    prefixes = weight_prefixes(index, roles)
    indexed_files = set(index.get("weight_map", {}).values()) if isinstance(index, dict) else set()
    extra_files = sorted(
        name for name in files["names"]
        if name.endswith(".safetensors") and "/" not in name and indexed_files and name not in indexed_files
    )
    code, sources = _code(client, repo, rev, files["summary"]["python"], warnings)
    missing = (
        [{"kind": "component", "name": c["role"]} for c in comps if c["status"] == "missing"]
        + [{"kind": "weights", "name": p["prefix"]} for p in prefixes if p["component"] is None]
        + [{"kind": "file", "name": name} for name in extra_files]
    )
    architectures = [item for item in config.get("architectures") or [] if isinstance(item, str)]
    payload = {
        "schema": PORT_ANALYSIS_SCHEMA,
        "source": {"input": text.strip(), "repo": repo, "revision": rev, "file": source["file"],
                   "url": "https://huggingface.co/{0}".format(repo)},
        "model_type": model_type,
        "architectures": architectures,
        "processor_class": config.get("processor_class") if isinstance(config.get("processor_class"), str) else None,
        "transformers_version": config.get("transformers_version") if isinstance(config.get("transformers_version"), str) else None,
        "components": [
            {"role": c["role"], "config_key": c["config_key"], "model_type": c["model_type"],
             "status": c["status"], "matches": c["matches"]}
            for c in comps
        ],
        "weights": {"index_available": index is not None, "prefixes": prefixes, "extra_files": extra_files},
        "code": code,
        "missing": missing,
        "warnings": warnings,
    }
    return payload, sources


def analyze(text, revision=None, client=None, manifests=None, registries=None):
    return analyze_with_sources(text, revision, client, manifests, registries)[0]
