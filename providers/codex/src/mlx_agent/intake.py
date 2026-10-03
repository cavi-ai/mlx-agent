"""Resolve a pasted Hugging Face model into a conversion verdict (read-only)."""

from __future__ import annotations

import fnmatch
import http.client
import json
import math

from .backends import (
    choose_backend, component_matches, load_manifests, load_registries, lookup, port_bits, port_files,
    port_for_files, port_for_pipeline, quantize_rule, recipe_for, snapshot_files,
)
from .huggingface import HuggingFaceClient, HuggingFaceHTTPError
from .intake_fetch import FETCH_IGNORE_PATTERNS
from .intake_source import parse_hf_source, validate_revision
from .taxonomy import classify

INTAKE_SCHEMA = "intake/1"
TRANSPORT_ERRORS = (OSError, TimeoutError, http.client.HTTPException, ValueError)
_BLOCKING_STATUSES = (401, 403, 404, 410)
ESTIMATE_BITS = (4, 8)
ESTIMATE_MAX_SHARDS = 16
QUANT_GROUP_SIZE = 64
_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}
_SUPPORT_SUFFIXES = (".json", ".jinja", ".model", ".tiktoken", ".txt")


def _text(value):
    return value if isinstance(value, str) else None


def summarize_files(info):
    siblings = info.get("siblings") if isinstance(info.get("siblings"), list) else []
    names, gguf, python = set(), [], []
    total, safetensors, sizes = 0, 0, {}
    for sibling in siblings:
        if not isinstance(sibling, dict) or not isinstance(sibling.get("rfilename"), str):
            continue
        name = sibling["rfilename"]
        size = sibling.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            size = 0
        names.add(name)
        sizes[name] = size
        total += size
        lowered = name.lower()
        if lowered.endswith(".gguf"):
            gguf.append({"name": name, "bytes": size})
        elif lowered.endswith(".safetensors"):
            safetensors += 1
        elif lowered.endswith(".py") and "/" not in name:
            python.append(name)
    return {
        "names": names,
        "sizes": sizes,
        "bytes": total,
        "summary": {
            "safetensors": safetensors,
            "gguf": sorted(gguf, key=lambda item: item["name"]),
            "python": sorted(python),
        },
    }


def fetch_config(client, repo, revision, warnings, prefix=""):
    try:
        value = json.loads(client.fetch_raw_text(repo, revision, prefix + "config.json"))
    except TRANSPORT_ERRORS as error:
        warnings.append("config.json unreadable: {0}".format(error))
        return {}
    if not isinstance(value, dict):
        warnings.append("config.json is not a JSON object")
        return {}
    return value


def _group_size(rule):
    return (rule or {}).get("group_size", QUANT_GROUP_SIZE)


def _quantizes(name, shape, rule):
    if len(shape) < 2 or shape[-1] % _group_size(rule):
        return False
    if rule is None:
        return name.endswith(".weight")
    return any(name.startswith(prefix) for prefix in rule.get("include", ())) and not any(
        part in name for part in rule.get("exclude", ())
    )


def estimate_output_bytes(client, repo, revision, files, rule, warnings, support_files=None, prefix=""):
    """Converted size per bit width from the weight files' headers (no weights read).

    Quantized tensors cost bits/8 bytes per weight plus an affine scale and
    bias per group (64, or the port's group size) in the weight's dtype;
    everything else keeps its source size. Supporting files (config,
    tokenizer) are added as listed: root-level files, or the non-weight
    files a port's signature names.
    """
    shards = sorted(name for name in files["names"] if name.endswith(".safetensors") and "/" not in name)
    if not shards or len(shards) > ESTIMATE_MAX_SHARDS:
        return None
    if any(name.endswith(".pth") for name in files["names"]):
        # PyTorch pickles (a Wan checkpoint's text encoder and VAE) hold weights the headers do not describe.
        return None
    totals = {bits: 0 for bits in ESTIMATE_BITS}
    for shard in shards:
        try:
            header = client.fetch_safetensors_header(repo, revision, prefix + shard)
        except TRANSPORT_ERRORS as error:
            warnings.append("output size estimate unavailable: {0} header unreadable: {1}".format(shard, error))
            return None
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            shape = tensor.get("shape") if isinstance(tensor, dict) else None
            width = _DTYPE_BYTES.get(tensor.get("dtype")) if isinstance(tensor, dict) else None
            if width is None or not isinstance(shape, list) or not all(
                isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape
            ):
                warnings.append("output size estimate unavailable: {0} has an unreadable tensor {1}".format(shard, name))
                return None
            count = math.prod(shape)
            for bits in totals:
                if _quantizes(name, shape, rule):
                    totals[bits] += -(-count * bits // 8) + (count // _group_size(rule)) * 2 * width
                else:
                    totals[bits] += count * width
    if support_files is not None:
        support = sum(files["sizes"].get(name, 0) for name in support_files if not name.endswith(".safetensors"))
    else:
        support = sum(
            size for name, size in files["sizes"].items()
            if "/" not in name and name.endswith(_SUPPORT_SUFFIXES) and name != "model.safetensors.index.json"
        )
    return {str(bits): total + support for bits, total in totals.items()}


def components(model_type, config, manifests, registries):
    result = []
    if model_type:
        result.append({
            "role": "model", "config_key": None, "model_type": model_type,
            "matches": component_matches(model_type, manifests, registries),
        })
    for key in sorted(config):
        value = config[key]
        if key.endswith("_config") and isinstance(value, dict) and isinstance(value.get("model_type"), str):
            result.append({
                "role": key[: -len("_config")], "config_key": key, "model_type": value["model_type"],
                "matches": component_matches(value["model_type"], manifests, registries),
            })
    return result


def scope_files(info, subfolder):
    """The repository's file summary, or one subfolder's with names relative to it (GGUF names stay full)."""
    if not subfolder:
        return summarize_files(info)
    prefix = subfolder + "/"
    siblings = [
        dict(sibling, rfilename=sibling["rfilename"][len(prefix):])
        for sibling in (info.get("siblings") if isinstance(info.get("siblings"), list) else [])
        if isinstance(sibling, dict) and isinstance(sibling.get("rfilename"), str) and sibling["rfilename"].startswith(prefix)
    ]
    files = summarize_files({"siblings": siblings})
    for item in files["summary"]["gguf"]:
        item["name"] = prefix + item["name"]
    return files


def _source_url(source):
    url = "https://huggingface.co/{0}".format(source["repo"])
    if source.get("subfolder"):
        url += "/tree/{0}/{1}".format(source["revision"], source["subfolder"])
    return url


def _empty_payload(text, source):
    return {
        "schema": INTAKE_SCHEMA,
        "source": {
            "input": text.strip(), "repo": source["repo"], "revision": source["revision"],
            "file": source["file"], "subfolder": source.get("subfolder"), "url": _source_url(source),
        },
        "verdict": "unknown", "reasons": [], "backend": None, "backend_installed": False,
        "model_type": None, "components": [], "task": None, "custom_code": False, "gated": False,
        "library_name": None, "pipeline_tag": None, "transformers_version": None, "bytes": 0,
        "download_bytes": 0, "estimated_output_bytes": None, "q_bits": list(ESTIMATE_BITS), "recipe": None,
        "files": {"safetensors": 0, "gguf": [], "python": []}, "warnings": [],
    }


def fetch_pipeline_class(client, repo, revision, warnings, prefix=""):
    """``_class_name`` of a diffusers ``model_index.json``, or None."""
    try:
        value = json.loads(client.fetch_raw_text(repo, revision, prefix + "model_index.json"))
    except TRANSPORT_ERRORS as error:
        warnings.append("model_index.json unreadable: {0}".format(error))
        return None
    return _text(value.get("_class_name")) if isinstance(value, dict) else None


def recipe_download_bytes(client, recipe, files, warnings):
    """The base snapshot (less ignored formats) plus the recipe's LoRA: what fetch downloads for a recipe."""
    lora = files["sizes"].get(recipe["lora"], 0)
    try:
        info = client.fetch_model_info(recipe["base"], revision=recipe["base_revision"])
    except (HuggingFaceHTTPError,) + TRANSPORT_ERRORS as error:
        warnings.append("{0} size unavailable: {1}".format(recipe["base"], error))
        return lora
    base = summarize_files(info if isinstance(info, dict) else {})
    return lora + download_bytes(base, None, {})


def download_bytes(files, model_type, manifests):
    """What ``intake fetch`` of the snapshot takes: a ported type's files, else every file but ignored formats."""
    ported = snapshot_files(model_type, manifests)
    if ported:
        return sum(files["sizes"].get(name, 0) for name in ported)
    return sum(
        size for name, size in files["sizes"].items()
        if not any(fnmatch.fnmatch(name, pattern) for pattern in FETCH_IGNORE_PATTERNS)
    )


def _verdict(payload, tags, files, config, manifests, registries):
    if payload["gated"]:
        return "blocked", ["gated"], None
    if payload["library_name"] == "mlx" or "mlx" in tags:
        return "already_mlx", [], None
    if files["summary"]["gguf"] and not files["summary"]["safetensors"]:
        return "gguf", [], None
    if not payload["model_type"]:
        return "unsupported", ["no_config"], None
    hits = lookup(payload["model_type"], manifests, registries)
    backend = choose_backend(hits, payload["task"]["type"], "vision_config" in config)
    if backend:
        installed = registries.get(backend, {}).get("installed", False)
        return ("convertible" if installed else "convertible_after_install"), [], backend
    reasons = ["arch_not_in_registry"]
    if payload["custom_code"]:
        reasons.append("custom_code")
    return "unsupported", reasons, None


def resolve(text, revision=None, client=None, manifests=None, registries=None):
    source = parse_hf_source(text)
    if revision:
        source["revision"] = validate_revision(revision)
    client = client or HuggingFaceClient()
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests) if registries is None else registries
    payload = _empty_payload(text, source)
    try:
        info = client.fetch_model_info(source["repo"], revision=source["revision"])
    except HuggingFaceHTTPError as error:
        if error.status in _BLOCKING_STATUSES:
            payload.update(verdict="blocked", reasons=["not_found_or_private"])
        else:
            payload.update(verdict="unknown", reasons=["hub_unreachable"])
            payload["warnings"].append(str(error))
        return payload
    except TRANSPORT_ERRORS as error:
        payload.update(verdict="unknown", reasons=["hub_unreachable"])
        payload["warnings"].append(str(error))
        return payload
    if not isinstance(info, dict):
        payload.update(verdict="unknown", reasons=["hub_unreachable"])
        payload["warnings"].append("the model API returned a non-object document")
        return payload

    subfolder = source.get("subfolder")
    prefix = subfolder + "/" if subfolder else ""
    files = scope_files(info, subfolder)
    tags = [tag for tag in info.get("tags") or [] if isinstance(tag, str)]
    # The model card's config describes the repository root, not a subfolder checkpoint.
    api_config = info.get("config") if isinstance(info.get("config"), dict) and not subfolder else {}
    payload.update(
        files=files["summary"], bytes=files["bytes"], gated=bool(info.get("gated")),
        library_name=_text(info.get("library_name")), pipeline_tag=_text(info.get("pipeline_tag")),
    )
    if subfolder and not files["names"]:
        payload.update(verdict="unsupported", reasons=["subfolder_not_found"])
        return payload
    config = {}
    if "config.json" in files["names"] and not payload["gated"]:
        config = fetch_config(client, source["repo"], source["revision"], payload["warnings"], prefix)
    recipe = None if subfolder or payload["gated"] else recipe_for(source["repo"], manifests)
    if recipe is not None and recipe["lora"] not in files["names"]:
        payload["warnings"].append("{0} no longer ships {1}; its recipe does not apply".format(source["repo"], recipe["lora"]))
        recipe = None
    pipeline = None
    if recipe is None and "model_index.json" in files["names"] and not payload["gated"]:
        pipeline = port_for_pipeline(
            fetch_pipeline_class(client, source["repo"], source["revision"], payload["warnings"], prefix), manifests,
        )
    payload["model_type"] = (
        (recipe["port"] if recipe else None) or pipeline
        or _text(config.get("model_type")) or _text(api_config.get("model_type"))
        or (None if payload["gated"] else port_for_files(files["names"], manifests))
    )
    if recipe is not None:
        payload["recipe"] = {key: recipe[key] for key in ("base", "base_revision", "lora", "lora_scale")}
    payload["transformers_version"] = _text(config.get("transformers_version"))
    payload["custom_code"] = (
        "custom_code" in tags or bool(config.get("auto_map")) or bool(api_config.get("auto_map"))
    )
    payload["download_bytes"] = (
        recipe_download_bytes(client, recipe, files, payload["warnings"]) if recipe
        else download_bytes(files, payload["model_type"], manifests)
    )
    payload["components"] = components(payload["model_type"], config, manifests, registries)
    payload["task"] = classify(
        source["repo"], tags=tags, pipeline_tag=payload["pipeline_tag"],
        model_type=payload["model_type"], config_keys=tuple(config),
        manifests=manifests, registries=registries, local=False,
    )
    verdict, reasons, backend = _verdict(payload, tags, files, config, manifests, registries)
    payload.update(
        verdict=verdict, reasons=reasons, backend=backend,
        backend_installed=bool(backend and registries.get(backend, {}).get("installed")),
    )
    if verdict in ("convertible", "convertible_after_install"):
        payload["q_bits"] = port_bits(payload["model_type"], backend, manifests) or list(ESTIMATE_BITS)
    # A recipe's weights come from its base repository and a pipeline's from component folders,
    # whose quantization the header estimate does not model: no estimate rather than a wrong one.
    if verdict in ("convertible", "convertible_after_install") and not recipe and not pipeline:
        payload["estimated_output_bytes"] = estimate_output_bytes(
            client, source["repo"], source["revision"], files,
            quantize_rule(payload["model_type"], backend, manifests), payload["warnings"],
            support_files=port_files(payload["model_type"], backend, manifests), prefix=prefix,
        )
    return payload
