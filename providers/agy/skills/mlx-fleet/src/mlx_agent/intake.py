"""Resolve a pasted Hugging Face model into a conversion verdict (read-only)."""

from __future__ import annotations

import http.client
import json

from .backends import choose_backend, component_matches, load_manifests, load_registries, lookup
from .huggingface import HuggingFaceClient, HuggingFaceHTTPError
from .intake_source import parse_hf_source, validate_revision
from .taxonomy import classify

INTAKE_SCHEMA = "intake/1"
TRANSPORT_ERRORS = (OSError, TimeoutError, http.client.HTTPException, ValueError)
_BLOCKING_STATUSES = (401, 403, 404, 410)


def _text(value):
    return value if isinstance(value, str) else None


def summarize_files(info):
    siblings = info.get("siblings") if isinstance(info.get("siblings"), list) else []
    names, gguf, python = set(), [], []
    total, safetensors = 0, 0
    for sibling in siblings:
        if not isinstance(sibling, dict) or not isinstance(sibling.get("rfilename"), str):
            continue
        name = sibling["rfilename"]
        size = sibling.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            size = 0
        names.add(name)
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
        "bytes": total,
        "summary": {
            "safetensors": safetensors,
            "gguf": sorted(gguf, key=lambda item: item["name"]),
            "python": sorted(python),
        },
    }


def fetch_config(client, repo, revision, warnings):
    try:
        value = json.loads(client.fetch_raw_text(repo, revision, "config.json"))
    except TRANSPORT_ERRORS as error:
        warnings.append("config.json unreadable: {0}".format(error))
        return {}
    if not isinstance(value, dict):
        warnings.append("config.json is not a JSON object")
        return {}
    return value


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


def _empty_payload(text, source):
    return {
        "schema": INTAKE_SCHEMA,
        "source": {
            "input": text.strip(), "repo": source["repo"], "revision": source["revision"],
            "file": source["file"], "url": "https://huggingface.co/{0}".format(source["repo"]),
        },
        "verdict": "unknown", "reasons": [], "backend": None, "backend_installed": False,
        "model_type": None, "components": [], "task": None, "custom_code": False, "gated": False,
        "library_name": None, "pipeline_tag": None, "transformers_version": None, "bytes": 0,
        "files": {"safetensors": 0, "gguf": [], "python": []}, "warnings": [],
    }


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

    files = summarize_files(info)
    tags = [tag for tag in info.get("tags") or [] if isinstance(tag, str)]
    api_config = info.get("config") if isinstance(info.get("config"), dict) else {}
    payload.update(
        files=files["summary"], bytes=files["bytes"], gated=bool(info.get("gated")),
        library_name=_text(info.get("library_name")), pipeline_tag=_text(info.get("pipeline_tag")),
    )
    config = {}
    if "config.json" in files["names"] and not payload["gated"]:
        config = fetch_config(client, source["repo"], source["revision"], payload["warnings"])
    payload["model_type"] = _text(config.get("model_type")) or _text(api_config.get("model_type"))
    payload["transformers_version"] = _text(config.get("transformers_version"))
    payload["custom_code"] = (
        "custom_code" in tags or bool(config.get("auto_map")) or bool(api_config.get("auto_map"))
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
    return payload
