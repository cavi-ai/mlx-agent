"""Local model status: where each model lives, how it arrived, its task type and duplicate groups.

Built from one read-only :func:`mlx_agent.gguf.inventory` pass annotated by
:mod:`mlx_agent.taxonomy`. Discovery matches Hugging Face repositories against it.
"""

from __future__ import annotations

from pathlib import Path

from . import gguf
from .taxonomy import annotate_inventory

# Storage item kinds that are whole models; standalone assets are files, not models.
MODEL_KINDS = ("gguf", "folder", "ollama")
# Model parts that share a base model's name but are not a copy of it.
_PART_TYPES = ("vision_projector", "speculative_draft")
_WEIGHT_SUFFIXES = (".safetensors", ".gguf")


def scan(gguf_roots, mlx_roots=(), receipts=(), signature=True, limit=gguf.MAX_SCAN_FILES,
         compare_assets=False, asset_exclude_roots=()):
    """The inventory report with task types, ``families`` and ``local_models``."""
    report = annotate_inventory(gguf.inventory(
        gguf_roots, mlx_roots=mlx_roots, receipts=receipts, signature=signature, limit=limit,
        compare_assets=compare_assets, asset_exclude_roots=asset_exclude_roots,
    ))
    task_types = {item["path"]: item["task"].get("type") for item in report["models"] + report["outputs"] + report["ollama"]
                  if isinstance(item.get("task"), dict)}
    report["families"] = gguf.families(report["storage"]["items"], task_types)
    report["local_models"] = local_models(report)
    return report


def _hf_cache_repo(parts):
    for index in range(1, len(parts) - 1):
        if parts[index] == "snapshots" and parts[index - 1].startswith("models--"):
            return parts[index - 1][len("models--"):].replace("--", "/", 1)
    return None


def _lmstudio_repo(location, kind, roots):
    """LM Studio keeps ``<root>/<publisher>/<repository>``: a folder at that depth, or a GGUF file inside one."""
    for root in roots:
        if location.is_relative_to(root):
            parts = location.relative_to(root).parts
            if (kind == "folder" and len(parts) == 2) or (kind == "gguf" and len(parts) == 3):
                return "/".join(parts[:2]), True
            return None, True
    return None, False


def local_models(report):
    """One record per local model in an annotated inventory report."""
    lmstudio = gguf.lmstudio_roots()
    sources = {**{item["path"]: item for item in report.get("models", [])},
               **{item["path"]: item for item in report.get("outputs", [])},
               **{item["path"]: item for item in report.get("ollama", [])}}
    converted = {path for item in report.get("models", []) for path in item.get("outputs") or []}
    groups = {}
    for group in report.get("duplicates", []):
        for path in [group.get("keep")] + list(group.get("redundant") or []) + list(group.get("members") or []):
            if path and group["group_id"] not in groups.get(path, []):
                groups.setdefault(path, []).append(group["group_id"])
    family = {path: entry["family_id"] for entry in report.get("families", []) for path in entry["members"]}
    records = []
    for item in report["storage"]["items"]:
        if item["kind"] not in MODEL_KINDS:
            continue
        source = sources.get(item["path"], {})
        location = Path(item["path"])
        repo = _hf_cache_repo(location.parts)
        store = item["store"]
        if store == "folder":
            lmstudio_repo, in_lmstudio = _lmstudio_repo(location, item["kind"], lmstudio)
            store = "lmstudio" if in_lmstudio else store
            repo = repo or lmstudio_repo
        provenance = source.get("provenance") or {}
        origin = provenance.get("source") or {}
        if provenance or item["path"] in converted:
            acquired = "converted"
        elif store == "huggingface":
            acquired = "cached"
        else:
            acquired = "downloaded"
        if item["kind"] == "folder":
            quantization = gguf._mlx_quantization(source) if source else None
        else:
            quantization = source.get("quantization")
        weights = [stored["bytes"] for stored in item["files"] if stored["path"].endswith(_WEIGHT_SUFFIXES)]
        records.append({
            "path": item["path"],
            "name": item["name"],
            "repo": repo,
            "store": store,
            "format": item["format"],
            "acquired": acquired,
            "converted_from": origin.get("repo") or origin.get("path"),
            "model_key": item["model_key"],
            "quantization": quantization,
            "task": source.get("task"),
            "duplicate_groups": groups.get(item["path"], []),
            "family_id": family.get(item["path"]),
            "bytes": item["bytes"],
            "weight_bytes": sum(weights) if weights and item["kind"] != "ollama" else None,
        })
    return records


def match(records, repo, weight_bytes=None):
    """How a Hugging Face repository is present locally.

    ``exact``: a copy of the same repository, or a same-named model whose weight
    files total the same bytes. ``variant``: another quantization or format of the
    same model name. ``absent``: neither.
    """
    key = gguf.name_key(repo)
    wanted = repo.lower()
    same_repo, same_key = [], []
    for record in records:
        if ((record.get("task") or {}).get("type")) in _PART_TYPES:
            continue
        if record["repo"] and record["repo"].lower() == wanted:
            same_repo.append(record)
        elif len(key) >= gguf._MIN_NAME_MATCH and record["model_key"] == key:
            same_key.append(record)
    same_weights = [record for record in same_key
                    if weight_bytes and record["weight_bytes"] == weight_bytes]
    if same_repo:
        status, basis = "exact", "repository"
    elif same_weights:
        status, basis = "exact", "weight_bytes"
    elif same_key:
        status, basis = "variant", "model_key"
    else:
        status, basis = "absent", None
    return {
        "status": status,
        "basis": basis,
        "model_key": key,
        "copies": [
            {field: record[field] for field in ("path", "repo", "store", "format", "quantization", "acquired")}
            for record in same_repo + same_key
        ],
    }
