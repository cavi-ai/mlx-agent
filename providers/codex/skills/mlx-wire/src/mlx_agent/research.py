"""Read-only domain research packs: discovery -> scoring -> project-local markdown.

No verification, no wiring, no downloads. Discovery and the Hugging Face client
are injected so this module is fully testable offline. Output is written only
inside the project-local ``mlx-research`` folder.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .blueprint import DatasetBlueprint, build_dataset_blueprint
from .discovery import DiscoveryRequest
from .interview import DomainIntent
from .modality import FACET_CHOICES, facet_label, modality_label
from .scoring import ScoreResult, rank_scored, score_candidate


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_CARD_EXCERPT_CHARS = 240
_DEFAULT_CATALOG_LIMIT = 5


@dataclass(frozen=True)
class ResearchCandidate:
    repo: str
    role: str
    score: float
    wiring: str
    signals: Tuple[dict, ...]
    provenance: Tuple[dict, ...]
    card_present: bool
    card_excerpt: str

    def to_dict(self):
        return {
            "repo": self.repo,
            "role": self.role,
            "score": self.score,
            "wiring": self.wiring,
            "signals": [dict(signal) for signal in self.signals],
            "provenance": [dict(record) for record in self.provenance],
            "card_present": self.card_present,
            "card_excerpt": self.card_excerpt,
        }


@dataclass(frozen=True)
class CatalogItem:
    repo: str
    kind: str  # adapter | dataset
    score: float
    signals: Tuple[dict, ...]
    provenance: Tuple[dict, ...]
    card_excerpt: str
    record: Dict[str, object]

    def to_dict(self):
        return {
            "repo": self.repo,
            "kind": self.kind,
            "score": self.score,
            "signals": [dict(signal) for signal in self.signals],
            "provenance": [dict(record) for record in self.provenance],
            "card_excerpt": self.card_excerpt,
            "record": dict(self.record),
        }


@dataclass(frozen=True)
class ResearchPack:
    intent: DomainIntent
    candidates: Tuple[ResearchCandidate, ...]
    generated_at: str
    warnings: Tuple[dict, ...] = field(default_factory=tuple)
    adapters: Tuple[CatalogItem, ...] = field(default_factory=tuple)
    datasets: Tuple[CatalogItem, ...] = field(default_factory=tuple)
    dataset_blueprint: Optional[DatasetBlueprint] = None

    def to_dict(self):
        return {
            "intent": self.intent.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "generated_at": self.generated_at,
            "warnings": [dict(warning) for warning in self.warnings],
            "adapters": [item.to_dict() for item in self.adapters],
            "datasets": [item.to_dict() for item in self.datasets],
            "dataset_blueprint": (
                self.dataset_blueprint.to_dict() if self.dataset_blueprint else None
            ),
        }


def slugify(domain: str) -> str:
    slug = _SLUG_RE.sub("-", domain.strip().lower()).strip("-")
    return slug or "domain"


def candidate_metadata(candidate: Dict[str, object]) -> Dict[str, object]:
    """Map a discovery candidate to the scoring core's metadata shape."""
    roles = candidate.get("roles") or ([candidate["role"]] if candidate.get("role") else [])
    return {
        "roles": list(roles),
        "license": candidate.get("license"),
        "downloads": candidate.get("downloads", 0),
        "likes": candidate.get("likes", 0),
        "est_ram_gb": candidate.get("est_ram_gb"),
        "tags": candidate.get("tags", []),
    }


def hub_row_metadata(row: Dict[str, object], intent: DomainIntent) -> Dict[str, object]:
    """Map a Hub adapter/dataset row into the scoring metadata shape."""
    tags = list(row.get("tags") or [])
    license_name = row.get("license")
    if not license_name:
        for tag in tags:
            text = str(tag)
            if text.startswith("license:"):
                license_name = text.split(":", 1)[1]
                break
    return {
        "roles": list(intent.roles),
        "license": license_name,
        "downloads": row.get("downloads", 0),
        "likes": row.get("likes", 0),
        "est_ram_gb": None,
        "tags": tags,
    }


def catalog_search_query(intent: DomainIntent) -> str:
    parts = list(intent.keywords)
    if intent.domain and intent.domain.lower() not in {p.lower() for p in parts}:
        parts.append(intent.domain)
    return " ".join(parts).strip()


def _excerpt(card_text: Optional[str]) -> str:
    if not card_text:
        return ""
    collapsed = " ".join(card_text.split())
    return collapsed[:_CARD_EXCERPT_CHARS]


def _repo_id(row: Dict[str, object]) -> Optional[str]:
    repo = row.get("id") or row.get("modelId") or row.get("repo")
    if not repo:
        return None
    return str(repo)


def _score_catalog_rows(
    intent: DomainIntent,
    rows: List[Dict[str, object]],
    kind: str,
    fetch_card,
    catalog_limit: int,
) -> Tuple[CatalogItem, ...]:
    scored: List[Tuple[str, ScoreResult]] = []
    detail: Dict[str, Dict[str, object]] = {}
    for row in rows:
        repo = _repo_id(row)
        if not repo or repo in detail:
            continue
        try:
            card_text = fetch_card(repo)
        except Exception:
            card_text = None
        result = score_candidate(intent, hub_row_metadata(row, intent), card_text)
        scored.append((repo, result))
        detail[repo] = {"row": row, "card_text": card_text}

    ranked = rank_scored(scored)[:max(0, catalog_limit)]
    items: List[CatalogItem] = []
    for repo, result in ranked:
        row = detail[repo]["row"]
        card_text = detail[repo]["card_text"]
        items.append(CatalogItem(
            repo=repo,
            kind=kind,
            score=result.score,
            signals=tuple(signal.to_dict() for signal in result.signals),
            provenance=result.provenance,
            card_excerpt=_excerpt(card_text),
            record=dict(row),
        ))
    return tuple(items)


def generate_pack(
    intent: DomainIntent,
    discovery_service,
    hf_client,
    limit: int = 6,
    catalog_limit: int = _DEFAULT_CATALOG_LIMIT,
    now: Optional[datetime] = None,
) -> ResearchPack:
    """Build a research pack. Read-only: never verifies, wires, or downloads."""
    moment = now or datetime.now(timezone.utc)
    seen: Dict[str, Dict[str, object]] = {}
    warnings: List[dict] = []
    for role in intent.roles:
        request = DiscoveryRequest(
            role=role,
            memory_gb=intent.memory_gb,
            licenses=list(intent.license_allow) or None,
            limit=limit,
        )
        envelope = discovery_service.discover(request)
        if envelope.status != "ok":
            detail = envelope.error
            warnings.append({
                "code": detail.code if detail else "discovery_failed",
                "message": "{0}: {1}".format(role, detail.message if detail else "discovery failed"),
            })
            continue
        for bucket in envelope.data.get("roles", {}).values():
            for candidate in bucket:
                repo = candidate.get("repo")
                if repo and repo not in seen:
                    seen[repo] = candidate

    scored: List[Tuple[str, ScoreResult]] = []
    detail_by_repo: Dict[str, Dict[str, object]] = {}
    for repo, candidate in seen.items():
        try:
            card_text = hf_client.fetch_model_card(repo)
        except Exception:
            card_text = None
        result = score_candidate(intent, candidate_metadata(candidate), card_text)
        scored.append((repo, result))
        detail_by_repo[repo] = {"candidate": candidate, "card_text": card_text}

    ranked = rank_scored(scored)[:max(0, limit)]
    candidates = []
    for repo, result in ranked:
        candidate = detail_by_repo[repo]["candidate"]
        card_text = detail_by_repo[repo]["card_text"]
        candidates.append(ResearchCandidate(
            repo=repo,
            role=candidate.get("role", ""),
            score=result.score,
            wiring=candidate.get("wiring", ""),
            signals=tuple(signal.to_dict() for signal in result.signals),
            provenance=result.provenance,
            card_present=bool(card_text),
            card_excerpt=_excerpt(card_text),
        ))

    search = catalog_search_query(intent)
    try:
        adapter_rows = list(hf_client.list_adapters(search=search, limit_fetch=max(catalog_limit * 4, 20)))
    except Exception as error:
        warnings.append({
            "code": "adapter_search_failed",
            "message": "adapter search failed: {0}".format(error),
        })
        adapter_rows = []
    try:
        dataset_rows = list(hf_client.list_datasets(search=search, limit_fetch=max(catalog_limit * 4, 20)))
    except Exception as error:
        warnings.append({
            "code": "dataset_search_failed",
            "message": "dataset search failed: {0}".format(error),
        })
        dataset_rows = []

    adapters = _score_catalog_rows(
        intent,
        adapter_rows,
        "adapter",
        hf_client.fetch_model_card,
        catalog_limit,
    )
    datasets = _score_catalog_rows(
        intent,
        dataset_rows,
        "dataset",
        hf_client.fetch_dataset_card,
        catalog_limit,
    )
    blueprint = build_dataset_blueprint(intent) if not datasets else None

    return ResearchPack(
        intent=intent,
        candidates=tuple(candidates),
        generated_at=moment.isoformat(),
        warnings=tuple(warnings),
        adapters=adapters,
        datasets=datasets,
        dataset_blueprint=blueprint,
    )


def _render_catalog_section(title: str, items: Tuple[CatalogItem, ...], empty_message: str) -> List[str]:
    lines = ["## {0}".format(title), ""]
    if not items:
        lines.append(empty_message)
        lines.append("")
        return lines
    for index, item in enumerate(items, start=1):
        lines.append("### {0}. `{1}` — score {2}/100".format(index, item.repo, item.score))
        lines.append("")
        lines.append("- Kind: {0}".format(item.kind))
        matched = [s for s in item.signals if s["applicable"] and s["matched"]]
        if matched:
            lines.append("- Why it ranked:")
            for signal in matched:
                lines.append("  - {0}: {1}".format(signal["id"], signal["detail"]))
        if item.card_excerpt:
            lines.append("- Card excerpt: {0}".format(item.card_excerpt))
        lines.append("")
    return lines


def render_pack(pack: ResearchPack) -> str:
    intent = pack.intent
    lines = [
        "# MLX Research Pack: {0}".format(intent.domain),
        "",
        "- Generated: {0}".format(pack.generated_at),
        "- Roles: {0}".format(", ".join(intent.roles) or "none"),
        "- Keywords: {0}".format(", ".join(intent.keywords) or "none"),
        "- License filter: {0}".format(", ".join(intent.license_allow) or "none"),
        "- Memory budget (GB): {0}".format(intent.memory_gb if intent.memory_gb is not None else "none"),
        "- Modalities: {0}".format(", ".join(intent.modalities) or "none"),
        "- Facets: {0}".format(", ".join(intent.facets) or "none"),
    ]
    if intent.notes:
        lines.append("- Notes: {0}".format(intent.notes))
    lines.extend(["", "## Summary", ""])
    if pack.candidates:
        lines.append(
            "{0} candidate(s) ranked by transparent scoring. Scores are read-only "
            "estimates from metadata and model-card text; no model was downloaded, "
            "verified, or wired.".format(len(pack.candidates))
        )
    else:
        lines.append("No candidates were found for this intent. See warnings below.")
    summary_bits = []
    if pack.adapters:
        summary_bits.append("{0} adapter(s)".format(len(pack.adapters)))
    if pack.datasets:
        summary_bits.append("{0} dataset(s)".format(len(pack.datasets)))
    if pack.dataset_blueprint is not None:
        summary_bits.append("dataset blueprint (no Hub datasets matched)")
    if summary_bits:
        lines.append("Catalog enrichment: {0}.".format("; ".join(summary_bits)))
    if pack.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in pack.warnings:
            lines.append("- [{0}] {1}".format(warning.get("code", "warning"), warning.get("message", "")))
    lines.extend(["", "## Modality foundations", ""])
    if intent.modalities:
        for modality in intent.modalities:
            allowed = set(FACET_CHOICES.get(modality, {}).values())
            facet_ids = [facet for facet in intent.facets if facet in allowed]
            lines.append("- `{0}` — {1}".format(modality, modality_label(modality)))
            if facet_ids:
                lines.append("  - Facets: {0}".format(
                    ", ".join(
                        "`{0}` ({1})".format(facet, facet_label(facet))
                        for facet in facet_ids
                    )
                ))
        lines.append("")
    else:
        lines.append("No foundational modalities were selected for this pack.")
        lines.append("")
    lines.extend(["", "## Candidates", ""])
    for index, candidate in enumerate(pack.candidates, start=1):
        lines.append("### {0}. `{1}` — score {2}/100".format(index, candidate.repo, candidate.score))
        lines.append("")
        lines.append("- Role: {0}".format(candidate.role))
        lines.append("- Suggested wiring: {0}".format(candidate.wiring))
        lines.append("- Model card: {0}".format("present" if candidate.card_present else "not found"))
        matched = [s for s in candidate.signals if s["applicable"] and s["matched"]]
        if matched:
            lines.append("- Why it ranked:")
            for signal in matched:
                lines.append("  - {0}: {1}".format(signal["id"], signal["detail"]))
        if candidate.card_excerpt:
            lines.append("- Card excerpt: {0}".format(candidate.card_excerpt))
        lines.append("")
    lines.extend(_render_catalog_section(
        "Adapters / LoRAs",
        pack.adapters,
        "No PEFT/LoRA adapters matched this intent on the Hub (read-only search).",
    ))
    lines.extend(_render_catalog_section(
        "Datasets",
        pack.datasets,
        "No datasets matched this intent on the Hub. See the dataset blueprint below.",
    ))
    if pack.dataset_blueprint is not None:
        lines.extend([
            "## Dataset blueprint",
            "",
            "No Hub datasets ranked for this intent. Use this guidance-only blueprint "
            "to create a local dataset — nothing is downloaded or trained by research.",
            "",
            pack.dataset_blueprint.to_markdown().rstrip(),
            "",
        ])
    lines.extend([
        "## Next steps",
        "",
        "1. Review candidates, adapters, and datasets above and pick a local stack.",
        "2. Install a base model in a supported local runtime (Ollama, LM Studio, or mlx_lm).",
        "3. Run `mlx-agent adopt start` to verify and wire it — research does not verify or download.",
        "4. If no datasets matched, follow the dataset blueprint to create labeled data locally.",
        "",
        "## Notes",
        "",
        "This pack is a read-only research artifact intended for ingestion by other agents. "
        "All scores are estimates with per-signal provenance; verify capability before relying on any model.",
        "",
    ])
    return "\n".join(lines)


def write_pack(
    markdown: str,
    intent: DomainIntent,
    root: Optional[object] = None,
    now: Optional[datetime] = None,
    pack: Optional[ResearchPack] = None,
) -> Path:
    """Write the pack under ``<root>/mlx-research`` and never outside it."""
    moment = now or datetime.now(timezone.utc)
    base = Path(root) if root is not None else Path.cwd()
    out_dir = (base / "mlx-research").resolve()
    unresolved_dir = base / "mlx-research"
    if unresolved_dir.is_symlink():
        raise ValueError("refusing to write research pack through a symlinked folder")
    timestamp = moment.strftime("%Y%m%dT%H%M%SZ")
    filename = "{0}-{1}.md".format(slugify(intent.domain), timestamp)
    path = (out_dir / filename).resolve()
    if os.path.commonpath([str(out_dir), str(path)]) != str(out_dir):
        raise ValueError("refusing to write research pack outside the project folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    if pack is not None:
        sidecar = path.with_suffix(".json")
        if os.path.commonpath([str(out_dir), str(sidecar.resolve())]) != str(out_dir):
            raise ValueError("refusing to write research pack outside the project folder")
        sidecar.write_text(json.dumps(pack.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
