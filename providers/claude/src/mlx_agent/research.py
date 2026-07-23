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
from typing import Any, Dict, List, Optional, Tuple

from .discovery import DiscoveryRequest
from .interview import DomainIntent
from .scoring import ScoreResult, rank_scored, score_candidate


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_CARD_EXCERPT_CHARS = 240


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
    record: Dict[str, Any] = field(default_factory=dict)

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
            "record": dict(self.record),
        }


@dataclass(frozen=True)
class ResearchPack:
    intent: DomainIntent
    candidates: Tuple[ResearchCandidate, ...]
    generated_at: str
    warnings: Tuple[dict, ...] = field(default_factory=tuple)

    def to_dict(self):
        return {
            "intent": self.intent.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "generated_at": self.generated_at,
            "warnings": [dict(warning) for warning in self.warnings],
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


def _excerpt(card_text: Optional[str]) -> str:
    if not card_text:
        return ""
    collapsed = " ".join(card_text.split())
    return collapsed[:_CARD_EXCERPT_CHARS]


def generate_pack(
    intent: DomainIntent,
    discovery_service,
    hf_client,
    limit: int = 6,
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
        record = dict(candidate)
        record["role"] = candidate.get("role") or intent.roles[0]
        candidates.append(ResearchCandidate(
            repo=repo,
            role=record["role"],
            score=result.score,
            wiring=candidate.get("wiring", ""),
            signals=tuple(signal.to_dict() for signal in result.signals),
            provenance=result.provenance,
            card_present=bool(card_text),
            card_excerpt=_excerpt(card_text),
            record=record,
        ))

    return ResearchPack(
        intent=intent,
        candidates=tuple(candidates),
        generated_at=moment.isoformat(),
        warnings=tuple(warnings),
    )


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
    if pack.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in pack.warnings:
            lines.append("- [{0}] {1}".format(warning.get("code", "warning"), warning.get("message", "")))
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
    lines.extend([
        "## Next steps",
        "",
        "1. Review candidates above and pick one to adopt.",
        "2. Install it in a supported local runtime (Ollama, LM Studio, or mlx_lm).",
        "3. Run `mlx-agent adopt start --from-research <this-pack>.json --state <state-path>` "
        "to verify ranked candidates — research does not verify or download.",
        "",
        "## Notes",
        "",
        "This pack is a read-only research artifact intended for ingestion by other agents. "
        "A sibling `.json` sidecar carries the machine-readable intent and candidate records. "
        "All scores are estimates with per-signal provenance; verify capability before relying on any model.",
        "",
    ])
    return "\n".join(lines)


def _assert_pack_path(out_dir: Path, path: Path) -> Path:
    resolved = path.resolve()
    if os.path.commonpath([str(out_dir), str(resolved)]) != str(out_dir):
        raise ValueError("refusing to write research pack outside the project folder")
    return resolved


def write_pack(
    markdown: str,
    intent: DomainIntent,
    root: Optional[object] = None,
    now: Optional[datetime] = None,
    pack: Optional[ResearchPack] = None,
) -> Path:
    """Write the pack under ``<root>/mlx-research`` and never outside it.

    When ``pack`` is provided, also writes a sibling ``.json`` sidecar with the
    structured pack payload for ``adopt start --from-research``.
    """
    moment = now or datetime.now(timezone.utc)
    base = Path(root) if root is not None else Path.cwd()
    out_dir = (base / "mlx-research").resolve()
    unresolved_dir = base / "mlx-research"
    if unresolved_dir.is_symlink():
        raise ValueError("refusing to write research pack through a symlinked folder")
    timestamp = moment.strftime("%Y%m%dT%H%M%SZ")
    stem = "{0}-{1}".format(slugify(intent.domain), timestamp)
    path = _assert_pack_path(out_dir, out_dir / "{0}.md".format(stem))
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    if pack is not None:
        json_path = _assert_pack_path(out_dir, out_dir / "{0}.json".format(stem))
        json_path.write_text(
            json.dumps(pack.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return path


def load_research_pack(path: object) -> Dict[str, Any]:
    """Load and lightly validate a research-pack JSON sidecar."""
    pack_path = Path(path)
    if pack_path.suffix.lower() == ".md":
        raise ValueError(
            "research packs for adopt must be the sibling .json sidecar, not the .md file"
        )
    if pack_path.suffix.lower() != ".json":
        raise ValueError("research pack path must be a .json sidecar")
    try:
        payload = json.loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("research pack could not be read: {0}".format(error)) from error
    if not isinstance(payload, dict):
        raise ValueError("research pack must be a JSON object")
    intent = payload.get("intent")
    candidates = payload.get("candidates")
    if not isinstance(intent, dict):
        raise ValueError("research pack is missing a valid intent object")
    if not isinstance(candidates, list):
        raise ValueError("research pack is missing a candidates array")
    if not str(intent.get("domain") or "").strip():
        raise ValueError("research pack intent.domain is required")
    roles = intent.get("roles")
    if not isinstance(roles, list) or not roles:
        raise ValueError("research pack intent.roles must be a non-empty array")
    return payload
