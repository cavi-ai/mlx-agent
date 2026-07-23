"""Deterministic, transparent scoring of a model against a DomainIntent.

Pure and side-effect free. Signals that do not apply to the intent are excluded
from both the numerator and denominator so a candidate is never penalized for a
constraint the user did not request. No network access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

from .interview import DomainIntent


SIGNAL_WEIGHTS = {
    "role_match": 30.0,
    "keyword_match": 25.0,
    "popularity": 15.0,
    "license_ok": 10.0,
    "memory_fit": 10.0,
    "card_quality": 10.0,
}

_SIGNAL_SOURCES = {
    "role_match": "local_role_derivation",
    "keyword_match": "card_text_and_tags",
    "popularity": "huggingface_model_list",
    "license_ok": "huggingface_model_metadata",
    "memory_fit": "local_memory_estimate",
    "card_quality": "card_text",
}

_POPULARITY_LOG_CEILING = 6.0  # ~1e6 downloads saturates the signal.
_MEMORY_HEADROOM_FRACTION = 0.8


@dataclass(frozen=True)
class Signal:
    id: str
    weight: float
    applicable: bool
    matched: Optional[bool]
    contribution: float
    source: str
    detail: str

    def to_dict(self):
        return {
            "id": self.id,
            "weight": self.weight,
            "applicable": self.applicable,
            "matched": self.matched,
            "contribution": round(self.contribution, 3),
            "source": self.source,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ScoreResult:
    score: float
    signals: Tuple[Signal, ...]
    provenance: Tuple[dict, ...]

    def to_dict(self):
        return {
            "score": self.score,
            "signals": [signal.to_dict() for signal in self.signals],
            "provenance": [dict(record) for record in self.provenance],
        }


def _signal(signal_id, applicable, matched, fraction, detail):
    weight = SIGNAL_WEIGHTS[signal_id]
    contribution = weight * fraction if applicable else 0.0
    return Signal(
        id=signal_id,
        weight=weight,
        applicable=applicable,
        matched=matched if applicable else None,
        contribution=contribution,
        source=_SIGNAL_SOURCES[signal_id],
        detail=detail,
    )


def _role_signal(intent: DomainIntent, metadata: Mapping[str, object]) -> Signal:
    model_roles = set(metadata.get("roles") or [])
    matched = [role for role in intent.roles if role in model_roles]
    fraction = len(matched) / len(intent.roles) if intent.roles else 0.0
    return _signal(
        "role_match", True, bool(matched), fraction,
        "matched roles: {0}".format(", ".join(matched) or "none"),
    )


def _keyword_signal(
    intent: DomainIntent, metadata: Mapping[str, object], card_text: Optional[str]
) -> Signal:
    if not intent.keywords:
        return _signal("keyword_match", False, None, 0.0, "no keywords requested")
    tags = " ".join(str(tag).lower() for tag in (metadata.get("tags") or []))
    haystack = "{0} {1} {2}".format((card_text or "").lower(), tags, intent.domain.lower())
    matched = [keyword for keyword in intent.keywords if keyword in haystack]
    fraction = len(matched) / len(intent.keywords)
    return _signal(
        "keyword_match", True, bool(matched), fraction,
        "matched keywords: {0}".format(", ".join(matched) or "none"),
    )


def _popularity_signal(metadata: Mapping[str, object]) -> Signal:
    downloads = metadata.get("downloads") or 0
    try:
        downloads = int(downloads)
    except (TypeError, ValueError):
        downloads = 0
    fraction = min(1.0, math.log10(downloads + 1) / _POPULARITY_LOG_CEILING)
    return _signal(
        "popularity", True, downloads > 0, fraction,
        "downloads: {0}".format(downloads),
    )


def _license_signal(intent: DomainIntent, metadata: Mapping[str, object]) -> Signal:
    if not intent.license_allow:
        return _signal("license_ok", False, None, 0.0, "no license restriction")
    license_name = (metadata.get("license") or "").lower()
    matched = license_name in set(intent.license_allow)
    return _signal(
        "license_ok", True, matched, 1.0 if matched else 0.0,
        "license: {0}".format(license_name or "unknown"),
    )


def _memory_signal(intent: DomainIntent, metadata: Mapping[str, object]) -> Signal:
    est_ram = metadata.get("est_ram_gb")
    if intent.memory_gb is None or est_ram is None:
        return _signal("memory_fit", False, None, 0.0, "no budget or estimate")
    fits = float(est_ram) < intent.memory_gb * _MEMORY_HEADROOM_FRACTION
    return _signal(
        "memory_fit", True, fits, 1.0 if fits else 0.0,
        "est_ram_gb {0} vs budget {1}".format(est_ram, intent.memory_gb),
    )


def _card_quality_signal(card_text: Optional[str]) -> Signal:
    if not card_text:
        return _signal("card_quality", True, False, 0.0, "no model card text")
    quality = 0.5
    lowered = card_text.lower()
    if "```" in card_text or "usage" in lowered or "example" in lowered or "how to" in lowered:
        quality += 0.5
    return _signal(
        "card_quality", True, True, quality,
        "card present; usage/example: {0}".format(quality > 0.5),
    )


def score_candidate(
    intent: DomainIntent,
    metadata: Mapping[str, object],
    card_text: Optional[str],
) -> ScoreResult:
    signals = (
        _role_signal(intent, metadata),
        _keyword_signal(intent, metadata, card_text),
        _popularity_signal(metadata),
        _license_signal(intent, metadata),
        _memory_signal(intent, metadata),
        _card_quality_signal(card_text),
    )
    applicable = [signal for signal in signals if signal.applicable]
    denominator = sum(signal.weight for signal in applicable)
    numerator = sum(signal.contribution for signal in applicable)
    score = round(100.0 * numerator / denominator, 1) if denominator else 0.0
    provenance = tuple(
        {
            "source": signal.source,
            "fields": [signal.id],
            "basis": signal.detail,
        }
        for signal in applicable
    )
    return ScoreResult(score=score, signals=signals, provenance=provenance)


def rank_scored(
    entries: Sequence[Tuple[str, ScoreResult]]
) -> List[Tuple[str, ScoreResult]]:
    """Sort (repo, ScoreResult) entries by score desc, then repo asc."""
    return sorted(entries, key=lambda entry: (-entry[1].score, entry[0]))
