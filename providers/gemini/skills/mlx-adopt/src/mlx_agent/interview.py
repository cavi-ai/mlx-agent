"""Domain-agnostic interview engine producing a validated DomainIntent.

The question set is a deterministic, config-driven backbone. An optional
``assist`` callable may refine answers, but every assist result is re-validated
through ``build_intent`` and can never bypass validation or inject unknown
roles. No network access, no LLM dependency for the deterministic path.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from .models import DISCOVERY_ROLES


ROLE_CHOICES: Dict[str, str] = {
    "Vision / OCR": "vision",
    "Embeddings": "embedding",
    "Coding": "coding",
    "Reasoning": "reasoning",
    "General chat": "general",
    "Tool use / function calling": "tool-use",
}


QUESTIONS: Tuple[Dict[str, object], ...] = (
    {
        "id": "domain",
        "prompt": "What are you building? Describe the domain in a sentence.",
        "kind": "text",
    },
    {
        "id": "roles",
        "prompt": "Which model roles do you need? (choose one or more)",
        "kind": "multi",
        "choices": tuple(ROLE_CHOICES.keys()),
    },
    {
        "id": "keywords",
        "prompt": "Any domain keywords to prioritize? (comma-separated, optional)",
        "kind": "text",
    },
    {
        "id": "license",
        "prompt": "Restrict to specific licenses? (comma-separated, optional)",
        "kind": "text",
    },
    {
        "id": "memory_gb",
        "prompt": "Memory budget in GB? (optional, e.g. 32)",
        "kind": "text",
    },
    {
        "id": "notes",
        "prompt": "Any other constraints or notes? (optional)",
        "kind": "text",
    },
)


@dataclass(frozen=True)
class DomainIntent:
    """A validated, deterministic description of what the user wants."""

    domain: str
    roles: Tuple[str, ...]
    keywords: Tuple[str, ...] = ()
    license_allow: Tuple[str, ...] = ()
    memory_gb: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "domain": self.domain,
            "roles": list(self.roles),
            "keywords": list(self.keywords),
            "license_allow": list(self.license_allow),
            "memory_gb": self.memory_gb,
            "notes": self.notes,
        }


def _dedupe(values: Sequence[str]) -> Tuple[str, ...]:
    seen = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


def _split_csv(raw: object) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        parts = [str(item) for item in raw]
    else:
        parts = str(raw).split(",")
    return _dedupe([part.strip().lower() for part in parts if part.strip()])


def _resolve_roles(raw: object) -> Tuple[str, ...]:
    if raw is None or raw == "":
        labels: Sequence[str] = ()
    elif isinstance(raw, (list, tuple)):
        labels = [str(item) for item in raw]
    else:
        labels = [part.strip() for part in str(raw).split(",")]
    roles = []
    for label in labels:
        cleaned = label.strip()
        if not cleaned:
            continue
        if cleaned in ROLE_CHOICES:
            roles.append(ROLE_CHOICES[cleaned])
        elif cleaned in DISCOVERY_ROLES:
            roles.append(cleaned)
        else:
            raise ValueError("unknown role: {0}".format(cleaned))
    resolved = _dedupe(roles)
    return resolved or ("general",)


def _resolve_memory(raw: object) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError("memory_gb must be a number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("memory_gb must be a positive number")
    return value


def build_intent(
    answers: Mapping[str, object],
    assist: Optional[Callable[["DomainIntent"], Mapping[str, object]]] = None,
) -> DomainIntent:
    """Validate raw answers into a DomainIntent. Deterministic given answers.

    If ``assist`` is provided it receives the validated intent and may return a
    mapping of overrides; those overrides are merged and the intent is rebuilt
    through this same validator (assist can never bypass validation).
    """
    domain = str(answers.get("domain", "")).strip()
    if not domain:
        raise ValueError("domain is required")
    intent = DomainIntent(
        domain=domain,
        roles=_resolve_roles(answers.get("roles")),
        keywords=_split_csv(answers.get("keywords")),
        license_allow=_split_csv(
            answers["license"] if "license" in answers else answers.get("license_allow")
        ),
        memory_gb=_resolve_memory(answers.get("memory_gb")),
        notes=str(answers.get("notes", "")).strip(),
    )
    if assist is None:
        return intent
    overrides = assist(intent)
    if not overrides:
        return intent
    merged = intent.to_dict()
    merged.update(dict(overrides))
    return build_intent(merged, assist=None)


def run_interview(
    reader: Callable[[Mapping[str, object]], object],
    assist: Optional[Callable[["DomainIntent"], Mapping[str, object]]] = None,
) -> DomainIntent:
    """Ask each question via ``reader`` and build a validated intent."""
    answers: Dict[str, object] = {}
    for question in QUESTIONS:
        answers[str(question["id"])] = reader(question)
    return build_intent(answers, assist=assist)
