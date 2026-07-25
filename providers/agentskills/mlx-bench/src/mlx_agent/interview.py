"""Domain-agnostic interview engine producing a validated DomainIntent.

The question set is a deterministic, config-driven backbone. An optional
``assist`` callable may refine answers, but every assist result is re-validated
through ``build_intent`` and can never bypass validation or inject unknown
roles. Modality foundations seed roles/keywords via ``modality`` profiles.
No network access, no LLM dependency for the deterministic path.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .modality import (
    FACET_CHOICES,
    MODALITY_CHOICES,
    apply_modality_profile,
    detect_facets,
    detect_modalities,
    resolve_facets,
    validate_facets,
    validate_modalities,
)
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

_FOLLOW_ON_QUESTIONS = tuple(question for question in QUESTIONS if question["id"] != "domain")


@dataclass(frozen=True)
class DomainIntent:
    """A validated, deterministic description of what the user wants."""

    domain: str
    roles: Tuple[str, ...]
    keywords: Tuple[str, ...] = ()
    license_allow: Tuple[str, ...] = ()
    memory_gb: Optional[float] = None
    notes: str = ""
    modalities: Tuple[str, ...] = ()
    facets: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "domain": self.domain,
            "roles": list(self.roles),
            "keywords": list(self.keywords),
            "license_allow": list(self.license_allow),
            "memory_gb": self.memory_gb,
            "notes": self.notes,
            "modalities": list(self.modalities),
            "facets": list(self.facets),
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


def _split_labels(raw: object) -> Tuple[str, ...]:
    if raw is None or raw == "":
        return ()
    if isinstance(raw, (list, tuple)):
        return _dedupe([str(item).strip() for item in raw if str(item).strip()])
    return _dedupe([part.strip() for part in str(raw).split(",") if part.strip()])


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
    return _dedupe(roles)


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


def _resolve_modality_ids(raw: object) -> Tuple[str, ...]:
    labels = _split_labels(raw)
    ids = []
    for label in labels:
        if label in MODALITY_CHOICES:
            ids.append(MODALITY_CHOICES[label])
        else:
            ids.append(label)
    return validate_modalities(ids)


def _resolve_facet_ids(raw: object, modalities: Sequence[str]) -> Tuple[str, ...]:
    labels = _split_labels(raw)
    if not labels:
        return ()
    label_to_id = {}
    for modality in modalities:
        label_to_id.update(FACET_CHOICES.get(modality, {}))
    ids = []
    for label in labels:
        if label in label_to_id:
            ids.append(label_to_id[label])
        else:
            ids.append(label)
    return validate_facets(ids, modalities=modalities)


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

    modalities_raw = answers.get("modalities")
    if modalities_raw is None or modalities_raw == "":
        modalities = detect_modalities(domain)
        keyword_text = " ".join(_split_csv(answers.get("keywords")))
        if keyword_text:
            modalities = _dedupe(list(modalities) + list(detect_modalities(keyword_text)))
    else:
        modalities = _resolve_modality_ids(modalities_raw)

    facets_raw = answers.get("facets")
    detect_text = "{0} {1}".format(domain, " ".join(_split_csv(answers.get("keywords"))))
    if facets_raw is None or facets_raw == "":
        facets = resolve_facets(modalities, cli=(), text=detect_text) if modalities else ()
    else:
        facets = _resolve_facet_ids(facets_raw, modalities)

    roles = _resolve_roles(answers.get("roles"))
    keywords = _split_csv(answers.get("keywords"))
    if modalities:
        roles, keywords = apply_modality_profile(roles, keywords, modalities, facets)
    if not roles:
        roles = ("general",)

    intent = DomainIntent(
        domain=domain,
        roles=roles,
        keywords=keywords,
        license_allow=_split_csv(
            answers["license"] if "license" in answers else answers.get("license_allow")
        ),
        memory_gb=_resolve_memory(answers.get("memory_gb")),
        notes=str(answers.get("notes", "")).strip(),
        modalities=modalities,
        facets=facets,
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
    preset_modalities: Sequence[str] = (),
    preset_facets: Sequence[str] = (),
) -> DomainIntent:
    """Ask questions via ``reader`` with detect-then-ask modality resolution."""
    answers: Dict[str, object] = {}
    domain_question = QUESTIONS[0]
    answers["domain"] = reader(domain_question)
    domain = str(answers["domain"]).strip()

    modalities = validate_modalities(preset_modalities) if preset_modalities else detect_modalities(domain)
    if not modalities:
        modality_question = {
            "id": "modalities",
            "prompt": (
                "Which foundational modalities apply? "
                "(audio, video, document extraction / vision)"
            ),
            "kind": "multi",
            "choices": tuple(MODALITY_CHOICES.keys()),
        }
        modalities = _resolve_modality_ids(reader(modality_question))
        if not modalities:
            raise ValueError("at least one foundational modality is required")
    answers["modalities"] = list(modalities)

    facets: List[str] = list(validate_facets(preset_facets, modalities=modalities)) if preset_facets else []
    if not facets:
        for modality in modalities:
            detected = detect_facets(modality, domain)
            if detected:
                facets.extend(detected)
            else:
                facet_question = {
                    "id": "facets:{0}".format(modality),
                    "prompt": "Which {0} facets matter? (optional multi-select)".format(modality),
                    "kind": "multi",
                    "choices": tuple(FACET_CHOICES[modality].keys()),
                }
                response = reader(facet_question)
                if response is None or response == "":
                    continue
                facets.extend(_resolve_facet_ids(response, (modality,)))
        if not facets and modalities:
            facets = list(resolve_facets(modalities, cli=(), text=domain))
    answers["facets"] = list(_dedupe(facets))

    for question in _FOLLOW_ON_QUESTIONS:
        answers[str(question["id"])] = reader(question)
    return build_intent(answers, assist=assist)
