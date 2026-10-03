"""Foundational modality layers for research intent seeding.

Three foundations (audio, video, document-vision) seed existing discovery roles
and keywords. No new discovery roles or runtimes. Detection is deterministic
lexicon matching only — no LLM classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FacetProfile:
    id: str
    label: str
    lexicon: Tuple[str, ...]
    seed_keywords: Tuple[str, ...]


@dataclass(frozen=True)
class ModalityProfile:
    id: str
    label: str
    lexicon: Tuple[str, ...]
    default_roles: Tuple[str, ...]
    seed_keywords: Tuple[str, ...]
    facets: Tuple[FacetProfile, ...]


MODALITY_PROFILES: Tuple[ModalityProfile, ...] = (
    ModalityProfile(
        id="audio",
        label="Audio (ASR / TTS / music)",
        lexicon=(
            "audio", "speech", "whisper", "asr", "tts", "voice", "music",
            "sound", "transcri",
        ),
        default_roles=("general",),
        seed_keywords=("audio", "speech", "whisper"),
        facets=(
            FacetProfile(
                id="asr",
                label="ASR / speech-to-text",
                lexicon=("asr", "speech-to-text", "speech to text", "transcri", "whisper", "stt"),
                seed_keywords=("asr", "whisper", "transcription", "speech-to-text"),
            ),
            FacetProfile(
                id="tts",
                label="TTS / text-to-speech",
                lexicon=("tts", "text-to-speech", "text to speech", "narrat", "voice synthes"),
                seed_keywords=("tts", "text-to-speech", "speech synthesis"),
            ),
            FacetProfile(
                id="music",
                label="Music",
                lexicon=("music", "melody", "song", "soundtrack", "audio generation"),
                seed_keywords=("music", "melody", "song"),
            ),
        ),
    ),
    ModalityProfile(
        id="video",
        label="Video",
        lexicon=("video", "footage", "clip", "mp4", "frames"),
        default_roles=("vision", "general"),
        seed_keywords=("video",),
        facets=(
            FacetProfile(
                id="understanding",
                label="Understanding / captioning",
                lexicon=("understanding", "caption", "describe video", "video-llm", "vlm video"),
                seed_keywords=("video", "caption", "understanding"),
            ),
            FacetProfile(
                id="generation",
                label="Generation",
                lexicon=("generation", "generate video", "diffusion", "text-to-video", "text to video"),
                seed_keywords=("video", "generation", "diffusion", "text-to-video"),
            ),
            FacetProfile(
                id="action",
                label="Action / tracking",
                lexicon=("action recognition", "action", "tracking", "pose", "motion"),
                seed_keywords=("video", "action", "tracking"),
            ),
        ),
    ),
    ModalityProfile(
        id="document-vision",
        label="Document extraction / vision",
        lexicon=(
            "ocr", "document", "pdf", "scanned", "invoice", "layout",
            "handwriting", "vision", "vlm", "image understanding",
        ),
        default_roles=("vision",),
        seed_keywords=("ocr", "document", "vision"),
        facets=(
            FacetProfile(
                id="ocr",
                label="OCR / text extraction",
                lexicon=("ocr", "optical character", "scanned", "handwriting", "text extraction"),
                seed_keywords=("ocr", "handwriting", "scanned"),
            ),
            FacetProfile(
                id="layout",
                label="Layout / structure",
                lexicon=("layout", "form parsing", "table extraction", "document structure"),
                seed_keywords=("layout", "document", "pdf"),
            ),
            FacetProfile(
                id="general-vision",
                label="General vision / VLM",
                lexicon=("general vision", "vlm", "image caption", "visual qa", "vision-language"),
                seed_keywords=("vision", "vlm", "image"),
            ),
        ),
    ),
)

FOUNDATION_IDS: Tuple[str, ...] = tuple(profile.id for profile in MODALITY_PROFILES)
ALL_FACET_IDS: Tuple[str, ...] = tuple(
    facet.id for profile in MODALITY_PROFILES for facet in profile.facets
)
MODALITY_CHOICES: Dict[str, str] = {profile.label: profile.id for profile in MODALITY_PROFILES}
FACET_CHOICES: Dict[str, Dict[str, str]] = {
    profile.id: {facet.label: facet.id for facet in profile.facets}
    for profile in MODALITY_PROFILES
}

_PROFILES_BY_ID: Mapping[str, ModalityProfile] = {profile.id: profile for profile in MODALITY_PROFILES}
_FACETS_BY_ID: Mapping[str, FacetProfile] = {
    facet.id: facet for profile in MODALITY_PROFILES for facet in profile.facets
}
_FACET_TO_MODALITY: Mapping[str, str] = {
    facet.id: profile.id for profile in MODALITY_PROFILES for facet in profile.facets
}


def _dedupe(values: Iterable[str]) -> Tuple[str, ...]:
    seen = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


def _haystack(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _lexicon_hit(haystack: str, lexicon: Sequence[str]) -> bool:
    return any(token in haystack for token in lexicon)


def detect_modalities(text: str) -> Tuple[str, ...]:
    """Return foundation ids matched by deterministic lexicon hits."""
    haystack = _haystack(text)
    if not haystack:
        return ()
    hits = []
    for profile in MODALITY_PROFILES:
        if _lexicon_hit(haystack, profile.lexicon):
            hits.append(profile.id)
    return tuple(hits)


def detect_facets(modality: str, text: str) -> Tuple[str, ...]:
    """Return facet ids for one foundation matched in text."""
    profile = _PROFILES_BY_ID.get(modality)
    if profile is None:
        raise ValueError("unknown modality: {0}".format(modality))
    haystack = _haystack(text)
    if not haystack:
        return ()
    hits = []
    for facet in profile.facets:
        if _lexicon_hit(haystack, facet.lexicon):
            hits.append(facet.id)
    return tuple(hits)


def validate_modalities(modalities: Sequence[str]) -> Tuple[str, ...]:
    resolved = _dedupe([str(item).strip() for item in modalities if str(item).strip()])
    for modality in resolved:
        if modality not in _PROFILES_BY_ID:
            raise ValueError("unknown modality: {0}".format(modality))
    return resolved


def validate_facets(facets: Sequence[str], modalities: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    resolved = _dedupe([str(item).strip() for item in facets if str(item).strip()])
    allowed = None
    if modalities is not None:
        allowed = set()
        for modality in validate_modalities(modalities):
            allowed.update(facet.id for facet in _PROFILES_BY_ID[modality].facets)
    for facet in resolved:
        if facet not in _FACETS_BY_ID:
            raise ValueError("unknown facet: {0}".format(facet))
        if allowed is not None and facet not in allowed:
            raise ValueError(
                "facet {0} is not valid for modalities {1}".format(
                    facet, ",".join(modalities) or "none"
                )
            )
    return resolved


def resolve_modalities(cli: Sequence[str] = (), text: str = "") -> Tuple[str, ...]:
    """CLI modalities win; otherwise detect from text."""
    if cli:
        return validate_modalities(cli)
    return detect_modalities(text)


def resolve_facets(
    modalities: Sequence[str],
    cli: Sequence[str] = (),
    text: str = "",
) -> Tuple[str, ...]:
    """Resolve facets: CLI wins; else detect; else seed all facets for each modality."""
    modalities = validate_modalities(modalities)
    if not modalities:
        return ()
    if cli:
        return validate_facets(cli, modalities=modalities)
    detected = []
    for modality in modalities:
        detected.extend(detect_facets(modality, text))
    detected = _dedupe(detected)
    if detected:
        return validate_facets(detected, modalities=modalities)
    # Lock: seed all facet ids for selected foundations when none detected.
    all_facets = []
    for modality in modalities:
        all_facets.extend(facet.id for facet in _PROFILES_BY_ID[modality].facets)
    return tuple(all_facets)


def apply_modality_profile(
    roles: Sequence[str],
    keywords: Sequence[str],
    modalities: Sequence[str],
    facets: Sequence[str] = (),
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Union user roles/keywords with profile seeds. User values are preserved."""
    modalities = validate_modalities(modalities)
    facets = validate_facets(facets, modalities=modalities) if facets else ()
    seeded_roles = list(roles)
    seeded_keywords = list(keywords)
    for modality in modalities:
        profile = _PROFILES_BY_ID[modality]
        for role in profile.default_roles:
            if role not in seeded_roles:
                seeded_roles.append(role)
        for keyword in profile.seed_keywords:
            if keyword not in seeded_keywords:
                seeded_keywords.append(keyword)
    facet_ids = facets
    if modalities and not facet_ids:
        facet_ids = resolve_facets(modalities, cli=(), text="")
    for facet_id in facet_ids:
        facet = _FACETS_BY_ID[facet_id]
        for keyword in facet.seed_keywords:
            if keyword not in seeded_keywords:
                seeded_keywords.append(keyword)
    return tuple(seeded_roles), tuple(seeded_keywords)


def modality_label(modality_id: str) -> str:
    profile = _PROFILES_BY_ID.get(modality_id)
    if profile is None:
        raise ValueError("unknown modality: {0}".format(modality_id))
    return profile.label


def facet_label(facet_id: str) -> str:
    facet = _FACETS_BY_ID.get(facet_id)
    if facet is None:
        raise ValueError("unknown facet: {0}".format(facet_id))
    return facet.label
