"""Deterministic task type and use-case labels for models (no model calls)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .backends import (
    BackendError, is_vision_type, load_manifests, load_registries, lookup, lookup_squashed, squash_type,
)
from .gguf import DRAFT_ARCHITECTURES, MAX_CONFIG_BYTES
from .modality import detect_facets, detect_modalities

TASK_TYPES = (
    "text_llm", "vision_language", "speech_to_text", "text_to_speech",
    "embedding", "classification", "image_generation", "video_generation", "speculative_draft", "other",
)
# Ordered specific-first; the last entry is the type's default use case.
USE_CASES = {
    "text_llm": ("coding", "reasoning", "general_chat"),
    "vision_language": ("ocr_documents", "video", "vision"),
    "speech_to_text": ("realtime_transcription", "diarization", "speech_translation", "transcription"),
    "text_to_speech": ("voice_cloning", "narration"),
    "embedding": ("reranking", "retrieval"),
    "classification": ("moderation", "routing", "classification"),
    "image_generation": ("image_generation",),
    "video_generation": ("video_generation",),
    "speculative_draft": ("speculative_decoding",),
    "other": (),
}
PIPELINE_TYPES = {
    "text-generation": "text_llm",
    "text2text-generation": "text_llm",
    "conversational": "text_llm",
    "image-text-to-text": "vision_language",
    "visual-question-answering": "vision_language",
    "image-to-text": "vision_language",
    "video-text-to-text": "vision_language",
    "document-question-answering": "vision_language",
    "any-to-any": "vision_language",
    "automatic-speech-recognition": "speech_to_text",
    "text-to-speech": "text_to_speech",
    "text-to-audio": "text_to_speech",
    "feature-extraction": "embedding",
    "sentence-similarity": "embedding",
    "text-ranking": "embedding",
    "text-classification": "classification",
    "zero-shot-classification": "classification",
    "text-to-image": "image_generation",
    "image-to-image": "image_generation",
    "text-to-video": "video_generation",
}
_SPECIFIC_TOKENS = (
    ("coding", ("code", "coder", "coding", "starcoder", "codestral", "devstral")),
    ("reasoning", ("reasoning", "reasoner", "thinking", "r1", "qwq")),
    ("ocr_documents", ("ocr", "document", "docvqa")),
    ("video", ("video",)),
    ("realtime_transcription", ("realtime", "real-time", "streaming")),
    ("diarization", ("diariz",)),
    ("speech_translation", ("translat",)),
    ("voice_cloning", ("clone", "cloning", "zero-shot")),
    ("reranking", ("rerank",)),
    ("moderation", ("moderation", "guardrail", "toxic")),
    ("routing", ("routing", "router")),
)
# llama.cpp multimodal projectors (mmproj files) are model parts, not models.
_GGUF_PROJECTOR_ARCHITECTURES = ("clip", "mmproj")
# A converted DSpark drafter's config.json (mlx-agent's port, mlx-community's sidecars).
_DRAFT_CONFIG_KEYS = ("dspark_target_layer_ids",)
_VISION_KEYS = ("vision_config", "vision_tower", "mm_vision_tower", "visual", "vision_encoder")
_EMBEDDING_NAME = re.compile(r"(?<![a-z0-9])(embed|embedding|embeddings|bge|e5|gte)(?![a-z])")
_VISION_NAME = re.compile(r"(?<![a-z0-9])(vl|vlm|vision|llava)(?![a-z])")
_IMAGE_NAME = re.compile(r"(?<![a-z0-9])(sdxl|flux|stable-diffusion)(?![a-z])")


def _has_token(haystack, token):
    if len(token) <= 3:
        pattern = r"(?<![a-z0-9]){0}(?![a-z0-9])".format(re.escape(token))
    else:
        pattern = r"(?<![a-z0-9]){0}".format(re.escape(token))
    return re.search(pattern, haystack) is not None


def use_cases_for(task_type, haystack):
    ordered = USE_CASES.get(task_type, ())
    if not ordered:
        return []
    haystack = haystack.lower()
    specific = {
        use_case for use_case, tokens in _SPECIFIC_TOKENS
        if use_case in ordered[:-1] and any(_has_token(haystack, token) for token in tokens)
    }
    return [use_case for use_case in ordered[:-1] if use_case in specific] + [ordered[-1]]


def _type_from_hits(hits, config_keys, haystack, vision_module=False, allow_vision=True):
    preferred = [hit for hit in hits if hit["match"] != "remap"] or hits
    categories = {hit["category"] for hit in preferred}
    if "text_llm" in categories and categories & {"speech_to_text", "text_to_speech"}:
        named = _type_from_name(haystack)
        return named if named in ("speech_to_text", "text_to_speech") else "text_llm"
    if "speech_to_text" in categories:
        return "speech_to_text"
    if "text_to_speech" in categories:
        return "text_to_speech"
    if "classification" in categories:
        return "classification"
    if "image_generation" in categories:
        return "image_generation"
    if "video_generation" in categories:
        return "video_generation"
    if allow_vision and "vision_language" in categories and (
        any(key in config_keys for key in _VISION_KEYS)
        or (vision_module and "text_llm" not in categories)
    ):
        return "vision_language"
    if categories & {"text_llm", "vision_language"}:
        return "text_llm"
    return None


def _type_from_name(haystack):
    if _EMBEDDING_NAME.search(haystack):
        return "embedding"
    modalities = detect_modalities(haystack)
    if "audio" in modalities:
        facets = detect_facets("audio", haystack)
        if "tts" in facets:
            return "text_to_speech"
        if "asr" in facets:
            return "speech_to_text"
    if "video" in modalities or "document-vision" in modalities or _VISION_NAME.search(haystack):
        return "vision_language"
    if _IMAGE_NAME.search(haystack):
        return "image_generation"
    return None


def _task_type(haystack, pipeline_tag, model_type, config_keys, gguf_architecture, manifests, registries, local):
    # A drafter is tagged like its target (text-generation) but cannot run alone.
    if gguf_architecture and squash_type(gguf_architecture) in DRAFT_ARCHITECTURES:
        return "speculative_draft", "gguf_architecture", "confirmed"
    if any(key in config_keys for key in _DRAFT_CONFIG_KEYS):
        return "speculative_draft", "config", "confirmed"
    if pipeline_tag in PIPELINE_TYPES:
        return PIPELINE_TYPES[pipeline_tag], "pipeline_tag", "confirmed"
    if model_type and manifests:
        found = _type_from_hits(
            lookup(model_type, manifests, registries), config_keys, haystack,
            vision_module=is_vision_type(model_type, manifests, registries),
        )
        if found:
            return found, "registry", "confirmed"
    if gguf_architecture and squash_type(gguf_architecture) in _GGUF_PROJECTOR_ARCHITECTURES:
        return "other", "gguf_architecture", "likely"
    if gguf_architecture and manifests:
        # GGUF main weights never carry a vision tower (projectors ship separately).
        found = _type_from_hits(
            lookup_squashed(gguf_architecture, manifests, registries), config_keys, haystack,
            allow_vision=False,
        )
        if found:
            return found, "gguf_architecture", "likely"
    found = _type_from_name(haystack)
    if found:
        return found, "name", "likely"
    return ("text_llm" if local else "other"), "default", "likely"


def classify(name, tags=(), pipeline_tag=None, model_type=None, config_keys=(),
             gguf_architecture=None, manifests=None, registries=None, local=False):
    haystack = " ".join([str(name or "")] + [str(tag) for tag in tags or ()]).lower()
    manifests = manifests or {}
    registries = registries or {}
    task_type, source, confidence = _task_type(
        haystack, pipeline_tag, model_type, tuple(config_keys), gguf_architecture,
        manifests, registries, local,
    )
    return {
        "type": task_type,
        "use_cases": use_cases_for(task_type, haystack),
        "source": source,
        "confidence": confidence,
    }


def read_config_type(directory):
    location = Path(str(directory)) / "config.json"
    try:
        if location.stat().st_size > MAX_CONFIG_BYTES:
            return None, ()
        value = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None, ()
    if not isinstance(value, dict):
        return None, ()
    model_type = value.get("model_type") if isinstance(value.get("model_type"), str) else None
    return model_type, tuple(sorted(key for key in value if isinstance(key, str)))


def annotate_inventory(report, manifests=None, registries=None):
    """Add a "task" object to every scanned GGUF model and MLX output."""
    if manifests is None:
        try:
            manifests = load_manifests()
        except (BackendError, OSError, ValueError):
            manifests = {}
    if registries is None:
        registries = load_registries(manifests)
    for item in report.get("models", []):
        item["task"] = classify(
            item.get("name"), gguf_architecture=item.get("architecture"),
            manifests=manifests, registries=registries, local=True,
        )
    for output in report.get("outputs", []):
        model_type, keys = read_config_type(output.get("path", ""))
        output["task"] = classify(
            output.get("name"), model_type=model_type, config_keys=keys,
            manifests=manifests, registries=registries, local=True,
        )
    return report
