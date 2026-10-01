"""Deterministic task type and use-case labels for models (no model calls)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .backends import BackendError, load_manifests, load_registries, lookup, lookup_squashed
from .gguf import MAX_CONFIG_BYTES
from .modality import detect_facets, detect_modalities

TASK_TYPES = (
    "text_llm", "vision_language", "speech_to_text", "text_to_speech",
    "embedding", "image_generation", "other",
)
# Ordered specific-first; the last entry is the type's default use case.
USE_CASES = {
    "text_llm": ("coding", "reasoning", "general_chat"),
    "vision_language": ("ocr_documents", "video", "vision"),
    "speech_to_text": ("realtime_transcription", "diarization", "speech_translation", "transcription"),
    "text_to_speech": ("voice_cloning", "narration"),
    "embedding": ("reranking", "retrieval"),
    "image_generation": ("image_generation",),
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
    "text-to-image": "image_generation",
    "image-to-image": "image_generation",
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
)
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


def _type_from_hits(hits, config_keys):
    preferred = [hit for hit in hits if hit["match"] != "remap"] or hits
    categories = {hit["category"] for hit in preferred}
    if "speech_to_text" in categories:
        return "speech_to_text"
    if "text_to_speech" in categories:
        return "text_to_speech"
    if "vision_language" in categories and ("vision_config" in config_keys or "text_llm" not in categories):
        return "vision_language"
    if "text_llm" in categories:
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
    if pipeline_tag in PIPELINE_TYPES:
        return PIPELINE_TYPES[pipeline_tag], "pipeline_tag", "confirmed"
    if model_type and manifests:
        found = _type_from_hits(lookup(model_type, manifests, registries), config_keys)
        if found:
            return found, "registry", "confirmed"
    if gguf_architecture and manifests:
        found = _type_from_hits(lookup_squashed(gguf_architecture, manifests, registries), config_keys)
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
