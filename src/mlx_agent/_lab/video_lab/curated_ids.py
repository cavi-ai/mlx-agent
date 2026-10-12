"""Canonical identities for curated-sequence contracts."""

from __future__ import annotations

from hashlib import sha256

from .curated_models import CuratedDatasetManifestV1, StructuredCaptionV1
from .domain import canonical_json


def structured_caption_digest(caption: StructuredCaptionV1) -> str:
    return sha256(canonical_json(caption).encode("utf-8")).hexdigest()


def curated_dataset_digest(manifest: CuratedDatasetManifestV1) -> str:
    return sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
