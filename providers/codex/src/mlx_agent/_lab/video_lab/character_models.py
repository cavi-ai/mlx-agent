from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from video_lab.domain import FrozenModel


class Framing(StrEnum):
    PORTRAIT = "portrait"
    FULL_BODY = "full_body"


class HeadAngle(StrEnum):
    FRONT = "front"
    THREE_QUARTER = "three_quarter"
    PROFILE = "profile"


class BodyVisibility(StrEnum):
    CLEAR_FACE = "clear_face"
    PARTIAL_BODY = "partial_body"
    FULL_BODY = "full_body"


class Expression(StrEnum):
    NEUTRAL = "neutral"
    SMILE = "smile"
    SERIOUS = "serious"
    ANGRY = "angry"
    SURPRISED = "surprised"


class Lighting(StrEnum):
    ORDINARY = "ordinary"
    DIRECTIONAL = "directional"


class Occlusion(StrEnum):
    NONE = "none"
    PARTIAL = "partial"
    HEAVY = "heavy"


class CuratorDisposition(StrEnum):
    ACTIVE = "active"
    REJECTED = "rejected"


class WarningResolution(StrEnum):
    UNRESOLVED = "unresolved"
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"


class CharacterAnnotation(FrozenModel):
    framing: Framing
    head_angle: HeadAngle
    body_visibility: BodyVisibility
    expression: Expression
    lighting: Lighting
    occlusion: Occlusion = Occlusion.NONE
    identity_confirmed: bool = False


class CharacterReferenceInput(FrozenModel):
    reference_id: str = Field(min_length=1)
    image_sha256: str = Field(min_length=1)
    media_type: Literal["image/jpeg", "image/png", "image/webp"]
    dhash_hex: str
    disposition: CuratorDisposition = CuratorDisposition.ACTIVE
    annotation: CharacterAnnotation

    @field_validator("dhash_hex")
    @classmethod
    def validate_dhash_hex(cls, value: str) -> str:
        normalized = value.lower()
        if re.fullmatch(r"[0-9a-f]{16}", normalized) is None:
            raise ValueError("dhash_hex must be exactly 16 lowercase hexadecimal characters")
        return normalized


class DuplicateWarning(FrozenModel):
    left_reference_id: str = Field(min_length=1)
    right_reference_id: str = Field(min_length=1)
    distance: int = Field(ge=0, le=64)
    resolution: WarningResolution = WarningResolution.UNRESOLVED

    @model_validator(mode="after")
    def validate_distinct_references(self) -> DuplicateWarning:
        if self.left_reference_id == self.right_reference_id:
            raise ValueError("duplicate warnings must refer to two distinct references")
        return self


class CharacterPackDraft(FrozenModel):
    character_id: str = Field(min_length=1)
    references: tuple[CharacterReferenceInput, ...]
    duplicate_warnings: tuple[DuplicateWarning, ...] = ()
    supersedes_pack_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_reference_ids(self) -> CharacterPackDraft:
        reference_ids = tuple(reference.reference_id for reference in self.references)
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("reference_id values must be unique")
        return self
