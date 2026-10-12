from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from PIL import Image, ImageOps

DUPLICATE_HASH_ALGORITHM_VERSION: Final[Literal["dhash64-exif-transpose-lanczos.v1"]] = (
    "dhash64-exif-transpose-lanczos.v1"
)


@dataclass(frozen=True)
class DecodedCharacterImage:
    width: int
    height: int
    media_type: Literal["image/jpeg", "image/png", "image/webp"]
    dhash_hex: str


def decode_character_image(path: Path) -> DecodedCharacterImage:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
        return DecodedCharacterImage(
            width=image.width,
            height=image.height,
            media_type=_media_type(opened.format),
            dhash_hex=f"{difference_hash(image):016x}",
        )


def difference_hash(image: Image.Image) -> int:
    luminance = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    hash_value = 0
    for row in range(8):
        for column in range(8):
            left = cast(int, luminance.getpixel((column, row)))
            right = cast(int, luminance.getpixel((column + 1, row)))
            hash_value = (hash_value << 1) | int(left > right)
    return hash_value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def is_near_duplicate(left: int, right: int, *, threshold: int = 5) -> bool:
    return hamming_distance(left, right) <= threshold


def _media_type(image_format: str | None) -> Literal["image/jpeg", "image/png", "image/webp"]:
    match image_format:
        case "JPEG":
            return "image/jpeg"
        case "PNG":
            return "image/png"
        case "WEBP":
            return "image/webp"
        case _:
            raise ValueError(f"unsupported character image format: {image_format!r}")
