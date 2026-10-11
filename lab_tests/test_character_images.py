from pathlib import Path

from PIL import Image

from video_lab.character_images import (
    decode_character_image,
    hamming_distance,
    is_near_duplicate,
)


def write_oriented_jpeg(path: Path, *, orientation: int) -> Path:
    image = Image.new("RGB", (64, 48), color=(64, 128, 192))
    exif = Image.Exif()
    exif[274] = orientation
    image.save(path, format="JPEG", exif=exif)
    return path


def test_decode_normalizes_exif_orientation_and_hashes_pixels(tmp_path: Path) -> None:
    source = write_oriented_jpeg(tmp_path / "profile.jpg", orientation=6)

    decoded = decode_character_image(source)

    assert (decoded.width, decoded.height) == (48, 64)
    assert decoded.media_type == "image/jpeg"
    assert len(decoded.dhash_hex) == 16


def test_decode_asymmetric_exif_oriented_jpeg_uses_transposed_pixels_and_pinned_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asymmetric.jpg"
    image = Image.new("RGB", (64, 48))
    for y in range(48):
        for x in range(64):
            image.putpixel(
                (x, y), ((17 * x + 3 * y) % 256, (5 * x + 19 * y) % 256, (11 * x + 7 * y) % 256)
            )
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, format="JPEG", quality=100, subsampling=0, exif=exif)

    decoded = decode_character_image(source)

    assert (decoded.width, decoded.height) == (48, 64)
    assert decoded.dhash_hex == "29a5b4d6524a2bad"


def test_decode_png_and_webp_preserve_media_type_and_pinned_hash(tmp_path: Path) -> None:
    image = Image.new("RGB", (64, 48))
    for y in range(48):
        for x in range(64):
            image.putpixel(
                (x, y), ((17 * x + 3 * y) % 256, (5 * x + 19 * y) % 256, (11 * x + 7 * y) % 256)
            )
    png = tmp_path / "profile.png"
    webp = tmp_path / "profile.webp"
    image.save(png, format="PNG")
    image.save(webp, format="WEBP", lossless=True, method=6)

    assert decode_character_image(png).media_type == "image/png"
    assert decode_character_image(png).dhash_hex == "16754555d73ca32e"
    assert decode_character_image(webp).media_type == "image/webp"
    assert decode_character_image(webp).dhash_hex == "16754555d73ca32e"


def test_near_duplicate_threshold_is_inclusive() -> None:
    assert hamming_distance(0, 0b11111) == 5
    assert is_near_duplicate(0, 0b11111, threshold=5)
    assert not is_near_duplicate(0, 0b111111, threshold=5)
