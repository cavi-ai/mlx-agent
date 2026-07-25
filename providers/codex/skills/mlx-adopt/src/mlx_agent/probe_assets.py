"""Deterministic synthetic assets for role-fit verification probes."""

from __future__ import annotations

import base64
import struct
import zlib


VISION_PROBE_TEXT = "MLX42"

_FONT_5X7 = {
    "M": (
        "X...X",
        "XX.XX",
        "X.X.X",
        "X.X.X",
        "X...X",
        "X...X",
        "X...X",
    ),
    "L": (
        "X....",
        "X....",
        "X....",
        "X....",
        "X....",
        "X....",
        "XXXXX",
    ),
    "X": (
        "X...X",
        "X...X",
        ".X.X.",
        "..X..",
        ".X.X.",
        "X...X",
        "X...X",
    ),
    "4": (
        "...X.",
        "..XX.",
        ".X.X.",
        "X..X.",
        "XXXXX",
        "...X.",
        "...X.",
    ),
    "2": (
        ".XXX.",
        "X...X",
        "....X",
        "...X.",
        "..X..",
        ".X...",
        "XXXXX",
    ),
}

_cached_image_b64 = None


def _png_chunk(tag, data):
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _png_gray(width, height, rows):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def _render_text_bitmap(text, scale=6, padding=8):
    glyphs = [_FONT_5X7[character] for character in text]
    glyph_width = 5
    spacing = 1
    bitmap_width = len(glyphs) * (glyph_width + spacing) - spacing
    bitmap_height = 7
    width = bitmap_width * scale + 2 * padding
    height = bitmap_height * scale + 2 * padding
    rows = [bytearray(width) for _ in range(height)]
    for index, glyph in enumerate(glyphs):
        origin_x = padding + index * (glyph_width + spacing) * scale
        for gy, line in enumerate(glyph):
            for gx, mark in enumerate(line):
                if mark != "X":
                    continue
                for dy in range(scale):
                    row = rows[padding + gy * scale + dy]
                    start = origin_x + gx * scale
                    for dx in range(scale):
                        row[start + dx] = 255
    return width, height, rows


def vision_probe_image_png():
    """Render the fixed probe word as a small deterministic grayscale PNG."""
    width, height, rows = _render_text_bitmap(VISION_PROBE_TEXT)
    return _png_gray(width, height, rows)


def vision_probe_image_base64():
    global _cached_image_b64
    if _cached_image_b64 is None:
        _cached_image_b64 = base64.b64encode(vision_probe_image_png()).decode("ascii")
    return _cached_image_b64
