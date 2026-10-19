"""Decode-check a generated PNG in the selected image backend environment."""

from __future__ import annotations

import sys
from pathlib import Path


def valid_png(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                return False
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except (ImportError, OSError, ValueError, SyntaxError):
        return False


if __name__ == "__main__":
    raise SystemExit(0 if len(sys.argv) == 2 and valid_png(Path(sys.argv[1])) else 1)
