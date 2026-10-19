"""Decode-check a generated MP4 in the selected video backend environment."""

from __future__ import annotations

import math
import sys
from pathlib import Path


def valid_mp4(path: Path) -> bool:
    capture = None
    try:
        import cv2

        if path.suffix.lower() != ".mp4" or not path.is_file():
            return False
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return False
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (not all(math.isfinite(value) for value in (fps, frame_count, width, height))
                or fps <= 0 or frame_count <= 0 or not frame_count.is_integer()
                or width <= 0 or height <= 0
                or frame_count / fps <= 0):
            return False
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or getattr(frame, "size", 0) <= 0:
                return False
            decoded += 1
        return decoded == int(frame_count)
    except Exception:
        return False
    finally:
        if capture is not None:
            capture.release()


if __name__ == "__main__":
    raise SystemExit(0 if len(sys.argv) == 2 and valid_mp4(Path(sys.argv[1])) else 1)
