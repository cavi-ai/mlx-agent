#!/usr/bin/env python3
"""Backward-compatible wrapper for the extracted mlx-scout core."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlx_agent.cli import legacy_scout_main


if __name__ == "__main__":
    sys.exit(legacy_scout_main())
