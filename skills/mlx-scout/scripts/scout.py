#!/usr/bin/env python3
"""Backward-compatible wrapper for the extracted mlx-scout core."""

from pathlib import Path
import sys


if sys.version_info < (3, 11):
    sys.exit("mlx-agent requires Python 3.11 or later; {0} is Python {1}.{2}.".format(sys.executable, *sys.version_info[:2]))
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlx_agent.cli import legacy_scout_main


if __name__ == "__main__":
    sys.exit(legacy_scout_main())
