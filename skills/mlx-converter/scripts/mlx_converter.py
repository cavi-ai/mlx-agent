#!/usr/bin/env python3
"""Stable entry point for the mlx-converter skill; delegates to the core."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlx_agent.cli import converter_main


if __name__ == "__main__":
    sys.exit(converter_main())
