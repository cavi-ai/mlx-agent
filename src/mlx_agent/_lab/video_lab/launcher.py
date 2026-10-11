from __future__ import annotations

_import = __import__


def main() -> None:
    """Load the CLI without importing optional inference dependencies."""
    cli = _import("video_lab.cli", fromlist=("app",))
    cli.app()


if __name__ == "__main__":
    main()
