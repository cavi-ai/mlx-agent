"""Internal process entry points for the engine-owned inference lifecycle."""

import argparse
from pathlib import Path

from .task_runtime import execute, supervise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("supervisor", "execute"), required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return (supervise if args.role == "supervisor" else execute)(args.id, args.state_dir)


if __name__ == "__main__":
    raise SystemExit(main())
