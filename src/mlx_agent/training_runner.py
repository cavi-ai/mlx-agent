"""Record the actual trainer exit without inferring success from a folder."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_agent.atomicio import write_text_atomic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not command:
        parser.error("training command is required")
    try:
        code = subprocess.run(command).returncode
        result = {"schema_version": "training-exit/1", "exit_code": code}
    except OSError as error:
        code = 127
        result = {"schema_version": "training-exit/1", "exit_code": code, "error": str(error)}
    write_text_atomic(arguments.result, json.dumps(result))
    return code

if __name__ == "__main__":
    raise SystemExit(main())
