"""Conversion queue calls the canonical agent CLI without a second scheduler."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

CLI_RELATIVE = Path("scripts/mlx-agent")
DEFAULT_TIMEOUT = 300
MAX_LOG_BYTES = 64 * 1024
_receipts_directory = None

def run(agent_path, argv, timeout=DEFAULT_TIMEOUT, runner=None):
    arguments = list(argv) + ["--json"]
    if runner is not None:
        result = runner([sys.executable, str(Path(agent_path) / "scripts/mlx-agent"), *arguments], timeout=timeout)
        try:
            return json.loads(result["stdout"])
        except (KeyError, ValueError, TypeError) as error:
            raise BridgeError("queue_execution_invalid", str(error), "Inspect the agent operation.") from error
    if _receipts_directory is not None:
        arguments += ["--receipts-dir", str(_receipts_directory)]
    from mlx_agent.cli import main
    output = io.StringIO()
    with redirect_stdout(output):
        main(arguments)
    return json.loads(output.getvalue())

class BridgeError(RuntimeError):
    """A classified failure to reach or run the agent CLI."""

    def __init__(self, code, message, remediation):
        super().__init__(message)
        self.code = code
        self.remediation = remediation

    def to_dict(self):
        return {
            "code": self.code,
            "message": str(self),
            "remediation": self.remediation,
        }

def unwrap(envelope):
    """Return the envelope's data, or raise its classified error."""
    if envelope.get("status") == "ok":
        return envelope.get("data") or {}
    error = envelope.get("error") or {}
    raise BridgeError(
        error.get("code", "skill_failed"),
        error.get("message", "The agent reported an error."),
        error.get("remediation", "Inspect the agent output and retry."),
    )

def start(agent_path, gguf_path, preview_hash, q_bits=4, out=None,
          timeout=DEFAULT_TIMEOUT, runner=None):
    """Start a reviewed GGUF conversion. The hash must come from a preview."""
    argv = [
        "convert", "start", "--gguf", gguf_path, "--q-bits", str(q_bits),
        "--confirm", "--preview-hash", preview_hash,
    ]
    if out:
        argv.extend(["--out", out])
    return unwrap(run(agent_path, argv, timeout=timeout, runner=runner))

def start_repo(agent_path, repo, preview_hash, q_bits=4, out=None, hf_cache=None,
               timeout=DEFAULT_TIMEOUT, runner=None):
    """Start a reviewed HF-cache conversion. The hash must come from a preview."""
    argv = [
        "convert", "start", "--repo", repo, "--q-bits", str(q_bits),
        "--confirm", "--preview-hash", preview_hash,
    ]
    if out:
        argv.extend(["--out", out])
    if hf_cache:
        argv.extend(["--hf-cache", hf_cache])
    return unwrap(run(agent_path, argv, timeout=timeout, runner=runner))

def jobs(agent_path, timeout=DEFAULT_TIMEOUT, runner=None):
    """Cross-check conversion receipts against live processes."""
    return unwrap(run(agent_path, ["convert", "status"], timeout=timeout, runner=runner))

def read_convert_receipts(entries, max_bytes=MAX_LOG_BYTES):
    """Read bounded receipt JSON from paths advertised by convert status."""
    receipts = []
    for entry in entries or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("receipt"), str):
            continue
        advertised = Path(entry["receipt"]).expanduser()
        try:
            location = advertised.resolve(strict=True)
        except FileNotFoundError as error:
            raise BridgeError(
                "receipt_missing",
                "A conversion receipt no longer exists: {0}".format(advertised),
                "Run convert status again or inspect the mlx-agent receipt directory.",
            ) from error
        except OSError as error:
            raise BridgeError(
                "receipt_unreadable",
                "A conversion receipt could not be resolved: {0}".format(error),
                "Check the receipt path and file permissions.",
            ) from error
        if not location.is_file():
            raise BridgeError(
                "receipt_missing",
                "A conversion receipt is not a regular file: {0}".format(location),
                "Run convert status again or inspect the mlx-agent receipt directory.",
            )
        try:
            size = location.stat().st_size
        except OSError as error:
            raise BridgeError(
                "receipt_unreadable",
                "A conversion receipt could not be inspected: {0}".format(error),
                "Check the receipt file permissions.",
            ) from error
        if size > max_bytes:
            raise BridgeError(
                "receipt_too_large",
                "A conversion receipt exceeds the {0}-byte read limit.".format(max_bytes),
                "Inspect or remove the malformed receipt in the mlx-agent state directory.",
            )
        try:
            receipt = json.loads(location.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise BridgeError(
                "receipt_unreadable",
                "A conversion receipt is not readable JSON: {0}".format(error),
                "Inspect or remove the malformed receipt in the mlx-agent state directory.",
            ) from error
        if not isinstance(receipt, dict):
            raise BridgeError(
                "receipt_unreadable",
                "A conversion receipt is not a JSON object.",
                "Inspect or remove the malformed receipt in the mlx-agent state directory.",
            )
        receipts.append(receipt)
    return receipts
