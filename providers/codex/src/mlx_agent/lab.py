"""Optional experiment engine; the core CLI never imports its dependencies."""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
import sys

ENGINE_ROOT = Path(__file__).resolve().parent / "_lab"
MAX_OUTPUT_BYTES = 8 * 1024 * 1024

def state_root():
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/mlx-agent/lab"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "mlx-agent/lab"

def _error(code, message, remediation="Install requirements-lab.txt in the selected Python environment."):
    print(json.dumps({"schema": "lab/1", "status": "error",
                      "error": {"code": code, "message": message,
                                "remediation": remediation}}))
    return 2

def _operation(arguments):
    value_options = {"--database", "--artifacts", "--reviewer-id"}
    iterator = iter(arguments)
    for token in iterator:
        if token in value_options:
            next(iterator, None)
        elif any(token.startswith(option + "=") for option in value_options):
            continue
        elif not token.startswith("-"):
            return token
    return None


def _run_child(command, env, timeout=86400, pass_fds=()):
    """Bound both output streams while the engine is running."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, pass_fds=pass_fds)
    output = {process.stdout: bytearray(), process.stderr: bytearray()}
    selector = selectors.DefaultSelector()
    for stream in output:
        selector.register(stream, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    try:
        while selector.get_map():
            if time.monotonic() >= deadline:
                raise TimeoutError("Experiment operation timed out.")
            for key, _ in selector.select(timeout=min(1, max(0, deadline - time.monotonic()))):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                output[key.fileobj].extend(chunk)
                if sum(len(value) for value in output.values()) > MAX_OUTPUT_BYTES:
                    raise OverflowError("Experiment output exceeded the supported size.")
        return subprocess.CompletedProcess(command, process.wait(timeout=max(0.1, deadline - time.monotonic())),
            output[process.stdout].decode("utf-8", errors="replace"),
            output[process.stderr].decode("utf-8", errors="replace"))
    finally:
        selector.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in output:
            stream.close()


def run(argv, legacy=False):
    arguments = list(argv)
    if not arguments:
        arguments = ["--help"]
    if "--database" not in arguments:
        arguments = ["--database", str(state_root() / "experiments.sqlite3")] + arguments
    if "--artifacts" not in arguments:
        arguments = ["--artifacts", str(state_root() / "artifacts")] + arguments
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ENGINE_ROOT), str(Path(__file__).resolve().parents[1])))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        module = {
            "workspace": "mlx_agent.lab_workspace",
            "queue": "mlx_agent.queue_cli",
            "training-data": "video_lab.training_data_cli",
        }.get(_operation(arguments), "video_lab.cli")
        from .compute import ComputeLease, ComputeBusy
        operation = _operation(arguments)
        heavy = (operation == "worker" and any(item in arguments for item in ("run-once", "recover"))) or (operation == "adapter" and "probe" in arguments) or (operation == "adapter" and "feasibility" in arguments and "resume" in arguments)
        with ComputeLease("lab-" + str(operation)) if heavy else nullcontext() as lease:
            result = _run_child([sys.executable, "-m", module, *arguments], env,
                                pass_fds=(lease.descriptor,) if lease is not None else ())
    except ComputeBusy as error:
        return _error("compute_busy", str(error), "Stop the owning model operation before retrying.")
    except OverflowError as error:
        return _error("lab_output_limit", str(error))
    except (TimeoutError, subprocess.TimeoutExpired):
        return _error("lab_timeout", "Experiment operation timed out.")
    except OSError:
        return _error("lab_unavailable", "The experiment runtime could not start.")
    if legacy or "--json" not in arguments:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode
    if len(result.stdout.encode()) > MAX_OUTPUT_BYTES:
        return _error("lab_output_limit", "Experiment output exceeded the supported size.")
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        return _error("lab_unavailable", "The experiment runtime returned no structured result.")
    if not isinstance(payload, dict):
        return _error("lab_invalid_output", "The experiment runtime returned an invalid result.")
    success = payload.get("ok") is True or payload.get("status") == "ok"
    data = payload.get("data")
    if isinstance(data, list):
        data = {"experiments": data} if "experiment" in arguments else {"items": data}
    envelope = {"schema": "lab/1", "engine_schema": payload.get("schema_version"),
                "status": "ok" if success and result.returncode == 0 else "error",
                "data": data}
    if envelope["status"] == "error":
        envelope["error"] = payload.get("error") or {"code": "lab_failed", "message": "Experiment operation failed."}
    print(json.dumps(envelope))
    return result.returncode if envelope["status"] == "ok" else (result.returncode or 1)


def worker_main(argv=None):
    from .compute import ComputeLease, ComputeBusy
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ENGINE_ROOT), str(Path(__file__).resolve().parents[1])))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        with ComputeLease("lab-worker") if "--help" not in arguments else nullcontext() as lease:
            result = _run_child([sys.executable, "-m", "video_lab.worker", *arguments], env,
                                pass_fds=(lease.descriptor,) if lease is not None else ())
    except ComputeBusy as error:
        return _error("compute_busy", str(error), "Stop the owning model operation before retrying.")
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode
