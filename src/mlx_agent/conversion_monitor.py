"""Observe one reviewed conversion worker and publish its actual completion."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_agent.source_evidence import completed_fetch, fingerprint, read_json, safe_path
from mlx_agent.transactions import _atomic_in_directory


@contextmanager
def receipt_lock(receipt_path):
    path = safe_path(receipt_path)
    descriptor = os.open(path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield path
    finally:
        os.close(descriptor)


def _matches(receipt, request):
    plan = request["plan"]
    return (receipt.get("launch_id") == request["launch_id"] and receipt.get("pid") == os.getpid()
            and receipt.get("attempt_id") == request.get("attempt_id")
            and receipt.get("require_completion") is True and receipt.get("preview_hash") == plan["preview_hash"]
            and receipt.get("argv") == plan["argv"] and receipt.get("source") == plan["source"]
            and receipt.get("out") == plan["out"] and receipt.get("started_at") == request["started_at"])


def _write(path, receipt):
    _atomic_in_directory(path.parent, path.name, (json.dumps(receipt, sort_keys=True) + "\n").encode(), 0o600)


def _check_sources(plan):
    source = plan["source"]
    if source.get("fetch_receipt"):
        current = completed_fetch(source["fetch_receipt"], source["path"])
        if current["origin"] != source["origin"] or current["fingerprint"] != source["fingerprint"]:
            raise ValueError("The reviewed source changed before conversion.")
    elif source.get("path") and source.get("fingerprint") and source.get("kind") == "hf-cache":
        from mlx_agent.convert import local_source_fingerprint
        if local_source_fingerprint(source["path"]) != source["fingerprint"]:
            raise ValueError("The reviewed local source changed.")
    if source.get("kind") == "gguf" and source.get("signature"):
        from mlx_agent.gguf import file_signature
        if file_signature(source["path"]) != source["signature"]:
            raise ValueError("The reviewed GGUF source changed.")
    for expected in source.get("dependencies", []):
        if completed_fetch(expected["fetch_receipt"], expected["path"], bundle_root=True) != expected:
            raise ValueError("A reviewed conversion dependency changed.")
    projector = source.get("projector")
    if projector:
        from mlx_agent.gguf import file_signature
        if (file_signature(projector["path"]) != projector["signature"]
                or fingerprint(Path(projector["path"]).resolve()) != projector["fingerprint"]):
            raise ValueError("The reviewed projector changed.")


def _owned_members():
    from mlx_agent.compute import group_members
    group = os.getpid()
    if os.getpgrp() != group:
        raise ValueError("The conversion monitor no longer owns its isolated process group.")
    return [pid for pid in group_members(group) if pid != os.getpid()]


def _wait_for_children(timeout=5):
    deadline = time.monotonic() + timeout
    while _owned_members():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _stop_owned_children():
    # The live monitor is the group leader, so this PGID cannot have been reused.
    # Ignore TERM in the monitor long enough to persist the failed outcome.
    if not _owned_members():
        return False
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if os.getpgrp() != os.getpid():
        raise ValueError("Conversion process-group ownership changed.")
    os.killpg(os.getpid(), signal.SIGTERM)
    return not _wait_for_children(timeout=2)


def run(request_path, readiness_timeout=15):
    request = read_json(request_path)
    path = safe_path(request["receipt_path"])
    deadline = time.monotonic() + readiness_timeout
    receipt = None
    while time.monotonic() < deadline:
        try:
            candidate = read_json(path)
            if (_matches(candidate, request) and
                    (not request.get("wait_for_compute_lease") or candidate.get("compute_lease_registered") is True)):
                receipt = candidate
                break
        except (OSError, ValueError):
            pass
        time.sleep(0.02)
    if receipt is None:
        return 2
    plan = request["plan"]
    worker = None
    observed = False
    error = None
    code = None
    kill_group = False
    try:
        _check_sources(plan)
        if Path(plan["out"]).exists() or Path(plan["out"]).is_symlink():
            raise ValueError("The conversion destination became occupied.")
        with receipt_lock(path):
            receipt = read_json(path)
            if not _matches(receipt, request):
                return 2
            from mlx_agent.compute import process_identity
            identity = process_identity(os.getpid())
            if os.getpgrp() != os.getpid() or not identity:
                raise ValueError("The conversion monitor's isolated process group could not be established.")
            receipt["execution_group"] = {"pgid": os.getpid(), "leader_identity": identity}
            worker = subprocess.Popen(plan["argv"], stdin=subprocess.DEVNULL)
            receipt["worker_pid"] = worker.pid
            _write(path, receipt)
        code = worker.wait()
        observed = True
        if not _wait_for_children():
            error = "Conversion descendants did not finish; the owned group was stopped and the output is incomplete."
            kill_group = _stop_owned_children()
        if code == 0 and error is None:
            _check_sources(plan)
        if code != 0:
            error = "The conversion worker exited with status {0}.".format(code)
        elif not safe_path(plan["out"]).is_dir():
            error = "The conversion worker exited without a model output directory."
    except (OSError, ValueError, subprocess.SubprocessError) as failure:
        error = str(failure)
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                code = worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                worker.kill()
                code = worker.wait()
            observed = True
            try:
                if not _wait_for_children():
                    kill_group = _stop_owned_children()
            except (OSError, ValueError, subprocess.SubprocessError):
                pass  # The retained compute group prevents another operation.
    try:
        with receipt_lock(path):
            current = read_json(path)
            if not _matches(current, request):
                return 2  # A later launch owns this receipt; publish nothing into it.
            if observed and code == 0 and error is None:
                try:
                    output = safe_path(plan["out"])
                    evidence = {"schema_version": "1.1", "tool": "mlx-agent.convert",
                        "created_at": datetime.now(timezone.utc).isoformat(), "source": plan["source"],
                        "q_bits": plan["q_bits"], "backend": plan.get("backend", "mlx-lm"),
                        "model_type": plan.get("model_type"), "recipe": plan.get("recipe"),
                        "launch_id": request["launch_id"], "attempt_id": request.get("attempt_id"),
                        "preview_hash": plan["preview_hash"], "out": str(output),
                        "output_fingerprint": fingerprint(output, exclude=("mlx-converter.json",))}
                    if plan.get("port_converter"):
                        evidence["port"] = plan["port_converter"]
                    _atomic_in_directory(output, "mlx-converter.json", (json.dumps(evidence, sort_keys=True) + "\n").encode(), 0o600)
                except (OSError, ValueError) as failure:
                    error = "Output provenance could not be saved: " + str(failure)
            current.update(completed_at=datetime.now(timezone.utc).isoformat(), exit_code=code,
                           exit_status="done" if observed and code == 0 and error is None else "failed",
                           completion_observed=observed)
            if error:
                current["error"] = error
            _write(path, current)
        return 0 if observed and code == 0 and error is None else 1

    finally:
        if kill_group and os.getpgrp() == os.getpid():
            try:
                Path(request_path).unlink(missing_ok=True)
            finally:
                os.killpg(os.getpid(), signal.SIGKILL)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.request)
    except (OSError, ValueError, KeyError, TypeError):
        return 2
    finally:
        candidate = Path(args.request)
        if candidate.name.startswith(".monitor-") and candidate.suffix == ".json":
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    sys.exit(main())
