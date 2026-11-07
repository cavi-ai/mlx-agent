"""Identity-checked cleanup of children captured from an owned POSIX process."""
import os
import signal
import subprocess
import time


def _snapshot():
    run = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,stat=,lstart=,command="],
                         capture_output=True, text=True, timeout=5, check=True)
    result = {}
    for line in run.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and not parts[2].startswith("Z"):
            result[int(parts[0])] = (int(parts[1]), parts[3])
    return result


def descendants(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError("An owned process requires a positive PID.")
    snapshot, found = _snapshot(), {}
    parents = {pid}
    while parents:
        children = {p for p, (parent, _) in snapshot.items() if parent in parents and p not in found and p != pid}
        found.update({p: snapshot[p][1] for p in children})
        parents = children
    return found


def remaining(identities):
    snapshot = _snapshot()
    return {pid: identity for pid, identity in identities.items()
            if pid in snapshot and snapshot[pid][1] == identity}


def terminate(identities, grace_seconds=3):
    """Never signal a replacement PID or an unrelated process from the same group."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in remaining(identities):
            # Recheck immediately before each signal, including the captured start and command.
            if pid in remaining({pid: identities[pid]}):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + grace_seconds
        while remaining(identities) and time.monotonic() < deadline:
            time.sleep(.1)
        if not remaining(identities):
            return
    raise OSError("A captured Strata engine is still running; model reload is blocked.")
