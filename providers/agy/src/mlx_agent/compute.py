"""One host compute owner shared by native and agent entrypoints."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


class ComputeBusy(ValueError):
    pass


def compute_root():
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/mlx-agent/compute"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "mlx-agent/compute"


def process_identity(pid):
    result = subprocess.run(["/bin/ps", "-ww", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                            capture_output=True, text=True, timeout=5)
    return result.stdout.strip() if result.returncode == 0 else None


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class ComputeLease:
    def __init__(self, operation, root=None, inspector=process_identity, alive=process_alive):
        self.operation = operation
        self.root = Path(root) if root is not None else compute_root()
        self.inspector, self.alive = inspector, alive
        self.descriptor = None
        self.detached = False

    def _write(self, value):
        encoded = json.dumps(value).encode()
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        os.write(self.descriptor, encoded)
        os.ftruncate(self.descriptor, len(encoded))
        os.fsync(self.descriptor)

    def __enter__(self):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for path in (self.root, *self.root.parents):
            if path.is_symlink():
                raise ValueError("Compute state cannot traverse symbolic links.")
        info = self.root.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Compute state must be a private owned directory.")
        descriptor = os.open(self.root / "owner.json",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self.descriptor = descriptor
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError("Compute ownership must be a private regular file.")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ComputeBusy("Another model operation owns this host.") from None
            raw = os.read(descriptor, 8193)
            if len(raw) > 8192:
                raise ValueError("Compute ownership record is invalid.")
            record = json.loads(raw) if raw else {}
            if record:
                if not isinstance(record, dict) or set(record) != {"schema", "pid", "identity", "operation"}:
                    raise ValueError("Compute ownership record is invalid.")
                pid = record["pid"]
                if type(pid) is not int or pid <= 0 or record["schema"] != 1:
                    raise ValueError("Compute ownership record is invalid.")
                observed = self.inspector(pid)
                if self.alive(pid) and (record["identity"] is None or observed is None or observed == record["identity"]):
                    raise ComputeBusy("Host compute is owned by " + str(record["operation"]) + " (pid " + str(pid) + ").")
            identity = self.inspector(os.getpid())
            if identity is None:
                raise ComputeBusy("The current process identity could not be established.")
            self._write({"schema": 1, "pid": os.getpid(), "identity": identity, "operation": self.operation})
            return self
        except BaseException:
            os.close(descriptor)
            self.descriptor = None
            raise

    def handoff(self, pid):
        if type(pid) is not int or pid <= 0:
            raise ValueError("Detached compute owner requires a positive PID.")
        identity = self.inspector(pid)
        if identity is None:
            if not self.alive(pid):
                return  # A process that already exited owns no compute.
            self._write({"schema": 1, "pid": pid, "identity": None, "operation": self.operation})
            self.detached = True
            raise ComputeBusy("The launched process identity could not be established; its ownership is retained.")
        self._write({"schema": 1, "pid": pid, "identity": identity, "operation": self.operation})
        self.detached = True

    def __exit__(self, *_):
        if self.descriptor is not None:
            try:
                if not self.detached:
                    self._write({})
            finally:
                os.close(self.descriptor)
                self.descriptor = None


def guarded_start(operation, launch, *args, **kwargs):
    with ComputeLease(operation) as lease:
        result = launch(*args, **kwargs)
        pid = result.get("receipt", {}).get("pid")
        if pid is not None:
            lease.handoff(pid)
        return result
