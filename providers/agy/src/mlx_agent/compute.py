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


def group_members(pgid):
    """Return live members, excluding zombies which no longer own compute."""
    if type(pgid) is not int or pgid <= 0:
        raise ValueError("A process group requires a positive ID.")
    result = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        raise OSError("The process table is unavailable.")
    members = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and int(fields[1]) == pgid and not fields[2].startswith("Z"):
            members.append(int(fields[0]))
    return members


class ComputeLease:
    def __init__(self, operation, root=None, inspector=process_identity, alive=process_alive, group_probe=group_members):
        self.operation = operation
        self.root = Path(root) if root is not None else compute_root()
        self.inspector, self.alive = inspector, alive
        self.descriptor = None
        self.detached = False
        self.group_probe = group_probe
        self.execution_group = None
        self.record = None

    def _write(self, value):
        encoded = json.dumps(value).encode()
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        os.write(self.descriptor, encoded)
        os.ftruncate(self.descriptor, len(encoded))
        os.fsync(self.descriptor)
        self.record = value

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
                fields = {"schema", "pid", "identity", "operation"}
                if isinstance(record, dict) and record.get("schema") == 2:
                    fields.add("execution_group")
                if not isinstance(record, dict) or set(record) != fields:
                    raise ValueError("Compute ownership record is invalid.")
                pid = record["pid"]
                if type(pid) is not int or pid <= 0 or record["schema"] not in (1, 2):
                    raise ValueError("Compute ownership record is invalid.")
                if record["schema"] == 2:
                    group = record["execution_group"]
                    if (not isinstance(group, dict) or set(group) != {"pgid", "leader_identity"}
                            or type(group["pgid"]) is not int or group["pgid"] <= 0
                            or not isinstance(group["leader_identity"], str) or not group["leader_identity"]):
                        raise ValueError("Compute execution group is invalid.")
                    try:
                        members = self.group_probe(group["pgid"])
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        raise ComputeBusy("The recorded execution group cannot be inspected; compute ownership is retained.") from error
                    if members:
                        raise ComputeBusy("A recorded task execution group still owns host compute (group {0}).".format(group["pgid"]))
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

    def bind_process_group(self, pid):
        if self.descriptor is None or type(pid) is not int or pid <= 0:
            raise ValueError("An active compute lease and positive group leader are required.")
        identity = self.inspector(pid)
        if not identity or os.getpgid(pid) != pid:
            raise ComputeBusy("The task execution group identity could not be established.")
        self.execution_group = {"pgid": pid, "leader_identity": identity}
        self._write(dict(self.record, schema=2, execution_group=self.execution_group))

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
                group_stopped = True
                if self.execution_group is not None:
                    try:
                        group_stopped = not self.group_probe(self.execution_group["pgid"])
                    except (OSError, ValueError, subprocess.SubprocessError):
                        group_stopped = False
                if not self.detached and group_stopped:
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
