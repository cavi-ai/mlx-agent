"""Persisted host ownership: exclusive operations and shared served executions."""
from __future__ import annotations

import fcntl
import json
import os
import secrets
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time


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
    probe = subprocess.Popen(["/bin/ps", "-axo", "pid=,pgid=,stat="], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        output, _ = probe.communicate(timeout=5)
    except BaseException:
        probe.kill()
        probe.wait()
        raise
    if probe.returncode != 0:
        raise OSError("The process table is unavailable.")
    members = []
    for line in output.splitlines():
        fields = line.split()
        # A probe executed inside the target group observes itself. Its own
        # short-lived PID must not keep that group permanently non-quiescent.
        if len(fields) >= 3 and int(fields[0]) != probe.pid and int(fields[1]) == pgid and not fields[2].startswith("Z"):
            members.append(int(fields[0]))
    return members


class ComputeLease:
    def __init__(self, operation, root=None, inspector=process_identity, alive=process_alive, group_probe=group_members, shared=False):
        self.shared = shared
        self.host_descriptor = None
        self.record_path = None
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
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.root / "owner.json", flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            descriptor = os.open(self.root / "owner.json", flags)
        self.descriptor = descriptor
        pending_path = None
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
                raise ValueError("Compute ownership must be a private regular file.")
            try:
                fcntl.flock(descriptor, (fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ComputeBusy("Another model operation owns this host.") from None
            self._check_record(descriptor)
            self._check_executions()
            if self.shared:
                self.host_descriptor = descriptor
                self.record_path = self.root / ("execution-" + secrets.token_hex(16) + ".json")
                staging = self.record_path.with_name('.' + self.record_path.name + '.pending')
                self.descriptor = os.open(staging, flags | os.O_CREAT | os.O_EXCL, 0o600)
                pending_path = staging
                try:
                    fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ComputeBusy("Another process is claiming this execution record.") from None
            identity = self.inspector(os.getpid())
            if identity is None:
                raise ComputeBusy("The current process identity could not be established.")
            self._write({"schema": 1, "pid": os.getpid(), "identity": identity, "operation": self.operation})
            if pending_path is not None:
                # Publish only a locked, complete record; scanners never see creation in flight.
                os.replace(pending_path, self.record_path)
                pending_path = None
            return self
        except BaseException:
            try:
                if pending_path is not None:
                    pending_path.unlink(missing_ok=True)
            finally:
                if self.descriptor is not None and self.descriptor != descriptor:
                    os.close(self.descriptor)
                os.close(descriptor)
                self.host_descriptor = self.descriptor = None
            raise

    def _check_record(self, descriptor):
        raw = os.pread(descriptor, 8193, 0)
        if len(raw) > 8192:
            raise ValueError("Compute ownership record is invalid.")
        record = json.loads(raw) if raw else {}
        if not isinstance(record, dict):
            raise ValueError("Compute ownership record is invalid.")
        if record:
            fields = {"schema", "pid", "identity", "operation"}
            if isinstance(record, dict) and record.get("schema") == 2:
                fields.add("execution_group")
            if not isinstance(record, dict) or set(record) != fields:
                raise ValueError("Compute ownership record is invalid.")
            pid = record["pid"]
            if (type(pid) is not int or pid <= 0 or type(record["schema"]) is not int
                    or record["schema"] not in (1, 2)
                    or not isinstance(record['operation'], str) or not record['operation']
                    or record['identity'] is not None and not isinstance(record['identity'], str)):
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

    def _check_executions(self):
        # A crashed gateway releases its flock, but its worker group may survive.
        # Active shared records are protected by their own flock and the host lock.
        for path in self.root.glob("execution-*.json"):
            try:
                descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
            except FileNotFoundError:
                continue
            try:
                info = os.fstat(descriptor)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_mode & 0o077 or info.st_nlink != 1):
                    raise ValueError("Compute ownership must be a private regular file.")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    if not self.shared:
                        raise ComputeBusy("A served execution still owns host compute.") from None
                    continue
                self._check_record(descriptor)
                path.unlink(missing_ok=True)
            finally:
                os.close(descriptor)

    def enter_wait(self, seconds=0, cancelled=lambda: False):
        """Bounded admission; cancellation never transfers ownership."""
        deadline = time.monotonic() + seconds
        while True:
            if cancelled():
                raise InterruptedError('Execution admission cancelled.')
            try:
                return self.__enter__()
            except ComputeBusy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(.05, remaining))

    def bind_process_group(self, pid):
        if self.descriptor is None or type(pid) is not int or pid <= 0:
            raise ValueError("An active compute lease and positive group leader are required.")
        self.execution_group = {"pgid": pid, "leader_identity": "unverified"}
        self._write(dict(self.record, schema=2, execution_group=self.execution_group))
        identity = self.inspector(pid)
        if not identity or os.getpgid(pid) != pid:
            raise ComputeBusy("The task execution group identity could not be established.")
        self.execution_group = {"pgid": pid, "leader_identity": identity}
        self._write(dict(self.record, schema=2, execution_group=self.execution_group))

    def release_idle_process_group(self, pid, members=None):
        """Release only after completion and inspection of the idle private group."""
        expected = {pid: self.execution_group["leader_identity"]} if members is None and self.execution_group else members
        if (self.descriptor is None or self.execution_group is None or self.execution_group['pgid'] != pid
                or self.inspector(pid) != self.execution_group['leader_identity']
                or not expected or set(self.group_probe(pid)) != set(expected)
                or any(self.inspector(member) != identity for member, identity in expected.items())):
            raise ComputeBusy("The resident worker has not returned to an inspected idle state.")
        self.execution_group = None
        self._write({key: value for key, value in self.record.items() if key != 'execution_group'} | {'schema': 1})

    def handoff(self, pid, process_group=False):
        if type(pid) is not int or pid <= 0:
            raise ValueError("Detached compute owner requires a positive PID.")
        if process_group:
            self.execution_group = {"pgid": pid, "leader_identity": "unverified"}
        record = {"schema": 1, "pid": pid, "identity": None, "operation": self.operation}
        if self.execution_group is not None:
            record.update(schema=2, execution_group=dict(self.execution_group))
        # Persist ownership before asking ps about a process that already exists.
        # Missing identity or a lost leader must never release surviving children.
        self._write(record)
        self.detached = True
        try:
            identity = self.inspector(pid)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise ComputeBusy("The launched process cannot be inspected; its ownership is retained.") from error
        if identity is None:
            if not self.alive(pid):
                return
            raise ComputeBusy("The launched process identity could not be established; its ownership is retained.")
        record["identity"] = identity
        if record.get("execution_group", {}).get("pgid") == pid:
            record["execution_group"]["leader_identity"] = identity
            self.execution_group = record["execution_group"]
        self._write(record)

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
                    if self.record_path is not None:
                        self.record_path.unlink()
            finally:
                os.close(self.descriptor)
                self.descriptor = None
                if self.host_descriptor is not None:
                    os.close(self.host_descriptor)
                    self.host_descriptor = None


def guarded_start(operation, launch, *args, **kwargs):
    with ComputeLease(operation) as lease:
        observed_conversion = operation == "convert" and bool(args and args[0].get("require_completion"))
        if observed_conversion:
            kwargs["wait_for_compute_lease"] = True
        result = launch(*args, **kwargs)
        pid = result.get("receipt", {}).get("pid")
        if pid is not None:
            lease.handoff(pid, process_group=observed_conversion)
            if observed_conversion:
                from .convert import register_monitor_lease
                register_monitor_lease(result["receipt_path"], result["receipt"])
        return result


class ExecutionAdmission:
    """Batch requests to one backend under one shared host lease."""
    def __init__(self, operation, root=None, shared=False):
        self.shared = shared
        self.operation, self.root = operation, root
        self.condition = threading.Condition()
        self.lease = None
        self.users = 0
        self.waiting = 0
        self.closed = False

    def acquire(self, seconds=0, cancelled=lambda: False):
        deadline = time.monotonic() + seconds
        with self.condition:
            self.waiting += 1
            try:
                while True:
                    if self.closed or cancelled():
                        raise InterruptedError('Execution admission cancelled.')
                    if self.users:
                        self.users += 1
                        return
                    lease = ComputeLease(self.operation, self.root, shared=self.shared() if callable(self.shared) else self.shared)
                    try:
                        lease.__enter__()
                    except ComputeBusy:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise
                        self.condition.wait(min(.05, remaining))
                        continue
                    if self.closed or cancelled():
                        lease.__exit__(None, None, None)
                        raise InterruptedError('Execution admission cancelled.')
                    self.lease, self.users = lease, 1
                    return
            finally:
                self.waiting -= 1

    def bind(self, pid, refresh=False):
        """Bind the backend group; `refresh` records the leader again after it execs."""
        with self.condition:
            if self.lease is None:
                raise ComputeBusy('Model startup requires execution admission.')
            if not refresh and self.lease.execution_group and self.lease.execution_group['pgid'] == pid:
                return
            self.lease.bind_process_group(pid)

    def release(self, idle):
        with self.condition:
            if self.users == 0:
                return
            self.users -= 1
            if self.users == 0:
                try:
                    idle(self.lease)
                except BaseException:
                    # Retain the descriptor until shutdown proves the group stopped.
                    self.closed = True
                    self.condition.notify_all()
                    raise
                try:
                    self.lease.__exit__(None, None, None)
                finally:
                    self.lease = None
                    self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            if self.lease is not None:
                self.lease.__exit__(None, None, None)
                self.lease = None
            self.users = 0
            self.condition.notify_all()
