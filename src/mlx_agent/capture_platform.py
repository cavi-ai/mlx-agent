"""Fail-closed Darwin evidence for an already connected loopback TCP socket."""

import ctypes as C
import errno
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Sequence

from .compute import process_identity
from .task_contracts import TaskError
from .capture_volume import CaptureVolume

ProcessIdentity = Dict[str, Any]


class _BSDInfo(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        "flags", "status", "xstatus", "pid", "ppid", "uid", "gid", "ruid", "rgid", "svuid", "svgid", "reserved")]
    _fields_ += [("comm", C.c_char * 16), ("name", C.c_char * 32)]
    _fields_ += [(name, C.c_uint32) for name in ("nfiles", "pgid", "jobc", "tdev", "tpgid", "nice")]
    _fields_ += [("start_seconds", C.c_uint64), ("start_microseconds", C.c_uint64)]


class _FDInfo(C.Structure):
    _fields_ = [("fd", C.c_int32), ("kind", C.c_uint32)]


class _FileInfo(C.Structure):
    _fields_ = [("flags", C.c_uint32), ("status", C.c_uint32), ("offset", C.c_int64),
                ("kind", C.c_int32), ("guards", C.c_uint32)]


class _VInfoStat(C.Structure):
    _fields_ = [("dev", C.c_uint32), ("mode", C.c_uint16), ("nlink", C.c_uint16),
                ("ino", C.c_uint64), ("uid", C.c_uint32), ("gid", C.c_uint32)]
    _fields_ += [(name, C.c_int64) for name in (
        "atime", "atimensec", "mtime", "mtimensec", "ctime", "ctimensec", "birthtime", "birthtimensec", "size", "blocks")]
    _fields_ += [("blksize", C.c_int32), ("flags", C.c_uint32), ("gen", C.c_uint32),
                ("rdev", C.c_uint32), ("spare", C.c_int64 * 2)]


class _SockBuf(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ("cc", "hiwat", "mbcnt", "mbmax", "lowat")]
    _fields_ += [("flags", C.c_int16), ("timeo", C.c_int16)]


class _Address(C.Union):
    _fields_ = [("bytes", C.c_ubyte * 16), ("words", C.c_uint32 * 4)]


class _IPv6Info(C.Structure):
    _fields_ = [("hlim", C.c_uint8), ("cksum", C.c_int), ("ifindex", C.c_uint16), ("hops", C.c_int16)]


class _InSock(C.Structure):
    _fields_ = [("foreign_port", C.c_int), ("local_port", C.c_int), ("generation", C.c_uint64),
                ("flags", C.c_uint32), ("flow", C.c_uint32), ("vflag", C.c_uint8),
                ("ttl", C.c_uint8), ("reserved", C.c_uint32), ("foreign", _Address),
                ("local", _Address), ("tos", C.c_uint8), ("v6", _IPv6Info)]


class _TCPSock(C.Structure):
    _fields_ = [("ip", _InSock), ("state", C.c_int), ("timers", C.c_int * 4),
                ("mss", C.c_int), ("flags", C.c_uint32), ("reserved", C.c_uint32), ("handle", C.c_uint64)]


class _Protocol(C.Union):
    # Darwin's largest public socket union member is un_sockinfo (528 bytes).
    _fields_ = [("tcp", _TCPSock), ("other", C.c_ubyte * 528)]


class _SocketInfo(C.Structure):
    _fields_ = [("stat", _VInfoStat), ("socket_id", C.c_uint64), ("pcb", C.c_uint64),
                ("type", C.c_int), ("protocol", C.c_int), ("family", C.c_int)]
    _fields_ += [(name, C.c_int16) for name in ("options", "linger", "state", "qlen", "incqlen", "qlimit", "timeo", "error")]
    _fields_ += [("oobmark", C.c_uint32), ("rcv", _SockBuf), ("snd", _SockBuf),
                 ("kind", C.c_int), ("reserved", C.c_uint32), ("proto", _Protocol)]


class _SocketFDInfo(C.Structure):
    _fields_ = [("file", _FileInfo), ("socket", _SocketInfo)]


def _libproc():
    if sys.platform != "darwin" or C.sizeof(_SocketFDInfo) != 792 or C.sizeof(_BSDInfo) != 136:
        raise OSError("The native process/socket ABI is unavailable")
    library = C.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_pidinfo.argtypes = [C.c_int, C.c_int, C.c_uint64, C.c_void_p, C.c_int]
    library.proc_pidinfo.restype = C.c_int
    library.proc_pidfdinfo.argtypes = [C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_int]
    library.proc_pidfdinfo.restype = C.c_int
    return library


def _started_at(pid):
    value = _BSDInfo()
    library = _libproc()
    if library.proc_pidinfo(pid, 3, 0, C.byref(value), C.sizeof(value)) != C.sizeof(value) or value.pid != pid:
        raise OSError(C.get_errno(), "Process start identity is unavailable")
    return [value.start_seconds, value.start_microseconds]


def capture_process_identity(pid: int) -> ProcessIdentity:
    """Bind existing command identity to the kernel's precise start time."""
    if type(pid) is not int or pid <= 0:
        raise TaskError("capture_peer_unavailable", "A positive owned process identity is required.")
    try:
        started = _started_at(pid)
        command = process_identity(pid)
        if not command or started != _started_at(pid):
            raise OSError("The process changed during inspection")
        return {"pid": pid, "identity": command, "started_at": started}
    except (OSError, ValueError) as error:
        raise TaskError("capture_peer_unavailable", "Cannot establish the serving process identity.") from error


def stop_owned(process, identity, grace=5):
    """Stop an owned process group, signalling only while it still has its recorded identity."""
    if process.poll() is not None:
        return
    if identity is None or capture_process_identity(process.pid) != identity:
        raise TaskError("capture_ownership_lost", "Cannot prove the owned process identity.")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        if capture_process_identity(process.pid) != identity:
            raise TaskError("capture_ownership_lost", "Cannot prove the owned process identity.")
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=grace)


def _socket_connections(pid):
    library = _libproc()
    size = library.proc_pidinfo(pid, 1, 0, None, 0)
    if size <= 0 or size > 1024 * 1024 - 256:
        raise OSError(C.get_errno(), "Owned process descriptor list is unavailable")
    descriptors = (_FDInfo * ((size + 256) // C.sizeof(_FDInfo)))()
    count = library.proc_pidinfo(pid, 1, 0, descriptors, C.sizeof(descriptors))
    if count <= 0 or count >= C.sizeof(descriptors) or count % C.sizeof(_FDInfo):
        raise OSError(C.get_errno(), "Owned process descriptor list changed or is incomplete")
    result = []
    for descriptor in descriptors[:count // C.sizeof(_FDInfo)]:
        if descriptor.kind != 2:  # PROX_FDTYPE_SOCKET
            continue
        info = _SocketFDInfo()
        count = library.proc_pidfdinfo(pid, descriptor.fd, 3, C.byref(info), C.sizeof(info))
        if count != C.sizeof(info):
            if count <= 0 and C.get_errno() in (errno.EBADF, errno.ENOENT):
                continue  # An unrelated descriptor closed; it proves no peer.
            raise OSError(C.get_errno(), "Owned socket details are unavailable")
        value, tcp = info.socket, info.socket.proto.tcp
        if (value.family, value.type, value.protocol, value.kind, tcp.state) != (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, 2, 4):
            continue
        if not tcp.ip.vflag & 1:
            continue
        result.append({"fd": descriptor.fd, "socket_id": value.socket_id, "generation": tcp.ip.generation,
                       "local": [socket.inet_ntop(socket.AF_INET, bytes(tcp.ip.local.bytes[12:])), socket.ntohs(tcp.ip.local_port & 0xffff)],
                       "peer": [socket.inet_ntop(socket.AF_INET, bytes(tcp.ip.foreign.bytes[12:])), socket.ntohs(tcp.ip.foreign_port & 0xffff)]})
    return result


def verify_connected_peer(sock: socket.socket, allowed: Sequence[ProcessIdentity]) -> Dict:
    """Inspect only authorized processes; send no HTTP headers or body."""
    if not allowed or len(allowed) > 64 or sock.family != socket.AF_INET:
        raise TaskError("capture_peer_unavailable", "Supported owned loopback peer identities are required.")
    try:
        local, peer = list(sock.getsockname()), list(sock.getpeername())
        if local[0] != "127.0.0.1" or peer[0] != "127.0.0.1":
            raise OSError("Capture requires an IPv4 loopback connection")
        deadline = time.monotonic() + 2
        while True:
            for expected in allowed:
                if not isinstance(expected, dict) or set(expected) != {"pid", "identity", "started_at"}:
                    raise OSError("The allowed process identity is invalid")
                if capture_process_identity(expected["pid"]) != expected:
                    raise TaskError("capture_peer_mismatch", "The pinned serving process changed.")
                matches = [entry for entry in _socket_connections(expected["pid"])
                           if entry["local"] == peer and entry["peer"] == local]
                for match in matches:
                    if not match["socket_id"]:
                        raise OSError("The kernel did not expose a socket identity")
                    if (capture_process_identity(expected["pid"]) == expected
                            and match in _socket_connections(expected["pid"])
                            and list(sock.getsockname()) == local and list(sock.getpeername()) == peer):
                        return dict(match, schema="connected-peer/1", pid=expected["pid"], identity=dict(expected))
            if time.monotonic() >= deadline:
                raise TaskError("capture_peer_mismatch", "The connected socket is not owned by the pinned server.")
            time.sleep(.02)
    except TaskError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise TaskError("capture_peer_unavailable", "Connected serving-process ownership cannot be inspected.") from error
