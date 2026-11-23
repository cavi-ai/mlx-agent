"""The live process table read in-process: libproc on macOS, /proc elsewhere.

Rows exclude zombies, which no longer hold compute or memory. Start times and command lines
still come from ps, for only the PIDs that need them, so recorded identities keep one format.
"""
import ctypes
import errno
import os
import subprocess
import sys
from functools import lru_cache

_SHORT_INFO = 13  # PROC_PIDT_SHORTBSDINFO is readable for every user's processes.
_ZOMBIE = 5  # SZOMB


class _ShortInfo(ctypes.Structure):
    # Public Darwin ABI: struct proc_bsdshortinfo in sys/proc_info.h.
    _fields_ = [('pid', ctypes.c_uint32), ('ppid', ctypes.c_uint32), ('pgid', ctypes.c_uint32),
                ('status', ctypes.c_uint32), ('comm', ctypes.c_char * 16), ('flags', ctypes.c_uint32),
                ('uid', ctypes.c_uint32), ('gid', ctypes.c_uint32), ('ruid', ctypes.c_uint32),
                ('rgid', ctypes.c_uint32), ('svuid', ctypes.c_uint32), ('svgid', ctypes.c_uint32),
                ('rfu', ctypes.c_uint32)]


@lru_cache(maxsize=1)
def _libproc():
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    library.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.proc_listpgrppids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
    for function in (library.proc_listallpids, library.proc_listpgrppids, library.proc_pidinfo):
        function.restype = ctypes.c_int
    return library


def _darwin_pids(group=None):
    capacity = 4096
    while True:
        buffer = (ctypes.c_int * capacity)()
        library = _libproc()
        ctypes.set_errno(0)
        found = (library.proc_listallpids(buffer, ctypes.sizeof(buffer)) if group is None
                 else library.proc_listpgrppids(group, buffer, ctypes.sizeof(buffer)))
        code = ctypes.get_errno()
        if found < 0 or found == 0 and code:
            raise OSError(code, 'The process table is unavailable.')
        if found < capacity:  # A full buffer may have dropped PIDs.
            return [pid for pid in buffer[:found] if pid > 0]
        capacity *= 2


def _darwin_row(pid):
    info = _ShortInfo()
    if _libproc().proc_pidinfo(pid, _SHORT_INFO, 0, ctypes.byref(info), ctypes.sizeof(info)) == ctypes.sizeof(info):
        return None if info.status == _ZOMBIE else (info.ppid, info.pgid)
    code = ctypes.get_errno()
    if code == errno.ESRCH:  # Exited, or a zombie awaiting its parent.
        return None
    raise OSError(code, 'Process {0} cannot be inspected.'.format(pid))


def _proc_pids():
    try:
        return [int(name) for name in os.listdir('/proc') if name.isdigit()]
    except OSError as error:
        raise OSError(error.errno, 'The process table is unavailable.') from error


def _proc_row(pid):
    try:
        with open('/proc/{0}/stat'.format(pid), 'rb') as stream:
            # State, parent and group follow the parenthesised command name, which may contain ')'.
            fields = stream.read().rsplit(b')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None if fields[0] in (b'Z', b'X') else (int(fields[1]), int(fields[2]))


def rows(pids=None):
    """{pid: (parent, group)} for live processes; all of them, or the PIDs asked."""
    darwin = sys.platform == 'darwin'
    if pids is None:
        pids = _darwin_pids() if darwin else _proc_pids()
    read = _darwin_row if darwin else _proc_row
    return {pid: row for pid in pids if (row := read(pid)) is not None}


def group(pgid):
    """Live members of one process group."""
    candidates = rows(_darwin_pids(pgid)) if sys.platform == 'darwin' else rows()
    return sorted(pid for pid, (_, member_group) in candidates.items() if member_group == pgid)


def identities(pids, *, timeout=5, env=None):
    """{pid: 'start command'} for the live PIDs asked, in the form ``compute.process_identity`` returns."""
    pids = sorted(set(pids))
    if not pids:
        return {}
    run = subprocess.run(['/bin/ps', '-ww', '-o', 'pid=,lstart=,command=', '-p', ','.join(map(str, pids))],
                         capture_output=True, text=True, timeout=timeout, env=env)
    # ps exits 1 when a listed PID is gone and still prints the live ones.
    if run.returncode not in (0, 1) or run.returncode == 1 and run.stderr.strip():
        raise OSError('The process table is unavailable.')
    result = {}
    for line in run.stdout.splitlines():
        fields = line.split(None, 1)
        if len(fields) == 2:
            result[int(fields[0])] = fields[1].strip()
    return result
