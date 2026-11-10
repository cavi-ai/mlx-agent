"""Private, host-wide agent memory policy and process-family accounting."""
import ctypes
import errno
import json
import os
import stat
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

SCHEMA = 'agent-memory-budget/1'


class _RusageInfoV0(ctypes.Structure):
    # Public Darwin ABI from sys/resource.h; flavor 0 is available since 10.9.
    _fields_ = [('uuid', ctypes.c_ubyte * 16)] + [
        (name, ctypes.c_uint64) for name in
        ('user_time', 'system_time', 'idle_wakeups', 'interrupt_wakeups',
         'pageins', 'wired_size', 'resident_size', 'phys_footprint',
         'start_time', 'exit_time')]


@lru_cache(maxsize=1)
def _rusage_function():
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    function = library.proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RusageInfoV0)]
    function.restype = ctypes.c_int
    return function


def _physical_bytes(pid):
    usage = _RusageInfoV0()
    if _rusage_function()(pid, 0, ctypes.byref(usage)) != 0:
        if ctypes.get_errno() == errno.ESRCH:
            raise ProcessLookupError(pid)
        return None  # An unreadable tracked process must block admission.
    return usage.phys_footprint


def process_snapshot():
    result = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss=,lstart=,command='],
                            capture_output=True, text=True, timeout=5, check=True)
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split(None, 4)
        if len(fields) == 5:
            pid, parent, group, rss = map(int, fields[:4])
            rows[pid] = {'parent': parent, 'group': group, 'rss': rss * 1024,
                         'identity': fields[4]}
    if sys.platform == 'darwin':
        for pid in list(rows):
            try:
                rows[pid]['physical_bytes'] = _physical_bytes(pid)
            except ProcessLookupError:
                del rows[pid]  # Short-lived children can exit after ps returned.
    return rows


def same_identity(left, right):
    return isinstance(left, str) and isinstance(right, str) and left.split() == right.split()


def read_policy(root):
    path = Path(root) / 'budget.json'
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_nlink != 1 or info.st_size > 65536):
            raise ValueError('Agent memory policy must be a private owned regular file.')
        value = json.loads(stream.read(65537))
    validate_policy(value)
    return value


def validate_policy(value):
    if (not isinstance(value, dict) or set(value) != {'schema', 'limit_bytes', 'reserve_bytes', 'external'}
            or value['schema'] != SCHEMA or type(value['limit_bytes']) is not int
            or not 0 < value['limit_bytes'] <= 2**60 or type(value['reserve_bytes']) is not int
            or not 0 <= value['reserve_bytes'] < value['limit_bytes']
            or not isinstance(value['external'], list) or len(value['external']) > 256):
        raise ValueError('Invalid agent memory budget.')
    for row in value['external']:
        if (not isinstance(row, dict) or set(row) != {'pid', 'identity', 'group'}
                or type(row['pid']) is not int or row['pid'] <= 0
                or type(row['group']) is not int or row['group'] <= 0
                or not isinstance(row['identity'], str) or not row['identity']):
            raise ValueError('Invalid external agent process identity.')


def family(rows, roots, groups=()):
    found = {pid for pid, row in rows.items() if pid in roots or row['group'] in groups}
    while True:
        children = {pid for pid, row in rows.items() if row['parent'] in found}
        if children <= found:
            return found
        found.update(children)


def account(policy, entries, rows, owner, replacement=None):
    """Count physical process observations once; pending allocations are additional."""
    from .residency import MemoryAdmissionError

    def rss(pids):
        values = []
        for pid in pids:
            row = rows[pid]
            resident = row.get('rss')
            physical = row.get('physical_bytes', resident)
            if any(type(value) is not int or value < 0 for value in (resident, physical)):
                raise MemoryAdmissionError('memory_unknown',
                    'Agent process memory is unavailable for PID {}.'.format(pid))
            # RSS misses Metal allocations; footprint also includes compressed memory.
            values.append(max(resident, physical))
        return sum(values)

    external = set()
    for binding in policy['external']:
        observed = rows.get(binding['pid'])
        if observed is not None and same_identity(binding['identity'], observed['identity']):
            external.update(family(rows, {binding['pid']}))
        elif observed is None:
            # A reparented child in the original group remains charged.
            external.update(family(rows, set(), {binding['group']}))
    counted = set(external)
    external_bytes = rss(external)
    resident = pending = overhead = 0
    allocations = {}
    owners = {owner[0]}
    for key, entry in entries.items():
        owners.add(entry['owner'][0])
        new_amount = replacement[1] if replacement is not None and replacement[0] == key else None
        if entry['phase'] != 'resident':
            amount = entry['reserved_bytes'] if new_amount is None else new_amount
            # Cold/launching reservations already include worker allocations.
            worker = entry['worker']
            if worker is not None:
                pids = family(rows, {worker['pid']}, {worker['pgid']})
                amount = max(amount, rss(pids - counted))
                counted.update(pids)
            pending += amount
            allocations[key] = {'resident_bytes': 0, 'pending_bytes': amount}
            continue
        worker = entry['worker']
        pids = set()
        if worker is not None:
            observed = rows.get(worker['pid'])
            if observed is not None and not same_identity(worker['identity'], observed['identity']):
                raise MemoryAdmissionError('memory_unknown', 'A managed worker identity changed.')
            pids = family(rows, {worker['pid']}, {worker['pgid']})
        uncounted = pids - counted
        # Overlapping process roots cannot be charged twice. The model estimate
        # is retained when the original process is absent or not yet observable.
        estimate = 0 if pids and not uncounted else entry['resident_bytes']
        charged = max(estimate, rss(uncounted))
        resident += charged
        counted.update(pids)
        pending += (entry['reserved_bytes'] if new_amount is None else new_amount) + entry['growth_bytes']
        allocations[key] = {'resident_bytes': charged,
                            'pending_bytes': (entry['reserved_bytes'] if new_amount is None else new_amount) + entry['growth_bytes']}
    if replacement is not None and replacement[0] not in entries:
        pending += replacement[1]
    for pid in owners - counted:
        if pid not in rows:
            raise MemoryAdmissionError('memory_unknown', 'An agent owner cannot be inspected.')
        overhead += rss({pid})
    total = external_bytes + resident + pending + overhead
    admission = policy['limit_bytes'] - policy['reserve_bytes']
    observed = rss(counted | owners)
    return {'schema': SCHEMA, 'limit_bytes': policy['limit_bytes'], 'reserve_bytes': policy['reserve_bytes'],
            'admission_bytes': admission, 'external_bytes': external_bytes, 'resident_bytes': resident,
            'pending_bytes': pending, 'overhead_bytes': overhead, 'accounted_bytes': total,
            'within_budget': total <= admission, 'observed_bytes': observed, 'allocations': allocations}
