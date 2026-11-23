"""Private, host-wide agent memory policy and process-family accounting."""
import ctypes
import datetime
import errno
import json
import os
import stat
import subprocess
import sys
import time
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


def _usage(pid):
    usage = _RusageInfoV0()
    if _rusage_function()(pid, 0, ctypes.byref(usage)) != 0:
        if ctypes.get_errno() == errno.ESRCH:
            raise ProcessLookupError(pid)
        return None  # An unreadable tracked process must block admission.
    if usage.exit_time:
        raise ProcessLookupError(pid)  # The process exited after the table was sampled.
    return usage


def _physical_bytes(pid):
    usage = _usage(pid)
    return None if usage is None else usage.phys_footprint


def identified_pids(policy, entries, external_pids=()):
    """The PIDs whose start and command ``account`` compares: external bindings and managed workers."""
    pids = {binding['pid'] for binding in policy['external']} if policy is not None else set()
    if policy is not None:
        pids.update(binding['group'] for binding in policy['external'])
    pids.update(external_pids or ())
    pids.update(entry['worker']['pid'] for entry in entries.values() if entry['worker'] is not None)
    return pids


def process_snapshot(identities=None):
    """Live processes. Identity for the PIDs asked, or for every process when None.

    A macOS snapshot taken for a known PID set does not read footprint. ``account``
    reads that for the process family it charges.
    """
    if identities is not None and sys.platform == 'darwin':
        from .process_table import identities as observe, rows as table
        rows = {pid: {'parent': parent, 'group': group} for pid, (parent, group) in table().items()}
        for pid, identity in observe(pid for pid in identities if pid in rows).items():
            rows[pid]['identity'] = identity
        # A missing identity can also be a permissions failure; retain the observed row.
        return rows
    result = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss=,stat=,lstart=,command='],
                            capture_output=True, text=True, timeout=5, check=True)
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and not fields[4].startswith('Z'):
            pid, parent, group, rss = map(int, fields[:4])
            rows[pid] = {'parent': parent, 'group': group, 'rss': rss * 1024,
                         'identity': fields[5]}
    if sys.platform == 'darwin':
        for pid in list(rows):
            try:
                rows[pid]['physical_bytes'] = _physical_bytes(pid)
            except ProcessLookupError:
                del rows[pid]  # Short-lived children can exit after ps returned.
    return rows


def same_identity(left, right):
    return isinstance(left, str) and isinstance(right, str) and left.split() == right.split()


def read_policy(root, with_mtime=False):
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
    return (value, info.st_mtime_ns) if with_mtime else value


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


def _earliest_start_ns(identity):
    """Parse ps lstart, choosing the earliest valid epoch when local time repeats."""
    if not isinstance(identity, str):
        return None
    fields = identity.split(None, 5)
    if len(fields) < 6:
        return None
    try:
        parsed = datetime.datetime.strptime(' '.join(fields[:5]), '%a %b %d %H:%M:%S %Y')
        stamps = []
        for dst in (0, 1):
            stamp = time.mktime((*parsed.timetuple()[:8], dst))
            if time.localtime(stamp)[:6] == parsed.timetuple()[:6]:
                stamps.append(stamp)
        return int(min(stamps) * 1_000_000_000) if stamps else None
    except (ValueError, OverflowError, OSError):
        return None


def expire_external(policy, rows, saved_ns):
    """Remove only bindings whose root and original process group have ended."""
    retained = []
    groups = {row['group'] for row in rows.values()}
    for binding in policy['external']:
        root = rows.get(binding['pid'])
        if root is not None and same_identity(binding['identity'], root.get('identity')):
            retained.append(binding)
            continue
        if root is not None:
            root_birth = _earliest_start_ns(root.get('identity'))
            if root_birth is None or root_birth <= saved_ns:
                retained.append(binding)
                continue
        if binding['group'] not in groups:
            continue
        leader = rows.get(binding['group'])
        birth = _earliest_start_ns(leader.get('identity')) if leader and leader['group'] == binding['group'] else None
        if birth is None or birth <= saved_ns:
            retained.append(binding)
    return dict(policy, external=retained)


def family(rows, roots, groups=()):
    found = {pid for pid, row in rows.items() if pid in roots or row['group'] in groups}
    while True:
        children = {pid for pid, row in rows.items() if row['parent'] in found}
        if children <= found:
            return found
        found.update(children)


def _charged_pids(policy, entries, rows, owner):
    """PIDs ``account`` can charge: external families, managed workers, and the owner."""
    watched = set()
    for binding in policy['external']:
        observed = rows.get(binding['pid'])
        if observed is not None and same_identity(binding['identity'], observed.get('identity')):
            watched.update(family(rows, {binding['pid']}))
        else:
            watched.update(family(rows, set(), {binding['group']}))
    for entry in entries.values():
        worker = entry.get('worker')
        if worker is not None:
            watched.update(family(rows, {worker['pid']}, {worker['pgid']}))
    if owner[0] in rows:
        watched.add(owner[0])
    return watched


def _fill_footprint(rows, pids):
    """Read resident and physical size for ``pids`` that do not already have it."""
    if sys.platform != 'darwin':
        return
    for pid in list(pids):
        row = rows.get(pid)
        if row is None or 'rss' in row or 'physical_bytes' in row:
            continue
        try:
            usage = _usage(pid)
        except ProcessLookupError:
            del rows[pid]  # Exited after the table was read.
            continue
        row['rss'] = None if usage is None else usage.resident_size
        row['physical_bytes'] = None if usage is None else usage.phys_footprint


def account(policy, entries, rows, owner, replacement=None):
    """Count physical process observations once; pending allocations are additional."""
    from .residency import MemoryAdmissionError

    _fill_footprint(rows, _charged_pids(policy, entries, rows, owner))

    def rss(pids):
        values = []
        for pid in pids:
            row = rows[pid]
            resident = row.get('rss')
            physical = row.get('physical_bytes', resident)
            if any(type(value) is not int or value < 0 for value in (resident, physical)):
                raise MemoryAdmissionError('memory_unknown',
                    f'Agent process memory is unavailable for PID {pid}.')
            # RSS misses Metal allocations; footprint also includes compressed memory.
            values.append(max(resident, physical))
        return sum(values)

    external = set()
    for binding in policy['external']:
        observed = rows.get(binding['pid'])
        if observed is not None and not observed.get('identity'):
            raise MemoryAdmissionError('memory_unknown', f'Agent process identity is unavailable for PID {binding["pid"]}.')
        if observed is not None and same_identity(binding['identity'], observed.get('identity')):
            external.update(family(rows, {binding['pid']}))
        else:
            # A surviving or ambiguous original group remains charged.
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
            allocations[key] = {'resident_bytes': 0, 'pending_bytes': amount,
                                'observed_bytes': rss(pids) if worker is not None else 0,
                                'execution_bytes': entry.get('execution_bytes')}
            continue
        worker = entry['worker']
        pids = set()
        if worker is not None:
            observed = rows.get(worker['pid'])
            if observed is not None and not same_identity(worker['identity'], observed.get('identity')):
                raise MemoryAdmissionError('memory_unknown', 'A managed worker identity changed.')
            pids = family(rows, {worker['pid']}, {worker['pgid']})
        uncounted = pids - counted
        # Overlapping process roots cannot be charged twice. The model estimate
        # is retained when the original process is absent or not yet observable.
        estimate = 0 if pids and not uncounted else entry['resident_bytes']
        charged = max(estimate, rss(uncounted))
        resident += charged
        counted.update(pids)
        if new_amount is None and 'execution_bytes' in entry:
            # Actual allocations consume their reservation rather than adding to it.
            workspace = max(0, entry['execution_bytes'] - charged)
        else:
            # A settled reservation already holds its growth allowance; measured
            # growth beyond the resident estimate consumes it rather than adding to it.
            grown = min(entry['growth_bytes'], max(0, charged - entry['resident_bytes']))
            workspace = max(0, (entry['reserved_bytes'] if new_amount is None else new_amount) - grown)
        pending += workspace
        allocations[key] = {'resident_bytes': charged, 'pending_bytes': workspace,
                            'observed_bytes': rss(pids), 'execution_bytes': entry.get('execution_bytes')}
    if replacement is not None and replacement[0] not in entries:
        pending += replacement[1]
    # An owner missing from the process table has exited and holds no memory;
    # its reservations stay charged through their workers and estimates above.
    owners &= rows.keys()
    for pid in owners - counted:
        overhead += rss({pid})
    total = external_bytes + resident + pending + overhead
    admission = policy['limit_bytes'] - policy['reserve_bytes']
    observed = rss(counted | owners)
    return {'schema': SCHEMA, 'limit_bytes': policy['limit_bytes'], 'reserve_bytes': policy['reserve_bytes'],
            'admission_bytes': admission, 'external_bytes': external_bytes, 'resident_bytes': resident,
            'pending_bytes': pending, 'overhead_bytes': overhead, 'accounted_bytes': total,
            'within_budget': total <= admission, 'observed_bytes': observed, 'allocations': allocations}
