"""Atomic host memory reservations for independently owned model workers."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat
from subprocess import SubprocessError
import time

from .compute import compute_root, process_alive, process_identity, group_members
from .transactions import _atomic_in_directory, _walk_directory

SCHEMA = 'model-residency/1'
MAX_BYTES = 1024 * 1024


class MemoryAdmissionError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}


def _bytes(value):
    return type(value) is int and 0 <= value <= 2**60


class MemoryRegistry:
    """Reserve future allocations; OS headroom already includes resident arrays."""
    def __init__(self, root=None, owner=None, inspector=process_identity,
                 alive=process_alive, group_probe=group_members, lock_timeout=5, process_probe=None):
        self.root = Path(root) if root is not None else compute_root()
        self.owner = owner
        self.inspector, self.alive, self.group_probe = inspector, alive, group_probe
        self.lock_timeout = lock_timeout
        self.process_probe = process_probe

    def _owner(self):
        if self.owner is None:
            identity = self.inspector(os.getpid())
            if not identity:
                raise MemoryAdmissionError('memory_owner_unknown', 'Cannot establish memory reservation ownership.')
            self.owner = os.getpid(), identity
        return list(self.owner)

    @staticmethod
    def _validate(value):
        if (not isinstance(value, dict) or set(value) != {'schema','entries'}
                or value['schema'] != SCHEMA or not isinstance(value['entries'], dict)
                or len(value['entries']) > 256):
            raise ValueError('Invalid residency registry.')
        fields = {'owner','worker','phase','model_key','resident_bytes','workspace_bytes','growth_bytes',
                  'reserved_bytes','headroom_bytes','telemetry'}
        for key, row in value['entries'].items():
            if (not isinstance(key,str) or not 1 <= len(key) <= 128 or not isinstance(row,dict)
                    or set(row) != fields or row['phase'] not in ('reserved','launching','resident')
                    or not isinstance(row['model_key'],str) or len(row['model_key']) != 64
                    or any(not _bytes(row[name]) for name in ('resident_bytes','workspace_bytes','growth_bytes','reserved_bytes','headroom_bytes'))):
                raise ValueError('Invalid residency entry.')
            owner = row['owner']
            if (not isinstance(owner,list) or len(owner) != 2 or type(owner[0]) is not int
                    or owner[0] <= 0 or not isinstance(owner[1],str) or not owner[1]):
                raise ValueError('Invalid reservation owner.')
            worker = row['worker']
            if worker is not None and (not isinstance(worker,dict) or set(worker) != {'pid','identity','pgid'}
                    or any(type(worker[n]) is not int or worker[n] <= 0 for n in ('pid','pgid'))
                    or not isinstance(worker['identity'],str) or not worker['identity']):
                raise ValueError('Invalid worker owner.')
            if row['telemetry'] is not None:
                from .worker_telemetry import validate
                validate(row['telemetry'])
        return value

    def _live(self, pid, identity):
        if not self.alive(pid):
            return False
        observed = self.inspector(pid)
        return not observed or observed == identity

    def _prune(self, rows):
        for key, row in list(rows.items()):
            owner = row['owner']
            if self.owner is not None and owner == list(self.owner):
                continue
            try:
                if self._live(*owner):
                    continue
                worker = row['worker']
                if worker is None:
                    if row['phase'] != 'reserved':
                        continue  # Launch may have occurred before its PID was persisted.
                elif self._live(worker['pid'],worker['identity']) or self.group_probe(worker['pgid']):
                    continue
            except (OSError, ValueError, SubprocessError):
                continue  # Unknown ownership never authorizes reclaiming memory.
            del rows[key]

    @contextmanager
    def _locked(self):
        descriptor = None
        directory_fd = None
        try:
            _, directory_fd = _walk_directory(self.root, create=True)
            info = os.fstat(directory_fd)
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError('Memory state requires a private owned directory.')
            try:
                descriptor = os.open('memory.lock',os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW,0o600,dir_fd=directory_fd)
            except FileExistsError:
                descriptor = os.open('memory.lock',os.O_RDWR|os.O_NOFOLLOW,dir_fd=directory_fd)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
                raise ValueError('Invalid memory lock.')
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise MemoryAdmissionError('memory_registry_busy','Another memory admission is still in progress.')
                    time.sleep(.01)
            current = os.stat('memory.lock',dir_fd=directory_fd,follow_symlinks=False)
            if (current.st_dev,current.st_ino) != (info.st_dev,info.st_ino):
                raise ValueError('Memory lock identity changed.')
            try:
                source = os.open('memory.json',os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory_fd)
            except FileNotFoundError:
                value = {'schema':SCHEMA,'entries':{}}
            else:
                with os.fdopen(source,'rb') as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1 or info.st_size > MAX_BYTES:
                        raise ValueError('Invalid memory registry file.')
                    value = self._validate(json.loads(stream.read(MAX_BYTES+1)))
            original = json.dumps(value,allow_nan=False,separators=(',',':')).encode()
            self._prune(value['entries'])
            yield value['entries']
            encoded = json.dumps(value,allow_nan=False,separators=(',',':')).encode()
            if len(encoded) > MAX_BYTES:
                raise ValueError('Memory registry exceeded its limit.')
            if encoded != original:
                _atomic_in_directory(self.root,'memory.json',encoded,0o600)
        except MemoryAdmissionError:
            raise
        except (OSError,ValueError,TypeError,KeyError,RecursionError) as error:
            raise MemoryAdmissionError('memory_registry_invalid',str(error)) from error
        finally:
            if descriptor is not None:os.close(descriptor)
            if directory_fd is not None:os.close(directory_fd)

    @staticmethod
    def _summary(rows):
        return {'schema':SCHEMA,'entries':len(rows),
                'resident_bytes':sum(r['resident_bytes'] for r in rows.values()),
                'reserved_bytes':sum(r['reserved_bytes'] for r in rows.values()),
                'headroom_bytes':max((r['headroom_bytes'] for r in rows.values()),default=0)}

    def _own(self, rows, key):
        row = rows[key]
        if row['owner'] != self._owner():
            raise MemoryAdmissionError('memory_owner_changed','The reservation belongs to another process.')
        return row

    def _admit(self, rows, key, amount, headroom, probe):
        """Inspect fresh headroom while all competing admissions hold this lock."""
        others = [r for k,r in rows.items() if k != key]
        pending = sum(r['reserved_bytes'] for r in others)
        available = probe()
        requested = headroom() if callable(headroom) else headroom
        if not _bytes(requested):
            raise MemoryAdmissionError('memory_unknown','Invalid headroom reserve.')
        floor = max([requested]+[r['headroom_bytes'] for r in others])
        details = {'available_bytes':available,'required_available_bytes':amount+pending+floor,
                   'other_reserved_bytes':pending,'headroom_bytes':floor,'requested_headroom_bytes':requested,
                   'captured_at':time.time(),'estimated':True}
        budget = self._agent_budget(rows, (key, amount))
        if budget is not None:
            details['agent_budget'] = budget
            if not budget['within_budget']:
                raise MemoryAdmissionError('agent_memory_budget_exceeded',
                    'The request exceeds the shared agent memory budget; unload an idle model first.', details)
        if not _bytes(available):
            raise MemoryAdmissionError('memory_unknown','OS memory headroom is unavailable.',details)
        if available < details['required_available_bytes']:
            raise MemoryAdmissionError('insufficient_headroom','Insufficient memory for pending model allocations and application headroom.',details)
        return details

    def reserve(self, key, model_key, resident_bytes, workspace_bytes, headroom_bytes, probe, growth_bytes=0):
        """Reserve a cold load before it can allocate weights or workspace."""
        if (any(not _bytes(v) for v in (resident_bytes,workspace_bytes,growth_bytes))
                or not (_bytes(headroom_bytes) or callable(headroom_bytes))):
            raise MemoryAdmissionError('memory_unknown','Model memory estimate is unavailable.')
        with self._locked() as rows:
            if key in rows:self._own(rows,key)
            details = self._admit(rows,key,resident_bytes+workspace_bytes+growth_bytes,headroom_bytes,probe)
            if key not in rows and len(rows) >= 256:
                raise MemoryAdmissionError('memory_registry_full','Too many model reservations.')
            rows[key] = {'owner':self._owner(),'worker':None,'phase':'reserved','model_key':model_key,
                'resident_bytes':resident_bytes,'workspace_bytes':workspace_bytes,'growth_bytes':growth_bytes,
                'reserved_bytes':resident_bytes+workspace_bytes+growth_bytes,'headroom_bytes':details['requested_headroom_bytes'],'telemetry':None}
            self._validate({'schema':SCHEMA,'entries':rows})
            return dict(details,registry=self._summary(rows),allocation=dict(rows[key]))

    def launching(self,key):
        with self._locked() as rows:
            row=self._own(rows,key)
            row['phase']='launching'
            return dict(registry=self._summary(rows),allocation=dict(row))

    def bind_worker(self,key,pid,identity,pgid):
        with self._locked() as rows:
            self._own(rows,key)['worker']={'pid':pid,'identity':identity,'pgid':pgid}
            self._validate({'schema':SCHEMA,'entries':rows})
            return dict(registry=self._summary(rows),allocation=dict(rows[key]))

    def settle(self,key,resident_bytes=None,workspace_bytes=0,telemetry=None):
        """Account for materialized arrays without charging them to headroom again."""
        if not _bytes(workspace_bytes) or resident_bytes is not None and not _bytes(resident_bytes):
            raise MemoryAdmissionError('memory_unknown','Invalid resident allocation.')
        if telemetry is not None:
            from .worker_telemetry import validate
            validate(telemetry)
        with self._locked() as rows:
            row=self._own(rows,key)
            row['phase']='resident'
            row['reserved_bytes']=workspace_bytes
            if resident_bytes is not None:row['resident_bytes']=resident_bytes
            if telemetry is not None:row['telemetry']=telemetry
            return dict(registry=self._summary(rows),allocation=dict(row))

    def reserve_workspace(self,key,amount,probe):
        """Reserve the next job's workspace without replacing its warm worker."""
        if not _bytes(amount):raise MemoryAdmissionError('memory_unknown','Invalid workspace estimate.')
        with self._locked() as rows:
            row=self._own(rows,key)
            details=self._admit(rows,key,amount,row['headroom_bytes'],probe)
            row['reserved_bytes']=amount
            return dict(details,registry=self._summary(rows),allocation=dict(row))

    def release(self,key):
        with self._locked() as rows:
            if key in rows:
                self._own(rows,key)
                del rows[key]
            return self._summary(rows)

    def set_headroom(self,key,amount):
        if not _bytes(amount):raise MemoryAdmissionError('memory_unknown','Invalid headroom reserve.')
        with self._locked() as rows:
            row=self._own(rows,key)
            row['headroom_bytes']=amount
            return dict(registry=self._summary(rows),allocation=dict(row))

    def _agent_budget(self, rows, replacement=None):
        from .memory_budget import read_policy, process_snapshot, account
        policy = read_policy(self.root)
        if policy is None:
            return None
        try:
            processes = (self.process_probe or process_snapshot)()
        except (OSError, ValueError, SubprocessError) as error:
            raise MemoryAdmissionError('memory_unknown', 'Agent process memory is unavailable.') from error
        return account(policy, rows, processes, self._owner(), replacement)

    def configure_budget(self, limit_bytes, reserve_bytes=0, external_pids=None):
        from .memory_budget import SCHEMA as BUDGET_SCHEMA, validate_policy, process_snapshot, read_policy
        with self._locked() as rows:
            processes = (self.process_probe or process_snapshot)()
            previous = read_policy(self.root)
            external = list(previous['external']) if previous is not None and external_pids is None else []
            for pid in sorted(set(external_pids or ())):
                if type(pid) is not int or pid <= 0 or pid not in processes:
                    raise MemoryAdmissionError('memory_unknown', 'An external agent process cannot be inspected.')
                row = processes[pid]
                external.append({'pid': pid, 'identity': row['identity'], 'group': row['group']})
            policy = {'schema': BUDGET_SCHEMA, 'limit_bytes': limit_bytes,
                      'reserve_bytes': reserve_bytes, 'external': external}
            validate_policy(policy)
            from .memory_budget import account
            summary = account(policy, rows, processes, self._owner())
            _atomic_in_directory(self.root, 'budget.json', json.dumps(policy).encode(), 0o600)
            return summary

    def reserve_execution(self, key):
        """Reserve active workspace before dispatch, so other loads cannot spend it."""
        with self._locked() as rows:
            row = self._own(rows, key)
            budget = self._agent_budget(rows, (key, 0))
            if budget is None:
                return None
            free = budget['admission_bytes'] - budget['accounted_bytes']
            if free < 0:
                raise MemoryAdmissionError('agent_memory_budget_exceeded',
                    'Unload an idle model before starting this model job.')
            current = budget['allocations'][key]
            capacity = current['resident_bytes'] + current['pending_bytes'] + free
            if row['phase'] == 'resident':
                row['reserved_bytes'] = current['pending_bytes'] + free - row['growth_bytes']
            else:
                row['reserved_bytes'] = capacity
            return capacity

    def snapshot(self):
        with self._locked() as rows:
            result = self._summary(rows)
            try:
                budget = self._agent_budget(rows)
            except MemoryAdmissionError as error:
                if error.code != 'memory_unknown':
                    raise
                budget = {'status': 'unavailable', 'error': str(error), 'within_budget': False}
            if budget is not None:
                result['agent_budget'] = budget
            return result
