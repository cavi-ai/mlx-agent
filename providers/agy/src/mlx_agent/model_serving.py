"""Host-wide model serving identity and duplicate policy."""
import json
import os
import stat
import subprocess
import time
import re
from pathlib import Path

from .service_registry import registry_root
from .transactions import _atomic_in_directory, _assert_safe_directory


def state_root():
    return registry_root().parent / 'model-serving'


def resolve_model(selector, hf_cache=None):
    """Resolve a local model directory or an already-cached repository."""
    from .convert import cached_snapshot
    if not isinstance(selector, str) or not selector or '*' in selector:
        raise ValueError('An exact model path or cached repository is required.')
    path = Path(selector).expanduser()
    if not path.exists():
        path = cached_snapshot(selector, hf_cache)
    if path is None:
        raise ValueError('The local model identity is unavailable.')
    path = path.resolve(strict=True)
    if path.is_file() and path.suffix.lower() == '.gguf':
        from .gguf import read_gguf_header
        read_gguf_header(path)
        return str(path)
    if not path.is_dir():
        raise ValueError('The local model identity is unavailable.')
    from .strata_runtime import MODEL_MARKER, read_model
    if (path / MODEL_MARKER).exists():
        return str(Path(read_model(path)['gguf']).resolve(strict=True))
    if not (path / 'config.json').exists():
        ggufs = sorted(item for item in path.glob('*.gguf') if 'mmproj' not in item.name.lower())
        if ggufs:
            from .splash_runtime import read_target
            read_target(path)
            return str(ggufs[0].resolve(strict=True))
    return str(path)



def read_policy(root=None):
    root = Path(root) if root is not None else state_root()
    path = root / 'policy.json'
    if not os.path.lexists(path):
        return {'schema_version': 1, 'duplicate_model_allowlist': []}
    _state_dir(root)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > 65536:
            raise ValueError('Invalid model serving policy file.')
        value = json.loads(stream.read(65537))
    if isinstance(value, dict):
        value.setdefault('duplicate_model_allowlist', [])
    if (not isinstance(value, dict) or set(value) - {'schema_version', 'duplicate_model_allowlist'}
            or type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or not isinstance(value.get('duplicate_model_allowlist', []), list)
            or len(value['duplicate_model_allowlist']) > 256
            or any(not isinstance(item, str) or not Path(item).is_absolute() or '*' in item for item in value['duplicate_model_allowlist'])
            or len(set(value['duplicate_model_allowlist'])) != len(value['duplicate_model_allowlist'])):
        raise ValueError('Invalid model serving policy.')
    value.setdefault('duplicate_model_allowlist', [])
    return value


def write_policy(selectors, root=None):
    if not isinstance(selectors, list) or len(selectors) > 256:
        raise ValueError('Policy permits at most 256 exact model selectors.')
    models = sorted(set(resolve_model(item) for item in selectors))
    value = {'schema_version': 1, 'duplicate_model_allowlist': models}
    root = Path(root) if root is not None else state_root()
    encoded = (json.dumps(value, sort_keys=True) + '\n').encode()
    if len(encoded) > 65536:
        raise ValueError('Model serving policy exceeds 64 KiB.')
    _state_dir(root)
    _atomic_in_directory(root, 'policy.json', encoded, 0o600)
    return value


class ModelServingError(RuntimeError):
    def __init__(self, code, message, remediation, data=None):
        super().__init__(message)
        self.code = code
        self.remediation = remediation
        self.data = data or {}


def _state_dir(root):
    _assert_safe_directory(root, create=True)
    info = root.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Model serving state must be a private owned directory.')


def _unknown():
    return ModelServingError('model_serving_state_unknown', 'Model ownership cannot be established.',
                             'Inspect the model service and its owned processes before retrying.')


def _lock(root, key):
    import contextlib
    import fcntl

    @contextlib.contextmanager
    def locked():
        _state_dir(root)
        fd = os.open(root / (key + '.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
                raise ValueError('Invalid model ownership lock.')
            deadline = time.monotonic() + 5
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise _unknown()
                    time.sleep(0.01)
            yield
        finally:
            os.close(fd)
    return locked()


def _read_claims(root, key):
    path = root / (key + '.json')
    if not os.path.lexists(path):
        return []
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > 262144:
            raise ValueError('Invalid model ownership record.')
        rows = json.loads(stream.read(262145))
    if not isinstance(rows, list) or len(rows) > 256:
        raise ValueError('Invalid model ownership record.')
    for row in rows:
        if (not isinstance(row, dict) or set(row) - {'generation', 'pid', 'birth', 'state', 'kind',
                'service', 'pgid', 'jit', 'ready', 'state_path', 'receipt_path', 'command_sha256'}
                or not isinstance(row.get('generation'), str) or not re.fullmatch('[0-9a-f]{64}', row['generation'])
                or type(row.get('pid')) is not int or row['pid'] <= 0
                or not isinstance(row.get('birth'), str) or not row['birth'] or len(row['birth']) > 256
                or row.get('state') not in ('reserved', 'launching', 'active')
                or row.get('kind') not in ('service', 'execution')
                or ('pgid' in row and (type(row['pgid']) is not int or row['pgid'] <= 0))
                or ('jit' in row and type(row['jit']) is not bool)
                or ('ready' in row and type(row['ready']) is not bool)
                or ('state_path' in row and row['state_path'] is not None and
                    (not isinstance(row['state_path'], str) or not Path(row['state_path']).is_absolute()))):
            raise ValueError('Invalid model ownership record.')
        if row['kind'] == 'service':
            _service_info(row, verify=False)
        elif row.get('service') is not None:
            raise ValueError('Invalid model ownership record.')
    return rows


def _write_claims(root, key, rows):
    encoded = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    if len(encoded) > 262144 or len(rows) > 256:
        raise ValueError('Too many model ownership records.')
    _atomic_in_directory(root, key + '.json', encoded, 0o600)


def _live(row):
    from .compute import process_alive, process_birth_identity, group_members
    pid, birth, state = row['pid'], row['birth'], row['state']
    if type(pid) is not int or not isinstance(birth, str) or not birth or state not in ('reserved', 'launching', 'active'):
        raise ValueError('Invalid model ownership record.')
    actual = process_birth_identity(pid)
    if actual == birth:
        return True
    if actual is None and process_alive(pid):
        raise _unknown()
    if state == 'reserved':
        return False
    if state == 'launching':
        raise _unknown()
    if row.get('jit'):
        _check_stale_worker(row)
    pgid = row.get('pgid')
    if type(pgid) is not int or pgid <= 0:
        raise _unknown()
    if group_members(pgid):
        raise _unknown()
    return False


def _check_stale_worker(row):
    from .compute import process_alive, process_birth_identity, group_members
    path = row.get('state_path')
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise _unknown()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > 65536:
            raise _unknown()
        state = json.loads(stream.read(65537))
    if (not isinstance(state, dict) or state.get('gateway_pid') != row['pid']
            or state.get('worker_launch_pending') is not False):
        raise _unknown()
    worker = state.get('worker_pid')
    if worker is None:
        return
    birth = state.get('worker_started_at')
    if type(worker) is not int or worker <= 0 or not isinstance(birth, str) or not birth:
        raise _unknown()
    actual = process_birth_identity(worker)
    if actual == birth or actual is None and process_alive(worker):
        raise _unknown()
    if group_members(worker):
        raise _unknown()


class Claim:
    def __init__(self, root, key, generation):
        self.root, self.key, self.generation = root, key, generation

    def update(self, **changes):
        with _lock(self.root, self.key):
            rows = _read_claims(self.root, self.key)
            matches = [row for row in rows if row.get('generation') == self.generation]
            if len(matches) != 1:
                raise _unknown()
            matches[0].update(changes)
            _write_claims(self.root, self.key, rows)

    def cancel_reserved(self):
        from .compute import process_birth_identity
        with _lock(self.root, self.key):
            rows = _read_claims(self.root, self.key)
            matches = [row for row in rows if row.get('generation') == self.generation]
            if len(matches) != 1:
                raise _unknown()
            row = matches[0]
            if row['state'] != 'reserved':
                return False
            if row['pid'] != os.getpid() or row['birth'] != process_birth_identity(os.getpid()):
                raise _unknown()
            _write_claims(self.root, self.key, [item for item in rows if item is not row])
            return True

    def cancel_unstarted_gateway(self):
        from .compute import process_birth_identity
        with _lock(self.root, self.key):
            rows = _read_claims(self.root, self.key)
            matches = [row for row in rows if row.get('generation') == self.generation]
            if len(matches) != 1:
                raise _unknown()
            row = matches[0]
            if (row['state'] != 'active' or not row.get('jit') or row.get('ready')
                    or row['pid'] != os.getpid() or row['birth'] != process_birth_identity(os.getpid())):
                raise _unknown()
            _write_claims(self.root, self.key, [item for item in rows if item is not row])

    def cancel_failed_exec(self):
        from .compute import process_birth_identity
        with _lock(self.root, self.key):
            rows = _read_claims(self.root, self.key)
            matches = [row for row in rows if row.get('generation') == self.generation]
            if len(matches) != 1:
                raise _unknown()
            row = matches[0]
            if (row['state'] != 'active' or row.get('kind') != 'service' or row.get('jit')
                    or row.get('ready') or row['pid'] != os.getpid()
                    or row['birth'] != process_birth_identity(os.getpid())):
                raise _unknown()
            _write_claims(self.root, self.key, [item for item in rows if item is not row])

    def complete_execution(self, pgid, birth):
        from .compute import group_members, process_birth_identity
        if type(pgid) is not int or pgid <= 0 or group_members(pgid):
            raise _unknown()
        if birth and process_birth_identity(pgid) == birth:
            raise _unknown()
        with _lock(self.root, self.key):
            rows = _read_claims(self.root, self.key)
            matches = [row for row in rows if row.get('generation') == self.generation]
            if len(matches) != 1:
                raise _unknown()
            row = matches[0]
            if row.get('kind') != 'execution':
                raise _unknown()
            if row['state'] == 'launching':
                if row['pid'] != os.getpid() or row['birth'] != process_birth_identity(os.getpid()):
                    raise _unknown()
            elif row['state'] != 'active' or row['pid'] != pgid or row['birth'] != birth or row.get('pgid') != pgid:
                raise _unknown()
            _write_claims(self.root, self.key, [item for item in rows if item is not row])

    def cancel_failed_spawn(self):
        from .compute import process_birth_identity
        with _lock(self.root, self.key):
            rows = _read_claims(self.root, self.key)
            matches = [row for row in rows if row.get('generation') == self.generation]
            if len(matches) != 1:
                raise _unknown()
            row = matches[0]
            if row['state'] != 'launching' or row['pid'] != os.getpid() or row['birth'] != process_birth_identity(os.getpid()):
                raise _unknown()
            _write_claims(self.root, self.key, [item for item in rows if item is not row])

    def release(self):
        from .compute import process_birth_identity
        with _lock(self.root, self.key):
            rows = _read_claims(self.root, self.key)
            matches = [row for row in rows if row.get('generation') == self.generation]
            if len(matches) != 1:
                raise _unknown()
            row = matches[0]
            owned_here = row['pid'] == os.getpid() and row['birth'] == process_birth_identity(os.getpid())
            if row['state'] == 'reserved' and owned_here:
                pass
            elif row['state'] == 'active' and owned_here and row.get('jit'):
                _check_stale_worker(row)
            elif row['state'] == 'active' and not _live(row):
                pass
            else:
                raise _unknown()
            _write_claims(self.root, self.key, [item for item in rows if item is not row])


def _legacy_matches(model, services):
    from .serve import _read_receipt, _jit_config, receipts_root
    matches = []
    for item in services:
        receipt = _read_receipt(Path(item['receipt']))
        if receipt is None:
            raise _unknown()
        root = receipts_root(item['receipts_dir'])
        if not item.get('alive') and not receipt.get('jit'):
            from .compute import process_alive, process_birth_identity, group_members
            try:
                pid = receipt['pid']
                if type(pid) is not int or pid <= 0 or process_alive(pid) or process_birth_identity(pid) is not None or group_members(pid):
                    raise _unknown()
            except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
                raise _unknown() from None
            continue
        try:
            selector = (_jit_config(receipt, root)['local_path'] if receipt.get('jit') else
                        receipt.get('model_artifact') or receipt.get('splash_model_dir') or receipt.get('path') or receipt.get('repo'))
            if resolve_model(selector) != model:
                continue
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError):
            raise _unknown() from None
        if not item.get('alive'):
            # Matching legacy JIT state has no pre-spawn marker.
            raise _unknown()
        if not item.get('argv_match') or not receipt.get('process_started_at') or item.get('process_started_at') != receipt['process_started_at']:
            raise _unknown()
        from .endpoint_contract import descriptor
        matches.append(descriptor(item['port'], item['runtime'], item['service_id'], 'unknown',
                                  bool(item.get('api_key_sha256'))))
    return sorted(matches, key=lambda item: item['service_id'])[:256]




def _verify_ready_service(row):
    import hashlib
    from .compute import process_birth_identity, process_identity
    from .serve import _read_receipt, _pid_command, _argv_matches
    if process_birth_identity(row['pid']) != row['birth']:
        raise _unknown()
    path = row.get('receipt_path')
    if path:
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise _unknown()
        receipt = _read_receipt(Path(path))
        if (not receipt or receipt.get('pid') != row['pid'] or receipt.get('port') != row['service']['port']
                or receipt.get('runtime') != row['service']['runtime']
                or receipt.get('process_started_at') != row['birth']
                or not _argv_matches(receipt, _pid_command(row['pid']))):
            raise _unknown()
    else:
        command = process_identity(row['pid'])
        expected = row.get('command_sha256')
        if not command or not isinstance(expected, str) or hashlib.sha256(command.encode()).hexdigest() != expected:
            raise _unknown()

def _service_info(row, verify=True):
    import hashlib
    service = row.get('service')
    if not isinstance(service, dict) or type(service.get('port')) is not int or not 1 <= service['port'] <= 65535:
        raise _unknown()
    if service.get('runtime') not in ('tasks', 'mlx_lm', 'mlx-vlm', 'strata', 'splash') or type(service.get('auth_required')) is not bool:
        raise _unknown()
    if verify and row.get('ready'):
        _verify_ready_service(row)
    from .endpoint_contract import descriptor
    return descriptor(service['port'], service['runtime'], hashlib.sha256(row['generation'].encode()).hexdigest(),
                      'ready' if row.get('ready') else 'starting', service['auth_required'])


def receipt_service_id(item):
    """Identify a verified receipt's current claim, or its legacy process generation."""
    import hashlib
    from .serve import _read_receipt, _jit_config, receipts_root
    receipt = _read_receipt(Path(item['receipt']))
    if (not receipt or not item.get('alive') or not item.get('argv_match')
            or not receipt.get('process_started_at')
            or item.get('pid') != receipt.get('pid') or item.get('port') != receipt.get('port')
            or item.get('runtime') != receipt.get('runtime')
            or receipt['process_started_at'] != item.get('process_started_at')):
        return None
    selector = (_jit_config(receipt, receipts_root(item['receipts_dir']))['local_path']
                if receipt.get('jit') else receipt.get('model_artifact') or receipt.get('splash_model_dir')
                or receipt.get('path') or receipt.get('repo'))
    model = resolve_model(selector)
    root = state_root()
    key = hashlib.sha256(model.encode()).hexdigest()
    if os.path.lexists(root):
        _assert_safe_directory(root)
    rows = _read_claims(root, key)
    matches = [row for row in rows if row.get('kind') == 'service'
               and row.get('pid') == receipt['pid'] and row.get('birth') == receipt['process_started_at']]
    if len(matches) > 1:
        raise _unknown()
    if matches:
        if not matches[0].get('ready') or matches[0].get('receipt_path') != item['receipt']:
            raise _unknown()
        return _service_info(matches[0])['service_id']
    identity = json.dumps([str(Path(item['receipt']).resolve()), receipt['pid'], receipt['process_started_at']], separators=(',', ':'))
    return hashlib.sha256(identity.encode()).hexdigest()


class _OwnedClaim:
    def update(self, **changes):
        pass

    def cancel_reserved(self):
        pass

    def release(self):
        pass

    def cancel_failed_spawn(self):
        pass

    def complete_execution(self, pgid, birth):
        pass


def _owned_ancestor(row):
    from .compute import process_birth_identity
    from .process_table import rows
    target = row.get('pid')
    if row.get('kind') != 'service' or row.get('state') != 'active':
        return False
    pid = os.getpid()
    for _ in range(64):
        if pid == target:
            return process_birth_identity(pid) == row.get('birth')
        item = rows([pid]).get(pid)
        if not item or item[0] <= 1 or item[0] == pid:
            return False
        pid = item[0]
    raise _unknown()

def acquire(selector, *, kind='execution', service=None, root=None, discover=None, owner_claim=None):
    import hashlib
    import secrets
    from .compute import process_birth_identity
    from .service_registry import discover_services
    try:
        model = resolve_model(selector)
        root = Path(root) if root is not None else state_root()
        key = hashlib.sha256(model.encode()).hexdigest()
        birth = process_birth_identity(os.getpid())
        if not birth:
            raise _unknown()
        with _lock(root, key):
            policy = read_policy(root)
            rows = [row for row in _read_claims(root, key) if _live(row)]
            if (kind == 'execution' and isinstance(owner_claim, Claim) and owner_claim.root == root
                    and owner_claim.key == key and any(row.get('generation') == owner_claim.generation
                    and _owned_ancestor(row) for row in rows)):
                return _OwnedClaim()
            if rows and model not in policy['duplicate_model_allowlist']:
                if any(row.get('kind') == 'service' for row in rows):
                    services = [_service_info(row) for row in rows if row.get('kind') == 'service']
                    raise ModelServingError('model_already_served', 'This model already has a service.',
                                            'Check its state and use the compatible endpoint with your authorized key.',
                                            {'existing_services': sorted(services, key=lambda value: value['service_id'])})
                raise ModelServingError('model_load_in_progress', 'This model is already in use.',
                                        'Wait for the existing model operation to finish.')
            found = (discover or discover_services)()
            if found.get('errors'):
                raise _unknown()
            legacy = _legacy_matches(model, found.get('services', []))
            if legacy and model not in policy['duplicate_model_allowlist']:
                raise ModelServingError('model_already_served', 'This model already has a service.',
                                        'Use the existing compatible endpoint with its authorized key.',
                                        {'existing_services': legacy})
            generation = secrets.token_hex(32)
            rows.append({'generation': generation, 'pid': os.getpid(), 'birth': birth,
                         'state': 'reserved', 'kind': kind, 'service': service})
            _write_claims(root, key, rows)
        return Claim(root, key, generation)
    except ModelServingError:
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError):
        raise _unknown() from None


def adopt_gateway(config):
    """Bind a private parent reservation, or claim a foreground gateway."""
    from .compute import process_birth_identity
    model = resolve_model(config['local_path'])
    token = config.get('model_serving_claim')
    if token is None:
        service = {'port': config['port'], 'runtime': config['runtime'],
                   'auth_required': bool(config.get('api_key_sha256') or config.get('access_key_file'))}
        claim = acquire(model, kind='service', service=service)
        birth = process_birth_identity(os.getpid())
        if not birth:
            raise _unknown()
        claim.update(state='active', pid=os.getpid(), birth=birth, pgid=os.getpgrp(), jit=True, ready=False,
                     state_path=config.get('state_path'))
        return claim
    import hashlib
    import re
    if not isinstance(token, dict) or set(token) != {'key', 'generation'} or not all(
            isinstance(token[x], str) and re.fullmatch('[0-9a-f]{64}', token[x]) for x in token):
        raise _unknown()
    if token['key'] != hashlib.sha256(model.encode()).hexdigest():
        raise _unknown()
    claim = Claim(state_root(), token['key'], token['generation'])
    with _lock(claim.root, claim.key):
        rows = _read_claims(claim.root, claim.key)
        matches = [row for row in rows if row.get('generation') == claim.generation]
        if len(matches) != 1:
            raise _unknown()
        row = matches[0]
        parent = os.getppid()
        own = os.getpid()
        if not ((row.get('state') == 'launching' and row.get('pid') == parent
                 and row.get('birth') == process_birth_identity(parent))
                or (row.get('state') == 'active' and row.get('pid') == own
                    and row.get('birth') == process_birth_identity(own))):
            raise _unknown()
        birth = process_birth_identity(own)
        if not birth:
            raise _unknown()
        row.update(state='active', pid=own, birth=birth, pgid=os.getpgrp(), jit=True)
        _write_claims(claim.root, claim.key, rows)
    return claim


def release_stopped_service(receipt):
    """Release only the exact receipt owner after its processes have exited."""
    import hashlib
    from .compute import group_members
    artifact = receipt.get('model_artifact')
    if artifact is None:
        return
    if (not isinstance(artifact, str) or not artifact.startswith('/') or '\x00' in artifact
            or os.path.normpath(artifact) != artifact):
        raise _unknown()
    key = hashlib.sha256(artifact.encode()).hexdigest()
    root = state_root()
    with _lock(root, key):
        rows = _read_claims(root, key)
        matching = [row for row in rows if row.get('kind') == 'service' and row.get('pid') == receipt.get('pid')
                    and row.get('birth') == receipt.get('process_started_at')]
        if not matching:
            return
        if len(matching) != 1:
            raise _unknown()
        row = matching[0]
        if _live(row) or group_members(row['pgid']):
            raise _unknown()
        _write_claims(root, key, [item for item in rows if item is not row])


def public_error_data(data):
    """Validate and copy only bounded endpoint details safe for result envelopes."""
    if not isinstance(data, dict) or set(data) != {'existing_services'} or not isinstance(data['existing_services'], list) or len(data['existing_services']) > 256:
        raise ValueError('Invalid service error details.')
    services = []
    for row in data['existing_services']:
        if not isinstance(row, dict) or set(row) != {'origin', 'api_base', 'service_id', 'runtime', 'state', 'auth_required'}:
            raise ValueError('Invalid service error details.')
        match = re.fullmatch(r'http://127\.0\.0\.1:([0-9]{1,5})', row.get('origin', ''))
        if (not match or not 1 <= int(match.group(1)) <= 65535 or row.get('api_base') != row['origin'] + '/v1'
                or not isinstance(row.get('service_id'), str) or not re.fullmatch('[0-9a-f]{64}', row['service_id'])
                or row.get('runtime') not in ('tasks', 'mlx_lm', 'mlx-vlm', 'strata', 'splash')
                or row.get('state') not in ('ready', 'starting', 'unknown') or type(row.get('auth_required')) is not bool):
            raise ValueError('Invalid service error details.')
        services.append(dict(row))
    return {'existing_services': services}


def mark_foreground_ready(claim):
    import hashlib
    from .compute import process_identity
    command = process_identity(os.getpid())
    if not command:
        raise _unknown()
    claim.update(ready=True, command_sha256=hashlib.sha256(command.encode()).hexdigest())


def verify_private_worker(selector, descriptor):
    """Require an inherited Unix listener and its live owning gateway ancestor."""
    import hashlib
    import socket
    try:
        if type(descriptor) is not int or descriptor < 3 or not os.get_inheritable(descriptor):
            raise _unknown()
        with socket.socket(fileno=os.dup(descriptor)) as listener:
            if listener.family != socket.AF_UNIX or listener.type != socket.SOCK_STREAM:
                raise _unknown()
            name = listener.getsockname()
            if not isinstance(name, str) or not name:
                raise _unknown()
            info = os.lstat(name)
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise _unknown()
            try:
                if listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) != 1:
                    raise _unknown()
            except OSError as error:
                import errno
                if error.errno != errno.ENOPROTOOPT:
                    raise
        model = resolve_model(selector)
        root = state_root()
        key = hashlib.sha256(model.encode()).hexdigest()
        with _lock(root, key):
            rows = _read_claims(root, key)
            if not any(row.get('jit') and row.get('pid') != os.getpid() and _live(row)
                       and _owned_ancestor(row) for row in rows):
                raise _unknown()
    except ModelServingError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError):
        raise _unknown() from None
