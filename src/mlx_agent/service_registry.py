"""Bounded discovery of engine-owned services across receipt locations."""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

from .transactions import (
    _assert_safe_directory,
    _atomic_in_directory,
    _physical_absolute,
    _target_locks,
    _walk_directory,
)

MAX_ROOTS = 32
MAX_INDEX_FILES = 256
MAX_RECORD_BYTES = 16_384
_LIVE_GATEWAY_SCRIPTS = ('/mlx_agent/jit_serve.py', '/mlx_agent/task_serve.py')


def _is_stale_root_error(error, target_path):
    message = str(error)
    if message.startswith('path component does not exist:'):
        return True
    cause = getattr(error, '__cause__', None)
    if isinstance(cause, OSError) and cause.errno == errno.ENOTDIR:
        try:
            physical = _physical_absolute(Path(target_path).expanduser().absolute())
        except ValueError:
            return False
        current = Path('/')
        for part in physical.parts[1:]:
            current = current / part
            try:
                st = os.lstat(current)
            except FileNotFoundError:
                return True
            except OSError:
                return False
            if stat.S_ISLNK(st.st_mode):
                return False
            if not stat.S_ISDIR(st.st_mode):
                return True
    return False


def registry_root():
    return Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local/state') / 'mlx-agent/services'


def _safe_root(value):
    return _assert_safe_directory(Path(value).expanduser().absolute()).resolve()


def _read_registration(location, dir_fd=None):
    descriptor = os.open(location, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    with os.fdopen(descriptor, 'rb') as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_size > MAX_RECORD_BYTES:
            raise ValueError('Invalid service location record')
        raw = source.read(MAX_RECORD_BYTES + 1)
        if len(raw) > MAX_RECORD_BYTES:
            raise ValueError('Service location record is too large')
        record = json.loads(raw)
    if not isinstance(record, dict) or record.get('schema_version') != 1 or not isinstance(record.get('receipts_dir'), str):
        raise ValueError('Unsupported service location record')
    return record, metadata


def _index_names(directory):
    """Read at most one bounded page, including unrelated directory entries."""
    names = []
    overflow = False
    with os.scandir(directory) as entries:
        for entry in entries:
            if len(names) == MAX_INDEX_FILES:
                overflow = True
                break
            names.append(entry.name)
    return sorted(name for name in names if name.endswith('.json')), overflow


def _prune_stale_registrations(index, keep_name=None):
    """Remove only unchanged owned records whose receipt roots have disappeared."""
    _, directory = _walk_directory(index)
    unlinked = False
    try:
        names, _overflow = _index_names(directory)
        for name in names:
            if name == keep_name:
                continue
            try:
                record, metadata = _read_registration(name, dir_fd=directory)
                try:
                    _safe_root(record['receipts_dir'])
                    continue
                except ValueError as error:
                    if not _is_stale_root_error(error, record['receipts_dir']):
                        continue
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if (not stat.S_ISREG(current.st_mode) or current.st_uid != os.getuid()
                        or (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino)):
                    continue
                os.unlink(name, dir_fd=directory)
                unlinked = True
            except (OSError, ValueError, TypeError):
                continue
        if unlinked:
            try:
                os.fsync(directory)
            except OSError:
                pass
    finally:
        os.close(directory)


def prune_stale_registrations(registry=None):
    """Bounded cleanup serialized with managed registrations."""
    index = Path(registry) if registry is not None else registry_root()
    if not os.path.lexists(index):
        return
    with _target_locks([index], blocking=True):
        _assert_safe_directory(index)
        _prune_stale_registrations(index)


def register_receipts(receipts_dir, registry=None):
    """Index one explicit location under the shared registry lock."""
    root = _safe_root(receipts_dir)
    index = Path(registry) if registry is not None else registry_root()
    name = hashlib.sha256(str(root).encode()).hexdigest() + '.json'
    with _target_locks([index], create_parents=True, blocking=True):
        _assert_safe_directory(index, create=True)
        _atomic_in_directory(index, name, json.dumps({
            'schema_version': 1, 'receipts_dir': str(root),
        }, sort_keys=True).encode(), 0o600)
        _prune_stale_registrations(index, keep_name=name)


def _read_roots(registry, errors):
    index = Path(registry) if registry is not None else registry_root()
    if not index.exists() and not index.is_symlink():
        return []
    _, directory = _walk_directory(index)
    roots, seen = [], set()
    try:
        names, limit_reached = _index_names(directory)
        for name in names:
            location = index / name
            try:
                try:
                    record, _metadata = _read_registration(name, dir_fd=directory)
                except FileNotFoundError:
                    continue
                try:
                    resolved = str(_safe_root(record['receipts_dir']))
                except ValueError as error:
                    if not _is_stale_root_error(error, record['receipts_dir']):
                        raise
                    continue
                if resolved in seen:
                    continue
                if len(roots) >= MAX_ROOTS:
                    limit_reached = True
                    break
                seen.add(resolved)
                roots.append(resolved)
            except (OSError, ValueError, TypeError) as error:
                errors.append({'source': str(location), 'message': str(error)})
    finally:
        os.close(directory)
    if limit_reached:
        errors.append({'source': str(index), 'message': 'Service location limit reached.'})
    return roots


def registered_roots(registry=None):
    return _read_roots(registry, [])


def _process_arguments(pid):
    try:
        if sys.platform != 'darwin':
            return [value.decode(errors='replace') for value in Path('/proc/{}/cmdline'.format(pid)).read_bytes().split(b'\0') if value]
        libc = ctypes.CDLL(None, use_errno=True)
        mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2
        size = ctypes.c_size_t()
        if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) or not 4 < size.value <= 1_048_576:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0):
            return None
        raw = buffer.raw[:size.value]
        count = int.from_bytes(raw[:4], sys.byteorder, signed=True)
        if not 0 < count <= 4096:
            return None
        offset = raw.index(b'\0', 4)
        while offset < len(raw) and raw[offset] == 0:
            offset += 1
        arguments = []
        for _ in range(count):
            end = raw.index(b'\0', offset)
            arguments.append(raw[offset:end].decode(errors='replace'))
            offset = end + 1
        return arguments
    except (OSError, ValueError):
        return None


def live_receipt_roots(process_rows=None, errors=None):
    """Locate legacy JIT receipts from explicit live gateway arguments, never a walk."""
    live = process_rows is None
    if live:
        try:
            result = subprocess.run(['ps', '-ww', '-axo', 'pid=,uid=,command='], capture_output=True,
                                    text=True, timeout=2, check=True)
            process_rows = result.stdout[:2_000_000]
        except (OSError, subprocess.SubprocessError) as error:
            if errors is not None:
                errors.append({'source': 'local processes', 'message': str(error)})
            return []
    roots = []
    for row in process_rows.splitlines()[:65_536]:
        fields = row.strip().split(None, 2)
        if len(fields) != 3 or fields[1] != str(os.getuid()):
            continue
        if not any(script in fields[2] for script in _LIVE_GATEWAY_SCRIPTS):
            continue
        try:
            args = _process_arguments(int(fields[0])) if live else shlex.split(fields[2])
            if not args:
                if errors is not None:
                    errors.append({'source': 'local processes', 'message': 'A gateway argument list could not be read.'})
                continue
            if not any(value.endswith(_LIVE_GATEWAY_SCRIPTS) for value in args[:4]) or '--config' not in args:
                continue
            config = Path(args[args.index('--config') + 1])
            if not config.is_absolute() or config.parent.name != 'serve' or config.parent.parent.name != '.mlx-agent-receipts':
                continue
            if not config.name.endswith('.jit-config.json'):
                continue
            root = str(config.parent.parent.parent)
            if root not in roots:
                roots.append(root)
            if len(roots) >= MAX_ROOTS:
                break
        except (ValueError, IndexError):
            continue
    return roots


def discover_services(receipts_dirs=(), registry=None, process_roots=None, inspect=None):
    """Read authoritative receipts without adopting or changing any service."""
    from .serve import status_serve
    errors = []
    try:
        indexed = _read_roots(registry, errors)
    except (OSError, ValueError) as error:
        indexed = []
        errors.append({'source': str(registry or registry_root()), 'message': str(error)})
    roots = list(dict.fromkeys([str(value) for value in receipts_dirs] + indexed +
                              (live_receipt_roots(errors=errors) if process_roots is None else list(process_roots))))
    if len(roots) > MAX_ROOTS:
        errors.append({'source': 'discovery', 'message': 'Service location limit reached; narrow registered locations.'})
    services = []
    seen = set()
    deadline = time.monotonic() + 8
    for value in roots[:MAX_ROOTS]:
        try:
            root = str(_safe_root(value))
            if root in seen:
                continue
            seen.add(root)
            if time.monotonic() >= deadline:
                raise TimeoutError('Service discovery deadline reached')
            if inspect is None:
                receipts = Path(root) / '.mlx-agent-receipts/serve'
                if receipts.exists() or receipts.is_symlink():
                    _assert_safe_directory(receipts)
            entries = inspect(root) if inspect else status_serve(root, deadline=deadline, errors=errors)
            for entry in entries:
                receipt = Path(entry['receipt'])
                if receipt.parent.resolve() != Path(root) / '.mlx-agent-receipts/serve':
                    raise ValueError('Receipt is outside its service location')
                item = dict(entry, receipts_dir=root)
                from .model_serving import resolve_receipt_endpoint
                try:
                    endpoint = resolve_receipt_endpoint(item)
                except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError):
                    endpoint = None
                item['service_id'] = endpoint['service_id'] if endpoint else None
                if endpoint:
                    item['endpoint'] = endpoint
                services.append(item)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({'source': str(value), 'message': str(error)})
    return {'schema_version': 1, 'services': services, 'errors': errors,
            'capabilities': {'scoped_doctor': True, 'scoped_watch': True, 'model_update_preparation': True, 'model_storage_admission': True, 'expected_pid_stop': True}}
