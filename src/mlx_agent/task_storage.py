"""Opt-in task storage accounting and explicit media expiration."""
import json
import hashlib
import os
import re
import stat
import time
from pathlib import Path

from .task_contracts import TERMINAL, TaskError, encoded

MAX_TASKS = 10000
MAX_FILES = 100000
INVENTORY_SECONDS = 5


def _tables(connection):
    connection.execute('CREATE TABLE IF NOT EXISTS storage_policy (id INTEGER PRIMARY KEY CHECK (id = 1), budget INTEGER NOT NULL)')
    connection.execute('CREATE TABLE IF NOT EXISTS media_expiry (id TEXT PRIMARY KEY, state TEXT NOT NULL, manifest TEXT NOT NULL)')


def _exists(connection, table):
    return bool(connection and connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone())


def set_budget(store, budget):
    if type(budget) is not int or not 0 <= budget < 2**63:
        raise TaskError('invalid_budget', 'Budget must be a nonnegative number of bytes; zero disables it.')
    with store._connection(write=True) as connection:
        _tables(connection)
        connection.execute('INSERT OR REPLACE INTO storage_policy VALUES (1, ?)', (budget,))
    return {'budget_bytes': budget or None}


def _budget(connection):
    if not _exists(connection, 'storage_policy'):
        return None
    row = connection.execute('SELECT budget FROM storage_policy WHERE id = 1').fetchone()
    return row[0] if row and row[0] else None


def _inventory(store, connection):
    from .task_store import ensure_private
    from .task_contracts import task_id
    deadline = time.monotonic() + INVENTORY_SECONDS
    used = files = tasks = 0
    cursor = connection.execute('SELECT id FROM tasks ORDER BY id')
    while rows := cursor.fetchmany(100):
        for (identifier,) in rows:
            task_id(identifier)
            tasks += 1
            if tasks > MAX_TASKS or time.monotonic() > deadline:
                raise TaskError('storage_inventory_incomplete', 'The bounded task storage inventory did not finish.')
            directory = store.root / 'artifacts' / identifier
            if not directory.exists() and not directory.is_symlink():
                continue
            ensure_private(directory)
            with os.scandir(directory) as entries:
                for entry in entries:
                    files += 1
                    if files > MAX_FILES or time.monotonic() > deadline:
                        raise TaskError('storage_inventory_incomplete', 'The bounded task storage inventory did not finish.')
                    info = entry.stat(follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                        raise TaskError('unsafe_task_storage', 'Task inventory found an unsafe artifact.')
                    used += info.st_size
    return used


def check_submission(store, connection, staged_bytes, request_bytes):
    budget = _budget(connection)
    if budget is None:
        return
    used = _inventory(store, connection)
    if used + staged_bytes + request_bytes > budget:
        raise TaskError('storage_budget_exceeded', 'Task storage exceeds the configured submission high-water mark.')


def status(store):
    with store._connection() as connection:
        budget = _budget(connection)
        used = _inventory(store, connection) if budget is not None else None
    return {'budget_bytes': budget, 'retained_bytes': used,
            'description': 'Submission high-water mark; active jobs can grow after the check.'}


def expiry_status(store, identifier):
    with store._connection() as connection:
        if not _exists(connection, 'media_expiry'):
            return None
        row = connection.execute('SELECT state FROM media_expiry WHERE id = ?', (identifier,)).fetchone()
        return row[0] if row else None


def provenance(store, identifier):
    """Read preserved task and result references without opening expired payloads."""
    record = store.get(identifier)
    result = None
    if record['result'] is not None:
        path = store.verify_artifact(identifier, record['result'], expected_name='result.json', code='result')
        if record['result']['bytes'] > 1024 * 1024:
            raise TaskError('result_too_large', 'The saved result is too large for a metadata response.')
        try:
            value = json.loads(path.read_bytes())
            result = value.get('output_artifact')
        except (OSError, ValueError, AttributeError) as error:
            raise TaskError('result_unavailable', 'The saved result metadata is unavailable.') from error
    return {'id': identifier, 'request': record['request'], 'result': record['result'],
            'input_artifact': record.get('input_artifact'), 'input_artifacts': record.get('input_artifacts'),
            'output_artifact': result, 'media_expiration': expiry_status(store, identifier)}


def preview(store, identifier):
    record = store.get(identifier)
    if record.get('state') not in TERMINAL or record.get('execution_unresolved'):
        raise TaskError('task_not_settled', 'Only terminal tasks with settled execution can expire media.')
    existing = expiry_status(store, identifier)
    if existing:
        with store._connection() as connection:
            row = connection.execute('SELECT manifest FROM media_expiry WHERE id = ?', (identifier,)).fetchone()
        manifest = json.loads(row[0])
        return {'id': identifier, 'state': existing, 'artifacts': manifest,
                'confirmation': hashlib.sha256(encoded(manifest)).hexdigest()}
    from . import task_kinds
    from .task_runtime import task_result
    kind = task_kinds.get(record['kind'])
    refs = []
    if record.get('schema') == 'task-run/3':
        request = store.read_request(identifier)
        paths = store.verify_named_inputs(identifier, request, record['input_artifacts'])
        for value in paths.values():
            refs.extend(value if isinstance(value, list) else [value])
    elif kind.input_media:
        refs.append(str(store.verify_input_artifact(identifier, record['input_artifact'], kind.input_media)))
    if record['state'] == 'succeeded' and kind.output:
        task_result(identifier, store.root)
        refs.append(str(store.artifact_path(identifier, kind.output)))
    if not refs:
        raise TaskError('no_media_artifacts', 'This task has no owned media eligible for expiration.')
    manifest = []
    for name in refs:
        path = Path(name)
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise TaskError('unsafe_task_storage', 'A media artifact is no longer a regular file.')
        manifest.append({'name': path.name, 'bytes': info.st_size, 'device': info.st_dev, 'inode': info.st_ino,
                         'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns})
    return {'id': identifier, 'state': 'preview', 'artifacts': manifest,
            'confirmation': hashlib.sha256(encoded(manifest)).hexdigest()}


def expire(store, identifier, confirmation):
    try:
        selected = preview(store, identifier)
    except TaskError:
        if expiry_status(store, identifier) is None:
            raise
        selected = preview(store, identifier)
    if confirmation != selected['confirmation']:
        raise TaskError('expiration_changed', 'The selected media changed; review a new preview.')
    if selected['state'] == 'expired':
        return selected
    if selected['state'] == 'preview':
        with store._connection(write=True) as connection:
            record = store._load(connection, identifier)
            if record['state'] not in TERMINAL or record['execution_unresolved']:
                raise TaskError('task_not_settled', 'Only terminal tasks with settled execution can expire media.')
            _tables(connection)
            connection.execute('INSERT OR IGNORE INTO media_expiry VALUES (?, ?, ?)', (identifier, 'pending', encoded(selected['artifacts']).decode()))
            state, manifest = connection.execute('SELECT state, manifest FROM media_expiry WHERE id = ?', (identifier,)).fetchone()
            if hashlib.sha256(encoded(json.loads(manifest))).hexdigest() != confirmation:
                raise TaskError('expiration_changed', 'The selected media changed; review a new preview.')
            if state == 'expired':
                return dict(selected, state='expired')
    directory = store.artifact_path(identifier, 'request.json').parent
    try:
        directory_fd = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0))
    except OSError as error:
        raise TaskError('media_expiration_incomplete', 'Media expiration was interrupted; retry with the same confirmation.') from error
    try:
        from . import task_kinds
        from .task_contracts import MEDIA_SUFFIXES
        kind = task_kinds.get(store.get(identifier)['kind'])
        for item in selected['artifacts']:
            name = item['name']
            path = Path(name)
            is_input = (path.suffix in {suffix for values in MEDIA_SUFFIXES.values() for suffix in values}
                        and re.fullmatch(r'input(?:-[0-9]+)?', path.stem))
            if path.name != name or not (is_input or name == kind.output):
                raise TaskError('invalid_artifact', 'The expiration manifest is invalid.')
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            expected = (item['device'], item['inode'], item['bytes'], item['mtime_ns'], item['ctime_ns'])
            if not stat.S_ISREG(info.st_mode) or identity != expected:
                raise TaskError('artifact_changed', 'A selected artifact changed during expiration.')
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.fsync(directory_fd)
    except OSError as error:
        raise TaskError('media_expiration_incomplete', 'Media expiration was interrupted; retry with the same confirmation.') from error
    finally:
        os.close(directory_fd)
    with store._connection(write=True) as connection:
        connection.execute("UPDATE media_expiry SET state = 'expired' WHERE id = ? AND state = 'pending'", (identifier,))
    return {'id': identifier, 'state': 'expired', 'artifacts': selected['artifacts'], 'confirmation': confirmation}
