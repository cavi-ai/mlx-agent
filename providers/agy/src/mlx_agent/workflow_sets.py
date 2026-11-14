"""Private workflow definitions and advisory, non-activating previews."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from .convert import cached_snapshot
from .jit_serve import available_memory_bytes
from .memory_policy import policy_from_profile, reservation, validate as validate_policy, enforce_launch
from .residency import MemoryRegistry
from .serve import (SERVE_RECEIPT_KIND, SERVE_RECEIPT_SCHEMA_VERSION, load_recipes, plan_start, receipts_root, resolve_serve_runtime,
                    _argv_matches, _pid_alive, _pid_command, _pid_start)
from .serve_profiles import load_profile
from .transactions import _atomic_in_directory, _target_locks, _walk_directory
from .wiring import _MODEL

SCHEMA = 'serving-workflow/1'
MAX_BYTES = 65536
_NAME = re.compile(r'[a-z][a-z0-9_-]{0,63}\Z')


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError('Invalid workflow or role name.')
    return value


def validate_workflow(value):
    if not isinstance(value, dict) or set(value) != {'schema', 'name', 'members'} or value['schema'] != SCHEMA:
        raise ValueError('Invalid workflow definition.')
    _name(value['name'])
    members = value['members']
    if not isinstance(members, dict) or not 1 <= len(members) <= 32:
        raise ValueError('A workflow requires between 1 and 32 members.')
    ports = set()
    for role, member in members.items():
        _name(role)
        required = {'runtime', 'port', 'profile'}
        if not isinstance(member, dict) or not required <= set(member) or set(member) - required - {'path', 'repo', 'api_key_file'}:
            raise ValueError('Invalid workflow member fields.')
        if ('path' in member) == ('repo' in member):
            raise ValueError('Select exactly one model path or cached repo.')
        for field in ('path', 'repo', 'api_key_file'):
            if field in member and (not isinstance(member[field], str) or not member[field] or len(member[field]) > 4096):
                raise ValueError('Invalid workflow member reference.')
        if 'repo' in member and not _MODEL.fullmatch(member['repo']):
            raise ValueError('Invalid cached model repository.')
        if member['runtime'] not in ('auto', 'mlx_lm', 'mlx-vlm', 'strata', 'splash', 'tasks'):
            raise ValueError('Invalid workflow runtime.')
        if member['runtime'] == 'strata' and 'repo' in member:
            raise ValueError('Strata requires a local path.')
        _name(member['profile'])
        port = member['port']
        if type(port) is not int or not 1 <= port <= 65535 or port in ports:
            raise ValueError('Workflow ports must be distinct valid ports.')
        ports.add(port)
    try:
        if len(json.dumps(value, allow_nan=False).encode()) > MAX_BYTES:
            raise ValueError('Workflow definition is too large.')
    except (TypeError, OverflowError) as error:
        raise ValueError('Invalid workflow definition.') from error
    return value


def _path(name, root):
    return receipts_root(root) / 'workflows' / (_name(name) + '.json')


def load_workflow(name, root=None):
    path = _path(name, root)
    _, directory = _walk_directory(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1 or info.st_size > MAX_BYTES:
                raise ValueError('Invalid workflow file.')
            value = validate_workflow(json.loads(stream.read(MAX_BYTES + 1)))
        if value['name'] != name:
            raise ValueError('Workflow name does not match its file.')
        return value
    finally:
        os.close(directory)


def save_workflow(name, value, root=None):
    path = _path(name, root)
    value = validate_workflow(value)
    if value['name'] != name:
        raise ValueError('Workflow name does not match the save name.')
    with _target_locks([path.parent / '.workflows'], create_parents=True):
        if os.path.lexists(path):
            raise ValueError('Workflow already exists.')
        if len(list_workflows(root)) >= 100:
            raise ValueError('Too many workflows.')
        _atomic_in_directory(path.parent, path.name, (json.dumps(value, sort_keys=True) + '\n').encode(), 0o600)
    return value


def list_workflows(root=None):
    path = _path('unused', root).parent
    if not path.exists() and not path.is_symlink():
        return {}
    _, fd = _walk_directory(path)
    try:
        names = sorted(name[:-5] for name in os.listdir(fd) if name.endswith('.json'))
        if len(names) > 100:
            raise ValueError('Too many workflows.')
        return {name: load_workflow(name, root) for name in names}
    finally:
        os.close(fd)


def delete_workflow(name, root=None):
    path = _path(name, root)
    with _target_locks([path.parent / '.workflows']):
        load_workflow(name, root)
        _, fd = _walk_directory(path.parent)
        try:
            os.unlink(path.name, dir_fd=fd)
            os.fsync(fd)
        finally:
            os.close(fd)


def _member_plan(member, root):
    repo, path = member.get('repo'), member.get('path')
    if repo is not None and cached_snapshot(repo) is None:
        raise ValueError('Model repository is not cached locally.')
    profile = load_profile(member['profile'], root)
    recipes = load_recipes()
    runtime, executable = resolve_serve_runtime(repo, path, member['runtime'], recipes)
    if runtime != 'strata' and ('strata_context_tokens' in profile or 'strata_decoding' in profile):
        raise ValueError('Strata profile requires a Strata runtime.')
    policy = validate_policy(policy_from_profile(profile))
    if runtime != 'tasks':
        enforce_launch(policy, runtime, True)
    return plan_start(repo, runtime, recipes, port=member['port'], path=path, jit=True,
                      receipts_dir=root, runtime_executable=executable, memory_policy=policy,
                      api_key_file=member.get('api_key_file'),
                      strata_context_tokens=profile.get('strata_context_tokens'),
                      strata_decoding=profile.get('strata_decoding'))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _private_json(path):
    _, directory = _walk_directory(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077
                    or info.st_nlink != 1 or info.st_size > 1024 * 1024):
                raise ValueError('Unsafe service sidecar.')
            value = json.loads(stream.read(1024 * 1024 + 1))
            if not isinstance(value, dict):
                raise ValueError('Invalid service sidecar.')
            return value
    finally:
        os.close(directory)


def _reuse_candidate(root, plan):
    port = plan['port']
    receipt = _private_json(root / (str(port) + '.json'))
    config = _private_json(root / (str(port) + '.jit-config.json'))
    state = _private_json(root / (str(port) + '.jit-state.json'))
    pid = receipt.get('pid')
    if type(pid) is not int or pid <= 0 or not _pid_alive(pid):
        raise ValueError('Service gateway identity is unavailable.')
    start, command = _pid_start(pid), _pid_command(pid)
    if not start or not _argv_matches(receipt, command) or state.get('gateway_pid') != pid:
        raise ValueError('Service gateway identity does not match its receipt and state.')
    if (receipt.get('kind') != SERVE_RECEIPT_KIND or receipt.get('schema_version') != SERVE_RECEIPT_SCHEMA_VERSION
            or receipt.get('preview_hash') != plan['preview_hash'] or receipt.get('port') != port
            or receipt.get('runtime') != plan['runtime'] or not receipt.get('jit')):
        raise ValueError('Service receipt does not match the reviewed member plan.')
    for field in ('local_path', 'fingerprint', 'runtime', 'worker_argv', 'memory_policy'):
        if config.get(field) != plan.get(field):
            raise ValueError('Persisted service configuration differs: ' + field)
    if config.get('api_key_sha256') != plan.get('api_key_sha256'):
        raise ValueError('Persisted service credential identity differs.')
    return {'pid': pid, 'process_started_at': start, 'receipt': str(root / (str(port) + '.json')),
            'receipt_preview_hash': receipt['preview_hash'], 'memory_policy': config['memory_policy']}


def preview_workflow(definition, *, receipts_dir=None):
    definition = validate_workflow(definition)
    members, binding, estimates, reserves = {}, {}, [], []
    root = receipts_root(receipts_dir)
    for role, member in sorted(definition['members'].items()):
        try:
            plan = _member_plan(member, receipts_dir)
            estimate = reservation(plan.get('memory_policy'), plan)
            receipt = root / (str(member['port']) + '.json')
            candidate = None
            if os.path.lexists(receipt):
                candidate = _reuse_candidate(root, plan)
            action = 'reuse' if candidate else 'start'
            member_hash = plan['preview_hash']
            members[role] = {'member_plan_hash': member_hash, 'action': action,
                             'runtime': plan['runtime'], 'task_kinds': plan.get('task_kinds', []),
                             'endpoint_state': 'unknown' if candidate else 'not_started',
                             'model_state': 'unknown' if candidate else 'unloaded',
                             'reason': None, 'candidate': candidate, 'estimate': estimate}
            binding[role] = {'member_plan_hash': member_hash, 'action': action, 'candidate': candidate}
            estimates.append(estimate)
            reserves.append(math.ceil((plan.get('memory_policy', {}).get('minimum_headroom_gb') or 0) * 1e9))
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            members[role] = {'member_plan_hash': None, 'action': 'blocked', 'endpoint_state': 'unknown',
                             'model_state': 'unknown', 'reason': str(error), 'estimate': None}
            binding[role] = {'member_plan_hash': None, 'action': 'blocked', 'reason': str(error)}
            estimates.append(None)
    inspection = MemoryRegistry().inspect()
    available = available_memory_bytes()
    budget = inspection.get('agent_budget')
    amount = sum(item['size'] for item in estimates if item is not None)
    forecast = {'status': 'unknown', 'reason': 'Insufficient reliable accounting for a joint estimate.',
                'estimates': {'incremental_bytes': amount if all(item is not None for item in estimates)
                              and all(row['action'] == 'start' for row in members.values())
                              and not inspection['entries'] else None,
                              'available_bytes': available, 'agent_budget': budget}}
    if (all(item is not None for item in estimates) and all(row['action'] == 'start' for row in members.values())
            and inspection['state'] == 'ok' and not inspection['entries'] and budget is not None
            and type(available) is int):
        reserve = max(reserves)
        fits = budget['accounted_bytes'] + amount <= budget['admission_bytes'] and amount + reserve <= available
        forecast.update(status='fits_estimate' if fits else 'does_not_fit', reason='Conservative concurrent allocation estimate.')
    definition_hash = _hash(definition)
    return {'schema': 'serving-workflow-preview/1', 'definition_hash': definition_hash,
            'preview_hash': _hash({'definition_hash': definition_hash, 'members': binding}),
            'members': members, 'resource_forecast': forecast}
