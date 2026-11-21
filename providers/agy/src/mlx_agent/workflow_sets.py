"""Private workflow definitions and advisory, non-activating previews."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from subprocess import SubprocessError
import time

from .convert import cached_snapshot
from .jit_serve import available_memory_bytes
from .memory_policy import policy_from_profile, reservation, validate as validate_policy, enforce_launch, model_key as _model_key
from .residency import MemoryRegistry, SCHEMA as REGISTRY_SCHEMA
from .compute import process_identity
from .memory_budget import same_identity
from .serve import (SERVE_RECEIPT_KIND, SERVE_RECEIPT_SCHEMA_VERSION, load_recipes, plan_start, receipts_root, resolve_serve_runtime,
                    _argv_matches, _pid_alive, _pid_command, _pid_start)
from .serve_profiles import SPLASH_FIELDS, SPLASH_TUNING, load_profile
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
        if not isinstance(member, dict) or not required <= set(member) or set(member) - required - {'path', 'repo', 'api_key_file', 'reuse_existing'}:
            raise ValueError('Invalid workflow member fields.')
        if 'reuse_existing' in member and type(member['reuse_existing']) is not bool:
            raise ValueError('Invalid existing endpoint preference.')
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
    if runtime != 'splash' and SPLASH_FIELDS & set(profile):
        raise ValueError('Splash profile requires a Splash runtime.')
    policy = validate_policy(policy_from_profile(profile))
    if runtime != 'tasks':
        enforce_launch(policy, runtime, True)
    return plan_start(repo, runtime, recipes, port=member['port'], path=path, jit=True,
                      receipts_dir=root, runtime_executable=executable, memory_policy=policy,
                      api_key_file=member.get('api_key_file'),
                      strata_context_tokens=profile.get('strata_context_tokens'),
                      strata_decoding=profile.get('strata_decoding'),
                      splash_max_memory_gb=profile.get('splash_max_memory_gb'),
                      splash_max_context_tokens=profile.get('splash_max_context_tokens'),
                      splash_tuning={key: profile[key] for key in SPLASH_TUNING if key in profile})


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
    try:
        state = _private_json(root / (str(port) + '.jit-state.json'))
    except FileNotFoundError:
        state = None
    pid = receipt.get('pid')
    if type(pid) is not int or pid <= 0 or not _pid_alive(pid):
        raise ValueError('Service gateway identity is unavailable.')
    start, command = _pid_start(pid), _pid_command(pid)
    if (not start or receipt.get('process_started_at') != start or not _argv_matches(receipt, command)
            or (state is not None and state.get('gateway_pid') != pid)):
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


def _existing_claim_candidate(plan):
    """Resolve a compatible claimed generation without attaching to it."""
    from .model_serving import resolve_model, state_root, _read_claims, _live, _service_info, _legacy_matches
    from .service_registry import discover_services
    from .transactions import _assert_safe_directory
    from .worker_transport import gateway_config
    artifact = resolve_model(plan.get('local_path') or plan.get('path') or plan.get('repo'))
    key = hashlib.sha256(artifact.encode()).hexdigest()
    claim_root = state_root()
    if os.path.lexists(claim_root):
        _assert_safe_directory(claim_root)
        info = claim_root.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('Model ownership directory is unsafe.')
    claims = [row for row in _read_claims(claim_root, key) if _live(row)]
    if not claims:
        found = discover_services()
        if found.get('errors'):
            raise ValueError('Existing service discovery is uncertain.')
        if _legacy_matches(artifact, found.get('services', [])):
            raise ValueError('An existing legacy service cannot be bound at this location.')
        return None
    if len(claims) != 1 or claims[0].get('kind') != 'service':
        raise ValueError('Existing model ownership is ambiguous or in use.')
    row = claims[0]
    if row.get('retiring') or not row.get('ready') or not row.get('jit'):
        raise ValueError('Existing service is not available for workflow binding.')
    endpoint = _service_info(row)
    receipt_path = Path(row.get('receipt_path') or '')
    if (not receipt_path.is_absolute() or receipt_path.parent.name != 'serve'
            or receipt_path.parent.parent.name != '.mlx-agent-receipts'):
        raise ValueError('Existing service receipt location is invalid.')
    root = receipt_path.parent
    receipt = _private_json(receipt_path)
    config = _private_json(root / (str(receipt['port']) + '.jit-config.json'))
    if endpoint['auth_required'] != bool(config.get('api_key_sha256') or config.get('access_key_file')):
        raise ValueError('Existing service authentication identity changed.')
    expected = gateway_config(plan, control_token='', state_path='', log_path='', config_path='', task_state_dir='')
    actual = dict(config)
    if actual.get('model_serving_claim') != {'key': key, 'generation': row['generation']}:
        raise ValueError('Existing service claim identity changed.')
    actual.pop('model_serving_claim')
    for field in ('control_token', 'state_path', 'log_path', 'config_path', 'task_state_dir'):
        actual.pop(field, None)
        expected.pop(field, None)
    if actual.pop('port', None) != receipt['port']:
        raise ValueError('Existing service transport identity changed.')
    if plan['runtime'] == 'tasks':
        if expected.pop('port', None) != plan['port']:
            raise ValueError('Existing service transport identity changed.')
        for config_value, port in ((actual, receipt['port']), (expected, plan['port'])):
            argv = list(config_value.get('worker_argv') or [])
            if argv.count('--port') != 1 or argv[argv.index('--port') + 1] != str(port):
                raise ValueError('Existing service worker transport is invalid.')
            argv[argv.index('--port') + 1] = '{port}'
            config_value['worker_argv'] = argv
    if actual != expected:
        raise ValueError('Existing service configuration or capabilities differ.')
    return {'pid': receipt['pid'], 'process_started_at': receipt['process_started_at'],
            'receipt': str(receipt_path), 'receipt_preview_hash': receipt['preview_hash'],
            'memory_policy': config['memory_policy'], 'endpoint': endpoint}



def _resource_forecast(members, inspection, available, root):
    """Project only identity-qualified reservations; unknowns stay unknown."""
    def valid(value):
        return type(value) is int and 0 <= value <= 2**60
    result = {'status':'unknown', 'reason':'Insufficient reliable accounting for a joint estimate.',
              'captured_at':time.time(), 'members':{}, 'estimates':{'incremental_bytes':None,
              'available_bytes':available if valid(available) else None, 'agent_budget':inspection.get('agent_budget')}}
    try:
        if inspection.get('state') not in ('ok','uninitialized') or inspection.get('errors'):
            raise ValueError('Memory registry inspection is uncertain.')
        entries = inspection.get('entries')
        if not isinstance(entries, dict): raise ValueError('Memory registry entries are unavailable.')
        MemoryRegistry._validate({'schema': REGISTRY_SCHEMA, 'entries': entries})
        budget = inspection.get('agent_budget')
        if budget is not None:
            if not isinstance(budget,dict) or any(not valid(budget.get(k)) for k in ('accounted_bytes','admission_bytes')):
                raise ValueError('Shared budget accounting is unavailable.')
            if entries and (any(not valid(budget.get(k)) for k in ('pending_bytes','resident_bytes','external_bytes','overhead_bytes','observed_bytes','limit_bytes','reserve_bytes'))
                    or not isinstance(budget.get('allocations'),dict)):
                raise ValueError('Shared budget allocations are unavailable.')
            if entries and any(not isinstance(budget['allocations'].get(key),dict) or any(
                    not valid(budget['allocations'][key].get(field)) for field in ('resident_bytes','pending_bytes','observed_bytes'))
                    for key in entries):
                raise ValueError('Shared budget allocation is invalid.')
            baseline = budget.get('pending_bytes', 0)
            if not valid(baseline): raise ValueError('Shared pending allocation is invalid.')
        else:
            baseline = sum(row['reserved_bytes'] for row in entries.values())
        floors = [row['headroom_bytes'] for row in entries.values()]
        total = 0
        cap_failure = None
        for role, member in members.items():
            plan, estimate, action = member.get('plan'), member.get('estimate'), member['action']
            if action == 'blocked' or not isinstance(plan,dict) or not isinstance(estimate,dict) or not valid(estimate.get('size')):
                raise ValueError(role + ': member estimate unavailable.')
            policy = plan.get('memory_policy') or {}
            floor = policy.get('minimum_headroom_gb') or 0
            cap = policy.get('model_memory_budget_gb')
            if isinstance(floor,bool) or not isinstance(floor,(int,float)) or not math.isfinite(floor) or floor < 0:
                raise ValueError(role + ': headroom is invalid.')
            floors.append(math.ceil(floor * 1e9))
            if cap is not None and (isinstance(cap,bool) or not isinstance(cap,(int,float)) or not math.isfinite(cap) or cap <= 0):
                raise ValueError(role + ': route cap is invalid.')
            cap_bytes = math.ceil(cap * 1e9) if cap is not None else None
            envelope = max(estimate['size'], cap_bytes or 0)
            delta, source = envelope, 'configured_reservation' if cap_bytes is None else 'route_cap'
            required_cap = estimate['size']
            if action == 'reuse':
                candidate = member['candidate']; pid = candidate['pid']
                try:
                    candidate_receipt = Path(candidate.get('receipt') or root / (str(plan['port']) + '.json'))
                    state = _private_json(candidate_receipt.parent / (candidate_receipt.stem + '.jit-state.json'))
                except (OSError, ValueError):
                    raise ValueError(role + ': service state is unavailable or unsafe.') from None
                if state.get('gateway_pid') != pid or _pid_start(pid) != candidate['process_started_at']:
                    raise ValueError(role + ': gateway state identity is uncertain.')
                identity = process_identity(pid)
                owned = [(key,row) for key,row in entries.items() if isinstance(row.get('owner'),(list,tuple)) and len(row['owner']) == 2 and row['owner'][0] == pid]
                if not identity or any(not same_identity(row['owner'][1],identity) for _,row in owned):
                    raise ValueError(role + ': gateway ownership is uncertain.')
                if state.get('model_state') == 'unloaded' and state.get('worker_pid') is None and not owned:
                    source = 'unloaded_reservation'
                else:
                    if len(owned) != 1: raise ValueError(role + ': allocation is missing or ambiguous.')
                    key,row = owned[0]; worker = row.get('worker')
                    if row.get('phase') != 'resident' or 'execution_bytes' in row or row.get('model_key') != _model_key(plan):
                        raise ValueError(role + ': allocation is active or mismatched.')
                    if any(not valid(row.get(k)) for k in ('resident_bytes','workspace_bytes','growth_bytes','reserved_bytes')):
                        raise ValueError(role + ': workspace or residency is invalid.')
                    if not isinstance(worker,dict) or type(worker.get('pid')) is not int or worker['pid'] <= 0:
                        raise ValueError(role + ': worker identity is unavailable.')
                    wp = worker['pid']; current = process_identity(wp)
                    if (state.get('model_state') != 'loaded' or state.get('worker_pid') != wp or not current
                            or not same_identity(worker.get('identity'),current) or _pid_start(wp) != state.get('worker_started_at')
                            or not state.get('worker_started_at') or os.getpgid(wp) != worker.get('pgid')):
                        raise ValueError(role + ': worker identity is uncertain.')
                    workspace = row['workspace_bytes']; telemetry = row.get('telemetry')
                    if isinstance(telemetry,dict) and valid(telemetry.get('peak_bytes')) and valid(telemetry.get('active_bytes')):
                        workspace = max(workspace, max(0,telemetry['peak_bytes']-telemetry['active_bytes']))
                        workspace_source = 'allocator_peak_delta'
                    else: workspace_source = 'registry_workspace'
                    # Resident process memory is already charged; only project extra pending work.
                    if budget is not None:
                        allocation = budget['allocations'].get(key)
                        if not isinstance(allocation,dict) or any(not valid(allocation.get(k)) for k in ('observed_bytes','pending_bytes','resident_bytes')):
                            raise ValueError(role + ': shared allocation is unavailable.')
                        credit, pending, source = allocation['observed_bytes'], allocation['pending_bytes'], 'observed_allocation'
                    else: credit, pending, source = row['resident_bytes'], row['reserved_bytes'], 'registry_resident'
                    growth = row['growth_bytes']
                    # Task jobs reserve fresh workspace; JIT growth may already be materialized.
                    if plan.get('runtime') == 'tasks':
                        charged = allocation['resident_bytes'] if budget is not None else row['resident_bytes']
                        required_cap = max(estimate['size'], row['resident_bytes']+workspace+growth,
                                           credit+workspace+growth, charged+workspace+growth)
                        if cap_bytes is not None and cap_bytes < required_cap:
                            cap_failure = role + ': route cap is below the working estimate.'
                        future = max(workspace+growth, max(0,envelope-credit))
                    else:
                        future = max(pending, max(0,envelope-credit))
                    delta = max(0,future-pending)
                    result['members'][role] = {'credit_bytes':credit,'current_pending_bytes':pending,'workspace_bytes':workspace,
                        'growth_bytes':growth,'workspace_source':workspace_source,'credit_source':source}
            if plan.get('runtime') == 'tasks' and cap_bytes is not None and cap_bytes < estimate['size']: cap_failure = role + ': route cap is below the working estimate.'
            result['members'].setdefault(role, {'credit_bytes':0,'current_pending_bytes':0,'credit_source':source})
            result['members'][role].update(envelope_bytes=envelope, incremental_bytes=delta,
                configured_limit_bytes=cap_bytes, required_limit_bytes=required_cap if plan.get('runtime') == 'tasks' else None,
                allowance_source='route_cap' if plan.get('runtime') == 'tasks' else 'model_allowance')
            total += delta
        floor = max(floors,default=0); required = baseline+total+floor
        projected = None if budget is None else budget['accounted_bytes']+total
        if not all(valid(v) for v in (baseline, total, floor, required)) or (projected is not None and not valid(projected)):
            raise ValueError('Joint allocation total is invalid.')
        result['estimates'].update(incremental_bytes=total, baseline_pending_bytes=baseline,
            headroom_bytes=floor, required_available_bytes=required, projected_accounted_bytes=projected)
        if cap_failure: result.update(status='does_not_fit',reason=cap_failure)
        elif budget is not None and projected > budget['admission_bytes']: result.update(status='does_not_fit',reason='Estimated shared budget is insufficient.')
        elif not valid(available): raise ValueError('OS available memory is unavailable.')
        elif available < required: result.update(status='does_not_fit',reason='Estimated OS headroom is insufficient.')
        else: result.update(status='fits_estimate',reason='Conservative concurrent allocation estimate.')
    except (OSError,ValueError,TypeError,KeyError,OverflowError,SubprocessError) as error:
        result['reason'] = str(error)
    return result

def preview_workflow(definition, *, receipts_dir=None):
    if receipts_dir is not None:
        receipts_dir = Path(receipts_dir).absolute()
    definition = validate_workflow(definition)
    members, binding, forecasts = {}, {}, {}
    root = receipts_root(receipts_dir)
    planned_artifacts = set()
    for role, member in sorted(definition['members'].items()):
        try:
            from .model_serving import resolve_model, read_policy
            plan = _member_plan(member, receipts_dir)
            estimate = reservation(plan.get('memory_policy'), plan)
            artifact = resolve_model(plan.get('local_path') or plan.get('path') or plan.get('repo') or member.get('path') or member.get('repo'))
            allow_duplicate = artifact in read_policy()['duplicate_model_allowlist']
            if artifact in planned_artifacts and not allow_duplicate:
                raise ValueError('Another workflow member selects the same model artifact.')
            planned_artifacts.add(artifact)
            receipt = root / (str(member['port']) + '.json')
            candidate = None
            if os.path.lexists(receipt):
                candidate = _reuse_candidate(root, plan)
            elif not allow_duplicate or member.get('reuse_existing', False):
                candidate = _existing_claim_candidate(plan)
                if candidate and not member.get('reuse_existing', False):
                    raise ValueError('An existing endpoint serves this model; enable reuse_existing to bind it.')
            action = 'reuse' if candidate else 'start'
            member_hash = plan['preview_hash']
            members[role] = {'member_plan_hash': member_hash, 'action': action,
                             'runtime': plan['runtime'], 'task_kinds': plan.get('task_kinds', []),
                             'endpoint_state': 'unknown' if candidate else 'not_started',
                             'model_state': 'unknown' if candidate else 'unloaded',
                             'reason': None, 'candidate': candidate, 'estimate': estimate}
            binding[role] = {'member_plan_hash': member_hash, 'action': action, 'candidate': candidate}
            forecasts[role] = {'plan':plan, 'estimate':estimate, 'action':action, 'candidate':candidate}
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            members[role] = {'member_plan_hash': None, 'action': 'blocked', 'endpoint_state': 'unknown',
                             'model_state': 'unknown', 'reason': str(error), 'estimate': None}
            binding[role] = {'member_plan_hash': None, 'action': 'blocked', 'reason': str(error)}
            forecasts[role] = {'plan':None,'estimate':None,'action':'blocked'}
    inspection = MemoryRegistry().inspect()
    forecast = _resource_forecast(forecasts, inspection, available_memory_bytes(), root)
    definition_hash = _hash(definition)
    return {'schema': 'serving-workflow-preview/1', 'definition_hash': definition_hash,
            'preview_hash': _hash({'definition_hash': definition_hash, 'members': binding}),
            'members': members, 'resource_forecast': forecast}
