"""Identity-bound activation of reviewed local serving workflows."""
import json
import os
from pathlib import Path
import uuid

from . import workflow_sets
from .serve import (SERVE_RECEIPT_KIND, SERVE_RECEIPT_SCHEMA_VERSION,
                    _argv_matches, _jit_control, _pid_alive, _pid_command, _pid_start,
                    receipts_root, start_serve, stop_serve, _default_port_free)
from .transactions import _atomic_in_directory, _target_locks

SCHEMA = 'serving-workflow-activation/1'


def _path(name, root):
    workflow_sets._name(name)
    return receipts_root(root) / 'workflow-activations' / (name + '.json')


def _read(path):
    value = workflow_sets._private_json(path)
    if (value.get('schema') != SCHEMA or value.get('state') not in ('active', 'partial', 'stopped')
            or not isinstance(value.get('members'), dict) or len(value['members']) > 32
            or not isinstance(value.get('definition_hash'), str) or not isinstance(value.get('preview_hash'), str)
            or not isinstance(value.get('activation_id'), str) or value.get('name') != path.stem):
        raise ValueError('Invalid workflow activation receipt.')
    for role, row in value['members'].items():
        workflow_sets._name(role)
        if (not isinstance(row, dict) or row.get('role') != role
                or row.get('ownership') not in ('created', 'reused')
                or row.get('state') not in ('launching', 'launch_failed', 'attaching', 'active', 'stopped')
                or type(row.get('port')) is not int or not 1 <= row['port'] <= 65535
                or not isinstance(row.get('member_plan_hash'), str)):
            raise ValueError('Invalid workflow activation member.')
    return value


def _write(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(data) > workflow_sets.MAX_BYTES:
        raise ValueError('Workflow activation receipt is too large.')
    _atomic_in_directory(path.parent, path.name, data, 0o600)


def _public(value):
    result = dict(value)
    result['members'] = {role: {key: value for key, value in row.items()
                               if key not in ('claim', 'config_identity')}
                         for role, row in value['members'].items()}
    return result


def _config_identity(member, root):
    config = workflow_sets._private_json(receipts_root(_member_root(member, root)) / (str(member['port']) + '.jit-config.json'))
    config.pop('control_token', None)
    return workflow_sets._hash(config)


def _owned_receipt(member, root):
    port, pid, start = member['port'], member['pid'], member['process_started_at']
    root = _member_root(member, root)
    if type(pid) is not int or pid <= 0 or not isinstance(start, str) or not start:
        raise ValueError('Service process identity is unavailable.')
    receipt = workflow_sets._private_json(receipts_root(root) / (str(port) + '.json'))
    if (receipt.get('kind') != SERVE_RECEIPT_KIND or receipt.get('schema_version') != SERVE_RECEIPT_SCHEMA_VERSION
            or receipt.get('port') != port or not receipt.get('jit')
            or receipt.get('pid') != pid or receipt.get('process_started_at') != start
            or receipt.get('preview_hash') != member.get('receipt_preview_hash', member['member_plan_hash'])
            or not _pid_alive(pid) or _pid_start(pid) != start or not _argv_matches(receipt, _pid_command(pid))):
        raise ValueError('Service process identity changed.')
    return receipt


def _member_root(member, default):
    root = member.get('service_receipts_dir', default)
    if root is not None and (not isinstance(root, (str, os.PathLike)) or not Path(root).is_absolute()):
        raise ValueError('Invalid service location.')
    return root


def _service_status(member, root, expected_plan=None):
    receipt = _owned_receipt(member, root)
    if member.get('config_identity') is not None and _config_identity(member, root) != member['config_identity']:
        raise ValueError('Service configuration identity changed.')
    state = _jit_control(receipt, receipts_root(_member_root(member, root)), 'status')
    if expected_plan is not None and (state.get('memory_policy') != expected_plan.get('memory_policy')
            or state.get('api_key_sha256') != expected_plan.get('api_key_sha256')):
        raise ValueError('Live service policy or credential identity changed.')
    return {'endpoint_state': 'available', 'model_state': state['model_state'],
            'active_requests': state.get('active_requests')}


def _member_identity(role, plan, receipt, ownership):
    if receipt.get('runtime') != plan['runtime'] or receipt.get('task_kinds', []) != plan.get('task_kinds', []):
        raise ValueError('Service capabilities differ from the reviewed plan.')
    return {'role': role, 'port': receipt['port'], 'pid': receipt['pid'],
            'process_started_at': receipt.get('process_started_at'),
            'member_plan_hash': plan['preview_hash'], 'ownership': ownership, 'state': 'active',
            'runtime': plan['runtime'], 'model_id': plan.get('repo') or plan.get('path'),
            'task_kinds': plan.get('task_kinds', []), 'credential_ref': plan.get('access_key_file')}


def _member_claim(member, root):
    """Read the exact claim token from the verified gateway configuration."""
    import hashlib
    from .model_serving import Claim, state_root
    receipt = _owned_receipt(member, root)
    config = workflow_sets._private_json(receipts_root(_member_root(member, root)) / (str(member['port']) + '.jit-config.json'))
    token = config.get('model_serving_claim')
    if not isinstance(token, dict) or set(token) != {'key', 'generation'}:
        return None
    artifact = receipt.get('model_artifact')
    if not isinstance(artifact, str) or token.get('key') != hashlib.sha256(artifact.encode()).hexdigest():
        raise ValueError('Service claim identity differs from its receipt.')
    if not all(isinstance(token.get(k), str) and len(token[k]) == 64 for k in ('key', 'generation')):
        raise ValueError('Invalid service claim identity.')
    return Claim(state_root(), token['key'], token['generation'])


def _repeat_activation(value, name, preview_hash, root):
    if value['state'] != 'active':
        raise ValueError('A workflow activation still owns or references services.')
    definition = workflow_sets.load_workflow(name, root)
    if workflow_sets._hash(definition) != value.get('definition_hash') or set(definition['members']) != set(value['members']):
        raise ValueError('Active workflow definition changed.')
    if value.get('preview_hash') != preview_hash:
        current = workflow_sets.preview_workflow(definition, receipts_dir=root)
        if current['preview_hash'] != preview_hash or any(row['action'] != 'reuse' for row in current['members'].values()):
            raise ValueError('Workflow preview changed; review it again.')
        for role, row in current['members'].items():
            candidate = row['candidate']
            member = value['members'][role]
            if candidate['pid'] != member.get('pid') or candidate['process_started_at'] != member.get('process_started_at'):
                raise ValueError('Active workflow service identity changed: ' + role)
    for role, definition_member in definition['members'].items():
        member = value['members'][role]
        plan = workflow_sets._member_plan(definition_member, root)
        if plan['preview_hash'] != member.get('member_plan_hash'):
            raise ValueError('Active workflow member plan changed: ' + role)
        if receipts_root(_member_root(member, root)) != receipts_root(root) or member['port'] != plan['port']:
            candidate = workflow_sets._existing_claim_candidate(plan)
        else:
            candidate = workflow_sets._reuse_candidate(receipts_root(root), plan)
        if candidate['pid'] != member.get('pid') or candidate['process_started_at'] != member.get('process_started_at'):
            raise ValueError('Active workflow service identity changed: ' + role)
        _service_status(member, root, plan)
        if member.get('claim') is not None:
            claim = _member_claim(member, root)
            if claim is None or member['claim'] != {'key': claim.key, 'generation': claim.generation}:
                raise ValueError('Active workflow service generation changed: ' + role)
            claim.workflow_reference(value['activation_id'] + ':' + role, 'validate')
    return value


def start_workflow(name, preview_hash, *, receipts_dir=None):
    path = _path(name, receipts_dir)
    with _target_locks([path], create_parents=True):
        if os.path.lexists(path):
            previous = _read(path)
            if previous['state'] != 'stopped':
                return _public(_repeat_activation(previous, name, preview_hash, receipts_dir))
        definition = workflow_sets.load_workflow(name, receipts_dir)
        preview = workflow_sets.preview_workflow(definition, receipts_dir=receipts_dir)
        if preview_hash != preview['preview_hash']:
            raise ValueError('Workflow preview changed; review it again.')
        if any(row['action'] == 'blocked' for row in preview['members'].values()):
            raise ValueError('A workflow member is blocked.')
        for role, row in preview['members'].items():
            if row['action'] == 'start' and not _default_port_free(definition['members'][role]['port']):
                raise ValueError('Workflow port is already occupied: ' + role)
        result = {'schema': SCHEMA, 'name': name, 'activation_id': str(uuid.uuid4()),
                  'definition_hash': preview['definition_hash'], 'preview_hash': preview_hash,
                  'state': 'partial', 'members': {}, 'resource_forecast': preview['resource_forecast']}
        _write(path, result)
        try:
            for role, member in sorted(definition['members'].items()):
                reviewed = preview['members'][role]
                plan = workflow_sets._member_plan(member, receipts_dir)
                if plan['preview_hash'] != reviewed['member_plan_hash']:
                    raise ValueError('Member plan changed after review: ' + role)
                if reviewed['action'] == 'reuse':
                    reviewed_path = Path(reviewed['candidate']['receipt'])
                    if reviewed_path == receipts_root(receipts_dir) / (str(plan['port']) + '.json'):
                        candidate = workflow_sets._reuse_candidate(receipts_root(receipts_dir), plan)
                    else:
                        candidate = workflow_sets._existing_claim_candidate(plan)
                    if candidate != reviewed['candidate']:
                        raise ValueError('Reused service identity changed: ' + role)
                    child = workflow_sets._private_json(Path(candidate['receipt']))
                    item = _member_identity(role, plan, child, 'reused')
                    item['receipt_preview_hash'] = candidate['receipt_preview_hash']
                    item['service_receipts_dir'] = str(Path(candidate['receipt']).parent.parent.parent)
                    _service_status(item, receipts_dir, plan)
                else:
                    # Mark in-flight before launch: an interruption must not silently discard ownership uncertainty.
                    result['members'][role] = {'role': role, 'port': member['port'], 'ownership': 'created',
                                               'state': 'launching', 'member_plan_hash': plan['preview_hash']}
                    _write(path, result)
                    outcome = start_serve(plan, receipts_dir=receipts_dir, confirm=True, preview_hash=plan['preview_hash'])
                    item = _member_identity(role, plan, outcome['receipt'], 'created')
                claim = _member_claim(item, receipts_dir)
                item['config_identity'] = _config_identity(item, receipts_dir)
                if claim is not None:
                    item['claim'] = {'key': claim.key, 'generation': claim.generation}
                    item['state'] = 'attaching'
                    result['members'][role] = item
                    _write(path, result)
                    claim.workflow_reference(result['activation_id'] + ':' + role, 'attach')
                    item['state'] = 'active'
                elif item['ownership'] == 'created':
                    raise ValueError('Created service claim is unavailable: ' + role)
                result['members'][role] = item
                _write(path, result)
                if not item['process_started_at']:
                    raise ValueError('Service start identity is unavailable: ' + role)
            result['state'] = 'active'
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            if (getattr(error, 'code', None) in ('port_in_use', 'model_already_served', 'model_load_in_progress')
                    and result['members'].get(role, {}).get('state') == 'launching'):
                result['members'][role]['state'] = 'launch_failed'
                result['members'][role]['error'] = str(error)
            for item in result['members'].values():
                if item.get('ownership') == 'created' and item.get('state') == 'active':
                    item['cleanup_required'] = 'Endpoint retained; use workflow stop to clean up.'
                elif item.get('state') == 'launching':
                    item['cleanup_required'] = 'Launch ownership is unresolved; inspect the service receipt.'
            # No atomic idle-stop exists. Preserve created endpoints, even apparently idle ones.
            result['error'] = str(error)
            if getattr(error, 'code', None) == 'model_already_served':
                from .model_serving import public_error_data
                result['error_code'] = error.code
                result['error_data'] = public_error_data(error.data)
            _write(path, result)
            return _public(result)
        _write(path, result)
        return _public(result)


def status_workflow(name, *, receipts_dir=None):
    value = _read(_path(name, receipts_dir))
    result = dict(value, recorded_state=value['state'], members={})
    for role, member in value['members'].items():
        item = {key: value for key, value in member.items() if key not in ('claim', 'config_identity')}
        try:
            if member['state'] == 'stopped':
                item.update(endpoint_state='stopped', model_state='unknown')
            else:
                item.update(_service_status(member, receipts_dir))
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            item.update(endpoint_state='unknown', model_state='unknown', error=str(error))
        result['members'][role] = item
    if value['state'] == 'active' and any(
            item.get('endpoint_state') != 'available' for item in result['members'].values()):
        result['state'] = 'partial'
    return result


def resolve_workflow_role(name, role, *, receipts_dir=None):
    workflow_sets._name(role)
    value = _read(_path(name, receipts_dir))
    if value['state'] == 'stopped':
        raise ValueError('Workflow activation is stopped.')
    member = value['members'].get(role)
    if not isinstance(member, dict) or member.get('state') != 'active':
        raise ValueError('Workflow role is missing or not active.')
    receipt = _owned_receipt(member, receipts_dir)
    if member.get('config_identity') is not None and _config_identity(member, receipts_dir) != member['config_identity']:
        raise ValueError('Workflow service configuration changed.')
    endpoint = None
    if member.get('claim') is not None:
        claim = _member_claim(member, receipts_dir)
        if claim is None or member['claim'] != {'key': claim.key, 'generation': claim.generation}:
            raise ValueError('Workflow service generation changed.')
        endpoint = claim.workflow_reference(value['activation_id'] + ':' + role, 'validate')
    runtime, model, kinds = member.get('runtime'), member.get('model_id'), member.get('task_kinds')
    credential = member.get('credential_ref')
    if (runtime not in ('tasks', 'mlx_lm', 'mlx-vlm', 'strata', 'splash')
            or not isinstance(model, str) or not model or not isinstance(kinds, list)
            or any(not isinstance(kind, str) for kind in kinds)
            or (credential is not None and (not isinstance(credential, str) or not Path(credential).is_absolute()))):
        raise ValueError('Workflow role metadata is unavailable.')
    if (receipt.get('runtime') != runtime or (receipt.get('repo') or receipt.get('path')) != model
            or receipt.get('task_kinds', []) != kinds):
        raise ValueError('Workflow role capability identity changed.')
    service_root = receipts_root(_member_root(member, receipts_dir))
    if endpoint is None:
        from .model_serving import resolve_receipt_endpoint
        endpoint = resolve_receipt_endpoint({'receipt': str(service_root / (str(member['port']) + '.json')),
            'receipts_dir': str(service_root.parent.parent), 'pid': member['pid'], 'port': member['port'],
            'runtime': receipt['runtime'], 'alive': True, 'argv_match': True,
            'process_started_at': member['process_started_at'], 'api_key_sha256': receipt.get('api_key_sha256')})
        if endpoint is None:
            raise ValueError('Workflow service identity is unavailable.')
    config = workflow_sets._private_json(service_root / (str(member['port']) + '.jit-config.json'))
    if (config.get('runtime') != runtime or config.get('model') != model
            or config.get('task_kinds', []) != kinds or config.get('access_key_file') != credential
            or config.get('api_key_sha256') != receipt.get('api_key_sha256')):
        raise ValueError('Workflow role configuration changed.')
    state = _jit_control(receipt, service_root, 'status')
    if state.get('api_key_sha256') != receipt.get('api_key_sha256'):
        raise ValueError('Workflow role credential identity changed.')
    from .endpoint_contract import operations
    origin = 'http://127.0.0.1:' + str(member['port'])
    return {'schema': 'serving-workflow-role/1', 'workflow': name, 'activation_id': value['activation_id'],
            'workflow_state': value['state'], 'role': role, 'runtime': runtime, 'origin': origin,
            'base_url': origin + '/v1', 'model_id': model, 'operations': operations(runtime, kinds),
            'task_kinds': kinds, 'credential_ref': credential, 'model_state': state['model_state'],
            'endpoint_identity': {'service_id': endpoint['service_id'], 'pid': member['pid'], 'process_started_at': member['process_started_at'],
                                  'member_plan_hash': member['member_plan_hash']}}


def stop_workflow(name, *, receipts_dir=None):
    path = _path(name, receipts_dir)
    with _target_locks([path]):
        result = _read(path)
        for role, member in result['members'].items():
            if member.get('state') in ('stopped', 'launch_failed'):
                continue
            try:
                from .model_serving import Claim, state_root
                token = member.get('claim')
                claim = None
                if token is not None:
                    if not isinstance(token, dict) or set(token) != {'key', 'generation'}:
                        raise ValueError('Invalid service claim identity.')
                    import re
                    if not all(isinstance(token[k], str) and re.fullmatch('[0-9a-f]{64}', token[k]) for k in token):
                        raise ValueError('Invalid service claim identity.')
                    claim = Claim(state_root(), token['key'], token['generation'])
                reference = result['activation_id'] + ':' + role
                if member['ownership'] == 'reused':
                    if claim:
                        claim.workflow_reference(reference, 'detach',
                                                 absent_owner=(member.get('pid'), member.get('process_started_at')))
                    member['state'] = 'stopped'
                    member.pop('error', None)
                    _write(path, result)
                    continue
                if not claim:
                    claim = _member_claim(member, receipts_dir)
                    if claim is None:
                        raise ValueError('Service references are unavailable; ownership requires reconciliation.')
                    member['claim'] = {'key': claim.key, 'generation': claim.generation}
                    member['state'] = 'attaching'
                    _write(path, result)
                try:
                    _owned_receipt(member, receipts_dir)
                except (OSError, ValueError, TypeError, KeyError) as identity_error:
                    try:
                        claim.workflow_reference(reference, 'confirm_absent',
                                                 absent_owner=(member.get('pid'), member.get('process_started_at')))
                    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
                        raise identity_error
                    member['state'] = 'stopped'
                    member.pop('error', None)
                    member.pop('error_code', None)
                    _write(path, result)
                    continue
                current = _member_claim(member, receipts_dir)
                if current is None or current.key != claim.key or current.generation != claim.generation:
                    raise ValueError('Service claim identity changed.')
                if member.get('state') == 'attaching':
                    claim.workflow_reference(reference, 'attach')
                if not claim.workflow_reference(reference, 'retire'):
                    member['error_code'] = 'service_referenced'
                    member['error'] = 'Service is referenced by another workflow; retry after it stops.'
                    _write(path, result)
                    continue
                stop_serve(member['port'], receipts_dir, expected_pid=member['pid'],
                           expected_start=member['process_started_at'])
                member['state'] = 'stopped'
                member.pop('error', None)
                member.pop('error_code', None)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                member['error'] = str(error)
            _write(path, result)
        pending = any(row.get('state') not in ('stopped', 'launch_failed') for row in result['members'].values())
        result['state'] = 'partial' if pending else 'stopped'
        _write(path, result)
        return _public(result)
