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
                or row.get('state') not in ('launching', 'launch_failed', 'active', 'stopped')
                or type(row.get('port')) is not int or not 1 <= row['port'] <= 65535
                or not isinstance(row.get('member_plan_hash'), str)):
            raise ValueError('Invalid workflow activation member.')
    return value


def _write(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(data) > workflow_sets.MAX_BYTES:
        raise ValueError('Workflow activation receipt is too large.')
    _atomic_in_directory(path.parent, path.name, data, 0o600)


def _owned_receipt(member, root):
    port, pid, start = member['port'], member['pid'], member['process_started_at']
    if type(pid) is not int or pid <= 0 or not isinstance(start, str) or not start:
        raise ValueError('Service process identity is unavailable.')
    receipt = workflow_sets._private_json(receipts_root(root) / (str(port) + '.json'))
    if (receipt.get('kind') != SERVE_RECEIPT_KIND or receipt.get('schema_version') != SERVE_RECEIPT_SCHEMA_VERSION
            or receipt.get('port') != port or not receipt.get('jit')
            or receipt.get('pid') != pid or receipt.get('process_started_at') != start
            or receipt.get('preview_hash') != member['member_plan_hash']
            or not _pid_alive(pid) or _pid_start(pid) != start or not _argv_matches(receipt, _pid_command(pid))):
        raise ValueError('Service process identity changed.')
    return receipt


def _service_status(member, root, expected_plan=None):
    receipt = _owned_receipt(member, root)
    state = _jit_control(receipt, receipts_root(root), 'status')
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
        candidate = workflow_sets._reuse_candidate(receipts_root(root), plan)
        if candidate['pid'] != member.get('pid') or candidate['process_started_at'] != member.get('process_started_at'):
            raise ValueError('Active workflow service identity changed: ' + role)
        _service_status(member, root, plan)
    return value


def start_workflow(name, preview_hash, *, receipts_dir=None):
    path = _path(name, receipts_dir)
    with _target_locks([path], create_parents=True):
        if os.path.lexists(path):
            previous = _read(path)
            if previous['state'] != 'stopped':
                return _repeat_activation(previous, name, preview_hash, receipts_dir)
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
                    candidate = workflow_sets._reuse_candidate(receipts_root(receipts_dir), plan)
                    if candidate != reviewed['candidate']:
                        raise ValueError('Reused service identity changed: ' + role)
                    child = workflow_sets._private_json(Path(candidate['receipt']))
                    item = _member_identity(role, plan, child, 'reused')
                    _service_status(item, receipts_dir, plan)
                else:
                    # Mark in-flight before launch: an interruption must not silently discard ownership uncertainty.
                    result['members'][role] = {'role': role, 'port': member['port'], 'ownership': 'created',
                                               'state': 'launching', 'member_plan_hash': plan['preview_hash']}
                    _write(path, result)
                    outcome = start_serve(plan, receipts_dir=receipts_dir, confirm=True, preview_hash=plan['preview_hash'])
                    item = _member_identity(role, plan, outcome['receipt'], 'created')
                result['members'][role] = item
                _write(path, result)
                if not item['process_started_at']:
                    raise ValueError('Service start identity is unavailable: ' + role)
            result['state'] = 'active'
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            if getattr(error, 'code', None) == 'port_in_use' and result['members'].get(role, {}).get('state') == 'launching':
                result['members'][role]['state'] = 'launch_failed'
                result['members'][role]['error'] = str(error)
            for item in result['members'].values():
                if item.get('ownership') == 'created' and item.get('state') == 'active':
                    item['cleanup_required'] = 'Endpoint retained; use workflow stop to clean up.'
                elif item.get('state') == 'launching':
                    item['cleanup_required'] = 'Launch ownership is unresolved; inspect the service receipt.'
            # No atomic idle-stop exists. Preserve created endpoints, even apparently idle ones.
            result['error'] = str(error)
            _write(path, result)
            return result
        _write(path, result)
        return result


def status_workflow(name, *, receipts_dir=None):
    value = _read(_path(name, receipts_dir))
    result = dict(value, recorded_state=value['state'], members={})
    for role, member in value['members'].items():
        item = dict(member)
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
    config = workflow_sets._private_json(receipts_root(receipts_dir) / (str(member['port']) + '.jit-config.json'))
    if (config.get('runtime') != runtime or config.get('model') != model
            or config.get('task_kinds', []) != kinds or config.get('access_key_file') != credential
            or config.get('api_key_sha256') != receipt.get('api_key_sha256')):
        raise ValueError('Workflow role configuration changed.')
    state = _jit_control(receipt, receipts_root(receipts_dir), 'status')
    if state.get('api_key_sha256') != receipt.get('api_key_sha256'):
        raise ValueError('Workflow role credential identity changed.')
    if runtime == 'tasks':
        operations = [{'method': method, 'path': path} for method, path in (
            ('POST', '/v1/tasks'), ('GET', '/v1/tasks/{id}'), ('GET', '/v1/tasks/{id}/result'),
            ('GET', '/v1/tasks/{id}/artifacts/{name}'), ('POST', '/v1/tasks/{id}/cancel'))]
        if 'embedding' in kinds:
            operations.append({'method': 'POST', 'path': '/v1/embeddings'})
    else:
        operations = [{'method': 'POST', 'path': path} for path in (
            '/v1/chat/completions', '/v1/messages', '/v1/responses')]
        operations.append({'method': 'GET', 'path': '/v1/responses/{id}'})
    operations.append({'method': 'GET', 'path': '/v1/models'})
    origin = 'http://127.0.0.1:' + str(member['port'])
    return {'schema': 'serving-workflow-role/1', 'workflow': name, 'activation_id': value['activation_id'],
            'workflow_state': value['state'], 'role': role, 'runtime': runtime, 'origin': origin,
            'base_url': origin + '/v1', 'model_id': model, 'operations': operations,
            'task_kinds': kinds, 'credential_ref': credential, 'model_state': state['model_state'],
            'endpoint_identity': {'pid': member['pid'], 'process_started_at': member['process_started_at'],
                                  'member_plan_hash': member['member_plan_hash']}}


def stop_workflow(name, *, receipts_dir=None):
    path = _path(name, receipts_dir)
    with _target_locks([path]):
        result = _read(path)
        for role, member in result['members'].items():
            if member.get('ownership') != 'created' or member.get('state') in ('stopped', 'launch_failed'):
                continue
            try:
                _owned_receipt(member, receipts_dir)
                stop_serve(member['port'], receipts_dir, expected_pid=member['pid'],
                           expected_start=member['process_started_at'])
                member['state'] = 'stopped'
                member.pop('error', None)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                member['error'] = str(error)
            _write(path, result)
        pending = any(row.get('ownership') == 'created' and row.get('state') not in ('stopped', 'launch_failed')
                      for row in result['members'].values())
        result['state'] = 'partial' if pending else 'stopped'
        _write(path, result)
        return result
