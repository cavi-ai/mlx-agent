"""Read-only stage estimates and identity-bound handoff of completed tasks."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from . import served, workflow_activation as activation, workflow_sets
from .jit_serve import available_memory_bytes
from .memory_policy import reservation
from .residency import MemoryRegistry
from .serve import receipts_root
from .serve_profiles import load_profile
from .task_contracts import MAX_REQUEST_BYTES, task_id
from .transactions import _assert_safe_directory


def _role_identity(value):
    return {key: value[key] for key in ('activation_id', 'workflow_state', 'role', 'runtime', 'origin', 'model_id', 'task_kinds', 'endpoint_identity')}


def stage(name, roles, *, receipts_dir=None):
    if not isinstance(roles, list) or not roles or len(roles) > 32 or len(set(roles)) != len(roles):
        raise ValueError('Select one or more distinct workflow roles.')
    for role in roles:
        workflow_sets._name(role)
    record = activation._read(activation._path(name, receipts_dir))
    if record['state'] != 'active':
        raise ValueError('Workflow activation is not active.')
    definition = workflow_sets.load_workflow(name, receipts_dir)
    if workflow_sets._hash(definition) != record['definition_hash']:
        raise ValueError('Workflow definition changed.')
    selected, forecasts, profiles = {}, {}, {}
    for role in roles:
        member = record['members'].get(role)
        if role not in definition['members'] or not isinstance(member, dict) or member.get('state') != 'active':
            raise ValueError('Workflow role is missing or not active: ' + role)
        member_definition = definition['members'][role]
        profiles[role] = load_profile(member_definition['profile'], receipts_dir)
        plan = workflow_sets._member_plan(member_definition, receipts_dir)
        if plan['preview_hash'] != member['member_plan_hash']:
            raise ValueError('Workflow member plan changed: ' + role)
        identity = _role_identity(activation.resolve_workflow_role(name, role, receipts_dir=receipts_dir, expected_plan=plan))
        if identity['activation_id'] != record['activation_id'] or identity['workflow_state'] != 'active':
            raise ValueError('Workflow activation changed: ' + role)
        candidate = {'pid': member['pid'], 'process_started_at': member['process_started_at'],
                     'receipt': str(receipts_root(activation._member_root(member, receipts_dir)) / (str(member['port']) + '.json'))}
        selected[role] = identity
        forecasts[role] = {'plan': plan, 'estimate': reservation(plan.get('memory_policy'), plan),
                           'action': 'reuse', 'candidate': candidate}
    forecast = workflow_sets._resource_forecast(forecasts, MemoryRegistry().inspect(),
                                                available_memory_bytes(), receipts_root(receipts_dir))
    if workflow_sets._hash(workflow_sets.load_workflow(name, receipts_dir)) != record['definition_hash']:
        raise ValueError('Workflow definition changed during stage assessment.')
    for role in roles:
        if load_profile(definition['members'][role]['profile'], receipts_dir) != profiles[role]:
            raise ValueError('Workflow profile changed during stage assessment: ' + role)
        if _role_identity(activation.resolve_workflow_role(name, role, receipts_dir=receipts_dir,
                                                           expected_plan=forecasts[role]['plan'])) != selected[role]:
            raise ValueError('Workflow role changed during stage assessment: ' + role)
    return {'schema': 'serving-workflow-stage/1', 'workflow': name, 'activation_id': record['activation_id'],
            'roles': selected, 'resource_forecast': forecast}


def _read_json(port, key, path, limit=MAX_REQUEST_BYTES):
    with served._open(port, key, 'GET', path) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Task response exceeds the size limit.')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Task response is not an object.')
    return value, raw


def _reference(value, limit):
    if (not isinstance(value, dict) or type(value.get('bytes')) is not int or not 0 <= value['bytes'] <= limit
            or not isinstance(value.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', value['sha256'])):
        raise ValueError('Task artifact reference is invalid.')
    return {'bytes': value['bytes'], 'sha256': value['sha256']}


def handoff(name, role, identifier, *, out=None, receipts_dir=None):
    task_id(identifier)
    if out is not None:
        if not isinstance(out, str) or not out or len(out) > 4096 or '\x00' in out or '..' in Path(out).parts:
            raise ValueError('Output path is invalid.')
        out = str(Path(out).expanduser().absolute())
        _assert_safe_directory(Path(out).parent)
    current = activation._read(activation._path(name, receipts_dir))
    role_info = activation.resolve_workflow_role(name, role, receipts_dir=receipts_dir)
    if role_info['runtime'] != 'tasks':
        raise ValueError('Workflow role does not serve typed tasks.')
    member = current['members'][role]
    endpoint = role_info['endpoint_identity']
    if (role_info['activation_id'] != current['activation_id'] or endpoint.get('pid') != member.get('pid')
            or endpoint.get('process_started_at') != member.get('process_started_at')
            or role_info['origin'] != 'http://127.0.0.1:' + str(member.get('port'))):
        raise ValueError('Workflow activation changed before task readback.')
    service_root = activation._member_root(member, receipts_dir)
    port = member['port']
    key = served._key({'port': port, 'receipts_dir': str(Path(service_root or Path.cwd()).absolute())})
    outcome, _ = _read_json(port, key, '/v1/tasks/{}/result'.format(identifier))
    record = outcome.get('task')
    if (not isinstance(record, dict) or record.get('id') != identifier or record.get('state') != 'succeeded'
            or record.get('kind') not in role_info['task_kinds']):
        raise ValueError('Task is not a successful result for this role.')
    config = workflow_sets._private_json(receipts_root(service_root) / (str(port) + '.jit-config.json'))
    if not isinstance(record.get('model_path'), str) or Path(record['model_path']).resolve() != Path(config['local_path']).resolve():
        raise ValueError('Task model differs from the workflow role.')
    recorded_identity, current_identity = record.get('identity'), config.get('task_identity')
    identity_fields = ('model_path', 'model_fingerprint', 'backend', 'backend_manifest_hash', 'executable_identity')
    if (not isinstance(recorded_identity, dict) or not isinstance(current_identity, dict)
            or any(key not in recorded_identity or recorded_identity[key] != current_identity.get(key)
                   for key in identity_fields)):
        raise ValueError('Task model or backend identity differs from the workflow role.')
    reference = _reference(outcome.get('artifact'), MAX_REQUEST_BYTES)
    result, raw = _read_json(port, key, '/v1/tasks/{}/artifacts/result'.format(identifier))
    if len(raw) != reference['bytes'] or hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('Task result does not match its recorded digest.')
    output = result.pop('output_artifact', None)
    artifact = _reference(output, 2**40) if output is not None else None
    if out is not None and artifact is None:
        raise ValueError('This task has no output artifact to download.')
    if _role_identity(activation.resolve_workflow_role(name, role, receipts_dir=receipts_dir)) != _role_identity(role_info):
        raise ValueError('Workflow role changed during task readback.')
    if out is not None:
        with tempfile.TemporaryDirectory(prefix='.mlx-handoff-', dir=str(Path(out).parent)) as directory:
            staged = str(Path(directory) / 'output')
            served._download(port, key, identifier, staged, artifact)
            if _role_identity(activation.resolve_workflow_role(name, role, receipts_dir=receipts_dir)) != _role_identity(role_info):
                raise ValueError('Workflow role changed during artifact download.')
            os.link(staged, out, follow_symlinks=False)
    return {'schema': 'serving-workflow-handoff/1', 'workflow': name, 'role': role,
            'activation_id': role_info['activation_id'], 'endpoint_identity': role_info['endpoint_identity'],
            'task_id': identifier, 'kind': record['kind'],
            'task_provenance': {'model_path': recorded_identity['model_path'],
                                'model_fingerprint': recorded_identity['model_fingerprint'],
                                'backend': recorded_identity['backend'], 'completed_at': record.get('completed_at')},
            'result': result, 'result_artifact': reference,
            'output_artifact': artifact, 'output_path': out}
