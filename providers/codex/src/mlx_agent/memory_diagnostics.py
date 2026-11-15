"""Bounded memory decision data for status and private state."""
import hashlib
import os
import time

FIELDS = ('available_bytes', 'required_available_bytes', 'headroom_bytes',
          'requested_headroom_bytes', 'required_bytes', 'other_reserved_bytes',
          'other_resident_bytes', 'requested_bytes', 'registry_resident_bytes',
          'workspace_bytes', 'growth_bytes', 'pending_bytes', 'cache_bytes', 'active_bytes',
          'observed_bytes', 'charged_resident_bytes', 'route_limit_bytes',
          'shared_limit_bytes', 'shared_admission_bytes', 'shared_accounted_bytes',
          'shared_external_bytes', 'shared_overhead_bytes', 'shared_observed_bytes',
          'shared_resident_bytes', 'shared_pending_bytes', 'accounted_bytes')
SOURCES = ('cold_admission', 'workspace_admission', 'task_admission', 'resident_guard', 'agent_guard')


def memory_decision(source, reason, details=None, *, pid=None, pgid=None, identity=None):
    if source not in SOURCES or not isinstance(reason, str) or not reason.isidentifier() or len(reason) > 64:
        raise ValueError('Invalid memory decision identity.')
    details = details if isinstance(details, dict) else {}
    budget = details.get('agent_budget') if isinstance(details.get('agent_budget'), dict) else {}
    owner_hash = details.get('registry_owner_identity_sha256')
    if not isinstance(owner_hash, str) or len(owner_hash) != 64 or any(c not in '0123456789abcdef' for c in owner_hash):
        owner_hash = None
    result = {'schema': 'memory-decision/1', 'captured_at': time.time(), 'source': source,
              'reason': reason, 'gateway_pid': os.getpid(), 'persistence_error': None,
              'registry_owner_identity_sha256': owner_hash, 'worker_pid': pid if type(pid) is int and pid > 0 else None,
              'worker_pgid': pgid if type(pgid) is int and pgid > 0 else None,
              'worker_identity_sha256': hashlib.sha256(identity.encode()).hexdigest() if isinstance(identity, str) else None}
    aliases = {'route_limit_bytes': details.get('route_limit_bytes', details.get('limit_bytes')),
               'charged_resident_bytes': details.get('charged_resident_bytes', details.get('resident_bytes')),
               'shared_limit_bytes': budget.get('limit_bytes'),
               'shared_admission_bytes': budget.get('admission_bytes'),
               'shared_accounted_bytes': budget.get('accounted_bytes'),
               'accounted_bytes': budget.get('accounted_bytes'),
               'shared_external_bytes': budget.get('external_bytes'),
               'shared_overhead_bytes': budget.get('overhead_bytes'),
               'shared_observed_bytes': budget.get('observed_bytes'),
               'shared_resident_bytes': budget.get('resident_bytes'),
               'shared_pending_bytes': budget.get('pending_bytes')}
    for field in FIELDS:
        value = aliases.get(field, details.get(field))
        result[field] = value if type(value) is int and 0 <= value <= 2**60 else None
    return result
