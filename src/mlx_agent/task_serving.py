"""Capability-bound serving plans and durable inference over a resident backend."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from . import task_kinds
from .backends import backend_environment, backend_python, load_manifests, load_registries, lookup, normalize_type
from .compute import ComputeBusy, ComputeLease, process_identity
from . import allocator_cache
from .inference_runner import InferenceError
from .jit_serve import GatewayError, ModelWorker, model_fingerprint, validate_memory_policy
from .resident_runner import ResidentRunner
from .residency import MemoryAdmissionError
from .task_contracts import TaskError, validate_request
from .task_identity import capture_identity, check_identity
from .task_runtime import _artifact_reference, _error, _finish, task_result, task_status
from .task_store import TaskStore

def runtime_fingerprint(backend):
    source = Path(__file__).parent
    from .resident_worker import RUNNERS
    files = [source / (name + '.py') for name in sorted(RUNNERS)]
    files += [source / name for name in ('resident_worker.py', 'runner_model_cache.py','worker_telemetry.py','allocator_cache.py')]
    files += sorted((source / 'resources' / 'ports' / backend).rglob('*.py'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(source)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


TASKS = {'speech_to_text': ['transcribe'], 'text_to_speech': ['speak'], 'embedding': ['embedding'],
         'classification': ['classification'], 'music_generation': ['music_generation'],
         'image_generation': ['image_generation'], 'video_generation': ['video_generation'],
         'mesh_generation': ['mesh_generation']}


def resolve_model(path, manifests=None, registries=None, root=None, capture=True):
    from .serve import ServeError
    from .taxonomy import classify
    manifests = load_manifests() if manifests is None else manifests
    registries = load_registries(manifests, root) if registries is None else registries
    if path is None:
        raise ServeError("model_not_local", "Select an existing local model.", "Serving never downloads models.")
    local = Path(path).expanduser().resolve()
    try:
        config_path = local / 'config.json'
        if config_path.stat().st_size > 1024 * 1024:
            raise ValueError('config too large')
        config = json.loads(config_path.read_text())
        name = local.name + ' ' + str(config.get('_name_or_path', ''))
        if local.parent.name == 'snapshots':
            name += ' ' + local.parent.parent.name
        task = classify(name, model_type=config.get('model_type'), config_keys=config.keys(),
                        manifests=manifests, registries=registries, local=True)
        task_type = task['type']
        if (local / 'modules.json').is_file() and task_type == 'text_llm':
            task_type = 'embedding'
        hits = [hit for hit in lookup(config.get('model_type'), manifests, registries) if hit['category'] == task_type]
        if task_type not in TASKS or not hits:
            raise ValueError('No vetted task adapter implements this model type.')
        if task_type == 'embedding':
            from .embed import TEXT_MODELS
            if normalize_type(config.get('model_type')) not in TEXT_MODELS:
                raise ValueError('No vetted text embedding adapter implements this architecture.')
        manifest = manifests[sorted(hits, key=lambda hit: hit['backend'])[0]['backend']]
        if not manifest['builtin'] and not registries.get(manifest['id'], {}).get('installed'):
            raise ServeError('runtime_not_installed', 'The model backend is not installed.', 'Install ' + manifest['id'] + ' first.')
        executable = str(backend_python(manifest, root))
        evidence = capture_identity(local, manifest, Path(executable)) if capture else None
        return TASKS[task_type], executable, evidence
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise ServeError('unsupported_runtime', str(error), 'Select a supported local model with an installed backend.') from error


def plan_start(repo, path, port, receipts_dir, hf_cache, memory_policy):
    from .convert import cached_snapshot
    from .serve import ServeError, _preview_hash, receipts_root
    has_repo, has_path = isinstance(repo, str) and bool(repo.strip()), isinstance(path, str) and bool(path.strip())
    if has_repo == has_path:
        raise ServeError('invalid_arguments', 'Select exactly one local path or cached repository.', 'Use --path or --repo.')
    if has_repo and '/' not in repo:
        raise ServeError('invalid_repo', 'A repository ID must be publisher/model.', 'Select an existing cached repository.')
    local = Path(path).expanduser() if path else cached_snapshot(repo, hf_cache) if repo else None
    if local is None or not local.is_dir():
        raise ServeError('model_not_local', 'Task serving needs existing local model files.', 'Select a local model; serving never downloads weights.')
    local = local.resolve()
    kinds, executable, identity = resolve_model(local)
    port = 8085 if port is None else port
    if type(port) is not int or not 1 <= port <= 65535:
        raise ServeError('invalid_arguments', 'Port must be between 1 and 65535.', 'Choose a valid loopback port.')
    try:
        policy = validate_memory_policy({'release_after_request': True} if memory_policy is None else memory_policy)
        if policy.get('context_tokens') is not None or policy.get('model_memory_budget_gb') is not None:
            raise ValueError('Task endpoints use headroom reserves without text KV or Strata budgets.')
    except ValueError as error:
        raise ServeError('invalid_arguments', str(error), 'Review the memory policy.') from error
    root = receipts_root(receipts_dir).absolute()
    config = str(root / '{}.jit-config.json'.format(port))
    model = repo or str(local)
    plan = {'repo': repo, 'path': str(local), 'runtime': 'tasks', 'port': port, 'max_tokens': 0, 'adapter_path': None,
            'jit': True, 'local_path': str(local), 'fingerprint': model_fingerprint(local), 'task_identity': identity,
            'task_kinds': kinds, 'task_runtime_fingerprint': runtime_fingerprint(identity['backend']), 'memory_policy': policy, 'control_config': config, 'bind': '127.0.0.1',
            'worker_argv': [executable, str(Path(__file__).with_name('resident_worker.py')), '--model', str(local), '--port', str(port)],
            'argv': [sys.executable, str(Path(__file__).with_name('task_serve.py')), '--model', model, '--port', str(port), '--config', config],
            'readiness': 'http://127.0.0.1:{}/v1/models'.format(port)}
    plan['preview_hash'] = _preview_hash(plan)
    return plan


class TaskModelWorker(ModelWorker):
    def __init__(self, config):
        super().__init__(config)
        self.store = TaskStore(config['task_state_dir'])
        self.resident = ResidentRunner(config['worker_argv'][0], config['local_path'],
            environment=dict(backend_environment(), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1'),
            log_path=config.get('log_path'), port=config['port'])
        self.job_thread = None

    def _check_runtime(self):
        expected = self.config.get('task_runtime_fingerprint')
        if expected is not None and runtime_fingerprint(self.config['task_identity']['backend']) != expected:
            raise TaskError('runtime_changed', 'The accepted inference adapter changed; review serving again.')

    def _allocator_cache_status(self):
        desired = allocator_cache.from_memory(self.memory_policy)
        state = self.resident.cache_state or {}
        applied = state.get('applied_policy')
        pending = (allocator_cache.requested(desired) if applied is None else applied != desired)
        return {'supported': True, 'available': state.get('available'),
                'policy_fields': sorted(allocator_cache.FIELDS), 'reclaim_modes': list(allocator_cache.MODES),
                'control_path': '/_mlx/cache', 'configured_policy': desired, 'applied_policy': applied,
                'pending': pending or bool(state.get('error')), 'active': self.cache_control_active,
                'last_reclaim': state.get('last_reclaim'), 'error': state.get('error')}

    def configure(self, changes):
        if changes.get('context_tokens') is not None or changes.get('model_memory_budget_gb') is not None:
            raise GatewayError(400, 'invalid_policy', 'Task endpoints use headroom reserves without text KV or Strata budgets.')
        try:
            desired = validate_memory_policy(dict(self.memory_policy, **changes))
        except (ValueError, TypeError) as error:
            raise GatewayError(400, 'invalid_policy', str(error)) from error
        if (self.resident.loaded and (self.resident.cache_state or {}).get('available') is False
                and allocator_cache.requested(desired)):
            raise GatewayError(400, 'cache_control_unavailable', 'This worker does not expose allocator cache controls.')
        super().configure(changes)
        if self._allocator_cache_status()['pending']:
            self._cache_operation(None)
        return self.status()

    def _cache_operation(self, reason, manual=False, expected=None):
        with self.condition:
            if (self.closed or self.active_requests or self.cache_control_active
                    or self.model_state != 'loaded' or not self.resident.loaded):
                if manual:
                    raise GatewayError(409, 'model_busy', 'Cache reclaim requires an idle loaded worker.')
                return False
            if expected is not None and expected != (self.process, self.activity_revision, self.policy_revision):
                return False
            self.cache_control_active = True
            desired = allocator_cache.from_memory(self.memory_policy)
            if reason is None and desired['cache_reclaim'] == 'idle':
                reason = 'policy'
        try:
            with ComputeLease('cache:' + self.config['local_path']) as lease:
                try:
                    lease.bind_process_group(self.resident.process.pid)
                    success = self.resident.control_cache(desired, reclaim=reason)
                    lease.release_idle_process_group(self.resident.process.pid)
                except (OSError, ValueError, subprocess.SubprocessError):
                    self._stop_worker()
                    with self.condition:
                        self.model_state = 'unloaded'
                    raise
                settled = self._settle_task_memory()
                if success and settled:
                    self.last_error = None
            if not success:
                self.last_error = (self.resident.cache_state or {}).get('error') or 'Allocator cache control is unavailable.'
                if manual:
                    raise GatewayError(503, 'cache_control_unavailable', self.last_error)
                return False
            return True
        except ComputeBusy as error:
            if manual:
                raise GatewayError(503, 'compute_busy', str(error)) from error
            return False
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            with self.condition:
                self.process = self.resident.process
                self.model_state = 'failed' if self.process else 'unloaded'
                self.execution_unresolved = self.process is not None
                self.last_error = str(error)
            if self.process is None:
                self._release_memory()
            if manual:
                raise GatewayError(503, 'cache_control_failed', str(error)) from error
            return False
        finally:
            with self.condition:
                self.cache_control_active = False
                self.condition.notify_all()

    def reclaim_cache(self):
        self._cache_operation('manual', manual=True)
        return self.status()

    def unload_if_idle(self):
        with self.condition:
            state = self._allocator_cache_status()
            if (self.model_state == 'loaded' and not self.active_requests and not self.cache_control_active
                    and not state['error']):
                pending = state['pending']
                pressure = (state['available'] and self.memory_policy.get('cache_reclaim') == 'pressure'
                            and (self.worker_memory or {}).get('cache_bytes', 0))
                expected = (self.process, self.activity_revision, self.policy_revision)
                reserve = self.memory_policy['minimum_headroom_gb']
            else:
                pending = pressure = False
        if pending:
            return self._cache_operation(None, expected=expected)
        if pressure:
            available = self.memory_probe()
            if type(available) is int and available < reserve * 1e9:
                if self._cache_operation('pressure', expected=expected):
                    return True
        return super().unload_if_idle()

    def _stop_worker(self):
        try:
            self.resident.close()
            self.execution_unresolved = False
            self._release_memory()
        finally:
            with self.condition:
                self.process = self.resident.process
                self.worker_argv = list(self.process.args) if self.process is not None else None

    def _settle_task_memory(self):
        memory = self.resident.telemetry
        self.worker_memory = memory
        resident = None
        if memory['active_bytes'] is not None and memory['cache_bytes'] is not None:
            resident = memory['active_bytes'] + memory['cache_bytes']
            if memory['peak_bytes'] is not None:
                self.workspace_bytes = max(self.workspace_bytes,memory['peak_bytes']-memory['active_bytes'])
        try:
            self.memory_reservation = self.memory_registry.settle(self.memory_token,resident_bytes=resident,telemetry=memory)
        except MemoryAdmissionError as error:
            if error.code != 'memory_registry_busy':
                raise
            self.last_error = 'Memory reservation remains pending: ' + str(error)
            return False
        self.memory_materialized = True
        return True

    def submit(self, body):
        if set(body) != {'kind', 'parameters'} or body['kind'] not in self.config['task_kinds']:
            raise GatewayError(400, 'unsupported_task', 'Use a kind advertised by this model endpoint and its parameters.')
        kind = task_kinds.get(body['kind'])
        request = validate_request({'schema': kind.request_schema, 'kind': kind.name,
            'model_path': self.config['local_path'], 'parameters': body['parameters']})
        with self.condition:
            if self.closed or self.active_requests or self.cache_control_active or self.execution_unresolved or self.model_state == 'unloading':
                raise GatewayError(409, 'model_busy', 'Finish the active model job before starting another.')
            self.active_requests = 1
            self.activity_revision += 1
        staged = None
        record = None
        try:
            self._check_runtime()
            check_identity(self.config['task_identity'])
            if kind.input_media:
                staged = self.store.stage_input(request['parameters'][kind.input_field], kind.input_media)
            with tempfile.TemporaryDirectory(prefix='mlx-task-endpoint-plan-') as directory:
                plan = kind.plan(request, Path(directory), staged['staging_path'] if staged else None)
                if plan['backend'] != self.config['task_identity']['backend'] or plan['argv'][0] != self.resident.executable:
                    raise TaskError('identity_changed', 'The model task backend changed; review serving again.')
            record = self.store.create(request, self.config['task_identity'],
                input_artifact=staged['reference'] if staged else None, staged_input=staged)
            staged = None
            owner = process_identity(os.getpid())
            if not owner:
                raise TaskError('task_identity_unavailable', 'Cannot establish the serving process identity.')
            record = self.store.transition(record['id'], ('starting',), {'supervisor': {'pid': os.getpid(), 'identity': owner}})
            self.job_thread = threading.Thread(target=self._execute, args=(record['id'],), daemon=True)
            self.job_thread.start()
            return record
        except BaseException as error:
            unresolved = False
            try:
                try:
                    self._stop_worker()
                except (OSError, subprocess.SubprocessError):
                    unresolved = True
                    error = TaskError('task_stop_unconfirmed', 'The owned model worker has not stopped safely.')
                with self.condition:
                    self.model_state = 'failed' if unresolved else 'unloaded'
                    self.last_error = str(error)
                    self._persist()
                if record is not None:
                    _finish(self.store, record['id'], 'failed', error=_error(error), execution_unresolved=unresolved)
                if staged:
                    self.store.discard_staged_input(staged)
            finally:
                self.release()
            raise error

    def _execute(self, identifier):
        lease = ComputeLease("task:" + identifier)
        leased = False
        dispatched = False
        try:
            lease.enter_wait(self.memory_policy.get('execution_wait_seconds', 0),
                cancelled=lambda: self.closed or self.store.get(identifier)['cancel_requested'])
            leased = True
            record = self.store.get(identifier)
            if record['cancel_requested'] or self.closed:
                raise InterruptedError('Inference cancelled.')
            self._check_runtime()
            check_identity(record['identity'])
            request = self.store.read_request(identifier)
            kind = task_kinds.get(request['kind'])
            directory = self.store.artifact_path(identifier, 'result.json').parent
            media = str(self.store.verify_input_artifact(identifier, record['input_artifact'], kind.input_media)) if kind.input_media else None
            plan = kind.plan(request, directory, media)
            if plan['backend'] != record['identity']['backend'] or plan['argv'][0] != self.resident.executable:
                raise TaskError('identity_changed', 'The model task backend changed.')
            self._check_memory() if not self.resident.loaded else self._reserve_workspace()
            try:
                capacity = self.memory_registry.reserve_execution(self.memory_token)
            except MemoryAdmissionError as error:
                raise GatewayError(503, error.code, str(error)) from error
            worker_limit = None if capacity is None else max(1, capacity - 512_000_000)
            with self.condition:
                if self.closed:
                    raise InterruptedError('Endpoint stopping.')
                self.model_state = 'loading' if not self.resident.loaded else 'loaded'
            if self.resident.process is None:
                self._launching_memory()
                self.resident._start(cancelled=lambda: self.closed or self.store.get(identifier)['cancel_requested'])
            with self.condition:
                if self.closed:
                    raise InterruptedError('Endpoint stopping.')
                self.process = self.resident.process
                self.worker_argv = list(self.process.args)
                self._persist()
            lease.bind_process_group(self.process.pid)
            worker_identity = process_identity(self.process.pid)
            if not worker_identity:
                raise TaskError('task_identity_unavailable', 'Cannot establish the model worker identity.')
            if not self.memory_materialized:
                self._bind_memory(self.process,worker_identity)
            self.store.transition(identifier, ('starting',), {'state': 'running', 'execution': {'pid': self.process.pid, 'identity': worker_identity}})
            dispatched = True
            result = kind.run(plan, runner=lambda argv, **options: self.resident.run(argv,
                cancelled=lambda: self.closed or self.store.get(identifier)['cancel_requested'],
                cache_policy=allocator_cache.from_memory(self.memory_policy), memory_limit_bytes=worker_limit, **options))
            self._check_runtime()
            check_identity(record['identity'])
            self.store.read_request(identifier)
            if kind.input_media:
                result['input_artifact'] = record['input_artifact']
            if kind.output:
                output = self.store.artifact_path(identifier, kind.output)
                kind.check_output(output, plan)
                result['output_artifact'] = _artifact_reference(output)
            if self.store.get(identifier)['cancel_requested']:
                raise InterruptedError('Inference cancelled.')
            reference = self.store.write_result(identifier, result)
            desired = allocator_cache.from_memory(self.memory_policy)
            if self.resident.cache_state and self.resident.cache_state['applied_policy'] != desired:
                self.resident.control_cache(desired, reclaim='policy' if desired['cache_reclaim'] == 'idle' else None)
            # The protocol acknowledges completion before retaining idle weights.
            lease.release_idle_process_group(self.process.pid)
            memory_settled = self._settle_task_memory()
            with self.condition:
                self.model_state = 'loaded' if self.resident.loaded else 'unloaded'
                self.last_error = None if memory_settled else self.last_error
                self._persist()
            _finish(self.store, identifier, 'succeeded', result=reference, error=None)
        except BaseException as error:
            unresolved = False
            admission_denied = isinstance(error, (ComputeBusy, InterruptedError)) and not leased
            admission_denied |= (not dispatched and isinstance(error,GatewayError)
                and error.code in ('insufficient_headroom','memory_unknown','memory_registry_busy', 'agent_memory_budget_exceeded'))
            # No work was dispatched if the host lease was denied. Keep the
            # existing idle worker and its state available for the next request.
            if not admission_denied:
                try:
                    self._stop_worker()
                except (OSError, subprocess.SubprocessError):
                    unresolved = True
            if isinstance(error, ComputeBusy):
                error = TaskError('compute_busy', str(error), 'Wait for the current host model operation to finish, then start a new job.')
            if unresolved:
                error = TaskError('task_stop_unconfirmed', 'The owned model worker has not stopped safely.', 'Inspect the worker and retry the owned endpoint stop; its identity is retained.')
            with self.condition:
                if not admission_denied:
                    self.model_state = 'failed' if unresolved else 'unloaded'
                self.last_error = str(error)
                self._persist()
            cancelled = isinstance(error, InterruptedError) or self.store.get(identifier)['cancel_requested']
            _finish(self.store, identifier, 'cancelled' if cancelled and not unresolved else 'failed',
                    error=None if cancelled and not unresolved else _error(error), execution_unresolved=unresolved)
        finally:
            if leased:
                lease.__exit__(None, None, None)
            self.release()

    def close(self):
        with self.condition:
            self.closed = True
        if self.job_thread:
            self.job_thread.join(timeout=10)
        if self.job_thread and self.job_thread.is_alive():
            raise OSError('The owned model job is still stopping.')
        super().close()

    def record(self, identifier):
        record = task_status(identifier, self.store.root)
        if record['model_path'] != self.config['local_path'] or record['kind'] not in self.config['task_kinds']:
            raise TaskError('task_not_found', 'This endpoint does not own that model job.')
        return record
