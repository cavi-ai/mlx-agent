"""Host model ownership and policy contracts."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent import model_serving


class ModelServingTests(unittest.TestCase):
    def test_admission_recovers_overflow_of_missing_registered_roots(self):
        import json
        import os
        from unittest.mock import patch
        from mlx_agent.service_registry import MAX_INDEX_FILES

        with TemporaryDirectory() as folder:
            root = Path(folder)
            state = root / 'state'
            index = state / 'mlx-agent/services'
            index.mkdir(parents=True)
            model = root / 'model'
            model.mkdir()
            for i in range(MAX_INDEX_FILES + 2):
                record = {'schema_version': 1, 'receipts_dir': str(root / f'missing-{i}')}
                (index / f'{i:04d}.json').write_text(json.dumps(record))
            with patch.dict(os.environ, {'XDG_STATE_HOME': str(state)}), \
                 patch('mlx_agent.service_registry.live_receipt_roots', return_value=[]):
                claim = model_serving.acquire(str(model), root=root / 'claims')
                claim.release()
            self.assertLess(len(list(index.glob('*.json'))), MAX_INDEX_FILES)

    def test_overflow_recovery_remains_fail_closed_for_live_roots_and_custom_discovery(self):
        import json
        import os
        from unittest.mock import patch
        from mlx_agent.service_registry import MAX_ROOTS

        with TemporaryDirectory() as folder:
            root = Path(folder)
            state = root / 'state'
            index = state / 'mlx-agent/services'
            index.mkdir(parents=True)
            model = root / 'model'
            model.mkdir()
            for i in range(MAX_ROOTS + 1):
                live = root / f'live-{i}'
                live.mkdir()
                record = {'schema_version': 1, 'receipts_dir': str(live)}
                (index / f'{i:04d}.json').write_text(json.dumps(record))
            with patch.dict(os.environ, {'XDG_STATE_HOME': str(state)}), \
                 patch('mlx_agent.service_registry.live_receipt_roots', return_value=[]):
                with self.assertRaises(model_serving.ModelServingError) as failure:
                    model_serving.acquire(str(model), root=root / 'claims')
                self.assertEqual(failure.exception.code, 'model_serving_state_unknown')
                self.assertEqual(len(list(index.glob('*.json'))), MAX_ROOTS + 1)
                result = {'services': [], 'errors': [
                    {'source': str(index), 'message': 'Service location limit reached.'}
                ]}
                with patch('mlx_agent.service_registry.prune_stale_registrations') as cleanup:
                    with self.assertRaises(model_serving.ModelServingError):
                        model_serving.acquire(str(model), root=root / 'claims', discover=lambda: result)
                    self.assertEqual(cleanup.call_count, 0)

    def test_model_identity_resolves_directory_aliases(self):
        with TemporaryDirectory() as directory:
            model = Path(directory) / 'model'
            model.mkdir()
            alias = Path(directory) / 'alias'
            alias.symlink_to(model, target_is_directory=True)
            self.assertEqual(model_serving.resolve_model(str(model)), model_serving.resolve_model(str(alias)))

    def test_policy_defaults_to_no_duplicates(self):
        with TemporaryDirectory() as directory:
            self.assertEqual(model_serving.read_policy(Path(directory)),
                             {'schema_version': 1, 'duplicate_model_allowlist': []})

    def test_duplicate_claim_rejects_and_release_allows_next_owner(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'state'
            model = Path(directory) / 'model'
            model.mkdir()
            first = model_serving.acquire(str(model), root=root, discover=lambda: {'services': [], 'errors': []})
            try:
                with self.assertRaises(model_serving.ModelServingError) as rejected:
                    model_serving.acquire(str(model), root=root, discover=lambda: {'services': [], 'errors': []})
                self.assertEqual(rejected.exception.code, 'model_load_in_progress')
            finally:
                first.release()
            second = model_serving.acquire(str(model), root=root, discover=lambda: {'services': [], 'errors': []})
            second.release()

    def test_gguf_files_have_separate_identities(self):
        from tests.unit.test_gguf import write_gguf
        with TemporaryDirectory() as directory:
            first, second = Path(directory) / 'first.gguf', Path(directory) / 'second.gguf'
            write_gguf(first)
            write_gguf(second)
            self.assertNotEqual(model_serving.resolve_model(str(first)), model_serving.resolve_model(str(second)))

    def test_launching_claim_survives_dead_initiator_as_unknown(self):
        from unittest.mock import patch
        row = {'pid': 42, 'birth': 'old', 'state': 'launching'}
        with patch('mlx_agent.compute.process_birth_identity', return_value=None), patch('mlx_agent.compute.process_alive', return_value=False):
            with self.assertRaises(model_serving.ModelServingError) as rejected:
                model_serving._live(row)
            self.assertEqual(rejected.exception.code, 'model_serving_state_unknown')

    def test_cli_accepts_policy_read(self):
        from mlx_agent.cli import build_parser
        args = build_parser('serve').parse_args(['serve', 'duplication-policy', '--json'])
        self.assertEqual(args.serve_command, 'duplication-policy')

    def test_error_result_preserves_safe_data(self):
        from mlx_agent.contracts import ResultEnvelope
        result = ResultEnvelope.fail('serve-start', 'model_already_served', 'Already served', 'Connect',
                                     data={'existing_services': [{'origin': 'http://127.0.0.1:8080'}]})
        self.assertEqual(result.to_dict()['data']['existing_services'][0]['origin'], 'http://127.0.0.1:8080')

    def test_concurrent_processes_admit_one_owner(self):
        import multiprocessing
        with TemporaryDirectory() as directory:
            model = Path(directory) / 'model'
            model.mkdir()
            state = Path(directory) / 'state'
            context = multiprocessing.get_context('fork')
            start = context.Event()
            finish = context.Event()
            results = context.Queue()
            def run():
                start.wait()
                try:
                    claim = model_serving.acquire(str(model), root=state, discover=lambda: {'services': [], 'errors': []})
                    results.put('claimed')
                    finish.wait(5)
                    claim.release()
                except model_serving.ModelServingError as error:
                    results.put(error.code)
            workers = [context.Process(target=run) for _ in range(2)]
            for worker in workers:
                worker.start()
            start.set()
            outcomes = sorted(results.get(timeout=10) for _ in workers)
            finish.set()
            for worker in workers:
                worker.join(10)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(outcomes, ['claimed', 'model_load_in_progress'])

    def test_stale_gateway_is_pruned_only_after_worker_state_proves_absence(self):
        import json
        from unittest.mock import patch
        with TemporaryDirectory() as directory:
            state = Path(directory) / 'state.json'
            state.write_text(json.dumps({'gateway_pid': 123, 'worker_pid': None, 'worker_launch_pending': False}))
            state.chmod(0o600)
            row = {'pid': 123, 'birth': 'old', 'state': 'active', 'pgid': 123, 'jit': True, 'state_path': str(state)}
            with patch('mlx_agent.compute.process_birth_identity', return_value=None), \
                 patch('mlx_agent.compute.process_alive', return_value=False), \
                 patch('mlx_agent.compute.group_members', return_value=[]):
                self.assertFalse(model_serving._live(row))
                state.write_text(json.dumps({'gateway_pid': 123, 'worker_pid': None, 'worker_launch_pending': True}))
                with self.assertRaises(model_serving.ModelServingError):
                    model_serving._live(row)

    def test_exact_policy_allows_only_the_selected_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'state'
            first = Path(directory) / 'first'; first.mkdir()
            second = Path(directory) / 'second'; second.mkdir()
            alias = Path(directory) / 'alias'; alias.symlink_to(first, target_is_directory=True)
            model_serving.write_policy([str(alias)], root)
            discovery = lambda: {'services': [], 'errors': []}
            one = model_serving.acquire(str(first), root=root, discover=discovery)
            two = model_serving.acquire(str(alias), root=root, discover=discovery)
            three = model_serving.acquire(str(second), root=root, discover=discovery)
            with self.assertRaises(model_serving.ModelServingError):
                model_serving.acquire(str(second), root=root, discover=discovery)
            one.release(); two.release(); three.release()
            with self.assertRaises(ValueError):
                model_serving.write_policy(['*'], root)

    def test_pending_service_error_contains_only_safe_endpoint_details(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'state'
            model = Path(directory) / 'model'; model.mkdir()
            discovery = lambda: {'services': [], 'errors': []}
            claim = model_serving.acquire(str(model), root=root, discover=discovery, kind='service',
                service={'port': 8011, 'runtime': 'tasks', 'auth_required': True, 'secret': 'do-not-return'})
            try:
                with self.assertRaises(model_serving.ModelServingError) as rejected:
                    model_serving.acquire(str(model), root=root, discover=discovery)
                error = rejected.exception
                self.assertEqual(error.code, 'model_already_served')
                self.assertEqual(error.data['existing_services'][0]['state'], 'starting')
                self.assertEqual(error.data['existing_services'][0]['origin'], 'http://127.0.0.1:8011')
                self.assertNotIn('do-not-return', str(error.data))
                self.assertEqual(error.data, model_serving.public_error_data(error.data))
            finally:
                claim.release()

    def test_policy_bad_schema_and_discovery_error_fail_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'state'; root.mkdir(mode=0o700)
            model = Path(directory) / 'model'; model.mkdir()
            (root / 'policy.json').write_text('{"schema_version": true}')
            (root / 'policy.json').chmod(0o600)
            with self.assertRaises(model_serving.ModelServingError) as invalid:
                model_serving.acquire(str(model), root=root, discover=lambda: {'services': [], 'errors': []})
            self.assertEqual(invalid.exception.code, 'model_serving_state_unknown')
            (root / 'policy.json').unlink()
            with self.assertRaises(model_serving.ModelServingError):
                model_serving.acquire(str(model), root=root, discover=lambda: {'services': [], 'errors': [{'message': 'failed'}]})

    def test_cancel_unstarted_gateway_removes_only_its_own_claim(self):
        import os
        from mlx_agent.compute import process_birth_identity
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'state'
            model = Path(directory) / 'model'; model.mkdir()
            service = {'port': 8012, 'runtime': 'tasks', 'auth_required': False}
            claim = model_serving.acquire(str(model), root=root, kind='service', service=service,
                                          discover=lambda: {'services': [], 'errors': []})
            claim.update(state='active', pid=os.getpid(), birth=process_birth_identity(os.getpid()),
                         pgid=os.getpgrp(), jit=True, ready=False)
            claim.cancel_unstarted_gateway()
            self.assertEqual(model_serving._read_claims(root, claim.key), [])

    def test_task_error_retains_safe_endpoint_details(self):
        from mlx_agent.task_runtime import _error
        service = {'origin': 'http://127.0.0.1:8013', 'api_base': 'http://127.0.0.1:8013/v1',
                   'service_id': 'a' * 64, 'runtime': 'tasks', 'state': 'ready', 'auth_required': True}
        failure = model_serving.ModelServingError('model_already_served', 'Already served', 'Connect',
                                                   {'existing_services': [service]})
        self.assertEqual(_error(failure)['data'], {'existing_services': [service]})

    def test_finished_leader_does_not_release_a_live_process_group(self):
        import os
        import signal
        import subprocess
        import sys
        from mlx_agent.compute import process_birth_identity
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'state'
            model = Path(directory) / 'model'; model.mkdir()
            claim = model_serving.acquire(str(model), root=root, discover=lambda: {'services': [], 'errors': []})
            script = ('import subprocess,sys,time; '
                      'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"], '
                      'stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
                      'print(p.pid,flush=True); time.sleep(.2)')
            leader = subprocess.Popen([sys.executable, '-c', script], start_new_session=True, stdout=subprocess.PIPE, text=True)
            child_pid = int(leader.stdout.readline())
            birth = process_birth_identity(leader.pid)
            claim.update(state='active', pid=leader.pid, birth=birth, pgid=leader.pid, jit=False)
            leader.wait(timeout=5)
            try:
                with self.assertRaises(model_serving.ModelServingError):
                    claim.release()
            finally:
                os.killpg(leader.pid, signal.SIGKILL)
                leader.stdout.close()
            # Orphan reaping is asynchronous; ownership stays retained until the group is gone.
            import time
            from mlx_agent.compute import group_members
            deadline = time.monotonic() + 5
            while group_members(leader.pid) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(group_members(leader.pid), [])
            claim.release()

    def test_oversized_policy_does_not_replace_existing_policy(self):
        from unittest.mock import patch
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'state'
            model_serving.write_policy([], root)
            before = (root / 'policy.json').read_bytes()
            with patch('mlx_agent.model_serving.resolve_model', side_effect=lambda value: '/' + value):
                with self.assertRaises(ValueError):
                    model_serving.write_policy([str(index) + 'a' * 300 for index in range(256)], root)
            self.assertEqual((root / 'policy.json').read_bytes(), before)

    def test_cli_policy_preview_and_confirm(self):
        import contextlib
        import io
        import json
        import os
        from unittest.mock import patch
        from mlx_agent.cli import main
        with TemporaryDirectory() as directory, patch.dict(os.environ, {'XDG_STATE_HOME': directory}):
            model = Path(directory) / 'model'; model.mkdir()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(['serve', 'duplication-policy', '--allow-model', str(model), '--json']), 0)
            self.assertEqual(json.loads(output.getvalue())['data']['status'], 'preview')
            self.assertFalse((model_serving.state_root() / 'policy.json').exists())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(['serve', 'duplication-policy', '--allow-model', str(model), '--confirm', '--json']), 0)
            self.assertEqual(model_serving.read_policy()['duplicate_model_allowlist'], [str(model.resolve())])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['serve', 'duplication-policy', '--clear', '--confirm', '--json']), 0)
            self.assertEqual(model_serving.read_policy()['duplicate_model_allowlist'], [])

    def test_inference_identity_probe_failure_stops_owned_child(self):
        import os
        import subprocess
        import sys
        from unittest.mock import patch
        from mlx_agent import transcribe
        from mlx_agent.inference_runner import invoke
        from mlx_agent.compute import process_birth_identity as real_birth
        with TemporaryDirectory() as directory:
            model = Path(directory) / 'model'; model.mkdir()
            state = Path(directory) / 'state'
            spawned = []
            original_popen = subprocess.Popen
            def popen(*args, **kwargs):
                process = original_popen(*args, **kwargs)
                if isinstance(args[0], list) and len(args[0]) > 1 and args[0][1] == '-c':
                    spawned.append(process)
                return process
            def birth(pid):
                if pid != os.getpid():
                    raise OSError('identity probe failed')
                return real_birth(pid)
            plan = {'backend': 'fixture', 'model': str(model),
                    'argv': [sys.executable, '-c', 'import time; time.sleep(30)']}
            with patch('mlx_agent.model_serving.state_root', return_value=state), \
                 patch('mlx_agent.service_registry.discover_services', return_value={'services': [], 'errors': []}), \
                 patch('mlx_agent.inference_runner.sync_ports'), \
                 patch('mlx_agent.compute.process_birth_identity', side_effect=birth), \
                 patch('subprocess.Popen', side_effect=popen):
                with self.assertRaisesRegex(OSError, 'identity probe failed'):
                    invoke(plan, transcribe.CONTRACT, {'fixture': {'id': 'fixture'}})
            self.assertEqual(len(spawned), 1)
            self.assertIsNotNone(spawned[0].returncode)
            self.assertEqual(model_serving._read_claims(state, next(state.glob('*.json')).stem), [])

    def test_private_worker_requires_listener_and_gateway_ancestor(self):
        import os
        import socket
        import subprocess
        import sys
        from unittest.mock import patch
        from mlx_agent.compute import process_birth_identity
        with TemporaryDirectory() as directory, patch.dict(os.environ, {'XDG_STATE_HOME': directory}):
            model = Path(directory) / 'model'; model.mkdir()
            service = {'port': 8014, 'runtime': 'mlx_lm', 'auth_required': True}
            claim = model_serving.acquire(str(model), kind='service', service=service,
                                          discover=lambda: {'services': [], 'errors': []})
            claim.update(state='active', pid=os.getpid(), birth=process_birth_identity(os.getpid()),
                         pgid=os.getpgrp(), jit=True, ready=False)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(Path(directory) / 'worker.sock')); listener.listen()
            try:
                code = ('from mlx_agent.model_serving import verify_private_worker; '
                        'import sys; verify_private_worker(sys.argv[1],int(sys.argv[2]))')
                result = subprocess.run([sys.executable, '-c', code, str(model), str(listener.fileno())],
                                        pass_fds=(listener.fileno(),), capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                os.set_inheritable(listener.fileno(), True)
                with self.assertRaises(model_serving.ModelServingError):
                    model_serving.verify_private_worker(str(model), listener.fileno())
            finally:
                listener.close()
                claim.cancel_unstarted_gateway()

    def test_unowned_custom_runner_is_rejected_before_work(self):
        from unittest.mock import Mock, patch
        from mlx_agent.inference_runner import invoke, InferenceError
        from mlx_agent import transcribe
        with TemporaryDirectory() as directory:
            model = Path(directory) / 'model'; model.mkdir()
            state = Path(directory) / 'state'
            runner = Mock()
            plan = {'backend': 'fixture', 'model': str(model), 'argv': ['/unused']}
            with patch('mlx_agent.model_serving.state_root', return_value=state), \
                 patch('mlx_agent.service_registry.discover_services', return_value={'services': [], 'errors': []}), \
                 patch('mlx_agent.inference_runner.sync_ports') as sync:
                with self.assertRaises(InferenceError) as rejected:
                    invoke(plan, transcribe.CONTRACT, {'fixture': {'id': 'fixture'}}, runner=runner)
            self.assertEqual(rejected.exception.code, 'runner_ownership_unknown')
            runner.assert_not_called(); sync.assert_not_called()
            self.assertEqual(model_serving._read_claims(state, next(state.glob('*.json')).stem), [])

    def test_task_mcp_status_preserves_safe_service_details(self):
        import json
        from mlx_agent.mcp_server import _tool_result
        from mlx_agent.task_store import TaskStore
        from tests.task_fixtures import identity_evidence
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve() / 'tasks'
            store = TaskStore(root)
            request = {'schema': 'task-request/1', 'kind': 'music_generation', 'model_path': '/models/local',
                       'parameters': {'caption': 'A quiet melody'}}
            record = store.create(request, identity_evidence(request['model_path']))
            service = {'origin': 'http://127.0.0.1:8013', 'api_base': 'http://127.0.0.1:8013/v1',
                       'service_id': 'a' * 64, 'runtime': 'tasks', 'state': 'ready', 'auth_required': True}
            error = {'code': 'model_already_served', 'message': 'Already served.', 'remediation': 'Connect.',
                     'data': {'existing_services': [service]}}
            store.transition(record['id'], ('starting',), {'state': 'failed', 'error': error})
            result = _tool_result({'capability': 'task', 'arguments':
                                   'status --state-dir "{}" --id {} --json'.format(root, record['id'])})
            payload = json.loads(result['structuredContent']['stdout'])
            self.assertEqual(payload['data']['task']['error']['data']['existing_services'], [service])

    def test_serve_cli_duplicate_error_contains_safe_endpoint_data(self):
        import contextlib
        import io
        import json
        import os
        import sys
        from unittest.mock import patch
        from mlx_agent.cli import main
        from mlx_agent.serve import load_recipes, plan_start
        with TemporaryDirectory() as directory, patch.dict(os.environ, {'XDG_STATE_HOME': directory}):
            root = Path(directory).resolve()
            model = root / 'model'; model.mkdir()
            service = {'port': 18015, 'runtime': 'mlx_lm', 'auth_required': True}
            claim = model_serving.acquire(str(model), kind='service', service=service,
                                          discover=lambda: {'services': [], 'errors': []})
            plan = plan_start(None, 'mlx_lm', load_recipes(), path=str(model), port=18016,
                              runtime_executable=sys.executable)
            output = io.StringIO()
            try:
                with patch('mlx_agent.serve.resolve_serve_runtime', return_value=('mlx_lm', sys.executable)), \
                     patch('mlx_agent.serve.plan_start', return_value=plan), \
                     patch('mlx_agent.service_registry.discover_services', return_value={'services': [], 'errors': []}), \
                     contextlib.redirect_stdout(output):
                    code = main(['serve', 'start', '--path', str(model), '--runtime', 'mlx_lm', '--port', '18016',
                                 '--confirm', '--preview-hash', plan['preview_hash'], '--json'])
                payload = json.loads(output.getvalue())
                self.assertEqual(code, 2)
                self.assertEqual(payload['error']['code'], 'model_already_served')
                self.assertEqual(payload['data']['existing_services'][0]['origin'], 'http://127.0.0.1:18015')
            finally:
                claim.release()

    def test_stopped_service_releases_claim_after_artifact_removed(self):
        import os
        import signal
        import subprocess
        import sys
        from unittest.mock import patch
        from mlx_agent.compute import process_birth_identity
        with TemporaryDirectory() as directory:
            model = (Path(directory) / 'model').resolve(); model.mkdir()
            state = Path(directory) / 'state'
            service = {'port': 8017, 'runtime': 'mlx_lm', 'auth_required': False}
            claim = model_serving.acquire(str(model), root=state, kind='service', service=service,
                                          discover=lambda: {'services': [], 'errors': []})
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
            birth = process_birth_identity(child.pid)
            claim.update(state='active', pid=child.pid, birth=birth, pgid=child.pid, jit=False)
            os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
            model.rmdir()
            receipt = {'model_artifact': str(model), 'pid': child.pid, 'process_started_at': birth}
            with patch('mlx_agent.model_serving.state_root', return_value=state):
                model_serving.release_stopped_service(receipt)
            self.assertEqual(model_serving._read_claims(state, claim.key), [])

    def test_duplicate_same_port_returns_existing_service(self):
        import sys
        from unittest.mock import patch
        from mlx_agent.serve import ServeError, load_recipes, plan_start, start_serve
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(); model = root / 'model'; model.mkdir()
            state = root / 'state'
            service = {'port': 18018, 'runtime': 'mlx_lm', 'auth_required': True}
            claim = model_serving.acquire(str(model), root=state, kind='service', service=service,
                                          discover=lambda: {'services': [], 'errors': []})
            plan = plan_start(None, 'mlx_lm', load_recipes(), path=str(model), port=18018,
                              runtime_executable=sys.executable)
            try:
                with patch('mlx_agent.model_serving.state_root', return_value=state), \
                     patch('mlx_agent.service_registry.discover_services', return_value={'services': [], 'errors': []}), \
                     self.assertRaises(ServeError) as rejected:
                    start_serve(plan, receipts_dir=root, confirm=True, preview_hash=plan['preview_hash'],
                                which=lambda _: sys.executable, port_free=lambda _: False)
                self.assertEqual(rejected.exception.code, 'model_already_served')
                self.assertEqual(rejected.exception.data['existing_services'][0]['origin'], 'http://127.0.0.1:18018')
            finally:
                claim.release()

    def test_different_model_busy_port_cancels_reservation(self):
        import hashlib
        import sys
        from unittest.mock import patch
        from mlx_agent.serve import ServeError, load_recipes, plan_start, start_serve
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(); model = root / 'model'; model.mkdir()
            state = root / 'state'
            plan = plan_start(None, 'mlx_lm', load_recipes(), path=str(model), port=18019,
                              runtime_executable=sys.executable)
            with patch('mlx_agent.model_serving.state_root', return_value=state), \
                 patch('mlx_agent.service_registry.discover_services', return_value={'services': [], 'errors': []}), \
                 self.assertRaises(ServeError) as rejected:
                start_serve(plan, receipts_dir=root, confirm=True, preview_hash=plan['preview_hash'],
                            which=lambda _: sys.executable, port_free=lambda _: False)
            self.assertEqual(rejected.exception.code, 'port_in_use')
            key = hashlib.sha256(str(model).encode()).hexdigest()
            self.assertEqual(model_serving._read_claims(state, key), [])

    def test_dead_legacy_eager_with_missing_model_is_skipped_only_when_absent(self):
        import json
        from unittest.mock import patch
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(); model = root / 'other'; model.mkdir()
            receipt_root = root / '.mlx-agent-receipts' / 'serve'; receipt_root.mkdir(parents=True)
            receipt_path = receipt_root / '8018.json'
            receipt_path.write_text(json.dumps({'kind': 'serve', 'pid': 97123, 'argv': ['server'],
                                                'port': 8018, 'runtime': 'mlx_lm', 'repo': None,
                                                'path': str(root / 'missing')}))
            entry = {'receipt': str(receipt_path), 'receipts_dir': str(root), 'alive': False,
                     'pid': 97123, 'port': 8018, 'runtime': 'mlx_lm'}
            with patch('mlx_agent.compute.process_alive', return_value=False), \
                 patch('mlx_agent.compute.process_birth_identity', return_value=None), \
                 patch('mlx_agent.compute.group_members', return_value=[]):
                self.assertEqual(model_serving._legacy_matches(str(model), [entry]), [])
            for target, value in [('process_alive', True), ('process_birth_identity', 'birth'), ('group_members', [97124])]:
                with self.subTest(target=target), \
                     patch('mlx_agent.compute.process_alive', return_value=False), \
                     patch('mlx_agent.compute.process_birth_identity', return_value=None), \
                     patch('mlx_agent.compute.group_members', return_value=[]), \
                     patch('mlx_agent.compute.' + target, return_value=value), \
                     self.assertRaises(model_serving.ModelServingError):
                    model_serving._legacy_matches(str(model), [entry])
            with patch('mlx_agent.compute.process_alive', side_effect=OSError('unavailable')), \
                 self.assertRaises(model_serving.ModelServingError):
                model_serving._legacy_matches(str(model), [entry])
            jit_receipt = json.loads(receipt_path.read_text()); jit_receipt['jit'] = True
            receipt_path.write_text(json.dumps(jit_receipt))
            with self.assertRaises(model_serving.ModelServingError):
                model_serving._legacy_matches(str(model), [entry])

    def test_dead_legacy_jit_filters_known_other_model_and_blocks_same_model(self):
        import json
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(); first = root / 'first'; first.mkdir()
            other = root / 'other'; other.mkdir()
            receipt_root = root / '.mlx-agent-receipts' / 'serve'; receipt_root.mkdir(parents=True)
            receipt_path = receipt_root / '8018.json'
            receipt_path.write_text(json.dumps({'kind': 'serve', 'pid': 97123, 'argv': ['server'],
                'port': 8018, 'runtime': 'mlx_lm', 'repo': None, 'path': str(first), 'jit': True}))
            (receipt_root / '8018.jit-config.json').write_text(json.dumps({'local_path': str(first),
                                                                          'control_token': 'a' * 32}))
            entry = {'receipt': str(receipt_path), 'receipts_dir': str(root), 'alive': False,
                     'pid': 97123, 'port': 8018, 'runtime': 'mlx_lm'}
            self.assertEqual(model_serving._legacy_matches(str(other), [entry]), [])
            with self.assertRaises(model_serving.ModelServingError):
                model_serving._legacy_matches(str(first), [entry])

    def test_legacy_splash_receipt_uses_its_bound_model_directory(self):
        import json
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(); model = root / 'cache' / 'snapshot'; model.mkdir(parents=True)
            receipt_root = root / '.mlx-agent-receipts' / 'serve'; receipt_root.mkdir(parents=True)
            receipt_path = receipt_root / '8018.json'
            receipt_path.write_text(json.dumps({'kind': 'serve', 'pid': 97123, 'argv': ['splash'],
                'port': 8018, 'runtime': 'splash', 'repo': 'pub/unavailable', 'path': None,
                'splash_model_dir': str(model), 'process_started_at': 'birth'}))
            entry = {'receipt': str(receipt_path), 'receipts_dir': str(root), 'alive': True,
                     'argv_match': True, 'process_started_at': 'birth', 'pid': 97123, 'port': 8018, 'runtime': 'splash',
                     'service_id': 'a' * 64, 'api_key_sha256': None}
            result = model_serving._legacy_matches(str(model), [entry])
            self.assertEqual(result[0]['origin'], 'http://127.0.0.1:8018')
