"""Durable conditioning snapshots preserve the accepted files and their order."""
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator
from mlx_agent.task_contracts import TaskError
from mlx_agent.task_store import TaskStore
from tests.task_fixtures import identity_evidence


class NamedInputTests(unittest.TestCase):
    def test_mesh_inputs_are_owned_ordered_and_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = [root / name for name in ('main.png', 'a.jpg', 'b.png', 'guide.glb')]
            for index, path in enumerate(paths):
                path.write_bytes(str(index).encode())
            request = {'schema': 'task-request/2', 'kind': 'mesh_generation', 'model_path': str(root / 'model'),
                       'parameters': {'image_path': str(paths[0]), 'views': [str(paths[1]), str(paths[2])], 'guide': str(paths[3])}}
            store = TaskStore(root / 'store')
            staged = store.stage_inputs(request)
            record = store.create(request, identity_evidence(request['model_path']), staged_input=staged)
            schema = json.loads((Path(__file__).resolve().parents[2] / 'schemas/task-run.schema.json').read_text())
            self.assertTrue(Draft202012Validator(schema).is_valid(record))
            for path in paths:
                path.write_bytes(b'changed')
            owned = store.verify_named_inputs(record['id'], request, record['input_artifacts'])
            self.assertEqual([Path(path).read_bytes() for path in owned['views']], [b'1', b'2'])
            self.assertEqual(Path(owned['image_path']).read_bytes(), b'0')
            Path(owned['guide']).write_bytes(b'changed')
            with self.assertRaises(TaskError):
                store.verify_named_inputs(record['id'], request, record['input_artifacts'])

    def test_file_count_limit(self):
        request = {'kind': 'mesh_generation', 'parameters': {'image_path': '/a.png', 'views': ['/b.png'] * 16}}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(TaskError) as failure:
                TaskStore(Path(temporary).resolve() / 'store').stage_inputs(request)
            self.assertEqual(failure.exception.code, 'too_many_inputs')

    def test_budget_and_explicit_media_expiration(self):
        from mlx_agent import task_storage
        from mlx_agent.task_runtime import task_result
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            image = root / 'image.png'
            image.write_bytes(b'input')
            request = {'schema': 'task-request/2', 'kind': 'mesh_generation', 'model_path': str(root / 'model'),
                       'parameters': {'image_path': str(image)}}
            store = TaskStore(root / 'store')
            task_storage.set_budget(store, 1)
            staged = store.stage_inputs(request)
            with self.assertRaises(TaskError) as failure:
                store.create(request, identity_evidence(request['model_path']), staged_input=staged)
            self.assertEqual(failure.exception.code, 'storage_budget_exceeded')
            store.discard_staged_input(staged)
            task_storage.set_budget(store, 0)
            staged = store.stage_inputs(request)
            record = store.create(request, identity_evidence(request['model_path']), staged_input=staged)
            identifier = record['id']
            with self.assertRaises(TaskError):
                task_storage.preview(store, identifier)
            store.transition(identifier, ('starting',), {'state': 'failed', 'error': {'code': 'failed', 'message': 'failed', 'remediation': 'retry'}})
            before = store.get(identifier)
            preview = task_storage.preview(store, identifier)
            with self.assertRaises(TaskError):
                task_storage.expire(store, identifier, 'wrong')
            self.assertTrue(Path(record['input_artifacts']['image_path']['path']).exists())
            self.assertEqual(task_storage.expire(store, identifier, preview['confirmation'])['state'], 'expired')
            self.assertFalse(Path(record['input_artifacts']['image_path']['path']).exists())
            self.assertEqual(store.get(identifier), before)
            self.assertEqual(task_storage.expiry_status(store, identifier), 'expired')
            self.assertEqual(image.read_bytes(), b'input')

    def test_local_and_served_acceptance_retain_video_conditioning(self):
        import threading
        from unittest.mock import patch
        from mlx_agent import task_runtime, task_serving
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            image, audio = root / 'image.png', root / 'audio.wav'
            image.write_bytes(b'image-before')
            audio.write_bytes(b'audio-before')
            request = {'schema': 'task-request/2', 'kind': 'video_generation', 'model_path': str(root / 'model'),
                       'parameters': {'prompt': 'test', 'image': str(image), 'audio': str(audio)}}
            identity = identity_evidence(request['model_path'])
            plan = {'backend': identity['backend'], 'argv': [identity['requested_executable']]}
            request_file = root / 'request.json'
            request_file.write_text(json.dumps(request))
            with patch('mlx_agent.task_kinds.TaskKind.plan', return_value=plan), \
                 patch('mlx_agent.task_runtime.capture_identity', return_value=identity), \
                 patch('mlx_agent.task_runtime.load_manifests', return_value={identity['backend']: {}}), \
                 patch('mlx_agent.task_runtime._spawn'):
                local = task_runtime.start_task(request_file, root / 'local')
            controller = task_serving.TaskModelWorker.__new__(task_serving.TaskModelWorker)
            controller.config = {'task_kinds': ['video_generation'], 'local_path': request['model_path'], 'task_identity': identity}
            controller.store = TaskStore(root / 'served')
            controller.condition = threading.Condition()
            controller.closed = controller.active_requests = controller.cache_control_active = controller.execution_unresolved = False
            controller.activity_revision = 0
            controller.model_state = 'unloaded'
            controller._recheck_identity = lambda: None
            controller.resident = type('Resident', (), {'executable': identity['requested_executable']})()
            with patch('mlx_agent.task_kinds.TaskKind.plan', return_value=plan), \
                 patch('mlx_agent.task_serving.threading.Thread'):
                served = controller.submit({'kind': 'video_generation', 'parameters': request['parameters']})
            image.write_bytes(b'image-after')
            audio.write_bytes(b'audio-after')
            for record, store in ((local, TaskStore(root / 'local')), (served, controller.store)):
                retained = store.verify_named_inputs(record['id'], request, record['input_artifacts'])
                self.assertEqual(Path(retained['image']).read_bytes(), b'image-before')
                self.assertEqual(Path(retained['audio']).read_bytes(), b'audio-before')

    def test_mesh_cli_submits_typed_route_and_preserves_failure(self):
        import contextlib
        import io
        import threading
        from http.server import HTTPServer
        from unittest.mock import patch
        from mlx_agent.cli import main
        from mlx_agent.task_contracts import validate_request
        from mlx_agent.task_serve import TaskHandler

        observed = []
        class Controller:
            def submit(self, body):
                observed.append(body)
                validate_request({'schema': 'task-request/2', 'kind': body['kind'],
                                  'model_path': model, 'parameters': body['parameters']})
                raise TaskError('mesh_rejected', 'The backend rejected this job.')
        class Handler(TaskHandler):
            def _guard(self, control=False):
                pass
            def _dispatch_control(self, body):
                return False
            def log_message(self, *args):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            model = str(root / 'model')
            server = HTTPServer(('127.0.0.1', 0), Handler)
            server.controller = Controller()
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            stdout = io.StringIO()
            with patch('mlx_agent.mesh.plan_mesh', return_value={}), \
                 patch('mlx_agent.served.endpoint', return_value={'port': server.server_port}), \
                 patch('mlx_agent.served._key', return_value=None), contextlib.redirect_stdout(stdout):
                code = main(['convert', 'mesh', '--path', model, '--image', str(root / 'input.png'),
                             '--view', str(root / 'a.png'), '--view', str(root / 'b.png'),
                             '--guide', str(root / 'guide.glb'), '--out', str(root / 'output.glb'), '--json'])
            self.assertNotEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())['error']['code'], 'mesh_rejected')
            self.assertEqual(observed[0]['kind'], 'mesh_generation')
            self.assertEqual(observed[0]['parameters']['views'], [str(root / 'a.png'), str(root / 'b.png')])
            self.assertEqual(observed[0]['parameters']['guide'], str(root / 'guide.glb'))

    def test_budget_equality_and_interrupted_expiration(self):
        from unittest.mock import patch
        from mlx_agent import task_storage
        from mlx_agent.task_contracts import encoded
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            image = root / 'input.png'
            image.write_bytes(b'abc')
            request = {'schema': 'task-request/2', 'kind': 'mesh_generation', 'model_path': str(root / 'model'),
                       'parameters': {'image_path': str(image)}}
            store = TaskStore(root / 'store')
            task_storage.set_budget(store, 3 + len(encoded(request)))
            staged = store.stage_inputs(request)
            record = store.create(request, identity_evidence(request['model_path']), staged_input=staged)
            identifier = record['id']
            store.transition(identifier, ('starting',), {'state': 'failed', 'error': {'code': 'x', 'message': 'x', 'remediation': 'x'}})
            preview = task_storage.preview(store, identifier)
            with patch('mlx_agent.task_storage.os.unlink', side_effect=OSError('interrupted')):
                with self.assertRaises(TaskError):
                    task_storage.expire(store, identifier, preview['confirmation'])
            self.assertEqual(task_storage.expiry_status(store, identifier), 'pending')
            self.assertEqual(task_storage.expire(store, identifier, preview['confirmation'])['state'], 'expired')

    def test_concurrent_expiration_and_provenance(self):
        from concurrent.futures import ThreadPoolExecutor
        from mlx_agent import task_storage
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            image = root / 'input.png'
            image.write_bytes(b'input')
            request = {'schema': 'task-request/2', 'kind': 'mesh_generation', 'model_path': str(root / 'model'),
                       'parameters': {'image_path': str(image)}}
            store = TaskStore(root / 'store')
            staged = store.stage_inputs(request)
            record = store.create(request, identity_evidence(request['model_path']), staged_input=staged)
            identifier = record['id']
            store.transition(identifier, ('starting',), {'state': 'failed', 'error': {'code': 'x', 'message': 'x', 'remediation': 'x'}})
            token = task_storage.preview(store, identifier)['confirmation']
            with ThreadPoolExecutor(max_workers=2) as executor:
                states = list(executor.map(lambda _: task_storage.expire(store, identifier, token)['state'], range(2)))
            self.assertEqual(states, ['expired', 'expired'])
            provenance = task_storage.provenance(store, identifier)
            self.assertEqual(provenance['input_artifacts'], record['input_artifacts'])
            self.assertEqual(provenance['media_expiration'], 'expired')

    def test_unconditioned_jobs_keep_the_existing_record_version(self):
        from mlx_agent.task_inputs import needs_named
        cases = [('video_generation', {'prompt': 'test'}, False),
                 ('video_generation', {'prompt': 'test', 'image': '/image.png'}, True),
                 ('mesh_generation', {'image_path': '/image.png', 'views': []}, False),
                 ('mesh_generation', {'image_path': '/image.png', 'guide': '/guide.glb'}, True)]
        for kind, parameters, expected in cases:
            self.assertEqual(needs_named({'kind': kind, 'parameters': parameters}), expected)

    def test_inventory_rejects_invalid_task_ids(self):
        import sqlite3
        from mlx_agent import task_storage
        with tempfile.TemporaryDirectory() as temporary:
            store = TaskStore(Path(temporary).resolve() / 'store')
            task_storage.set_budget(store, 1000)
            with sqlite3.connect(store.database) as connection:
                connection.execute('INSERT INTO tasks VALUES (?, ?, ?)', ('../outside', 'now', '{}'))
            with self.assertRaises(TaskError) as failure:
                task_storage.status(store)
            self.assertEqual(failure.exception.code, 'invalid_task_id')

    def test_video_audio_formats_preserve_existing_input_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for suffix in ('.m4a', '.aac'):
                audio = root / ('audio' + suffix)
                audio.write_bytes(b'audio')
                request = {'schema': 'task-request/2', 'kind': 'video_generation', 'model_path': str(root / 'model'),
                           'parameters': {'prompt': 'test', 'audio': str(audio)}}
                store = TaskStore(root / ('store-' + suffix[1:]))
                staged = store.stage_inputs(request)
                record = store.create(request, identity_evidence(request['model_path']), staged_input=staged)
                self.assertEqual(Path(store.verify_named_inputs(record['id'], request, record['input_artifacts'])['audio']).read_bytes(), b'audio')
                with self.assertRaises(TaskError):
                    store.stage_input(str(audio), 'audio')

    def test_served_video_paths_are_absolute_without_resolving_links(self):
        from argparse import Namespace
        from mlx_agent.served import request
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            target = root / 'source.png'
            target.write_bytes(b'image')
            link = root / 'link.png'
            link.symlink_to(target)
            relative = str(link.relative_to(Path.cwd()))
            values = dict(prompt='test', width=64, height=64, frames=9, steps=1, seed=42, fps=24,
                          denoiser_cache='off', cache_memory_mib=256, image=relative, audio='audio.m4a', audio_start=None)
            kind, parameters = request('video', Namespace(**values))
            self.assertEqual(kind, 'video_generation')
            self.assertEqual(parameters['image'], str(link))
            self.assertTrue(Path(parameters['image']).is_symlink())
            self.assertEqual(parameters['audio'], str(Path('audio.m4a').absolute()))

    def test_fifo_rejected_and_known_size_checked_before_copy(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            store = TaskStore(root / 'store')
            fifo = root / 'input.png'
            os.mkfifo(fifo)
            with self.assertRaises(TaskError) as failure:
                store.stage_input(str(fifo), 'image')
            self.assertEqual(failure.exception.code, 'invalid_input')
            source = root / 'input.jpg'
            source.write_bytes(b'abc')
            with patch('mlx_agent.task_store.os.fdopen', side_effect=AssertionError('must reject before streaming')):
                with self.assertRaises(TaskError) as failure:
                    store.stage_input(str(source), 'image', byte_limit=2)
            self.assertEqual(failure.exception.code, 'input_too_large')
