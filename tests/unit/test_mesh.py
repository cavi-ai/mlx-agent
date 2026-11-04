import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent import cli
from mlx_agent.backends import INSTALL_MARKER, load_manifests, load_registries
from mlx_agent.mesh import MeshError, plan_mesh
from mlx_agent.taxonomy import classify


class MeshPlanningTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifests = load_manifests()
        backend = self.root / 'backends' / 'mlx-3d'
        (backend / 'bin').mkdir(parents=True)
        (backend / 'bin/python').touch()
        (backend / INSTALL_MARKER).write_text(json.dumps({'id':'mlx-3d','version':self.manifests['mlx-3d']['version']}))
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text(json.dumps({'model_type':'triposg'}))
        self.image = self.root / 'object image.png'
        self.image.touch()
        self.out = self.root / 'shape.glb'

    def plan(self, **kwargs):
        return plan_mesh(str(self.model), str(self.image), str(self.out), root=self.root/'backends', **kwargs)

    def test_dispatches_the_native_port_and_argv_paths(self):
        plan = self.plan()
        self.assertEqual(plan['backend'], 'mlx-3d')
        args = plan['argv']
        self.assertEqual(args[args.index('--module')+1], 'mlx.mlx_agent_ports.triposg')
        self.assertEqual(args[args.index('--image')+1], str(self.image))
        self.assertEqual(args[args.index('--out')+1], str(self.out))

    def test_defaults_to_the_upstream_resolution_and_face_cap(self):
        args = self.plan()['argv']
        self.assertEqual(args[args.index('--resolution')+1], '512')
        self.assertEqual(args[args.index('--max-faces')+1], '500000')

    def test_accepts_every_grid_resolution_from_32_to_512(self):
        for resolution in (32, 64, 128, 256, 512):
            with self.subTest(resolution=resolution):
                args = self.plan(resolution=resolution, max_faces=1000)['argv']
                self.assertEqual(args[args.index('--resolution')+1], str(resolution))
                self.assertEqual(args[args.index('--max-faces')+1], '1000')

    def cli(self, *extra, model=None):
        argv = ['convert', 'mesh', '--path', str(model or self.model), '--image', str(self.image), '--out', str(self.out), '--json', *extra]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
            try:
                code = cli.main(argv)
            except SystemExit as stop:
                code = stop.code
        return code, buffer.getvalue()

    def test_cli_refuses_resolutions_and_face_caps_outside_the_supported_range(self):
        for extra in (['--resolution', '1024'], ['--resolution', '96'], ['--max-faces', '999'], ['--max-faces', '10000001']):
            with self.subTest(extra=extra):
                code, output = self.cli(*extra)
                self.assertEqual(code, 2)
                if output:
                    self.assertEqual(json.loads(output)['error']['code'], 'invalid_arguments')
        chat = self.root / 'chat'
        chat.mkdir()
        (chat / 'config.json').write_text(json.dumps({'model_type': 'bert'}))
        for extra in ([], ['--resolution', '512'], ['--max-faces', '1000'], ['--resolution', '32', '--max-faces', '10000000']):
            with self.subTest(accepted=extra):
                code, output = self.cli(*extra, model=chat)
                self.assertEqual(json.loads(output)['error']['code'], 'not_mesh_model')

    def test_rejects_invalid_settings_and_existing_outputs(self):
        for settings in ({'steps':0}, {'steps':True}, {'seed':-1}, {'resolution':96}, {'resolution':1024}, {'guidance':float('nan')},
                         {'max_faces':999}, {'max_faces':True}, {'max_faces':1000.0}, {'max_faces':10_000_001}):
            with self.subTest(settings=settings), self.assertRaises(MeshError) as caught:
                self.plan(**settings)
            self.assertEqual(caught.exception.code, 'invalid_arguments')
        self.out.write_bytes(b'keep')
        with self.assertRaises(MeshError):
            self.plan()
        self.assertEqual(self.out.read_bytes(), b'keep')

    def test_scan_taxonomy_distinguishes_mesh_from_chat(self):
        result = classify('TripoSG', model_type='triposg', manifests=self.manifests,
                          registries=load_registries(self.manifests, self.root/'backends'), local=True)
        self.assertEqual(result['type'], 'mesh_generation')
