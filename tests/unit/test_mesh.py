import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def test_rejects_invalid_settings_and_existing_outputs(self):
        for settings in ({'steps':0}, {'steps':True}, {'seed':-1}, {'resolution':96}, {'guidance':float('nan')}):
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
