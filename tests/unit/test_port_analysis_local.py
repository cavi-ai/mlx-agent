import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from mlx_agent.port_analysis import analyze, analyze_with_sources
from mlx_agent.backends import load_manifests

class LocalPortAnalysisTests(unittest.TestCase):
    def test_analyzes_diffusers_components_without_a_root_config(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'model_index.json').write_text(json.dumps({'_class_name':'TripoSGPipeline','transformer':['triposg','TripoSGDiTModel']}))
            (root/'transformer').mkdir()
            (root/'transformer/config.json').write_text(json.dumps({'_class_name':'TripoSGDiTModel','width':2048}))
            (root/'transformer/model.py').write_text('raise RuntimeError("must never execute")\nclass TripoSGDiTModel:\n    pass\n')
            payload,sources=analyze_with_sources(folder)
            self.assertEqual(payload['model_type'],'triposg')
            self.assertEqual(payload['architectures'],['TripoSGPipeline'])
            self.assertEqual(payload['source']['revision'],'local')
            self.assertEqual(payload['components'][0]['role'],'transformer')
            self.assertIn('transformer/model.py',sources)

    def test_source_only_analysis_reports_unknown_support(self):
        with TemporaryDirectory() as folder:
            (Path(folder)/'model.py').write_text('class SomeModel:\n    pass\n')
            payload=analyze(folder)
            self.assertIsNone(payload['model_type'])
            self.assertTrue(any('Source-only' in w for w in payload['warnings']))
            self.assertEqual(payload['code']['files'][0]['classes'][0]['name'],'SomeModel')

    def test_external_source_links_are_not_read(self):
        with TemporaryDirectory() as folder, TemporaryDirectory() as outside:
            (Path(outside)/'private.py').write_text('SECRET')
            (Path(folder)/'model.py').symlink_to(Path(outside)/'private.py')
            payload,sources=analyze_with_sources(folder)
            self.assertEqual(sources,{})
            self.assertTrue(payload['warnings'])
