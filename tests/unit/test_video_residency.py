"""The Wan pipeline must retain actual loaded components, not only its directory metadata."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch



class VideoResidencyTests(unittest.TestCase):
    def test_adaptive_session_closes_on_pipeline_failure_and_budget_falls_back(self):
        path = Path(__file__).resolve().parents[2] / 'src/mlx_agent/resources/ports/mlx-video/t2v/t2v.py'
        package = types.ModuleType('adaptive_port_fixture')
        package.__path__ = [str(path.parent)]
        spec = importlib.util.spec_from_file_location('adaptive_port_fixture.t2v', path)
        port = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(port)
        sessions = []

        class Session:
            def __init__(self, *args):
                self.closed = False
                sessions.append(self)

            def close(self):
                self.closed = True

        cache_module = types.SimpleNamespace(WanDenoiserCache=Session, required_cache_bytes=lambda *a: 128 * 1024 ** 2)
        wan = types.ModuleType('mlx_video.generate_wan')
        for name in port.LOADERS:
            setattr(wan, name, lambda: None)
        wan.save_video = lambda *a, **k: None
        calls = []

        def fail(**kwargs):
            calls.append(kwargs)
            raise RuntimeError('generation failed')

        wan.generate_video = fail
        core = types.SimpleNamespace(reset_peak_memory=lambda: None)
        modules = {'adaptive_port_fixture': package, 'adaptive_port_fixture.denoiser_cache': cache_module,
                   'mlx': types.SimpleNamespace(core=core), 'mlx.core': core,
                   'mlx_video': types.SimpleNamespace(generate_wan=wan), 'mlx_video.generate_wan': wan,
                   'transformers': types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: None))}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            model = port.Model(Path(directory), {'model_version':'2.1', 'model_type':'t2v'})
            for budget in (256, 8):
                with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                    port.generate(model, 'A pan.', str(Path(directory) / 'out.mp4'),
                                  denoiser_cache='adaptive', cache_memory_mib=budget)
            self.assertEqual(len(sessions), 1)
            self.assertTrue(sessions[0].closed)
            self.assertTrue(calls[0]['no_compile'])
            self.assertNotIn('no_compile', calls[1])

    def test_repeated_generations_reuse_components_only_in_resident_mode(self):
        path = Path(__file__).resolve().parents[2] / 'src/mlx_agent/resources/ports/mlx-video/t2v/t2v.py'
        spec = importlib.util.spec_from_file_location('video_residency_fixture', path)
        port = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(port)
        for retain, expected in ((False, 2), (True, 1)):
            with self.subTest(retain=retain), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                calls = {}
                wan = types.ModuleType('mlx_video.generate_wan')
                for name in port.LOADERS:
                    def loader(*args, key=name, **kwargs):
                        calls[key] = calls.get(key, 0) + 1
                        return object()
                    loader.__name__ = name
                    setattr(wan, name, loader)
                class Tokenizer:
                    @staticmethod
                    def from_pretrained(*args, **kwargs):
                        self.assertTrue(kwargs['local_files_only'])
                        self.assertFalse(kwargs['trust_remote_code'])
                        calls['tokenizer'] = calls.get('tokenizer', 0) + 1
                        return object()
                def generate_video(**options):
                    for name in port.LOADERS:
                        getattr(wan, name)(options['model_dir'])
                    Tokenizer.from_pretrained('google/umt5-xxl')
                    wan.save_video(types.SimpleNamespace(shape=(2, 4, 4, 3)), options['output_path'])
                wan.generate_video = generate_video
                wan.save_video = lambda *a, **k: None
                core = types.SimpleNamespace(reset_peak_memory=lambda: None, get_peak_memory=lambda: 0)
                modules = {'mlx': types.SimpleNamespace(core=core), 'mlx.core': core,
                    'mlx_video': types.SimpleNamespace(generate_wan=wan), 'mlx_video.generate_wan': wan,
                    'transformers': types.SimpleNamespace(AutoTokenizer=Tokenizer)}
                model = port.Model(root, {})
                model.retain_weights = retain
                with patch.dict(sys.modules, modules), patch.object(port, '_encode'), patch.object(port, '_pixel_std', return_value=0):
                    for seed in (1, 2):
                        port.generate(model, 'test', str(root / ('out%d.mp4' % seed)), seed=seed)
                self.assertEqual(calls, {name: expected for name in (*port.LOADERS, 'tokenizer')})
                self.assertEqual(len(model.components), 4 if retain else 0)
