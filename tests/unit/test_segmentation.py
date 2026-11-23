"""Contracts for local image-segmentation tasks."""

import importlib.util
import unittest
from pathlib import Path

from mlx_agent.task_contracts import TaskError
from mlx_agent.task_kinds import get, validate_parameters


class SegmentationTests(unittest.TestCase):
    def test_control_keeps_torch_telemetry_after_segmentation(self):
        import io
        import json
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        from mlx_agent import resident_worker

        class Output(io.BytesIO):
            def close(self):
                pass

        requests = [{'runner': 'segment_runner', 'arguments': []},
                    {'control': 'cache', 'policy': {}, 'reclaim': None}]
        input_stream = SimpleNamespace(buffer=io.BytesIO(b''.join(json.dumps(x).encode() + b'\n' for x in requests)))
        output = Output()
        cache = SimpleNamespace(desired={'cache_reclaim': 'none'}, configure=lambda *x: None,
                                apply_limit=lambda **x: None, snapshot=dict)
        frameworks = []
        def capture(stats, framework):
            frameworks.append(framework)
            return {}
        with (patch.object(sys, 'argv', ['worker', '--response-fd', '7', '--model', '/tmp/model']),
              patch.object(sys, 'stdin', input_stream), patch.object(resident_worker.os, 'fdopen', return_value=output),
              patch.object(resident_worker.threading.Thread, 'start'),
              patch.object(resident_worker, 'AllocatorCache', return_value=cache),
              patch.object(resident_worker.runner_model_cache, 'loaded', return_value=True),
              patch.object(resident_worker.importlib, 'import_module', return_value=SimpleNamespace(main=lambda args: 0)),
              patch.object(resident_worker.worker_telemetry, 'capture', side_effect=capture)):
            resident_worker.main()
        self.assertEqual(frameworks, ['torch', 'torch'])

    def test_worker_applies_torch_grant_without_mlx(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from mlx_agent.resident_worker import apply_memory_limit
        values = []
        torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                                mps=SimpleNamespace(recommended_max_memory=lambda: 1000,
                                                    set_per_process_memory_fraction=values.append))
        with patch('mlx_agent.resident_worker.importlib.import_module', return_value=torch):
            apply_memory_limit('segment_runner', 500)
            with self.assertRaises(ValueError):
                apply_memory_limit('segment_runner', 2001)
        self.assertEqual(values, [0.5])

    def test_requires_valid_box_points_and_threshold(self):
        kind = get('image_segmentation')
        value = {'image_path': '/tmp/image.png', 'box': [1, 2, 3, 4], 'mask_threshold': 0.0}
        self.assertEqual(validate_parameters(kind, value), value)
        for change in ({'box': [1, 2, 1, 4]}, {'mask_threshold': float('nan')},
                       {'points': [[1, 2]], 'labels': []}, {'labels': [2], 'points': [[1, 2]]}):
            with self.subTest(change=change), self.assertRaises(TaskError):
                validate_parameters(kind, dict(value, **change))

    def test_runner_digests_the_image_without_reading_it_all_at_once(self):
        import sys
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch

        from mlx_agent.resources.ports.sam2 import segment

        with TemporaryDirectory() as directory:
            source = Path(directory) / 'input.png'
            source.write_bytes(b'image data')
            # Stop at decoding so this check needs neither Pillow nor Torch.
            def stop_at_decode(path):
                self.assertEqual(path, source)
                raise ValueError('reached image decoding')

            pillow = SimpleNamespace(Image=SimpleNamespace(open=stop_at_decode), ImageOps=None)
            with (patch.dict(sys.modules, {'PIL': pillow, 'torch': SimpleNamespace()}),
                  patch.object(Path, 'read_bytes', side_effect=AssertionError('unbounded file read')),
                  self.assertRaisesRegex(ValueError, 'reached image decoding')):
                segment.generate((None, None, [], {}), source, source.with_name('mask.png'),
                                 [0, 0, 1, 1], [], [], 0.0)

    def test_loader_rejects_non_video_extras_and_missing_weights(self):
        path = Path(__file__).parents[2] / 'src/mlx_agent/resources/ports/sam2/segment/__init__.py'
        spec = importlib.util.spec_from_file_location('segment_port', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        empty = {'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': [], 'error_msgs': []}
        self.assertEqual(module.validate_loading(dict(empty, unexpected_keys=['memory_encoder.foo'])), ['memory_encoder.foo'])
        with self.assertRaises(ValueError):
            module.validate_loading({})
        for info in ({'missing_keys': ['vision_encoder.foo']}, {'unexpected_keys': ['vision_encoder.foo']},
                     {'mismatched_keys': ['foo']}, {'error_msgs': ['error']}):
            with self.subTest(info=info), self.assertRaises(ValueError):
                module.validate_loading(dict(empty, **info))
        self.assertEqual(module.validate_checkpoint_keys({'image.weight'}, {'image.weight', 'memory_encoder.foo'}),
                         ['memory_encoder.foo'])
        for keys in ({'memory_encoder.foo'}, {'image.weight', 'unknown.weight'},
                     {'image.weight', 'no_object_pointer_bogus'}):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                module.validate_checkpoint_keys({'image.weight'}, keys)
